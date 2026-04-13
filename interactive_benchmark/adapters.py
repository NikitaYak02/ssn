from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw

from .contracts import MaskProposal, MethodAdapter, PromptPayload
from .resource_monitor import run_monitored_command
from .session import SessionState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rasterize_points_mask(
    image_shape: tuple[int, int],
    points: list[tuple[float, float]],
    radius_px: int,
) -> np.ndarray:
    h, w = image_shape
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x_norm, y_norm in points:
        x = int(round(float(x_norm) * float(w)))
        y = int(round(float(y_norm) * float(h)))
        draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=255)
    return np.array(mask, dtype=np.uint8) > 0


def _rasterize_polyline_mask(
    image_shape: tuple[int, int],
    points: list[tuple[float, float]],
    width_px: int,
) -> np.ndarray:
    h, w = image_shape
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if len(points) == 1:
        return _rasterize_points_mask(image_shape, points, radius_px=max(1, width_px // 2))
    pts_px = [(float(x) * float(w), float(y) * float(h)) for x, y in points]
    draw.line(pts_px, fill=255, width=int(width_px))
    return np.array(mask, dtype=np.uint8) > 0


class MockGeometryAdapter(MethodAdapter):
    def __init__(self, method_id: str, display_name: str, prompt_type: str) -> None:
        self.method_id = method_id
        self.display_name = display_name
        self.prompt_type = prompt_type
        self.last_resource_metrics: dict[str, Any] | None = None

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    def predict(
        self,
        image_rgb: np.ndarray,
        session_state: SessionState,
        prompt: PromptPayload,
        work_dir: Path,
    ) -> list[MaskProposal]:
        start = time.perf_counter()
        h, w = image_rgb.shape[:2]
        scale = max(2, int(round(min(h, w) * 0.08)))
        if self.prompt_type == "point":
            mask = _rasterize_points_mask((h, w), prompt.points, radius_px=scale)
        elif self.prompt_type == "line":
            mask = _rasterize_polyline_mask((h, w), prompt.points, width_px=max(2, scale // 2))
        else:
            mask = _rasterize_polyline_mask((h, w), prompt.points, width_px=max(3, scale))
        proposals = [
            MaskProposal(
                class_id=int(prompt.class_id),
                mask=mask,
                score=1.0,
                source=self.method_id,
                candidate_id=f"{self.method_id}_default",
            )
        ]
        self.last_resource_metrics = {
            "wall_time_sec": float(time.perf_counter() - start),
            "peak_rss_bytes": 0,
            "peak_gpu_memory_mib": 0,
            "memory_limit_exceeded": 0,
        }
        return proposals


class ExternalSubprocessAdapter(MethodAdapter):
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.method_id = str(self.manifest["method_id"])
        self.display_name = str(self.manifest.get("display_name") or self.method_id)
        self.prompt_type = str(self.manifest["prompt_type"])
        self.last_resource_metrics: dict[str, Any] | None = None

    def _expand_string(self, value: Any) -> str:
        text = str(value)
        return (
            text.replace("{repo_root}", str(PROJECT_ROOT))
            .replace("{manifest_dir}", str(self.manifest_path.parent.resolve()))
        )

    def _resolve_path_value(self, value: Any) -> Path:
        raw = self._expand_string(value)
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        manifest_relative = (self.manifest_path.parent / candidate).resolve()
        if manifest_relative.exists():
            return manifest_relative
        repo_relative = (PROJECT_ROOT / candidate).resolve()
        if repo_relative.exists():
            return repo_relative
        return manifest_relative

    def _resolve_command_executable(self, value: Any) -> str:
        expanded = self._expand_string(value)
        if os.sep in expanded or expanded.startswith("."):
            candidate = Path(expanded).expanduser()
            if candidate.is_absolute():
                return str(candidate)
            manifest_relative = self.manifest_path.parent / candidate
            if manifest_relative.exists():
                return str(manifest_relative)
            repo_relative = PROJECT_ROOT / candidate
            if repo_relative.exists():
                return str(repo_relative)
            return str(manifest_relative)
        executable = shutil.which(expanded)
        if executable is not None:
            return executable
        path_candidate = self._resolve_path_value(expanded)
        return str(path_candidate)

    def is_available(self) -> tuple[bool, Optional[str]]:
        disabled_reason = self.manifest.get("disabled_reason")
        if disabled_reason:
            return False, str(disabled_reason)
        runtime = self.manifest.get("runtime") or {}
        entrypoint = self.manifest.get("entrypoint") or {}
        command = entrypoint.get("command") or []
        if not command:
            return False, "manifest has no entrypoint.command configured"
        runtime_cwd = runtime.get("cwd")
        if runtime_cwd:
            resolved_cwd = self._resolve_path_value(runtime_cwd)
            if not resolved_cwd.exists():
                return False, f"missing runtime.cwd: {resolved_cwd}"
        for item in self.manifest.get("required_paths") or []:
            resolved = self._resolve_path_value(item)
            if not resolved.exists():
                return False, f"missing required path: {resolved}"
        executable = self._resolve_command_executable(command[0])
        if not Path(executable).exists() and shutil.which(str(command[0])) is None:
            return False, f"missing executable: {command[0]}"
        return True, None

    def _write_runtime_input(
        self,
        image_rgb: np.ndarray,
        session_state: SessionState,
        prompt: PromptPayload,
        temp_dir: Path,
    ) -> tuple[Path, Path, Path]:
        image_path = temp_dir / "image.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_path)
        input_path = temp_dir / "input.json"
        output_path = temp_dir / "output.json"
        payload = {
            "method_id": self.method_id,
            "manifest_path": str(self.manifest_path.resolve()),
            "image_path": str(image_path.resolve()),
            "prompt": prompt.to_dict(),
            "session": session_state.to_runtime_payload(),
        }
        with open(input_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return image_path, input_path, output_path

    def _load_mask(self, payload: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
        if "mask" in payload:
            return np.asarray(payload["mask"], dtype=bool)
        if "mask_path" in payload:
            mask_path = Path(str(payload["mask_path"]))
            if mask_path.suffix.lower() == ".npy":
                return np.load(mask_path).astype(bool)
            return np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        raise ValueError("Proposal is missing 'mask' or 'mask_path'.")

    def predict(
        self,
        image_rgb: np.ndarray,
        session_state: SessionState,
        prompt: PromptPayload,
        work_dir: Path,
    ) -> list[MaskProposal]:
        self.last_resource_metrics = None
        available, reason = self.is_available()
        if not available:
            raise RuntimeError(reason or f"{self.method_id} is unavailable")
        work_dir.mkdir(parents=True, exist_ok=True)
        runtime = self.manifest.get("runtime") or {}
        entrypoint = self.manifest.get("entrypoint") or {}
        command_template = [str(item) for item in (entrypoint.get("command") or [])]
        with tempfile.TemporaryDirectory(prefix=f"{self.method_id}_", dir=str(work_dir)) as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            _, input_path, output_path = self._write_runtime_input(image_rgb, session_state, prompt, temp_dir)
            command: list[str] = []
            for idx, item in enumerate(command_template):
                expanded = (
                    self._expand_string(item)
                    .replace("{input_json}", str(input_path))
                    .replace("{output_json}", str(output_path))
                )
                command.append(
                    self._resolve_command_executable(expanded) if idx == 0 else expanded
                )
            env = {
                str(key): self._expand_string(value)
                for key, value in dict(runtime.get("env") or {}).items()
            }
            runtime_cwd = runtime.get("cwd")
            resolved_cwd = self._resolve_path_value(runtime_cwd) if runtime_cwd else work_dir
            proc = run_monitored_command(
                command,
                cwd=str(resolved_cwd),
                env=None if not env else {**os.environ, **env},
                memory_limit_gb=getattr(self, "memory_limit_gb", None),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.method_id} failed: {proc.returncode}\n"
                    f"stdout:\n{proc.stdout[-2000:]}\n"
                    f"stderr:\n{proc.stderr[-2000:]}"
                )
            with open(output_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.last_resource_metrics = {
                "wall_time_sec": float(proc.usage.wall_time_sec),
                "peak_rss_bytes": int(proc.usage.peak_rss_bytes),
                "peak_gpu_memory_mib": int(proc.usage.peak_gpu_memory_mib),
                "memory_limit_exceeded": int(bool(proc.usage.memory_limit_exceeded)),
            }
        proposals: list[MaskProposal] = []
        for idx, item in enumerate(payload.get("proposals") or []):
            proposals.append(
                MaskProposal(
                    class_id=int(item.get("class_id", prompt.class_id)),
                    mask=self._load_mask(item, image_rgb.shape[:2]),
                    score=None if item.get("score") is None else float(item["score"]),
                    source=self.method_id,
                    candidate_id=str(item.get("candidate_id") or idx),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        if not proposals:
            raise RuntimeError(f"{self.method_id} produced no proposals.")
        return proposals
