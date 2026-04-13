from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .resource_monitor import run_monitored_command


def _coerce_value(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except Exception:
        return value


def load_metrics_csv(metrics_path: Path) -> list[dict[str, Any]]:
    with open(metrics_path, "r", encoding="utf-8", newline="") as fh:
        return [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(fh)]


def normalize_current_pipeline_metrics(
    rows: list[dict[str, Any]],
    *,
    interaction_budgets: list[int],
    image_name: str,
    method_id: str = "current_pipeline",
    display_name: str = "Current Pipeline",
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda item: int(float(item.get("step") or 0)))
    normalized: list[dict[str, Any]] = []
    for budget in sorted(int(item) for item in interaction_budgets):
        chosen = rows_sorted[0]
        for row in rows_sorted:
            step = int(float(row.get("step") or 0))
            if step <= int(budget):
                chosen = row
            else:
                break
        normalized.append(
            {
                "image": image_name,
                "method_id": method_id,
                "display_name": display_name,
                "interaction_budget": int(budget),
                "step": int(float(chosen.get("step") or 0)),
                "n_interactions": int(float(chosen.get("n_scribbles") or 0)),
                "coverage": float(chosen.get("coverage") or 0.0),
                "annotation_precision": float(chosen.get("annotation_precision") or 0.0),
                "miou": float(chosen.get("miou") or 0.0),
                "total_ink_px": float(chosen.get("total_ink_px") or 0.0),
                "status": "ok",
                "prompt_type": "scribble",
                "source": "evaluate_interactive_annotation.py",
            }
        )
    return normalized


def _save_current_pipeline_final_mask(output_dir: Path) -> None:
    state_candidates = sorted(output_dir.glob("state_*.json"))
    if not state_candidates:
        return
    state_path = state_candidates[-1]
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    image_meta = dict(payload.get("_meta", {}).get("image") or {})
    size_wh = image_meta.get("size_wh") or []
    if len(size_wh) != 2:
        return
    width, height = int(size_wh[0]), int(size_wh[1])
    if width <= 0 or height <= 0:
        return
    annotations_map = dict(payload.get("annotations") or {})
    if not annotations_map:
        return
    method_key = next(iter(annotations_map.keys()))
    annotations = list(annotations_map.get(method_key) or [])
    if not annotations:
        return

    canvas = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for item in annotations:
        if not isinstance(item, dict):
            continue
        points01 = item.get("border") or []
        if len(points01) < 3:
            continue
        try:
            class_id = max(0, int(item.get("code", 0)))
        except Exception:
            class_id = 0
        points_px: list[tuple[int, int]] = []
        for xy in points01:
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                continue
            x = int(round(float(xy[0]) * float(width - 1)))
            y = int(round(float(xy[1]) * float(height - 1)))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            points_px.append((x, y))
        if len(points_px) >= 3:
            draw.polygon(points_px, fill=int(class_id))

    arr = np.array(canvas, dtype=np.uint16)
    Image.fromarray(arr, mode="I;16").save(output_dir / "final_mask.png")


def _cleanup_intermediate_visuals(output_dir: Path) -> None:
    for frame_path in output_dir.glob("frame_*.png"):
        try:
            frame_path.unlink()
        except OSError:
            continue


def run_current_pipeline(
    *,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    interaction_budgets: list[int],
    pipeline_args: dict[str, Any] | None = None,
    python_bin: str | None = None,
    memory_limit_gb: float | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_args = dict(pipeline_args or {})
    cmd = [
        str(python_bin or sys.executable),
        "evaluate_interactive_annotation.py",
        "--image",
        str(image_path.resolve()),
        "--mask",
        str(mask_path.resolve()),
        "--out",
        str(output_dir.resolve()),
        "--save_every",
        str(max(int(item) for item in interaction_budgets)),
        "--scribbles",
        str(max(int(item) for item in interaction_budgets)),
    ]
    for key, value in pipeline_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, ensure_ascii=False)])
            continue
        cmd.extend([flag, str(value)])

    proc = run_monitored_command(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        memory_limit_gb=memory_limit_gb,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"current_pipeline failed: {proc.returncode}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    _save_current_pipeline_final_mask(output_dir)
    _cleanup_intermediate_visuals(output_dir)
    metrics_rows = load_metrics_csv(output_dir / "metrics.csv")
    normalized = normalize_current_pipeline_metrics(
        metrics_rows,
        interaction_budgets=interaction_budgets,
        image_name=image_path.stem,
    )
    for row in normalized:
        row["run_wall_time_sec"] = float(proc.usage.wall_time_sec)
        row["run_peak_rss_bytes"] = int(proc.usage.peak_rss_bytes)
        row["run_peak_gpu_memory_mib"] = int(proc.usage.peak_gpu_memory_mib)
        row["run_memory_limit_exceeded"] = int(bool(proc.usage.memory_limit_exceeded))
    return normalized
