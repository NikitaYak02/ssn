#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_benchmark.runners.interactive_scribble_seg_runner import run as run_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render step-by-step overlays for Interactive Scribble Segmentation.")
    parser.add_argument("--run-dir", required=True, help="Path to method run dir with step_XXX subdirs.")
    parser.add_argument("--image", required=True, help="Source RGB image path.")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "interactive_benchmark" / "manifests" / "interactive_scribble_seg.json"),
        help="Method manifest path.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for debug visualizations.")
    parser.add_argument("--crop-margin", type=int, default=96, help="Pixel margin around current prompt bbox.")
    parser.add_argument("--downscale", type=float, default=1.0, help="Output scale factor, e.g. 0.5.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _norm_points_to_px(points: list[list[float]], h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for p in points:
        x = int(round(float(p[0]) * float(w - 1)))
        y = int(round(float(p[1]) * float(h - 1)))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        out.append((x, y))
    return out


def _draw_polyline(draw: ImageDraw.ImageDraw, points_xy: list[tuple[int, int]], color: tuple[int, int, int], width: int) -> None:
    if not points_xy:
        return
    if len(points_xy) == 1:
        x, y = points_xy[0]
        r = max(1, width // 2)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, fill=color)
        return
    draw.line(points_xy, fill=color, width=width)


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    mixed = base.copy()
    m = mask.astype(bool)
    mixed[m] = (1.0 - alpha) * base[m] + alpha * color[m]
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _maybe_downscale(img: Image.Image, scale: float) -> Image.Image:
    s = float(scale)
    if s <= 0:
        raise ValueError("--downscale must be > 0")
    if abs(s - 1.0) < 1e-8:
        return img
    w, h = img.size
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return img.resize((nw, nh), resample=Image.Resampling.BILINEAR)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image).resolve()
    manifest_path = Path(args.manifest).resolve()
    image_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    h, w = image_rgb.shape[:2]

    step_dirs = sorted([p for p in run_dir.glob("step_*") if p.is_dir()])
    prompt_history: list[dict[str, Any]] = []

    for step_dir in step_dirs:
        prompt_path = step_dir / "prompt.json"
        if not prompt_path.exists():
            continue
        payload = _load_json(prompt_path)
        prompt = dict(payload["prompt"])

        runner_payload = {
            "method_id": "interactive_scribble_seg",
            "manifest_path": str(manifest_path),
            "image_path": str(image_path),
            "prompt": prompt,
            "session": {"prompt_history": list(prompt_history)},
        }
        result = run_method(runner_payload)
        proposal = (result.get("proposals") or [])[0]
        mask = np.asarray(proposal["mask"], dtype=bool)

        overlay = _overlay_mask(image_rgb, mask)
        canvas = Image.fromarray(overlay, mode="RGB")
        draw = ImageDraw.Draw(canvas)

        for old in prompt_history:
            old_pts = _norm_points_to_px(old.get("points") or [], h, w)
            _draw_polyline(draw, old_pts, color=(70, 200, 255), width=2)
        cur_pts = _norm_points_to_px(prompt.get("points") or [], h, w)
        _draw_polyline(draw, cur_pts, color=(255, 230, 0), width=3)

        if cur_pts:
            xs = [p[0] for p in cur_pts]
            ys = [p[1] for p in cur_pts]
            x0 = max(0, min(xs) - int(args.crop_margin))
            y0 = max(0, min(ys) - int(args.crop_margin))
            x1 = min(w, max(xs) + int(args.crop_margin))
            y1 = min(h, max(ys) + int(args.crop_margin))
            crop = canvas.crop((x0, y0, x1, y1))
            _maybe_downscale(crop, args.downscale).save(output_dir / f"{step_dir.name}_zoom.png")

        _maybe_downscale(canvas, args.downscale).save(output_dir / f"{step_dir.name}_overlay.png")
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        _maybe_downscale(mask_img, args.downscale).save(output_dir / f"{step_dir.name}_mask.png")

        prompt_history.append(prompt)

    meta = {
        "run_dir": str(run_dir),
        "image": str(image_path),
        "manifest": str(manifest_path),
        "n_steps_rendered": len(step_dirs),
    }
    (output_dir / "debug_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
