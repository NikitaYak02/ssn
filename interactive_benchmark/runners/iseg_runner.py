#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_benchmark.runners.gpu_prompt_ops import rasterize_points_gpu, rasterize_polyline_gpu, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="iSeg runtime bridge for interactive_benchmark.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def point_to_pixel(point: list[float] | tuple[float, float], height: int, width: int) -> tuple[int, int]:
    x = int(round(float(point[0]) * float(width - 1)))
    y = int(round(float(point[1]) * float(height - 1)))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def prompt_history_for_class(payload: dict[str, Any], class_id: int) -> list[dict[str, Any]]:
    history = []
    for item in payload.get("session", {}).get("prompt_history") or []:
        if int(item.get("class_id", -1)) == int(class_id):
            history.append(item)
    history.append(payload["prompt"])
    return history


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    image_rgb = np.array(Image.open(payload["image_path"]).convert("RGB"), dtype=np.uint8)
    h, w = image_rgb.shape[:2]
    class_id = int(payload["prompt"]["class_id"])
    prompts = prompt_history_for_class(payload, class_id)
    width = max(3, int(round(min(h, w) * 0.01)))
    device = resolve_device(os.environ.get("INTERACTIVE_BENCHMARK_DEVICE", "cuda:0"))

    agg = torch.zeros((h, w), dtype=torch.bool, device=device)
    for item in prompts:
        points = item.get("points") or []
        points_xy = [point_to_pixel(pt, h, w) for pt in points]
        if len(points_xy) <= 1:
            agg |= rasterize_points_gpu(h=h, w=w, points_xy=points_xy, radius_px=width, device=device)
        else:
            agg |= rasterize_polyline_gpu(h=h, w=w, points_xy=points_xy, width_px=width, device=device)
    out_mask = agg.detach().cpu().numpy().astype(bool)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "proposals": [
            {
                "class_id": class_id,
                "mask": out_mask.tolist(),
                "score": float(out_mask.mean()),
                "candidate_id": "compat",
                "metadata": {
                    "mode": "compat_fallback_gpu",
                    "prompt_type": "line",
                    "n_prompts": len(prompts),
                    "device": str(device),
                },
            }
        ]
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = load_json(Path(args.input_json))
    write_json(Path(args.output_json), run(payload))


if __name__ == "__main__":
    main()
