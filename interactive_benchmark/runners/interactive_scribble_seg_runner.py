#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interactive_benchmark.runners.gpu_prompt_ops import (  # noqa: E402
    rasterize_points_gpu,
    rasterize_polyline_gpu,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Scribble Segmentation runtime bridge.")
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


def expand_manifest_value(manifest_path: Path, value: Any) -> str:
    return (
        str(value)
        .replace("{repo_root}", str(PROJECT_ROOT))
        .replace("{manifest_dir}", str(manifest_path.parent.resolve()))
    )


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
    manifest_path = Path(payload["manifest_path"])
    manifest = load_json(manifest_path)
    inference = dict(manifest.get("inference") or {})
    external_repo_path = Path(expand_manifest_value(manifest_path, inference["external_repo_path"]))
    checkpoint_path = Path(expand_manifest_value(manifest_path, inference["checkpoint_path"]))
    model_name = str(inference.get("model_name") or "amix_150000")
    resize_size = int(inference.get("resize_size") or 128)
    click_radius_px = int(inference.get("click_radius_px") or 5)
    line_width_px = int(inference.get("line_width_px") or 5)
    prob_thresh = float(inference.get("prob_thresh") or 0.5)
    prefer = str(inference.get("device") or os.environ.get("INTERACTIVE_BENCHMARK_DEVICE") or "cuda:0")
    device = resolve_device(prefer)

    if str(external_repo_path) not in sys.path:
        sys.path.insert(0, str(external_repo_path))
    from config import get_args  # type: ignore  # noqa: E402
    from model_unet import AbstractUNet  # type: ignore  # noqa: E402

    image_rgb = np.array(Image.open(payload["image_path"]).convert("RGB"), dtype=np.uint8)
    h, w = image_rgb.shape[:2]
    class_id = int(payload["prompt"]["class_id"])
    prompts = prompt_history_for_class(payload, class_id)

    skeleton_t = torch.zeros((h, w), dtype=torch.float32, device=device)
    for item in prompts:
        points = item.get("points") or []
        if not points:
            continue
        points_xy = [point_to_pixel(pt, h, w) for pt in points]
        prompt_type = str(item.get("prompt_type") or "").lower()
        if prompt_type == "point":
            stroke = rasterize_points_gpu(
                h=h,
                w=w,
                points_xy=points_xy,
                radius_px=click_radius_px,
                device=device,
            )
        else:
            stroke = rasterize_polyline_gpu(
                h=h,
                w=w,
                points_xy=points_xy,
                width_px=line_width_px,
                device=device,
            )
        skeleton_t = torch.where(stroke, torch.ones_like(skeleton_t), skeleton_t)

    im = cv2.resize(image_rgb, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
    im_t = torch.from_numpy(im).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    im_t = im_t / 255.0 * 2.0 - 1.0
    skelet_small = cv2.resize(
        skeleton_t.detach().cpu().numpy(),
        (resize_size, resize_size),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    skelet_t = torch.from_numpy(skelet_small).to(device=device, dtype=torch.float32).unsqueeze(0)
    batch = torch.cat((im_t, skelet_t), dim=0).unsqueeze(0)

    args = get_args(name=model_name.split("_")[0])
    net = AbstractUNet(args).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(ckpt["net"])
    net.eval()

    with torch.no_grad():
        seg = net(batch)
        seg = torch.clamp(seg.squeeze(1), min=0.0, max=1.0).squeeze(0)
    pred = cv2.resize(seg.detach().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.asarray(pred >= prob_thresh, dtype=bool)
    score = float(pred[mask].mean()) if mask.any() else float(np.mean(pred))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "proposals": [
            {
                "class_id": class_id,
                "mask": mask.astype(bool).tolist(),
                "score": score,
                "candidate_id": "main",
                "metadata": {
                    "mode": "interactive_scribble_seg",
                    "device": str(device),
                    "num_prompts": len(prompts),
                    "prob_thresh": prob_thresh,
                    "resize_size": resize_size,
                },
            }
        ]
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = load_json(Path(args.input_json))
    result = run(payload)
    write_json(Path(args.output_json), result)


if __name__ == "__main__":
    main()
