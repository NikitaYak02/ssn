#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SegNext runtime bridge for interactive_benchmark.")
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


def resolve_device(preference: str):
    import torch

    pref = str(preference).strip().lower()
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prompt_history_for_class(payload: dict[str, Any], class_id: int) -> list[dict[str, Any]]:
    history = []
    for item in payload.get("session", {}).get("prompt_history") or []:
        if int(item.get("class_id", -1)) == int(class_id):
            history.append(item)
    history.append(payload["prompt"])
    return history


def point_to_pixel(point: list[float] | tuple[float, float], height: int, width: int) -> tuple[int, int]:
    x = int(round(float(point[0]) * float(width - 1)))
    y = int(round(float(point[1]) * float(height - 1)))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from isegm.inference.clicker import Click, Clicker
    from isegm.inference.predictor import BasePredictor
    from isegm.inference.utils import load_is_model

    manifest_path = Path(payload["manifest_path"])
    manifest = load_json(manifest_path)
    inference = dict(manifest.get("inference") or {})
    checkpoint_path = Path(expand_manifest_value(manifest_path, inference["checkpoint_path"]))
    device = resolve_device(str(inference.get("device") or "cpu"))
    prob_thresh = float(inference.get("prob_thresh", 0.5))

    image_rgb = np.array(Image.open(payload["image_path"]).convert("RGB"), dtype=np.uint8)
    class_id = int(payload["prompt"]["class_id"])
    prompts = prompt_history_for_class(payload, class_id)

    model = load_is_model(str(checkpoint_path), device, cpu_dist_maps=True)
    predictor = BasePredictor(model)
    predictor.set_image(image_rgb)
    clicker = Clicker()

    prediction = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    for item in prompts:
        prompt_type = str(item.get("prompt_type") or "").lower()
        if prompt_type != "point":
            raise ValueError(f"SegNext runner only supports point prompts, got {prompt_type!r}")
        points = item.get("points") or []
        if not points:
            continue
        x, y = point_to_pixel(points[0], image_rgb.shape[0], image_rgb.shape[1])
        clicker.add_click(Click(is_positive=True, coords=(y, x)))
        prediction = predictor.predict(clicker)

    mask = np.asarray(prediction >= prob_thresh, dtype=bool)
    score = float(prediction[mask].mean()) if mask.any() else float(np.mean(prediction))
    if device.type == "mps":
        torch.mps.empty_cache()

    return {
        "proposals": [
            {
                "class_id": class_id,
                "mask": mask.astype(bool).tolist(),
                "score": score,
                "candidate_id": "main",
                "metadata": {
                    "device": str(device),
                    "prob_thresh": prob_thresh,
                    "num_clicks": len(prompts),
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
