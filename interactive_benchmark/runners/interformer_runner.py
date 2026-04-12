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
    parser = argparse.ArgumentParser(description="InterFormer runtime bridge for interactive_benchmark.")
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


def set_test_cfg(cfg, size_divisor: int) -> Any:
    data_test_cfg = dict(
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="Resize", img_scale=None, ratio_range=(1.0, 1.0), keep_ratio=True),
            dict(type="RandomFlip", prob=0.0),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=False,
            ),
            dict(type="Pad", size=None, size_divisor=size_divisor, pad_val=0, seg_pad_val=0),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ]
    )
    cfg.merge_from_dict(
        {
            "model.backbone.type": "MAEWithSimpleFPN",
            "model.type": "ClickSegmentorZoomIn",
            "data.test": data_test_cfg,
        }
    )
    cfg.model.decode_head.norm_cfg = dict(type="BN", requires_grad=True)
    if cfg.model.get("auxiliary_head") is not None:
        cfg.model.auxiliary_head.norm_cfg = dict(type="BN", requires_grad=True)
    return cfg


def run(payload: dict[str, Any]) -> dict[str, Any]:
    import mmcv
    import torch
    from mmseg.apis import init_segmentor

    from demo.gui import clicker
    from demo.gui.predictor import Predictor

    manifest_path = Path(payload["manifest_path"])
    manifest = load_json(manifest_path)
    inference = dict(manifest.get("inference") or {})
    checkpoint_path = Path(expand_manifest_value(manifest_path, inference["checkpoint_path"]))
    config_path = Path(expand_manifest_value(manifest_path, inference["config_path"]))
    size_divisor = int(inference.get("size_divisor", 32))
    device = torch.device(str(inference.get("device") or "cpu"))

    cfg = mmcv.Config.fromfile(str(config_path))
    cfg = set_test_cfg(cfg, size_divisor=size_divisor)
    model = init_segmentor(cfg, str(checkpoint_path), device=str(device))
    predictor = Predictor(
        model=model,
        device=device,
        predictor_params={
            "inner_radius": int(inference.get("inner_radius", 5)),
            "outer_radius": int(inference.get("outer_radius", 0)),
            "zoom_in_params": None,
        },
    )

    image_rgb = np.array(Image.open(payload["image_path"]).convert("RGB"), dtype=np.uint8)
    predictor.set_input_image(image_rgb)

    class_id = int(payload["prompt"]["class_id"])
    prompts = prompt_history_for_class(payload, class_id)
    prediction = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    for idx, item in enumerate(prompts, start=1):
        prompt_type = str(item.get("prompt_type") or "").lower()
        if prompt_type != "point":
            raise ValueError(f"InterFormer runner only supports point prompts, got {prompt_type!r}")
        points = item.get("points") or []
        if not points:
            continue
        x, y = point_to_pixel(points[0], image_rgb.shape[0], image_rgb.shape[1])
        predictor.update_ref_label_by_new_click(
            clicker.Click(is_positive=True, coords=(y, x))
        )
        prediction = predictor.get_prediction(idx, prev_mask=None)

    mask = np.asarray(prediction > 0, dtype=bool)
    return {
        "proposals": [
            {
                "class_id": class_id,
                "mask": mask.astype(bool).tolist(),
                "score": float(mask.mean()),
                "candidate_id": "main",
                "metadata": {
                    "device": str(device),
                    "num_clicks": len(prompts),
                    "size_divisor": size_divisor,
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
