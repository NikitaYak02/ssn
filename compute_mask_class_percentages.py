#!/usr/bin/env python3
"""
Compute class content percentages in annotation masks.

The script reports:
- global class share across all valid pixels in the dataset
- mean per-image class share (including zeros for images without the class)
- mean per-image class share only over images where the class is present
- how many images contain each class

Example:
  superpixel_annotator/superpixel_annotator_venv/bin/python \
    compute_mask_class_percentages.py \
    --masks ../target_dataset/S1_v2/masks/test \
    --class-codes 0,1,2,3,4,5,6,7,8,11
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_superpixel_postprocessing import (
    IMAGE_EXTENSIONS,
    get_named_class_sets,
    load_mask_codes,
    maybe_add_petroscope_root,
    parse_class_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute class-content percentages from segmentation masks."
    )
    parser.add_argument("--masks", required=True, help="Directory with masks.")
    parser.add_argument(
        "--class-codes",
        default=None,
        help=(
            "Optional comma-separated class-code order. If omitted, the script "
            "uses the sorted codes observed in masks."
        ),
    )
    parser.add_argument(
        "--ignore-codes",
        default=None,
        help=(
            "Optional comma-separated raw codes to ignore, for example 255. "
            "Ignored pixels are excluded from all percentages."
        ),
    )
    parser.add_argument(
        "--petroscope-root",
        default=None,
        help="Optional local petroscope checkout for recovering class names.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save the class-percentage table as CSV.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def discover_masks(masks_dir: str) -> list[Path]:
    paths = []
    for mask_path in sorted(Path(masks_dir).iterdir()):
        if mask_path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(mask_path)
    return paths


def parse_codes(raw: str | None) -> set[int]:
    values = parse_class_codes(raw)
    return set(values or [])


def load_code_name_map(petroscope_root: str | None) -> dict[int, str]:
    maybe_add_petroscope_root(petroscope_root)
    try:
        from petroscope.segmentation.classes import LumenStoneClasses
    except Exception:
        return {}

    named_sets = get_named_class_sets(LumenStoneClasses)
    code_to_name: dict[int, str] = {}
    for infos in named_sets.values():
        for info in infos:
            code_to_name.setdefault(int(info.code), str(info.name))
    return code_to_name


def format_percent(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    mask_paths = discover_masks(args.masks)
    if not mask_paths:
        raise RuntimeError("No mask files were found.")

    ignore_codes = parse_codes(args.ignore_codes)
    requested_codes = parse_class_codes(args.class_codes)
    code_to_name = load_code_name_map(args.petroscope_root)

    total_pixels = 0
    total_counts: dict[int, int] = defaultdict(int)
    per_image_counts: list[dict[int, int]] = []
    per_image_valid_pixels: list[int] = []
    observed_codes: set[int] = set()

    for mask_path in mask_paths:
        mask_codes = load_mask_codes(mask_path)
        valid_mask = np.ones(mask_codes.shape, dtype=bool)
        if ignore_codes:
            valid_mask &= ~np.isin(mask_codes, list(ignore_codes))

        valid_codes = mask_codes[valid_mask]
        valid_pixels = int(valid_codes.size)
        per_image_valid_pixels.append(valid_pixels)
        image_counts: dict[int, int] = {}
        if valid_pixels > 0:
            unique_codes, unique_counts = np.unique(valid_codes, return_counts=True)
            for code, count in zip(unique_codes.tolist(), unique_counts.tolist()):
                code_i = int(code)
                count_i = int(count)
                image_counts[code_i] = count_i
                total_counts[code_i] += count_i
                observed_codes.add(code_i)
            total_pixels += valid_pixels
        per_image_counts.append(image_counts)

    if total_pixels == 0:
        raise RuntimeError("All pixels were ignored. Nothing to analyze.")

    if requested_codes is not None:
        class_codes = requested_codes
    else:
        class_codes = sorted(observed_codes)

    rows: list[dict[str, object]] = []
    for code in class_codes:
        total_count = int(total_counts.get(code, 0))
        global_percent = (100.0 * total_count / total_pixels) if total_pixels > 0 else 0.0

        image_percents: list[float] = []
        present_image_percents: list[float] = []
        images_with_class = 0
        for valid_pixels, image_counts in zip(per_image_valid_pixels, per_image_counts):
            if valid_pixels <= 0:
                image_percent = 0.0
            else:
                image_percent = 100.0 * image_counts.get(code, 0) / valid_pixels
            image_percents.append(float(image_percent))
            if image_counts.get(code, 0) > 0:
                present_image_percents.append(float(image_percent))
                images_with_class += 1

        mean_image_percent = float(np.mean(image_percents)) if image_percents else 0.0
        std_image_percent = float(np.std(image_percents)) if image_percents else 0.0
        mean_present_image_percent = (
            float(np.mean(present_image_percents)) if present_image_percents else 0.0
        )

        rows.append(
            {
                "class_code": int(code),
                "class_name": code_to_name.get(int(code), f"class_{code}"),
                "global_percent": global_percent,
                "mean_image_percent": mean_image_percent,
                "std_image_percent": std_image_percent,
                "mean_present_image_percent": mean_present_image_percent,
                "images_with_class": images_with_class,
                "n_images": len(mask_paths),
                "total_pixels": total_count,
            }
        )

    rows.sort(key=lambda row: float(row["global_percent"]), reverse=True)

    print()
    print(f"Masks: {Path(args.masks).resolve()}")
    print(f"Images analyzed: {len(mask_paths)}")
    print(f"Valid pixels total: {total_pixels}")
    if ignore_codes:
        print(f"Ignored codes: {sorted(ignore_codes)}")
    print()
    print(
        f"{'code':>6} {'name':<12} {'global_%':>10} {'mean_img_%':>12} "
        f"{'std_img_%':>10} {'mean_present_%':>15} {'images':>8}"
    )
    for row in rows:
        print(
            f"{int(row['class_code']):>6d} "
            f"{str(row['class_name'])[:12]:<12} "
            f"{format_percent(float(row['global_percent'])):>10} "
            f"{format_percent(float(row['mean_image_percent'])):>12} "
            f"{format_percent(float(row['std_image_percent'])):>10} "
            f"{format_percent(float(row['mean_present_image_percent'])):>15} "
            f"{int(row['images_with_class']):>3d}/{int(row['n_images']):<4d}"
        )

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"Saved CSV to: {output_csv}")

    if args.output_json:
        output_json = Path(args.output_json).resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "masks_dir": str(Path(args.masks).resolve()),
            "n_images": len(mask_paths),
            "total_valid_pixels": total_pixels,
            "ignored_codes": sorted(ignore_codes),
            "rows": rows,
        }
        with open(output_json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        print(f"Saved JSON to: {output_json}")


if __name__ == "__main__":
    main()
