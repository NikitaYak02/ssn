#!/usr/bin/env python3
"""
Tune confidence_threshold for low_confidence_mean_proba using cached logits.

The script reuses cache files produced by `tune_hybrid_conservative.py` and
evaluates only the low-confidence overwrite stage for a range of thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_superpixel_postprocessing import (
    ClassInfo,
    compute_confusion,
    compute_superpixel_mean_probs,
    metrics_from_confusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune confidence_threshold for low_confidence_mean_proba using an "
            "existing on-disk logits cache."
        )
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory containing cache_manifest.json and per-image .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where tuning CSV/JSON will be written.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "Optional comma-separated explicit threshold list. If omitted, the "
            "script uses threshold-start/stop/step."
        ),
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.0,
        help="Start of the threshold sweep, inclusive.",
    )
    parser.add_argument(
        "--threshold-stop",
        type=float,
        default=1.0,
        help="End of the threshold sweep, inclusive within rounding tolerance.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step for the threshold sweep.",
    )
    parser.add_argument(
        "--max-acc-drop",
        type=float,
        default=0.005,
        help="Max allowed pixel-accuracy drop for the 'safe' best threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top thresholds to include in the summary JSON.",
    )
    return parser.parse_args()


def parse_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds:
        items = [item.strip() for item in args.thresholds.split(",") if item.strip()]
        if not items:
            raise ValueError("Expected a non-empty comma-separated threshold list.")
        thresholds = [float(item) for item in items]
    else:
        if args.threshold_step <= 0:
            raise ValueError("--threshold-step must be positive.")
        thresholds = []
        current = float(args.threshold_start)
        stop = float(args.threshold_stop)
        while current <= stop + 1e-9:
            thresholds.append(round(current, 10))
            current += float(args.threshold_step)

    unique_sorted = sorted({float(threshold) for threshold in thresholds})
    for threshold in unique_sorted:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                f"Threshold {threshold} is outside [0.0, 1.0], which is not valid."
            )
    return unique_sorted


def class_infos_from_json(items: list[dict[str, object]]) -> list[ClassInfo]:
    return [
        ClassInfo(
            code=int(item["mask_code"]),
            name=str(item["name"]),
            color_rgb=tuple(int(v) for v in item["color_rgb"]),
        )
        for item in items
    ]


def load_cache(
    cache_dir: Path,
    num_classes: int,
) -> list[dict[str, object]]:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    prepared: list[dict[str, object]] = []
    for entry in manifest["entries"]:
        cache_file = cache_dir / str(entry["cache_file"])
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing cache file: {cache_file}")

        with np.load(cache_file, allow_pickle=False) as data:
            logits_np = data["logits"].astype(np.float32)
            superpixels = data["superpixels"].astype(np.int32)
            gt_idx = data["gt_idx"].astype(np.int32)
            valid_mask = data["valid_mask"].astype(bool)

        flat_sp = superpixels.reshape(-1)
        num_sp = int(flat_sp.max()) + 1
        probs = torch.softmax(torch.from_numpy(logits_np), dim=0).numpy().astype(
            np.float32
        )
        flat_probs = probs.reshape(num_classes, -1)
        pixel_labels = flat_probs.argmax(axis=0).astype(np.int32)
        pixel_conf = flat_probs.max(axis=0).astype(np.float32)
        mean_probs, _ = compute_superpixel_mean_probs(probs, flat_sp, num_sp)
        sp_mean_labels = mean_probs.argmax(axis=0).astype(np.int32)
        baseline_pred = pixel_labels.reshape(superpixels.shape)
        baseline_conf = compute_confusion(
            gt_idx,
            baseline_pred,
            num_classes,
            valid_mask,
        )

        prepared.append(
            {
                "image_name": str(entry["image_name"]),
                "shape": superpixels.shape,
                "gt_idx": gt_idx,
                "valid_mask": valid_mask,
                "flat_sp": flat_sp,
                "pixel_labels": pixel_labels,
                "pixel_conf": pixel_conf,
                "sp_mean_labels": sp_mean_labels,
                "baseline_conf": baseline_conf,
            }
        )

    return prepared


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    class_infos = class_infos_from_json(manifest["class_mapping"])
    num_classes = len(class_infos)
    thresholds = parse_thresholds(args)
    prepared = load_cache(cache_dir, num_classes)

    baseline_confusion = sum(
        (item["baseline_conf"] for item in prepared),
        np.zeros((num_classes, num_classes), dtype=np.int64),
    )
    baseline_metrics = metrics_from_confusion(baseline_confusion, class_infos)

    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        per_image_metrics: list[dict[str, object]] = []
        for item in prepared:
            out_flat = item["pixel_labels"].copy()
            low_conf_pixels = item["pixel_conf"] < float(threshold)
            out_flat[low_conf_pixels] = item["sp_mean_labels"][
                item["flat_sp"][low_conf_pixels]
            ]
            pred = out_flat.reshape(item["shape"])
            image_confusion = compute_confusion(
                item["gt_idx"],
                pred,
                num_classes,
                item["valid_mask"],
            )
            confusion += image_confusion
            image_metrics = metrics_from_confusion(image_confusion, class_infos)
            per_image_metrics.append(
                {
                    "image": item["image_name"],
                    "miou": image_metrics["miou"],
                    "pixel_accuracy": image_metrics["pixel_accuracy"],
                }
            )

        metrics = metrics_from_confusion(confusion, class_infos)
        rows.append(
            {
                "confidence_threshold": threshold,
                "miou": metrics["miou"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "delta_miou_vs_baseline": metrics["miou"] - baseline_metrics["miou"],
                "delta_acc_vs_baseline": (
                    metrics["pixel_accuracy"] - baseline_metrics["pixel_accuracy"]
                ),
                "per_image": json.dumps(per_image_metrics, ensure_ascii=False),
            }
        )
        print(
            f"[threshold {threshold:.3f}] "
            f"mIoU={metrics['miou']:.4f} "
            f"acc={metrics['pixel_accuracy']:.4f} "
            f"delta_vs_baseline={metrics['miou'] - baseline_metrics['miou']:+.4f}"
        )

    rows_sorted = sorted(
        rows,
        key=lambda row: (float(row["miou"]), float(row["pixel_accuracy"])),
        reverse=True,
    )

    csv_path = output_dir / "low_conf_threshold_sweep.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    best_by_miou = rows_sorted[0]
    safe_rows = [
        row
        for row in rows_sorted
        if float(row["delta_acc_vs_baseline"]) >= -float(args.max_acc_drop)
    ]
    best_safe = safe_rows[0] if safe_rows else None

    summary = {
        "cache_dir": str(cache_dir),
        "device": manifest["device"],
        "images_dir": manifest["images_dir"],
        "masks_dir": manifest["masks_dir"],
        "checkpoint": manifest["checkpoint"],
        "n_images": len(prepared),
        "baseline": baseline_metrics,
        "threshold_grid": thresholds,
        "best_by_miou": best_by_miou,
        "best_with_acc_drop_limit": best_safe,
        "top_thresholds": rows_sorted[: args.top_k],
        "cache_manifest": manifest,
    }
    with open(
        output_dir / "low_conf_threshold_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print("Baseline:", baseline_metrics)
    print(
        "Best by mIoU:",
        {key: value for key, value in best_by_miou.items() if key != "per_image"},
    )
    if best_safe is not None:
        print(
            "Best safe:",
            {key: value for key, value in best_safe.items() if key != "per_image"},
        )
    print(f"Saved tuning report to: {output_dir}")


if __name__ == "__main__":
    main()
