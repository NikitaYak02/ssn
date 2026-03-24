#!/usr/bin/env python3
"""
Benchmark simple superpixel-based postprocessing methods on a small subset.

The benchmark compares:
  - baseline pixel argmax
  - mean_proba
  - confidence_gated_mean_proba
  - low_confidence_mean_proba
  - prior_corrected_mean_proba
  - small_region_cleanup
  - hybrid_conservative
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_superpixel_postprocessing import (
    build_hrnet_model,
    compute_confusion,
    compute_superpixels,
    discover_image_pairs,
    get_named_class_sets,
    import_petroscope_bits,
    infer_class_infos,
    load_image_bgr,
    load_mask_codes,
    metrics_from_confusion,
    parse_class_codes,
    perturb_checkpoint,
    remap_mask_codes,
    resolve_device,
    resolved_noise_std,
    run_model_logits,
    superpixel_postprocess,
)


METHODS: list[tuple[str, str]] = [
    ("baseline", "Pixel argmax without superpixels"),
    ("mean_proba", "Average probabilities per superpixel"),
    ("confidence_gated_mean_proba", "Apply superpixel relabeling only for confident regions"),
    ("low_confidence_mean_proba", "Only overwrite low-confidence pixels"),
    ("prior_corrected_mean_proba", "Average probabilities with image-level prior correction"),
    ("small_region_cleanup", "Merge tiny superpixel islands into dominant neighbors"),
    ("hybrid_conservative", "Low-confidence overwrite plus conservative island cleanup"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark simple superpixel postprocessing methods."
    )
    parser.add_argument("--images", required=True, help="Directory with images.")
    parser.add_argument("--masks", required=True, help="Directory with masks.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison CSV/JSON will be written.",
    )
    parser.add_argument(
        "--petroscope-root",
        default=None,
        help="Optional local petroscope checkout.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many image/mask pairs to evaluate from the start of the test set.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Std of Gaussian noise added to the selected tensor.",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help=(
            "Disable checkpoint degradation entirely. Equivalent to running "
            "with the original weights regardless of --noise-std."
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Seed for weight perturbation.",
    )
    parser.add_argument(
        "--noise-weight-key",
        default="model.backbone.conv1.weight",
        help="Checkpoint tensor key to perturb.",
    )
    parser.add_argument(
        "--sp-method",
        default="felzenszwalb",
        choices=["felzenszwalb", "slic"],
        help="Superpixel algorithm.",
    )
    parser.add_argument("--n-segments", type=int, default=800)
    parser.add_argument("--compactness", type=float, default=20.0)
    parser.add_argument("--slic-sigma", type=float, default=0.0)
    parser.add_argument("--felz-scale", type=float, default=200.0)
    parser.add_argument("--felz-sigma", type=float, default=1.0)
    parser.add_argument("--felz-min-size", type=int, default=10)
    parser.add_argument("--pad-align", type=int, default=16)
    parser.add_argument("--patch-size-limit", type=int, default=1800)
    parser.add_argument("--patch-size", type=int, default=1536)
    parser.add_argument("--patch-stride", type=int, default=1024)
    parser.add_argument(
        "--class-codes",
        default=None,
        help="Optional comma-separated class-code mapping.",
    )
    parser.add_argument(
        "--unknown-label-policy",
        default="error",
        choices=["error", "ignore"],
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Threshold for confidence-gated and low-confidence methods.",
    )
    parser.add_argument(
        "--prior-power",
        type=float,
        default=0.5,
        help="Exponent for prior correction.",
    )
    parser.add_argument(
        "--small-component-superpixels",
        type=int,
        default=3,
        help="Max island size in superpixels for cleanup.",
    )
    parser.add_argument(
        "--hybrid-neighbor-ratio",
        type=float,
        default=0.6,
        help="Minimum neighbor support ratio for hybrid_conservative cleanup.",
    )
    return parser.parse_args()


def method_prediction(
    method: str,
    logits_np: np.ndarray,
    superpixels: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if method == "baseline":
        return logits_np.argmax(axis=0).astype(np.int32)
    return superpixel_postprocess(
        logits_np=logits_np,
        superpixels=superpixels,
        vote_mode=method,
        confidence_threshold=args.confidence_threshold,
        prior_power=args.prior_power,
        small_component_superpixels=args.small_component_superpixels,
        hybrid_neighbor_ratio=args.hybrid_neighbor_ratio,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_image_pairs(args.images, args.masks)
    if not pairs:
        raise RuntimeError("No matching image/mask pairs were found.")
    pairs = pairs[: args.limit]

    device = resolve_device(args.device)
    hrnet_cls, lumenstone_classes = import_petroscope_bits(args.petroscope_root)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    perturb_checkpoint(
        checkpoint,
        weight_key=args.noise_weight_key,
        noise_std=resolved_noise_std(args),
        noise_seed=args.noise_seed,
    )
    model = build_hrnet_model(hrnet_cls, checkpoint, device)

    mask_codes_seen: set[int] = set()
    for _, mask_path in pairs:
        mask_codes_seen.update(np.unique(load_mask_codes(mask_path)).tolist())

    class_infos = infer_class_infos(
        class_codes_arg=parse_class_codes(args.class_codes),
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        named_sets=get_named_class_sets(lumenstone_classes),
        mask_codes_seen=mask_codes_seen,
    )

    agg_confusions = {
        method: np.zeros((len(class_infos), len(class_infos)), dtype=np.int64)
        for method, _ in METHODS
    }

    rows: list[dict[str, object]] = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        image_bgr = load_image_bgr(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_codes = load_mask_codes(mask_path)
        gt_idx, valid_mask = remap_mask_codes(
            mask_codes, class_infos, args.unknown_label_policy
        )

        logits_np = run_model_logits(
            model=model,
            image_rgb=image_rgb,
            device=device,
            pad_align=args.pad_align,
            patch_size_limit=args.patch_size_limit,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
        )
        superpixels = compute_superpixels(
            image_bgr=image_bgr,
            method=args.sp_method,
            slic_n_segments=args.n_segments,
            slic_compactness=args.compactness,
            slic_sigma=args.slic_sigma,
            felz_scale=args.felz_scale,
            felz_sigma=args.felz_sigma,
            felz_min_size=args.felz_min_size,
        )

        image_metrics: dict[str, dict[str, float]] = {}
        for method, description in METHODS:
            pred_idx = method_prediction(method, logits_np, superpixels, args)
            confusion = compute_confusion(
                gt_idx, pred_idx, len(class_infos), valid_mask
            )
            agg_confusions[method] += confusion
            metrics = metrics_from_confusion(confusion, class_infos)
            image_metrics[method] = {
                "miou": float(metrics["miou"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
            }
            rows.append(
                {
                    "image": image_path.name,
                    "method": method,
                    "description": description,
                    "miou": metrics["miou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "valid_pixels": int(valid_mask.sum()),
                    "n_superpixels": int(np.unique(superpixels).size),
                }
            )

        baseline = image_metrics["baseline"]
        best_method = max(
            image_metrics.items(),
            key=lambda item: item[1]["miou"],
        )
        print(
            f"[{idx}/{len(pairs)}] {image_path.name}: "
            f"baseline mIoU={baseline['miou']:.4f}, "
            f"best={best_method[0]} ({best_method[1]['miou']:.4f})"
        )

    csv_path = output_dir / "comparison_per_image.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "method",
                "description",
                "miou",
                "pixel_accuracy",
                "valid_pixels",
                "n_superpixels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_methods: dict[str, dict[str, object]] = {}
    baseline_metrics = metrics_from_confusion(agg_confusions["baseline"], class_infos)
    mean_proba_metrics = metrics_from_confusion(
        agg_confusions["mean_proba"], class_infos
    )
    for method, description in METHODS:
        metrics = metrics_from_confusion(agg_confusions[method], class_infos)
        summary_methods[method] = {
            "description": description,
            "miou": metrics["miou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "delta_miou_vs_baseline": metrics["miou"] - baseline_metrics["miou"],
            "delta_acc_vs_baseline": (
                metrics["pixel_accuracy"] - baseline_metrics["pixel_accuracy"]
            ),
            "delta_miou_vs_mean_proba": metrics["miou"] - mean_proba_metrics["miou"],
            "delta_acc_vs_mean_proba": (
                metrics["pixel_accuracy"] - mean_proba_metrics["pixel_accuracy"]
            ),
            "per_class_iou": metrics["per_class_iou"],
        }

    summary = {
        "device": device,
        "images_dir": str(Path(args.images).resolve()),
        "masks_dir": str(Path(args.masks).resolve()),
        "checkpoint": str(checkpoint_path),
        "n_images": len(pairs),
        "methods": summary_methods,
        "class_mapping": [
            {
                "model_index": idx,
                "mask_code": info.code,
                "name": info.name,
                "color_rgb": list(info.color_rgb),
            }
            for idx, info in enumerate(class_infos)
        ],
        "settings": {
            "sp_method": args.sp_method,
            "confidence_threshold": args.confidence_threshold,
            "prior_power": args.prior_power,
            "small_component_superpixels": args.small_component_superpixels,
            "hybrid_neighbor_ratio": args.hybrid_neighbor_ratio,
            "noise_enabled": not args.no_noise,
            "noise_std": resolved_noise_std(args),
            "noise_seed": args.noise_seed,
            "noise_weight_key": args.noise_weight_key,
        },
    }
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print("Aggregated over", len(pairs), "images:")
    for method, _ in METHODS:
        method_summary = summary_methods[method]
        print(
            f"{method:30s} "
            f"mIoU={method_summary['miou']:.4f} "
            f"acc={method_summary['pixel_accuracy']:.4f} "
            f"delta_vs_baseline={method_summary['delta_miou_vs_baseline']:+.4f}"
        )
    print(f"Saved report to: {output_dir}")


if __name__ == "__main__":
    main()
