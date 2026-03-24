#!/usr/bin/env python3
"""
Tune hybrid_conservative superpixel postprocessing with on-disk logits cache.

The script saves per-image logits, superpixels, and remapped ground truth into
`--cache-dir`, then reuses that cache for repeated parameter sweeps.

Example:
  superpixel_annotator/superpixel_annotator_venv/bin/python \
    tune_hybrid_conservative.py \
    --images ../target_dataset/S1_v2/imgs/test \
    --masks ../target_dataset/S1_v2/masks/test \
    --checkpoint S1v2_S2v2_x05.pth \
    --output-dir out/hybrid_tune \
    --device mps \
    --no-noise \
    --limit 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_superpixel_postprocessing import (
    ClassInfo,
    build_hrnet_model,
    build_superpixel_adjacency,
    compute_confusion,
    compute_superpixel_majority_labels,
    compute_superpixel_mean_probs,
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
)


def parse_float_list(raw: str) -> list[float]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(item) for item in items]


def parse_int_list(raw: str) -> list[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return [int(item) for item in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune hybrid_conservative parameters on a subset while caching "
            "logits and superpixels on disk."
        )
    )
    parser.add_argument("--images", required=True, help="Directory with images.")
    parser.add_argument("--masks", required=True, help="Directory with masks.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where tuning CSV/JSON will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory for cached logits/superpixels. Defaults to "
            "<output-dir>/logits_cache."
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore existing cache files and rebuild them from the model.",
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
        "--confidence-thresholds",
        default="0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
        help="Comma-separated confidence thresholds to test.",
    )
    parser.add_argument(
        "--small-component-sizes",
        default="1,2,3,4,5",
        help="Comma-separated component-size limits to test.",
    )
    parser.add_argument(
        "--neighbor-ratios",
        default="0.5,0.6,0.7,0.8,0.9,0.95",
        help="Comma-separated neighbor support ratios to test.",
    )
    parser.add_argument(
        "--reference-confidence-threshold",
        type=float,
        default=0.75,
        help="Reference threshold for low_confidence_mean_proba comparisons.",
    )
    parser.add_argument(
        "--max-acc-drop",
        type=float,
        default=0.005,
        help="Max allowed pixel-accuracy drop for the 'safe' best config.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top configs to include in the summary JSON.",
    )
    return parser.parse_args()


def class_infos_to_json(class_infos: list[ClassInfo]) -> list[dict[str, object]]:
    return [
        {
            "mask_code": info.code,
            "name": info.name,
            "color_rgb": list(info.color_rgb),
        }
        for info in class_infos
    ]


def class_infos_from_json(items: list[dict[str, object]]) -> list[ClassInfo]:
    return [
        ClassInfo(
            code=int(item["mask_code"]),
            name=str(item["name"]),
            color_rgb=tuple(int(v) for v in item["color_rgb"]),
        )
        for item in items
    ]


def build_cache(
    args: argparse.Namespace,
    cache_dir: Path,
) -> tuple[list[dict[str, object]], list[ClassInfo], str, dict[str, object]]:
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

    manifest_entries: list[dict[str, object]] = []
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
        ).astype(np.float32)
        superpixels = compute_superpixels(
            image_bgr=image_bgr,
            method=args.sp_method,
            slic_n_segments=args.n_segments,
            slic_compactness=args.compactness,
            slic_sigma=args.slic_sigma,
            felz_scale=args.felz_scale,
            felz_sigma=args.felz_sigma,
            felz_min_size=args.felz_min_size,
        ).astype(np.int32)

        cache_file = cache_dir / f"{idx:03d}_{image_path.stem}.npz"
        np.savez_compressed(
            cache_file,
            logits=logits_np,
            superpixels=superpixels,
            gt_idx=gt_idx.astype(np.int32),
            valid_mask=valid_mask.astype(bool),
        )
        manifest_entries.append(
            {
                "image_name": image_path.name,
                "mask_name": mask_path.name,
                "cache_file": cache_file.name,
            }
        )
        print(f"[cache {idx}/{len(pairs)}] saved {cache_file.name}")

    manifest = {
        "images_dir": str(Path(args.images).resolve()),
        "masks_dir": str(Path(args.masks).resolve()),
        "checkpoint": str(checkpoint_path),
        "device": device,
        "n_images": len(pairs),
        "class_mapping": class_infos_to_json(class_infos),
        "entries": manifest_entries,
        "settings": {
            "sp_method": args.sp_method,
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "slic_sigma": args.slic_sigma,
            "felz_scale": args.felz_scale,
            "felz_sigma": args.felz_sigma,
            "felz_min_size": args.felz_min_size,
            "pad_align": args.pad_align,
            "patch_size_limit": args.patch_size_limit,
            "patch_size": args.patch_size,
            "patch_stride": args.patch_stride,
            "noise_enabled": not args.no_noise,
            "noise_std": resolved_noise_std(args),
            "noise_seed": args.noise_seed,
            "noise_weight_key": args.noise_weight_key,
        },
    }
    with open(cache_dir / "cache_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    return manifest_entries, class_infos, device, manifest


def load_cache(
    cache_dir: Path,
) -> tuple[list[dict[str, object]], list[ClassInfo], str, dict[str, object]]:
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    entries = manifest["entries"]
    for entry in entries:
        cache_file = cache_dir / str(entry["cache_file"])
        if not cache_file.exists():
            raise FileNotFoundError(f"Missing cache file: {cache_file}")

    class_infos = class_infos_from_json(manifest["class_mapping"])
    return entries, class_infos, str(manifest["device"]), manifest


def component_stats(
    sp_labels: np.ndarray,
    adjacency: list[dict[int, int]],
    sp_confidence: np.ndarray,
    counts: np.ndarray,
    eligible_mask: np.ndarray,
) -> list[dict[str, object]]:
    visited = np.zeros(len(sp_labels), dtype=bool)
    components: list[dict[str, object]] = []

    for start in range(len(sp_labels)):
        if visited[start]:
            continue

        target_label = int(sp_labels[start])
        stack = [start]
        component: list[int] = []
        visited[start] = True

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if visited[neighbor] or int(sp_labels[neighbor]) != target_label:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        comp_arr = np.array(component, dtype=np.int32)
        component_set = set(component)
        neighbor_votes: dict[int, int] = {}
        for node in component:
            for neighbor, weight in adjacency[node].items():
                if neighbor in component_set:
                    continue
                neighbor_label = int(sp_labels[neighbor])
                if neighbor_label == target_label:
                    continue
                neighbor_votes[neighbor_label] = (
                    neighbor_votes.get(neighbor_label, 0) + int(weight)
                )

        total_support = int(sum(neighbor_votes.values()))
        if neighbor_votes:
            best_label, best_support = max(
                neighbor_votes.items(),
                key=lambda item: (item[1], -item[0]),
            )
            best_label = int(best_label)
            best_support = int(best_support)
        else:
            best_label = None
            best_support = 0

        total_component_px = float(np.maximum(counts[comp_arr].sum(), 1.0))
        comp_conf = float(
            (sp_confidence[comp_arr] * counts[comp_arr]).sum() / total_component_px
        )
        components.append(
            {
                "nodes": comp_arr,
                "size": int(len(component)),
                "eligible": bool(np.any(eligible_mask[comp_arr])),
                "component_conf": comp_conf,
                "best_label": best_label,
                "best_support": best_support,
                "total_support": total_support,
            }
        )

    return components


def prepare_cached_item(
    entry: dict[str, object],
    cache_dir: Path,
    num_classes: int,
) -> dict[str, object]:
    cache_file = cache_dir / str(entry["cache_file"])
    with np.load(cache_file, allow_pickle=False) as data:
        logits_np = data["logits"].astype(np.float32)
        superpixels = data["superpixels"].astype(np.int32)
        gt_idx = data["gt_idx"].astype(np.int32)
        valid_mask = data["valid_mask"].astype(bool)

    flat_sp = superpixels.reshape(-1)
    num_sp = int(flat_sp.max()) + 1
    probs = torch.softmax(torch.from_numpy(logits_np), dim=0).numpy().astype(np.float32)
    flat_probs = probs.reshape(num_classes, -1)
    pixel_labels = flat_probs.argmax(axis=0).astype(np.int32)
    pixel_conf = flat_probs.max(axis=0).astype(np.float32)
    mean_probs, counts = compute_superpixel_mean_probs(probs, flat_sp, num_sp)
    sp_mean_labels = mean_probs.argmax(axis=0).astype(np.int32)
    sp_confidence = mean_probs.max(axis=0).astype(np.float32)
    adjacency = build_superpixel_adjacency(superpixels)
    baseline_sp_labels = compute_superpixel_majority_labels(
        pixel_labels,
        flat_sp,
        num_sp,
        num_classes,
    )
    baseline_pred = pixel_labels.reshape(superpixels.shape)
    baseline_conf = compute_confusion(gt_idx, baseline_pred, num_classes, valid_mask)

    return {
        "image_name": str(entry["image_name"]),
        "shape": superpixels.shape,
        "gt_idx": gt_idx,
        "valid_mask": valid_mask,
        "flat_sp": flat_sp,
        "pixel_labels": pixel_labels,
        "pixel_conf": pixel_conf,
        "sp_mean_labels": sp_mean_labels,
        "sp_confidence": sp_confidence,
        "counts": counts,
        "adjacency": adjacency,
        "baseline_sp_labels": baseline_sp_labels,
        "baseline_conf": baseline_conf,
    }


def evaluate_low_confidence(
    prepared: list[dict[str, object]],
    num_classes: int,
    class_infos: list[ClassInfo],
    threshold: float,
) -> dict[str, object]:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for item in prepared:
        out_flat = item["pixel_labels"].copy()
        low_conf_pixels = item["pixel_conf"] < float(threshold)
        out_flat[low_conf_pixels] = item["sp_mean_labels"][item["flat_sp"][low_conf_pixels]]
        pred = out_flat.reshape(item["shape"])
        confusion += compute_confusion(
            item["gt_idx"],
            pred,
            num_classes,
            item["valid_mask"],
        )
    return metrics_from_confusion(confusion, class_infos)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "logits_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cache or not (cache_dir / "cache_manifest.json").exists():
        entries, class_infos, device, cache_manifest = build_cache(args, cache_dir)
    else:
        entries, class_infos, device, cache_manifest = load_cache(cache_dir)
        print(f"Loaded existing cache from: {cache_dir}")

    num_classes = len(class_infos)
    prepared = [prepare_cached_item(entry, cache_dir, num_classes) for entry in entries]

    baseline_confusion = sum(
        (item["baseline_conf"] for item in prepared),
        np.zeros((num_classes, num_classes), dtype=np.int64),
    )
    baseline_metrics = metrics_from_confusion(baseline_confusion, class_infos)
    reference_low_conf = evaluate_low_confidence(
        prepared,
        num_classes,
        class_infos,
        threshold=args.reference_confidence_threshold,
    )

    thresholds = parse_float_list(args.confidence_thresholds)
    sizes = parse_int_list(args.small_component_sizes)
    ratios = parse_float_list(args.neighbor_ratios)

    precomputed_by_threshold: dict[float, list[dict[str, object]]] = {}
    for threshold in thresholds:
        per_threshold_items: list[dict[str, object]] = []
        for item in prepared:
            out_flat = item["pixel_labels"].copy()
            low_conf_pixels = item["pixel_conf"] < float(threshold)
            out_flat[low_conf_pixels] = item["sp_mean_labels"][item["flat_sp"][low_conf_pixels]]

            stage1_sp_labels = compute_superpixel_majority_labels(
                out_flat,
                item["flat_sp"],
                len(item["sp_mean_labels"]),
                num_classes,
            )
            eligible_mask = stage1_sp_labels != item["baseline_sp_labels"]
            components = component_stats(
                stage1_sp_labels,
                item["adjacency"],
                item["sp_confidence"],
                item["counts"],
                eligible_mask,
            )
            per_threshold_items.append(
                {
                    "image_name": item["image_name"],
                    "shape": item["shape"],
                    "gt_idx": item["gt_idx"],
                    "valid_mask": item["valid_mask"],
                    "flat_sp": item["flat_sp"],
                    "stage1_sp_labels": stage1_sp_labels,
                    "components": components,
                }
            )
        precomputed_by_threshold[threshold] = per_threshold_items
        print(f"[prepare] threshold={threshold:.3f} ready")

    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        for size, ratio in itertools.product(sizes, ratios):
            confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
            per_image_metrics: list[dict[str, object]] = []
            for item in precomputed_by_threshold[threshold]:
                final_labels = item["stage1_sp_labels"].copy()
                for component in item["components"]:
                    if int(component["size"]) > size:
                        continue
                    if not bool(component["eligible"]):
                        continue
                    if float(component["component_conf"]) >= float(threshold):
                        continue
                    if component["best_label"] is None or int(component["total_support"]) <= 0:
                        continue
                    if (
                        float(component["best_support"])
                        / float(component["total_support"])
                    ) < ratio:
                        continue
                    final_labels[component["nodes"]] = int(component["best_label"])

                pred = final_labels[item["flat_sp"]].reshape(item["shape"])
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
                    "small_component_superpixels": size,
                    "hybrid_neighbor_ratio": ratio,
                    "miou": metrics["miou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "delta_miou_vs_baseline": metrics["miou"] - baseline_metrics["miou"],
                    "delta_acc_vs_baseline": (
                        metrics["pixel_accuracy"] - baseline_metrics["pixel_accuracy"]
                    ),
                    "delta_miou_vs_low_conf_reference": (
                        metrics["miou"] - reference_low_conf["miou"]
                    ),
                    "delta_acc_vs_low_conf_reference": (
                        metrics["pixel_accuracy"]
                        - reference_low_conf["pixel_accuracy"]
                    ),
                    "per_image": json.dumps(per_image_metrics, ensure_ascii=False),
                }
            )
        print(f"[sweep] threshold={threshold:.3f} done")

    rows_sorted = sorted(
        rows,
        key=lambda row: (float(row["miou"]), float(row["pixel_accuracy"])),
        reverse=True,
    )

    csv_path = output_dir / "hybrid_grid_search.csv"
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
        "device": device,
        "cache_dir": str(cache_dir),
        "n_images": len(prepared),
        "baseline": baseline_metrics,
        "low_confidence_reference": {
            "threshold": args.reference_confidence_threshold,
            "metrics": reference_low_conf,
        },
        "grid_sizes": {
            "confidence_thresholds": thresholds,
            "small_component_sizes": sizes,
            "neighbor_ratios": ratios,
            "n_configs": len(rows_sorted),
        },
        "best_by_miou": best_by_miou,
        "best_with_acc_drop_limit": best_safe,
        "top_configs": rows_sorted[: args.top_k],
        "cache_manifest": cache_manifest,
    }
    with open(
        output_dir / "hybrid_grid_search_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print("Baseline:", baseline_metrics)
    print(
        "Low-confidence reference:",
        {
            "threshold": args.reference_confidence_threshold,
            "miou": reference_low_conf["miou"],
            "pixel_accuracy": reference_low_conf["pixel_accuracy"],
        },
    )
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
    print(f"Saved logits cache to: {cache_dir}")


if __name__ == "__main__":
    main()
