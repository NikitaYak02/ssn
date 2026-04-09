#!/usr/bin/env python3
"""
Batch evaluation of segmentation quality before and after superpixel postprocessing.

The script mirrors the notebook logic from test_sp_postproc.ipynb:
1. load an HRNet checkpoint from petroscope,
2. perturb a selected weight tensor with Gaussian noise,
3. run segmentation on a folder of images,
4. postprocess predictions inside superpixels,
5. compute metrics before and after postprocessing,
6. save regions where the postprocessing helped or hurt.

Example:
  superpixel_annotator/superpixel_annotator_venv/bin/python \
    evaluate_superpixel_postprocessing.py \
    --images ../target_dataset/S1_v2/imgs/test \
    --masks ../target_dataset/S1_v2/masks/test \
    --checkpoint S1v2_S2v2_x05.pth \
    --output-dir out/sp_postproc_eval \
    --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as cc_label
from skimage.segmentation import felzenszwalb, slic

from superpixel_refinement_strategies import (
    SuperpixelRefinementStrategy,
    build_legacy_strategy,
    named_strategy_catalog,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ClassInfo:
    code: int
    name: str
    color_rgb: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate segmentation quality before and after superpixel "
            "postprocessing."
        )
    )
    parser.add_argument("--images", required=True, help="Directory with images.")
    parser.add_argument("--masks", required=True, help="Directory with masks.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a petroscope HRNet checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save metrics and change visualizations.",
    )
    parser.add_argument(
        "--petroscope-root",
        default=None,
        help=(
            "Optional root directory of the petroscope repository. "
            "Useful if the installed package does not contain HRNet."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. auto prefers mps, then cuda, then cpu.",
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
    parser.add_argument(
        "--vote-mode",
        default="mean_proba",
        choices=[
            "mean_proba",
            "majority_argmax",
            "confidence_gated_mean_proba",
            "low_confidence_mean_proba",
            "prior_corrected_mean_proba",
            "small_region_cleanup",
            "hybrid_conservative",
        ],
        help="How to aggregate predictions inside a superpixel.",
    )
    parser.add_argument(
        "--strategy-id",
        default=None,
        help=(
            "Optional named refinement strategy. Supports legacy aliases and "
            "the generated novel_XX_YY candidate pool."
        ),
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print available strategy ids and exit.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help=(
            "Threshold used by confidence-gated and low-confidence "
            "superpixel modes."
        ),
    )
    parser.add_argument(
        "--prior-power",
        type=float,
        default=0.5,
        help=(
            "Exponent for prior correction in "
            "prior_corrected_mean_proba."
        ),
    )
    parser.add_argument(
        "--small-component-superpixels",
        type=int,
        default=3,
        help=(
            "Maximum connected-component size in superpixels for "
            "small_region_cleanup."
        ),
    )
    parser.add_argument(
        "--hybrid-neighbor-ratio",
        type=float,
        default=0.6,
        help=(
            "Minimum fraction of neighboring boundary support needed to "
            "merge a small island in hybrid_conservative."
        ),
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=800,
        help="SLIC n_segments.",
    )
    parser.add_argument(
        "--compactness",
        type=float,
        default=20.0,
        help="SLIC compactness.",
    )
    parser.add_argument(
        "--slic-sigma",
        type=float,
        default=0.0,
        help="SLIC sigma.",
    )
    parser.add_argument(
        "--felz-scale",
        type=float,
        default=200.0,
        help="Felzenszwalb scale.",
    )
    parser.add_argument(
        "--felz-sigma",
        type=float,
        default=1.0,
        help="Felzenszwalb sigma.",
    )
    parser.add_argument(
        "--felz-min-size",
        type=int,
        default=10,
        help="Felzenszwalb min_size.",
    )
    parser.add_argument(
        "--pad-align",
        type=int,
        default=16,
        help="Pad images to a multiple of this value before inference.",
    )
    parser.add_argument(
        "--patch-size-limit",
        type=int,
        default=1800,
        help=(
            "If H or W is larger than this value, run tiled inference "
            "to reduce memory pressure."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1536,
        help="Tile size for large-image inference.",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=1024,
        help="Stride for tiled inference.",
    )
    parser.add_argument(
        "--class-codes",
        default=None,
        help=(
            "Comma-separated raw mask codes in model channel order. "
            "Example: 0,1,2,3,4,5,6,7,8,11"
        ),
    )
    parser.add_argument(
        "--unknown-label-policy",
        default="error",
        choices=["error", "ignore"],
        help="What to do with mask labels absent from class-codes.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save baseline and postprocessed predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N matched image/mask pairs.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        if device_arg == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device_arg

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def maybe_add_petroscope_root(petroscope_root: str | None) -> None:
    if not petroscope_root:
        return

    root = Path(petroscope_root).resolve()
    if (root / "petroscope").is_dir():
        sys.path.insert(0, str(root))
        return
    if root.name == "petroscope" and root.is_dir():
        sys.path.insert(0, str(root.parent))
        return
    raise FileNotFoundError(
        f"Could not locate a petroscope package under: {root}"
    )


def import_petroscope_bits(
    petroscope_root: str | None,
) -> tuple[type[Any], Any | None]:
    maybe_add_petroscope_root(petroscope_root)
    try:
        from petroscope.segmentation.models.hrnet.model import HRNet
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise RuntimeError(
            "Failed to import petroscope HRNet. If your installed petroscope "
            "package does not contain HRNet, pass --petroscope-root pointing "
            "to a newer petroscope checkout."
        ) from exc

    try:
        from petroscope.segmentation.classes import LumenStoneClasses
    except Exception:
        LumenStoneClasses = None

    return HRNet, LumenStoneClasses


def discover_image_pairs(images_dir: str, masks_dir: str) -> list[tuple[Path, Path]]:
    images = []
    for image_path in sorted(Path(images_dir).iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = find_mask_for_image(image_path, masks_dir)
        if mask_path is not None:
            images.append((image_path, mask_path))
    return images


def find_mask_for_image(image_path: Path, masks_dir: str) -> Path | None:
    direct = Path(masks_dir) / image_path.name
    if direct.exists():
        return direct
    for ext in IMAGE_EXTENSIONS:
        alt = Path(masks_dir) / f"{image_path.stem}{ext}"
        if alt.exists():
            return alt
    return None


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_mask_codes(path: Path) -> np.ndarray:
    mask = Image.open(path)
    if mask.mode in ("P", "L", "I", "I;16"):
        arr = np.array(mask)
    else:
        arr = np.array(mask.convert("RGB"))[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2-D after conversion, got {arr.shape}")
    return arr.astype(np.int32)


def parse_class_codes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    items = [item.strip() for item in raw.replace(" ", "").split(",") if item.strip()]
    if not items:
        raise ValueError("--class-codes was provided but is empty.")
    return [int(item) for item in items]


def default_colors(n: int) -> list[tuple[int, int, int]]:
    palette = [
        (0, 0, 0),
        (255, 165, 0),
        (154, 205, 50),
        (255, 69, 0),
        (0, 191, 255),
        (169, 169, 169),
        (47, 79, 79),
        (255, 255, 0),
        (238, 130, 238),
        (85, 107, 47),
        (160, 82, 45),
        (72, 61, 139),
        (0, 128, 0),
        (0, 0, 139),
        (139, 0, 139),
    ]
    if n <= len(palette):
        return palette[:max(1, n)]

    colors: list[tuple[int, int, int]] = list(palette)
    for idx in range(len(palette), max(1, n)):
        extra_idx = idx - len(palette)
        hue = ((extra_idx * 0.6180339887498949) % 1.0) * 179.0
        hsv = np.uint8([[[int(hue), 200, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return colors


def get_named_class_sets(
    lumenstone_classes: Any | None,
) -> dict[str, list[ClassInfo]]:
    if lumenstone_classes is None:
        return {}

    def classset_to_infos(classset: Any) -> list[ClassInfo]:
        return [
            ClassInfo(
                code=int(cl.code),
                name=str(cl.label),
                color_rgb=tuple(int(v) for v in cl.color_rgb),
            )
            for cl in classset.classes
        ]

    named: dict[str, list[ClassInfo]] = {}
    try:
        named["S1"] = classset_to_infos(lumenstone_classes.S1())
    except Exception:
        pass
    try:
        named["S2"] = classset_to_infos(lumenstone_classes.S2())
    except Exception:
        pass
    try:
        named["S3"] = classset_to_infos(lumenstone_classes.S3())
    except Exception:
        pass
    if "S1" in named and "S2" in named:
        merged: dict[int, ClassInfo] = {
            info.code: info for info in named["S1"] + named["S2"]
        }
        named["S1_S2"] = [merged[code] for code in sorted(merged)]
    return named


def infer_class_infos(
    class_codes_arg: list[int] | None,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    named_sets: dict[str, list[ClassInfo]],
    mask_codes_seen: set[int],
) -> list[ClassInfo]:
    n_classes = int(checkpoint["n_classes"])

    if class_codes_arg is not None:
        class_codes = class_codes_arg
        if len(class_codes) != n_classes:
            raise ValueError(
                f"--class-codes has {len(class_codes)} items, but checkpoint "
                f"expects {n_classes} classes."
            )
        colors = default_colors(len(class_codes))
        return [
            ClassInfo(code=code, name=f"class_{code}", color_rgb=colors[i])
            for i, code in enumerate(class_codes)
        ]

    stem = checkpoint_path.stem.lower()
    for key in ("S1_S2", "S3", "S1", "S2"):
        key_l = key.lower()
        if key == "S1_S2":
            matched = "s1" in stem and "s2" in stem
        else:
            matched = key_l in stem
        if matched and key in named_sets:
            infos = named_sets[key]
            if len(infos) == n_classes:
                return infos

    sorted_codes = sorted(mask_codes_seen)
    if len(sorted_codes) == n_classes:
        colors = default_colors(len(sorted_codes))
        return [
            ClassInfo(code=code, name=f"class_{code}", color_rgb=colors[i])
            for i, code in enumerate(sorted_codes)
        ]

    raise RuntimeError(
        "Could not infer class-code mapping automatically. Pass --class-codes "
        "explicitly in the order expected by the model outputs."
    )


def remap_mask_codes(
    mask_codes: np.ndarray,
    class_infos: list[ClassInfo],
    unknown_label_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    code_to_index = {info.code: idx for idx, info in enumerate(class_infos)}
    gt = np.full(mask_codes.shape, -1, dtype=np.int32)
    valid_mask = np.zeros(mask_codes.shape, dtype=bool)

    for code, idx in code_to_index.items():
        mask = mask_codes == code
        if np.any(mask):
            gt[mask] = idx
            valid_mask[mask] = True

    if unknown_label_policy == "error":
        unknown = sorted(int(v) for v in np.unique(mask_codes[~valid_mask]))
        if unknown:
            raise ValueError(
                "Mask contains codes that are absent from class mapping: "
                f"{unknown}"
            )

    return gt, valid_mask


def compute_confusion(
    gt_idx: np.ndarray,
    pred_idx: np.ndarray,
    num_classes: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    gt_flat = gt_idx[valid_mask].ravel()
    pred_flat = pred_idx[valid_mask].ravel()
    if gt_flat.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    bins = np.bincount(
        num_classes * gt_flat + pred_flat,
        minlength=num_classes * num_classes,
    )
    return bins.reshape(num_classes, num_classes).astype(np.int64)


def metrics_from_confusion(
    confusion: np.ndarray,
    class_infos: list[ClassInfo],
) -> dict[str, Any]:
    total = int(confusion.sum())
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    unions = tp + fp + fn

    per_class_iou: dict[str, float | None] = {}
    valid_ious: list[float] = []
    for idx, info in enumerate(class_infos):
        key = f"{info.code}:{info.name}"
        if unions[idx] <= 0:
            per_class_iou[key] = None
            continue
        value = float(tp[idx] / unions[idx])
        per_class_iou[key] = value
        valid_ious.append(value)

    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return {
        "pixel_accuracy": accuracy,
        "miou": miou,
        "valid_pixels": total,
        "per_class_iou": per_class_iou,
    }


def summarize_prediction(
    gt_idx: np.ndarray,
    pred_idx: np.ndarray,
    class_infos: list[ClassInfo],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    confusion = compute_confusion(gt_idx, pred_idx, len(class_infos), valid_mask)
    return metrics_from_confusion(confusion, class_infos)


def pad_image_to_multiple(image_rgb: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    height, width = image_rgb.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image_rgb, height, width
    padded = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, height, width


def run_model_logits_full(
    model: Any,
    image_rgb: np.ndarray,
    device: str,
    pad_align: int,
) -> np.ndarray:
    padded, height, width = pad_image_to_multiple(image_rgb, pad_align)
    tensor = (
        torch.from_numpy(padded)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    with torch.inference_mode():
        logits = model.model(tensor)
    if isinstance(logits, (tuple, list)):
        logits = logits[-1]
    logits_np = logits.detach().cpu().numpy()[0, :, :height, :width]
    return logits_np.astype(np.float32, copy=False)


def sliding_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_model_logits(
    model: Any,
    image_rgb: np.ndarray,
    device: str,
    pad_align: int,
    patch_size_limit: int,
    patch_size: int,
    patch_stride: int,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height <= patch_size_limit and width <= patch_size_limit:
        return run_model_logits_full(model, image_rgb, device, pad_align)

    channels = int(model.n_classes)
    logits_sum = np.zeros((channels, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    for y0 in sliding_starts(height, patch_size, patch_stride):
        for x0 in sliding_starts(width, patch_size, patch_stride):
            patch = image_rgb[y0 : y0 + patch_size, x0 : x0 + patch_size]
            patch_logits = run_model_logits_full(model, patch, device, pad_align)
            ph, pw = patch_logits.shape[1:]
            logits_sum[:, y0 : y0 + ph, x0 : x0 + pw] += patch_logits
            counts[y0 : y0 + ph, x0 : x0 + pw] += 1.0

    counts = np.maximum(counts, 1.0)
    return logits_sum / counts[None, ...]


def compute_superpixels(
    image_bgr: np.ndarray,
    method: str,
    slic_n_segments: int,
    slic_compactness: float,
    slic_sigma: float,
    felz_scale: float,
    felz_sigma: float,
    felz_min_size: int,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
    if method == "slic":
        labels = slic(
            image_rgb,
            n_segments=slic_n_segments,
            compactness=slic_compactness,
            sigma=slic_sigma,
            start_label=0,
            channel_axis=-1,
        )
    elif method == "felzenszwalb":
        labels = felzenszwalb(
            image_rgb,
            scale=felz_scale,
            sigma=felz_sigma,
            min_size=felz_min_size,
            channel_axis=-1,
        )
    else:  # pragma: no cover - argparse already protects this
        raise ValueError(f"Unknown superpixel method: {method}")
    return labels.astype(np.int32)


def compute_superpixel_mean_probs(
    probs: np.ndarray,
    flat_sp: np.ndarray,
    num_sp: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(flat_sp, minlength=num_sp).astype(np.float32)
    sums = np.stack(
        [
            np.bincount(flat_sp, weights=probs[c].reshape(-1), minlength=num_sp)
            for c in range(probs.shape[0])
        ],
        axis=0,
    )
    mean_probs = sums / np.maximum(counts, 1.0)[None, :]
    return mean_probs.astype(np.float32), counts


def compute_superpixel_weighted_probs(
    probs: np.ndarray,
    flat_sp: np.ndarray,
    num_sp: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    flat_weights = np.maximum(flat_weights, 1e-6)
    counts = np.bincount(
        flat_sp,
        weights=flat_weights,
        minlength=num_sp,
    ).astype(np.float32)
    sums = np.stack(
        [
            np.bincount(
                flat_sp,
                weights=probs[c].reshape(-1) * flat_weights,
                minlength=num_sp,
            )
            for c in range(probs.shape[0])
        ],
        axis=0,
    )
    mean_probs = sums / np.maximum(counts, 1e-6)[None, :]
    return mean_probs.astype(np.float32), counts


def compute_superpixel_mean_logits(
    logits_np: np.ndarray,
    flat_sp: np.ndarray,
    num_sp: int,
) -> np.ndarray:
    channels = logits_np.shape[0]
    sums = np.stack(
        [
            np.bincount(
                flat_sp,
                weights=logits_np[c].reshape(-1),
                minlength=num_sp,
            )
            for c in range(channels)
        ],
        axis=0,
    ).astype(np.float32)
    counts = np.bincount(flat_sp, minlength=num_sp).astype(np.float32)
    return sums / np.maximum(counts, 1.0)[None, :]


def apply_temperature_to_probs(
    probs: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if abs(float(temperature) - 1.0) < 1e-6:
        return probs.astype(np.float32, copy=False)
    adjusted = np.power(np.maximum(probs, 1e-8), 1.0 / float(temperature))
    adjusted /= np.maximum(adjusted.sum(axis=0, keepdims=True), 1e-8)
    return adjusted.astype(np.float32)


def normalize_superpixel_scores(scores: np.ndarray) -> np.ndarray:
    clipped = np.maximum(scores.astype(np.float32, copy=False), 1e-8)
    denom = np.maximum(clipped.sum(axis=0, keepdims=True), 1e-8)
    return (clipped / denom).astype(np.float32)


def compute_pixel_confidence_features(
    flat_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_conf = flat_probs.max(axis=0).astype(np.float32)
    if flat_probs.shape[0] > 1:
        top2 = np.partition(flat_probs, kth=flat_probs.shape[0] - 2, axis=0)[-2:]
        pixel_margin = (top2[-1] - top2[-2]).astype(np.float32)
    else:
        pixel_margin = pixel_conf.copy()
    entropy = -np.sum(
        flat_probs * np.log(np.maximum(flat_probs, 1e-8)),
        axis=0,
    )
    entropy /= max(np.log(max(flat_probs.shape[0], 2)), 1e-8)
    entropy = entropy.astype(np.float32)
    return pixel_conf, pixel_margin, entropy


def compute_superpixel_majority_labels(
    flat_labels: np.ndarray,
    flat_sp: np.ndarray,
    num_sp: int,
    num_classes: int,
) -> np.ndarray:
    votes = np.bincount(
        flat_sp * num_classes + flat_labels.astype(np.int32),
        minlength=num_sp * num_classes,
    ).reshape(num_sp, num_classes)
    return votes.argmax(axis=1).astype(np.int32)


def build_superpixel_adjacency(superpixels: np.ndarray) -> list[dict[int, int]]:
    num_sp = int(superpixels.max()) + 1
    adjacency = [dict() for _ in range(num_sp)]

    def add_edges(lhs: np.ndarray, rhs: np.ndarray) -> None:
        mask = lhs != rhs
        if not np.any(mask):
            return
        pairs = np.stack([lhs[mask], rhs[mask]], axis=1).astype(np.int32)
        pairs.sort(axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (u, v), count in zip(unique_pairs, counts):
            adjacency[int(u)][int(v)] = adjacency[int(u)].get(int(v), 0) + int(count)
            adjacency[int(v)][int(u)] = adjacency[int(v)].get(int(u), 0) + int(count)

    add_edges(superpixels[:, :-1], superpixels[:, 1:])
    add_edges(superpixels[:-1, :], superpixels[1:, :])
    return adjacency


def smooth_superpixel_scores(
    scores: np.ndarray,
    adjacency: list[dict[int, int]],
    steps: int,
    alpha: float,
) -> np.ndarray:
    if steps <= 0 or alpha <= 0.0:
        return normalize_superpixel_scores(scores)

    current = normalize_superpixel_scores(scores)
    for _ in range(int(steps)):
        smoothed = (1.0 - float(alpha)) * current
        for node, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            neigh_ids = np.fromiter(neighbors.keys(), dtype=np.int32)
            neigh_weights = np.fromiter(neighbors.values(), dtype=np.float32)
            neigh_weights /= np.maximum(neigh_weights.sum(), 1.0)
            smoothed[:, node] += float(alpha) * (
                current[:, neigh_ids] * neigh_weights[None, :]
            ).sum(axis=1)
        current = normalize_superpixel_scores(smoothed)
    return current


def cleanup_small_superpixel_components(
    sp_labels: np.ndarray,
    adjacency: list[dict[int, int]],
    max_component_size: int,
) -> np.ndarray:
    cleaned = sp_labels.copy()
    visited = np.zeros(len(cleaned), dtype=bool)

    for start in range(len(cleaned)):
        if visited[start]:
            continue

        target_label = int(cleaned[start])
        stack = [start]
        component: list[int] = []
        visited[start] = True

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if visited[neighbor] or int(cleaned[neighbor]) != target_label:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        if len(component) > max_component_size:
            continue

        neighbor_votes: dict[int, int] = {}
        component_set = set(component)
        for node in component:
            for neighbor, weight in adjacency[node].items():
                if neighbor in component_set:
                    continue
                neighbor_label = int(cleaned[neighbor])
                if neighbor_label == target_label:
                    continue
                neighbor_votes[neighbor_label] = (
                    neighbor_votes.get(neighbor_label, 0) + int(weight)
                )

        if not neighbor_votes:
            continue

        best_label = max(
            neighbor_votes.items(),
            key=lambda item: (item[1], -item[0]),
        )[0]
        for node in component:
            cleaned[node] = best_label

    return cleaned


def cleanup_small_superpixel_components_conservative(
    sp_labels: np.ndarray,
    adjacency: list[dict[int, int]],
    max_component_size: int,
    sp_confidence: np.ndarray,
    counts: np.ndarray,
    confidence_threshold: float,
    neighbor_ratio_threshold: float,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    cleaned = sp_labels.copy()
    visited = np.zeros(len(cleaned), dtype=bool)

    for start in range(len(cleaned)):
        if visited[start]:
            continue

        target_label = int(cleaned[start])
        stack = [start]
        component: list[int] = []
        visited[start] = True

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if visited[neighbor] or int(cleaned[neighbor]) != target_label:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)

        if len(component) > max_component_size:
            continue
        if not np.any(eligible_mask[component]):
            continue

        component_counts = counts[component]
        total_component_px = float(np.maximum(component_counts.sum(), 1.0))
        component_conf = float(
            (sp_confidence[component] * component_counts).sum() / total_component_px
        )
        if component_conf >= float(confidence_threshold):
            continue

        neighbor_votes: dict[int, int] = {}
        component_set = set(component)
        for node in component:
            for neighbor, weight in adjacency[node].items():
                if neighbor in component_set:
                    continue
                neighbor_label = int(cleaned[neighbor])
                if neighbor_label == target_label:
                    continue
                neighbor_votes[neighbor_label] = (
                    neighbor_votes.get(neighbor_label, 0) + int(weight)
                )

        if not neighbor_votes:
            continue

        total_support = sum(neighbor_votes.values())
        best_label, best_support = max(
            neighbor_votes.items(),
            key=lambda item: (item[1], -item[0]),
        )
        if total_support <= 0:
            continue
        if (best_support / float(total_support)) < float(neighbor_ratio_threshold):
            continue

        for node in component:
            cleaned[node] = best_label

    return cleaned


def resolve_refinement_strategy(
    *,
    vote_mode: str,
    strategy_id: str | None,
    confidence_threshold: float,
    prior_power: float,
    small_component_superpixels: int,
    hybrid_neighbor_ratio: float,
) -> SuperpixelRefinementStrategy:
    if strategy_id:
        catalog = named_strategy_catalog(
            confidence_threshold=confidence_threshold,
            prior_power=prior_power,
            small_component_superpixels=small_component_superpixels,
            hybrid_neighbor_ratio=hybrid_neighbor_ratio,
            include_legacy=True,
            novel_limit=100,
        )
        if strategy_id in catalog:
            return catalog[strategy_id]
        if strategy_id in {
            "mean_proba",
            "majority_argmax",
            "confidence_gated_mean_proba",
            "low_confidence_mean_proba",
            "prior_corrected_mean_proba",
            "small_region_cleanup",
            "hybrid_conservative",
        }:
            return build_legacy_strategy(
                strategy_id,
                confidence_threshold=confidence_threshold,
                prior_power=prior_power,
                small_component_superpixels=small_component_superpixels,
                hybrid_neighbor_ratio=hybrid_neighbor_ratio,
            )
        raise KeyError(f"Unknown strategy-id: {strategy_id}")
    return build_legacy_strategy(
        vote_mode,
        confidence_threshold=confidence_threshold,
        prior_power=prior_power,
        small_component_superpixels=small_component_superpixels,
        hybrid_neighbor_ratio=hybrid_neighbor_ratio,
    )


def compute_superpixel_scores(
    *,
    logits_np: np.ndarray,
    flat_sp: np.ndarray,
    num_sp: int,
    flat_probs: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_conf: np.ndarray,
    pixel_margin: np.ndarray,
    pixel_entropy: np.ndarray,
    strategy: SuperpixelRefinementStrategy,
) -> np.ndarray:
    if strategy.aggregate_mode == "majority_argmax":
        votes = np.bincount(
            flat_sp * logits_np.shape[0] + pixel_labels.astype(np.int32),
            minlength=num_sp * logits_np.shape[0],
        ).reshape(num_sp, logits_np.shape[0]).astype(np.float32)
        return normalize_superpixel_scores(votes.T)

    if strategy.aggregate_mode == "logit_mean":
        mean_logits = compute_superpixel_mean_logits(logits_np, flat_sp, num_sp)
        mean_logits_t = torch.from_numpy(mean_logits / float(strategy.temperature))
        scores = torch.softmax(mean_logits_t, dim=0).numpy().astype(np.float32)
    else:
        probs_for_aggregation = apply_temperature_to_probs(
            flat_probs,
            float(strategy.temperature),
        )
        if strategy.aggregate_mode == "mean_proba":
            scores, _ = compute_superpixel_mean_probs(
                probs_for_aggregation.reshape(logits_np.shape),
                flat_sp,
                num_sp,
            )
        else:
            if strategy.aggregate_mode == "confidence_weighted_mean":
                weights = np.power(np.maximum(pixel_conf, 1e-6), float(strategy.weight_power))
            elif strategy.aggregate_mode == "margin_weighted_mean":
                weights = np.power(np.maximum(pixel_margin, 1e-6), float(strategy.weight_power))
            elif strategy.aggregate_mode == "entropy_weighted_mean":
                weights = np.power(
                    np.maximum(1.0 - pixel_entropy, 1e-6),
                    float(strategy.weight_power),
                )
            else:  # pragma: no cover - validated by strategy registry
                raise ValueError(f"Unknown aggregate mode: {strategy.aggregate_mode}")
            scores, _ = compute_superpixel_weighted_probs(
                probs_for_aggregation.reshape(logits_np.shape),
                flat_sp,
                num_sp,
                weights,
            )

    if float(strategy.prior_power) > 0.0:
        priors = np.maximum(flat_probs.mean(axis=1), 1e-6).astype(np.float32)
        scores = scores / np.power(priors[:, None], float(strategy.prior_power))
    return normalize_superpixel_scores(scores)


def apply_overwrite_policy(
    *,
    strategy: SuperpixelRefinementStrategy,
    pixel_labels: np.ndarray,
    pixel_conf: np.ndarray,
    flat_sp: np.ndarray,
    sp_labels: np.ndarray,
    sp_conf: np.ndarray,
    baseline_sp_labels: np.ndarray,
) -> np.ndarray:
    sp_flat_labels = sp_labels[flat_sp]
    if strategy.overwrite_policy == "all":
        return sp_flat_labels.astype(np.int32)

    out_flat = pixel_labels.copy()
    if strategy.overwrite_policy == "low_pixel_conf":
        mask = pixel_conf < float(strategy.pixel_confidence_threshold)
    elif strategy.overwrite_policy == "disagree_low_pixel_conf":
        mask = (
            (pixel_labels != sp_flat_labels)
            & (pixel_conf < float(strategy.pixel_confidence_threshold))
        )
    elif strategy.overwrite_policy == "high_sp_conf":
        mask = sp_conf[flat_sp] >= float(strategy.superpixel_confidence_threshold)
    elif strategy.overwrite_policy == "changed_high_sp_conf":
        mask = (
            (baseline_sp_labels[flat_sp] != sp_flat_labels)
            & (sp_conf[flat_sp] >= float(strategy.superpixel_confidence_threshold))
        )
    else:  # pragma: no cover - validated by strategy registry
        raise ValueError(f"Unknown overwrite policy: {strategy.overwrite_policy}")

    out_flat[mask] = sp_flat_labels[mask]
    return out_flat.astype(np.int32)


def superpixel_postprocess_strategy(
    *,
    logits_np: np.ndarray,
    superpixels: np.ndarray,
    strategy: SuperpixelRefinementStrategy,
) -> np.ndarray:
    channels, height, width = logits_np.shape
    if superpixels.shape != (height, width):
        raise ValueError(
            "Superpixel labels and logits have different spatial shapes: "
            f"{superpixels.shape} vs {(height, width)}"
        )

    flat_sp = superpixels.reshape(-1)
    num_sp = int(flat_sp.max()) + 1
    probs = torch.softmax(torch.from_numpy(logits_np), dim=0).numpy().astype(np.float32)
    flat_probs = probs.reshape(channels, -1)
    pixel_labels = flat_probs.argmax(axis=0).astype(np.int32)
    pixel_conf, pixel_margin, pixel_entropy = compute_pixel_confidence_features(flat_probs)

    scores = compute_superpixel_scores(
        logits_np=logits_np,
        flat_sp=flat_sp,
        num_sp=num_sp,
        flat_probs=flat_probs,
        pixel_labels=pixel_labels,
        pixel_conf=pixel_conf,
        pixel_margin=pixel_margin,
        pixel_entropy=pixel_entropy,
        strategy=strategy,
    )
    if int(strategy.graph_steps) > 0 and float(strategy.graph_alpha) > 0.0:
        adjacency = build_superpixel_adjacency(superpixels)
        scores = smooth_superpixel_scores(
            scores,
            adjacency,
            steps=int(strategy.graph_steps),
            alpha=float(strategy.graph_alpha),
        )
    else:
        adjacency = None

    sp_labels = scores.argmax(axis=0).astype(np.int32)
    sp_conf = scores.max(axis=0).astype(np.float32)
    baseline_sp_labels = compute_superpixel_majority_labels(
        pixel_labels,
        flat_sp,
        num_sp,
        channels,
    )
    out_flat = apply_overwrite_policy(
        strategy=strategy,
        pixel_labels=pixel_labels,
        pixel_conf=pixel_conf,
        flat_sp=flat_sp,
        sp_labels=sp_labels,
        sp_conf=sp_conf,
        baseline_sp_labels=baseline_sp_labels,
    )

    if strategy.cleanup_mode == "none":
        return out_flat.reshape(height, width)

    if adjacency is None:
        adjacency = build_superpixel_adjacency(superpixels)
    stage1_sp_labels = compute_superpixel_majority_labels(
        out_flat,
        flat_sp,
        num_sp,
        channels,
    )
    if strategy.cleanup_mode == "simple":
        stage2_sp_labels = cleanup_small_superpixel_components(
            stage1_sp_labels,
            adjacency,
            max_component_size=int(strategy.small_component_superpixels),
        )
    elif strategy.cleanup_mode == "conservative":
        counts = np.bincount(flat_sp, minlength=num_sp).astype(np.float32)
        eligible_mask = stage1_sp_labels != baseline_sp_labels
        stage2_sp_labels = cleanup_small_superpixel_components_conservative(
            stage1_sp_labels,
            adjacency,
            max_component_size=int(strategy.small_component_superpixels),
            sp_confidence=sp_conf,
            counts=counts,
            confidence_threshold=float(strategy.superpixel_confidence_threshold),
            neighbor_ratio_threshold=float(strategy.neighbor_ratio_threshold),
            eligible_mask=eligible_mask,
        )
    else:  # pragma: no cover - validated by strategy registry
        raise ValueError(f"Unknown cleanup mode: {strategy.cleanup_mode}")
    return stage2_sp_labels[flat_sp].reshape(height, width)


def superpixel_postprocess(
    logits_np: np.ndarray,
    superpixels: np.ndarray,
    vote_mode: str,
    confidence_threshold: float = 0.75,
    prior_power: float = 0.5,
    small_component_superpixels: int = 3,
    hybrid_neighbor_ratio: float = 0.6,
) -> np.ndarray:
    strategy = resolve_refinement_strategy(
        vote_mode=vote_mode,
        strategy_id=None,
        confidence_threshold=confidence_threshold,
        prior_power=prior_power,
        small_component_superpixels=small_component_superpixels,
        hybrid_neighbor_ratio=hybrid_neighbor_ratio,
    )
    return superpixel_postprocess_strategy(
        logits_np=logits_np,
        superpixels=superpixels,
        strategy=strategy,
    )


def perturb_checkpoint(
    checkpoint: dict[str, Any],
    weight_key: str,
    noise_std: float,
    noise_seed: int | None,
) -> None:
    if noise_std == 0.0:
        return
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict) or weight_key not in model_state:
        raise KeyError(f"Weight key not found in checkpoint: {weight_key}")
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    tensor = model_state[weight_key]
    model_state[weight_key] = tensor + torch.randn_like(tensor) * noise_std


def resolved_noise_std(args: argparse.Namespace) -> float:
    return 0.0 if args.no_noise else float(args.noise_std)


def build_hrnet_model(
    hrnet_cls: type[Any],
    checkpoint: dict[str, Any],
    device: str,
) -> Any:
    checkpoint_copy = dict(checkpoint)
    checkpoint_copy["pretrained"] = False
    model = hrnet_cls._create_from_checkpoint(checkpoint_copy, device)
    model.model.load_state_dict(checkpoint["model_state"])
    model.model.eval()
    return model


def decode_prediction(pred_idx: np.ndarray, class_infos: list[ClassInfo]) -> np.ndarray:
    raw_codes = np.array([info.code for info in class_infos], dtype=np.uint8)
    return raw_codes[pred_idx]


def colorize_prediction(pred_idx: np.ndarray, class_infos: list[ClassInfo]) -> np.ndarray:
    palette = np.array([info.color_rgb for info in class_infos], dtype=np.uint8)
    return palette[pred_idx]


def binary_mask_image(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def overlay_binary_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: float = 0.6,
) -> np.ndarray:
    result = image_rgb.astype(np.float32).copy()
    color = np.array(color_rgb, dtype=np.float32)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * color
    return np.clip(result, 0, 255).astype(np.uint8)


def save_change_maps(
    image_rgb: np.ndarray,
    image_stem: str,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    out_dir: Path,
) -> tuple[int, int]:
    pos_dir = out_dir / "changed_regions" / "positive"
    neg_dir = out_dir / "changed_regions" / "negative"
    comb_dir = out_dir / "changed_regions" / "combined"
    for directory in (pos_dir, neg_dir, comb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pos_labels, pos_regions = cc_label(positive_mask.astype(np.uint8))
    neg_labels, neg_regions = cc_label(negative_mask.astype(np.uint8))

    Image.fromarray(binary_mask_image(positive_mask)).save(pos_dir / f"{image_stem}_mask.png")
    Image.fromarray(binary_mask_image(negative_mask)).save(neg_dir / f"{image_stem}_mask.png")
    Image.fromarray(
        overlay_binary_mask(image_rgb, positive_mask, (0, 255, 0))
    ).save(pos_dir / f"{image_stem}_overlay.png")
    Image.fromarray(
        overlay_binary_mask(image_rgb, negative_mask, (255, 0, 0))
    ).save(neg_dir / f"{image_stem}_overlay.png")
    np.save(pos_dir / f"{image_stem}_labels.npy", pos_labels.astype(np.int32))
    np.save(neg_dir / f"{image_stem}_labels.npy", neg_labels.astype(np.int32))

    combined = image_rgb.copy()
    combined = overlay_binary_mask(combined, positive_mask, (0, 255, 0), alpha=0.55)
    combined = overlay_binary_mask(combined, negative_mask, (255, 0, 0), alpha=0.55)
    Image.fromarray(combined).save(comb_dir / f"{image_stem}_overlay.png")

    return int(pos_regions), int(neg_regions)


def save_prediction_maps(
    image_stem: str,
    pred_base_idx: np.ndarray,
    pred_post_idx: np.ndarray,
    class_infos: list[ClassInfo],
    out_dir: Path,
) -> None:
    pred_dir = out_dir / "predictions"
    base_dir = pred_dir / "baseline"
    post_dir = pred_dir / "postprocessed"
    vis_dir = pred_dir / "visualizations"
    for directory in (base_dir, post_dir, vis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    Image.fromarray(decode_prediction(pred_base_idx, class_infos)).save(
        base_dir / f"{image_stem}.png"
    )
    Image.fromarray(decode_prediction(pred_post_idx, class_infos)).save(
        post_dir / f"{image_stem}.png"
    )
    Image.fromarray(colorize_prediction(pred_base_idx, class_infos)).save(
        vis_dir / f"{image_stem}_baseline_rgb.png"
    )
    Image.fromarray(colorize_prediction(pred_post_idx, class_infos)).save(
        vis_dir / f"{image_stem}_post_rgb.png"
    )


def save_summary_json(
    args: argparse.Namespace,
    class_infos: list[ClassInfo],
    before_conf: np.ndarray,
    after_conf: np.ndarray,
    image_count: int,
    device: str,
    output_dir: Path,
    strategy: SuperpixelRefinementStrategy,
) -> None:
    before = metrics_from_confusion(before_conf, class_infos)
    after = metrics_from_confusion(after_conf, class_infos)
    summary = {
        "device": device,
        "n_images": image_count,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "images_dir": str(Path(args.images).resolve()),
        "masks_dir": str(Path(args.masks).resolve()),
        "postprocessing": {
            "sp_method": args.sp_method,
            "vote_mode": args.vote_mode,
            "strategy_id": strategy.strategy_id,
            "strategy": strategy.to_dict(),
            "confidence_threshold": args.confidence_threshold,
            "prior_power": args.prior_power,
            "small_component_superpixels": args.small_component_superpixels,
            "hybrid_neighbor_ratio": args.hybrid_neighbor_ratio,
            "noise_enabled": not args.no_noise,
            "noise_std": resolved_noise_std(args),
            "noise_seed": args.noise_seed,
            "noise_weight_key": args.noise_weight_key,
        },
        "class_mapping": [
            {
                "model_index": idx,
                "mask_code": info.code,
                "name": info.name,
                "color_rgb": list(info.color_rgb),
            }
            for idx, info in enumerate(class_infos)
        ],
        "before": before,
        "after": after,
        "delta": {
            "pixel_accuracy": after["pixel_accuracy"] - before["pixel_accuracy"],
            "miou": after["miou"] - before["miou"],
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    strategy = resolve_refinement_strategy(
        vote_mode=args.vote_mode,
        strategy_id=args.strategy_id,
        confidence_threshold=args.confidence_threshold,
        prior_power=args.prior_power,
        small_component_superpixels=args.small_component_superpixels,
        hybrid_neighbor_ratio=args.hybrid_neighbor_ratio,
    )
    if args.list_strategies:
        catalog = named_strategy_catalog(
            confidence_threshold=args.confidence_threshold,
            prior_power=args.prior_power,
            small_component_superpixels=args.small_component_superpixels,
            hybrid_neighbor_ratio=args.hybrid_neighbor_ratio,
            include_legacy=True,
            novel_limit=100,
        )
        for strategy_name in sorted(catalog):
            print(strategy_name)
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_image_pairs(args.images, args.masks)
    if not pairs:
        raise RuntimeError("No matching image/mask pairs were found.")
    if args.limit is not None:
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
    for _, mask_path in pairs[: min(8, len(pairs))]:
        mask_codes_seen.update(np.unique(load_mask_codes(mask_path)).tolist())

    class_infos = infer_class_infos(
        class_codes_arg=parse_class_codes(args.class_codes),
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        named_sets=get_named_class_sets(lumenstone_classes),
        mask_codes_seen=mask_codes_seen,
    )

    before_conf = np.zeros((len(class_infos), len(class_infos)), dtype=np.int64)
    after_conf = np.zeros((len(class_infos), len(class_infos)), dtype=np.int64)

    per_image_path = output_dir / "per_image_metrics.csv"
    with open(per_image_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "mask",
                "height",
                "width",
                "valid_pixels",
                "before_pixel_accuracy",
                "after_pixel_accuracy",
                "delta_pixel_accuracy",
                "before_miou",
                "after_miou",
                "delta_miou",
                "positive_pixels",
                "negative_pixels",
                "positive_regions",
                "negative_regions",
            ],
        )
        writer.writeheader()

        for index, (image_path, mask_path) in enumerate(pairs, start=1):
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
            pred_base_idx = logits_np.argmax(axis=0).astype(np.int32)
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
            if args.strategy_id:
                pred_post_idx = superpixel_postprocess_strategy(
                    logits_np=logits_np,
                    superpixels=superpixels,
                    strategy=strategy,
                )
            else:
                pred_post_idx = superpixel_postprocess(
                    logits_np=logits_np,
                    superpixels=superpixels,
                    vote_mode=args.vote_mode,
                    confidence_threshold=args.confidence_threshold,
                    prior_power=args.prior_power,
                    small_component_superpixels=args.small_component_superpixels,
                    hybrid_neighbor_ratio=args.hybrid_neighbor_ratio,
                )

            image_before_conf = compute_confusion(
                gt_idx, pred_base_idx, len(class_infos), valid_mask
            )
            image_after_conf = compute_confusion(
                gt_idx, pred_post_idx, len(class_infos), valid_mask
            )
            before_conf += image_before_conf
            after_conf += image_after_conf

            before_metrics = metrics_from_confusion(image_before_conf, class_infos)
            after_metrics = metrics_from_confusion(image_after_conf, class_infos)

            positive_mask = valid_mask & (pred_base_idx != gt_idx) & (pred_post_idx == gt_idx)
            negative_mask = valid_mask & (pred_base_idx == gt_idx) & (pred_post_idx != gt_idx)
            positive_regions, negative_regions = save_change_maps(
                image_rgb=image_rgb,
                image_stem=image_path.stem,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
                out_dir=output_dir,
            )

            if args.save_predictions:
                save_prediction_maps(
                    image_stem=image_path.stem,
                    pred_base_idx=pred_base_idx,
                    pred_post_idx=pred_post_idx,
                    class_infos=class_infos,
                    out_dir=output_dir,
                )

            writer.writerow(
                {
                    "image": image_path.name,
                    "mask": mask_path.name,
                    "height": image_rgb.shape[0],
                    "width": image_rgb.shape[1],
                    "valid_pixels": int(valid_mask.sum()),
                    "before_pixel_accuracy": before_metrics["pixel_accuracy"],
                    "after_pixel_accuracy": after_metrics["pixel_accuracy"],
                    "delta_pixel_accuracy": (
                        after_metrics["pixel_accuracy"] - before_metrics["pixel_accuracy"]
                    ),
                    "before_miou": before_metrics["miou"],
                    "after_miou": after_metrics["miou"],
                    "delta_miou": after_metrics["miou"] - before_metrics["miou"],
                    "positive_pixels": int(positive_mask.sum()),
                    "negative_pixels": int(negative_mask.sum()),
                    "positive_regions": positive_regions,
                    "negative_regions": negative_regions,
                }
            )

            print(
                f"[{index}/{len(pairs)}] {image_path.name}: "
                f"mIoU {before_metrics['miou']:.4f} -> {after_metrics['miou']:.4f} "
                f"(delta {after_metrics['miou'] - before_metrics['miou']:+.4f}), "
                f"acc {before_metrics['pixel_accuracy']:.4f} -> "
                f"{after_metrics['pixel_accuracy']:.4f}"
            )

    save_summary_json(
        args=args,
        class_infos=class_infos,
        before_conf=before_conf,
        after_conf=after_conf,
        image_count=len(pairs),
        device=device,
        output_dir=output_dir,
        strategy=strategy,
    )

    before = metrics_from_confusion(before_conf, class_infos)
    after = metrics_from_confusion(after_conf, class_infos)
    print()
    print(f"Processed {len(pairs)} image(s) on device={device}")
    print(
        "Dataset metrics: "
        f"mIoU {before['miou']:.4f} -> {after['miou']:.4f} "
        f"(delta {after['miou'] - before['miou']:+.4f}), "
        f"accuracy {before['pixel_accuracy']:.4f} -> {after['pixel_accuracy']:.4f}"
    )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
