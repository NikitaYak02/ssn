#!/usr/bin/env python3
"""Batch run the interactive scribble pipeline with SSN precomputed upfront.

This wrapper is meant for test-set evaluation on one subset, for example
`S1_v2`. It:

1. loads each image/mask pair,
2. resizes both by `--resize-scale` (default: 0.5),
3. precomputes a full-image SSN `.spanno.json.gz` for the resized image,
4. runs the existing interactive annotation pipeline with that precomputed
   superpixel layout,
5. writes per-image metrics inside `<output-dir>/<image-stem>/`,
6. writes an aggregate `batch_summary.csv` with one row per image plus a final
   `MEAN` row.

The script reuses the existing logic from `evaluate_interactive_annotation.py`
so the stroke-generation and metric computation remain consistent with the
rest of the repository.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "superpixel_annotator"))
sys.path.insert(0, str(_SCRIPT_DIR))

import structs  # noqa: E402
from evaluate_interactive_annotation import (  # noqa: E402
    DEFAULT_CLASS_INFO,
    discover_image_pairs,
    ensure_same_size,
    load_mask_as_ids,
    run_single_image,
    save_batch_summary,
    setup_logger,
)
from precompute_superpixels import (  # noqa: E402
    build_sp_method,
    populate_full_image_superpixels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SSN-backed scribble evaluation on a whole test subset."
    )

    parser.add_argument("--images", required=True, help="Directory with test images.")
    parser.add_argument("--masks", required=True, help="Directory with GT masks.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-image runs and the summary CSV will be written.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Optional directory for cached resized inputs and SSN spanno files.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=0.5,
        help="Resize images and masks by this factor before running evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of image/mask pairs to process.",
    )
    parser.add_argument(
        "--overwrite-spanno",
        action="store_true",
        help="Recompute cached SSN spanno files even if they already exist.",
    )

    # Interactive evaluation settings. These mirror evaluate_interactive_annotation.py.
    parser.add_argument("--scribbles", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--border_margin", type=int, default=3)
    parser.add_argument("--no_overlap", action="store_true")
    parser.add_argument("--max_no_progress", type=int, default=12)
    parser.add_argument(
        "--region_selection_cycle",
        default="miou_gain,largest_error,unannotated",
    )
    parser.add_argument("--sensitivity", type=float, default=1.8)
    parser.add_argument("--emb_weights", default=None)
    parser.add_argument("--emb_threshold", type=float, default=0.988)
    parser.add_argument("--num_classes", type=int, default=None)

    # SSN superpixel settings.
    parser.add_argument("--method", default="ssn", choices=["ssn"])
    parser.add_argument("--ssn_weights", required=True, help="Path to SSN checkpoint (.pth).")
    parser.add_argument("--ssn_nspix", type=int, default=100)
    parser.add_argument("--ssn_fdim", type=int, default=20)
    parser.add_argument("--ssn_niter", type=int, default=5)
    parser.add_argument("--ssn_color_scale", type=float, default=0.26)
    parser.add_argument("--ssn_pos_scale", type=float, default=2.5)

    return parser


def _build_class_info(num_classes: int | None) -> list[tuple[str, str]]:
    if num_classes is None:
        return list(DEFAULT_CLASS_INFO)
    if num_classes <= len(DEFAULT_CLASS_INFO):
        return list(DEFAULT_CLASS_INFO[:num_classes])
    return list(DEFAULT_CLASS_INFO) + [
        (f"cls{i}", "#aaaaaa") for i in range(len(DEFAULT_CLASS_INFO), num_classes)
    ]


def _resize_pair(img: Image.Image, gt: np.ndarray, scale: float) -> tuple[Image.Image, np.ndarray]:
    if scale <= 0.0:
        raise ValueError("--resize-scale must be positive")
    if math.isclose(scale, 1.0):
        return img, gt

    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized_img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    resized_gt = Image.fromarray(gt.astype(np.int32), mode="I").resize(
        (new_w, new_h), resample=Image.Resampling.NEAREST
    )
    return resized_img, np.array(resized_gt).astype(np.int32)


def _maybe_make_spanno(
    img: Image.Image,
    sp_method,
    spanno_path: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    if spanno_path.exists() and not overwrite:
        return

    spanno_path.parent.mkdir(parents=True, exist_ok=True)
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=img,
        image_path=None,
        auto_propagation_sensitivity=0.0,
    )
    algo.add_superpixel_method(sp_method)
    n_superpixels = populate_full_image_superpixels(algo, sp_method)
    algo.serialize(str(spanno_path), pretty=False)
    logger.info(
        "Cached SSN spanno: %s | %s | superpixels=%d",
        spanno_path,
        sp_method.short_string(),
        n_superpixels,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve() if args.work_dir else (out_dir / "_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    spanno_dir = work_dir / "spanno"
    spanno_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir, name="ssn_scribble_batch")
    logger.info("Images: %s", Path(args.images).resolve())
    logger.info("Masks: %s", Path(args.masks).resolve())
    logger.info("Output: %s", out_dir)
    logger.info("Work dir: %s", work_dir)
    logger.info("Resize scale: %.3f", float(args.resize_scale))

    pairs = discover_image_pairs(args.images, args.masks)
    if not pairs:
        raise RuntimeError("No matching image/mask pairs were found.")
    if args.limit is not None:
        pairs = pairs[: int(args.limit)]

    logger.info("Found %d image/mask pairs", len(pairs))

    class_info = _build_class_info(args.num_classes)
    logger.info("Classes (%d): %s", len(class_info), [c[0] for c in class_info])

    sp_method_proto = build_sp_method(args)
    logger.info("SP method: %s", sp_method_proto.short_string())
    logger.info("Propagation sensitivity: %.2f", float(args.sensitivity))
    if args.emb_weights:
        logger.info(
            "Embedding propagation: %s | threshold=%.3f",
            args.emb_weights,
            float(args.emb_threshold),
        )

    run_args = copy.deepcopy(args)
    run_args.downscale = 1.0
    run_args.no_borders = False
    run_args.no_annos = False
    run_args.no_scribbles = False

    summaries: dict[str, list] = {}
    for image_path, mask_path in pairs:
        img = Image.open(image_path).convert("RGB")
        gt = load_mask_as_ids(str(mask_path))
        img, gt = ensure_same_size(img, gt)
        img, gt = _resize_pair(img, gt, float(args.resize_scale))

        image_name = image_path.stem
        image_out = out_dir / image_name
        image_out.mkdir(parents=True, exist_ok=True)

        spanno_path = spanno_dir / f"{image_name}.spanno.json.gz"
        logger.info("=== Processing %s ===", image_path.name)
        _maybe_make_spanno(
            img=img,
            sp_method=sp_method_proto,
            spanno_path=spanno_path,
            overwrite=bool(args.overwrite_spanno),
            logger=logger,
        )

        history = run_single_image(
            img=img,
            gt=gt,
            sp_method_proto=sp_method_proto,
            args=run_args,
            out_dir=image_out,
            class_info=class_info,
            logger=logger,
            spanno_path=str(spanno_path),
        )
        summaries[image_name] = history

        if history:
            final = history[-1]
            logger.info(
                "Final %s | steps=%d | scribbles=%d | mIoU=%.4f | cov=%.4f | prec=%.4f",
                image_name,
                final.step,
                final.n_scribbles,
                final.miou,
                final.coverage,
                final.annotation_precision,
            )

    batch_summary = out_dir / "batch_summary.csv"
    save_batch_summary(summaries, class_info, batch_summary)
    logger.info("Batch summary: %s", batch_summary)


if __name__ == "__main__":
    main()
