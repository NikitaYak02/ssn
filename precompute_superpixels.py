#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precompute_superpixels.py

Пакетное предварительное разбиение изображений на суперпиксели с сохранением
результата в формат `.spanno.json.gz`.

Это полезно запускать до `evaluate_interactive_annotation.py`, чтобы оценка шла
по уже готовой автоматической сегментации изображения и не возникали странные
объединения разных регионов в один суперпиксель.

Пример:
    python precompute_superpixels.py \
        --img_dir /data/images \
        --out_dir /data/spanno \
        --method slic --n_segments 3000 --compactness 15 --sigma 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "superpixel_annotator"))
sys.path.insert(0, str(_SCRIPT_DIR))

import structs  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def setup_logger() -> logging.Logger:
    log = logging.getLogger("precompute_sp")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(handler)
    return log


def discover_images(img_dir: Path, recursive: bool = True) -> List[Path]:
    iterator: Iterable[Path] = img_dir.rglob("*") if recursive else img_dir.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_sp_method(args: argparse.Namespace) -> structs.SuperPixelMethod:
    return structs.build_superpixel_method_from_args(args)


def populate_full_image_superpixels(
    algo: structs.SuperPixelAnnotationAlgo,
    sp_method: structs.SuperPixelMethod,
) -> int:
    labels = structs.compute_superpixels(algo.image_lab, sp_method)
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        return 0

    means, variances, valid_labels = structs.parallel_stats_rgb(
        algo.image_lab.astype(np.float32, copy=False),
        labels.astype(np.int32, copy=False),
        max_label,
    )
    polys = structs.labels_to_polygons(
        labels,
        bbox01=(0.0, 0.0, 1.0, 1.0),
        out_h=algo.image_lab.shape[0],
        out_w=algo.image_lab.shape[1],
        filter_labels=valid_labels,
    )

    algo.superpixels[sp_method] = []
    algo._annotations[sp_method] = structs.ImageAnnotation(annotations=[])
    algo._superpixel_ind[sp_method] = 0
    algo._annotation_ind[sp_method] = 0

    count = 0
    for i, lab in enumerate(valid_labels):
        lab_i = int(lab)
        if lab_i <= 0:
            continue
        poly = polys.get(lab_i)
        if poly is None:
            continue
        if isinstance(poly, Polygon):
            border = np.asarray(poly.exterior.coords, dtype=np.float32)
            holes = [
                np.asarray(interior.coords, dtype=np.float32)
                for interior in poly.interiors
                if len(interior.coords) >= 3
            ]
        else:
            border = np.asarray(poly, dtype=np.float32)
            holes = []
        if border.ndim != 2 or border.shape[0] < 3:
            continue
        props = np.concatenate([means[i], variances[i]]).astype(np.float32)
        algo.superpixels[sp_method].append(
            structs.SuperPixel(
                id=int(algo._superpixel_ind[sp_method]),
                method=sp_method.short_string(),
                border=border,
                parents=[],
                props=props,
                holes=holes,
                emb=None,
            )
        )
        algo._superpixel_ind[sp_method] += 1
        count += 1

    algo._mark_sp_index_dirty()
    return count


def process_one_image(
    image_path: Path,
    out_path: Path,
    sp_method: structs.SuperPixelMethod,
    downscale: float,
    overwrite: bool,
    logger: logging.Logger,
) -> bool:
    if out_path.exists() and not overwrite:
        logger.info("Skip existing: %s", out_path)
        return True

    img = Image.open(image_path).convert("RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=float(downscale),
        superpixel_methods=[],
        image_path=image_path,
        image=img,
    )
    algo.add_superpixel_method(sp_method)

    n_sp = populate_full_image_superpixels(algo, sp_method)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    algo.serialize(str(out_path), pretty=False)
    logger.info("Saved %s | method=%s | superpixels=%d", out_path, sp_method.short_string(), n_sp)
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--img_dir", required=True, help="Папка с изображениями")
    ap.add_argument("--out_dir", required=True, help="Куда сохранять .spanno.json.gz")
    ap.add_argument("--method", default="slic",
                    choices=structs.SUPPORTED_SUPERPIXEL_METHOD_CHOICES)
    ap.add_argument("--method_config", default=None,
                    help="JSON string or path to JSON config for neural methods.")
    ap.add_argument("--weights", default=None,
                    help="Checkpoint for neural methods (and optional alias for ssn).")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="downscale_coeff для SuperPixelAnnotationAlgo")
    ap.add_argument("--overwrite", action="store_true",
                    help="Перезаписывать уже существующие .spanno.json.gz")
    ap.add_argument("--no_recursive", action="store_true",
                    help="Не обходить вложенные директории")

    grp_sp = ap.add_argument_group("Superpixel method")
    grp_sp.add_argument("--n_segments", type=int, default=3000)
    grp_sp.add_argument("--compactness", type=float, default=20.0)
    grp_sp.add_argument("--sigma", type=float, default=1.0)
    grp_sp.add_argument("--scale", type=float, default=400.0)
    grp_sp.add_argument("--f_sigma", type=float, default=1.0)
    grp_sp.add_argument("--min_size", type=int, default=50)
    grp_sp.add_argument("--ws_compactness", type=float, default=1e-4)
    grp_sp.add_argument("--ws_components", type=int, default=500)
    grp_sp.add_argument("--ssn_weights", default=None, help="Чекпоинт SSN (.pth)")
    grp_sp.add_argument("--ssn_nspix", type=int, default=100)
    grp_sp.add_argument("--ssn_fdim", type=int, default=20)
    grp_sp.add_argument("--ssn_niter", type=int, default=5)
    grp_sp.add_argument("--ssn_color_scale", type=float, default=0.26)
    grp_sp.add_argument("--ssn_pos_scale", type=float, default=2.5)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger()

    img_dir = Path(args.img_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        logger.error("Image directory does not exist: %s", img_dir)
        return 1

    sp_method = build_sp_method(args)
    images = discover_images(img_dir, recursive=not bool(args.no_recursive))
    if not images:
        logger.error("No images found in %s", img_dir)
        return 1

    logger.info("Found %d images", len(images))
    logger.info("Method: %s", sp_method.short_string())
    logger.info(
        "Hint: these precomputed .spanno files can be passed into "
        "evaluate_interactive_annotation.py via --spanno."
    )

    ok_count = 0
    for image_path in images:
        rel = image_path.relative_to(img_dir)
        out_path = out_dir / rel.parent / f"{image_path.stem}.spanno.json.gz"
        try:
            if process_one_image(
                image_path=image_path,
                out_path=out_path,
                sp_method=sp_method,
                downscale=float(args.downscale),
                overwrite=bool(args.overwrite),
                logger=logger,
            ):
                ok_count += 1
        except Exception as exc:
            logger.exception("Failed on %s: %s", image_path, exc)

    logger.info("Done: %d/%d images processed successfully", ok_count, len(images))
    return 0 if ok_count == len(images) else 2


if __name__ == "__main__":
    raise SystemExit(main())
