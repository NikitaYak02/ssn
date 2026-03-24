#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_class_miou.py

Строит графики per-class IoU / mIoU по файлу metrics.csv, который создаёт
evaluate_interactive_annotation.py.

Сохраняет:
  - class_iou_over_time.png        — все классы на одном графике + mIoU
  - class_iou_grid.png             — отдельный subplot для каждого класса

Пример:
    python plot_class_miou.py \
        --metrics /path/to/metrics.csv \
        --out_dir /path/to/plots
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_CLASS_INFO = [
    ("BG", "#1c1818"),
    ("Ccp", "#ff0000"),
    ("Gl", "#cbff00"),
    ("Mag", "#00ff66"),
    ("Brt", "#0065ff"),
    ("Po", "#cc00ff"),
    ("Pn", "#dbff4c"),
    ("Sph", "#4cff93"),
    ("Apy", "#4c93ff"),
    ("Hem", "#db4cff"),
    ("Kvl", "#eaff99"),
    ("Py/Mrc", "#ff4c4c"),
    ("Tnt/Ttr", "#ff9999"),
]


def hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def parse_float(value: str) -> float:
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return float("nan")
    return float(s)


def load_mask_as_ids(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.mode in ("P", "L", "I;16", "I"):
        arr = np.array(mask)
    else:
        arr = np.array(mask.convert("RGB"))[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got shape={arr.shape}")
    return arr.astype(np.int32)


def load_metrics(metrics_path: Path) -> Tuple[List[int], List[float], Dict[str, List[float]]]:
    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"metrics.csv is empty: {metrics_path}")

    iou_cols = [c for c in reader.fieldnames or [] if c.startswith("iou_")]
    if not iou_cols:
        raise ValueError(f"No per-class IoU columns found in: {metrics_path}")

    steps = [int(float(r["step"])) for r in rows]
    miou = [parse_float(r["miou"]) for r in rows]
    per_class = {col.removeprefix("iou_"): [parse_float(r[col]) for r in rows] for col in iou_cols}
    return steps, miou, per_class


def filter_present_classes(
    per_class: Dict[str, List[float]],
    mask_path: Path | None,
) -> Dict[str, List[float]]:
    if mask_path is not None:
        mask = load_mask_as_ids(mask_path)
        present_ids = {int(v) for v in np.unique(mask)}
        present_names = {
            DEFAULT_CLASS_INFO[idx][0]
            for idx in present_ids
            if 0 <= idx < len(DEFAULT_CLASS_INFO)
        }
        filtered = {name: vals for name, vals in per_class.items() if name in present_names}
        if filtered:
            return filtered

    # Fallback: keep only classes that have at least one finite IoU value.
    return {
        name: vals
        for name, vals in per_class.items()
        if any(np.isfinite(v) for v in vals)
    }


def build_color_map(class_names: List[str]) -> Dict[str, Tuple[float, float, float]]:
    cmap: Dict[str, Tuple[float, float, float]] = {}
    for name, color in DEFAULT_CLASS_INFO:
        cmap[name] = hex_to_rgb01(color)

    fallback = plt.cm.tab20(np.linspace(0.0, 1.0, max(1, len(class_names))))
    for idx, name in enumerate(class_names):
        if name not in cmap:
            cmap[name] = tuple(float(x) for x in fallback[idx][:3])
    return cmap


def plot_all_classes(
    steps: List[int],
    miou: List[float],
    per_class: Dict[str, List[float]],
    out_path: Path,
) -> None:
    class_names = list(per_class.keys())
    colors = build_color_map(class_names)

    fig, ax = plt.subplots(figsize=(14, 8))
    for name in class_names:
        ax.plot(
            steps,
            per_class[name],
            label=name,
            color=colors[name],
            linewidth=1.8,
            alpha=0.9,
        )

    ax.plot(
        steps,
        miou,
        label="mIoU",
        color="black",
        linewidth=2.5,
        linestyle="--",
    )

    ax.set_title("Per-class IoU over annotation steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_class_grid(
    steps: List[int],
    miou: List[float],
    per_class: Dict[str, List[float]],
    out_path: Path,
) -> None:
    class_names = list(per_class.keys())
    colors = build_color_map(class_names)

    n = len(class_names)
    ncols = 3 if n >= 3 else max(1, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, name in zip(axes_flat, class_names):
        ax.plot(steps, per_class[name], color=colors[name], linewidth=2.0, label=name)
        ax.plot(steps, miou, color="black", linestyle="--", linewidth=1.3, alpha=0.85, label="mIoU")
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel("IoU")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes_flat[n:]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="Путь к metrics.csv")
    ap.add_argument("--out_dir", required=True, help="Куда сохранить графики")
    ap.add_argument(
        "--mask",
        default=None,
        help="GT-маска изображения. Если указана, графики строятся только для классов, "
             "которые реально присутствуют на изображении.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    metrics_path = Path(args.metrics).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, miou, per_class = load_metrics(metrics_path)
    per_class = filter_present_classes(per_class, Path(args.mask).resolve() if args.mask else None)
    if not per_class:
        raise ValueError("No present classes found to plot.")
    plot_all_classes(steps, miou, per_class, out_dir / "class_iou_over_time.png")
    plot_class_grid(steps, miou, per_class, out_dir / "class_iou_grid.png")

    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
