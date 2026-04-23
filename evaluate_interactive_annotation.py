#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_interactive_annotation.py

Пайплайн автоматического проставления штрихов с детальными метриками
качества аннотации относительно стоимости взаимодействий пользователя.

Метрики качества:
    mIoU                    — средний IoU по классам
    per-class IoU           — IoU для каждого класса
    coverage                — доля размеченных пикселей от общего числа
    annotation_precision    — доля правильных пикселей среди размеченных

Метрики стоимости взаимодействия:
    n_scribbles             — число нанесённых штрихов
    total_ink_px            — суммарная длина всех штрихов (пиксели)
    mean_ink_px             — средняя длина одного штриха (пиксели)
    per_class_n_scribbles   — штрихов по каждому классу
    per_class_ink_px        — длина штрихов по каждому классу

Метрики эффективности (качество / стоимость):
    miou_per_scribble       — mIoU делённый на число штрихов
    miou_per_1kpx           — mIoU на 1 000 пикселей суммарной длины
    correct_px_per_scribble — правильно размеченных пикселей на штрих
    correct_px_per_px_ink   — правильно размеченных пикселей на пиксель длины
    quality_x_coverage      — mIoU × coverage (совместная метрика)

Пример запуска (одно изображение):
    python evaluate_interactive_annotation.py \\
        --image img.png --mask gt.png --out results/ \\
        --method slic --n_segments 3000 --compactness 15 \\
        --sensitivity 2.0 --scribbles 500 --save_every 50

Пакетный режим (директория изображений):
    python evaluate_interactive_annotation.py \\
        --img_dir /data/images --mask_dir /data/masks --out results/ \\
        --method ssn --ssn_weights model.pth \\
        --sensitivity 1.5 --scribbles 300 --save_every 50

Флаги пропагации:
    --sensitivity 0.0      — отключить распространение штрихов (только прямые аннотации)
    --sensitivity 1.8      — рекомендуемый консервативный режим для SSN+embeddings
    --emb_weights path.pth — использовать cosine-similarity по эмбедингам вместо LAB-цвета
    --emb_threshold 0.988  — строгий порог косинусного сходства для аккуратного propagation

Выбор региона для нового штриха:
    --region_selection_cycle miou_gain,largest_error,unannotated
                          — чередовать стратегии по шагам внутри одной разметки
"""

import argparse
import csv
import gzip
import json
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.morphology import medial_axis, skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Resolve path to structs.py ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "superpixel_annotator"))
sys.path.insert(0, str(_SCRIPT_DIR))

import structs  # noqa: E402 (must come after sys.path hack)


# ─────────────────────────────────────────────────────────────────────────────────
#  Константы / цвета
# ─────────────────────────────────────────────────────────────────────────────────
DEFAULT_CLASS_INFO = [
    ("bg",        "#000000"),
    ("ccp",       "#ffa500"),
    ("gl",        "#9acd32"),
    ("mag",       "#ff4500"),
    ("br",        "#00bfff"),
    ("po",        "#a9a9a9"),
    ("py",        "#2f4f4f"),
    ("pn",        "#ffff00"),
    ("sh",        "#ee82ee"),
    ("apy",       "#556b2f"),
    ("gmt",       "#a0522d"),
    ("tnt",       "#483d8b"),
    ("cv",        "#008000"),
    ("mrc",       "#00008b"),
    ("au",        "#8b008b"),
]
_SUPERPIXEL_BORDER_RGBA = (255, 255, 0, int(round(255 * 0.4)))


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgba(h: str, alpha: int) -> Tuple[int, int, int, int]:
    r, g, b = _hex_to_rgb(h)
    return r, g, b, alpha


def _edt_inside(mask: np.ndarray) -> np.ndarray:
    """EDT that also treats the image frame as outside the region."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)
    return dist[1:-1, 1:-1]


def _smooth_region_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Smooth a binary region with explicit dilation/erosion passes.

    We use the smoothed mask only to estimate the central corridor of the
    region; the final scribble is still traced inside the original allowed mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask.copy()

    structure = np.ones((2 * int(radius) + 1, 2 * int(radius) + 1), dtype=bool)
    closed = ndimage.binary_erosion(
        ndimage.binary_dilation(mask, structure=structure, iterations=1),
        structure=structure,
        iterations=1,
    )
    opened = ndimage.binary_dilation(
        ndimage.binary_erosion(closed, structure=structure, iterations=1),
        structure=structure,
        iterations=1,
    )

    min_keep = max(1, int(0.20 * float(np.count_nonzero(mask))))
    if int(np.count_nonzero(opened)) >= min_keep:
        return opened.astype(bool, copy=False)
    if closed.any():
        return closed.astype(bool, copy=False)
    return mask.copy()


# ─────────────────────────────────────────────────────────────────────────────────
#  Загрузка данных
# ─────────────────────────────────────────────────────────────────────────────────

def load_mask_as_ids(mask_path: str) -> np.ndarray:
    """Загружает маску (L, P, I, RGB) как 2D int32 с id классов."""
    m = Image.open(mask_path)
    if m.mode in ("P", "L", "I;16", "I"):
        arr = np.array(m)
    else:
        arr = np.array(m.convert("RGB"))[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {arr.shape}")
    return arr.astype(np.int32)


def ensure_same_size(img: Image.Image, mask: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
    if (img.height, img.width) == mask.shape[:2]:
        return img, mask
    m = Image.fromarray(mask.astype(np.uint8), mode="L")
    m = m.resize((img.width, img.height), resample=Image.NEAREST)
    return img, np.array(m).astype(np.int32)


def discover_image_pairs(img_dir: str, mask_dir: str) -> List[Tuple[Path, Path]]:
    """Пары (image_path, mask_path) по совпадению stem."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    pairs = []
    for img_path in sorted(Path(img_dir).iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        mask_path = Path(mask_dir) / img_path.name
        if not mask_path.exists():
            # try any extension with same stem
            for ext in exts:
                alt = Path(mask_dir) / (img_path.stem + ext)
                if alt.exists():
                    mask_path = alt
                    break
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def load_spanno_image_size(spanno_path: str) -> Optional[Tuple[int, int]]:
    """
    Read image size from .spanno metadata when available.

    Returns `(width, height)` or `None` if the file has no compatible metadata.
    """
    open_fn = gzip.open if str(spanno_path).endswith(".gz") else open
    try:
        with open_fn(spanno_path, mode="rt", encoding="utf-8") as f:
            root = json.load(f)
    except Exception:
        return None

    meta = root.get("_meta") if isinstance(root, dict) else None
    image_meta = meta.get("image") if isinstance(meta, dict) else None
    size_wh = image_meta.get("size_wh") if isinstance(image_meta, dict) else None
    if not isinstance(size_wh, list) or len(size_wh) != 2:
        return None
    try:
        return int(size_wh[0]), int(size_wh[1])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────────
#  Метрики
# ─────────────────────────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    """Все метрики на одном шаге (checkpoint)."""
    step: int
    n_scribbles: int
    total_ink_px: float            # суммарная длина штрихов в пикселях
    mean_ink_px: float             # средняя длина одного штриха
    annotated_px: int              # пикселей с любой аннотацией
    correctly_annotated_px: int   # пикселей, где pred == gt (среди размеченных)
    total_px: int                  # всего пикселей
    coverage: float                # annotated_px / total_px
    annotation_precision: float   # correctly_annotated_px / annotated_px
    miou: float
    per_class_iou: List[float]     # len == num_classes
    per_class_n_scribbles: List[int]
    per_class_ink_px: List[float]

    # Эффективность ──────────────────────────────────────────────────────────────
    @property
    def miou_per_scribble(self) -> float:
        return self.miou / max(1, self.n_scribbles)

    @property
    def miou_per_1kpx(self) -> float:
        return self.miou / max(1e-6, self.total_ink_px / 1000.0)

    @property
    def correct_px_per_scribble(self) -> float:
        return self.correctly_annotated_px / max(1, self.n_scribbles)

    @property
    def correct_px_per_px_ink(self) -> float:
        return self.correctly_annotated_px / max(1e-6, self.total_ink_px)

    @property
    def quality_x_coverage(self) -> float:
        return self.miou * self.coverage

    def as_flat_dict(self, class_names: List[str]) -> dict:
        d: dict = {
            "step": self.step,
            "n_scribbles": self.n_scribbles,
            "total_ink_px": round(self.total_ink_px, 2),
            "mean_ink_px": round(self.mean_ink_px, 2),
            "annotated_px": self.annotated_px,
            "correctly_annotated_px": self.correctly_annotated_px,
            "total_px": self.total_px,
            "coverage": round(self.coverage, 6),
            "annotation_precision": round(self.annotation_precision, 6),
            "miou": round(self.miou, 6),
            "miou_per_scribble": round(self.miou_per_scribble, 8),
            "miou_per_1kpx": round(self.miou_per_1kpx, 8),
            "correct_px_per_scribble": round(self.correct_px_per_scribble, 4),
            "correct_px_per_px_ink": round(self.correct_px_per_px_ink, 8),
            "quality_x_coverage": round(self.quality_x_coverage, 6),
        }
        for i, name in enumerate(class_names):
            iou_v = self.per_class_iou[i] if i < len(self.per_class_iou) else float("nan")
            d[f"iou_{name}"] = round(iou_v, 6) if not math.isnan(iou_v) else float("nan")
            ns = self.per_class_n_scribbles[i] if i < len(self.per_class_n_scribbles) else 0
            d[f"n_scrib_{name}"] = ns
            ink = self.per_class_ink_px[i] if i < len(self.per_class_ink_px) else 0.0
            d[f"ink_px_{name}"] = round(ink, 2)
        return d


def compute_ious(pred: np.ndarray, gt: np.ndarray,
                 num_classes: int) -> Tuple[float, List[float]]:
    """
    pred: (H,W) int32, где -1 — неразмечено, 0..C-1 — классы.
    gt:   (H,W) int32, 0-indexed class ids.
    """
    ious: List[float] = []
    for c in range(num_classes):
        p_c = (pred == c)
        g_c = (gt == c)
        inter = int(np.logical_and(p_c, g_c).sum())
        union = int(np.logical_or(p_c, g_c).sum())
        ious.append(inter / union if union > 0 else float("nan"))
    valid = [v for v in ious if not math.isnan(v)]
    miou = float(np.mean(valid)) if valid else float("nan")
    return miou, ious


def scribble_length_px(pts_norm: np.ndarray, H: int, W: int) -> float:
    """Евклидова длина штриха в пикселях."""
    if len(pts_norm) < 2:
        return 0.0
    pts = pts_norm.astype(np.float64) * np.array([[W, H]], dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))


# ─────────────────────────────────────────────────────────────────────────────────
#  PredMaskUpdater  (быстрая инкрементальная покраска полигонов)
# ─────────────────────────────────────────────────────────────────────────────────

class PredMaskUpdater:
    """Хранит текущую pred-маску и отдельную маску явной разметки."""

    def __init__(self, H: int, W: int):
        self.H, self.W = H, W
        # По умолчанию пиксели не размечены.
        self.pred_np = np.full((H, W), -1, dtype=np.int32)
        self.explicit_np = np.zeros((H, W), dtype=bool)
        self._pred_pil = Image.fromarray(self.pred_np, mode="I")
        self._explicit_pil = Image.fromarray(
            np.zeros((H, W), dtype=np.uint8), mode="L"
        )

    def reset(self) -> None:
        self.pred_np[:] = -1
        self.explicit_np[:] = False
        self._pred_pil = Image.fromarray(self.pred_np, mode="I")
        self._explicit_pil = Image.fromarray(
            np.zeros((self.H, self.W), dtype=np.uint8), mode="L"
        )

    def paint_polygon(
        self,
        poly_xy: List[Tuple[float, float]],
        value: int,
        *,
        explicit: bool = True,
    ) -> None:
        if len(poly_xy) < 3:
            return
        xs = [p[0] for p in poly_xy]
        ys = [p[1] for p in poly_xy]
        x0 = max(0, int(np.floor(min(xs))))
        y0 = max(0, int(np.floor(min(ys))))
        x1 = min(self.W, int(np.ceil(max(xs))) + 1)
        y1 = min(self.H, int(np.ceil(max(ys))) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        crop_pred = self._pred_pil.crop((x0, y0, x1, y1))
        ImageDraw.Draw(crop_pred).polygon(
            [(float(x - x0), float(y - y0)) for x, y in poly_xy],
            fill=int(value),
        )
        self._pred_pil.paste(crop_pred, (x0, y0))
        self.pred_np[y0:y1, x0:x1] = np.array(crop_pred, dtype=np.int32)

        if explicit:
            crop_exp = self._explicit_pil.crop((x0, y0, x1, y1))
            ImageDraw.Draw(crop_exp).polygon(
                [(float(x - x0), float(y - y0)) for x, y in poly_xy],
                fill=255,
            )
            self._explicit_pil.paste(crop_exp, (x0, y0))
            self.explicit_np[y0:y1, x0:x1] = np.array(crop_exp, dtype=np.uint8) > 0

    def repaint_all(self, algo: "structs.SuperPixelAnnotationAlgo",
                    sp_method: "structs.SuperPixelMethod",
                    num_classes: int) -> None:
        """Полная перерисовка всех аннотаций из algo."""
        self.reset()
        annos_obj = algo._annotations.get(sp_method)
        if annos_obj is None:
            return
        for anno in annos_obj.annotations:
            border = np.asarray(anno.border, dtype=np.float32)
            poly = [
                (float(border[i, 0] * self.W), float(border[i, 1] * self.H))
                for i in range(len(border))
            ]
            pred_class = max(0, min(int(anno.code) - 1, num_classes - 1))
            self.paint_polygon(poly, pred_class, explicit=True)


# ─────────────────────────────────────────────────────────────────────────────────
#  Генератор штрихов: самая большая ошибочная компонента
# ─────────────────────────────────────────────────────────────────────────────────

class LargestBadRegionGenerator:
    """
    На каждом шаге выбирает компоненту ошибки, которая даёт наибольший
    ожидаемый прирост mean IoU.

    Для каждого класса берётся его крупнейшая ошибочная компонента внутри GT
    этого класса, а затем оценивается приблизительный вклад в mIoU как
    component_area / union(class). Это даёт приоритет не только большим
    регионам, но и классам с низким текущим IoU.
    """

    VALID_REGION_SELECTION_MODES = (
        "miou_gain",
        "largest_error",
        "unannotated",
    )

    def __init__(self, gt_mask: np.ndarray, num_classes: int,
                 seed: int = 0, margin: int = 2, border_margin: int = 3,
                 no_overlap: bool = True, max_retries: int = 200,
                 region_selection_cycle: Optional[Sequence[str]] = None):
        self.gt = gt_mask.astype(np.int32)
        self.H, self.W = gt_mask.shape
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        self.margin = max(0, margin)
        self.border_margin = max(0, border_margin)
        self.no_overlap = no_overlap
        self.max_retries = max_retries
        self.smoothing_radius = 1
        self._diag = 0.5 * math.sqrt(self.W ** 2 + self.H ** 2)
        self._stall_steps = 0
        self._recent_signatures = deque(maxlen=12)
        self._component_failures: Dict[Tuple[int, int, int, int, int, int], int] = {}
        self._class_failures: Dict[int, int] = {}
        self._last_selected_signature: Optional[Tuple[int, int, int, int, int, int]] = None
        self._last_selected_class: Optional[int] = None
        cycle = list(region_selection_cycle or ["miou_gain"])
        normalized_cycle: List[str] = []
        for mode in cycle:
            mode_norm = str(mode).strip().lower()
            if not mode_norm:
                continue
            if mode_norm not in self.VALID_REGION_SELECTION_MODES:
                raise ValueError(
                    f"Unknown region selection mode: {mode!r}. "
                    f"Expected one of {self.VALID_REGION_SELECTION_MODES}."
                )
            normalized_cycle.append(mode_norm)
        if not normalized_cycle:
            normalized_cycle = ["miou_gain"]
        self.region_selection_cycle: Tuple[str, ...] = tuple(normalized_cycle)
        self._selection_step: int = 0
        self._class_area = np.array(
            [int(np.count_nonzero(self.gt == cid)) for cid in range(num_classes)],
            dtype=np.int64,
        )
        self._gt_component_count = np.zeros(num_classes, dtype=np.int64)
        for cid in range(num_classes):
            if self._class_area[cid] <= 0:
                continue
            _, ncc = cc_label(self.gt == cid)
            self._gt_component_count[cid] = int(ncc)
        balance_weights = np.zeros(num_classes, dtype=np.float64)
        for cid in range(num_classes):
            area = float(self._class_area[cid])
            if area <= 0:
                continue
            frag_bonus = 1.0 + 0.35 * math.log1p(float(self._gt_component_count[cid]))
            weight = math.sqrt(area) * frag_bonus
            if cid == 0:
                weight *= 0.45
            balance_weights[cid] = weight
        bal_sum = float(balance_weights.sum())
        self._class_balance_target = (
            balance_weights / bal_sum
            if bal_sum > 1e-12
            else np.full(num_classes, 1.0 / max(1, num_classes), dtype=np.float64)
        )

        # внутренние маски каждого класса (с отступом margin)
        self._gt_inner: List[np.ndarray] = []
        for cid in range(num_classes):
            cls = (self.gt == cid)
            if self.margin > 0 and cls.any():
                inner = cls & (_edt_inside(cls) > self.margin)
            else:
                inner = cls.copy()
            self._gt_inner.append(inner)

    def set_selection_step(self, step: int) -> None:
        self._selection_step = max(0, int(step))

    def _current_region_selection_mode(self) -> str:
        return self.region_selection_cycle[
            self._selection_step % len(self.region_selection_cycle)
        ]

    def _advance_region_selection_cycle(self) -> None:
        self._selection_step += 1

    def _region_error_mask(
        self,
        cid: int,
        pred_mask: np.ndarray,
        mode: str,
        annotated_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        gt_c = (self.gt == cid)
        known_mask = (
            np.asarray(annotated_mask, dtype=bool)
            if annotated_mask is not None
            else (pred_mask >= 0)
        )
        if mode == "miou_gain":
            return gt_c & (pred_mask != cid)
        if mode == "largest_error":
            return gt_c & known_mask & (pred_mask != cid)
        if mode == "unannotated":
            return gt_c & (~known_mask)
        raise ValueError(f"Unsupported region selection mode: {mode!r}")

    # -- helpers ---
    def _largest_component(self, bad: np.ndarray) -> Optional[np.ndarray]:
        lab, n = cc_label(bad)
        if n == 0:
            return None
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        k = int(np.argmax(counts))
        return (lab == k) if counts[k] > 0 else None

    @staticmethod
    def _center_pixel(region_mask: np.ndarray) -> Optional[Tuple[int, int, np.ndarray]]:
        if not region_mask.any():
            return None
        dist = _edt_inside(region_mask)
        max_dt = float(dist.max())
        if max_dt <= 0.0:
            return None
        ys, xs = np.where(dist >= (max_dt - 1e-6))
        all_ys, all_xs = np.where(region_mask)
        target_x = float(all_xs.mean())
        target_y = float(all_ys.mean())
        d2 = (xs.astype(np.float64) - target_x) ** 2 + (ys.astype(np.float64) - target_y) ** 2
        idx = int(np.argmin(d2))
        return int(xs[idx]), int(ys[idx]), dist

    def _ranked_components(self, bad: np.ndarray, limit: int = 3) -> List[np.ndarray]:
        lab, n = cc_label(bad)
        if n == 0:
            return []
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        ranked = np.argsort(counts)[::-1]
        out: List[np.ndarray] = []
        for k in ranked:
            if int(k) == 0 or counts[int(k)] <= 0:
                continue
            out.append(lab == int(k))
            if len(out) >= int(limit):
                break
        return out

    def _component_signature(
        self,
        cid: int,
        comp: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, int, int]]:
        ys, xs = np.where(comp)
        if xs.size == 0:
            return None
        return (
            int(cid),
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
            int(comp.sum()),
        )

    def _class_inter_union(self, pred_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inter = np.zeros(self.num_classes, dtype=np.int64)
        union = np.zeros(self.num_classes, dtype=np.int64)
        for cid in range(self.num_classes):
            pred_c = (pred_mask == cid)
            gt_c = (self.gt == cid)
            inter[cid] = int(np.logical_and(pred_c, gt_c).sum())
            union[cid] = int(np.logical_or(pred_c, gt_c).sum())
        return inter, union

    def _class_priority(
        self,
        cid: int,
        inter: np.ndarray,
        union: np.ndarray,
        bad_c: np.ndarray,
        class_scribble_counts: Optional[List[int]],
    ) -> float:
        union_c = max(1, int(union[cid]))
        iou_c = float(inter[cid]) / float(union_c)
        bad_ratio = float(np.count_nonzero(bad_c)) / float(union_c)
        class_fail = float(self._class_failures.get(int(cid), 0))
        n_scr = 0 if class_scribble_counts is None else int(class_scribble_counts[cid])
        total_scr = (
            0
            if class_scribble_counts is None
            else int(sum(int(v) for v in class_scribble_counts))
        )
        actual_share = (float(n_scr) / float(total_scr)) if total_scr > 0 else 0.0
        target_share = float(self._class_balance_target[cid])
        balance_deficit = max(0.0, target_share - actual_share)
        frag_bonus = 1.0 + 0.35 * math.log1p(float(self._gt_component_count[cid]))
        scarcity_bonus = 1.0 / float(1 + n_scr)
        score = (
            (1.0 - iou_c) * frag_bonus
            + 0.75 * balance_deficit
            + 0.50 * bad_ratio
            + 0.20 * scarcity_bonus
            - 0.25 * class_fail
        )
        return float(score)

    def _build_allowed_mask(
        self,
        cid: int,
        comp: np.ndarray,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
    ) -> np.ndarray:
        bad_c = (self.gt == cid) & (pred_mask != cid)
        if self.border_margin > 0 and comp.any():
            comp_inner = comp & (_edt_inside(comp) > self.border_margin)
        else:
            comp_inner = comp

        gt_inner = self._gt_inner[cid]
        allowed = comp_inner & gt_inner & bad_c
        if self.no_overlap:
            allowed &= ~used_mask
        return allowed

    def _analysis_region(self, allowed: np.ndarray, comp: np.ndarray) -> np.ndarray:
        focus = allowed if allowed.any() else comp
        if not focus.any():
            return focus.copy()
        smoothed = _smooth_region_mask(focus, radius=self.smoothing_radius)
        largest = self._largest_component(smoothed)
        if largest is not None and largest.any():
            min_keep = max(2, int(0.65 * float(np.count_nonzero(focus))))
            if int(np.count_nonzero(largest)) >= min_keep:
                return largest
        return focus.copy()

    def _nearest_allowed_point(
        self,
        allowed_xs: np.ndarray,
        allowed_ys: np.ndarray,
        x: float,
        y: float,
    ) -> Tuple[int, int]:
        d2 = (allowed_xs.astype(np.float64) - float(x)) ** 2 + (allowed_ys.astype(np.float64) - float(y)) ** 2
        idx = int(np.argmin(d2))
        return int(allowed_xs[idx]), int(allowed_ys[idx])

    def _principal_directions(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return [(1.0, 0.0), (0.0, 1.0)]
        if xs.size == 1:
            return [(1.0, 0.0), (0.0, 1.0)]
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = np.cov(pts, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]
        norm = float(np.linalg.norm(major))
        if norm <= 1e-8:
            return [(1.0, 0.0), (0.0, 1.0)]
        major = major / norm
        minor = np.array([-major[1], major[0]], dtype=np.float64)
        dirs = [
            (float(major[0]), float(major[1])),
            (float(minor[0]), float(minor[1])),
            (1.0, 0.0),
            (0.0, 1.0),
        ]
        unique_dirs: List[Tuple[float, float]] = []
        for dx, dy in dirs:
            if any(abs(dx * ox + dy * oy) > 0.98 for ox, oy in unique_dirs):
                continue
            unique_dirs.append((dx, dy))
        return unique_dirs

    def _sample_anchor_points(
        self,
        focus_mask: np.ndarray,
        allowed: np.ndarray,
        center_dt: np.ndarray,
    ) -> List[Tuple[int, int]]:
        allowed_ys, allowed_xs = np.where(allowed)
        if allowed_xs.size == 0:
            return []

        ys, xs = np.where(focus_mask)
        if xs.size == 0:
            xs = allowed_xs
            ys = allowed_ys

        top_k = min(24, xs.size)
        weights = center_dt[ys, xs]
        ranked = np.argsort(weights)[::-1][:top_k]
        candidates: List[Tuple[int, int]] = []

        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        candidates.append(self._nearest_allowed_point(allowed_xs, allowed_ys, centroid_x, centroid_y))

        best_idx = int(ranked[0]) if ranked.size else 0
        candidates.append(
            self._nearest_allowed_point(
                allowed_xs,
                allowed_ys,
                float(xs[best_idx]),
                float(ys[best_idx]),
            )
        )

        if ranked.size > 1:
            far_rank = max(
                ranked,
                key=lambda idx: (
                    (float(xs[idx]) - centroid_x) ** 2
                    + (float(ys[idx]) - centroid_y) ** 2
                ),
            )
            candidates.append(
                self._nearest_allowed_point(
                    allowed_xs,
                    allowed_ys,
                    float(xs[int(far_rank)]),
                    float(ys[int(far_rank)]),
                )
            )

        for idx in ranked:
            pt = self._nearest_allowed_point(
                allowed_xs,
                allowed_ys,
                float(xs[int(idx)]),
                float(ys[int(idx)]),
            )
            if pt not in candidates:
                candidates.append(pt)
            if len(candidates) >= 4:
                break
        return candidates

    @staticmethod
    def _normalized_dt_map(dist_map: np.ndarray) -> np.ndarray:
        max_dt = float(np.max(dist_map))
        if max_dt <= 1e-6:
            return np.zeros_like(dist_map, dtype=np.float32)
        return np.clip(dist_map.astype(np.float32) / float(max_dt), 0.0, 1.0)

    def _edt_corridor_masks(
        self,
        allowed: np.ndarray,
        dist_map: np.ndarray,
    ) -> List[np.ndarray]:
        allowed = np.asarray(allowed, dtype=bool)
        if not allowed.any():
            return []

        allowed_vals = dist_map[allowed]
        max_dt = float(np.max(allowed_vals)) if allowed_vals.size > 0 else 0.0
        if max_dt <= 1e-6:
            return [allowed.copy()]

        norm_dt = self._normalized_dt_map(dist_map)
        strict_interior = allowed & (dist_map > (1.0 + 1e-6))
        corridor_base = strict_interior if strict_interior.any() else allowed
        corridor = corridor_base & (norm_dt >= (0.40 - 1e-6))

        masks: List[np.ndarray] = []
        if corridor.any():
            masks.append(corridor)
        elif corridor_base.any():
            masks.append(corridor_base)

        plateau = allowed & (dist_map >= (max_dt - 1e-6))
        if plateau.any():
            masks.append(plateau)
        return masks

    def _build_scribble_core(
        self,
        allowed: np.ndarray,
        center_dt: np.ndarray,
    ) -> np.ndarray:
        """
        Builds an inner-core mask of the current bad region. The scribble is
        preferably traced through this core, so the line follows the interior
        of the unlabeled / mislabeled region rather than grazing its border.
        """
        if not allowed.any():
            return allowed

        min_pixels = max(6, int(0.03 * float(np.count_nonzero(allowed))))
        best_core: Optional[np.ndarray] = None
        best_size = -1

        for core in self._edt_corridor_masks(allowed, center_dt):
            if not core.any():
                continue
            core_largest = self._largest_component(core)
            if core_largest is None:
                continue
            cur_size = int(np.count_nonzero(core_largest))
            if cur_size > best_size:
                best_size = cur_size
                best_core = core_largest
            if cur_size >= min_pixels:
                return core_largest

        if best_core is not None and best_core.any():
            return best_core
        return allowed & (center_dt >= (float(center_dt.max()) - 1e-6))

    def _segment_pixels(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        xs = np.clip(xs, 0, self.W - 1)
        ys = np.clip(ys, 0, self.H - 1)
        return xs, ys

    def _polyline_pixels(self, pts_px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts_px = np.asarray(pts_px, dtype=np.int32)
        if pts_px.ndim != 2 or pts_px.shape[0] == 0:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        if pts_px.shape[0] == 1:
            return pts_px[:, 0].copy(), pts_px[:, 1].copy()

        xs_parts: List[np.ndarray] = []
        ys_parts: List[np.ndarray] = []
        for i in range(pts_px.shape[0] - 1):
            xs, ys = self._segment_pixels(
                int(pts_px[i, 0]),
                int(pts_px[i, 1]),
                int(pts_px[i + 1, 0]),
                int(pts_px[i + 1, 1]),
            )
            if i > 0 and xs.size > 0:
                xs = xs[1:]
                ys = ys[1:]
            xs_parts.append(xs)
            ys_parts.append(ys)
        return np.concatenate(xs_parts), np.concatenate(ys_parts)

    def _polyline_band_mask(
        self,
        pts_px: np.ndarray,
        radius: int = 2,
    ) -> np.ndarray:
        mask = np.zeros((self.H, self.W), dtype=bool)
        xs, ys = self._polyline_pixels(pts_px)
        r = max(0, int(radius))
        for x, y in zip(xs, ys):
            mask[max(0, y - r):min(self.H, y + r + 1),
                 max(0, x - r):min(self.W, x + r + 1)] = True
        return mask

    def _nearest_mask_point(
        self,
        mask: np.ndarray,
        x: float,
        y: float,
    ) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(mask)
        if xs.size == 0:
            return None
        return self._nearest_allowed_point(xs, ys, x, y)

    @staticmethod
    def _simplify_polyline_pixels(pts_px: np.ndarray) -> np.ndarray:
        pts_px = np.asarray(pts_px, dtype=np.int32)
        if pts_px.ndim != 2 or pts_px.shape[0] <= 2:
            return pts_px.copy()

        deduped: List[np.ndarray] = [pts_px[0]]
        for pt in pts_px[1:]:
            if not np.array_equal(pt, deduped[-1]):
                deduped.append(pt)
        if len(deduped) <= 2:
            return np.asarray(deduped, dtype=np.int32)

        simplified: List[np.ndarray] = [deduped[0]]
        for i in range(1, len(deduped) - 1):
            a = simplified[-1]
            b = deduped[i]
            c = deduped[i + 1]
            v1 = b - a
            v2 = c - b
            cross = int(v1[0]) * int(v2[1]) - int(v1[1]) * int(v2[0])
            dot = int(v1[0]) * int(v2[0]) + int(v1[1]) * int(v2[1])
            if cross == 0 and dot >= 0:
                continue
            simplified.append(b)
        simplified.append(deduped[-1])
        return np.asarray(simplified, dtype=np.int32)

    @staticmethod
    def _reconstruct_path(
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        end_xy: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        cur: Optional[Tuple[int, int]] = end_xy
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

    def _centerline_path_from_skeleton(
        self,
        skeleton: np.ndarray,
        allowed: np.ndarray,
        center_xy: Tuple[int, int],
        axis_mask: np.ndarray,
        dt_map: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, int, float]]:
        if not skeleton.any():
            return None

        center_skel = self._nearest_mask_point(skeleton, center_xy[0], center_xy[1])
        if center_skel is None:
            return None

        skel_labels, _ = ndimage.label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
        label_id = int(skel_labels[center_skel[1], center_skel[0]])
        if label_id <= 0:
            return None
        skel_comp = skel_labels == label_id

        queue = deque([center_skel])
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {center_skel: None}
        dist_steps: Dict[Tuple[int, int], int] = {center_skel: 0}
        degrees: Dict[Tuple[int, int], int] = {}

        while queue:
            cur_x, cur_y = queue.popleft()
            deg = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = cur_x + dx
                    ny = cur_y + dy
                    if not (0 <= nx < self.W and 0 <= ny < self.H):
                        continue
                    if not skel_comp[ny, nx]:
                        continue
                    deg += 1
                    if (nx, ny) in prev:
                        continue
                    prev[(nx, ny)] = (cur_x, cur_y)
                    dist_steps[(nx, ny)] = dist_steps[(cur_x, cur_y)] + 1
                    queue.append((nx, ny))
            degrees[(cur_x, cur_y)] = deg

        nodes = list(prev.keys())
        if len(nodes) <= 1:
            return None

        axis_xy = self._principal_directions(axis_mask)[0]
        endpoints = [node for node, deg in degrees.items() if deg <= 1 and node != center_skel]
        endpoint_pool = endpoints if endpoints else [node for node in nodes if node != center_skel]
        if not endpoint_pool:
            return None

        cx, cy = center_skel
        ax, ay = axis_xy
        endpoint_data: List[Tuple[List[Tuple[int, int]], int, Tuple[int, int], float]] = []
        for end_xy in endpoint_pool:
            path = self._reconstruct_path(prev, end_xy)
            if len(path) <= 1:
                continue
            branch_key = path[1]
            proj = (float(end_xy[0]) - float(cx)) * float(ax) + (float(end_xy[1]) - float(cy)) * float(ay)
            endpoint_data.append((path, int(dist_steps.get(end_xy, -1)), branch_key, float(proj)))
        if not endpoint_data:
            return None

        best_pair: Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = None
        best_pair_score: Optional[Tuple[int, int, float]] = None
        for i in range(len(endpoint_data)):
            path_i, dist_i, branch_i, proj_i = endpoint_data[i]
            for j in range(i + 1, len(endpoint_data)):
                path_j, dist_j, branch_j, proj_j = endpoint_data[j]
                if branch_i == branch_j:
                    continue
                score = (
                    int(dist_i + dist_j),
                    int(min(dist_i, dist_j)),
                    float(abs(proj_i - proj_j)),
                )
                if best_pair_score is None or score > best_pair_score:
                    best_pair_score = score
                    best_pair = (path_i, path_j)

        if best_pair is not None:
            path_a, path_b = best_pair
            path_xy = list(reversed(path_a)) + path_b[1:]
        else:
            neg_end = self._select_centerline_endpoint(
                endpoint_pool, center_skel, axis_xy, dist_steps, dt_map, sign=-1.0
            )
            pos_end = self._select_centerline_endpoint(
                endpoint_pool, center_skel, axis_xy, dist_steps, dt_map, sign=+1.0
            )
            if neg_end == center_skel and pos_end == center_skel:
                far_end = max(nodes, key=lambda node: dist_steps.get(node, -1))
                if far_end == center_skel:
                    return None
                path_xy = self._reconstruct_path(prev, far_end)
            else:
                neg_path = self._reconstruct_path(prev, neg_end)
                pos_path = self._reconstruct_path(prev, pos_end)
                path_xy = list(reversed(neg_path)) + pos_path[1:]

        pts_px = np.asarray(path_xy, dtype=np.int32)
        if pts_px.shape[0] < 2:
            return None

        if not np.all(allowed[pts_px[:, 1], pts_px[:, 0]]):
            _, nearest_allowed = ndimage.distance_transform_edt(~allowed, return_indices=True)
            ys = pts_px[:, 1]
            xs = pts_px[:, 0]
            proj_y = nearest_allowed[0, ys, xs]
            proj_x = nearest_allowed[1, ys, xs]
            pts_px = np.stack([proj_x, proj_y], axis=1).astype(np.int32, copy=False)

        pts_px = self._simplify_polyline_pixels(pts_px)
        if pts_px.shape[0] < 2:
            return None

        center_allowed = self._nearest_mask_point(allowed, center_xy[0], center_xy[1])
        if center_allowed is None:
            return None
        center_index = int(
            np.argmin(
                (pts_px[:, 0].astype(np.float64) - float(center_allowed[0])) ** 2
                + (pts_px[:, 1].astype(np.float64) - float(center_allowed[1])) ** 2
            )
        )
        seg = np.diff(pts_px.astype(np.float64), axis=0)
        path_len = float(np.sum(np.sqrt((seg ** 2).sum(axis=1))))
        return pts_px, center_index, path_len

    def _select_centerline_endpoint(
        self,
        nodes: List[Tuple[int, int]],
        center_xy: Tuple[int, int],
        axis_xy: Tuple[float, float],
        dist_steps: Dict[Tuple[int, int], int],
        dt_map: np.ndarray,
        sign: float,
    ) -> Tuple[int, int]:
        cx, cy = center_xy
        ax, ay = axis_xy
        directed: List[Tuple[float, int, float, Tuple[int, int]]] = []
        fallback: List[Tuple[int, float, float, Tuple[int, int]]] = []
        for x, y in nodes:
            if (x, y) == center_xy:
                continue
            dist = int(dist_steps.get((x, y), -1))
            if dist < 0:
                continue
            proj = float(sign) * (
                (float(x) - float(cx)) * float(ax) + (float(y) - float(cy)) * float(ay)
            )
            dt = float(dt_map[y, x]) if 0 <= x < self.W and 0 <= y < self.H else 0.0
            fallback.append((dist, abs(proj), dt, (x, y)))
            if proj > 0.0:
                directed.append((proj, dist, dt, (x, y)))
        if directed:
            directed.sort(reverse=True)
            return directed[0][3]
        if fallback:
            fallback.sort(reverse=True)
            return fallback[0][3]
        return center_xy

    def _build_centerline_path(
        self,
        allowed: np.ndarray,
        analysis_mask: np.ndarray,
        focus_mask: np.ndarray,
        comp: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, int]]:
        source_mask = analysis_mask if analysis_mask.any() else allowed
        if not source_mask.any():
            return None

        center_data = self._center_pixel(source_mask)
        if center_data is None:
            return None
        cx, cy, center_dt = center_data
        focus_for_axis = focus_mask if focus_mask.any() else (analysis_mask if analysis_mask.any() else comp)
        max_dt = float(center_dt.max())
        skeleton_candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        for ridge_mask in self._edt_corridor_masks(source_mask, center_dt):
            if not ridge_mask.any():
                continue
            skeleton_candidates.append((skeletonize(ridge_mask).astype(bool, copy=False), ridge_mask))
            skeleton_candidates.append((medial_axis(ridge_mask).astype(bool, copy=False), ridge_mask))

        best_path: Optional[Tuple[np.ndarray, int]] = None
        best_score = -1.0
        for skeleton, corridor_mask in skeleton_candidates:
            path_choice = self._centerline_path_from_skeleton(
                skeleton=skeleton,
                allowed=corridor_mask,
                center_xy=(cx, cy),
                axis_mask=focus_for_axis,
                dt_map=center_dt,
            )
            if path_choice is None:
                continue
            pts_px, center_index, path_len = path_choice
            xs, ys = self._polyline_pixels(pts_px)
            if xs.size == 0:
                continue
            center_support = (
                float(np.mean(center_dt[ys, xs])) / float(max(1e-6, max_dt))
                if max_dt > 1e-6
                else 0.0
            )
            score = float(path_len) + 0.5 * float(center_support)
            if score <= best_score:
                continue
            best_score = score
            best_path = (pts_px, center_index)
        return best_path

    def _straight_fallback_path(
        self,
        allowed: np.ndarray,
        focus_mask: np.ndarray,
        center_dt: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, int]]:
        anchors = self._sample_anchor_points(focus_mask if focus_mask.any() else allowed, allowed, center_dt)
        directions = self._principal_directions(focus_mask if focus_mask.any() else allowed)
        best_pts: Optional[np.ndarray] = None
        best_center_idx = 0
        best_len = -1.0

        for x0, y0 in anchors:
            for dx, dy in directions:
                x1a, y1a = self._trace(x0, y0, +dx, +dy, allowed, np.zeros_like(allowed, dtype=bool))
                x1b, y1b = self._trace(x0, y0, -dx, -dy, allowed, np.zeros_like(allowed, dtype=bool))
                path_px = np.array([[x1b, y1b], [x0, y0], [x1a, y1a]], dtype=np.int32)
                path_px = self._simplify_polyline_pixels(path_px)
                if path_px.shape[0] < 2:
                    continue
                seg = np.diff(path_px.astype(np.float64), axis=0)
                cur_len = float(np.sum(np.sqrt((seg ** 2).sum(axis=1))))
                if cur_len <= best_len:
                    continue
                best_len = cur_len
                best_pts = path_px
                best_center_idx = int(
                    np.argmin(
                        (path_px[:, 0].astype(np.float64) - float(x0)) ** 2
                        + (path_px[:, 1].astype(np.float64) - float(y0)) ** 2
                    )
                )

        if best_pts is None:
            return None
        return best_pts, best_center_idx

    def _candidate_scribble_score(
        self,
        cid: int,
        comp: np.ndarray,
        allowed: np.ndarray,
        scribble_core: np.ndarray,
        allowed_dt: np.ndarray,
        union_c: int,
        pts_px: np.ndarray,
        center_index: int,
    ) -> Tuple[float, float, float, float, float, float, float]:
        band = self._polyline_band_mask(pts_px, radius=2)
        direct_correct = int(np.count_nonzero(band & allowed))
        direct_gain = float(direct_correct) / float(max(1, union_c))
        comp_covered = float(np.count_nonzero(band & comp)) / float(max(1, int(np.count_nonzero(comp))))
        core_covered = float(np.count_nonzero(band & scribble_core)) / float(
            max(1, int(np.count_nonzero(scribble_core)))
        )
        xs, ys = self._polyline_pixels(pts_px)
        center_support = (
            float(np.mean(allowed_dt[ys, xs])) / float(max(1e-6, float(allowed_dt.max())))
            if xs.size > 0 and float(allowed_dt.max()) > 1e-6
            else 0.0
        )
        center_index = int(max(0, min(center_index, pts_px.shape[0] - 1)))
        cx = int(pts_px[center_index, 0])
        cy = int(pts_px[center_index, 1])
        seed_support = (
            float(allowed_dt[cy, cx]) / float(max(1e-6, float(allowed_dt.max())))
            if 0 <= cx < self.W and 0 <= cy < self.H and float(allowed_dt.max()) > 1e-6
            else 0.0
        )
        min_support = (
            float(np.min(allowed_dt[ys, xs])) / float(max(1e-6, float(allowed_dt.max())))
            if xs.size > 0 and float(allowed_dt.max()) > 1e-6
            else 0.0
        )
        path_xy = pts_px.astype(np.float64)
        seg_lens = np.sqrt((np.diff(path_xy, axis=0) ** 2).sum(axis=1))
        arm_a = float(seg_lens[:center_index].sum()) if center_index > 0 else 0.0
        arm_b = float(seg_lens[center_index:].sum()) if center_index < seg_lens.size else 0.0
        center_balance = (
            float(min(arm_a, arm_b) / max(1e-6, max(arm_a, arm_b)))
            if max(arm_a, arm_b) > 1e-6
            else 0.0
        )
        length_norm = float(seg_lens.sum()) / float(max(1, max(self.H, self.W)))
        # Prefer lines that stay on the interior core of the erroneous region.
        # This makes the scribble follow the central part of the bad component,
        # which is usually more informative and less brittle near boundaries.
        score = (
            4.0 * direct_gain
            + 0.90 * core_covered
            + 0.55 * comp_covered
            + 0.60 * center_support
            + 0.30 * min_support
            + 0.35 * length_norm
        )
        return (
            float(score),
            float(direct_gain),
            float(comp_covered),
            float(center_support),
            float(length_norm),
            float(seed_support),
            float(center_balance),
        )

    def _best_scribble_for_component(
        self,
        cid: int,
        comp: np.ndarray,
        allowed: np.ndarray,
        union_c: int,
    ) -> Optional[Tuple[Tuple[float, float, float, float, float, float, float, float], np.ndarray]]:
        if not allowed.any():
            return None
        analysis_mask = self._analysis_region(allowed, comp)
        center_dt = _edt_inside(analysis_mask if analysis_mask.any() else allowed)
        allowed_dt = _edt_inside(allowed)
        scribble_core = self._build_scribble_core(allowed, center_dt)
        focus_mask = scribble_core if scribble_core.any() else analysis_mask
        path_allowed = scribble_core if int(np.count_nonzero(scribble_core)) >= 2 else allowed
        path_choice = self._build_centerline_path(
            allowed=allowed,
            analysis_mask=analysis_mask,
            focus_mask=focus_mask,
            comp=comp,
        )
        if path_choice is None:
            fallback_dt = _edt_inside(path_allowed)
            path_choice = self._straight_fallback_path(path_allowed, focus_mask if focus_mask.any() else path_allowed, fallback_dt)
        if path_choice is None:
            return None

        pts_px, center_index = path_choice
        score_parts = self._candidate_scribble_score(
            cid=cid,
            comp=comp,
            allowed=allowed,
            scribble_core=scribble_core,
            allowed_dt=allowed_dt,
            union_c=union_c,
            pts_px=pts_px,
            center_index=center_index,
        )
        pts = pts_px.astype(np.float32, copy=False)
        pts01 = np.empty_like(pts, dtype=np.float32)
        pts01[:, 0] = pts[:, 0] / float(self.W)
        pts01[:, 1] = pts[:, 1] / float(self.H)
        payload = score_parts + (float(self.rng.random()),)
        return payload, pts01

    def _select_class_component(
        self,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
        annotated_mask: Optional[np.ndarray] = None,
        class_scribble_counts: Optional[List[int]] = None,
        selection_mode: str = "miou_gain",
    ) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        inter, union = self._class_inter_union(pred_mask)
        raw_candidates = []

        for cid in range(self.num_classes):
            bad_c = self._region_error_mask(
                cid,
                pred_mask,
                selection_mode,
                annotated_mask=annotated_mask,
            )
            if not bad_c.any():
                continue

            union_c = max(1, int(union[cid]))
            class_priority = self._class_priority(
                cid=cid,
                inter=inter,
                union=union,
                bad_c=bad_c,
                class_scribble_counts=class_scribble_counts,
            )
            for comp in self._ranked_components(bad_c, limit=3):
                allowed = self._build_allowed_mask(cid, comp, pred_mask, used_mask)
                if not allowed.any():
                    continue

                signature = self._component_signature(cid, comp)
                if signature is None:
                    continue

                comp_area = int(comp.sum())
                comp_fail = float(self._component_failures.get(signature, 0))
                recent_pen = 1.0 if signature in self._recent_signatures else 0.0
                comp_share = math.sqrt(float(comp_area) / float(max(1, union_c)))
                if selection_mode == "miou_gain":
                    primary = class_priority + 0.65 * comp_share
                    secondary = float(comp_area)
                elif selection_mode == "largest_error":
                    primary = float(comp_area)
                    secondary = class_priority
                elif selection_mode == "unannotated":
                    primary = float(comp_area)
                    secondary = class_priority + 0.25 * comp_share
                else:
                    raise ValueError(f"Unsupported region selection mode: {selection_mode!r}")
                raw_score = (
                    float(primary) - comp_fail - 0.05 * recent_pen,
                    float(secondary),
                    float(self.rng.random()),
                )
                raw_candidates.append((raw_score, int(cid), comp, allowed, signature, union_c))

        if not raw_candidates:
            return None, None, None, None

        raw_candidates.sort(key=lambda item: item[0], reverse=True)
        eval_budget = min(len(raw_candidates), 4 + min(2, self._stall_steps))
        candidates = []
        for raw_score, cid, comp, allowed, signature, union_c in raw_candidates[:eval_budget]:
            scribble_choice = self._best_scribble_for_component(
                cid=cid,
                comp=comp,
                allowed=allowed,
                union_c=union_c,
            )
            if scribble_choice is None:
                continue
            scribble_score, pts01 = scribble_choice
            comp_area = int(comp.sum())
            score = (
                float(raw_score[0]) + float(scribble_score[0]),
                float(scribble_score[0]),
                scribble_score[1],
                scribble_score[2],
                scribble_score[3],
                float(comp_area),
                scribble_score[4],
                float(self.rng.random()),
            )
            candidates.append((score, int(cid), comp, allowed, signature, pts01))

        if not candidates:
            return None, None, None, None

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_n = 1
        if self._stall_steps > 0:
            top_n = min(len(candidates), 1 + min(3, self._stall_steps))

        if self._stall_steps > 0 and self._last_selected_class is not None:
            alternate = [
                item for item in candidates
                if int(item[1]) != int(self._last_selected_class)
            ]
            if alternate:
                candidates = alternate
                top_n = min(len(candidates), max(1, top_n))

        if top_n > 1:
            rank_weights = np.linspace(top_n, 1, top_n, dtype=np.float64)
            rank_weights /= rank_weights.sum()
            pick = int(self.rng.choice(np.arange(top_n), p=rank_weights))
            chosen = candidates[pick]
        else:
            chosen = candidates[0]

        _, best_cid, best_comp, best_allowed, signature, best_pts = chosen
        self._last_selected_signature = signature
        self._last_selected_class = int(best_cid)
        return best_cid, best_comp, best_allowed, best_pts

    def report_last_result(self, progress: bool) -> None:
        sig = self._last_selected_signature
        cid = self._last_selected_class
        if sig is None or cid is None:
            return

        self._recent_signatures.append(sig)
        if progress:
            self._stall_steps = 0
            if sig in self._component_failures:
                self._component_failures[sig] -= 1
                if self._component_failures[sig] <= 0:
                    self._component_failures.pop(sig, None)
            if cid in self._class_failures:
                self._class_failures[cid] -= 1
                if self._class_failures[cid] <= 0:
                    self._class_failures.pop(cid, None)
        else:
            self._stall_steps += 1
            self._component_failures[sig] = self._component_failures.get(sig, 0) + 1
            self._class_failures[cid] = self._class_failures.get(cid, 0) + 1

    def _mode_gt(self, mask: np.ndarray) -> int:
        vals = self.gt[mask]
        return int(np.argmax(np.bincount(vals, minlength=self.num_classes))) if vals.size else 0

    def _trace(self, x0: int, y0: int, dx: float, dy: float,
               allowed: np.ndarray, used: np.ndarray) -> Tuple[int, int]:
        x, y = float(x0), float(y0)
        lx, ly = x0, y0
        for _ in range(int(self._diag)):
            x += dx; y += dy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < self.W and 0 <= yi < self.H):
                break
            if not allowed[yi, xi]:
                break
            if self.no_overlap and used[yi, xi]:
                break
            lx, ly = xi, yi
        return lx, ly

    def make_scribble(
        self,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
        annotated_mask: Optional[np.ndarray] = None,
        class_scribble_counts: Optional[List[int]] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Returns:
            gt_id   — 0-indexed class id that the scribble annotates
            pts01   — (N,2) float32 array of normalised [0..1] polyline points
        Raises StopIteration when all pixels are correctly labelled.
        Raises RuntimeError if a scribble cannot be placed (relax --margin).
        """
        bad = (pred_mask != self.gt)
        if not bad.any():
            raise StopIteration("All pixels correctly labelled.")

        selection_mode = self._current_region_selection_mode()
        cid, comp, allowed, pts01 = self._select_class_component(
            pred_mask,
            used_mask,
            annotated_mask=annotated_mask,
            class_scribble_counts=class_scribble_counts,
            selection_mode=selection_mode,
        )
        if cid is None and selection_mode != "miou_gain":
            cid, comp, allowed, pts01 = self._select_class_component(
                pred_mask,
                used_mask,
                annotated_mask=annotated_mask,
                class_scribble_counts=class_scribble_counts,
                selection_mode="miou_gain",
            )
        if cid is None or comp is None or allowed is None:
            comp = self._largest_component(bad)
            if comp is None:
                raise StopIteration("No bad connected component.")
            cid = self._mode_gt(comp)
            allowed = self._build_allowed_mask(cid, comp, pred_mask, used_mask)
            if not allowed.any():
                raise RuntimeError(
                    "Cannot place scribble in selected bad region. "
                    "Try smaller --border_margin/--margin or remove --no_overlap."
                )
            self._last_selected_signature = self._component_signature(cid, comp)
            self._last_selected_class = int(cid)
            fallback_choice = self._best_scribble_for_component(
                cid=cid,
                comp=comp,
                allowed=allowed,
                union_c=max(1, int(np.logical_or(pred_mask == cid, self.gt == cid).sum())),
            )
            if fallback_choice is not None:
                _, pts01 = fallback_choice

        if pts01 is not None:
            self._advance_region_selection_cycle()
            return cid, pts01

        raise RuntimeError("Failed to generate scribble after max_retries.")


def mark_line_used(used: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                   width: int = 3, pad: int = 2) -> None:
    H, W = used.shape
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, n).round().astype(int)
    ys = np.linspace(y0, y1, n).round().astype(int)
    r = width // 2 + pad
    for x, y in zip(xs, ys):
        used[max(0, y - r):min(H, y + r + 1),
            max(0, x - r):min(W, x + r + 1)] = True


def mark_polyline_used_norm(
    used: np.ndarray,
    pts01: np.ndarray,
    H: int,
    W: int,
    *,
    width: int = 3,
    pad: int = 2,
) -> None:
    pts = np.asarray(pts01, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return
    pts_px = np.empty_like(pts, dtype=np.int32)
    pts_px[:, 0] = np.round(pts[:, 0] * float(W)).astype(np.int32)
    pts_px[:, 1] = np.round(pts[:, 1] * float(H)).astype(np.int32)
    pts_px[:, 0] = np.clip(pts_px[:, 0], 0, W - 1)
    pts_px[:, 1] = np.clip(pts_px[:, 1], 0, H - 1)
    for i in range(pts_px.shape[0] - 1):
        structs.mark_line_used(
            used,
            int(pts_px[i, 0]),
            int(pts_px[i, 1]),
            int(pts_px[i + 1, 0]),
            int(pts_px[i + 1, 1]),
            width=width,
            pad=pad,
        )


def _class_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, class_id: int) -> float:
    pred_c = (pred_mask == int(class_id))
    gt_c = (gt_mask == int(class_id))
    union = int(np.logical_or(pred_c, gt_c).sum())
    if union <= 0:
        return 0.0
    inter = int(np.logical_and(pred_c, gt_c).sum())
    return float(inter) / float(union)


def _build_propagation_profiles(
    algo: "structs.SuperPixelAnnotationAlgo",
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_id: int,
    class_scribble_counts: List[int],
    sensitivity: float,
) -> List[Dict[str, float]]:
    if sensitivity <= 0.0:
        return []

    def _clamp_similarity_threshold(value: float) -> float:
        return float(min(1.0, max(0.0, value)))

    class_iou = _class_iou(pred_mask, gt_mask, gt_id)
    n_class_scribbles = int(class_scribble_counts[gt_id]) if class_scribble_counts else 0
    present_classes = [int(c) for c in np.unique(gt_mask)]
    late_stage = int(sum(int(v) for v in class_scribble_counts)) >= max(6, len(present_classes))
    base_threshold = _clamp_similarity_threshold(
        float(getattr(algo, "_embedding_threshold", 0.99))
    )
    active_method = algo.superpixel_methods[0] if algo.superpixel_methods else None
    use_embeddings = bool(getattr(algo, "embedding_weight_path", None)) or isinstance(
        active_method,
        structs.SSNSuperpixel,
    )

    if use_embeddings:
        profiles: List[Dict[str, float]] = [
            {
                "sens": float(max(1e-6, sensitivity * 0.55)),
                "radius_scale": 0.65,
                "property_scale": 0.85,
                "embedding_threshold_override": _clamp_similarity_threshold(
                    max(base_threshold, base_threshold + 0.01)
                ),
            }
        ]
        should_aggressive = (
            (late_stage and (n_class_scribbles >= 3 or class_iou >= 0.45))
            or (n_class_scribbles >= 4 and class_iou >= 0.60)
        )
        if should_aggressive:
            profiles.append(
                {
                    "sens": float(max(1e-6, sensitivity * (0.75 if late_stage else 0.65))),
                    "radius_scale": 0.90 if late_stage else 0.75,
                    "property_scale": 0.95,
                    "embedding_threshold_override": _clamp_similarity_threshold(
                        base_threshold
                    ),
                }
            )
        return profiles

    profiles = [
        {
            "sens": float(max(1e-6, sensitivity * 0.45)),
            "radius_scale": 0.55,
            "property_scale": 0.55,
            "embedding_threshold_override": _clamp_similarity_threshold(base_threshold),
        }
    ]
    should_aggressive = (
        (late_stage and (n_class_scribbles >= 3 or class_iou >= 0.45))
        or (n_class_scribbles >= 4 and class_iou >= 0.60)
    )
    if should_aggressive:
        profiles.append(
            {
                "sens": float(max(1e-6, sensitivity * (0.70 if late_stage else 0.60))),
                "radius_scale": 0.85 if late_stage else 0.75,
                "property_scale": 0.80,
                "embedding_threshold_override": _clamp_similarity_threshold(base_threshold),
            }
        )
    return profiles


# ─────────────────────────────────────────────────────────────────────────────────
#  Распространение штрихов (propagation)
# ─────────────────────────────────────────────────────────────────────────────────

def run_propagation(
    algo: "structs.SuperPixelAnnotationAlgo",
    sp_method: "structs.SuperPixelMethod",
    scrib: "structs.Scribble",
    sensitivity: float,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_scribble_counts: List[int],
) -> None:
    """
    Запускает use_sensitivity_for_region для всех суперпикселей,
    которые получили аннотацию от данного штриха.
    """
    if sensitivity <= 0.0:
        return
    sp_list = algo.superpixels.get(sp_method, [])
    if not sp_list:
        return

    # Строим карту id → list-index (нужна ПОСЛЕ add_scribble, т.к. список мог измениться)
    sp_id_to_idx = {sp.id: idx for idx, sp in enumerate(sp_list)}

    annos_obj = algo._annotations.get(sp_method)
    if annos_obj is None:
        return

    seen: set = set()
    gt_id = max(0, int(scrib.params.code) - 1)
    profiles = _build_propagation_profiles(
        algo=algo,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        gt_id=gt_id,
        class_scribble_counts=class_scribble_counts,
        sensitivity=float(sensitivity),
    )
    if not profiles:
        return
    for anno in annos_obj.annotations:
        if scrib.id in (anno.parent_scribble or []):
            sp_id = anno.parent_superpixel
            if sp_id not in seen:
                seen.add(sp_id)
                idx = sp_id_to_idx.get(sp_id)
                if idx is not None:
                    for profile in profiles:
                        algo.use_sensitivity_for_region(
                            idx,
                            float(profile["sens"]),
                            scrib,
                            radius_scale=float(profile["radius_scale"]),
                            property_scale=float(profile["property_scale"]),
                            embedding_threshold_override=float(profile["embedding_threshold_override"]),
                        )


# ─────────────────────────────────────────────────────────────────────────────────
#  Построение метода суперпикселей из аргументов
# ─────────────────────────────────────────────────────────────────────────────────

def build_sp_method(args) -> "structs.SuperPixelMethod":
    return structs.build_superpixel_method_from_args(args)


def parse_region_selection_cycle(raw_value: str) -> List[str]:
    items = [part.strip().lower() for part in str(raw_value).split(",")]
    items = [part for part in items if part]
    if not items:
        raise ValueError("Region selection cycle must not be empty.")
    invalid = [
        part for part in items
        if part not in LargestBadRegionGenerator.VALID_REGION_SELECTION_MODES
    ]
    if invalid:
        raise ValueError(
            "Unknown region selection mode(s): "
            + ", ".join(repr(x) for x in invalid)
            + ". Expected one of "
            + ", ".join(LargestBadRegionGenerator.VALID_REGION_SELECTION_MODES)
        )
    return items


# ─────────────────────────────────────────────────────────────────────────────────
#  Визуализация
# ─────────────────────────────────────────────────────────────────────────────────

def render_snapshot(
    img: Image.Image,
    algo: "structs.SuperPixelAnnotationAlgo",
    sp_method: "structs.SuperPixelMethod",
    class_info: List[Tuple[str, str]],
    out_png: Path,
    draw_borders: bool = True,
    draw_annos: bool = True,
    draw_scribbles: bool = True,
    anno_alpha: int = 110,
) -> None:
    W, H = img.size
    overlay = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if draw_annos:
        ann_obj = algo._annotations.get(sp_method)
        if ann_obj:
            for anno in ann_obj.annotations:
                b = np.asarray(anno.border, np.float32)
                poly = [(float(b[i, 0] * W), float(b[i, 1] * H)) for i in range(len(b))]
                code = int(anno.code)
                gt_id = max(0, min(code - 1, len(class_info) - 1))
                if gt_id == 0:
                    continue
                draw.polygon(poly, fill=_hex_to_rgba(class_info[gt_id][1], anno_alpha))

    if draw_borders:
        for sp in algo.superpixels.get(sp_method, []):
            b = np.asarray(sp.border, np.float32)
            poly = [(float(b[i, 0] * W), float(b[i, 1] * H)) for i in range(len(b))]
            draw.polygon(poly, outline=_SUPERPIXEL_BORDER_RGBA)

    composite = Image.alpha_composite(img.convert("RGBA"), overlay)

    if draw_scribbles:
        d2 = ImageDraw.Draw(composite)
        for s in getattr(algo, "_scribbles", []):
            pts = np.asarray(s.points, np.float32)
            line_pts = [(float(pts[i, 0] * W), float(pts[i, 1] * H)) for i in range(len(pts))]
            code = int(s.params.code)
            gt_id = max(0, min(code - 1, len(class_info) - 1))
            if gt_id == 0:
                continue
            d2.line(line_pts, fill=_hex_to_rgba(class_info[gt_id][1], 255), width=5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(str(out_png), quality=95)


def plot_metrics(
    metrics_history: List[StepMetrics],
    class_info: List[Tuple[str, str]],
    out_path: Path,
) -> None:
    """
    Строит 4-панельный график:
      (a) mIoU, coverage, annotation_precision vs шаги
      (b) mIoU, coverage vs суммарная длина штрихов (эффективность по ink)
      (c) miou_per_scribble, miou_per_1kpx vs шаги
      (d) correct_px_per_scribble, correct_px_per_px_ink vs шаги (нормированные)
    """
    if not metrics_history:
        return

    steps = [m.step for m in metrics_history]
    mious = [m.miou for m in metrics_history]
    covs = [m.coverage for m in metrics_history]
    precs = [m.annotation_precision for m in metrics_history]
    inks = [m.total_ink_px for m in metrics_history]
    miou_per_scr = [m.miou_per_scribble for m in metrics_history]
    miou_per_ink = [m.miou_per_1kpx for m in metrics_history]
    qxc = [m.quality_x_coverage for m in metrics_history]
    cp_per_scr = [m.correct_px_per_scribble for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Interactive Annotation Evaluation", fontsize=13, fontweight="bold")

    # (a) качество по шагам
    ax = axes[0, 0]
    ax.plot(steps, mious, "b-o", ms=3, label="mIoU")
    ax.plot(steps, covs, "g--s", ms=3, label="coverage")
    ax.plot(steps, precs, "r:^", ms=3, label="annotation precision")
    ax.plot(steps, qxc, "m-.", ms=3, label="mIoU × coverage")
    ax.set_xlabel("Scribble step")
    ax.set_ylabel("Value")
    ax.set_title("(a) Quality vs. Number of scribbles")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (b) качество по суммарной длине штрихов (эффективность)
    ax = axes[0, 1]
    ax.plot(inks, mious, "b-o", ms=3, label="mIoU")
    ax.plot(inks, covs, "g--s", ms=3, label="coverage")
    ax.plot(inks, qxc, "m-.", ms=3, label="mIoU × coverage")
    ax.set_xlabel("Total ink length (pixels)")
    ax.set_ylabel("Value")
    ax.set_title("(b) Quality vs. Total ink length (efficiency)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (c) удельные метрики по шагам
    ax = axes[1, 0]
    ax.plot(steps, miou_per_scr, "b-o", ms=3, label="mIoU / scribble")
    ax.plot(steps, miou_per_ink, "r--s", ms=3, label="mIoU / 1k-px")
    ax.set_xlabel("Scribble step")
    ax.set_ylabel("Value")
    ax.set_title("(c) Efficiency: mIoU per unit of interaction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) per-class IoU финального шага
    ax = axes[1, 1]
    final = metrics_history[-1]
    nc = len(class_info)
    x_pos = np.arange(nc)
    bar_vals = [
        v if not math.isnan(v) else 0.0 for v in final.per_class_iou[:nc]
    ]
    bar_colors = [_hex_to_rgb(class_info[i][1]) for i in range(nc)]
    bar_colors_norm = [(r / 255, g / 255, b / 255) for r, g, b in bar_colors]
    bars = ax.bar(x_pos, bar_vals, color=bar_colors_norm, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in class_info], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU")
    ax.set_title(f"(d) Per-class IoU at step {final.step}\n"
                 f"n_scribbles={final.n_scribbles}  ink={final.total_ink_px:.0f}px")
    ax.grid(True, alpha=0.3, axis="y")
    # Аннотируем бары: число штрихов на класс
    for i, (bar, val) in enumerate(zip(bars, bar_vals)):
        ns = final.per_class_n_scribbles[i] if i < len(final.per_class_n_scribbles) else 0
        ax.text(bar.get_x() + bar.get_width() / 2, min(val + 0.02, 1.0),
                f"n={ns}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────────
#  Главный цикл: одно изображение
# ─────────────────────────────────────────────────────────────────────────────────

def run_single_image(
    img: Image.Image,
    gt: np.ndarray,
    sp_method_proto: "structs.SuperPixelMethod",
    args,
    out_dir: Path,
    class_info: List[Tuple[str, str]],
    logger: logging.Logger,
    spanno_path: Optional[str] = None,
) -> List[StepMetrics]:
    """
    Запускает цикл авторазметки для одного изображения.
    Returns список StepMetrics (по одной записи на checkpoint).
    """
    num_classes = len(class_info)
    H, W = gt.shape

    # ── Инициализация algo ──────────────────────────────────────────────────────
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=float(args.downscale),
        superpixel_methods=[],
        image_path="",
        image=img,
        auto_propagation_sensitivity=0.0,
    )

    # embedding propagation
    if args.emb_weights:
        algo.embedding_weight_path = str(args.emb_weights)
        algo.embedding_fdim = int(args.ssn_fdim)
        algo.embedding_color_scale = float(args.ssn_color_scale)
        algo.embedding_pos_scale = float(args.ssn_pos_scale)
        algo._embedding_threshold = float(args.emb_threshold)

    # precomputed spanno или resume-state, или добавляем метод свежий
    use_precomputed = False
    resumed_from_state = False
    if spanno_path and Path(spanno_path).exists():
        spanno_size = load_spanno_image_size(spanno_path)
        if spanno_size is not None and tuple(spanno_size) != tuple(img.size):
            raise ValueError(
                "The provided --spanno was created for a different image size: "
                f"spanno={spanno_size[0]}x{spanno_size[1]}, "
                f"current={img.size[0]}x{img.size[1]}. "
                "For a cropped image, recompute superpixels for that crop and pass the new .spanno file."
            )
        algo.deserialize(spanno_path)
        resumed_from_state = len(getattr(algo, "_scribbles", [])) > 0
        use_precomputed = not resumed_from_state
        # найти совпадающий метод в загруженном файле
        want = sp_method_proto.short_string()
        sp_method = next(
            (m for m in algo.superpixel_methods if m.short_string() == want),
            None,
        )
        if sp_method is None:
            sp_method = algo.superpixel_methods[0] if algo.superpixel_methods else sp_method_proto
        logger.info(
            "Loaded %s: %s  active=%s",
            "resume-state" if resumed_from_state else "precomputed spanno",
            spanno_path,
            sp_method.short_string(),
        )
    else:
        sp_method = sp_method_proto
        algo.add_superpixel_method(sp_method)
        logger.info(
            "Hint: для более стабильной оценки лучше сначала автоматически сегментировать "
            "изображение суперпикселями и передать precomputed --spanno. Это помогает "
            "избежать странных объединений разных регионов в один суперпиксель."
        )

    # Убеждаемся, что sp_method является первым (некоторые методы смотрят на [0])
    if algo.superpixel_methods and algo.superpixel_methods[0] is not sp_method:
        try:
            idx = [m.short_string() for m in algo.superpixel_methods].index(sp_method.short_string())
            algo.superpixel_methods.insert(0, algo.superpixel_methods.pop(idx))
        except ValueError:
            algo.superpixel_methods.insert(0, sp_method)

    # ── Состояние ───────────────────────────────────────────────────────────────
    pred_upd = PredMaskUpdater(H, W)
    used_mask = np.zeros((H, W), dtype=bool)

    generator = LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=num_classes,
        seed=int(args.seed),
        margin=int(args.margin),
        border_margin=int(args.border_margin),
        no_overlap=bool(args.no_overlap),
        max_retries=200,
        region_selection_cycle=parse_region_selection_cycle(args.region_selection_cycle),
    )

    # Накопители стоимости взаимодействия
    total_ink: float = 0.0
    ink_list: List[float] = []          # длина каждого штриха
    per_class_n_scr: List[int] = [0] * num_classes
    per_class_ink: List[float] = [0.0] * num_classes
    start_step = 0
    next_scribble_id = 1

    if resumed_from_state and getattr(algo, "_scribbles", None):
        existing_scribbles = list(algo._scribbles)
        start_step = len(existing_scribbles)
        generator.set_selection_step(start_step)
        existing_ids = [int(getattr(s, "id", 0)) for s in existing_scribbles]
        next_scribble_id = (max(existing_ids) + 1) if existing_ids else 1

        pred_upd.repaint_all(algo, sp_method, num_classes)

        if args.no_overlap:
            for scrib in existing_scribbles:
                pts = np.asarray(scrib.points, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                mark_polyline_used_norm(used_mask, pts, H, W, width=3, pad=2)

        for scrib in existing_scribbles:
            pts = np.asarray(scrib.points, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            gt_id = max(0, min(int(scrib.params.code) - 1, num_classes - 1))
            length = scribble_length_px(pts, H, W)
            ink_list.append(length)
            total_ink += length
            per_class_n_scr[gt_id] += 1
            per_class_ink[gt_id] += length

    metrics_history: List[StepMetrics] = []
    dynamic_history: List[dict[str, object]] = []

    # ── CSV ─────────────────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer: Optional[csv.DictWriter] = None   # инициализируем на первой записи

    dynamic_csv_path = out_dir / "dynamic_metrics.csv"
    dynamic_csv_file = open(dynamic_csv_path, "w", newline="", encoding="utf-8")
    dynamic_csv_writer: Optional[csv.DictWriter] = None

    def write_dynamic_metrics(
        step: int,
        pred: np.ndarray,
        n_scr: int,
        total_ink_now: float,
        explicit_mask: Optional[np.ndarray] = None,
    ) -> None:
        nonlocal dynamic_csv_writer

        annotated_mask = (
            np.asarray(explicit_mask, dtype=bool)
            if explicit_mask is not None
            else (pred >= 0)
        )
        annotated_px = int(annotated_mask.sum())
        correct_mask = annotated_mask & (pred == gt)
        correctly_px = int(correct_mask.sum())
        total_px = H * W
        coverage = annotated_px / total_px
        ann_prec = correctly_px / annotated_px if annotated_px > 0 else 0.0
        dyn_miou, dyn_per_iou = compute_ious(pred, gt, num_classes)

        row: dict[str, object] = {
            "step": int(step),
            "n_scribbles": int(n_scr),
            "total_ink_px": round(float(total_ink_now), 2),
            "annotated_px": annotated_px,
            "correctly_annotated_px": correctly_px,
            "total_px": total_px,
            "coverage": round(float(coverage), 6),
            "annotation_precision": round(float(ann_prec), 6),
            "dynamic_miou": round(float(dyn_miou), 6) if not math.isnan(dyn_miou) else float("nan"),
        }
        for i, name in enumerate([c[0] for c in class_info]):
            iou_v = dyn_per_iou[i] if i < len(dyn_per_iou) else float("nan")
            row[f"dynamic_iou_{name}"] = round(float(iou_v), 6) if not math.isnan(iou_v) else float("nan")

        dynamic_history.append(row)
        if dynamic_csv_writer is None:
            dynamic_csv_writer = csv.DictWriter(dynamic_csv_file, fieldnames=list(row.keys()))
            dynamic_csv_writer.writeheader()
        dynamic_csv_writer.writerow(row)
        dynamic_csv_file.flush()

        logger.info(
            "dynamic step=%d scribbles=%d dyn_mIoU=%.4f cov=%.4f prec=%.4f",
            step,
            n_scr,
            float(dyn_miou) if not math.isnan(dyn_miou) else 0.0,
            coverage,
            ann_prec,
        )

    def checkpoint(step: int) -> StepMetrics:
        nonlocal csv_writer
        # Пересчитываем pred_mask полностью (включает propagated аннотации)
        pred_upd.repaint_all(algo, sp_method, num_classes)
        pred = pred_upd.pred_np

        annotated_mask = pred_upd.explicit_np
        annotated_px = int(annotated_mask.sum())
        correct_mask = annotated_mask & (pred == gt)
        correctly_px = int(correct_mask.sum())
        total_px = H * W
        coverage = annotated_px / total_px
        ann_prec = correctly_px / annotated_px if annotated_px > 0 else 0.0

        miou, per_iou = compute_ious(pred, gt, num_classes)
        n_scr = len(ink_list)
        mean_ink = (total_ink / n_scr) if n_scr > 0 else 0.0

        m = StepMetrics(
            step=step,
            n_scribbles=n_scr,
            total_ink_px=total_ink,
            mean_ink_px=mean_ink,
            annotated_px=annotated_px,
            correctly_annotated_px=correctly_px,
            total_px=total_px,
            coverage=coverage,
            annotation_precision=ann_prec,
            miou=miou if not math.isnan(miou) else 0.0,
            per_class_iou=per_iou,
            per_class_n_scribbles=list(per_class_n_scr),
            per_class_ink_px=list(per_class_ink),
        )
        metrics_history.append(m)

        flat = m.as_flat_dict([c[0] for c in class_info])
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(flat.keys()))
            csv_writer.writeheader()
        csv_writer.writerow(flat)
        csv_file.flush()

        logger.info(
            "step=%d scribbles=%d ink=%.0fpx  mIoU=%.4f  cov=%.4f  prec=%.4f  "
            "eff_scr=%.6f  eff_ink=%.8f",
            step, n_scr, total_ink, m.miou, coverage, ann_prec,
            m.miou_per_scribble, m.miou_per_1kpx,
        )
        return m

    # ── Базовые метрики до новых штрихов ───────────────────────────────────────
    checkpoint(start_step)
    write_dynamic_metrics(
        step=start_step,
        pred=pred_upd.pred_np,
        n_scr=len(ink_list),
        total_ink_now=total_ink,
        explicit_mask=pred_upd.explicit_np,
    )
    no_progress_steps = 0

    # ── Главный цикл штрихов ────────────────────────────────────────────────────
    for step_idx in range(1, int(args.scribbles) + 1):
        sid = start_step + step_idx
        scribble_id = next_scribble_id + step_idx - 1
        prev_pred = pred_upd.pred_np
        prev_annotated_px = int(pred_upd.explicit_np.sum())
        prev_correct_px = int((pred_upd.explicit_np & (prev_pred == gt)).sum())

        # --- Генерируем штрих ---
        try:
            gt_id, pts01 = generator.make_scribble(
                pred_upd.pred_np,
                used_mask,
                annotated_mask=pred_upd.explicit_np,
                class_scribble_counts=per_class_n_scr,
            )
        except StopIteration as e:
            logger.info("Early stop at step %d: %s", sid, e)
            break
        except RuntimeError as e:
            logger.warning("Cannot generate scribble at step %d: %s", sid, e)
            break

        # --- Стоимость штриха ---
        length = scribble_length_px(pts01, H, W)
        ink_list.append(length)
        total_ink += length
        per_class_n_scr[gt_id] += 1
        per_class_ink[gt_id] += length

        # --- Пометить used_mask ---
        if args.no_overlap:
            mark_polyline_used_norm(used_mask, pts01, H, W, width=3, pad=2)

        code = gt_id + 1  # code 1-indexed
        scrib = structs.Scribble(
            id=scribble_id,
            points=pts01,
            params=structs.ScribbleParams(radius=1, code=int(code)),
        )

        # --- Создаём суперпиксели вокруг штриха (если не precomputed) ---
        if not use_precomputed:
            algo._create_superpixel_for_scribble(scrib, sp_method)

        # --- Добавляем штрих (триггерит _update_annotations) ---
        algo.add_scribble(scrib)

        # --- Распространение аннотации (BFS по соседям) ---
        run_propagation(
            algo,
            sp_method,
            scrib,
            float(args.sensitivity),
            gt_mask=gt,
            pred_mask=prev_pred,
            class_scribble_counts=per_class_n_scr,
        )

        # --- Инкрементальная перерисовка pred_mask для генератора ---
        ann_obj = algo._annotations.get(sp_method)
        if ann_obj:
            for anno in ann_obj.annotations:
                b = np.asarray(anno.border, np.float32)
                poly = [
                    (float(b[i, 0] * W), float(b[i, 1] * H))
                    for i in range(len(b))
                ]
                pred_class = max(0, min(int(anno.code) - 1, num_classes - 1))
                pred_upd.paint_polygon(poly, pred_class)

        cur_pred = pred_upd.pred_np
        cur_annotated_px = int(pred_upd.explicit_np.sum())
        cur_correct_px = int((pred_upd.explicit_np & (cur_pred == gt)).sum())
        write_dynamic_metrics(
            step=sid,
            pred=cur_pred,
            n_scr=len(ink_list),
            total_ink_now=total_ink,
            explicit_mask=pred_upd.explicit_np,
        )
        made_progress = (
            cur_annotated_px > prev_annotated_px
            or cur_correct_px > prev_correct_px
        )
        generator.report_last_result(made_progress)
        if made_progress:
            no_progress_steps = 0
        else:
            no_progress_steps += 1
            logger.info(
                "No progress after step %d; increasing randomness for the next scribble (streak=%d)",
                sid,
                no_progress_steps,
            )
            if no_progress_steps >= int(args.max_no_progress):
                logger.info(
                    "Early stop at step %d: no progress for %d consecutive scribbles.",
                    sid,
                    no_progress_steps,
                )
                break

        # --- Checkpoint: метрики + PNG + JSON ---
        is_checkpoint = (sid % int(args.save_every) == 0) or (sid == args.scribbles)
        if is_checkpoint:
            m = checkpoint(sid)

            out_png = out_dir / f"frame_{sid:06d}.png"
            structs.render_annotation_snapshot(
                img=img,
                algo=algo,
                sp_method=sp_method,
                class_info=class_info,
                out_png=out_png,
                draw_borders=not args.no_borders,
                draw_annos=not args.no_annos,
                draw_scribbles=not args.no_scribbles,
            )

            out_json = out_dir / f"state_{sid:06d}.json"
            try:
                algo.serialize(str(out_json))
            except Exception as ex:
                logger.warning("Cannot save spanno: %s", ex)

    # --- Финальный checkpoint если не попал ---
    last_step = len(ink_list)
    if not metrics_history or metrics_history[-1].step != last_step:
        checkpoint(last_step)

    csv_file.close()

    # --- График кривых обучения ---
    plot_metrics(metrics_history, class_info, out_dir / "learning_curves.png")

    dynamic_csv_file.close()
    return metrics_history


# ─────────────────────────────────────────────────────────────────────────────────
#  Батч: сводная таблица по всем изображениям
# ─────────────────────────────────────────────────────────────────────────────────

def save_batch_summary(
    summaries: Dict[str, List[StepMetrics]],
    class_info: List[Tuple[str, str]],
    out_path: Path,
) -> None:
    """Сохраняет CSV с финальными метриками для каждого изображения."""
    if not summaries:
        return
    rows = []
    for img_name, history in summaries.items():
        if not history:
            continue
        m = history[-1]
        row = {"image": img_name}
        row.update(m.as_flat_dict([c[0] for c in class_info]))
        rows.append(row)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    # Средние
    numeric_keys = [k for k in fieldnames if k != "image"]
    means = {}
    for k in numeric_keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and not math.isnan(float(r[k]))]
        means[k] = round(float(np.mean(vals)), 6) if vals else float("nan")
    means["image"] = "MEAN"
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writerow(means)


# ─────────────────────────────────────────────────────────────────────────────────
#  Логгер
# ─────────────────────────────────────────────────────────────────────────────────

def setup_logger(out_dir: Path, name: str = "eval") -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(str(out_dir / "run.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# Shared benchmark primitives live in interactive_benchmark so the legacy
# pipeline and the new benchmark layer reuse the same metric/oracle behavior.
from interactive_benchmark.legacy_oracle import (  # noqa: E402
    LargestBadRegionGenerator as _SharedLargestBadRegionGenerator,
    parse_region_selection_cycle as _shared_parse_region_selection_cycle,
)
from interactive_benchmark.shared import (  # noqa: E402
    StepMetrics as _SharedStepMetrics,
    compute_ious as _shared_compute_ious,
    plot_metrics as _shared_plot_metrics,
    save_batch_summary as _shared_save_batch_summary,
)

StepMetrics = _SharedStepMetrics
compute_ious = _shared_compute_ious
LargestBadRegionGenerator = _SharedLargestBadRegionGenerator
parse_region_selection_cycle = _shared_parse_region_selection_cycle
plot_metrics = _shared_plot_metrics
save_batch_summary = _shared_save_batch_summary


# ─────────────────────────────────────────────────────────────────────────────────
#  Аргументы
# ─────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    hint_text = (
        "Hint: before interactive evaluation it is preferable to pre-segment the image "
        "into superpixels (for example, save and pass --spanno). This reduces strange "
        "merging of different regions into the same superpixel and makes scribble-based "
        "evaluation more stable."
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog=hint_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Вход ---
    grp_in = ap.add_argument_group("Input")
    grp_in.add_argument("--image", help="Путь к одному RGB-изображению")
    grp_in.add_argument("--mask", help="Путь к GT-маске (одно изображение)")
    grp_in.add_argument("--img_dir", help="Директория с изображениями (пакетный режим)")
    grp_in.add_argument("--mask_dir", help="Директория с масками (пакетный режим)")
    grp_in.add_argument("--spanno", default=None,
                        help="Заранее вычисленный .spanno.json[.gz] (одиночный режим). "
                             "Рекомендуется: сначала автоматически сегментировать изображение "
                             "суперпикселями, чтобы избежать странных объединений разных регионов.")
    grp_in.add_argument("--downscale", type=float, default=1.0,
                        help="Коэффициент уменьшения изображения для algo (default 1.0)")

    # --- Выход ---
    ap.add_argument("--out", required=True, help="Директория для результатов")

    # --- Симуляция ---
    grp_sim = ap.add_argument_group("Simulation")
    grp_sim.add_argument("--scribbles", type=int, default=500, help="Максимум штрихов")
    grp_sim.add_argument("--save_every", type=int, default=50, help="Checkpoint каждые N штрихов")
    grp_sim.add_argument("--seed", type=int, default=0)
    grp_sim.add_argument("--margin", type=int, default=2, help="Отступ от границы GT (пиксели)")
    grp_sim.add_argument("--border_margin", type=int, default=3,
                         help="Минимальный отступ штриха от границы bad-region (пиксели)")
    grp_sim.add_argument("--no_overlap", action="store_true",
                         help="Запрет перекрытия новых штрихов с ранее нанесёнными")
    grp_sim.add_argument("--max_no_progress", type=int, default=12,
                         help="Ранний стоп после N штрихов подряд без прогресса")
    grp_sim.add_argument(
        "--region_selection_cycle",
        default="miou_gain,largest_error,unannotated",
        help=(
            "Comma-separated cycle of region selection modes for new scribbles. "
            "Supported: miou_gain, largest_error, unannotated."
        ),
    )

    # --- Метод суперпикселей ---
    grp_sp = ap.add_argument_group("Superpixel method")
    grp_sp.add_argument("--method", default="slic",
                        choices=structs.SUPPORTED_SUPERPIXEL_METHOD_CHOICES)
    grp_sp.add_argument("--method_config", default=None,
                        help="JSON string or path to JSON config for neural methods.")
    grp_sp.add_argument("--weights", default=None,
                        help="Checkpoint for neural methods (and optional alias for ssn).")
    # SLIC
    grp_sp.add_argument("--n_segments", type=int, default=3000)
    grp_sp.add_argument("--compactness", type=float, default=20.0)
    grp_sp.add_argument("--sigma", type=float, default=1.0)
    # Felzenszwalb
    grp_sp.add_argument("--scale", type=float, default=400.0)
    grp_sp.add_argument("--f_sigma", type=float, default=1.0)
    grp_sp.add_argument("--min_size", type=int, default=50)
    # Watershed
    grp_sp.add_argument("--ws_compactness", type=float, default=1e-4)
    grp_sp.add_argument("--ws_components", type=int, default=500)
    # SSN
    grp_sp.add_argument("--ssn_weights", default=None, help="Чекпоинт SSN (.pth)")
    grp_sp.add_argument("--ssn_nspix", type=int, default=100)
    grp_sp.add_argument("--ssn_fdim", type=int, default=20)
    grp_sp.add_argument("--ssn_niter", type=int, default=5)
    grp_sp.add_argument("--ssn_color_scale", type=float, default=0.26)
    grp_sp.add_argument("--ssn_pos_scale", type=float, default=2.5)

    # --- Распространение ---
    grp_prop = ap.add_argument_group("Propagation")
    grp_prop.add_argument("--sensitivity", type=float, default=1.8,
                          help="Чувствительность BFS-распространения (0 = выкл., default 1.8)")
    grp_prop.add_argument("--emb_weights", default=None,
                          help="Чекпоинт для эмбединг-пропагации (.pth). "
                               "Если задан, использует cosine-similarity вместо LAB")
    grp_prop.add_argument("--emb_threshold", type=float, default=0.988,
                          help="Порог косинусного сходства (default 0.988)")

    # --- Визуализация ---
    grp_viz = ap.add_argument_group("Visualization")
    grp_viz.add_argument("--no_borders", action="store_true")
    grp_viz.add_argument("--no_annos", action="store_true")
    grp_viz.add_argument("--no_scribbles", action="store_true")

    # --- Классы ---
    ap.add_argument("--num_classes", type=int, default=None,
                    help="Число классов (default — авто из DEFAULT_CLASS_INFO или GT уникальных)")

    return ap


# ─────────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────────

def main():
    ap = build_parser()
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    # ── Определяем class_info ────────────────────────────────────────────────────
    if args.num_classes is not None:
        nc = int(args.num_classes)
        if nc <= len(DEFAULT_CLASS_INFO):
            class_info = DEFAULT_CLASS_INFO[:nc]
        else:
            class_info = DEFAULT_CLASS_INFO + [
                (f"cls{i}", "#aaaaaa") for i in range(len(DEFAULT_CLASS_INFO), nc)
            ]
    else:
        class_info = DEFAULT_CLASS_INFO

    logger.info("Classes (%d): %s", len(class_info), [c[0] for c in class_info])

    # ── Метод суперпикселей ──────────────────────────────────────────────────────
    sp_method_proto = build_sp_method(args)
    logger.info("SP method: %s", sp_method_proto.short_string())

    if args.sensitivity > 0:
        logger.info("Propagation: sensitivity=%.2f", args.sensitivity)
        if args.emb_weights:
            logger.info("Embedding propagation: %s  threshold=%.2f",
                        args.emb_weights, args.emb_threshold)
    else:
        logger.info("Propagation: disabled (--sensitivity 0)")

    # ── Одиночный режим ──────────────────────────────────────────────────────────
    if args.image and args.mask:
        img = Image.open(args.image).convert("RGB")
        gt = load_mask_as_ids(args.mask)
        img, gt = ensure_same_size(img, gt)
        logger.info("Image: %s  size=%s  GT classes=%s",
                    args.image, img.size, sorted(np.unique(gt).tolist()))

        run_single_image(
            img=img, gt=gt,
            sp_method_proto=sp_method_proto,
            args=args,
            out_dir=out_dir,
            class_info=class_info,
            logger=logger,
            spanno_path=args.spanno,
        )
        logger.info("Done. Results: %s", out_dir)
        return

    # ── Пакетный режим ───────────────────────────────────────────────────────────
    if args.img_dir and args.mask_dir:
        pairs = discover_image_pairs(args.img_dir, args.mask_dir)
        if not pairs:
            logger.error("No image-mask pairs found in %s / %s", args.img_dir, args.mask_dir)
            sys.exit(1)
        logger.info("Found %d image-mask pairs", len(pairs))

        all_summaries: Dict[str, List[StepMetrics]] = {}
        for img_path, mask_path in pairs:
            img = Image.open(img_path).convert("RGB")
            gt = load_mask_as_ids(str(mask_path))
            img, gt = ensure_same_size(img, gt)

            img_out = out_dir / img_path.stem
            img_out.mkdir(parents=True, exist_ok=True)
            logger.info("=== Processing %s ===", img_path.name)

            history = run_single_image(
                img=img, gt=gt,
                sp_method_proto=sp_method_proto,
                args=args,
                out_dir=img_out,
                class_info=class_info,
                logger=logger,
            )
            all_summaries[img_path.stem] = history

        save_batch_summary(all_summaries, class_info, out_dir / "batch_summary.csv")
        logger.info("Batch done. Summary: %s", out_dir / "batch_summary.csv")
        return

    ap.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
