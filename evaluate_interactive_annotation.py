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
    --sensitivity 1.0..5.0 — включить BFS-распространение по соседним суперпикселям
    --emb_weights path.pth — использовать cosine-similarity по эмбедингам вместо LAB-цвета
    --emb_threshold 0.75   — порог косинусного сходства (default 0.99)
"""

import argparse
import csv
import json
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label

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
#  Константы / цвета (13-классовая задача минералов; заменить при необходимости)
# ─────────────────────────────────────────────────────────────────────────────────
DEFAULT_CLASS_INFO = [
    ("BG",        "#1c1818"),
    ("Ccp",       "#ff0000"),
    ("Gl",        "#cbff00"),
    ("Mag",       "#00ff66"),
    ("Brt",       "#0065ff"),
    ("Po",        "#cc00ff"),
    ("Pn",        "#dbff4c"),
    ("Sph",       "#4cff93"),
    ("Apy",       "#4c93ff"),
    ("Hem",       "#db4cff"),
    ("Kvl",       "#eaff99"),
    ("Py/Mrc",    "#ff4c4c"),
    ("Tnt/Ttr",   "#ff9999"),
]


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgba(h: str, alpha: int) -> Tuple[int, int, int, int]:
    r, g, b = _hex_to_rgb(h)
    return r, g, b, alpha


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
    pred: (H,W) int32, -1 = не размечено.
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
    """Хранит текущую pred-маску и эффективно перерисовывает аннотации."""

    def __init__(self, H: int, W: int):
        self.H, self.W = H, W
        self.pred_np = np.full((H, W), -1, dtype=np.int32)
        self._pil = Image.fromarray(self.pred_np, mode="I")

    def reset(self) -> None:
        self.pred_np[:] = -1
        self._pil = Image.fromarray(self.pred_np, mode="I")

    def paint_polygon(self, poly_xy: List[Tuple[float, float]], value: int) -> None:
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
        crop = self._pil.crop((x0, y0, x1, y1))
        ImageDraw.Draw(crop).polygon(
            [(float(x - x0), float(y - y0)) for x, y in poly_xy],
            fill=int(value),
        )
        self._pil.paste(crop, (x0, y0))
        self.pred_np[y0:y1, x0:x1] = np.array(crop, dtype=np.int32)

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
            self.paint_polygon(poly, pred_class)


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

    def __init__(self, gt_mask: np.ndarray, num_classes: int,
                 seed: int = 0, margin: int = 2, border_margin: int = 3,
                 no_overlap: bool = True, max_retries: int = 200):
        self.gt = gt_mask.astype(np.int32)
        self.H, self.W = gt_mask.shape
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)
        self.margin = max(0, margin)
        self.border_margin = max(0, border_margin)
        self.no_overlap = no_overlap
        self.max_retries = max_retries
        self._diag = 0.5 * math.sqrt(self.W ** 2 + self.H ** 2)
        self._stall_steps = 0
        self._recent_signatures = deque(maxlen=12)
        self._component_failures: Dict[Tuple[int, int, int, int, int, int], int] = {}
        self._class_failures: Dict[int, int] = {}
        self._last_selected_signature: Optional[Tuple[int, int, int, int, int, int]] = None
        self._last_selected_class: Optional[int] = None

        # внутренние маски каждого класса (с отступом margin)
        self._gt_inner: List[np.ndarray] = []
        for cid in range(num_classes):
            cls = (self.gt == cid)
            if self.margin > 0 and cls.any():
                inner = cls & (distance_transform_edt(cls) > self.margin)
            else:
                inner = cls.copy()
            self._gt_inner.append(inner)

    # -- helpers ---
    def _largest_component(self, bad: np.ndarray) -> Optional[np.ndarray]:
        lab, n = cc_label(bad)
        if n == 0:
            return None
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        k = int(np.argmax(counts))
        return (lab == k) if counts[k] > 0 else None

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

    def _build_allowed_mask(
        self,
        cid: int,
        comp: np.ndarray,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
    ) -> np.ndarray:
        bad_c = (self.gt == cid) & (pred_mask != cid)
        if self.border_margin > 0 and comp.any():
            comp_inner = comp & (distance_transform_edt(comp) > self.border_margin)
        else:
            comp_inner = comp

        gt_inner = self._gt_inner[cid]
        allowed = comp_inner & gt_inner & bad_c
        if self.no_overlap:
            allowed &= ~used_mask
        return allowed

    def _select_class_component(
        self,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
    ) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
        inter, union = self._class_inter_union(pred_mask)
        candidates = []

        for cid in range(self.num_classes):
            bad_c = (self.gt == cid) & (pred_mask != cid)
            if not bad_c.any():
                continue

            comp = self._largest_component(bad_c)
            if comp is None:
                continue

            allowed = self._build_allowed_mask(cid, comp, pred_mask, used_mask)
            if not allowed.any():
                continue

            union_c = max(1, int(union[cid]))
            comp_area = int(comp.sum())
            iou_c = float(inter[cid]) / float(union_c)
            gain_score = float(comp_area) / float(union_c)
            signature = self._component_signature(cid, comp)
            if signature is None:
                continue

            comp_fail = float(self._component_failures.get(signature, 0))
            class_fail = float(self._class_failures.get(int(cid), 0))
            recent_pen = 1.0 if signature in self._recent_signatures else 0.0

            # После бесполезных шагов уводим выбор от тех же регионов и классов.
            effective_gain = gain_score - comp_fail - 0.25 * class_fail - 0.05 * recent_pen
            score = (
                effective_gain,
                1.0 - iou_c - 0.05 * class_fail,
                comp_area,
                float(self.rng.random()),
            )
            candidates.append((score, int(cid), comp, allowed, signature))

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_n = 1
        if self._stall_steps > 0:
            top_n = min(len(candidates), 1 + min(3, self._stall_steps))

        if top_n > 1:
            rank_weights = np.linspace(top_n, 1, top_n, dtype=np.float64)
            rank_weights /= rank_weights.sum()
            pick = int(self.rng.choice(np.arange(top_n), p=rank_weights))
            chosen = candidates[pick]
        else:
            chosen = candidates[0]

        _, best_cid, best_comp, best_allowed, signature = chosen
        self._last_selected_signature = signature
        self._last_selected_class = int(best_cid)
        return best_cid, best_comp, best_allowed

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
    ) -> Tuple[int, np.ndarray]:
        """
        Returns:
            gt_id   — 0-indexed class id that the scribble annotates
            pts01   — (2,2) float32 array of normalised [0..1] endpoints
        Raises StopIteration when all pixels are correctly labelled.
        Raises RuntimeError if a scribble cannot be placed (relax --margin).
        """
        bad = (pred_mask != self.gt)
        if not bad.any():
            raise StopIteration("All pixels correctly labelled.")

        cid, comp, allowed = self._select_class_component(pred_mask, used_mask)
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

        ys, xs = np.where(allowed)
        for _ in range(self.max_retries):
            i = int(self.rng.integers(0, xs.size))
            x0, y0 = int(xs[i]), int(ys[i])
            theta = float(self.rng.uniform(0, 2 * math.pi))
            dx, dy = math.cos(theta), math.sin(theta)
            x1a, y1a = self._trace(x0, y0, +dx, +dy, allowed, used_mask)
            x1b, y1b = self._trace(x0, y0, -dx, -dy, allowed, used_mask)
            if (x1a, y1a) == (x1b, y1b):
                continue
            pts = np.array([[x1b / self.W, y1b / self.H],
                             [x1a / self.W, y1a / self.H]], dtype=np.float32)
            return cid, pts

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


# ─────────────────────────────────────────────────────────────────────────────────
#  Распространение штрихов (propagation)
# ─────────────────────────────────────────────────────────────────────────────────

def run_propagation(
    algo: "structs.SuperPixelAnnotationAlgo",
    sp_method: "structs.SuperPixelMethod",
    scrib: "structs.Scribble",
    sensitivity: float,
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
    for anno in annos_obj.annotations:
        if scrib.id in (anno.parent_scribble or []):
            sp_id = anno.parent_superpixel
            if sp_id not in seen:
                seen.add(sp_id)
                idx = sp_id_to_idx.get(sp_id)
                if idx is not None:
                    algo.use_sensitivity_for_region(idx, float(sensitivity), scrib)


# ─────────────────────────────────────────────────────────────────────────────────
#  Построение метода суперпикселей из аргументов
# ─────────────────────────────────────────────────────────────────────────────────

def build_sp_method(args) -> "structs.SuperPixelMethod":
    m = args.method.lower()
    if m == "slic":
        return structs.SLICSuperpixel(
            n_clusters=int(args.n_segments),
            compactness=float(args.compactness),
            sigma=float(args.sigma),
        )
    if m in ("felzenszwalb", "fwb"):
        return structs.FelzenszwalbSuperpixel(
            min_size=int(args.min_size),
            sigma=float(args.f_sigma),
            scale=float(args.scale),
        )
    if m in ("watershed", "ws"):
        return structs.WatershedSuperpixel(
            compactness=float(args.ws_compactness),
            n_components=int(args.ws_components),
        )
    if m == "ssn":
        if not args.ssn_weights:
            raise ValueError("--ssn_weights required for method=ssn")
        return structs.SSNSuperpixel(
            weight_path=args.ssn_weights,
            nspix=int(args.ssn_nspix),
            fdim=int(args.ssn_fdim),
            niter=int(args.ssn_niter),
            color_scale=float(args.ssn_color_scale),
            pos_scale=float(args.ssn_pos_scale),
        )
    raise ValueError(f"Unknown method: {args.method!r}")


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
                draw.polygon(poly, fill=_hex_to_rgba(class_info[gt_id][1], anno_alpha))

    if draw_borders:
        for sp in algo.superpixels.get(sp_method, []):
            b = np.asarray(sp.border, np.float32)
            poly = [(float(b[i, 0] * W), float(b[i, 1] * H)) for i in range(len(b))]
            draw.polygon(poly, outline=(255, 255, 0, 255))

    composite = Image.alpha_composite(img.convert("RGBA"), overlay)

    if draw_scribbles:
        d2 = ImageDraw.Draw(composite)
        for s in getattr(algo, "_scribbles", []):
            pts = np.asarray(s.points, np.float32)
            line_pts = [(float(pts[i, 0] * W), float(pts[i, 1] * H)) for i in range(len(pts))]
            code = int(s.params.code)
            gt_id = max(0, min(code - 1, len(class_info) - 1))
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

    # precomputed spanno или добавляем метод свежий
    use_precomputed = False
    if spanno_path and Path(spanno_path).exists():
        algo.deserialize(spanno_path)
        use_precomputed = True
        # найти совпадающий метод в загруженном файле
        want = sp_method_proto.short_string()
        sp_method = next(
            (m for m in algo.superpixel_methods if m.short_string() == want),
            None,
        )
        if sp_method is None:
            sp_method = algo.superpixel_methods[0] if algo.superpixel_methods else sp_method_proto
        logger.info("Loaded precomputed spanno: %s  active=%s", spanno_path, sp_method.short_string())
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
    )

    # Накопители стоимости взаимодействия
    total_ink: float = 0.0
    ink_list: List[float] = []          # длина каждого штриха
    per_class_n_scr: List[int] = [0] * num_classes
    per_class_ink: List[float] = [0.0] * num_classes

    metrics_history: List[StepMetrics] = []

    # ── CSV ─────────────────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer: Optional[csv.DictWriter] = None   # инициализируем на первой записи

    def checkpoint(step: int) -> StepMetrics:
        nonlocal csv_writer
        # Пересчитываем pred_mask полностью (включает propagated аннотации)
        pred_upd.repaint_all(algo, sp_method, num_classes)
        pred = pred_upd.pred_np

        annotated_mask = (pred >= 0)
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

    # ── Шаг 0: базовые метрики до любых штрихов ────────────────────────────────
    checkpoint(0)
    no_progress_steps = 0

    # ── Главный цикл штрихов ────────────────────────────────────────────────────
    for sid in range(1, int(args.scribbles) + 1):
        prev_pred = pred_upd.pred_np
        prev_annotated_px = int((prev_pred >= 0).sum())
        prev_correct_px = int(((prev_pred >= 0) & (prev_pred == gt)).sum())

        # --- Генерируем штрих ---
        try:
            gt_id, pts01 = generator.make_scribble(pred_upd.pred_np, used_mask)
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
            xA = int(round(pts01[0, 0] * W)); yA = int(round(pts01[0, 1] * H))
            xB = int(round(pts01[1, 0] * W)); yB = int(round(pts01[1, 1] * H))
            structs.mark_line_used(used_mask, xA, yA, xB, yB, width=3, pad=2)

        code = gt_id + 1  # code 1-indexed
        scrib = structs.Scribble(
            id=sid,
            points=pts01,
            params=structs.ScribbleParams(radius=1, code=int(code)),
        )

        # --- Создаём суперпиксели вокруг штриха (если не precomputed) ---
        if not use_precomputed:
            algo._create_superpixel_for_scribble(scrib, sp_method)

        # --- Добавляем штрих (триггерит _update_annotations) ---
        algo.add_scribble(scrib)

        # --- Распространение аннотации (BFS по соседям) ---
        run_propagation(algo, sp_method, scrib, float(args.sensitivity))

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
        cur_annotated_px = int((cur_pred >= 0).sum())
        cur_correct_px = int(((cur_pred >= 0) & (cur_pred == gt)).sum())
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

    # --- Метод суперпикселей ---
    grp_sp = ap.add_argument_group("Superpixel method")
    grp_sp.add_argument("--method", default="slic",
                        choices=["slic", "felzenszwalb", "fwb", "watershed", "ws", "ssn"])
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
    grp_prop.add_argument("--sensitivity", type=float, default=0.0,
                          help="Чувствительность BFS-распространения (0 = выкл.)")
    grp_prop.add_argument("--emb_weights", default=None,
                          help="Чекпоинт для эмбединг-пропагации (.pth). "
                               "Если задан, использует cosine-similarity вместо LAB")
    grp_prop.add_argument("--emb_threshold", type=float, default=0.99,
                          help="Порог косинусного сходства (default 0.99)")

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
