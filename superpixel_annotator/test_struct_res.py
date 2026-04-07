#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_superpixel_annotator_with_scribble_gen.py

Тестирование SuperPixelAnnotationAlgo через генератор скриблов по GT-маске.
Скриблы ставятся ТОЛЬКО в пиксели, где текущая предразметка (pred_mask)
не совпадает с GT (или pred=-1, т.е. не размечено).

Каждые N (по умолчанию 200) скриблов:
- сохраняет PNG-визуализации
- считает и логирует IoU (per-class + mIoU) и пишет в CSV.

Пример:
python test_superpixel_annotator_with_scribble_gen.py \
  --image /path/img.png \
  --mask  /path/mask.png \
  --out   /path/out_dir \
  --method slic --n_segments 800 --compactness 15 --sigma 1.0 \
  --scribbles 2000 --save_every 200 --seed 0
"""

import argparse
import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

import structs  # твой /mnt/data/structs.py


# ---- Классы (0..12 в GT), имена/цвета для отрисовки ----
# Важно: в tk_service "code" маркера начинается с 1, поэтому code = gt_id + 1.
CLASS_INFO = [
    ("bg", "#000000"),
    ("ccp", "#ffa500"),
    ("gl", "#9acd32"),
    ("mag", "#ff4500"),
    ("br", "#00bfff"),
    ("po", "#a9a9a9"),
    ("py", "#2f4f4f"),
    ("pn", "#ffff00"),
    ("sh", "#ee82ee"),
    ("apy", "#556b2f"),
    ("gmt", "#a0522d"),
    ("tnt", "#483d8b"),
    ("cv", "#008000"),
    ("mrc", "#00008b"),
    ("au", "#8b008b"),
]

GTID_TO_CODE = {gt_id: gt_id + 1 for gt_id in range(len(CLASS_INFO))}


def hex_to_rgba(hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def load_mask_as_ids(mask_path: str) -> np.ndarray:
    """
    Загружаем разметку как индексы классов. Поддержка:
      - L (grayscale) — значения пикселей и есть id
      - RGB/палитра — берём 1 канал (обычно достаточно для индексов)
    """
    m = Image.open(mask_path)
    if m.mode in ("P", "L", "I;16", "I"):
        arr = np.array(m)
    else:
        arr = np.array(m.convert("RGB"))[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after conversion, got shape={arr.shape}")
    return arr.astype(np.int32)


def ensure_same_size(img: Image.Image, mask_ids: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
    if (img.height, img.width) == mask_ids.shape[:2]:
        return img, mask_ids
    # если размеры не совпали — приводим маску nearest-neighbor
    m = Image.fromarray(mask_ids.astype(np.uint8), mode="L")
    m = m.resize((img.width, img.height), resample=Image.NEAREST)
    return img, np.array(m).astype(np.int32)


# -------------------- IoU --------------------

def compute_ious(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[float, List[float]]:
    """
    pred, gt: [H,W] int32, pred может содержать -1 (не размечено).
    IoU считаем для классов 0..num_classes-1, pred=-1 просто никуда не попадает.
    """
    ious: List[float] = []
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        inter = int(np.logical_and(pred_c, gt_c).sum())
        union = int(np.logical_or(pred_c, gt_c).sum())
        if union == 0:
            iou = float("nan")  # класс отсутствует и в gt и в pred
        else:
            iou = inter / union
        ious.append(iou)

    # mIoU: среднее по тем классам, у которых union>0
    valid = [v for v in ious if not np.isnan(v)]
    miou = float(np.mean(valid)) if valid else float("nan")
    return miou, ious


# -------------------- Pred mask updater (ускорение) --------------------

class PredMaskUpdater:
    """
    Держит временную маску предразметки pred_mask:
      - numpy pred_np: [H,W] int32, init=-1
      - PIL pred_pil: mode 'I' (32-bit signed int not guaranteed, but works for small ints)
    Обновляет только bbox полигона, чтобы не пересобирать массив целиком.
    """

    def __init__(self, h: int, w: int):
        self.H, self.W = h, w
        self.pred_np = np.full((h, w), -1, dtype=np.int32)
        self.pred_pil = Image.fromarray(self.pred_np.astype(np.int32), mode="I")

    @staticmethod
    def _poly_bbox(poly_xy: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in poly_xy]
        ys = [p[1] for p in poly_xy]
        x0 = int(max(0, np.floor(min(xs))))
        y0 = int(max(0, np.floor(min(ys))))
        x1 = int(min(np.ceil(max(xs)) + 1, 10**9))
        y1 = int(min(np.ceil(max(ys)) + 1, 10**9))
        return x0, y0, x1, y1

    def paint_polygon(self, poly_xy: List[Tuple[float, float]], value: int) -> None:
        """
        poly_xy: координаты в пикселях (float ok)
        value: класс 0..C-1
        """
        if len(poly_xy) < 3:
            return

        x0, y0, x1, y1 = self._poly_bbox(poly_xy)
        x0 = max(0, min(x0, self.W))
        x1 = max(0, min(x1, self.W))
        y0 = max(0, min(y0, self.H))
        y1 = max(0, min(y1, self.H))
        if x1 <= x0 or y1 <= y0:
            return

        # работаем по кропу, чтобы обновить только bbox
        crop = self.pred_pil.crop((x0, y0, x1, y1))
        dc = ImageDraw.Draw(crop)

        # смещаем полигон в координаты кропа
        shifted = [(float(x - x0), float(y - y0)) for (x, y) in poly_xy]
        dc.polygon(shifted, fill=int(value))

        # вставляем обратно и обновляем numpy только в bbox
        self.pred_pil.paste(crop, (x0, y0))
        self.pred_np[y0:y1, x0:x1] = np.array(crop, dtype=np.int32)

def mark_line_used(used_mask: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                   width: int = 3, pad: int = 2) -> None:
    H, W = used_mask.shape
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n).round().astype(np.int32)
    ys = np.linspace(y0, y1, n).round().astype(np.int32)

    r = max(0, width // 2) + max(0, pad)

    for x, y in zip(xs, ys):
        x0b = max(0, x - r); x1b = min(W, x + r + 1)
        y0b = max(0, y - r); y1b = min(H, y + r + 1)
        used_mask[y0b:y1b, x0b:x1b] = True


# -------------------- Scribble generator (только в ошибках) --------------------

@dataclass
class ScribbleGenCfg:
    max_steps: int = 60
    step_px: int = 6
    max_retries: int = 400
    sample_by_area: bool = True


class ScribbleGeneratorOnErrors:
    """
    Генерирует скриблы строго из областей, где pred_mask != gt_mask (или pred==-1).
    При этом лейбл скрибла берётся из GT класса в точке старта.
    """

    def __init__(self, gt_mask: np.ndarray, seed: int, cfg: ScribbleGenCfg):
        self.gt = gt_mask
        self.H, self.W = gt_mask.shape
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg

        # для ускорения — храним все пиксели каждого класса GT
        self.class_pixels: Dict[int, np.ndarray] = {}
        counts = []
        for cid in range(len(CLASS_INFO)):
            ys, xs = np.where(self.gt == cid)
            coords = np.stack([xs, ys], axis=1) if len(xs) else np.zeros((0, 2), dtype=np.int32)
            self.class_pixels[cid] = coords
            counts.append(coords.shape[0])
        self.counts = np.array(counts, dtype=np.int64)

        if self.cfg.sample_by_area and self.counts.sum() > 0:
            self.p_class = self.counts / self.counts.sum()
        else:
            self.p_class = np.ones(len(CLASS_INFO), dtype=np.float64) / len(CLASS_INFO)

    def _pick_class(self) -> int:
        return int(self.rng.choice(len(CLASS_INFO), p=self.p_class))

    def _pick_start_in_errors(self, cid: int, bad_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        coords = self.class_pixels[cid]
        if coords.shape[0] == 0:
            return None

        # несколько быстрых попыток наугад (без построения списка всех bad координат)
        for _ in range(128):
            x, y = coords[self.rng.integers(0, coords.shape[0])]
            if bad_mask[int(y), int(x)]:
                return int(x), int(y)

        # если не нашли — значит плохих пикселей в этом классе мало/нет
        # fallback: выборка подмассива и проверка
        k = min(4096, coords.shape[0])
        idx = self.rng.integers(0, coords.shape[0], size=k)
        sub = coords[idx]
        ok = bad_mask[sub[:, 1], sub[:, 0]]
        if np.any(ok):
            x, y = sub[np.argmax(ok)]
            return int(x), int(y)
        return None

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        dirs = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]
        random.shuffle(dirs)
        res = []
        for dx, dy in dirs:
            nx = x + dx * self.cfg.step_px
            ny = y + dy * self.cfg.step_px
            if 0 <= nx < self.W and 0 <= ny < self.H:
                res.append((nx, ny))
        return res

    def make_scribble(self, pred_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        pred_mask: [H,W] int32, -1 allowed.
        returns: (gt_class_id, points01 [N,2])
        """
        bad = (pred_mask != self.gt)  # pred=-1 тоже "bad", потому что != gt

        # если всё идеально — больше нечего делать
        if not np.any(bad):
            raise StopIteration("No incorrect pixels left (pred matches GT everywhere).")

        for _ in range(self.cfg.max_retries):
            cid = self._pick_class()
            start = self._pick_start_in_errors(cid, bad)
            if start is None:
                continue

            x, y = start
            pts = [(x, y)]

            for _step in range(self.cfg.max_steps):
                # продолжаем только внутри GT-класса
                cands = [p for p in self._neighbors(x, y) if self.gt[p[1], p[0]] == cid]
                if not cands:
                    break
                x, y = cands[self.rng.integers(0, len(cands))]
                if pts[-1] != (x, y):
                    pts.append((x, y))

            if len(pts) >= 2:
                pts = np.array(pts, dtype=np.float32)
                pts01 = np.empty_like(pts, dtype=np.float32)
                pts01[:, 0] = pts[:, 0] / float(self.W)
                pts01[:, 1] = pts[:, 1] / float(self.H)
                return cid, pts01

        raise RuntimeError("Failed to generate a valid error-focused scribble after many retries.")


@dataclass
class ScribbleGenCfg:
    max_retries: int = 400
    strict_bad: bool = True     # весь штрих только по (pred!=gt)
    margin: int = 2             # эрозия класса
    no_overlap: bool = True     # запрещать пересечение с used_mask



class ScribbleGeneratorOnErrors:
    """
    Прямой отрезок (2 точки) внутри одного GT-класса.
    Дополнительно: всё строится внутри внутренней области класса (erosion на margin px),
    чтобы штрих не залезал на границу.
    """

    def __init__(self, gt_mask: np.ndarray, seed: int, cfg: ScribbleGenCfg):
        self.gt = gt_mask
        self.H, self.W = gt_mask.shape
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg

        self.L = 0.5 * float(np.sqrt(self.W * self.W + self.H * self.H))
        self.num_classes = len(CLASS_INFO)

        # inner masks per class: distance-to-boundary >= margin
        self.inner_masks: List[np.ndarray] = []
        self.inner_coords: List[np.ndarray] = []
        counts = []

        margin = int(max(0, cfg.margin))
        for cid in range(self.num_classes):
            cls = (self.gt == cid)

            if margin <= 0:
                inner = cls
            else:
                # distance inside class to nearest boundary (0 at boundary)
                # edt works on True/False: distance to nearest False. We want inside class => edt(cls)
                dist = distance_transform_edt(cls)
                inner = cls & (dist > margin)  # строго “глубже” чем margin

            self.inner_masks.append(inner)

            ys, xs = np.where(inner)
            coords = np.stack([xs, ys], axis=1) if len(xs) else np.zeros((0, 2), dtype=np.int32)
            self.inner_coords.append(coords)
            counts.append(coords.shape[0])

        counts = np.array(counts, dtype=np.int64)
        self.counts = counts
        self.p_class = (counts / counts.sum()) if counts.sum() > 0 else np.ones(self.num_classes) / self.num_classes

    def _pick_class(self) -> int:
        return int(self.rng.choice(self.num_classes, p=self.p_class))

    def _pick_start_in_errors(self, cid: int, allowed_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        coords = self.inner_coords[cid]
        if coords.shape[0] == 0:
            return None

        for _ in range(128):
            x, y = coords[self.rng.integers(0, coords.shape[0])]
            if allowed_mask[int(y), int(x)]:
                return int(x), int(y)

        k = min(4096, coords.shape[0])
        idx = self.rng.integers(0, coords.shape[0], size=k)
        sub = coords[idx]
        ok = allowed_mask[sub[:, 1], sub[:, 0]]
        if np.any(ok):
            x, y = sub[np.argmax(ok)]
            return int(x), int(y)

        return None


    def _trace_until_boundary(
        self,
        x0: int,
        y0: int,
        dx: float,
        dy: float,
        cid: int,
        allowed_mask: np.ndarray,
        used_mask: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Идём шагом 1px по направлению.
        Разрешены пиксели только там, где allowed_mask==True.
        Если cfg.no_overlap=True — запрещаем used_mask==True.
        """
        x = float(x0)
        y = float(y0)
        last_x, last_y = x0, y0

        for _ in range(int(self.L)):
            x += dx
            y += dy
            xi = int(round(x))
            yi = int(round(y))

            if xi < 0 or xi >= self.W or yi < 0 or yi >= self.H:
                break
            if not allowed_mask[yi, xi]:
                break
            if self.cfg.no_overlap and used_mask[yi, xi]:
                break

            last_x, last_y = xi, yi

        return last_x, last_y


    def make_scribble(self, pred_mask: np.ndarray, used_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        bad = (pred_mask != self.gt)
        if not np.any(bad):
            raise StopIteration("No incorrect pixels left (pred matches GT everywhere).")

        # used_mask может быть None
        if used_mask is None:
            used_mask = np.zeros_like(bad, dtype=bool)

        for _ in range(self.cfg.max_retries):
            cid = self._pick_class()

            # allowed: внутри класса (inner) AND плохие пиксели AND не занято предыдущими штрихами
            allowed = self.inner_masks[cid] & bad
            if self.cfg.no_overlap:
                allowed = allowed & (~used_mask)

            if not np.any(allowed):
                continue

            start = self._pick_start_in_errors(cid, allowed)
            if start is None:
                continue

            x0, y0 = start

            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            dx = float(np.cos(theta))
            dy = float(np.sin(theta))

            x1a, y1a = self._trace_until_boundary(x0, y0, +dx, +dy, cid, allowed, used_mask)
            x1b, y1b = self._trace_until_boundary(x0, y0, -dx, -dy, cid, allowed, used_mask)

            if (x1a, y1a) == (x1b, y1b):
                continue

            pts = np.array([[x1b, y1b], [x1a, y1a]], dtype=np.float32)
            pts01 = np.empty_like(pts, dtype=np.float32)
            pts01[:, 0] = pts[:, 0] / float(self.W)
            pts01[:, 1] = pts[:, 1] / float(self.H)
            return cid, pts01

        raise RuntimeError("Failed to generate a valid straight-line scribble after many retries.")



# -------------------- Superpixel method selection --------------------

def build_superpixel_method(args) -> structs.SuperPixelMethod:
    m = args.method.lower()
    if m == "slic":
        return structs.SLICSuperpixel(
            n_clusters=int(args.n_segments),
            compactness=float(args.compactness),
            sigma=float(args.sigma),
        )
    if m in ("fwb", "felzenszwalb"):
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
    raise ValueError(f"Unknown method: {args.method}")


# -------------------- Rendering --------------------

def render_snapshot(
    base_img: Image.Image,
    algo: "structs.SuperPixelAnnotationAlgo",
    sp_method: "structs.SuperPixelMethod",
    out_png: str,
    draw_borders: bool = True,
    draw_annos: bool = True,
    draw_scribbles: bool = True,
    anno_alpha: int = 110,
) -> None:
    img = base_img.convert("RGB").copy()
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # аннотации
    if draw_annos:
        annos_obj = algo._annotations.get(sp_method, None)
        if annos_obj is not None:
            for sp in annos_obj.annotations:
                border = np.array(sp.border, dtype=np.float32).copy()
                border[:, 0] *= W
                border[:, 1] *= H
                poly = [(float(x), float(y)) for x, y in border]
                code = int(sp.code)
                gt_id = max(0, min(code - 1, len(CLASS_INFO) - 1))
                color = hex_to_rgba(CLASS_INFO[gt_id][1], alpha=anno_alpha)
                draw.polygon(poly, fill=color)

    # границы всех суперпикселей
    if draw_borders:
        sps = algo.superpixels.get(sp_method, None)
        if sps is not None:
            for sp in sps:
                border = np.array(sp.border, dtype=np.float32).copy()
                border[:, 0] *= W
                border[:, 1] *= H
                poly = [(float(x), float(y)) for x, y in border]
                draw.polygon(poly, outline=(255, 255, 0, 255))

    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)

    # скриблы поверх
    if draw_scribbles:
        d2 = ImageDraw.Draw(img_rgba)
        for s in getattr(algo, "_scribbles", []):
            pts01 = np.array(s.points, dtype=np.float32)
            pts = [(float(pts01[i, 0] * W), float(pts01[i, 1] * H)) for i in range(pts01.shape[0])]
            code = int(s.params.code)
            gt_id = max(0, min(code - 1, len(CLASS_INFO) - 1))
            rgb = hex_to_rgba(CLASS_INFO[gt_id][1], 255)
            d2.line(pts, fill=rgb, width=5)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img_rgba.convert("RGB").save(out_png, quality=95)


# -------------------- Main --------------------

def setup_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("sp_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(out_dir / "run.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB изображение")
    ap.add_argument("--mask", required=True, help="GT маска (индексы классов в пикселях)")
    ap.add_argument("--out", required=True, help="папка для результатов")

    ap.add_argument("--scribbles", type=int, default=20000, help="сколько скриблов сгенерировать")
    ap.add_argument("--save_every", type=int, default=200, help="шаг сохранения визуализаций")
    ap.add_argument("--seed", type=int, default=0)
    
    ap.add_argument("--sensitivity", type=float, default=None,
                help="Чувствительность")
    ap.add_argument("--margin", type=int, default=2,
                help="Отступ от границы класса (в пикселях). "
                     "Штрихи генерируются внутри класса, эрозированного на margin.")

    ap.add_argument("--method", default="slic", choices=["slic", "felzenszwalb", "fwb", "watershed", "ws"])

    ap.add_argument("--no_overlap", action="store_true",
                help="Запретить пересечение новых штрихов с предыдущими.")
    ap.add_argument("--scribble_width", type=int, default=3,
                    help="Толщина линии (в пикселях) для маски занятых пикселей.")
    ap.add_argument("--overlap_pad", type=int, default=2,
                    help="Доп. паддинг вокруг линии (в пикселях), тоже считаем занятым.")

    # SLIC
    ap.add_argument("--n_segments", type=int, default=5000)
    ap.add_argument("--compactness", type=float, default=20.0)
    ap.add_argument("--sigma", type=float, default=1.0)

    # Felzenszwalb
    ap.add_argument("--scale", type=float, default=400.0)
    ap.add_argument("--f_sigma", type=float, default=1.0)
    ap.add_argument("--min_size", type=int, default=50)

    # Watershed
    ap.add_argument("--ws_compactness", type=float, default=1e-4)
    ap.add_argument("--ws_components", type=int, default=500)

    # Scribble generator params
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--step_px", type=int, default=6)
    ap.add_argument("--uniform_classes", action="store_true", help="выбирать классы равновероятно")

    # viz switches
    ap.add_argument("--no_borders", action="store_true")
    ap.add_argument("--no_annos", action="store_true")
    ap.add_argument("--no_scribbles", action="store_true")
    
    ap.add_argument("--spanno", default=None,
                help="Путь к заранее сгенерированному spanno (.json или .json.gz). "
                     "Если задан, суперпиксели берём из файла и не пересчитываем.")


    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    img = Image.open(args.image).convert("RGB")
    gt = load_mask_as_ids(args.mask)
    img, gt = ensure_same_size(img, gt)
    H, W = gt.shape
    num_classes = len(CLASS_INFO)

    # временная маска предсказания (ускорение)
    used_mask = np.zeros((H, W), dtype=bool)
    pred_upd = PredMaskUpdater(H, W)

    # генератор скриблов по ошибкам
    cfg = ScribbleGenCfg(
        max_retries=400,
        margin=int(args.margin),
        strict_bad=True,
        no_overlap=bool(args.no_overlap),
    )
    gen = ScribbleGeneratorOnErrors(gt_mask=gt, seed=int(args.seed), cfg=cfg)

    # algo
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1,
        superpixel_methods=[],
        image_path="",
        image=img,
    )
    sp_method = build_superpixel_method(args)
    
    use_precomputed = False

    if args.spanno:
        # Загружаем состояние (включая superpixels) из файла
        algo.deserialize(args.spanno)

        # Проверим, что нужный метод реально есть в загруженных суперпикселях
        if sp_method not in algo.superpixels and sp_method.short_string() not in [m.short_string() for m in algo.superpixels.keys()]:
            # Иногда ключи - это объекты методов; иногда важно совпадение short_string.
            # Если short_string совпадает с одним из имеющихся методов, подберём реальный объект-ключ.
            target = None
            for m in algo.superpixels.keys():
                if getattr(m, "short_string", lambda: None)() == sp_method.short_string():
                    target = m
                    break
            if target is not None:
                sp_method = target
            else:
                raise RuntimeError(
                    f"В spanno нет суперпикселей для метода '{sp_method.short_string()}'. "
                    f"Доступно: {[m.short_string() for m in algo.superpixels.keys()]}"
                )
        use_precomputed = True

    
    algo.add_superpixel_method(sp_method)

    # CSV для IoU
    iou_csv = out_dir / "iou.csv"
    with open(iou_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "mIoU"] + [f"IoU_{CLASS_INFO[c][0]}" for c in range(num_classes)])

    def log_and_write_iou(step: int):
        miou, ious = compute_ious(pred_upd.pred_np, gt, num_classes)
        msg = f"step={step} mIoU={miou:.4f} " + " ".join(
            f"{CLASS_INFO[i][0]}={('nan' if np.isnan(v) else f'{v:.4f}')}" for i, v in enumerate(ious)
        )
        logger.info(msg)
        with open(iou_csv, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([step, miou] + ious)

    # главный цикл
    for sid in range(1, args.scribbles + 1):
        try:
            gt_id, pts01 = gen.make_scribble(pred_upd.pred_np, used_mask)
        except StopIteration as e:
            logger.info(str(e))
            break
        if args.no_overlap:
            xA = int(round(pts01[0, 0] * W)); yA = int(round(pts01[0, 1] * H))
            xB = int(round(pts01[1, 0] * W)); yB = int(round(pts01[1, 1] * H))
            mark_line_used(used_mask, xA, yA, xB, yB,
                        width=int(args.scribble_width),
                        pad=int(args.overlap_pad))


        code = GTID_TO_CODE.get(gt_id, 1)
        scrib = structs.Scribble(
            id=sid,
            points=np.array(pts01, dtype=np.float32),
            params=structs.ScribbleParams(radius=1, code=int(code)),
        )

        # как в tk_service: сначала создать суперпиксели для области, затем добавить штрих
        if not use_precomputed:
            algo._create_superpixel_for_scribble(scrib, sp_method)

        algo.add_scribble(scrib)


        # --- обновляем pred_mask ИНКРЕМЕНТАЛЬНО: красим полигон аннотированного суперпикселя под этот скрибл ---
        # Важно: мы не перебираем все аннотации, только ту, которая появилась/обновилась из-за текущего штриха.
        # Метод: ищем аннотированный суперпиксель, в который попала первая точка штриха.
        annos_obj = algo._annotations.get(sp_method, None)
        if annos_obj is not None:
            x0 = float(pts01[0, 0] * W)
            y0 = float(pts01[0, 1] * H)
            target = None
            for sp in annos_obj.annotations:
                border = np.array(sp.border, dtype=np.float32).copy()
                border[:, 0] *= W
                border[:, 1] *= H
                poly = [(float(x), float(y)) for x, y in border]
                pred_class = int(sp.code) - 1
                pred_class = max(0, min(pred_class, num_classes - 1))
                pred_upd.paint_polygon(poly, pred_class)

        # сохранение
        if (sid % args.save_every) == 0 or sid == args.scribbles:
            out_png = out_dir / f"frame_{sid:06d}.png"
            render_snapshot(
                base_img=img,
                algo=algo,
                sp_method=sp_method,
                out_png=str(out_png),
                draw_borders=not args.no_borders,
                draw_annos=not args.no_annos,
                draw_scribbles=not args.no_scribbles,
            )

            # сериализация состояния (не критична)
            out_json = out_dir / f"state_{sid:06d}.json"
            try:
                algo.serialize(str(out_json))
            except Exception:
                pass

            # IoU лог
            log_and_write_iou(sid)

    logger.info(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
