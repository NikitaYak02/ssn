#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_superpixel_annotator_with_largest_bad_region.py

Тестирование SuperPixelAnnotationAlgo через генератор скриблов по GT-маске.

Ключевая идея генератора:
- на каждом шаге берём САМУЮ БОЛЬШУЮ связную область bad = (pred != gt) (включая pred=-1)
- выбираем целевой класс как моду GT внутри этой области
- генерим ПРЯМОЙ отрезок строго внутри:
    * bad-области,
    * GT-класса (и его "внутренности" на margin px),
    * (опционально) без пересечений с предыдущими штрихами (used_mask)

Каждые N (по умолчанию 200) скриблов:
- сохраняет PNG-визуализации
- считает и логирует IoU (per-class + mIoU) и пишет в CSV.

Опционально:
- можно использовать заранее посчитанные суперпиксели (spanno .json/.json.gz), чтобы не пересчитывать.

Пример:
python test_superpixel_annotator_with_largest_bad_region.py \
  --image img.png \
  --mask  gt.png \
  --out   out_dir \
  --method slic --n_segments 3000 --compactness 15 --sigma 1.0 \
  --scribbles 2000 --save_every 200 --seed 0 --no_overlap --margin 2

С precomputed:
python test_superpixel_annotator_with_largest_bad_region.py \
  --image img.png --mask gt.png --out out_dir \
  --method slic --spanno img.spanno.json.gz \
  --scribbles 2000 --save_every 200 --seed 0 --no_overlap --margin 2
"""

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label

import structs  # твой structs.py


# ---- Классы (0..12 в GT), имена/цвета для отрисовки ----
# Важно: в tk_service "code" начинается с 1, поэтому code = gt_id + 1.
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
      - RGB/палитра — берём 1 канал
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
            iou = float("nan")
        else:
            iou = inter / union
        ious.append(iou)

    valid = [v for v in ious if not np.isnan(v)]
    miou = float(np.mean(valid)) if valid else float("nan")
    return miou, ious


# -------------------- Pred mask updater (ускорение) --------------------

class PredMaskUpdater:
    """
    Держит временную маску предразметки pred_mask:
      - numpy pred_np: [H,W] int32, init=-1
      - PIL pred_pil: mode 'I'
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
        x1 = int(np.ceil(max(xs)) + 1)
        y1 = int(np.ceil(max(ys)) + 1)
        return x0, y0, x1, y1

    def paint_polygon(self, poly_xy: List[Tuple[float, float]], value: int) -> None:
        if len(poly_xy) < 3:
            return

        x0, y0, x1, y1 = self._poly_bbox(poly_xy)
        x0 = max(0, min(x0, self.W))
        x1 = max(0, min(x1, self.W))
        y0 = max(0, min(y0, self.H))
        y1 = max(0, min(y1, self.H))
        if x1 <= x0 or y1 <= y0:
            return

        crop = self.pred_pil.crop((x0, y0, x1, y1))
        dc = ImageDraw.Draw(crop)
        shifted = [(float(x - x0), float(y - y0)) for (x, y) in poly_xy]
        dc.polygon(shifted, fill=int(value))
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


# -------------------- Scribble generator: largest bad region --------------------

@dataclass
class ScribbleGenCfg:
    max_retries: int = 200
    margin: int = 2
    no_overlap: bool = True


class LargestBadRegionScribbleGenerator:
    """
    Берёт самую большую связную компоненту bad=(pred!=gt), размечает именно её.
    Делает прямой отрезок (2 точки), который лежит:
      - внутри bad-области
      - внутри целевого GT класса (и с отступом от границы на margin)
      - (опционально) не пересекает уже нарисованные штрихи (used_mask)
    """

    def __init__(self, gt_mask: np.ndarray, seed: int, cfg: ScribbleGenCfg):
        self.gt = gt_mask.astype(np.int32, copy=False)
        self.H, self.W = gt_mask.shape
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.num_classes = len(CLASS_INFO)

        # длина — половина диагонали изображения
        self.L = 0.5 * float(np.sqrt(self.W * self.W + self.H * self.H))

        # inner mask внутри каждого GT класса (отступ margin)
        self.gt_inner_masks: List[np.ndarray] = []
        margin = int(max(0, cfg.margin))
        for cid in range(self.num_classes):
            cls = (self.gt == cid)
            if margin <= 0:
                inner = cls
            else:
                dist = distance_transform_edt(cls)
                inner = cls & (dist > margin)
            self.gt_inner_masks.append(inner)

    def _largest_component_mask(self, bad: np.ndarray) -> Optional[np.ndarray]:
        lab, n = cc_label(bad)
        if n <= 0:
            return None
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        k = int(np.argmax(counts))
        if counts[k] == 0:
            return None
        return (lab == k)

    def _mode_gt_in_mask(self, m: np.ndarray) -> int:
        vals = self.gt[m]
        if vals.size == 0:
            return 0
        bc = np.bincount(vals, minlength=self.num_classes)
        return int(np.argmax(bc))

    def _pick_start(self, allowed: np.ndarray) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(allowed)
        if xs.size == 0:
            return None
        i = int(self.rng.integers(0, xs.size))
        return int(xs[i]), int(ys[i])

    def _trace(self, x0: int, y0: int, dx: float, dy: float,
               allowed: np.ndarray, used: np.ndarray) -> Tuple[int, int]:
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
            if not allowed[yi, xi]:
                break
            if self.cfg.no_overlap and used[yi, xi]:
                break
            last_x, last_y = xi, yi
        return last_x, last_y

    def make_scribble(self, pred_mask: np.ndarray, used_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        bad = (pred_mask != self.gt)
        if not np.any(bad):
            raise StopIteration("No incorrect pixels left (pred matches GT everywhere).")

        comp = self._largest_component_mask(bad)
        if comp is None:
            raise StopIteration("No bad connected component found.")

        # внутренняя часть компоненты (чтобы не липнуть к границе ошибки)
        margin = int(max(0, self.cfg.margin))
        if margin > 0:
            comp_dist = distance_transform_edt(comp)
            comp_inner = comp & (comp_dist > margin)
        else:
            comp_inner = comp

        cid = self._mode_gt_in_mask(comp)
        gt_inner = self.gt_inner_masks[cid]

        # строго внутри bad и класса
        allowed = comp_inner & gt_inner & (self.gt == cid) & bad
        if self.cfg.no_overlap:
            allowed = allowed & (~used_mask)

        if not np.any(allowed):
            # ослабим чуть-чуть: без comp_inner, но с gt_inner и классом
            allowed = comp & gt_inner & (self.gt == cid) & bad
            if self.cfg.no_overlap:
                allowed = allowed & (~used_mask)
            if not np.any(allowed):
                raise RuntimeError(
                    "Largest bad region exists, but no allowed pixels to place a scribble. "
                    "Try lowering --margin or disabling --no_overlap."
                )

        for _ in range(self.cfg.max_retries):
            start = self._pick_start(allowed)
            if start is None:
                break
            x0, y0 = start

            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            dx = float(np.cos(theta))
            dy = float(np.sin(theta))

            x1a, y1a = self._trace(x0, y0, +dx, +dy, allowed, used_mask)
            x1b, y1b = self._trace(x0, y0, -dx, -dy, allowed, used_mask)

            if (x1a, y1a) == (x1b, y1b):
                continue

            pts = np.array([[x1b, y1b], [x1a, y1a]], dtype=np.float32)
            pts01 = np.empty_like(pts, dtype=np.float32)
            pts01[:, 0] = pts[:, 0] / float(self.W)
            pts01[:, 1] = pts[:, 1] / float(self.H)
            return cid, pts01

        raise RuntimeError("Failed to generate a non-degenerate scribble in the largest bad region.")


# -------------------- Superpixel method selection --------------------

def build_superpixel_method(args) -> structs.SuperPixelMethod:
    sens = args.sensitivity
    m = args.method.lower()

    if m == "slic":
        nseg = int(sens) if sens is not None else int(args.n_segments)
        return structs.SLICSuperpixel(
            n_clusters=nseg,
            compactness=float(args.compactness),
            sigma=float(args.sigma),
        )
    if m in ("fwb", "felzenszwalb"):
        scale = float(sens) if sens is not None else float(args.scale)
        return structs.FelzenszwalbSuperpixel(
            min_size=int(args.min_size),
            sigma=float(args.f_sigma),
            scale=float(scale),
        )
    if m in ("watershed", "ws"):
        comps = int(sens) if sens is not None else int(args.ws_components)
        return structs.WatershedSuperpixel(
            compactness=float(args.ws_compactness),
            n_components=comps,
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

    img_rgba = Image.alpha_composite(img.convert("RGBA"), overlay)

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


# -------------------- Logging helpers --------------------

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


def pick_method_key(algo: "structs.SuperPixelAnnotationAlgo", want: "structs.SuperPixelMethod") -> "structs.SuperPixelMethod":
    """
    Возвращает именно тот объект метода, который используется как ключ в algo.* dict-ах.
    """
    w = want.short_string()
    for m in algo.superpixel_methods:
        if m.short_string() == w:
            return m
    for m in algo.superpixels.keys():
        if getattr(m, "short_string", lambda: None)() == w:
            return m
    return want


def make_method_first(algo: "structs.SuperPixelAnnotationAlgo", method: "structs.SuperPixelMethod") -> None:
    """
    Важно для structs: часть кода смотрит на self.superpixel_methods[0].
    """
    if not algo.superpixel_methods:
        algo.superpixel_methods = [method]
        return
    for i, m in enumerate(algo.superpixel_methods):
        if m is method:
            if i != 0:
                algo.superpixel_methods.insert(0, algo.superpixel_methods.pop(i))
            return
    algo.superpixel_methods.insert(0, method)


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB изображение")
    ap.add_argument("--mask", required=True, help="GT маска (индексы классов в пикселях)")
    ap.add_argument("--out", required=True, help="папка для результатов")

    ap.add_argument("--scribbles", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--sensitivity", type=float, default=None,
                    help="Единая чувствительность: SLIC->n_segments, Felz->scale, WS->n_components")

    ap.add_argument("--margin", type=int, default=2,
                    help="Отступ от границы в пикселях (и для bad-компоненты, и для GT класса).")

    ap.add_argument("--no_overlap", action="store_true",
                    help="Не допускать пересечения новых штрихов с предыдущими.")
    ap.add_argument("--scribble_width", type=int, default=3)
    ap.add_argument("--overlap_pad", type=int, default=2)

    ap.add_argument("--method", default="slic", choices=["slic", "felzenszwalb", "fwb", "watershed", "ws"])

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

    # viz switches
    ap.add_argument("--no_borders", action="store_true")
    ap.add_argument("--no_annos", action="store_true")
    ap.add_argument("--no_scribbles", action="store_true")

    ap.add_argument("--spanno", default=None,
                    help="Путь к заранее сгенерированному spanno (.json или .json.gz). "
                         "Если задан, суперпиксели берём из файла и не пересчитываем.")

    ap.add_argument("--downscale", type=float, default=1.0,
                    help="downscale_coeff для algo. Для precomputed обычно держи 1.0 и совпадай со spanno.")

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    img = Image.open(args.image).convert("RGB")
    gt = load_mask_as_ids(args.mask)
    img, gt = ensure_same_size(img, gt)
    H, W = gt.shape
    num_classes = len(CLASS_INFO)

    # временная маска предсказания
    pred_upd = PredMaskUpdater(H, W)
    used_mask = np.zeros((H, W), dtype=bool)

    # algo
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=float(args.downscale),
        superpixel_methods=[],
        image_path="",
        image=img,
    )
    sp_method = build_superpixel_method(args)

    use_precomputed = False
    if args.spanno:
        algo.deserialize(args.spanno)
        use_precomputed = True

    # важно: sp_method должен быть именно объектом-ключом из algo
    if use_precomputed:
        sp_method = pick_method_key(algo, sp_method)
        make_method_first(algo, sp_method)
        logger.info("Using precomputed superpixels from spanno. Active method=%s", sp_method.short_string())
    else:
        algo.add_superpixel_method(sp_method)
        make_method_first(algo, sp_method)
        logger.info("No spanno: superpixels will be created on-the-fly. Active method=%s", sp_method.short_string())

    logger.info("Methods order: %s", [m.short_string() for m in algo.superpixel_methods])

    # генератор: largest bad region
    cfg = ScribbleGenCfg(
        max_retries=200,
        margin=int(args.margin),
        no_overlap=bool(args.no_overlap),
    )
    gen = LargestBadRegionScribbleGenerator(gt_mask=gt, seed=int(args.seed), cfg=cfg)

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
    for sid in range(0, args.scribbles + 1):
        try:
            gt_id, pts01 = gen.make_scribble(pred_upd.pred_np, used_mask)
        except StopIteration as e:
            logger.info(str(e))
            break

        # отметим used_mask (чтобы новые штрихи не пересекались)
        if args.no_overlap:
            xA = int(round(float(pts01[0, 0] * W))); yA = int(round(float(pts01[0, 1] * H)))
            xB = int(round(float(pts01[1, 0] * W))); yB = int(round(float(pts01[1, 1] * H)))
            mark_line_used(
                used_mask, xA, yA, xB, yB,
                width=int(args.scribble_width),
                pad=int(args.overlap_pad),
            )

        code = GTID_TO_CODE.get(gt_id, 1)
        scrib = structs.Scribble(
            id=sid,
            points=np.array(pts01, dtype=np.float32),
            params=structs.ScribbleParams(radius=1, code=int(code)),
        )

        # создать суперпиксели для области (только если не precomputed)
        if not use_precomputed:
            algo._create_superpixel_for_scribble(scrib, sp_method)

        algo.add_scribble(scrib)

        # обновляем pred_mask инкрементально: красим только тот аннотированный SP, где лежит первая точка
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

            out_json = out_dir / f"state_{sid:06d}.json"
            try:
                algo.serialize(str(out_json))
            except Exception:
                pass

            log_and_write_iou(sid)

    logger.info(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
