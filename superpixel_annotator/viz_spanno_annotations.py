#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
viz_spanno_annotations.py

Визуализирует аннотации из dump/spanno файла:
- создаёт чёрный фон
- рисует каждый аннотированный регион (полигон) цветом его класса (code -> class_id = code-1)
- сохраняет PNG

НОВОЕ:
- можно подать "цветную" GT-разметку (mask), и тогда цвета предсказания
  будут взяты ИЗ ЭТОЙ разметки, чтобы совпадали 1-в-1.

Поддержка:
- dump: .json / .json.gz (serialize/deserialize твоего structs.SuperPixelAnnotationAlgo)
- размеры:
    * через --image (берём W,H из изображения)
    * или через --width/--height
- выбор метода: --method (short_string)

Цвета:
- по умолчанию используются CLASS_INFO (жёстко заданные)
- если задан --color_mask:
    * если mask в режиме 'P' (paletted) или 'L' (индексы) — mapping class_id->RGB берётся корректно
    * если mask в RGB:
         - ЛУЧШИЙ вариант: вместе с --id_mask (маска с id классов). Тогда для каждого id берём
           доминирующий цвет из color_mask и получаем точную палитру.
         - если --id_mask не задан: используем эвристику (по частоте цветов в mask) и маппим
           самые частые цвета к классам 0..K-1 (может не совпасть с твоим id-энкодингом).

Примеры:
  # обычная палитра CLASS_INFO
  python viz_spanno_annotations.py --dump state_000200.json --image img.png --out ann.png

  # цвета из paletted/индексной маски (P/L)
  python viz_spanno_annotations.py --dump state_000200.json --image img.png --color_mask gt_color.png --out ann.png

  # цвета из RGB маски + точное соответствие через id_mask
  python viz_spanno_annotations.py --dump state_000200.json --image img.png \
      --color_mask gt_rgb.png --id_mask gt_ids.png --out ann.png
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw

import structs  # твой structs.py


# Дефолтные цвета (если не задан --color_mask)
CLASS_INFO = [
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


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def pick_method(algo: "structs.SuperPixelAnnotationAlgo", method_short: Optional[str]):
    """
    Выбрать метод для отрисовки:
    - если --method не задан: берём первый, у которого есть аннотации
    - если задан: ищем по short_string()
    """
    if not getattr(algo, "_annotations", None):
        return None

    if method_short is None:
        for m, ann in algo._annotations.items():
            if ann is not None and getattr(ann, "annotations", None):
                return m
        return next(iter(algo._annotations.keys()))

    want = method_short.strip().lower()
    for m in algo._annotations.keys():
        ms = getattr(m, "short_string", lambda: str(m))()
        if str(ms).strip().lower() == want:
            return m
    return None


def load_id_mask(path: str, W: int, H: int) -> np.ndarray:
    m = Image.open(path)
    if m.mode not in ("P", "L", "I;16", "I"):
        m = m.convert("RGB")
        arr = np.array(m)[..., 0]
    else:
        arr = np.array(m)

    if arr.ndim != 2:
        raise ValueError(f"id_mask must be 2D, got {arr.shape}")

    if (arr.shape[0], arr.shape[1]) != (H, W):
        m2 = Image.fromarray(arr.astype(np.uint16 if arr.max() > 255 else np.uint8))
        m2 = m2.resize((W, H), resample=Image.NEAREST)
        arr = np.array(m2)

    return arr.astype(np.int32)


def resize_color_mask(m: Image.Image, W: int, H: int) -> Image.Image:
    if m.size != (W, H):
        return m.resize((W, H), resample=Image.NEAREST)
    return m


def palette_from_color_mask(color_mask_path: str,
                            W: int, H: int,
                            id_mask_path: Optional[str],
                            num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    """
    Возвращает mapping: class_id -> (r,g,b) из color_mask.
    """
    m = Image.open(color_mask_path)
    m = resize_color_mask(m, W, H)

    # 1) Paletted ('P'): используем palette и индексы пикселей (класс==индекс)
    if m.mode == "P":
        idx = np.array(m, dtype=np.int32)
        pal = m.getpalette()  # list length 768
        if pal is None or len(pal) < 3:
            raise ValueError("Paletted mask has no palette.")

        out: Dict[int, Tuple[int, int, int]] = {}
        for cid in range(num_classes):
            # берём цвет из палитры по индексу cid (если есть)
            pi = 3 * cid
            if pi + 2 < len(pal):
                out[cid] = (int(pal[pi]), int(pal[pi + 1]), int(pal[pi + 2]))
        return out

    # 2) Grayscale ('L'/'I'): считаем что значения пикселей = class_id,
    #    но нужен "цвет". Тогда делаем pseudo-color через CLASS_INFO.
    if m.mode in ("L", "I", "I;16"):
        return {cid: hex_to_rgb(CLASS_INFO[min(cid, len(CLASS_INFO) - 1)][1]) for cid in range(num_classes)}

    # 3) RGB: лучше всего иметь id_mask для точного сопоставления
    rgb = np.array(m.convert("RGB"), dtype=np.uint8)

    if id_mask_path:
        ids = load_id_mask(id_mask_path, W, H)
        out: Dict[int, Tuple[int, int, int]] = {}
        for cid in range(num_classes):
            sel = (ids == cid)
            if not np.any(sel):
                continue
            cols = rgb[sel]  # [N,3]
            # доминирующий цвет
            cols_view = cols[:, 0].astype(np.uint32) << 16 | cols[:, 1].astype(np.uint32) << 8 | cols[:, 2].astype(np.uint32)
            uniq, cnt = np.unique(cols_view, return_counts=True)
            v = int(uniq[int(np.argmax(cnt))])
            out[cid] = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
        return out

    # 4) RGB без id_mask: эвристика — маппим самые частые цвета к class_id 0..K-1
    flat = rgb.reshape(-1, 3)
    packed = flat[:, 0].astype(np.uint32) << 16 | flat[:, 1].astype(np.uint32) << 8 | flat[:, 2].astype(np.uint32)
    uniq, cnt = np.unique(packed, return_counts=True)
    order = np.argsort(-cnt)  # по убыванию частоты

    out: Dict[int, Tuple[int, int, int]] = {}
    for cid in range(min(num_classes, order.size)):
        v = int(uniq[order[cid]])
        out[cid] = ((v >> 16) & 255, (v >> 8) & 255, v & 255)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="Файл состояния (state_*.json / *.spanno.json[.gz])")
    ap.add_argument("--out", required=True, help="Куда сохранить PNG")

    ap.add_argument("--image", default=None, help="Исходное изображение, чтобы взять W,H")
    ap.add_argument("--width", type=int, default=None, help="Ширина (если нет --image)")
    ap.add_argument("--height", type=int, default=None, help="Высота (если нет --image)")

    ap.add_argument("--method", default=None,
                    help="Какой метод рисовать (short_string), например: slic / felz / ws. "
                         "Если не задан — выберется автоматически.")

    ap.add_argument("--alpha", type=int, default=255, help="Прозрачность заливки 0..255")
    ap.add_argument("--draw_borders", action="store_true", help="Рисовать контуры регионов")

    # NEW: цвета из GT цветной разметки
    ap.add_argument("--color_mask", default=None,
                    help="Путь к цветной маске разметки (P/L/RGB). "
                         "Если задано — цвета предсказания берутся отсюда.")
    ap.add_argument("--id_mask", default=None,
                    help="Путь к id-маске (2D значения классов). "
                         "Нужно для точного сопоставления цветов, если --color_mask в RGB.")

    args = ap.parse_args()

    dump_path = str(args.dump)

    if args.image:
        im = Image.open(args.image)
        W, H = im.size
    else:
        if args.width is None or args.height is None:
            raise SystemExit("Нужно указать --image или пару --width/--height.")
        W, H = int(args.width), int(args.height)

    # Загружаем состояние
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image_path=args.image,
        image=None,
    )
    algo.deserialize(dump_path)

    sp_method = pick_method(algo, args.method)
    if sp_method is None:
        raise SystemExit("В dump нет аннотаций (_annotations пуст).")

    annos_obj = algo._annotations.get(sp_method, None)
    annos = getattr(annos_obj, "annotations", []) if annos_obj is not None else []
    if not annos:
        ms = getattr(sp_method, "short_string", lambda: str(sp_method))()
        raise SystemExit(f"Для метода '{ms}' аннотаций нет.")

    # Определяем, сколько классов нужно (по max(code-1))
    max_cid = 0
    for sp in annos:
        code = int(getattr(sp, "code", 1))
        max_cid = max(max_cid, max(0, code - 1))
    num_classes = max(max_cid + 1, 1)

    # Палитра class_id -> rgb
    if args.color_mask:
        palette = palette_from_color_mask(args.color_mask, W, H, args.id_mask, num_classes)
        # fallback если каких-то cid нет
        for cid in range(num_classes):
            if cid not in palette:
                palette[cid] = hex_to_rgb(CLASS_INFO[min(cid, len(CLASS_INFO) - 1)][1])
    else:
        palette = {cid: hex_to_rgb(CLASS_INFO[min(cid, len(CLASS_INFO) - 1)][1]) for cid in range(num_classes)}

    # Рендер: чёрный фон + заливки
    bg = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    for sp in annos:
        border01 = np.asarray(sp.border, dtype=np.float32)
        if border01.ndim != 2 or border01.shape[0] < 3:
            continue

        border_px = border01.copy()
        border_px[:, 0] *= float(W)
        border_px[:, 1] *= float(H)
        poly: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in border_px]

        code = int(getattr(sp, "code", 1))
        cid = max(0, code - 1)
        rgb = palette.get(cid, (255, 255, 255))
        fill = (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(args.alpha))

        d.polygon(poly, fill=fill)
        if args.draw_borders:
            d.line(poly + [poly[0]], fill=(255, 255, 255, 255), width=1)

    out = Image.alpha_composite(bg, overlay).convert("RGB")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, quality=95)

    ms = getattr(sp_method, "short_string", lambda: str(sp_method))()
    print(f"Saved: {args.out} | method={ms} | annos={len(annos)} | size={W}x{H} | classes={num_classes}")
    if args.color_mask:
        print("Palette source: color_mask"
              + (" + id_mask" if args.id_mask else " (heuristic if RGB without id_mask)"))


if __name__ == "__main__":
    main()
