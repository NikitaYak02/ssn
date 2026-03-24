# vizualize_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Optional, List, Tuple, Union

import numpy as np
import cv2
from PIL import Image

import shapely
from shapely.geometry import Polygon, MultiPolygon
# BaseGeometry может отсутствовать в shapely.geometry; берём из .base, а если что — заглушка для type hints
try:
    from shapely.geometry.base import BaseGeometry  # type: ignore
except Exception:  # на старых/нестандартных сборках
    class BaseGeometry:  # type: ignore
        pass

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]

# ---------- утилиты ----------

def to_px(coords01: np.ndarray, H: int, W: int) -> np.ndarray:
    """Нормализованные (x,y) -> пиксели int32, форма Nx1x2 для cv2."""
    pts = np.empty_like(coords01, dtype=np.float32)
    pts[:, 0] = coords01[:, 0] * W
    pts[:, 1] = coords01[:, 1] * H
    pts = np.round(pts).astype(np.int32)
    return pts.reshape((-1, 1, 2))

def poly_to_px(poly: BaseGeometry, H: int, W: int) -> List[np.ndarray]:
    """Shapely Polygon/MultiPolygon -> список контуров в пикселях для cv2."""
    out: List[np.ndarray] = []
    if isinstance(poly, Polygon):
        ext = np.asarray(poly.exterior.coords, dtype=np.float32)
        out.append(to_px(ext, H, W))
        for r in poly.interiors:
            hole = np.asarray(r.coords, dtype=np.float32)
            if hole.shape[0] >= 3:
                out.append(to_px(hole, H, W))
    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            out += poly_to_px(p, H, W)
    return out

def random_color(seed: int) -> Tuple[int, int, int]:
    """Детерминированный цвет по индексу (BGR)."""
    rng = np.random.default_rng(seed)
    c = rng.integers(60, 255, size=3, endpoint=True, dtype=np.int32)
    return int(c[2]), int(c[1]), int(c[0])  # B, G, R

def _as_bgr(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Привести вход к BGR numpy (H,W,3)."""
    if isinstance(image, Image.Image):
        arr = np.array(image)  # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = image
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB/BGR array or PIL.Image")
    # грубая проверка порядка каналов
    if arr[..., 2].mean() > arr[..., 0].mean() + 5.0:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.copy()

# ---------- основная функция визуализации ----------

def debug_visualize_candidates(
    *,
    image: Union[np.ndarray, Image.Image],
    sp_polys: Sequence[BaseGeometry],
    scribble_points01: np.ndarray,                # Nx2, в [0..1]
    candidate_indices: Sequence[int],
    out_path: Union[str, Path],
    aabb_extra_indices: Optional[Sequence[int]] = None,
    alpha_fill: float = 0.35,
    max_w: int = 4096,
    draw_labels: bool = True,
    superpixels: Optional[Sequence[object]] = None,  # объекты с .centroid_xy (нормализованными)
) -> Path:
    """
    Сохраняет PNG с наложением всех кандидатов и штриха.
    - sp_polys: список shapely-полигонов (в нормализованных [0..1])
    - candidate_indices: индексы кандидатов из STRtree/query_bulk
    - aabb_extra_indices: индексы, добавленные AABB-fallback (рисуются оранжевым)
    - superpixels: опционально, объекты с атрибутом .centroid_xy (нормализованные)
    """
    base = _as_bgr(image)
    H, W = base.shape[:2]

    overlay = base.copy()
    vis = base.copy()

    used = set()
    for idx in candidate_indices:
        if idx is None or idx in used or idx < 0 or idx >= len(sp_polys):
            continue
        used.add(idx)
        poly = sp_polys[idx]
        contours = poly_to_px(poly, H, W)
        if not contours:
            continue
        color = random_color(idx)
        cv2.fillPoly(overlay, contours, color)
        cv2.polylines(overlay, contours, isClosed=True, color=(0, 0, 0), thickness=1)

        if draw_labels:
            if superpixels is not None and 0 <= idx < len(superpixels) and hasattr(superpixels[idx], "centroid_xy"):
                cx01, cy01 = superpixels[idx].centroid_xy
            else:
                c = poly.centroid
                cx01, cy01 = float(c.x), float(c.y)
            cx, cy = int(cx01 * W), int(cy01 * H)
            cv2.putText(vis, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # AABB fallback — оранжевый
    if aabb_extra_indices:
        for idx in aabb_extra_indices:
            if idx in used or idx < 0 or idx >= len(sp_polys):
                continue
            poly = sp_polys[idx]
            contours = poly_to_px(poly, H, W)
            if not contours:
                continue
            cv2.fillPoly(overlay, contours, (0, 165, 255))
            cv2.polylines(overlay, contours, isClosed=True, color=(40, 40, 40), thickness=1)

    # альфа-слияние
    vis = cv2.addWeighted(overlay, alpha_fill, vis, 1 - alpha_fill, 0.0)

    # штрих — красным
    if isinstance(scribble_points01, np.ndarray) and scribble_points01.size >= 4:
        pts_px = (scribble_points01 * np.array([W, H], dtype=np.float32)).astype(np.int32)
        cv2.polylines(vis, [pts_px.reshape((-1, 1, 2))], isClosed=False, color=(0, 0, 255), thickness=max(2, W // 1000))
        for p in pts_px:
            cv2.circle(vis, tuple(p.tolist()), radius=max(2, W // 1200), color=(0, 0, 255), thickness=-1)

    # downscale для больших кадров
    if W > max_w:
        scale = max_w / float(W)
        vis = cv2.resize(vis, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return out_path
