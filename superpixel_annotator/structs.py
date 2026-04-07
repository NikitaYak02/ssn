# -*- coding: utf-8 -*-
from __future__ import annotations

# ── Stdlib ─────────────────────────────────────────────────────────────────────
import json
import logging
import os
import io
import gzip
import tempfile
import hashlib
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Sequence, Any

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
import cv2
from PIL import Image, ImageDraw

import shapely
from shapely.geometry import LineString, Polygon
from shapely import prepared
from shapely.strtree import STRtree
from shapely.ops import split as shp_split
from shapely.ops import unary_union  # union геометрий

from shapely.errors import GEOSException
from lib.utils.torch_device import get_torch_device

# --- Geometry sanitization ----------------------------------------------------
# GEOS may raise TopologyException ("side location conflict") during set-ops on
# invalid polygons (self-intersections, tiny slivers after split, etc.).
# We sanitize polygon geometry once (lazy, per-superpixel) and cache it.
try:  # shapely >= 2.0
    from shapely import make_valid as _shp_make_valid  # type: ignore
    from shapely import set_precision as _shp_set_precision  # type: ignore
except Exception:  # pragma: no cover
    _shp_make_valid = None
    _shp_set_precision = None


def _largest_polygon(geom):
    """Return the largest Polygon from Polygon/MultiPolygon/GeometryCollection."""
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    geoms = getattr(geom, "geoms", None)
    if not geoms:
        return None
    polys = [g for g in geoms if isinstance(g, Polygon) and (not g.is_empty)]
    if not polys:
        return None
    return max(polys, key=lambda p: float(p.area))


def _sanitize_polygon(poly: Polygon, grid_size: float = 1e-9) -> Polygon:
    """Best-effort repair for invalid polygons. Returns a (usually) valid Polygon."""
    if poly is None or poly.is_empty:
        return poly

    g = poly

    # (1) Quantize coordinates to a grid to remove near-degenerate slivers.
    if _shp_set_precision is not None and grid_size and grid_size > 0:
        try:
            g = _shp_set_precision(g, float(grid_size))
        except Exception:
            pass

    # (2) Repair invalid geometry.
    try:
        if not g.is_valid:
            if _shp_make_valid is not None:
                g = _shp_make_valid(g)
            else:
                g = g.buffer(0)
    except Exception:
        try:
            g = g.buffer(0)
        except Exception:
            return poly

    out = _largest_polygon(g)
    if out is None:
        return poly

    # (3) Final guard: some edge cases still produce invalid polygon; buffer(0) once.
    try:
        if not out.is_valid:
            out = out.buffer(0)
            out2 = _largest_polygon(out)
            if out2 is not None:
                out = out2
    except Exception:
        pass

    return out

from skimage.filters import sobel
from skimage.measure import find_contours
from skimage.morphology import medial_axis, skeletonize
from skimage.segmentation import felzenszwalb as sk_fz
from skimage.segmentation import slic as sk_slic
from skimage.segmentation import watershed as sk_ws

from scipy import ndimage
from scipy.spatial import cKDTree
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt  # (возможен для отладки, не обязателен)


import sys as _sys_color
_sys_color.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.utils.color_conv import rgb2lab, lab2rgb  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("sp_anno")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")


def enable_file_logging(log_path: str = "./sp_anno.log"):
    """Опционально писать логи в файл."""
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        logger.info("File logging enabled: %s", log_path)
    except Exception as e:
        logger.warning("Failed to enable file logging: %s", e)


# ── Optional fast_slic ────────────────────────────────────────────────────────
_HAS_FAST_SLIC = False
FastSlic = None
FastSlicAvx2 = None
try:
    from fast_slic import Slic as FastSlic  # pip install fast_slic
    try:
        from fast_slic.avx2 import SlicAvx2 as FastSlicAvx2
    except Exception:
        FastSlicAvx2 = None
    _HAS_FAST_SLIC = True
except Exception:
    pass

# ── Optional SSN model ─────────────────────────────────────────────────────────
import sys as _sys_mod

# The annotator lives at  .../ssn/superpixel_annotator/structs.py
# The SSN model lives at  .../ssn/model.py
_SSN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Cache checkpoint contents once per path and keep reusable models per device.
_ssn_state_cache: dict = {}
_ssn_model_cache: dict = {}


def _get_ssn_state_dict(abs_path: str):
    """
    Read the checkpoint from disk only once per process.
    """
    import torch as _torch

    if abs_path not in _ssn_state_cache:
        state = _torch.load(abs_path, map_location="cpu")
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        _ssn_state_cache[abs_path] = state
        logger.info("SSN checkpoint loaded once: %s", abs_path)

    return _ssn_state_cache[abs_path]


def _get_ssn_model(weight_path: str, fdim: int, nspix: int, niter: int, device: str = "cpu"):
    """
    Return a cached SSNModel instance on the requested device.

    The checkpoint is read from disk only once. ``nspix`` and ``niter`` are
    runtime parameters for SSN iteration, so they are updated on the cached
    model before each use.
    """
    abs_path = os.path.abspath(weight_path)
    key = (abs_path, int(fdim), str(device))

    if key not in _ssn_model_cache:
        if _SSN_DIR not in _sys_mod.path:
            _sys_mod.path.insert(0, _SSN_DIR)
        try:
            from model import SSNModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"Cannot import SSNModel from '{_SSN_DIR}'. "
                "Make sure model.py is present there."
            ) from exc

        state = _get_ssn_state_dict(abs_path)
        model = SSNModel(fdim, nspix, niter).to(device)
        model.load_state_dict(state)
        model.eval()
        _ssn_model_cache[key] = model
        logger.info(
            "SSN model initialised: %s  (fdim=%d  device=%s)",
            abs_path, fdim, device,
        )

    model = _ssn_model_cache[key]
    model.nspix = int(nspix)
    model.n_iter = int(niter)
    model.eval()
    return model


def _compute_roi_embeddings(
    image_lab_roi: np.ndarray,
    weight_path: str,
    fdim: int = 20,
    color_scale: float = 0.26,
    pos_scale: float = 2.5,
) -> np.ndarray:
    """
    Run SSNModel.feature_extract on a LAB ROI patch and return per-pixel
    L2-normalised embedding vectors.

    Parameters
    ----------
    image_lab_roi : np.ndarray  (H, W, 3)  float32  LAB image patch
    weight_path   : str         path to .pth checkpoint
    fdim          : int         feature dimension (must match checkpoint)
    color_scale   : float       LAB channel scale (same as used for superpixels)
    pos_scale     : float       coordinate scale base (same as used for superpixels)

    Returns
    -------
    emb : np.ndarray  (H, W, fdim)  float32, each pixel vector is L2-normalised
    """
    import math as _math
    import torch as _torch

    H, W = image_lab_roi.shape[:2]
    device = get_torch_device(_torch)

    # load/cache model (nspix/niter irrelevant for feature_extract; use canonical defaults)
    model = _get_ssn_model(weight_path, fdim, nspix=100, niter=5, device=device)

    img_t = (
        _torch.from_numpy(image_lab_roi.astype(np.float32))
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )  # (1, 3, H, W)

    # same coordinate scaling formula as compute_superpixels
    nspix_per_axis = int(_math.sqrt(100))
    ps = pos_scale * max(nspix_per_axis / H, nspix_per_axis / W)
    coords = _torch.stack(
        _torch.meshgrid(
            _torch.arange(H, device=device, dtype=_torch.float32),
            _torch.arange(W, device=device, dtype=_torch.float32),
            indexing="ij",
        ), 0
    ).unsqueeze(0)  # (1, 2, H, W)

    model_input = _torch.cat([color_scale * img_t, ps * coords], dim=1)  # (1, 5, H, W)

    with _torch.no_grad():
        pixel_f = model.feature_extract(model_input)  # (1, fdim, H, W)

    emb = pixel_f.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, fdim)

    # L2-normalise per pixel
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb = (emb / np.maximum(norms, 1e-8)).astype(np.float32)
    return emb


# ── Typing aliases ─────────────────────────────────────────────────────────────
BBox = Tuple[float, float, float, float]  # (x0,y0,x1,y1) in [0..1]

# ── Save/Load constants & exceptions ───────────────────────────────────────────
SCHEMA_MAGIC = "spanno"
SCHEMA_VERSION = 2


class SaveError(RuntimeError):
    ...


class LoadError(RuntimeError):
    ...


# ── Save/Load helpers ──────────────────────────────────────────────────────────
def _is_gzip_path(path: str) -> bool:
    return str(path).lower().endswith(".gz")


def _open_text_for_write_atomic(path: str, is_gz: bool):
    """
    Возвращает (tmp_path, fp). Писать и закрыть, затем os.replace(tmp_path, path).
    """
    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    suffix = ".json.gz" if is_gz else ".json"
    tmp = tempfile.NamedTemporaryFile(prefix=".tmp_spanno_", suffix=suffix, dir=dir_, delete=False)
    tmp_path = tmp.name
    tmp.close()  # откроем заново в нужном режиме
    if is_gz:
        fp = gzip.open(tmp_path, mode="wt", encoding="utf-8")
    else:
        fp = open(tmp_path, mode="w", encoding="utf-8")
    return tmp_path, fp


def _open_text_for_read(path: str):
    if _is_gzip_path(path):
        return gzip.open(path, mode="rt", encoding="utf-8")
    return open(path, mode="r", encoding="utf-8")


def _rotate_backup(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    bak = path + ".bak"
    try:
        if os.path.exists(bak):
            os.remove(bak)
    except Exception:
        pass
    try:
        os.rename(path, bak)
        return bak
    except Exception:
        # если не удалось — тихо продолжаем, атомарность всё равно сохранится
        return None


def _sha256_of_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _round_float_list(x: Iterable[float], ndigits: int = 7) -> List[float]:
    return [round(float(v), ndigits) for v in x]


# ── Small utils ────────────────────────────────────────────────────────────────
def _maybe_gaussian(img_u8: np.ndarray, sigma: float) -> np.ndarray:
    if sigma and sigma > 0:
        k = int(max(3, 2 * round(3 * sigma) + 1))
        return cv2.GaussianBlur(img_u8, (k, k), sigmaX=float(sigma),
                                sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
    return img_u8


def _normalize_bbox01(bbox: BBox) -> BBox:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    return (x0, y0, x1, y1)


def _bbox_to_pixel_rect(bbox: BBox, width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = _normalize_bbox01(bbox)
    ix0 = max(0, min(width, int(round(x0 * width))))
    iy0 = max(0, min(height, int(round(y0 * height))))
    ix1 = max(0, min(width, int(round(x1 * width))))
    iy1 = max(0, min(height, int(round(y1 * height))))
    return ix0, iy0, ix1, iy1


def bbox_is_intersect(b1: BBox, b2: BBox) -> bool:
    b1 = _normalize_bbox01(b1)
    b2 = _normalize_bbox01(b2)
    return max(b1[0], b2[0]) <= min(b1[2], b2[2]) and max(b1[1], b2[1]) <= min(b1[3], b2[3])


def bbox_intersect(b1: BBox, b2: BBox) -> BBox:
    b1 = _normalize_bbox01(b1)
    b2 = _normalize_bbox01(b2)
    return (max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3]))


def simplify(polygon: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Упростить полигон через shapely и вернуть np.array Nx2 (float32)."""
    if polygon is None or len(polygon) < 3:
        return polygon.astype(np.float32, copy=False)
    poly = Polygon(polygon)
    poly_s = poly.simplify(tolerance=tolerance)
    if poly_s.is_empty:
        return polygon.astype(np.float32, copy=False)
    return np.array(poly_s.boundary.coords[:], dtype=np.float32)


def _strtree_query_indices(tree: STRtree, geom) -> List[int]:
    """
    Совместимость с Shapely 1.x/2.x:
    - Может вернуть индексы, геометрии или ndarray смешанного типа.
    Возвращаем список индексов в исходном порядке без дубликатов.
    """
    res = None
    try:
        # Shapely 2.x часто возвращает np.ndarray индексов
        res = tree.query(geom)
        if isinstance(res, np.ndarray):
            return [int(i) for i in res.tolist()]
        # список геометрий:
        if res and hasattr(res[0], "geom_type"):
            geom_to_idx = {id(g): i for i, g in enumerate(tree.geometries)}
            return [geom_to_idx.get(id(g)) for g in res if geom_to_idx.get(id(g)) is not None]
        # список индексов:
        return [int(i) for i in res] if res is not None else []
    except Exception:
        # older path
        try:
            _, right_idx = tree.query_bulk([geom], predicate="intersects")
            return sorted(set(int(i) for i in right_idx.tolist()))
        except Exception:
            return []


def labels_to_polygons(
    labels: np.ndarray,
    bbox01: BBox,
    out_h: int,
    out_w: int,
    filter_labels: Optional[Iterable[int]] = None,
    simplify_tol: float = 1e-10,
) -> Dict[int, np.ndarray]:
    """Контуризация меток в нормированных координатах исходного изображения."""
    H, W = labels.shape[:2]
    labels_extended = np.zeros((H + 2, W + 2), dtype=labels.dtype)
    labels_extended[1:-1, 1:-1] = labels

    result: Dict[int, np.ndarray] = {}
    if filter_labels is None:
        lbls = sorted(int(v) for v in np.unique(labels) if v > 0)
    else:
        lbls = [int(v) for v in filter_labels if int(v) > 0]

    x0, y0, x1, y1 = bbox01
    for lab in lbls:
        binary = (labels_extended == lab).astype(np.uint8)
        contours = find_contours(binary, level=0.5)
        if not contours:
            continue
        external = max(contours, key=len)[:, ::-1]  # (x, y)
        poly = (external - 1).astype(np.float32)
        poly[:, 0] /= float(W)
        poly[:, 1] /= float(H)
        poly[:, 0] = poly[:, 0] * (x1 - x0) + x0
        poly[:, 1] = poly[:, 1] * (y1 - y0) + y0
        poly_s = simplify(poly, tolerance=simplify_tol)
        if poly_s is not None and len(poly_s) >= 3:
            result[lab] = np.around(poly_s, decimals=7)
    return result


@njit(cache=True)
def parallel_stats_rgb(image: np.ndarray, mask: np.ndarray, max_label: int):
    """
    Быстрые средние и дисперсии по меткам (uint16/int32). Возвращает только по валидным меткам.

    Важно: здесь намеренно нет ``parallel=True``.
    Эта функция может вызываться из нескольких Python-потоков во время split/update,
    а numba ``workqueue`` не thread-safe при nested/concurrent parallel regions.
    image: float32/float64 (H,W,C), mask: int32/uint16 (H,W), labels>=1.
    """
    h, w, c = image.shape
    sums = np.zeros((max_label + 1, c), dtype=np.float64)
    sumsq = np.zeros((max_label + 1, c), dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)

    for i in range(h):
        for j in range(w):
            label = mask[i, j]
            if label <= 0:
                continue
            for k in range(c):
                val = image[i, j, k]
                sums[label, k] += val
                sumsq[label, k] += val * val
            counts[label] += 1

    # count valid
    n_valid = 0
    for lab in range(max_label + 1):
        if counts[lab] > 0:
            n_valid += 1

    means = np.zeros((n_valid, c), dtype=np.float64)
    variances = np.zeros((n_valid, c), dtype=np.float64)
    valid_labels = np.empty(n_valid, dtype=np.int64)

    idx = 0
    for lab in range(max_label + 1):
        cnt = counts[lab]
        if cnt <= 0:
            continue
        valid_labels[idx] = lab
        for k in range(c):
            m = sums[lab, k] / cnt
            means[idx, k] = m
            variances[idx, k] = (sumsq[lab, k] / cnt) - (m * m)
        idx += 1

    return means, variances, valid_labels


def check_bbox_contain_scribble(polyline_points: np.ndarray, rectangles: List[List[float]]) -> bool:
    """Проверяет, что ломаная целиком покрыта объединением bbox-прямоугольников."""
    if polyline_points is None or len(polyline_points) < 2:
        return False
    polygons = []
    for rect in rectangles:
        a, b, c, d = _normalize_bbox01(tuple(rect))
        if c <= a or d <= b:
            continue
        polygons.append(Polygon([(a, b), (c, b), (c, d), (a, d)]))
    if not polygons:
        return False
    line = LineString(polyline_points)
    try:
        return bool(unary_union(polygons).covers(line))
    except Exception:
        return any(poly.covers(line) for poly in polygons)


def find_holes(mask: np.ndarray) -> List[np.ndarray]:
    """Найти все «дыры» в бинарной маске, исключив компоненты, касающиеся рамки."""
    inverted = ~mask
    labeled, num_features = ndimage.label(inverted)
    if num_features == 0:
        return []
    borders = np.zeros_like(inverted, dtype=bool)
    borders[[0, -1], :] = True
    borders[:, [0, -1]] = True
    touch_border = ndimage.labeled_comprehension(
        input=borders, labels=labeled,
        index=np.arange(1, num_features + 1),
        func=np.any, out_dtype=bool, default=False
    )
    return [labeled == i for i in range(1, num_features + 1) if not touch_border[i - 1]]


def remove_small_components(mask: np.ndarray, min_size=100) -> np.ndarray:
    """Удаляет мелкие компоненты и артефакты."""
    cleaned = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    labeled, num_labels = ndimage.label(cleaned)
    if num_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(cleaned, labeled, range(1, num_labels + 1))
    keep = sizes >= min_size
    return np.isin(labeled, np.where(keep)[0] + 1)


def split_recursive(mask: np.ndarray) -> List[np.ndarray]:
    """Рекурсивно разделяет маску до устранения дыр (эвристика по главной оси)."""
    holes = find_holes(mask)
    if not holes:
        return [mask.copy()]
    hole_sizes = [np.count_nonzero(h) for h in holes]
    main_hole = holes[int(np.argmax(hole_sizes))]
    y, x = np.where(main_hole)
    points = np.column_stack((x, y))
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if len(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal_axis = vh[0]
    else:
        principal_axis = np.array([1.0, 0.0])
    perp_vec = np.array([-principal_axis[1], principal_axis[0]])
    perp_angle = np.arctan2(perp_vec[1], perp_vec[0])
    yy, xx = np.indices(mask.shape)
    split_line = (xx - centroid[0]) * np.cos(perp_angle) + (yy - centroid[1]) * np.sin(perp_angle)
    masks = [mask & (split_line >= -0.5), mask & (split_line < 0.5)]
    result: List[np.ndarray] = []
    for m in masks:
        m_clean = remove_small_components(m)
        if np.any(m_clean):
            result.extend(split_recursive(m_clean))
    return result


def split_mask(mask: np.ndarray) -> List[np.ndarray]:
    """Вернуть список масок без дырок."""
    final_masks = split_recursive(mask)
    return [m for m in final_masks if np.any(m)]


# ── Data models ────────────────────────────────────────────────────────────────
@dataclass
class ScribbleParams:
    radius: float
    code: int

    def dict_to_save(self) -> Dict:
        return {"radius": float(self.radius), "code": int(self.code)}

    @staticmethod
    def from_dict(d: Dict) -> "ScribbleParams":
        return ScribbleParams(radius=float(d["radius"]), code=int(d["code"]))


@dataclass
class Scribble:
    id: int
    points: np.ndarray              # [N,2] float in absolute normalized coords
    params: ScribbleParams
    creation_time: Optional[np.datetime64] = None

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = np.datetime64("now")
        self._recompute_bbox()

    def _recompute_bbox(self):
        if isinstance(self.points, np.ndarray) and self.points.size:
            x = self.points[:, 0]
            y = self.points[:, 1]
            self._bbox = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        else:
            self._bbox = None

    def __len__(self):
        return 0 if self.points is None else len(self.points)

    def dict_to_save(self) -> Dict:
        return {
            "id": int(self.id),
            "points": self.points.astype(float).tolist(),
            "params": self.params.dict_to_save(),
            "creation_time": str(self.creation_time) if self.creation_time is not None else "",
        }

    @staticmethod
    def from_dict(d: Dict) -> "Scribble":
        ct = d.get("creation_time") or None
        if ct:
            try:
                ct = np.datetime64(ct)
            except Exception:
                ct = None
        return Scribble(
            id=int(d["id"]),
            points=np.array(d["points"], dtype=np.float32),
            params=ScribbleParams.from_dict(d["params"]),
            creation_time=ct,
        )

    def set_from_dict(self, loaded_dict: Dict):
        s = Scribble.from_dict(loaded_dict)
        self.id, self.points, self.params, self.creation_time = s.id, s.points, s.params, s.creation_time
        self._recompute_bbox()

    @property
    def bbox(self) -> Optional[BBox]:
        return self._bbox

    def __hash__(self):
        s = float(self.points.sum()) if isinstance(self.points, np.ndarray) else 0.0
        t = str(self.creation_time) if self.creation_time is not None else ""
        return hash((s, t))


@dataclass
class SuperPixel:
    id: int
    method: str                     # e.g. 'SLIC_100_20.00_1.00'
    border: np.ndarray              # [N,2] float normalized absolute points
    parents: Optional[List[int]]    # parent SP ids if split performed
    props: Optional[np.ndarray]     # [6] mean+var LAB
    emb: Optional[np.ndarray] = None  # [fdim] L2-normalised mean embedding (not serialised)

    # Lazy caches (not serialized)
    _poly_cache: object = None
    _prep_cache: object = None
    _centroid_cache: Optional[Tuple[float, float]] = None

    def __hash__(self):
        return hash(self.method + f"_{self.id}")

    def __eq__(self, other):
        return isinstance(other, SuperPixel) and (self.method + f"_{self.id}") == (other.method + f"_{other.id}")

    @property
    def poly(self) -> Polygon:
        if self._poly_cache is None:
            # Build and sanitize polygon once, then cache.
            p = Polygon(self.border)
            p2 = _sanitize_polygon(p)
            self._poly_cache = p2

            # If sanitization changed geometry, update stored border too
            # (so any downstream Polygon(self.border) stays healthy-ish).
            try:
                if (p2 is not None) and (not p2.is_empty) and (not p2.equals(p)):
                    coords = np.asarray(p2.exterior.coords, dtype=np.float32)
                    if coords.ndim == 2 and coords.shape[0] >= 3:
                        self.border = np.around(coords, 7)
                        # invalidate dependent caches
                        self._prep_cache = None
                        self._centroid_cache = None
            except Exception:
                pass
        return self._poly_cache

    @property
    def prepared_poly(self):
        if self._prep_cache is None:
            self._prep_cache = prepared.prep(self.poly)
        return self._prep_cache

    @property
    def centroid_xy(self) -> Tuple[float, float]:
        if self._centroid_cache is None:
            c = self.poly.centroid
            self._centroid_cache = (float(c.x), float(c.y))
        return self._centroid_cache

    def dict_to_save(self) -> Dict:
        return {
            "id": int(self.id),
            "method": str(self.method),
            "border": [[round(float(y), 7) for y in x] for x in self.border],
            "parents": [] if self.parents is None else [int(i) for i in self.parents],
            "props": [] if self.props is None else [float(v) for v in np.asarray(self.props).ravel()],
        }

    @staticmethod
    def from_dict(d: Dict) -> "SuperPixel":
        parents = d.get("parents", []) or None
        if parents is not None:
            parents = [int(i) for i in parents]
        props = d.get("props", []) or None
        if props is not None:
            props = np.array(props, dtype=np.float32)
        return SuperPixel(
            id=int(d["id"]),
            method=str(d["method"]),
            border=np.array(d["border"], dtype=np.float32),
            parents=parents,
            props=props,
        )


@dataclass
class AnnotationInstance:
    id: int
    code: int
    border: np.ndarray
    parent_superpixel: int
    parent_scribble: List[int] = field(default_factory=list)
    parent_intersect: bool = True

    def dict_to_save(self) -> Dict:
        return {
            "id": int(self.id),
            "code": int(self.code),
            "border": [[round(float(y), 7) for y in x] for x in self.border],
            "parent_superpixel": int(self.parent_superpixel),
            "parent_scribble": [int(i) for i in self.parent_scribble],
            "parent_intersect": bool(self.parent_intersect),
        }

    @staticmethod
    def from_dict(d: Dict) -> "AnnotationInstance":
        return AnnotationInstance(
            id=int(d["id"]),
            code=int(d["code"]),
            border=np.array(d["border"], dtype=np.float32),
            parent_superpixel=int(d.get("parent_superpixel", -1)),
            parent_scribble=[int(i) for i in d.get("parent_scribble", [])],
            parent_intersect=bool(d.get("parent_intersect", True)),
        )

    def set_from_dict(self, loaded_dict: Dict):
        a = AnnotationInstance.from_dict(loaded_dict)
        self.id = a.id
        self.code = a.code
        self.border = a.border
        self.parent_superpixel = a.parent_superpixel
        self.parent_scribble = a.parent_scribble
        self.parent_intersect = a.parent_intersect


@dataclass
class ImageAnnotation:
    annotations: List[AnnotationInstance]

    def append(self, anno: AnnotationInstance):
        self.annotations.append(anno)

    def to_array(self) -> np.ndarray:
        raise NotImplementedError("Implement if нужно экспортировать в растр.")


# ── Superpixel methods ─────────────────────────────────────────────────────────
class SuperPixelMethod(ABC):
    @abstractmethod
    def short_string(self) -> str:
        ...

    def __le__(self, other) -> bool:
        return self.short_string() <= other.short_string()

    def __ge__(self, other) -> bool:
        return self.short_string() >= other.short_string()

    def __lt__(self, other) -> bool:
        return self.short_string() < other.short_string()

    def __gt__(self, other) -> bool:
        return self.short_string() > other.short_string()

    def __hash__(self):
        return hash(self.short_string())

    def __eq__(self, other):
        return isinstance(other, SuperPixelMethod) and self.short_string() == other.short_string()


@dataclass(frozen=True)
class WatershedSuperpixel(SuperPixelMethod):
    compactness: float
    n_components: int

    def short_string(self) -> str:
        return f"Watershed_{self.compactness:.6f}_{self.n_components}"


@dataclass(frozen=True)
class SLICSuperpixel(SuperPixelMethod):
    n_clusters: int
    compactness: float
    sigma: float

    def short_string(self) -> str:
        return f"SLIC_{int(self.n_clusters)}_{float(self.compactness):.6f}_{float(self.sigma):.6f}"


@dataclass(frozen=True)
class FelzenszwalbSuperpixel(SuperPixelMethod):
    min_size: int
    sigma: float
    scale: float

    def short_string(self) -> str:
        return f"Felzenszwalb_{self.min_size}_{self.sigma}_{self.scale}"


@dataclass(frozen=True)
class SSNSuperpixel(SuperPixelMethod):
    """
    Superpixel method backed by the trained Superpixel Sampling Network (SSN).

    Parameters
    ----------
    weight_path : str
        Path to the .pth checkpoint (absolute or relative to CWD).
    nspix : int
        Target number of superpixels (default 100).
    fdim : int
        Feature dimension — must match the checkpoint (default 20).
    niter : int
        Number of differentiable SLIC iterations (default 5).
    color_scale : float
        LAB channel scale factor (default 0.26).
    pos_scale : float
        Coordinate scale factor (default 2.5).
    """
    weight_path: str
    nspix: int         = 100
    fdim: int          = 20
    niter: int         = 5
    color_scale: float = 0.26
    pos_scale: float   = 2.5

    def short_string(self) -> str:
        # Use "|" to separate numeric params from the path (path may contain "_").
        # Format:  SSN_{nspix}_{fdim}_{niter}_{color_scale:.4f}_{pos_scale:.4f}|{abs_path}
        return (
            f"SSN_{self.nspix}_{self.fdim}_{self.niter}"
            f"_{self.color_scale:.4f}_{self.pos_scale:.4f}"
            f"|{os.path.abspath(self.weight_path)}"
        )


def find_sp_key_in_dict(sp_method: SuperPixelMethod, sp_dict: Dict[SuperPixelMethod, List]) -> bool:
    return any(key.short_string() == sp_method.short_string() for key in sp_dict)


def _hex_to_rgba_tuple(hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {hex_color!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(alpha)


def _edt_inside(mask: np.ndarray) -> np.ndarray:
    """EDT that treats the image border as outside the region."""
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    dist = ndimage.distance_transform_edt(padded)
    return dist[1:-1, 1:-1]


def _smooth_region_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Smooth a binary region with explicit dilation/erosion passes.

    The result is intended for geometric analysis of the region centerline; the
    final scribble still stays inside the original allowed mask.
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


def _split_disconnected_superpixels(labels: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ensure each label corresponds to exactly one connected component.
    Disconnected islands of the same label get new unique ids.
    """
    labels = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    valid = labels > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    uniq = [int(v) for v in np.unique(labels[valid]) if int(v) > 0]
    for lab in uniq:
        comp_mask = valid & (labels == lab)
        cc, ncc = ndimage.label(comp_mask)
        for k in range(1, int(ncc) + 1):
            out[cc == k] = next_id
            next_id += 1
    return out


def _filter_small_and_thin_superpixels(
    labels: np.ndarray,
    nspix_hint: int,
    mask: Optional[np.ndarray] = None,
    min_area_ratio: float = 0.02,
    thin_radius_px: float = 1.5,
) -> np.ndarray:
    """
    Mark very small or very thin superpixel components as background (0).
    """
    labels = np.asarray(labels, dtype=np.int32).copy()
    valid = labels > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
        labels[~np.asarray(mask, dtype=bool)] = 0

    if not valid.any():
        return labels

    approx_segment_area = max(1.0, float(np.count_nonzero(valid)) / float(max(1, nspix_hint)))
    min_area = max(8, int(round(min_area_ratio * approx_segment_area)))

    for lab in [int(v) for v in np.unique(labels[valid]) if int(v) > 0]:
        comp = (labels == lab)
        area = int(np.count_nonzero(comp))
        if area == 0:
            continue
        max_radius = float(_edt_inside(comp).max()) if comp.any() else 0.0
        if area < min_area or max_radius < float(thin_radius_px):
            labels[comp] = 0

    return labels


def _relabel_sequential(labels: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(labels, dtype=np.int32)
    valid = labels > 0
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    uniq = [int(v) for v in np.unique(labels[valid]) if int(v) > 0]
    for new_id, old_id in enumerate(uniq, start=1):
        out[labels == old_id] = int(new_id)
    return out


def _postprocess_superpixel_labels(
    labels: np.ndarray,
    nspix_hint: int,
    mask: Optional[np.ndarray] = None,
    prune_small_thin: bool = False,
) -> np.ndarray:
    """
    Connectivity-aware cleanup for superpixel labels:
    1. Split disconnected regions into different superpixels.
    2. Optionally drop very small or thin regions to background.
    3. Split again if background removal disconnected regions further.
    4. Relabel sequentially.
    """
    labels = _split_disconnected_superpixels(labels, mask=mask)
    if prune_small_thin:
        labels = _filter_small_and_thin_superpixels(labels, nspix_hint=nspix_hint, mask=mask)
        labels = _split_disconnected_superpixels(labels, mask=mask)
    labels = _relabel_sequential(labels, mask=mask)
    if mask is not None:
        labels = labels.astype(np.int32, copy=False)
        labels[~np.asarray(mask, dtype=bool)] = 0
    return labels


def mark_line_used(
    used_mask: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int = 3,
    pad: int = 2,
) -> None:
    """Mark a scribble segment plus a small safety band as already occupied."""
    H, W = used_mask.shape
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n).round().astype(np.int32)
    ys = np.linspace(y0, y1, n).round().astype(np.int32)
    r = max(0, width // 2) + max(0, pad)
    for x, y in zip(xs, ys):
        x0b = max(0, x - r)
        x1b = min(W, x + r + 1)
        y0b = max(0, y - r)
        y1b = min(H, y + r + 1)
        used_mask[y0b:y1b, x0b:x1b] = True


@dataclass
class LargestBadRegionGenerator:
    """
    Generate a scribble inside the largest connected bad region of a single GT class.

    Unlike the older "largest bad component + dominant class" heuristic, this version
    first splits errors by GT class and then picks the largest class-specific component.
    The line is then traced through the EDT maximum so it stays near the component
    centerline and does not drift into neighbouring class regions.
    """

    gt_mask: np.ndarray
    num_classes: int
    seed: int = 0
    margin: int = 2
    no_overlap: bool = True
    max_retries: int = 200
    center_quantile: float = 0.8
    smoothing_radius: int = 1

    def __post_init__(self) -> None:
        self.gt = np.asarray(self.gt_mask, dtype=np.int32)
        self.H, self.W = self.gt.shape
        self.rng = np.random.default_rng(int(self.seed))
        self.margin = max(0, int(self.margin))
        self.max_retries = max(1, int(self.max_retries))
        self.center_quantile = float(np.clip(self.center_quantile, 0.0, 1.0))
        self.smoothing_radius = max(0, int(self.smoothing_radius))
        self._diag = 0.5 * float(np.hypot(self.W, self.H))
        self._gt_inner: List[np.ndarray] = []
        for cid in range(int(self.num_classes)):
            cls_mask = (self.gt == cid)
            if self.margin > 0 and cls_mask.any():
                dist = _edt_inside(cls_mask)
                inner = cls_mask & (dist > self.margin)
            else:
                inner = cls_mask.copy()
            self._gt_inner.append(inner)

    def _largest_class_component(self, bad_mask: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
        best_cid: Optional[int] = None
        best_comp: Optional[np.ndarray] = None
        best_area = 0
        for cid in range(int(self.num_classes)):
            class_bad = bad_mask & (self.gt == cid)
            if not class_bad.any():
                continue
            labels, nlab = ndimage.label(class_bad)
            if nlab <= 0:
                continue
            counts = np.bincount(labels.ravel())
            if counts.size <= 1:
                continue
            counts[0] = 0
            comp_idx = int(np.argmax(counts))
            comp_area = int(counts[comp_idx])
            if comp_area > best_area:
                best_area = comp_area
                best_cid = cid
                best_comp = (labels == comp_idx)
        return best_cid, best_comp

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

    @staticmethod
    def _nearest_allowed_point(
        allowed_mask: np.ndarray,
        x: float,
        y: float,
    ) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(allowed_mask)
        if xs.size == 0:
            return None
        d2 = (xs.astype(np.float64) - float(x)) ** 2 + (ys.astype(np.float64) - float(y)) ** 2
        idx = int(np.argmin(d2))
        return int(xs[idx]), int(ys[idx])

    def _analysis_region(self, allowed: np.ndarray, comp: np.ndarray) -> np.ndarray:
        focus = allowed if allowed.any() else comp
        if not focus.any():
            return focus.copy()
        smoothed = _smooth_region_mask(focus, radius=self.smoothing_radius)
        labels, nlab = ndimage.label(smoothed)
        if nlab <= 0:
            return focus.copy()
        counts = np.bincount(labels.ravel())
        if counts.size <= 1:
            return focus.copy()
        counts[0] = 0
        comp_idx = int(np.argmax(counts))
        if counts[comp_idx] <= 0:
            return focus.copy()
        return labels == comp_idx

    def _principal_direction(
        self,
        allowed: np.ndarray,
        dist_map: np.ndarray,
        x0: int,
        y0: int,
    ) -> List[Tuple[float, float]]:
        positive = dist_map[allowed]
        if positive.size == 0:
            return [(1.0, 0.0), (0.0, 1.0)]

        cutoff = float(np.quantile(positive, self.center_quantile))
        core = allowed & (dist_map >= cutoff)
        ys, xs = np.where(core)
        if xs.size < 2:
            ys, xs = np.where(allowed)
        if xs.size < 2:
            return [(1.0, 0.0), (0.0, 1.0)]

        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        weights = dist_map[ys, xs].astype(np.float64)
        weights = np.maximum(weights, 1e-6)
        center = np.array([float(x0), float(y0)], dtype=np.float64)
        pts = pts - center[None, :]
        cov = np.cov(pts.T, aweights=weights)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return [(1.0, 0.0), (0.0, 1.0)]

        eigvals, eigvecs = np.linalg.eigh(cov)
        main = eigvecs[:, int(np.argmax(eigvals))]
        dx, dy = float(main[0]), float(main[1])
        norm = float(np.hypot(dx, dy))
        if not np.isfinite(norm) or norm <= 1e-8:
            return [(1.0, 0.0), (0.0, 1.0)]
        dx /= norm
        dy /= norm
        return [
            (dx, dy),
            (-dy, dx),
            (dy, -dx),
            (1.0, 0.0),
            (0.0, 1.0),
        ]

    def _segment_pixels(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(np.int32)
        ys = np.linspace(y0, y1, n).round().astype(np.int32)
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

    def _nearest_mask_point(
        self,
        mask: np.ndarray,
        x: float,
        y: float,
    ) -> Optional[Tuple[int, int]]:
        return self._nearest_allowed_point(mask, x, y)

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

        axis_xy = self._principal_direction(axis_mask, dt_map, center_xy[0], center_xy[1])[0]
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

    def _build_centerline_path(
        self,
        allowed: np.ndarray,
        analysis_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        source_mask = analysis_mask if analysis_mask.any() else allowed
        if not source_mask.any():
            return None

        center_data = self._center_pixel(source_mask)
        if center_data is None:
            return None
        cx, cy, center_dt = center_data
        axis_mask = source_mask if source_mask.any() else allowed
        max_dt = float(center_dt.max())

        skeleton_candidates: List[np.ndarray] = []
        seen_thresholds: set[float] = set()
        for thr in [
            max(1.0, 0.70 * max_dt),
            max(1.0, 0.55 * max_dt),
            max(1.0, 0.40 * max_dt),
            1.0,
        ]:
            key = round(float(thr), 4)
            if key in seen_thresholds:
                continue
            seen_thresholds.add(key)
            ridge_mask = source_mask & (center_dt >= (float(thr) - 1e-6))
            if not ridge_mask.any():
                continue
            skeleton_candidates.append(skeletonize(ridge_mask).astype(bool, copy=False))
        skeleton_candidates.append(skeletonize(source_mask).astype(bool, copy=False))
        skeleton_candidates.append(medial_axis(source_mask).astype(bool, copy=False))

        best_pts: Optional[np.ndarray] = None
        best_score = -1.0
        for skeleton in skeleton_candidates:
            path_choice = self._centerline_path_from_skeleton(
                skeleton=skeleton,
                allowed=allowed,
                center_xy=(cx, cy),
                axis_mask=axis_mask,
                dt_map=center_dt,
            )
            if path_choice is None:
                continue
            pts_px, _, path_len = path_choice
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
            best_pts = pts_px.astype(np.float32, copy=False)
        return best_pts

    def _trace(
        self,
        x0: int,
        y0: int,
        dx: float,
        dy: float,
        allowed: np.ndarray,
        used_mask: np.ndarray,
    ) -> Tuple[int, int]:
        x = float(x0)
        y = float(y0)
        last_x, last_y = int(x0), int(y0)
        for _ in range(int(self._diag)):
            x += float(dx)
            y += float(dy)
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0 or xi >= self.W or yi < 0 or yi >= self.H:
                break
            if not allowed[yi, xi]:
                break
            if self.no_overlap and used_mask[yi, xi]:
                break
            last_x, last_y = xi, yi
        return last_x, last_y

    def _best_segment(
        self,
        x0: int,
        y0: int,
        allowed: np.ndarray,
        used_mask: np.ndarray,
        dist_map: np.ndarray,
    ) -> Optional[np.ndarray]:
        best_pts: Optional[np.ndarray] = None
        best_score = -1.0
        directions = self._principal_direction(allowed, dist_map, x0, y0)
        for dx, dy in directions:
            x1a, y1a = self._trace(x0, y0, +dx, +dy, allowed, used_mask)
            x1b, y1b = self._trace(x0, y0, -dx, -dy, allowed, used_mask)
            total_len = float(np.hypot(x1a - x1b, y1a - y1b))
            arm_a = float(np.hypot(x1a - x0, y1a - y0))
            arm_b = float(np.hypot(x1b - x0, y1b - y0))
            balance = float(min(arm_a, arm_b) / max(1e-6, max(arm_a, arm_b)))
            cur_score = total_len + 0.75 * balance
            if cur_score <= best_score:
                continue
            best_score = cur_score
            best_pts = np.array([[x1b, y1b], [x1a, y1a]], dtype=np.float32)
        return best_pts if best_pts is not None and best_score > 0.0 else None

    def make_scribble(self, pred_mask: np.ndarray, used_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        bad_mask = (np.asarray(pred_mask, dtype=np.int32) != self.gt)
        if not bad_mask.any():
            raise StopIteration("All pixels correctly labelled.")

        cid, comp = self._largest_class_component(bad_mask)
        if cid is None or comp is None or not comp.any():
            raise StopIteration("No bad connected component.")

        if self.margin > 0:
            comp_dist = _edt_inside(comp)
            comp_inner = comp & (comp_dist > self.margin)
        else:
            comp_inner = comp

        allowed = comp_inner & self._gt_inner[cid] & bad_mask
        if self.no_overlap:
            allowed &= ~used_mask

        if not allowed.any():
            allowed = comp & self._gt_inner[cid] & bad_mask
            if self.no_overlap:
                allowed &= ~used_mask
        if not allowed.any():
            allowed = comp & bad_mask
            if self.no_overlap:
                allowed &= ~used_mask
        if not allowed.any():
            raise RuntimeError(
                "Cannot place scribble in largest bad region of the target class. "
                "Try --margin 0 or remove --no_overlap."
            )

        analysis_mask = self._analysis_region(allowed, comp)
        center_data = self._center_pixel(analysis_mask if analysis_mask.any() else allowed)
        if center_data is None:
            raise RuntimeError("Failed to find a center pixel for the bad region.")
        x0, y0, dist_map = center_data
        projected_center = self._nearest_allowed_point(allowed, x0, y0)
        if projected_center is None:
            raise RuntimeError("Failed to project the scribble center to the allowed region.")
        x0, y0 = projected_center

        pts = self._build_centerline_path(allowed, analysis_mask)
        if pts is None:
            pts = self._best_segment(x0, y0, allowed, np.asarray(used_mask, dtype=bool), dist_map)
        if pts is None:
            ys, xs = np.where(allowed)
            for _ in range(self.max_retries):
                idx = int(self.rng.integers(0, xs.size))
                x0 = int(xs[idx])
                y0 = int(ys[idx])
                pts = self._best_segment(
                    x0,
                    y0,
                    allowed,
                    np.asarray(used_mask, dtype=bool),
                    _edt_inside(allowed),
                )
                if pts is not None:
                    break
        if pts is None:
            raise RuntimeError("Failed to generate scribble after max_retries.")

        pts01 = np.empty_like(pts, dtype=np.float32)
        pts01[:, 0] = pts[:, 0] / float(self.W)
        pts01[:, 1] = pts[:, 1] / float(self.H)
        return int(cid), pts01


def render_annotation_snapshot(
    img: Image.Image,
    algo: "SuperPixelAnnotationAlgo",
    sp_method: SuperPixelMethod,
    class_info: Sequence[Tuple[str, str]],
    out_png: Path | str,
    draw_borders: bool = True,
    draw_annos: bool = True,
    draw_scribbles: bool = True,
    anno_alpha: int = 110,
) -> None:
    """Render annotations with class colors and optional superpixel borders/scribbles."""
    base = img.convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    if draw_annos:
        ann_obj = algo._annotations.get(sp_method)
        if ann_obj is not None:
            for anno in ann_obj.annotations:
                border = np.asarray(anno.border, dtype=np.float32)
                if border.ndim != 2 or border.shape[0] < 3:
                    continue
                poly = [(float(x * W), float(y * H)) for x, y in border]
                gt_id = max(0, min(int(anno.code) - 1, len(class_info) - 1))
                fill = _hex_to_rgba_tuple(class_info[gt_id][1], int(anno_alpha))
                draw.polygon(poly, fill=fill)

    if draw_borders:
        for sp in algo.superpixels.get(sp_method, []):
            border = np.asarray(sp.border, dtype=np.float32)
            if border.ndim != 2 or border.shape[0] < 3:
                continue
            poly = [(float(x * W), float(y * H)) for x, y in border]
            draw.polygon(poly, outline=(255, 255, 0, 255))

    composite = Image.alpha_composite(base, overlay)
    if draw_scribbles:
        draw_scr = ImageDraw.Draw(composite)
        for scribble in getattr(algo, "_scribbles", []):
            pts = np.asarray(scribble.points, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            line_pts = [(float(x * W), float(y * H)) for x, y in pts]
            gt_id = max(0, min(int(scribble.params.code) - 1, len(class_info) - 1))
            line_color = _hex_to_rgba_tuple(class_info[gt_id][1], 255)
            draw_scr.line(line_pts, fill=line_color, width=5)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(str(out_path), quality=95)


# ── Superpixel computation ─────────────────────────────────────────────────────
def compute_superpixels(
    image_lab: np.ndarray,
    method: SuperPixelMethod,
    n_segments_hint: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    embedding_guided_cleanup: bool = False,
) -> np.ndarray:
    """
    Возвращает labels (H,W) int32, где 0 — фон (если mask задан), валидные метки начинаются с 1.
    image_lab — float ndarray (H,W,3) в Lab.
    """
    H, W = image_lab.shape[:2]
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.int32)
    num_pixel_in_roi = H * W
    sp_size = 1400 * 1400 / 500
    n_segment_dynamic = num_pixel_in_roi // sp_size
    has_mask = mask is not None
    if has_mask:
        mask = mask.astype(bool)
        if not np.any(mask):
            return np.zeros((H, W), dtype=np.int32)

    # --- SLIC ---
    if isinstance(method, SLICSuperpixel):
        n_segments = int(n_segments_hint or method.n_clusters or 2)
        n_segments = max(2, n_segments)
        if _HAS_FAST_SLIC:
            # lab2rgb -> [0,1] -> u8
            rgb = (lab2rgb(image_lab) * 255.0).astype(np.uint8)
            rgb = _maybe_gaussian(rgb, method.sigma)
            Engine = FastSlicAvx2 if (FastSlicAvx2 is not None) else FastSlic
            slic_engine = Engine(num_components=n_segment_dynamic, compactness=float(method.compactness))
            labels0 = slic_engine.iterate(rgb).astype(np.int32)  # 0..K-1
            labels = labels0 + 1
            if has_mask:
                out = np.zeros_like(labels, dtype=np.int32)
                out[mask] = labels[mask]
                labels = out
        else:
            labels = sk_slic(
                image_lab,
                n_segments=int(n_segment_dynamic),
                compactness=float(method.compactness),
                sigma=float(method.sigma),
                start_label=1,
                mask=mask if has_mask else None,
            ).astype(np.int32)
        return labels

    # --- Felzenszwalb ---
    if isinstance(method, FelzenszwalbSuperpixel):
        labels = sk_fz(
            image_lab, scale=float(method.scale),
            sigma=float(method.sigma), min_size=int(method.min_size),
        ).astype(np.int32)
        labels += 1
        if has_mask:
            labels[~mask] = 0
        return labels

    # --- Watershed ---
    if isinstance(method, WatershedSuperpixel):
        elev = sobel(lab2rgb(image_lab).astype(np.float32))
        labels = sk_ws(
            elev, markers=int(n_segment_dynamic),
            compactness=float(method.compactness),
        ).astype(np.int32)
        labels += 1
        if has_mask:
            labels[~mask] = 0
        return labels

    # --- SSN (Superpixel Sampling Network) ---
    if isinstance(method, SSNSuperpixel):
        import math as _math
        import torch as _torch

        device = get_torch_device(_torch)
        model = _get_ssn_model(
            method.weight_path,
            method.fdim,
            n_segment_dynamic,
            method.niter,
            device=device,
        )

        # LAB image → float32 tensor (1, 3, H, W)
        img_t = (
            _torch.from_numpy(image_lab.astype(np.float32))
            .permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # Coordinate grid (1, 2, H, W) — plain pixel indices
        nspix_per_axis = int(_math.sqrt(n_segment_dynamic))
        ps = method.pos_scale * max(nspix_per_axis / H, nspix_per_axis / W)
        coords = _torch.stack(
            _torch.meshgrid(
                _torch.arange(H, device=device, dtype=_torch.float32),
                _torch.arange(W, device=device, dtype=_torch.float32),
                indexing="ij",
            ), 0
        ).unsqueeze(0)  # (1, 2, H, W)

        model_input = _torch.cat(
            [method.color_scale * img_t, ps * coords], dim=1
        )  # (1, 5, H, W)

        with _torch.no_grad():
            _, hard_labels, _ = model(model_input)
        # hard_labels: (1, H*W) — 0-indexed superpixel IDs
        labels = (
            hard_labels.reshape(H, W).cpu().numpy().astype(np.int32) + 1
        )  # shift to 1-indexed (0 = background/masked)

        if has_mask:
            labels[~mask] = 0
        return _postprocess_superpixel_labels(
            labels,
            nspix_hint=max(2, int(method.nspix)),
            mask=mask if has_mask else None,
            # For SSN we always want tiny / thread-like fragments to collapse
            # to background after connectivity splitting.
            prune_small_thin=True,
        )

    raise TypeError(f"Unsupported superpixel type: {type(method)}")


# ── Orchestrator ───────────────────────────────────────────────────────────────
@dataclass
class _CandResult:
    superpixel_ind_to_del: List[int]
    annotations_to_del: List[int]
    superpixel_to_append: List[SuperPixel]
    scribbles_to_check: List[int]
    new_annotations: List[AnnotationInstance]
    update_existing_annos: List[Tuple[int, Dict]]


class SuperPixelAnnotationAlgo:
    """Высокоуровневый менеджер суперпикселей/аннотаций и конфликтов."""

    # ---- Construction ----
    def __init__(
        self,
        downscale_coeff: float,
        superpixel_methods: List[SuperPixelMethod],
        image_path: Optional[Path] = None,
        image: Optional[Image.Image] = None,
        auto_propagation_sensitivity: float = 1.0,
    ) -> None:
        assert 0 < downscale_coeff <= 1.0
        self.downscale_coeff = float(downscale_coeff)  # сохраняем для метаданных
        self.debug_candidates_dir = "./_debug_candidates"
        self.image_path = image_path
        self.superpixel_methods: List[SuperPixelMethod] = list(superpixel_methods)
        self._sp_to_anno_idx: Dict[SuperPixelMethod, Dict[int, int]] = {
            m: {} for m in self.superpixel_methods
        }
        if image is not None:
            self.image = image.convert("RGB")
        else:
            if image_path is None:
                raise ValueError("Either image or image_path must be provided")
            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Can't read image: {image_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).ascontiguousarray()
            self.image = Image.fromarray(img_rgb, mode="RGB")

        self._preprocess_image(downscale_coeff)

        self.superpixels: Dict[SuperPixelMethod, List[SuperPixel]] = {}
        self._annotations: Dict[SuperPixelMethod, ImageAnnotation] = {}
        self._superpixel_ind: Dict[SuperPixelMethod, int] = {}
        self._annotation_ind: Dict[SuperPixelMethod, int] = {}

        # spatial indices / caches
        self._sp_index_dirty: bool = True
        self._sp_polys: List[Polygon] | None = None
        self._sp_prepared: List[prepared.PreparedGeometry] | None = None
        self._sp_centroids: Optional[np.ndarray] = None
        self._sp_kdtree: Optional[cKDTree] = None

        # user interaction state
        self.ind_scrible = 0
        self.prev_ind_scrible = 0
        self._scribbles: List[Scribble] = []
        self.scribbles_id_sequence: List[int] = []

        # already annotated regions (list of normalized bbox)
        self.annotated_bbox: List[List[float]] = []
        self.bbox_size: int = 700
        self.refine_bbox_size: int = 256
        self._property_dist = 5.0
        self._superpixel_radius = 0.08  # in normalized coords (KD radius)
        self.auto_propagation_sensitivity = float(auto_propagation_sensitivity)

        # ── Embedding-based propagation (optional) ────────────────────────────
        # Set embedding_weight_path to enable per-superpixel embedding computation.
        # When set, use_sensitivity_for_region uses cosine similarity instead of
        # LAB property distance for the feature-similarity gate.
        self.embedding_weight_path: Optional[str] = None
        self.embedding_fdim: int = 20
        self.embedding_color_scale: float = 0.26
        self.embedding_pos_scale: float = 2.5
        self._embedding_threshold: float = 0.99   # minimum cosine similarity to propagate
        self._pixel_embedding_cache: Dict[Tuple[str, int, float, float], np.ndarray] = {}

        self._create_superpixels()
        self._mark_sp_index_dirty()

    # ---- Spatial index maintenance ----
    def _mark_sp_index_dirty(self):
        self._sp_index_dirty = True

    def _ensure_spatial_index(self, sp_method: Optional[SuperPixelMethod] = None):
        if not self.superpixel_methods:
            return
        if sp_method is None:
            sp_method = self.superpixel_methods[0]
        if not self._sp_index_dirty and self._sp_polys is not None and self._sp_kdtree is not None:
            return

        sps = self.superpixels.get(sp_method, [])
        self._sp_polys = [sp.poly for sp in sps]
        self._sp_prepared = [sp.prepared_poly for sp in sps]
        if len(sps) > 0:
            self._sp_centroids = np.array([sp.centroid_xy for sp in sps], dtype=np.float64)
            self._sp_kdtree = cKDTree(self._sp_centroids)
        else:
            self._sp_centroids = None
            self._sp_kdtree = None
        self._sp_index_dirty = False

    # ---- Imaging ----
    def _preprocess_image(self, downscale_coeff: float) -> None:
        # Ensure image is in RGB mode first (handles RGBA, grayscale, etc.)
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')

        # Apply downscaling if needed
        if downscale_coeff < 1.0:
            w, h = self.image.size
            new_w = int(max(1, w * downscale_coeff))
            new_h = int(max(1, h * downscale_coeff))
            self.image = self.image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Convert PIL image to float RGB and then to LAB
        rgb = np.array(self.image, dtype=np.float32) / 255.0
        self.image_lab = rgb2lab(rgb).astype(np.float32)

    # ---- Methods management ----
    def add_superpixel_method(self, superpixel_method: SuperPixelMethod) -> None:
        if any(m.short_string() == superpixel_method.short_string() for m in self.superpixel_methods):
            logger.debug("Method already exists: %s", superpixel_method.short_string())
            return
        self._ensure_embedding_defaults_for_method(superpixel_method)
        self.superpixel_methods.append(superpixel_method)
        self._create_superpixel(superpixel_method)
        self._annotations[superpixel_method] = ImageAnnotation(annotations=[])
        logger.info("Added method %s (methods=%d)", superpixel_method.short_string(), len(self.superpixel_methods))

    def _ensure_embedding_defaults_for_method(self, sp_method: SuperPixelMethod) -> None:
        """
        When SSN is the active method, default propagation should use the same
        checkpoint for per-superpixel mean embeddings unless the user
        explicitly configured a different embedding model.
        """
        if not isinstance(sp_method, SSNSuperpixel):
            return
        if self.embedding_weight_path:
            return
        self.embedding_weight_path = os.path.abspath(sp_method.weight_path)
        self.embedding_fdim = int(sp_method.fdim)
        self.embedding_color_scale = float(sp_method.color_scale)
        self.embedding_pos_scale = float(sp_method.pos_scale)

    # ---- Superpixel generation around scribble ----
    def _exist_sp_mask(self, scribble: Scribble) -> bool:
        return check_bbox_contain_scribble(scribble.points, self.annotated_bbox)

    def _create_superpixel_for_mask(
        self,
        superpixel_method: SuperPixelMethod,
        image_roi: np.ndarray,
        mask: np.ndarray,
        bbox: List[float],
        scribble: Scribble,  # kept for future hooks
        forbidden_bboxes: Optional[List[List[float]]] = None,
    ) -> int:
        self._ensure_embedding_defaults_for_method(superpixel_method)
        if image_roi.size == 0 or mask.size == 0 or not np.any(mask):
            return 0
        sp_mask = compute_superpixels(
            image_roi,
            superpixel_method,
            mask=mask,
            embedding_guided_cleanup=bool(self.embedding_weight_path),
        )
        if sp_mask.size == 0:
            return 0
        means, variances, valid_labels = parallel_stats_rgb(
            image_roi.astype(np.float32, copy=False), sp_mask, int(np.max(sp_mask))
        )
        polys = labels_to_polygons(sp_mask, tuple(bbox), self.image_lab.shape[0], self.image_lab.shape[1], valid_labels)

        # ── Per-label mean embeddings (optional) ─────────────────────────────
        label_embs: dict = {}
        if self.embedding_weight_path:
            try:
                pixel_emb = _compute_roi_embeddings(
                    image_roi,
                    self.embedding_weight_path,
                    fdim=self.embedding_fdim,
                    color_scale=self.embedding_color_scale,
                    pos_scale=self.embedding_pos_scale,
                )  # (H_roi, W_roi, fdim)
                for lab in valid_labels:
                    if lab <= 0:
                        continue
                    lmask = (sp_mask == int(lab))
                    if not lmask.any():
                        continue
                    raw_mean = pixel_emb[lmask].mean(axis=0)  # (fdim,)
                    norm = np.linalg.norm(raw_mean)
                    label_embs[int(lab)] = (raw_mean / max(norm, 1e-8)).astype(np.float32)
            except Exception as _emb_err:
                logger.warning("Embedding computation failed (skipping emb for this ROI): %s", _emb_err)

        allowed_geom = None
        if forbidden_bboxes:
            try:
                x0, y0, x1, y1 = _normalize_bbox01(tuple(bbox))
                roi_poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                forbid_polys = []
                for fb in forbidden_bboxes:
                    fx0, fy0, fx1, fy1 = _normalize_bbox01(tuple(fb))
                    if fx1 <= fx0 or fy1 <= fy0:
                        continue
                    forbid_polys.append(Polygon([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)]))
                if forbid_polys:
                    allowed_geom = roi_poly.difference(unary_union(forbid_polys))
            except Exception as clip_err:
                logger.warning("Failed to build allowed ROI geometry: %s", clip_err)
                allowed_geom = None

        created_count = 0
        for i, lab in enumerate(valid_labels):
            if lab <= 0:
                continue
            polygon = polys.get(int(lab))
            if polygon is None or len(polygon) < 3:
                continue
            if allowed_geom is not None:
                try:
                    clipped = Polygon(polygon).intersection(allowed_geom)
                    clipped_poly = _largest_polygon(_sanitize_polygon(clipped))
                except Exception as clip_err:
                    logger.warning("Failed to clip ROI polygon to allowed bbox area: %s", clip_err)
                    continue
                if clipped_poly is None or clipped_poly.is_empty or clipped_poly.area < 1e-8:
                    continue
                polygon = np.asarray(clipped_poly.exterior.coords, dtype=np.float32)
                if polygon.ndim != 2 or polygon.shape[0] < 3:
                    continue
            region_props = []
            region_props.extend(means[i])
            region_props.extend(variances[i])
            self.superpixels[superpixel_method].append(
                SuperPixel(
                    id=self._superpixel_ind[superpixel_method],
                    method=superpixel_method.short_string(),
                    border=np.asarray(polygon, dtype=np.float32),
                    parents=[],
                    props=np.array(region_props, dtype=np.float32),
                    emb=label_embs.get(int(lab)),
                )
            )
            self._superpixel_ind[superpixel_method] += 1
            created_count += 1
        self._mark_sp_index_dirty()
        return created_count

    def _rasterize_scribble_to_roi_mask(
        self,
        scribble: Scribble,
        roi: Tuple[int, int, int, int],
        *,
        width_px: int,
    ) -> Optional[np.ndarray]:
        if len(scribble.points) < 2:
            return None
        x0, y0, x1, y1 = roi
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return None
        H, W = self.image_lab.shape[:2]
        pts = np.asarray(scribble.points, dtype=np.float32)
        pts_px = np.empty_like(pts)
        pts_px[:, 0] = pts[:, 0] * W - x0
        pts_px[:, 1] = pts[:, 1] * H - y0
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        draw.line([tuple(p) for p in pts_px], fill=1, width=max(1, int(width_px)))
        return np.array(canvas, dtype=bool)

    def _create_superpixel_for_scribble(self, scribble: Scribble, superpixel_method: SuperPixelMethod) -> None:
        self._ensure_embedding_defaults_for_method(superpixel_method)
        if not find_sp_key_in_dict(superpixel_method, self._superpixel_ind):
            self._superpixel_ind[superpixel_method] = 0
        if not find_sp_key_in_dict(superpixel_method, self._annotations):
            self._annotations[superpixel_method] = ImageAnnotation(annotations=[])
        if not find_sp_key_in_dict(superpixel_method, self.superpixels):
            self.superpixels[superpixel_method] = []

        if len(scribble) < 2:
            return
        already_covered = self._exist_sp_mask(scribble)

        # Allow local refinement even inside an old bbox. In that case we use a
        # smaller ROI and avoid subtracting the containing bbox, otherwise later
        # scribbles often fail to create any new local superpixels at all.
        H, W = self.image_lab.shape[:2]
        pts = scribble.points
        bbox = [float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max())]
        local_bbox_size = self.refine_bbox_size if already_covered else self.bbox_size
        pad_x = 1.0 * local_bbox_size / (2 * W)
        pad_y = 1.0 * local_bbox_size / (2 * H)
        bbox = list(_normalize_bbox01((
            max(bbox[0] - pad_x, 0.0),
            max(bbox[1] - pad_y, 0.0),
            min(bbox[2] + pad_x, 1.0),
            min(bbox[3] + pad_y, 1.0),
        )))

        x0, y0, x1, y1 = bbox
        ix0, iy0, ix1, iy1 = _bbox_to_pixel_rect(tuple(bbox), W, H)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        roi_px = (ix0, iy0, ix1, iy1)
        image_roi = self.image_lab[iy0:iy1, ix0:ix1]
        mask = np.ones((image_roi.shape[0], image_roi.shape[1]), dtype=bool)
        scribble_line = LineString(scribble.points)
        forbidden_bboxes: List[List[float]] = []

        # вычитаем пересечение уже размеченных bbox
        for existed_bbox in self.annotated_bbox:
            existed_bbox = list(_normalize_bbox01(tuple(existed_bbox)))
            existed_poly = Polygon([
                (existed_bbox[0], existed_bbox[1]),
                (existed_bbox[2], existed_bbox[1]),
                (existed_bbox[2], existed_bbox[3]),
                (existed_bbox[0], existed_bbox[3]),
            ])
            if already_covered and existed_poly.covers(scribble_line):
                continue
            if not bbox_is_intersect(tuple(existed_bbox), tuple(bbox)):
                continue
            ex_ix0, ex_iy0, ex_ix1, ex_iy1 = _bbox_to_pixel_rect(tuple(existed_bbox), W, H)
            jx0 = max(ix0, ex_ix0) - ix0
            jy0 = max(iy0, ex_iy0) - iy0
            jx1 = min(ix1, ex_ix1) - ix0
            jy1 = min(iy1, ex_iy1) - iy0
            jx0, jy0 = max(jx0, 0), max(jy0, 0)
            jx1, jy1 = min(jx1, image_roi.shape[1]), min(jy1, image_roi.shape[0])
            if jx1 > jx0 and jy1 > jy0:
                mask[jy0:jy1, jx0:jx1] = 0
                forbidden_bboxes.append(existed_bbox)

        if already_covered:
            scribble_radius = int(getattr(scribble.params, "radius", 1))
            scribble_mask = self._rasterize_scribble_to_roi_mask(
                scribble,
                roi_px,
                width_px=max(3, 2 * scribble_radius + 1),
            )
            if scribble_mask is not None:
                mask |= scribble_mask


        created_count = self._create_superpixel_for_mask(
            superpixel_method,
            image_roi,
            mask,
            bbox,
            scribble,
            forbidden_bboxes=forbidden_bboxes if not already_covered else None,
        )
        if created_count > 0:
            self.annotated_bbox.append(bbox)

    def _create_superpixel(self, superpixel_method: SuperPixelMethod) -> None:
        self._superpixel_ind[superpixel_method] = 0
        self.superpixels[superpixel_method] = []
        self._annotation_ind[superpixel_method] = 0

    def _create_superpixels(self) -> None:
        for m in self.superpixel_methods:
            self._create_superpixel(m)
            self._annotations[m] = ImageAnnotation(annotations=[])

    # ---- Undo ----
    def cancel_prev_act(self):
        if not self.scribbles_id_sequence or not self._scribbles:
            logger.warning("Nothing to cancel")
            return
        scribble_to_del = self.scribbles_id_sequence.pop()

        for method in list(self._annotations.keys()):
            anno_to_del: List[int] = []
            for a_idx, anno in enumerate(self._annotations[method].annotations):
                if scribble_to_del in (anno.parent_scribble or []):
                    anno.parent_scribble = [sid for sid in anno.parent_scribble if sid != scribble_to_del]
                    if len(anno.parent_scribble) == 0:
                        anno_to_del.append(a_idx)
            for j in sorted(anno_to_del, reverse=True):
                self._annotations[method].annotations.pop(j)

        # удаляем последний добавленный штрих (LIFO)
        last = self._scribbles[-1]
        if last.id == scribble_to_del:
            self._scribbles.pop()
        else:
            self._scribbles = [s for s in self._scribbles if s.id != scribble_to_del]

    # ── ВНУТРИ класса SuperPixelAnnotationAlgo ────────────────────────────────────
    def clear_existing_data(self, *, keep_image: bool = True) -> None:
        """
        Полный сброс оперативного состояния перед загрузкой разметки.
        По умолчанию сохраняем self.image_lab (данные изображения) и набор методов.
        """
        # сохранить то, что должно пережить сброс
        saved_image_lab = getattr(self, "image_lab", None) if keep_image else None
        saved_methods   = list(getattr(self, "superpixel_methods", [])) if hasattr(self, "superpixel_methods") else []

        # контейнеры данных
        self.superpixels: dict = {}
        self._annotations: dict = {}
        self._superpixel_ind: dict = {}
        self._annotation_ind: dict = {}

        for m in saved_methods:
            self.superpixels[m]      = []
            self._annotations[m]     = ImageAnnotation(annotations=[])
            self._superpixel_ind[m]  = 0
            self._annotation_ind[m]  = 0

        # штрихи и вспомогательные структуры
        self._scribbles: list[Scribble] = []
        self.annotated_bbox: list[list[float]] = []

        # индексы/кэши/флаги
        self._spatial_index = {}                  # любые R-деревья/индексы по SP
        self._spatial_index_dirty = True
        self._prepared_scr_geoms = {}             # буферизованные геометрии штрихов
        self._prepared_sp_polys = {}              # prepared-polygons для SP
        self._sp_index_version = 0                # инкрементировать при любых изменениях SP

        # вернуть картинку при необходимости
        if keep_image:
            self.image_lab = saved_image_lab

    # Алиас для старого кода, который звал приватную версию
    def _clear_existing_data(self, *args, **kwargs) -> None:
        return self.clear_existing_data(*args, **kwargs)

    
    # ── (De)serialization ───────────────────────────────────────────────────────
    def _to_state_dict(self) -> Dict[str, Any]:
        """Готовит словарь состояния (включая мета/версию), БЕЗ кешей."""
        methods = [m.short_string() for m in self.superpixel_methods]
        sp_map = {
            m.short_string(): [sp.dict_to_save() for sp in self.superpixels.get(m, [])]
            for m in self.superpixel_methods
        }
        ann_map = {
            m.short_string(): [a.dict_to_save() for a in self._annotations.get(m, ImageAnnotation([])).annotations]
            for m in self.superpixel_methods
        }
        scribbles = [s.dict_to_save() for s in self._scribbles]
        bbox = [[float(y) for y in x] for x in self.annotated_bbox]
        img_w, img_h = self.image.size
        meta = {
            "magic": SCHEMA_MAGIC,
            "version": SCHEMA_VERSION,
            "image": {
                "path": str(self.image_path) if self.image_path is not None else None,
                "size_wh": [int(img_w), int(img_h)],
                "lab_shape": list(map(int, self.image_lab.shape)),
                "downscale_coeff": float(self.downscale_coeff),
            },
            "methods": methods,
            "checks": {
                "n_scribbles": len(scribbles),
                "n_superpixels": {k: len(v) for k, v in sp_map.items()},
                "n_annotations": {k: len(v) for k, v in ann_map.items()},
            },
        }
        payload = {
            "scribbles": scribbles,
            "superpixels": sp_map,
            "annotations": ann_map,
            "bbox": bbox,
        }
        return {"_meta": meta, **payload}

    def _validate_loaded(self, root: Dict[str, Any]) -> None:
        """Быстрая структурная проверка, бросает LoadError."""
        def _fail(msg: str):
            raise LoadError(msg)

        if "_meta" in root:
            meta = root["_meta"] or {}
            if meta.get("magic") != SCHEMA_MAGIC:
                _fail("Invalid magic or file type.")
            ver = int(meta.get("version", 0))
            if ver > SCHEMA_VERSION:
                _fail(f"Unsupported future schema version: {ver} > {SCHEMA_VERSION}")

        superpixels = root.get("superpixels", {})
        annotations = root.get("annotations", {})
        try:
            _scrib_id_set = {int(s.get("id")) for s in (root.get("scribbles") or []) if isinstance(s, dict) and "id" in s}
        except Exception:
            _scrib_id_set = set()
        if not isinstance(superpixels, dict) or not isinstance(annotations, dict):
            _fail("Invalid maps for 'superpixels'/'annotations'.")

        methods = sorted(set(list(superpixels.keys()) + list(annotations.keys())))
        for m in methods:
            sps = superpixels.get(m, [])
            ann = annotations.get(m, [])
            sp_ids = set()
            for sp in sps:
                if "id" not in sp or "border" not in sp:
                    _fail(f"SP missing fields in method '{m}'.")
                if sp["id"] in sp_ids:
                    _fail(f"Duplicate superpixel id {sp['id']} in method '{m}'.")
                sp_ids.add(int(sp["id"]))
                br = sp["border"]
                if not isinstance(br, list) or len(br) < 3:
                    _fail(f"SP border degenerate (len<3) in method '{m}', id={sp['id']}.")

            for a in ann:
                if "parent_superpixel" not in a:
                    _fail(f"Annotation missing parent_superpixel in method '{m}'.")
                pid = int(a["parent_superpixel"])
                if pid not in sp_ids:
                    _fail(f"Annotation references unknown SP id={pid} in method '{m}'.")
                # optional: validate parent_scribble ids if present
                ps_list = a.get('parent_scribble') or []
                for _sid in ps_list:
                    try:
                        _sid_i = int(_sid)
                    except Exception:
                        _fail(f"Annotation has non-integer scribble id in method '{m}'.")
                        continue
                    if _scrib_id_set and _sid_i not in _scrib_id_set:
                        _fail(f"Annotation references unknown scribble id={_sid_i} in method '{m}'.")

    def _migrate_if_needed(self, root: Dict[str, Any]) -> Dict[str, Any]:
        """Принимает старые файлы без _meta/magic/version, возвращает нормализованный словарь."""
        if "_meta" in root:
            return root  # уже новая схема
        meta = {
            "magic": SCHEMA_MAGIC,
            "version": 1,
            "image": {
                "path": str(self.image_path) if self.image_path is not None else None,
                "size_wh": [int(self.image.size[0]), int(self.image.size[1])],
                "lab_shape": list(map(int, getattr(self, "image_lab", np.zeros((1, 1, 3))).shape)),
                "downscale_coeff": float(getattr(self, "downscale_coeff", 1.0)),
            },
            "methods": sorted(list((root.get("superpixels") or {}).keys())),
            "checks": {},
        }
        return {"_meta": meta, **root}

    def serialize(self, path: str, make_backup: bool = True, pretty: bool = False) -> None:
        """
        Сохраняет состояние в .json или .json.gz (по расширению). Атомарно.
        Оставлен старый метод-имя для совместимости.
        """
        state = self._to_state_dict()
        try:
            self._validate_loaded(state)
        except LoadError as e:
            logger.warning("Self-validation before save failed: %s", e)

        is_gz = _is_gzip_path(path)
        tmp_path, fp = _open_text_for_write_atomic(path, is_gz)
        try:
            if pretty:
                json.dump(state, fp, ensure_ascii=False, indent=2)
            else:
                json.dump(state, fp, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            fp.close()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise SaveError(f"Serialization error: {e}") from e
        finally:
            try:
                fp.close()
            except Exception:
                pass

        try:
            if make_backup:
                _rotate_backup(path)
            os.replace(tmp_path, path)
        except Exception as e:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise SaveError(f"Atomic replace failed: {e}") from e
        logger.info("Saved state → %s", path)

    def save(self, path: str, **kwargs) -> None:
        """Синоним serialize()."""
        return self.serialize(path, **kwargs)

    def deserialize(self, path: str, strict: bool = True) -> None:
        """
        Загружает .json или .json.gz, валидирует, мигрирует при необходимости.
        strict=True → бросает LoadError при любой несостыковке.
        """


        # IMPORTANT: deserialize() is a clean load, not a merge.
        # If we merge over an already-initialized instance (e.g. test script created methods),
        # SuperPixelMethod keys in dicts stop matching and label propagation/sensitivity
        # works only for "new" superpixels created after load.
        self.superpixel_methods = []
        self.superpixels = {}
        self._superpixel_ind = {}
        self._annotations = {}
        self._annotation_ind = {}
        self._scribbles = []
        self.scribbles_id_sequence = []
        self.ind_scrible = 0
        self.annotated_bbox = []

        # spatial index caches (rebuilt lazily)
        self._sp_kdtree = None
        self._sp_centroids = None
        self._sp_polys = None
        self._sp_prepared = None
        self._sp_bbox = None

        try:
            with _open_text_for_read(path) as f:
                loaded = json.load(f)
        except FileNotFoundError as e:
            raise LoadError(f"File not found: {path}") from e
        except json.JSONDecodeError as e:
            raise LoadError(f"Invalid JSON: {e}") from e
        except OSError as e:
            raise LoadError(f"Open error: {e}") from e

        try:
            normalized = self._migrate_if_needed(loaded)
            self._validate_loaded(normalized)
        except LoadError:
            if strict:
                raise
            logger.warning("Loaded with warnings; trying to continue.")
            normalized = self._migrate_if_needed(loaded)

        self._clear_existing_data()

        sp_map = normalized.get("superpixels", {})
        ann_map = normalized.get("annotations", {})
        method_names = sorted(set(list(sp_map.keys()) + list(ann_map.keys())))
        method_objs: Dict[str, SuperPixelMethod] = {}
        for mstr in method_names:
            try:
                mobj = self._parse_method_from_string(mstr)
                method_objs[mstr] = mobj
                self.superpixel_methods.append(mobj)
                self._superpixel_ind[mobj] = 0
                self._annotation_ind[mobj] = 0
                self.superpixels[mobj] = []
                self._annotations[mobj] = ImageAnnotation(annotations=[])
            except Exception as e:
                if strict:
                    raise LoadError(f"Failed to parse method '{mstr}': {e}") from e
                logger.warning("Skip unknown method '%s': %s", mstr, e)

        # суперпиксели
        for mstr, arr in sp_map.items():
            mobj = method_objs.get(mstr)
            if mobj is None:
                continue
            sps = [SuperPixel.from_dict(sp) for sp in (arr or [])]
            self.superpixels[mobj] = sps
            for sp in sps:
                self._superpixel_ind[mobj] = max(self._superpixel_ind[mobj], int(sp.id) + 1)

        # аннотации
        for mstr, arr in ann_map.items():
            mobj = method_objs.get(mstr)
            if mobj is None:
                continue
            annos = [AnnotationInstance.from_dict(a) for a in (arr or [])]
            self._annotations[mobj] = ImageAnnotation(annotations=annos)
            for a in annos:
                self._annotation_ind[mobj] = max(self._annotation_ind[mobj], int(a.id) + 1)

        # штрихи/bbox
        self._scribbles = [Scribble.from_dict(s) for s in normalized.get("scribbles", [])]
        self.scribbles_id_sequence = [int(s.id) for s in self._scribbles]
        self.annotated_bbox = [[float(x) for x in y] for y in normalized.get("bbox", [])]

        # восстановим image_path если в meta присутствует
        try:
            meta = normalized.get("_meta", {})
            imeta = meta.get("image", {}) if isinstance(meta, dict) else {}
            if self.image_path is None and imeta.get("path"):
                self.image_path = Path(imeta["path"])
        except Exception:
            pass

        self._mark_sp_index_dirty()
        self._assert_annotation_invariants()
        logger.info("Loaded state ← %s (methods=%d, SP=%d, annos=%d, scribbles=%d)",
                    path,
                    len(self.superpixel_methods),
                    sum(len(v) for v in self.superpixels.values()),
                    sum(len(v.annotations) for v in self._annotations.values()),
                    len(self._scribbles))

    def load(self, path: str, **kwargs) -> None:
        """Синоним deserialize()."""
        return self.deserialize(path, **kwargs)

    @staticmethod
    def _parse_method_from_string(method_str: str) -> SuperPixelMethod:
        # SSN strings contain "|" to separate numeric params from the path.
        if method_str.startswith("SSN_") and "|" in method_str:
            try:
                pipe = method_str.index("|")
                prefix_parts = method_str[:pipe].split("_")
                # prefix_parts: ["SSN", nspix, fdim, niter, color_scale, pos_scale]
                weight_path = method_str[pipe + 1:]
                return SSNSuperpixel(
                    nspix=int(prefix_parts[1]),
                    fdim=int(prefix_parts[2]),
                    niter=int(prefix_parts[3]),
                    color_scale=float(prefix_parts[4]),
                    pos_scale=float(prefix_parts[5]),
                    weight_path=weight_path,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to parse SSN method string '{method_str}': {e}"
                )

        parts = method_str.split("_")
        try:
            mt = parts[0]
            if mt == "SLIC" and len(parts) == 4:
                return SLICSuperpixel(n_clusters=int(parts[1]),
                                      compactness=float(parts[2]),
                                      sigma=float(parts[3]))
            if mt == "Felzenszwalb" and len(parts) == 4:
                return FelzenszwalbSuperpixel(min_size=int(parts[1]),
                                              sigma=float(parts[2]),
                                              scale=float(parts[3]))
            if mt == "Watershed" and len(parts) == 3:
                return WatershedSuperpixel(compactness=float(parts[1]),
                                           n_components=int(parts[2]))
        except Exception as e:
            raise ValueError(f"Failed to parse method string '{method_str}': {e}")
        raise ValueError(f"Unknown/invalid method string: {method_str}")

    # ---- Helpers (factored out from _update_annotations) ----
    def _roi_from_norm_poly(self, poly: Polygon, pad: int = 2) -> Tuple[int, int, int, int]:
        H, W = self.image_lab.shape[:2]
        minx, miny, maxx, maxy = poly.bounds
        x0 = max(0, int(np.floor(minx * W)) - pad)
        y0 = max(0, int(np.floor(miny * H)) - pad)
        x1 = min(W, int(np.ceil(maxx * W)) + pad)
        y1 = min(H, int(np.ceil(maxy * H)) + pad)
        if x1 <= x0:
            x1 = min(W, x0 + 2)
        if y1 <= y0:
            y1 = min(H, y0 + 2)
        return x0, y0, x1, y1

    def _rasterize_poly_to_roi_mask(self, poly: Polygon, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        H, W = self.image_lab.shape[:2]
        x0, y0, x1, y1 = roi
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return None
        pts = np.asarray(poly.exterior.coords, dtype=np.float32)
        pts_px = np.empty_like(pts)
        pts_px[:, 0] = pts[:, 0] * W - x0
        pts_px[:, 1] = pts[:, 1] * H - y0
        ov = Image.new("L", (w, h), 0)
        ImageDraw.Draw(ov).polygon([tuple(p) for p in pts_px], outline=1, fill=1)
        return np.array(ov, dtype=bool)

    def _props_on_roi_mask(self, roi: Tuple[int, int, int, int], mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        x0, y0, x1, y1 = roi
        if mask is None or not mask.any():
            return None
        arr = self.image_lab[y0:y1, x0:x1][mask]
        if arr.size == 0:
            return None
        m = arr.mean(axis=0)
        v = arr.var(axis=0)
        return np.concatenate([m, v]).astype(np.float32)

    def _clip_poly_to_superpixel(self, poly_norm: Polygon, sp_poly: Polygon) -> List[Polygon]:
        try:
            geom = poly_norm.intersection(sp_poly.buffer(0))
            if geom.is_empty:
                return []
            if isinstance(geom, Polygon):
                geoms = [geom]
            elif isinstance(geom, shapely.MultiPolygon):
                geoms = list(geom.geoms)
            else:
                return []
            out = []
            for g in geoms:
                g = g.buffer(0)
                if g.is_empty or not isinstance(g, Polygon) or g.area < 1e-8:
                    continue
                out.append(g)
            return out
        except Exception as _e:
            logger.debug("clip failed: %s", _e)
            return []

    def _get_full_image_pixel_embeddings(self) -> Optional[np.ndarray]:
        if not self.embedding_weight_path:
            return None
        key = (
            os.path.abspath(self.embedding_weight_path),
            int(self.embedding_fdim),
            float(self.embedding_color_scale),
            float(self.embedding_pos_scale),
        )
        if key not in self._pixel_embedding_cache:
            try:
                self._pixel_embedding_cache[key] = _compute_roi_embeddings(
                    self.image_lab,
                    self.embedding_weight_path,
                    fdim=self.embedding_fdim,
                    color_scale=self.embedding_color_scale,
                    pos_scale=self.embedding_pos_scale,
                )
            except Exception as exc:
                logger.warning("Full-image embedding computation failed: %s", exc)
                return None
        return self._pixel_embedding_cache.get(key)

    def _ensure_superpixel_embeddings(self, sp_method: SuperPixelMethod) -> None:
        self._ensure_embedding_defaults_for_method(sp_method)
        if not self.embedding_weight_path:
            return
        sp_list = self.superpixels.get(sp_method, [])
        if not sp_list or all(sp.emb is not None for sp in sp_list):
            return
        pixel_emb = self._get_full_image_pixel_embeddings()
        if pixel_emb is None:
            return

        for sp in sp_list:
            if sp.emb is not None:
                continue
            poly = sp.poly
            if poly is None or poly.is_empty:
                continue
            roi = self._roi_from_norm_poly(poly, pad=1)
            mask = self._rasterize_poly_to_roi_mask(poly, roi)
            if mask is None or not mask.any():
                continue
            x0, y0, x1, y1 = roi
            region_vecs = pixel_emb[y0:y1, x0:x1][mask]
            if region_vecs.size == 0:
                continue
            mean_vec = region_vecs.mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm <= 1e-8:
                continue
            sp.emb = (mean_vec / norm).astype(np.float32)

    def _compute_code_embedding_prototype(
        self,
        sp_method: SuperPixelMethod,
        code: int,
    ) -> Optional[np.ndarray]:
        self._ensure_superpixel_embeddings(sp_method)
        sp_by_id = {int(sp.id): sp for sp in self.superpixels.get(sp_method, [])}
        vecs: List[np.ndarray] = []
        for anno in self._annotations.get(sp_method, ImageAnnotation([])).annotations:
            if int(anno.code) != int(code):
                continue
            sp = sp_by_id.get(int(anno.parent_superpixel))
            if sp is None or sp.emb is None:
                continue
            vecs.append(np.asarray(sp.emb, dtype=np.float32))
        if not vecs:
            return None
        proto = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = float(np.linalg.norm(proto))
        if norm <= 1e-8:
            return None
        return (proto / norm).astype(np.float32)

    # ---- Public API ----
    def add_scribble(self, scribble: Scribble) -> None:
        scribble.id = self.ind_scrible
        self.scribbles_id_sequence.append(scribble.id)
        self.ind_scrible += 1

        self._scribbles.append(scribble)
        self._update_annotations(scribble)

    # alias with typo for backward compatibility
    def sp_annoted_before(self, sp_id: int, sp_method: Optional[SuperPixelMethod] = None) -> bool:
        return self.sp_annotated_before(sp_id, sp_method)

    def sp_annotated_before(self, sp_id: int, sp_method: Optional[SuperPixelMethod] = None) -> bool:
        """Return True if superpixel with given *id* is already annotated.

        Prefer checking inside the given *sp_method*. If method is not provided,
        checks all methods (slower) which may lead to false positives if ids overlap across methods.
        """
        if not self._annotations:
            return False
        if sp_method is not None and sp_method in self._annotations:
            return any(a.parent_superpixel == sp_id for a in self._annotations[sp_method].annotations)
        # fallback: check across all methods
        for img_ann in self._annotations.values():
            if any(a.parent_superpixel == sp_id for a in img_ann.annotations):
                return True
        return False

    def _find_annotation_for_superpixel(
        self,
        sp_id: int,
        sp_method: SuperPixelMethod,
    ) -> tuple[Optional[int], Optional[AnnotationInstance]]:
        ann_obj = self._annotations.get(sp_method)
        if ann_obj is None:
            return None, None
        for idx, anno in enumerate(ann_obj.annotations):
            if int(anno.parent_superpixel) == int(sp_id):
                return idx, anno
        return None, None

    @staticmethod
    def _is_split_descendant(sp: SuperPixel) -> bool:
        return bool(getattr(sp, "parents", None))

    def use_sensitivity_for_region(
        self,
        sp_idx: int,
        sens: float,
        scribble: Scribble,
        *,
        radius_scale: float = 1.0,
        property_scale: float = 1.0,
        embedding_threshold_override: Optional[float] = None,
    ) -> None:
        """
        Безопасное расширение аннотации по соседям:
        - работает ТОЛЬКО в главном потоке (вызывается снаружи после merge)
        - не трогает суперпиксели, пересекающиеся с ЛЮБЫМИ штрихами другого класса
        - parent_intersect выставляется по факту buffered-пересечения с родительским штрихом
        - топологическое ограничение по расстоянию между полигонами не применяется:
          распространение контролируется пространственным радиусом и feature-gate
        """
        if not self.superpixel_methods:
            return
        sp_method = self.superpixel_methods[0]
        self._ensure_embedding_defaults_for_method(sp_method)
        self._ensure_superpixel_embeddings(sp_method)
        self._ensure_spatial_index(sp_method)
        if self._sp_kdtree is None or self._sp_centroids is None:
            return
        if sp_idx is None or sp_idx < 0 or sp_idx >= len(self.superpixels[sp_method]):
            return
        if len(self._scribbles) == 0:
            return

        H, W = self.image_lab.shape[:2]
        all_lines = [LineString(s.points) for s in self._scribbles if len(s.points) >= 2]
        all_codes = [int(s.params.code) for s in self._scribbles if len(s.points) >= 2]
        if not all_lines:
            return

        # Буфер ≈ 2 пикселя в нормированных координатах
        buf = max(1e-5, 2.0 / float(max(H, W)))
        all_buffers = [ln.buffer(buf, cap_style=2, join_style=2) for ln in all_lines]
        lines_tree = STRtree(all_buffers)

        seed_code = int(scribble.params.code)
        seed_line_buf = LineString(scribble.points).buffer(buf, cap_style=2, join_style=2)

        radius = float(self._superpixel_radius * max(0.0, sens) * max(0.0, radius_scale))
        seed_sp = self.superpixels[sp_method][sp_idx]
        seed_props = seed_sp.props
        proto_emb = self._compute_code_embedding_prototype(sp_method, int(scribble.params.code))
        emb_threshold = (
            float(embedding_threshold_override)
            if embedding_threshold_override is not None
            else float(self._embedding_threshold)
        )
        ref_emb = None
        if proto_emb is not None and seed_sp.emb is not None:
            merged = 0.5 * (np.asarray(proto_emb, dtype=np.float32) + np.asarray(seed_sp.emb, dtype=np.float32))
            merged_norm = float(np.linalg.norm(merged))
            if merged_norm > 1e-8:
                ref_emb = (merged / merged_norm).astype(np.float32)
        elif proto_emb is not None:
            ref_emb = np.asarray(proto_emb, dtype=np.float32)
        elif seed_sp.emb is not None:
            ref_emb = np.asarray(seed_sp.emb, dtype=np.float32)
        visited = {sp_idx}
        queue = [sp_idx]

        while queue:
            cur = queue.pop()
            cur_pt = np.array(self._sp_centroids[cur])

            # соседи по центроидам
            idxs = self._sp_kdtree.query_ball_point(cur_pt, r=radius)
            for nb in idxs:
                if nb in visited or nb == cur:
                    continue
                cand_sp = self.superpixels[sp_method][nb]
                cand_poly = cand_sp.poly

                # 1) Жёсткая защита: если пересекается с ЛЮБЫМ штрихом другого класса — пропускаем
                hit_idx = _strtree_query_indices(lines_tree, cand_poly)
                conflict = False
                for h in hit_idx:
                    if all_codes[h] != seed_code:
                        conflict = True
                        break
                if conflict:
                    continue

                # 2) Сходство признаков — эмбединги (если есть) или LAB props
                if ref_emb is not None and cand_sp.emb is not None:
                    # cosine similarity against the class prototype / seed mean embedding
                    cos_sim = float(np.dot(ref_emb, cand_sp.emb))
                    if cos_sim < emb_threshold:
                        continue
                else:
                    if cand_sp.props is None or seed_props is None:
                        continue
                    if not np.all(
                        np.abs(cand_sp.props - seed_props)
                        < (self._property_dist * sens * max(0.0, property_scale))
                    ):
                        continue

                # 3) Аннотирование
                parent_intersect = cand_poly.intersects(seed_line_buf)

                anno_idx, anno_obj = self._find_annotation_for_superpixel(int(cand_sp.id), sp_method)
                if anno_obj is not None:
                    # Directly scribbled / intersected regions stay "hard". Areas
                    # produced only by previous propagation can be reused as a
                    # target and recolored by the new propagation pass.
                    if bool(anno_obj.parent_intersect):
                        visited.add(nb)
                        continue

                    prev_sids = [int(s) for s in (anno_obj.parent_scribble or [])]
                    if int(scribble.id) not in prev_sids:
                        prev_sids.append(int(scribble.id))
                    anno_obj.code = seed_code
                    anno_obj.parent_scribble = prev_sids
                    anno_obj.parent_intersect = bool(parent_intersect)
                    visited.add(nb)
                    continue
                else:
                    self._annotations[sp_method].annotations.append(
                        AnnotationInstance(
                            id=self._annotation_ind[sp_method],
                            code=seed_code,
                            border=cand_sp.border.astype(np.float32),
                            parent_superpixel=int(cand_sp.id),
                            parent_scribble=[scribble.id],
                            parent_intersect=bool(parent_intersect),
                        )
                    )
                    self._annotation_ind[sp_method] += 1
                visited.add(nb)
                queue.append(nb)

    # backward-compat alias
    def use_sensetivity_for_region(self, *args, **kwargs):
        return self.use_sensitivity_for_region(*args, **kwargs)

    # ---- Core update ----
    def _update_annotations(self, last_scribble: Scribble) -> None:
        """Главная процедура добавления/сплита/перекраски по новому штриху."""
        if len(last_scribble) < 2:
            return

        scribble_line = LineString(last_scribble.points).simplify(1e-6, preserve_topology=True)
        H, W = self.image_lab.shape[:2]
        px_buf = max(1e-5, 2.0 / float(max(H, W)))  # ~2 px в норм. координатах
        scribble_line_buf = scribble_line.buffer(px_buf, cap_style=2, join_style=2)
        if not self.superpixel_methods:
            return
        superpixel_method = self.superpixel_methods[0]

        # 1) Индексы
        self._ensure_spatial_index(superpixel_method)
        if not getattr(self, "_sp_polys", None):
            return
        preps = self._sp_prepared

        # 2) Кандидаты (пересекающиеся AABB + STRtree)
        tree = STRtree(self._sp_polys)
        candidates = sorted(set(_strtree_query_indices(tree, scribble_line)))

        sx0, sy0, sx1, sy1 = scribble_line.bounds
        aabb_extra: List[int] = []
        for i, p in enumerate(self._sp_polys):
            px0, py0, px1, py1 = p.bounds
            if (sx0 <= px1) and (sx1 >= px0) and (sy0 <= py1) and (sy1 >= py0):
                aabb_extra.append(i)
        candidates = sorted(set(candidates) | set(aabb_extra))

        # 3) Параллельные воркеры (только планы действий)
        def _process_candidate(cur_superpixel_ind: int) -> _CandResult:
            if cur_superpixel_ind is None or cur_superpixel_ind < 0 or cur_superpixel_ind >= len(self._sp_polys):
                return _CandResult([], [], [], [], [], [])
            sp_poly_geom = self._sp_polys[cur_superpixel_ind]
            if not isinstance(sp_poly_geom, Polygon):
                return _CandResult([], [], [], [], [], [])

            does_intersect = preps[cur_superpixel_ind].intersects(scribble_line_buf)
            cur_superpixel = self.superpixels[superpixel_method][cur_superpixel_ind]

            superpixel_ind_to_del: List[int] = []
            superpixel_to_append: List[SuperPixel] = []
            scribbles_id_to_check: List[int] = []
            annotations_to_del: List[int] = []
            new_annotations: List[AnnotationInstance] = []
            update_existing_annos: List[Tuple[int, Dict]] = []

            # была ли аннотация на этом SP
            annotated_before = False
            base_anno_ind = None
            anno_obj: Optional[AnnotationInstance] = None
            for ai, a in enumerate(self._annotations[superpixel_method].annotations):
                if a.parent_superpixel == cur_superpixel.id:
                    annotated_before = True
                    base_anno_ind = ai
                    anno_obj = a
                    break

            if annotated_before:
                prev_code = int(anno_obj.code)
                new_code = int(last_scribble.params.code)

                # тот же класс — только цепляем штрих
                if prev_code == new_code:
                    if does_intersect:
                        ps = (anno_obj.parent_scribble or [])
                        update_existing_annos.append((base_anno_ind, {
                            "parent_scribble": ps + [int(last_scribble.id)],
                            "parent_intersect": True
                        }))
                    return _CandResult([], [], [], [], [], update_existing_annos)

                # РАЗНЫЙ класс + пересечение (буфером) => сплит
                if does_intersect:
                    # Do not geometrically split the same region twice. If the
                    # current superpixel is already a child produced by a
                    # previous split, treat a new direct scribble as relabeling
                    # that refined piece instead of creating second-level
                    # fragments.
                    if self._is_split_descendant(cur_superpixel):
                        new_anno = AnnotationInstance(
                            id=-1,
                            code=last_scribble.params.code,
                            border=anno_obj.border,
                            parent_superpixel=anno_obj.parent_superpixel,
                            parent_scribble=[int(last_scribble.id)],
                            parent_intersect=True,
                        )
                        return _CandResult(
                            superpixel_ind_to_del=[],
                            annotations_to_del=[base_anno_ind],
                            superpixel_to_append=[],
                            scribbles_to_check=[],
                            new_annotations=[new_anno],
                            update_existing_annos=[]
                        )
                    if anno_obj.parent_intersect == False:
                        new_anno = AnnotationInstance(
                            id=-1,
                            code=last_scribble.params.code,
                            border=anno_obj.border,
                            parent_superpixel=anno_obj.parent_superpixel,
                            parent_scribble=[int(last_scribble.id)],
                            parent_intersect=True,
                        )
                        return _CandResult(
                            superpixel_ind_to_del=[],
                            annotations_to_del=[base_anno_ind],
                            superpixel_to_append=[],
                            scribbles_to_check=[],
                            new_annotations=[new_anno],
                            update_existing_annos=[]
                        )

                    prev_parent_scribbles = anno_obj.parent_scribble or []
                    ids_to_check = [int(s) for s in prev_parent_scribbles] + [int(last_scribble.id)]

                    sp_poly = cur_superpixel.poly
                    roi = self._roi_from_norm_poly(sp_poly, pad=3)
                    x0, y0, x1, y1 = roi
                    image_roi = self.image_lab[y0:y1, x0:x1]
                    sp_mask_roi = self._rasterize_poly_to_roi_mask(sp_poly, roi)
                    if sp_mask_roi is None:
                        return _CandResult([], [], [], [], [], [])

                    # чуть чистим, не склеиваем
                    kernel = np.ones((3, 3), np.uint8)
                    sp_mask_roi = cv2.morphologyEx(sp_mask_roi.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)

                    # адаптивные пределы SLIC
                    roi_h, roi_w = image_roi.shape[:2]
                    area_px = max(1, roi_h * roi_w)
                    cap = min(1024, max(66, int(area_px / 512)))
                    cur_n_segments = min(64, max(4, int(area_px / 4096)))

                    is_ok = False
                    tmp: List[SuperPixel] = []
                    while not is_ok and cur_n_segments <= cap:
                        try:
                            sp_mask = sk_slic(image_roi, n_segments=int(cur_n_segments),
                                              compactness=20, sigma=1, start_label=1, mask=sp_mask_roi).astype(np.int32)
                        except Exception:
                            break
                        try:
                            means, vars_, valid_labels = parallel_stats_rgb(
                                image_roi.astype(np.float32, copy=False),
                                sp_mask, int(np.max(sp_mask))
                            )
                        except Exception:
                            break

                        tmp.clear()
                        bbox01 = (x0 / W, y0 / H, x1 / W, y1 / H)
                        polys = labels_to_polygons(sp_mask, bbox01=bbox01, out_h=H, out_w=W, filter_labels=valid_labels)
                        for i_lbl, lab in enumerate(valid_labels):
                            if int(lab) <= 0:
                                continue
                            poly_arr = polys.get(int(lab))
                            if poly_arr is None or len(poly_arr) < 3:
                                continue
                            raw_piece = Polygon(poly_arr)
                            if raw_piece.is_empty:
                                continue
                            for piece in self._clip_poly_to_superpixel(raw_piece, sp_poly):
                                try:
                                    piece = piece.buffer(0)  # чинит самопересечения
                                except Exception as _e:
                                    logger.warning("SP split: piece.buffer(0) failed (sp_id=%d): %s", int(cur_superpixel.id), _e)
                                    continue
                                if piece.is_empty or (not isinstance(piece, Polygon)) or piece.area < 1e-8:
                                    logger.warning("SP split: dropped degenerate piece (area=%.3g, sp_id=%d).", float(getattr(piece, 'area', 0.0)), int(cur_superpixel.id))
                                    continue
                                coords = np.asarray(piece.exterior.coords, dtype=np.float32)
                                if coords.shape[0] < 3:
                                    print(i_lbl, " exited by coords < 3")
                                    continue
                                rp = self._roi_from_norm_poly(piece, pad=1)
                                mp = self._rasterize_poly_to_roi_mask(piece, rp)
                                props_arr = self._props_on_roi_mask(rp, mp)
                                if props_arr is None:
                                    try:
                                        props_arr = np.concatenate([means[i_lbl], vars_[i_lbl]]).astype(np.float32)
                                    except Exception:
                                        props_arr = None
                                tmp.append(SuperPixel(
                                    id=-1,
                                    method=superpixel_method.short_string(),
                                    border=np.around(coords, 7),
                                    parents=list((cur_superpixel.parents or [])) + [cur_superpixel.id],
                                    props=props_arr
                                ))

                        # проверка: один кусок — коды штрихов не смешиваются
                        def _check_non_overlap_by_classes(scribble_ids: List[int], regions: List[SuperPixel]) -> bool:
                            if not regions:
                                return False
                            region_polys = [r.poly for r in regions]
                            prepared_polys = [prepared.prep(p) for p in region_polys]
                            tree_local = STRtree(region_polys)
                            buf = max(1e-5, 2.0 / float(max(H, W)))
                            id_to_geom, id_to_code = {}, {}
                            for s in self._scribbles:
                                if s.id in set(scribble_ids) and len(s.points) >= 2:
                                    ln = LineString(s.points)
                                    id_to_geom[s.id] = ln.buffer(buf, cap_style=2, join_style=2)
                                    id_to_code[s.id] = int(s.params.code)
                            region_codes: List[set] = [set() for _ in region_polys]
                            for sid in scribble_ids:
                                g = id_to_geom.get(sid)
                                if g is None:
                                    continue
                                cands = _strtree_query_indices(tree_local, g)
                                for idx in cands:
                                    if prepared_polys[idx].intersects(g):
                                        region_codes[idx].add(id_to_code.get(sid, -9999))
                            return all(len(c) <= 1 for c in region_codes)

                        is_ok = _check_non_overlap_by_classes(ids_to_check, tmp)
                        cur_n_segments += 10

                    # fallback split по самой линии
                    if (not tmp) or (not is_ok):
                        try:
                            pieces = shp_split(sp_poly.buffer(0), scribble_line)
                            tmp = []
                            if getattr(pieces, "geoms", None) and len(pieces.geoms) >= 2:
                                for piece in pieces.geoms:
                                    try:
                                        piece = piece.buffer(0)  # чинит самопересечения
                                    except Exception as _e:
                                        logger.warning("SP split: piece.buffer(0) failed (sp_id=%d): %s", int(cur_superpixel.id), _e)
                                        continue
                                    if piece.is_empty or (not isinstance(piece, Polygon)) or piece.area < 1e-8:
                                        logger.warning("SP split: dropped degenerate piece (area=%.3g, sp_id=%d).", float(getattr(piece, 'area', 0.0)), int(cur_superpixel.id))
                                        continue
                                    coords = np.asarray(piece.exterior.coords, dtype=np.float32)
                                    if coords.shape[0] < 3:
                                        continue
                                    rp = self._roi_from_norm_poly(piece, pad=1)
                                    mp = self._rasterize_poly_to_roi_mask(piece, rp)
                                    props_arr = self._props_on_roi_mask(rp, mp)
                                    tmp.append(SuperPixel(
                                        id=-1,
                                        method=superpixel_method.short_string(),
                                        border=np.around(coords, 7),
                                        parents=list((cur_superpixel.parents or [])) + [cur_superpixel.id],
                                        props=props_arr
                                    ))
                            else:
                                tmp = []
                                is_ok = False
                        except Exception:
                            tmp = []
                            is_ok = False

                    # применяем ТОЛЬКО если реально появились куски
                    # Гарантируем покрытие исходного SP
                    try:
                        fixed_pieces, coverage = self._heal_and_complete_coverage(sp_poly, tmp, area_tol=1e-8, miss_tol_rel=0.02)
                        tmp = fixed_pieces
                        if tmp:
                            logger.info("SP split: sp_id=%d pieces=%d coverage=%.4f", int(cur_superpixel.id), len(tmp), coverage)
                            # Сохраним диагностический дамп (недорого: WKT)
                            self._debug_dump_split(int(cur_superpixel.id), sp_poly, tmp, last_scribble, tag="slic", coverage=coverage)
                        # Предохранитель: если покрытие ниже 90% — считаем сплит сомнительным и отменяем его.
                        if tmp and coverage < 0.90:
                            logger.warning("SP split: coverage too low (%.3f) for sp_id=%d, aborting split.", coverage, int(cur_superpixel.id))
                            tmp = []
                    except Exception as e:
                        logger.exception("SP split coverage check failed (sp_id=%d): %s", int(cur_superpixel.id), e)
                        tmp = []

                    if tmp:
                        # print(len(tmp))
                        superpixel_to_append.extend(tmp)
                        return _CandResult(
                            superpixel_ind_to_del=[cur_superpixel_ind],
                            annotations_to_del=[base_anno_ind],
                            superpixel_to_append=superpixel_to_append,
                            scribbles_to_check=ids_to_check,
                            new_annotations=[],
                            update_existing_annos=[]
                        )

                    # сплит не удался — ничего не меняем
                    return _CandResult([], [], [], [], [], [])

                # разный класс, но пересечения нет — ничего не делаем
                return _CandResult([], [], [], [], [], [])

            # новой аннотации ещё не было
            if does_intersect:
                new_annotations.append(
                    AnnotationInstance(
                        id=-1,
                        code=last_scribble.params.code,
                        border=cur_superpixel.border.astype(np.float32),
                        parent_superpixel=int(cur_superpixel.id),
                        parent_scribble=[int(last_scribble.id)],
                        parent_intersect=True,
                    )
                )
            return _CandResult([], [], [], [], new_annotations, [])

        results: List[_CandResult] = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
            futures = [ex.submit(_process_candidate, idx) for idx in candidates]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    logger.warning("candidate failed: %s", e)

        # 5) Применение планов
        superpixel_ind_to_del = sorted({i for r in results for i in r.superpixel_ind_to_del})
        annotations_to_del = sorted({i for r in results for i in r.annotations_to_del})
        superpixel_to_append = [sp for r in results for sp in r.superpixel_to_append]
        scribbles_to_check = sorted(set(int(sid) for r in results for sid in r.scribbles_to_check))
        new_annotations = [a for r in results for a in r.new_annotations]
        update_existing = [u for r in results for u in r.update_existing_annos]

        # обновление существующих аннотаций
        for anno_ind, upd in update_existing:
            anno = self._annotations[superpixel_method].annotations[anno_ind]
            if "parent_scribble" in upd:
                anno.parent_scribble = list(upd["parent_scribble"])
            if "code" in upd:
                anno.code = int(upd["code"])
            if "parent_intersect" in upd:
                anno.parent_intersect = bool(upd["parent_intersect"])

        # удаление SP и аннотаций под сплит
        for sp_to_del in sorted(superpixel_ind_to_del, reverse=True):
            self.superpixels[superpixel_method].pop(sp_to_del)
            self._mark_sp_index_dirty()
        for anno_to_del in sorted(annotations_to_del, reverse=True):
            self._annotations[superpixel_method].annotations.pop(anno_to_del)

        # новые аннотации по пересечению
        for a in new_annotations:
            dup = any((aa.parent_superpixel == a.parent_superpixel) and (aa.code == a.code)
                      for aa in self._annotations[superpixel_method].annotations)
            if dup:
                continue
            a.id = self._annotation_ind[superpixel_method]
            self._annotations[superpixel_method].annotations.append(a)
            self._annotation_ind[superpixel_method] += 1

        # добавление новых SP от сплита
        new_sp_ids: List[int] = []
        for sp in superpixel_to_append:
            sp.id = self._superpixel_ind[superpixel_method]
            self._superpixel_ind[superpixel_method] += 1
            self.superpixels[superpixel_method].append(sp)
            new_sp_ids.append(int(sp.id))
        if superpixel_to_append:
            self._mark_sp_index_dirty()
        
        sens_tasks = []  # (sp_idx, scribble)
        # привязка аннотаций только к новым кускам
        if new_sp_ids:
            id_to_sp = {int(sp.id): sp for sp in self.superpixels[superpixel_method]}
            impacted_scribbles = set(scribbles_to_check) | {int(last_scribble.id)}
            id2scr = {s.id: s for s in self._scribbles if int(s.id) in impacted_scribbles and len(s.points) >= 2}
            buf = max(1e-8, 2.0 / float(max(H, W)))
            id2geom = {}
            for sid in id2scr:
                ln = LineString(id2scr[sid].points)
                if _shp_set_precision is not None:
                    try:
                        ln = _shp_set_precision(ln, float(1e-9))
                    except Exception:
                        pass
                g = ln.buffer(buf, cap_style=2, join_style=2)
                try:
                    if not g.is_valid:
                        g = g.buffer(0)
                except Exception:
                    pass
                id2geom[sid] = g
            id2code = {sid: int(id2scr[sid].params.code) for sid in id2scr}

            def _pick_code_for_piece(poly: Polygon) -> Optional[int]:
                # выбираем код по максимальной площади пересечения
                best_sid, best_area = None, 0.0
                for sid, g in id2geom.items():
                    inter = poly.intersection(g)
                    a = inter.area if not inter.is_empty else 0.0
                    if a > best_area:
                        best_sid, best_area = sid, a
                if best_sid is not None and best_area > 0.0:
                    return id2code[best_sid]
                # фолбэк: ближний по расстоянию
                best_sid, best_d = None, 1e9
                for sid, g in id2geom.items():
                    d = poly.distance(g)
                    if d < best_d:
                        best_sid, best_d = sid, d
                return id2code[best_sid] if best_sid is not None else None

            for sp_id in new_sp_ids:
                sp = id_to_sp.get(sp_id)
                if sp is None:
                    continue
                if self.sp_annotated_before(sp_id, superpixel_method):  # write-once guard
                    continue
                poly = sp.poly
                code = _pick_code_for_piece(poly)
                if code is None:
                    continue

                # связываем с подходящим штрихом(ами)
                sids = [sid for sid in id2geom if id2code[sid] == code and not poly.intersection(id2geom[sid]).is_empty]
                if not sids:
                    candidates = [sid for sid in id2geom if id2code[sid] == code]
                    if candidates:
                        sids = [min(candidates, key=lambda sid: poly.distance(id2geom[sid]))]

                self._annotations[superpixel_method].annotations.append(
                    AnnotationInstance(
                        id=self._annotation_ind[superpixel_method],
                        code=int(code),
                        border=sp.border.astype(np.float32),
                        parent_superpixel=int(sp.id),
                        parent_scribble=[int(s) for s in sids] if sids else [int(last_scribble.id)],
                        parent_intersect=bool(any(poly.intersects(id2geom[s]) for s in sids)) if sids else True,
                    )
                )
                self._annotation_ind[superpixel_method] += 1

                sp_idx = next((i for i, spp in enumerate(self.superpixels[superpixel_method]) if int(spp.id) == int(sp.id)), None)
                if sp_idx is not None:
                    sens_tasks.append((sp_idx, last_scribble))
                    
        for a in new_annotations:
            sp_idx = next(
                (i for i, spp in enumerate(self.superpixels[superpixel_method]) if int(spp.id) == int(a.parent_superpixel)),
                None,
            )
            if sp_idx is not None:
                sens_tasks.append((sp_idx, last_scribble))
        # Выполняем «чувствительность» после merge, если сценарий это разрешает.
        auto_sens = float(getattr(self, "auto_propagation_sensitivity", 1.0))
        if auto_sens > 0.0:
            for sp_idx, scr in sens_tasks:
                self.use_sensitivity_for_region(sp_idx=sp_idx, sens=auto_sens, scribble=scr)

        self._assert_annotation_invariants()
        logger.info("Annotations: %d | Superpixels: %d",
                    len(self._annotations[superpixel_method].annotations),
                    len(self.superpixels[superpixel_method]))

    def get_annotation(self, method: SuperPixelMethod) -> ImageAnnotation:
        return self._annotations[method]

    def _assert_annotation_invariants(self, method=None):
        if method is None:
            if not self.superpixel_methods:
                return
            method = self.superpixel_methods[0]
        seen = {}
        for a in self._annotations.get(method, ImageAnnotation([])).annotations:
            pid = int(a.parent_superpixel)
            if pid in seen and seen[pid] != id(a):
                raise AssertionError(f"Duplicate annotation for SP {pid}")
            seen[pid] = id(a)

    # === DEBUG-дамп результатов сплита (по желанию) ============================
    def _debug_dump_split(self, sp_id: int, sp_poly: Polygon, pieces: List[SuperPixel],
                          scribble: Scribble, tag: str, coverage: float):
        try:
            os.makedirs(self.debug_candidates_dir, exist_ok=True)
            base = Path(self.debug_candidates_dir) / f"sp{sp_id}_{tag}_{coverage:.3f}"
            # Геометрии в WKT (быстро и читабельно)
            with open(str(base) + ".wkt", "w", encoding="utf-8") as f:
                f.write(f"# coverage={coverage:.5f}\n")
                f.write(f"SP: {sp_poly.wkt}\n")
                f.write(f"SCRIBBLE: {LineString(scribble.points).wkt}\n")
                for i, p in enumerate(pieces):
                    f.write(f"PIECE[{i}]: {Polygon(p.border).wkt}\n")
        except Exception as e:
            logger.debug("debug dump failed: %s", e)

    # === Починка покрытия: достроить остатки и почистить геометрию =============
    def _heal_and_complete_coverage(self, sp_poly: Polygon, pieces: List[SuperPixel],
                                    area_tol: float = 1e-7, miss_tol_rel: float = 0.02) -> Tuple[List[SuperPixel], float]:
        """
        Возвращает (pieces_fixed, coverage), где coverage = area(union(pieces))/area(sp_poly).
        Если есть «дыры» > miss_tol_rel — достраиваем их отдельными кусками.
        """
        if not pieces:
            return [], 0.0

        # Шаг 1: чистим и собираем юнион текущих кусков
        valid_polys = []
        for sp in pieces:
            try:
                g = sp.poly.buffer(0)
                if isinstance(g, Polygon) and (not g.is_empty) and (g.area > area_tol):
                    valid_polys.append(g)
            except Exception:
                continue

        if not valid_polys:
            return [], 0.0

        try:
            uni = unary_union(valid_polys).buffer(0)
        except Exception:
            # На всякий — fallback как сумма площадей без топ. юниона
            area_sum = sum(p.area for p in valid_polys)
            cov = min(1.0, float(area_sum / max(sp_poly.area, 1e-12)))
            return pieces, cov

        try:
            # Шаг 2: считаем недостающую область
            miss = sp_poly.buffer(0).difference(uni)
        except Exception:
            miss = None

        total_area = float(sp_poly.area)
        covered_area = float(uni.area) if hasattr(uni, "area") else 0.0
        coverage = covered_area / max(total_area, 1e-12)

        # Если дыры «существенные» — раскладываем и добавляем
        if (miss is not None) and (not miss.is_empty):
            try:
                geoms = []
                if isinstance(miss, Polygon):
                    geoms = [miss]
                elif isinstance(miss, shapely.MultiPolygon):
                    geoms = list(miss.geoms)
                add_list: List[SuperPixel] = []
                for g in geoms:
                    g = g.buffer(0)
                    if (not isinstance(g, Polygon)) or g.is_empty:
                        continue
                    if g.area < area_tol:
                        continue
                    coords = np.asarray(g.exterior.coords, dtype=np.float32)
                    if coords.shape[0] < 3:
                        continue
                    # props по ROI
                    rp = self._roi_from_norm_poly(g, pad=1)
                    mp = self._rasterize_poly_to_roi_mask(g, rp)
                    props_arr = self._props_on_roi_mask(rp, mp)
                    add_list.append(SuperPixel(
                        id=-1,
                        method=self.superpixel_methods[0].short_string(),
                        border=np.around(coords, 7),
                        parents=None,
                        props=props_arr
                    ))
                if add_list:
                    pieces = pieces + add_list
                    # обновим coverage после добавления
                    try:
                        uni2 = unary_union([Polygon(p.border) for p in pieces]).buffer(0)
                        coverage = float(uni2.area) / max(total_area, 1e-12)
                    except Exception:
                        # ок, оставим прежний coverage
                        pass
            except Exception as e:
                logger.debug("coverage healing failed: %s", e)

        return pieces, float(coverage)
