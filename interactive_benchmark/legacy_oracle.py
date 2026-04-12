from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from skimage.morphology import medial_axis, skeletonize


def _edt_inside(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)
    return dist[1:-1, 1:-1]


def _smooth_region_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
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


class LargestBadRegionGenerator:
    VALID_REGION_SELECTION_MODES = (
        "miou_gain",
        "largest_error",
        "unannotated",
    )

    def __init__(
        self,
        gt_mask: np.ndarray,
        num_classes: int,
        seed: int = 0,
        margin: int = 2,
        border_margin: int = 3,
        no_overlap: bool = True,
        max_retries: int = 200,
        region_selection_cycle: Optional[Sequence[str]] = None,
    ):
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

        self._gt_inner: List[np.ndarray] = []
        for cid in range(num_classes):
            cls = self.gt == cid
            if self.margin > 0 and cls.any():
                inner = cls & (_edt_inside(cls) > self.margin)
            else:
                inner = cls.copy()
            self._gt_inner.append(inner)

    def set_selection_step(self, step: int) -> None:
        self._selection_step = max(0, int(step))

    def _current_region_selection_mode(self) -> str:
        return self.region_selection_cycle[self._selection_step % len(self.region_selection_cycle)]

    def _advance_region_selection_cycle(self) -> None:
        self._selection_step += 1

    def _region_error_mask(self, cid: int, pred_mask: np.ndarray, mode: str) -> np.ndarray:
        gt_c = self.gt == cid
        if mode == "miou_gain":
            return gt_c & (pred_mask != cid)
        if mode == "largest_error":
            return gt_c & (pred_mask >= 0) & (pred_mask != cid)
        if mode == "unannotated":
            return gt_c & (pred_mask < 0)
        raise ValueError(f"Unsupported region selection mode: {mode!r}")

    def _largest_component(self, bad: np.ndarray) -> Optional[np.ndarray]:
        lab, n = cc_label(bad)
        if n == 0:
            return None
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        k = int(np.argmax(counts))
        return lab == k if counts[k] > 0 else None

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

    def _component_signature(self, cid: int, comp: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
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
            pred_c = pred_mask == cid
            gt_c = self.gt == cid
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
        total_scr = 0 if class_scribble_counts is None else int(sum(int(v) for v in class_scribble_counts))
        actual_share = float(n_scr) / float(total_scr) if total_scr > 0 else 0.0
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

    def _build_allowed_mask(self, cid: int, comp: np.ndarray, pred_mask: np.ndarray, used_mask: np.ndarray) -> np.ndarray:
        bad_c = (self.gt == cid) & (pred_mask != cid)
        if self.border_margin > 0 and comp.any():
            comp_inner = comp & (_edt_inside(comp) > self.border_margin)
        else:
            comp_inner = comp
        allowed = comp_inner & self._gt_inner[cid] & bad_c
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
        if xs.size <= 1:
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

    def _sample_anchor_points(self, focus_mask: np.ndarray, allowed: np.ndarray, center_dt: np.ndarray) -> List[Tuple[int, int]]:
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
        candidates.append(self._nearest_allowed_point(allowed_xs, allowed_ys, float(xs[best_idx]), float(ys[best_idx])))
        if ranked.size > 1:
            far_rank = max(
                ranked,
                key=lambda idx: (float(xs[idx]) - centroid_x) ** 2 + (float(ys[idx]) - centroid_y) ** 2,
            )
            candidates.append(
                self._nearest_allowed_point(allowed_xs, allowed_ys, float(xs[int(far_rank)]), float(ys[int(far_rank)]))
            )
        for idx in ranked:
            pt = self._nearest_allowed_point(allowed_xs, allowed_ys, float(xs[int(idx)]), float(ys[int(idx)]))
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

    def _edt_corridor_masks(self, allowed: np.ndarray, dist_map: np.ndarray) -> List[np.ndarray]:
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

    def _build_scribble_core(self, allowed: np.ndarray, center_dt: np.ndarray) -> np.ndarray:
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

    def _segment_pixels(self, x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
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
            xs, ys = self._segment_pixels(int(pts_px[i, 0]), int(pts_px[i, 1]), int(pts_px[i + 1, 0]), int(pts_px[i + 1, 1]))
            if i > 0 and xs.size > 0:
                xs = xs[1:]
                ys = ys[1:]
            xs_parts.append(xs)
            ys_parts.append(ys)
        return np.concatenate(xs_parts), np.concatenate(ys_parts)

    def _polyline_band_mask(self, pts_px: np.ndarray, radius: int = 2) -> np.ndarray:
        mask = np.zeros((self.H, self.W), dtype=bool)
        xs, ys = self._polyline_pixels(pts_px)
        r = max(0, int(radius))
        for x, y in zip(xs, ys):
            mask[max(0, y - r):min(self.H, y + r + 1), max(0, x - r):min(self.W, x + r + 1)] = True
        return mask

    def _nearest_mask_point(self, mask: np.ndarray, x: float, y: float) -> Optional[Tuple[int, int]]:
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
    def _reconstruct_path(prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], end_xy: Tuple[int, int]) -> List[Tuple[int, int]]:
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
                score = (int(dist_i + dist_j), int(min(dist_i, dist_j)), float(abs(proj_i - proj_j)))
                if best_pair_score is None or score > best_pair_score:
                    best_pair_score = score
                    best_pair = (path_i, path_j)
        if best_pair is not None:
            path_a, path_b = best_pair
            path_xy = list(reversed(path_a)) + path_b[1:]
        else:
            neg_end = self._select_centerline_endpoint(endpoint_pool, center_skel, axis_xy, dist_steps, dt_map, sign=-1.0)
            pos_end = self._select_centerline_endpoint(endpoint_pool, center_skel, axis_xy, dist_steps, dt_map, sign=+1.0)
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
            proj = float(sign) * ((float(x) - float(cx)) * float(ax) + (float(y) - float(cy)) * float(ay))
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
        skeleton_candidates: List[Tuple[np.ndarray, np.ndarray]] = []
        for ridge_mask in self._edt_corridor_masks(source_mask, center_dt):
            if not ridge_mask.any():
                continue
            skeleton_candidates.append((skeletonize(ridge_mask).astype(bool, copy=False), ridge_mask))
            skeleton_candidates.append((medial_axis(ridge_mask).astype(bool, copy=False), ridge_mask))
        best_path: Optional[Tuple[np.ndarray, int]] = None
        best_score = -1.0
        max_dt = float(center_dt.max())
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
            center_support = float(np.mean(center_dt[ys, xs])) / float(max(1e-6, max_dt)) if max_dt > 1e-6 else 0.0
            score = float(path_len) + 0.5 * float(center_support)
            if score <= best_score:
                continue
            best_score = score
            best_path = (pts_px, center_index)
        return best_path

    def _trace(self, x0: int, y0: int, dx: float, dy: float, allowed: np.ndarray, used: np.ndarray) -> Tuple[int, int]:
        x, y = float(x0), float(y0)
        lx, ly = x0, y0
        for _ in range(int(self._diag)):
            x += dx
            y += dy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < self.W and 0 <= yi < self.H):
                break
            if not allowed[yi, xi]:
                break
            if self.no_overlap and used[yi, xi]:
                break
            lx, ly = xi, yi
        return lx, ly

    def _straight_fallback_path(self, allowed: np.ndarray, focus_mask: np.ndarray, center_dt: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
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
        core_covered = float(np.count_nonzero(band & scribble_core)) / float(max(1, int(np.count_nonzero(scribble_core))))
        xs, ys = self._polyline_pixels(pts_px)
        center_support = float(np.mean(allowed_dt[ys, xs])) / float(max(1e-6, float(allowed_dt.max()))) if xs.size > 0 and float(allowed_dt.max()) > 1e-6 else 0.0
        center_index = int(max(0, min(center_index, pts_px.shape[0] - 1)))
        cx = int(pts_px[center_index, 0])
        cy = int(pts_px[center_index, 1])
        seed_support = float(allowed_dt[cy, cx]) / float(max(1e-6, float(allowed_dt.max()))) if 0 <= cx < self.W and 0 <= cy < self.H and float(allowed_dt.max()) > 1e-6 else 0.0
        min_support = float(np.min(allowed_dt[ys, xs])) / float(max(1e-6, float(allowed_dt.max()))) if xs.size > 0 and float(allowed_dt.max()) > 1e-6 else 0.0
        path_xy = pts_px.astype(np.float64)
        seg_lens = np.sqrt((np.diff(path_xy, axis=0) ** 2).sum(axis=1))
        arm_a = float(seg_lens[:center_index].sum()) if center_index > 0 else 0.0
        arm_b = float(seg_lens[center_index:].sum()) if center_index < seg_lens.size else 0.0
        center_balance = float(min(arm_a, arm_b) / max(1e-6, max(arm_a, arm_b))) if max(arm_a, arm_b) > 1e-6 else 0.0
        length_norm = float(seg_lens.sum()) / float(max(1, max(self.H, self.W)))
        score = 4.0 * direct_gain + 0.90 * core_covered + 0.55 * comp_covered + 0.60 * center_support + 0.30 * min_support + 0.35 * length_norm
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
        path_choice = self._build_centerline_path(allowed=allowed, analysis_mask=analysis_mask, focus_mask=focus_mask, comp=comp)
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
        class_scribble_counts: Optional[List[int]] = None,
        selection_mode: str = "miou_gain",
    ) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        inter, union = self._class_inter_union(pred_mask)
        raw_candidates = []
        for cid in range(self.num_classes):
            bad_c = self._region_error_mask(cid, pred_mask, selection_mode)
            if not bad_c.any():
                continue
            union_c = max(1, int(union[cid]))
            class_priority = self._class_priority(cid=cid, inter=inter, union=union, bad_c=bad_c, class_scribble_counts=class_scribble_counts)
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
                raw_score = (float(primary) - comp_fail - 0.05 * recent_pen, float(secondary), float(self.rng.random()))
                raw_candidates.append((raw_score, int(cid), comp, allowed, signature, union_c))
        if not raw_candidates:
            return None, None, None, None
        raw_candidates.sort(key=lambda item: item[0], reverse=True)
        eval_budget = min(len(raw_candidates), 4 + min(2, self._stall_steps))
        candidates = []
        for raw_score, cid, comp, allowed, signature, union_c in raw_candidates[:eval_budget]:
            scribble_choice = self._best_scribble_for_component(cid=cid, comp=comp, allowed=allowed, union_c=union_c)
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
            alternate = [item for item in candidates if int(item[1]) != int(self._last_selected_class)]
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

    def make_scribble(
        self,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
        class_scribble_counts: Optional[List[int]] = None,
    ) -> Tuple[int, np.ndarray]:
        bad = pred_mask != self.gt
        if not bad.any():
            raise StopIteration("All pixels correctly labelled.")
        selection_mode = self._current_region_selection_mode()
        cid, comp, allowed, pts01 = self._select_class_component(
            pred_mask,
            used_mask,
            class_scribble_counts=class_scribble_counts,
            selection_mode=selection_mode,
        )
        if cid is None and selection_mode != "miou_gain":
            cid, comp, allowed, pts01 = self._select_class_component(
                pred_mask,
                used_mask,
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
                raise RuntimeError("Cannot place scribble in selected bad region.")
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


def parse_region_selection_cycle(raw_value: str) -> List[str]:
    items = [part.strip().lower() for part in str(raw_value).split(",")]
    items = [part for part in items if part]
    if not items:
        raise ValueError("Region selection cycle must not be empty.")
    invalid = [part for part in items if part not in LargestBadRegionGenerator.VALID_REGION_SELECTION_MODES]
    if invalid:
        raise ValueError(
            "Unknown region selection mode(s): "
            + ", ".join(repr(x) for x in invalid)
            + ". Expected one of "
            + ", ".join(LargestBadRegionGenerator.VALID_REGION_SELECTION_MODES)
        )
    return items
