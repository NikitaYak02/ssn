from __future__ import annotations

from typing import Iterable

import torch


def resolve_device(preference: str | None = None) -> torch.device:
    pref = str(preference or "").strip().lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _to_xy_tensor(points_xy: Iterable[tuple[float, float]], *, h: int, w: int, device: torch.device) -> torch.Tensor:
    pts = list(points_xy)
    if not pts:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    out = torch.tensor(pts, dtype=torch.float32, device=device)
    out[:, 0] = out[:, 0].clamp(0, float(w - 1))
    out[:, 1] = out[:, 1].clamp(0, float(h - 1))
    return out


def rasterize_points_gpu(
    *,
    h: int,
    w: int,
    points_xy: Iterable[tuple[float, float]],
    radius_px: int,
    device: torch.device,
) -> torch.Tensor:
    radius = max(1, int(radius_px))
    pts = _to_xy_tensor(points_xy, h=h, w=w, device=device)
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    if pts.numel() == 0:
        return mask
    ys = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
    for point in pts:
        px, py = float(point[0]), float(point[1])
        circle = (xs - px) ** 2 + (ys - py) ** 2 <= float(radius * radius)
        mask |= circle
    return mask


def rasterize_polyline_gpu(
    *,
    h: int,
    w: int,
    points_xy: Iterable[tuple[float, float]],
    width_px: int,
    device: torch.device,
) -> torch.Tensor:
    pts = _to_xy_tensor(points_xy, h=h, w=w, device=device)
    if pts.shape[0] <= 1:
        return rasterize_points_gpu(
            h=h,
            w=w,
            points_xy=[(float(pts[0, 0]), float(pts[0, 1]))] if pts.shape[0] == 1 else [],
            radius_px=max(1, int(width_px // 2)),
            device=device,
        )
    half_w = max(1.0, float(width_px) / 2.0)
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    xx = xs.expand(h, w)
    yy = ys.expand(h, w)
    out = torch.zeros((h, w), dtype=torch.bool, device=device)
    for i in range(pts.shape[0] - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        vx = x2 - x1
        vy = y2 - y1
        seg_len2 = vx * vx + vy * vy
        if float(seg_len2) < 1e-6:
            out |= ((xx - x1) ** 2 + (yy - y1) ** 2) <= half_w * half_w
            continue
        t = ((xx - x1) * vx + (yy - y1) * vy) / seg_len2
        t = t.clamp(0.0, 1.0)
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
        out |= dist2 <= half_w * half_w
    return out

