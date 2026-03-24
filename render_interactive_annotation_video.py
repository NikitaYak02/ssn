#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_interactive_annotation_video.py

Build an MP4 replay from `evaluate_interactive_annotation.py` outputs.

The script reads `state_*.json` / `state_*.json.gz` checkpoints and visualizes:
  1. current annotated superpixels,
  2. newly added scribbles,
  3. direct superpixel hits (`parent_intersect=True`),
  4. propagated labels (`parent_intersect=False`).

If checkpoints were saved sparsely (for example `--save_every 50`), the replay is
coarse between checkpoints because exact per-scribble intermediate states are no
longer available. For an exact step-by-step replay, run
`evaluate_interactive_annotation.py --save_every 1`.

Examples:
    superpixel_annotator/superpixel_annotator_venv/bin/python \
        render_interactive_annotation_video.py \
        --input out/interactive_repro_train01_ssn \
        --image /path/to/image.png

    superpixel_annotator/superpixel_annotator_venv/bin/python \
        render_interactive_annotation_video.py \
        --input out/batch_results \
        --image_dir /data/images \
        --fps 8
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
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

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_DIRECT_BRG = (0, 170, 255)
_PROP_BRG = (255, 255, 0)
_BORDER_BGR = (80, 80, 80)
_TEXT_MAIN_BGR = (245, 245, 245)
_TEXT_SUB_BGR = (180, 180, 180)
_BG_BGR = (22, 24, 28)
_PANEL_BG_BGR = (30, 33, 38)


def _hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return b, g, r


def _even(v: int) -> int:
    v = max(2, int(v))
    return v if (v % 2 == 0) else (v - 1)


def _open_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_state_index(path: Path) -> int:
    stem = path.name
    if stem.endswith(".json.gz"):
        stem = stem[:-8]
    elif stem.endswith(".json"):
        stem = stem[:-5]
    try:
        return int(stem.rsplit("_", 1)[-1])
    except Exception:
        return -1


def _build_default_class_info(num_classes: int) -> List[Tuple[str, str]]:
    if num_classes <= len(DEFAULT_CLASS_INFO):
        return DEFAULT_CLASS_INFO[:num_classes]
    extra = [(f"cls{i}", "#aaaaaa") for i in range(len(DEFAULT_CLASS_INFO), num_classes)]
    return DEFAULT_CLASS_INFO + extra


def _scaled_poly(coords01: Sequence[Sequence[float]], sx: float, sy: float) -> np.ndarray:
    arr = np.asarray(coords01, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        raise ValueError(f"Invalid polygon shape: {arr.shape}")
    pts = np.empty_like(arr, dtype=np.float32)
    pts[:, 0] = arr[:, 0] * sx
    pts[:, 1] = arr[:, 1] * sy
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, int(round(sx)) - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, int(round(sy)) - 1))
    return pts.reshape((-1, 1, 2))


def _scaled_line(coords01: Sequence[Sequence[float]], sx: float, sy: float) -> np.ndarray:
    arr = np.asarray(coords01, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != 2:
        raise ValueError(f"Invalid line shape: {arr.shape}")
    pts = np.empty_like(arr, dtype=np.float32)
    pts[:, 0] = arr[:, 0] * sx
    pts[:, 1] = arr[:, 1] * sy
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, int(round(sx)) - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, int(round(sy)) - 1))
    return pts.reshape((-1, 1, 2))


def _read_metrics_csv(path: Path) -> Dict[int, Dict[str, float]]:
    if not path.exists():
        return {}
    metrics: Dict[int, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                step = int(float(row["step"]))
            except Exception:
                continue
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if key == "step" or value in ("", None):
                    continue
                try:
                    parsed[key] = float(value)
                except Exception:
                    continue
            metrics[step] = parsed
    return metrics


@dataclass(frozen=True)
class ScribbleData:
    sid: int
    code: int
    points01: Tuple[Tuple[float, float], ...]
    points_px: np.ndarray


@dataclass(frozen=True)
class AnnotationData:
    sp_id: int
    code: int
    parent_scribble: Tuple[int, ...]
    parent_intersect: bool
    poly_px: np.ndarray

    def signature(self) -> Tuple[int, bool, Tuple[int, ...]]:
        return self.code, self.parent_intersect, self.parent_scribble


@dataclass
class LoadedState:
    path: Path
    checkpoint_index: int
    n_scribbles: int
    width: int
    height: int
    render_width: int
    render_height: int
    method: str
    scribbles: List[ScribbleData]
    annotations_by_sp: Dict[int, AnnotationData]
    superpixels_by_id: Dict[int, np.ndarray]


@dataclass(frozen=True)
class Event:
    prev_step: int
    cur_step: int
    checkpoint_index: int
    step_delta: int
    exact: bool
    new_scribbles: Tuple[ScribbleData, ...]
    direct_sp_ids: Tuple[int, ...]
    propagated_sp_ids: Tuple[int, ...]


@dataclass(frozen=True)
class TimingConfig:
    intro_frames: int
    pre_frames: int
    direct_frames: int
    prop_frames: int
    final_frames: int
    outro_frames: int


def choose_method(state_dict: dict, method: Optional[str]) -> str:
    available = sorted(
        set(list((state_dict.get("superpixels") or {}).keys()) + list((state_dict.get("annotations") or {}).keys()))
    )
    if not available:
        raise ValueError("State file does not contain any methods.")
    if method:
        if method not in available:
            raise ValueError(f"Requested method {method!r} not found. Available: {available}")
        return method
    meta_methods = list(((state_dict.get("_meta") or {}).get("methods") or []))
    for candidate in meta_methods:
        if candidate in available:
            return candidate
    return available[0]


def load_state(path: Path, max_side: int, method: Optional[str]) -> LoadedState:
    state = _open_json(path)
    chosen_method = choose_method(state, method)
    meta = state.get("_meta") or {}
    image_meta = meta.get("image") or {}
    width, height = map(int, image_meta.get("size_wh", [0, 0]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing image size in {path}")

    scale = min(1.0, float(max_side) / float(max(width, height)))
    render_width = _even(round(width * scale))
    render_height = _even(round(height * scale))

    scribbles: List[ScribbleData] = []
    for scrib in state.get("scribbles") or []:
        sid = int(scrib["id"])
        code = int((scrib.get("params") or {}).get("code", 1))
        pts01 = tuple((float(x), float(y)) for x, y in (scrib.get("points") or []))
        if len(pts01) < 2:
            continue
        scribbles.append(
            ScribbleData(
                sid=sid,
                code=code,
                points01=pts01,
                points_px=_scaled_line(pts01, render_width, render_height),
            )
        )

    superpixels_by_id: Dict[int, np.ndarray] = {}
    for sp in (state.get("superpixels") or {}).get(chosen_method, []):
        sp_id = int(sp["id"])
        try:
            superpixels_by_id[sp_id] = _scaled_poly(sp["border"], render_width, render_height)
        except Exception:
            continue

    annotations_by_sp: Dict[int, AnnotationData] = {}
    for anno in (state.get("annotations") or {}).get(chosen_method, []):
        sp_id = int(anno["parent_superpixel"])
        border = anno.get("border")
        if border is None and sp_id in superpixels_by_id:
            poly_px = superpixels_by_id[sp_id]
        else:
            try:
                poly_px = _scaled_poly(border, render_width, render_height)
            except Exception:
                if sp_id not in superpixels_by_id:
                    continue
                poly_px = superpixels_by_id[sp_id]
        annotations_by_sp[sp_id] = AnnotationData(
            sp_id=sp_id,
            code=int(anno["code"]),
            parent_scribble=tuple(int(v) for v in (anno.get("parent_scribble") or [])),
            parent_intersect=bool(anno.get("parent_intersect", True)),
            poly_px=poly_px,
        )

    return LoadedState(
        path=path,
        checkpoint_index=_extract_state_index(path),
        n_scribbles=len(scribbles),
        width=width,
        height=height,
        render_width=render_width,
        render_height=render_height,
        method=chosen_method,
        scribbles=scribbles,
        annotations_by_sp=annotations_by_sp,
        superpixels_by_id=superpixels_by_id,
    )


def discover_state_files(run_dir: Path) -> List[Path]:
    files = list(run_dir.glob("state_*.json")) + list(run_dir.glob("state_*.json.gz"))
    return sorted(files, key=_extract_state_index)


def discover_run_dirs(input_path: Path) -> List[Path]:
    if discover_state_files(input_path):
        return [input_path]

    run_dirs = sorted({p.parent for p in input_path.rglob("state_*.json")})
    run_dirs += sorted({p.parent for p in input_path.rglob("state_*.json.gz")})
    unique = sorted({p.resolve() for p in run_dirs})
    return [Path(p) for p in unique]


def build_events(states: Sequence[LoadedState]) -> List[Event]:
    if not states:
        return []

    events: List[Event] = []
    prev = LoadedState(
        path=Path("<synthetic>"),
        checkpoint_index=0,
        n_scribbles=0,
        width=states[0].width,
        height=states[0].height,
        render_width=states[0].render_width,
        render_height=states[0].render_height,
        method=states[0].method,
        scribbles=[],
        annotations_by_sp={},
        superpixels_by_id={},
    )

    for cur in states:
        prev_sids = {s.sid for s in prev.scribbles}
        new_scribbles = tuple(s for s in cur.scribbles if s.sid not in prev_sids)

        direct_sp_ids: List[int] = []
        propagated_sp_ids: List[int] = []
        for sp_id, anno in cur.annotations_by_sp.items():
            prev_anno = prev.annotations_by_sp.get(sp_id)
            if prev_anno is not None and prev_anno.signature() == anno.signature():
                continue
            if anno.parent_intersect:
                direct_sp_ids.append(sp_id)
            else:
                propagated_sp_ids.append(sp_id)

        events.append(
            Event(
                prev_step=prev.n_scribbles,
                cur_step=cur.n_scribbles,
                checkpoint_index=cur.checkpoint_index,
                step_delta=max(0, cur.n_scribbles - prev.n_scribbles),
                exact=(cur.n_scribbles - prev.n_scribbles) == 1,
                new_scribbles=new_scribbles,
                direct_sp_ids=tuple(sorted(direct_sp_ids)),
                propagated_sp_ids=tuple(sorted(propagated_sp_ids)),
            )
        )
        prev = cur

    return events


def load_optional_image(image_path: Optional[Path], render_width: int, render_height: int) -> Optional[np.ndarray]:
    if image_path is None:
        return None
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB").resize((render_width, render_height), Image.Resampling.BILINEAR)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def find_image_for_run(run_dir: Path, image: Optional[Path], image_dir: Optional[Path]) -> Optional[Path]:
    if image is not None:
        return image
    if image_dir is None:
        return None
    for ext in _IMAGE_EXTS:
        candidate = image_dir / f"{run_dir.name}{ext}"
        if candidate.exists():
            return candidate
    return None


def blank_background(width: int, height: int) -> np.ndarray:
    x_grad = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = 28 + 18 * x_grad
    base[..., 1] = 30 + 22 * y_grad
    base[..., 2] = 34 + 28 * (1.0 - x_grad * 0.6)
    return np.clip(base, 0, 255).astype(np.uint8)


def _blend_fill(img: np.ndarray, color_to_polys: Dict[Tuple[int, int, int], List[np.ndarray]], alpha: float) -> None:
    if not color_to_polys:
        return
    overlay = img.copy()
    for color, polys in color_to_polys.items():
        if polys:
            cv2.fillPoly(overlay, polys, color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def _draw_superpixel_borders(img: np.ndarray, polys: Iterable[np.ndarray], color: Tuple[int, int, int], thickness: int) -> None:
    for poly in polys:
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def render_panel(
    state: LoadedState,
    base_image: Optional[np.ndarray],
    annotations_by_sp: Dict[int, AnnotationData],
    highlight_direct: Sequence[int],
    highlight_prop: Sequence[int],
    scribbles: Sequence[ScribbleData],
    class_info: Sequence[Tuple[str, str]],
    show_borders: bool = True,
) -> np.ndarray:
    panel = base_image.copy() if base_image is not None else blank_background(state.render_width, state.render_height)

    anno_groups: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
    for anno in annotations_by_sp.values():
        class_idx = max(0, min(len(class_info) - 1, int(anno.code) - 1))
        anno_groups[_hex_to_bgr(class_info[class_idx][1])].append(anno.poly_px)
    _blend_fill(panel, anno_groups, alpha=0.42)

    if show_borders:
        _draw_superpixel_borders(panel, state.superpixels_by_id.values(), _BORDER_BGR, thickness=1)

    if highlight_direct:
        overlay = panel.copy()
        direct_polys = [state.annotations_by_sp[sp_id].poly_px for sp_id in highlight_direct if sp_id in state.annotations_by_sp]
        if direct_polys:
            cv2.fillPoly(overlay, direct_polys, _DIRECT_BRG)
            cv2.addWeighted(overlay, 0.58, panel, 0.42, 0.0, dst=panel)
            _draw_superpixel_borders(panel, direct_polys, (15, 15, 15), thickness=2)

    if highlight_prop:
        overlay = panel.copy()
        prop_polys = [state.annotations_by_sp[sp_id].poly_px for sp_id in highlight_prop if sp_id in state.annotations_by_sp]
        if prop_polys:
            cv2.fillPoly(overlay, prop_polys, _PROP_BRG)
            cv2.addWeighted(overlay, 0.50, panel, 0.50, 0.0, dst=panel)
            _draw_superpixel_borders(panel, prop_polys, (30, 30, 30), thickness=2)

    for scrib in scribbles:
        class_idx = max(0, min(len(class_info) - 1, int(scrib.code) - 1))
        color = _hex_to_bgr(class_info[class_idx][1])
        pts = scrib.points_px
        cv2.polylines(panel, [pts], False, (255, 255, 255), thickness=6, lineType=cv2.LINE_AA)
        cv2.polylines(panel, [pts], False, color, thickness=3, lineType=cv2.LINE_AA)
        first_pt = tuple(int(v) for v in pts[0, 0])
        label_pt = (min(panel.shape[1] - 80, first_pt[0] + 6), max(18, first_pt[1] - 6))
        cv2.putText(panel, f"#{scrib.sid}", label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(panel, f"#{scrib.sid}", label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return panel


def _draw_text_block(canvas: np.ndarray, lines: Sequence[str], origin: Tuple[int, int], color: Tuple[int, int, int], scale: float, line_gap: int) -> None:
    x, y = origin
    for idx, line in enumerate(lines):
        yy = y + idx * line_gap
        cv2.putText(canvas, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 5, 5), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose_canvas(
    panel: np.ndarray,
    run_name: str,
    title: str,
    subtitle: str,
    footer: str,
) -> np.ndarray:
    header_h = 96
    footer_h = 64
    h, w = panel.shape[:2]
    canvas = np.full((header_h + h + footer_h, w, 3), _BG_BGR, dtype=np.uint8)
    canvas[:header_h] = _PANEL_BG_BGR
    canvas[header_h:header_h + h] = panel
    canvas[header_h + h:] = _PANEL_BG_BGR

    _draw_text_block(canvas, [run_name], (18, 28), _TEXT_SUB_BGR, 0.55, 24)
    _draw_text_block(canvas, [title], (18, 58), _TEXT_MAIN_BGR, 0.78, 26)
    _draw_text_block(canvas, [subtitle], (18, 84), _TEXT_SUB_BGR, 0.52, 24)
    _draw_text_block(canvas, [footer], (18, header_h + h + 38), _TEXT_SUB_BGR, 0.50, 22)
    return canvas


def make_intro_frame(run_name: str, width: int, height: int, final_step: int, max_gap: int) -> np.ndarray:
    canvas = np.full((height, width, 3), _BG_BGR, dtype=np.uint8)
    lines = [
        "Interactive Annotation Replay",
        run_name,
        f"Final saved step: {final_step}",
    ]
    if max_gap <= 1:
        lines.append("Exact replay from saved checkpoints.")
    else:
        lines.append(f"Coarse replay: the largest gap is {max_gap} scribbles.")
        lines.append("For exact replay, rerun evaluate_interactive_annotation.py with --save_every 1.")
    y0 = max(80, height // 3)
    for idx, line in enumerate(lines):
        scale = 1.0 if idx == 0 else 0.72
        gap = 44 if idx == 0 else 36
        yy = y0 + idx * gap
        _draw_text_block(canvas, [line], (36, yy), _TEXT_MAIN_BGR if idx == 0 else _TEXT_SUB_BGR, scale, gap)
    return canvas


def metrics_footer(metrics: Optional[Dict[str, float]]) -> str:
    if not metrics:
        return "Direct hits = orange | Propagation = cyan | Colored fill = current annotation"
    parts = []
    if "miou" in metrics:
        parts.append(f"mIoU {metrics['miou']:.3f}")
    if "coverage" in metrics:
        parts.append(f"coverage {metrics['coverage']:.3f}")
    if "annotation_precision" in metrics:
        parts.append(f"precision {metrics['annotation_precision']:.3f}")
    if "annotated_px" in metrics:
        parts.append(f"annotated_px {int(round(metrics['annotated_px']))}")
    legend = "Direct = orange | Propagation = cyan"
    return " | ".join(parts + [legend]) if parts else legend


def render_run_video(
    run_dir: Path,
    out_path: Path,
    image_path: Optional[Path],
    fps: float,
    max_side: int,
    method: Optional[str],
    timing: TimingConfig,
    logger: logging.Logger,
) -> Path:
    state_files = discover_state_files(run_dir)
    if not state_files:
        raise FileNotFoundError(f"No state_*.json files found in {run_dir}")

    states = [load_state(path, max_side=max_side, method=method) for path in state_files]
    events = build_events(states)
    metrics_by_step = _read_metrics_csv(run_dir / "metrics.csv")

    base_image = load_optional_image(image_path, states[0].render_width, states[0].render_height) if image_path else None

    frame_w = states[0].render_width
    frame_h = states[0].render_height + 96 + 64
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")
    try:
        max_gap = max((event.step_delta for event in events), default=0)
        intro = make_intro_frame(run_dir.name, frame_w, frame_h, states[-1].n_scribbles, max_gap)
        for _ in range(max(1, timing.intro_frames)):
            writer.write(intro)

        prev_annotations: Dict[int, AnnotationData] = {}
        prev_step = 0
        for state, event in zip(states, events):
            metrics = metrics_by_step.get(state.n_scribbles)
            footer = metrics_footer(metrics)

            before_panel = render_panel(
                state=state,
                base_image=base_image,
                annotations_by_sp=prev_annotations,
                highlight_direct=[],
                highlight_prop=[],
                scribbles=[],
                class_info=_build_default_class_info(max(1, max((a.code for a in state.annotations_by_sp.values()), default=1))),
            )
            before_title = f"Before step {state.n_scribbles}" if event.step_delta == 1 else f"Before checkpoint {state.checkpoint_index}"
            before_sub = f"Saved annotations: {len(prev_annotations)} superpixels"
            before_canvas = compose_canvas(before_panel, run_dir.name, before_title, before_sub, footer)
            for _ in range(max(1, timing.pre_frames)):
                writer.write(before_canvas)

            class_info = _build_default_class_info(max(1, max((a.code for a in state.annotations_by_sp.values()), default=1)))
            new_scribbles = list(event.new_scribbles)
            focus_title = (
                f"Step {state.n_scribbles}: scribble {new_scribbles[0].sid}"
                if event.exact and len(new_scribbles) == 1
                else f"Checkpoint {state.checkpoint_index}: +{event.step_delta} scribbles"
            )
            focus_sub = (
                f"Direct hits: {len(event.direct_sp_ids)} | Propagated: {len(event.propagated_sp_ids)}"
                if event.step_delta > 0
                else "No new scribbles in this checkpoint."
            )

            if event.step_delta > 0:
                direct_panel = render_panel(
                    state=state,
                    base_image=base_image,
                    annotations_by_sp=prev_annotations,
                    highlight_direct=event.direct_sp_ids,
                    highlight_prop=[],
                    scribbles=new_scribbles,
                    class_info=class_info,
                )
                direct_canvas = compose_canvas(direct_panel, run_dir.name, focus_title, focus_sub, footer)
                for _ in range(max(1, timing.direct_frames)):
                    writer.write(direct_canvas)

                prop_panel = render_panel(
                    state=state,
                    base_image=base_image,
                    annotations_by_sp=prev_annotations,
                    highlight_direct=event.direct_sp_ids,
                    highlight_prop=event.propagated_sp_ids,
                    scribbles=new_scribbles,
                    class_info=class_info,
                )
                prop_canvas = compose_canvas(prop_panel, run_dir.name, focus_title, focus_sub, footer)
                for _ in range(max(1, timing.prop_frames)):
                    writer.write(prop_canvas)

            final_panel = render_panel(
                state=state,
                base_image=base_image,
                annotations_by_sp=state.annotations_by_sp,
                highlight_direct=[],
                highlight_prop=[],
                scribbles=[],
                class_info=class_info,
            )
            final_title = f"Committed state after step {state.n_scribbles}"
            if event.step_delta > 1:
                final_title += f" (+{event.step_delta} saved together)"
            final_sub = f"Annotated superpixels: {len(state.annotations_by_sp)} | Previous step: {prev_step}"
            final_canvas = compose_canvas(final_panel, run_dir.name, final_title, final_sub, footer)
            for _ in range(max(1, timing.final_frames)):
                writer.write(final_canvas)

            prev_annotations = dict(state.annotations_by_sp)
            prev_step = state.n_scribbles

        outro_panel = render_panel(
            state=states[-1],
            base_image=base_image,
            annotations_by_sp=states[-1].annotations_by_sp,
            highlight_direct=[],
            highlight_prop=[],
            scribbles=[],
            class_info=_build_default_class_info(max(1, max((a.code for a in states[-1].annotations_by_sp.values()), default=1))),
        )
        outro_canvas = compose_canvas(
            outro_panel,
            run_dir.name,
            f"Final saved state: step {states[-1].n_scribbles}",
            f"Total annotated superpixels: {len(states[-1].annotations_by_sp)}",
            metrics_footer(metrics_by_step.get(states[-1].n_scribbles)),
        )
        for _ in range(max(1, timing.outro_frames)):
            writer.write(outro_canvas)
    finally:
        writer.release()
    logger.info("Saved video: %s", out_path)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Single result directory or batch root with state_*.json files")
    ap.add_argument("--out", default=None, help="Output .mp4 path for a single run or directory for batch mode")
    ap.add_argument("--image", default=None, help="Optional original image for a single run")
    ap.add_argument("--image_dir", default=None, help="Optional image directory for batch mode; matched by run directory stem")
    ap.add_argument("--method", default=None, help="Method key from the saved state; defaults to the first available one")
    ap.add_argument("--fps", type=float, default=8.0, help="Output video FPS")
    ap.add_argument("--max_side", type=int, default=1600, help="Resize the longest rendered side to at most this value")
    ap.add_argument("--intro_seconds", type=float, default=1.2)
    ap.add_argument("--pre_seconds", type=float, default=0.35)
    ap.add_argument("--direct_seconds", type=float, default=0.55)
    ap.add_argument("--prop_seconds", type=float, default=0.70)
    ap.add_argument("--final_seconds", type=float, default=0.35)
    ap.add_argument("--outro_seconds", type=float, default=1.0)
    return ap


def output_targets(run_dirs: Sequence[Path], out_arg: Optional[str]) -> Dict[Path, Path]:
    if not run_dirs:
        return {}

    if out_arg is None:
        return {run_dir: run_dir / "interactive_annotation_replay.mp4" for run_dir in run_dirs}

    out_path = Path(out_arg)
    if len(run_dirs) == 1 and out_path.suffix.lower() == ".mp4":
        return {run_dirs[0]: out_path}
    if len(run_dirs) > 1 and out_path.suffix.lower() == ".mp4":
        raise ValueError("--out must point to a directory when --input contains multiple runs.")

    out_dir = out_path
    return {run_dir: out_dir / f"{run_dir.name}.mp4" for run_dir in run_dirs}


def setup_logger() -> logging.Logger:
    log = logging.getLogger("render_interactive_annotation_video")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(handler)
    return log


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logger()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    run_dirs = discover_run_dirs(input_path)
    if not run_dirs:
        raise FileNotFoundError(f"No result directories with state_*.json found under {input_path}")

    outputs = output_targets(run_dirs, args.out)
    image_dir = Path(args.image_dir) if args.image_dir else None
    single_image = Path(args.image) if args.image else None
    if len(run_dirs) > 1 and single_image is not None:
        raise ValueError("--image is only supported for a single run. Use --image_dir for batch mode.")
    if image_dir is not None and not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    timing = TimingConfig(
        intro_frames=max(1, int(round(args.intro_seconds * args.fps))),
        pre_frames=max(1, int(round(args.pre_seconds * args.fps))),
        direct_frames=max(1, int(round(args.direct_seconds * args.fps))),
        prop_frames=max(1, int(round(args.prop_seconds * args.fps))),
        final_frames=max(1, int(round(args.final_seconds * args.fps))),
        outro_frames=max(1, int(round(args.outro_seconds * args.fps))),
    )

    for run_dir in run_dirs:
        image_path = find_image_for_run(run_dir, single_image if len(run_dirs) == 1 else None, image_dir)
        render_run_video(
            run_dir=run_dir,
            out_path=outputs[run_dir],
            image_path=image_path,
            fps=float(args.fps),
            max_side=int(args.max_side),
            method=args.method,
            timing=timing,
            logger=logger,
        )


if __name__ == "__main__":
    main()
