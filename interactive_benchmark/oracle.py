from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import OracleSelection, PromptPayload
from .legacy_oracle import LargestBadRegionGenerator, parse_region_selection_cycle


PROMPT_PROFILE_MAP = {
    "interformer": "point",
    "seem": "scribble",
    "semantic_sam": "point",
    "segnext": "point",
    "iseg": "line",
    "mock_click": "point",
    "mock_line": "line",
    "mock_scribble": "scribble",
}


def mark_prompt_used_mask(used_mask: np.ndarray, prompt: PromptPayload) -> None:
    h, w = used_mask.shape
    prompt_type = str(prompt.prompt_type).lower()
    if prompt_type == "box" and prompt.box is not None:
        x0, y0, x1, y1 = prompt.box
        ix0 = max(0, min(w - 1, int(round(float(x0) * float(w)))))
        ix1 = max(0, min(w - 1, int(round(float(x1) * float(w)))))
        iy0 = max(0, min(h - 1, int(round(float(y0) * float(h)))))
        iy1 = max(0, min(h - 1, int(round(float(y1) * float(h)))))
        used_mask[min(iy0, iy1):max(iy0, iy1) + 1, min(ix0, ix1):max(ix0, ix1) + 1] = True
        return

    if not prompt.points:
        return
    pts = np.asarray(prompt.points, dtype=np.float32)
    pts_px = np.empty_like(pts, dtype=np.int32)
    pts_px[:, 0] = np.clip(np.round(pts[:, 0] * float(w)).astype(np.int32), 0, w - 1)
    pts_px[:, 1] = np.clip(np.round(pts[:, 1] * float(h)).astype(np.int32), 0, h - 1)
    radius = 2 if prompt_type == "point" else 3
    for x, y in pts_px:
        used_mask[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)] = True
    for idx in range(max(0, pts_px.shape[0] - 1)):
        x0, y0 = pts_px[idx]
        x1, y1 = pts_px[idx + 1]
        n = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1
        xs = np.linspace(x0, x1, n).round().astype(int)
        ys = np.linspace(y0, y1, n).round().astype(int)
        for x, y in zip(xs, ys):
            used_mask[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)] = True


@dataclass
class InteractionOracle:
    gt_mask: np.ndarray
    num_classes: int
    prompt_type: str
    seed: int = 0
    margin: int = 2
    border_margin: int = 3
    no_overlap: bool = True
    region_selection_cycle: str = "miou_gain,largest_error,unannotated"

    def __post_init__(self) -> None:
        self._generator = LargestBadRegionGenerator(
            gt_mask=np.asarray(self.gt_mask, dtype=np.int32),
            num_classes=int(self.num_classes),
            seed=int(self.seed),
            margin=int(self.margin),
            border_margin=int(self.border_margin),
            no_overlap=bool(self.no_overlap),
            max_retries=200,
            region_selection_cycle=parse_region_selection_cycle(self.region_selection_cycle),
        )

    def _prompt_from_polyline(
        self,
        *,
        class_id: int,
        interaction_id: int,
        pts01: np.ndarray,
        component_mask: np.ndarray,
    ) -> PromptPayload:
        points = [(float(item[0]), float(item[1])) for item in np.asarray(pts01, dtype=np.float32)]
        prompt_type = str(self.prompt_type).lower()
        if prompt_type == "point":
            if points:
                point = points[len(points) // 2]
            else:
                ys, xs = np.where(component_mask)
                point = (float(xs.mean()) / float(component_mask.shape[1]), float(ys.mean()) / float(component_mask.shape[0]))
            return PromptPayload(prompt_type="point", class_id=class_id, interaction_id=interaction_id, points=[point])
        if prompt_type == "line":
            if len(points) >= 2:
                line_points = [points[0], points[-1]]
            else:
                line_points = list(points)
            return PromptPayload(prompt_type="line", class_id=class_id, interaction_id=interaction_id, points=line_points)
        if prompt_type in {"scribble", "mark"}:
            return PromptPayload(prompt_type="scribble", class_id=class_id, interaction_id=interaction_id, points=points)
        ys, xs = np.where(component_mask)
        if xs.size > 0:
            box = (
                float(xs.min()) / float(component_mask.shape[1]),
                float(ys.min()) / float(component_mask.shape[0]),
                float(xs.max()) / float(component_mask.shape[1]),
                float(ys.max()) / float(component_mask.shape[0]),
            )
            return PromptPayload(prompt_type="box", class_id=class_id, interaction_id=interaction_id, box=box)
        return PromptPayload(prompt_type="point", class_id=class_id, interaction_id=interaction_id, points=points[:1])

    def next_interaction(
        self,
        pred_mask: np.ndarray,
        used_mask: np.ndarray,
        class_interaction_counts: Optional[list[int]],
        interaction_id: int,
    ) -> OracleSelection:
        pred_mask = np.asarray(pred_mask, dtype=np.int32)
        used_mask = np.asarray(used_mask, dtype=bool)
        selection_mode = self._generator._current_region_selection_mode()
        cid, comp, allowed, pts01 = self._generator._select_class_component(
            pred_mask,
            used_mask,
            class_scribble_counts=class_interaction_counts,
            selection_mode=selection_mode,
        )
        if cid is None and selection_mode != "miou_gain":
            cid, comp, allowed, pts01 = self._generator._select_class_component(
                pred_mask,
                used_mask,
                class_scribble_counts=class_interaction_counts,
                selection_mode="miou_gain",
            )
        if cid is None or comp is None or allowed is None or pts01 is None:
            cid, pts01 = self._generator.make_scribble(
                pred_mask,
                used_mask,
                class_scribble_counts=class_interaction_counts,
            )
            comp = (self.gt_mask == int(cid)) & (pred_mask != int(cid))
            allowed = comp.copy()
        else:
            self._generator._advance_region_selection_cycle()

        prompt = self._prompt_from_polyline(
            class_id=int(cid),
            interaction_id=int(interaction_id),
            pts01=np.asarray(pts01, dtype=np.float32),
            component_mask=np.asarray(comp, dtype=bool),
        )
        return OracleSelection(
            class_id=int(cid),
            prompt=prompt,
            target_mask=np.asarray(comp, dtype=bool),
            allowed_mask=np.asarray(allowed, dtype=bool),
            component_mask=np.asarray(comp, dtype=bool),
        )

    def report_result(self, progress: bool) -> None:
        self._generator.report_last_result(bool(progress))
