from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _polyline_length_px(points: list[tuple[float, float]], height: int, width: int) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    scale = np.array([[float(width), float(height)]], dtype=np.float64)
    pts_px = pts * scale
    deltas = np.diff(pts_px, axis=0)
    return float(np.sum(np.sqrt(np.sum(deltas ** 2, axis=1))))


@dataclass(frozen=True)
class PromptPayload:
    prompt_type: str
    class_id: int
    interaction_id: int
    points: list[tuple[float, float]] = field(default_factory=list)
    box: Optional[tuple[float, float, float, float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_type": self.prompt_type,
            "class_id": int(self.class_id),
            "interaction_id": int(self.interaction_id),
            "points": [[float(x), float(y)] for x, y in self.points],
            "box": (
                None
                if self.box is None
                else [float(self.box[0]), float(self.box[1]), float(self.box[2]), float(self.box[3])]
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptPayload":
        points = [
            (float(item[0]), float(item[1]))
            for item in (payload.get("points") or [])
        ]
        box_raw = payload.get("box")
        box = None if box_raw is None else tuple(float(v) for v in box_raw)
        return cls(
            prompt_type=str(payload["prompt_type"]),
            class_id=int(payload["class_id"]),
            interaction_id=int(payload["interaction_id"]),
            points=points,
            box=box,
            metadata=dict(payload.get("metadata") or {}),
        )

    def ink_px(self, height: int, width: int) -> float:
        prompt_type = str(self.prompt_type).lower()
        if prompt_type == "point":
            return 1.0
        if prompt_type in {"line", "scribble", "mark"}:
            return _polyline_length_px(self.points, height, width)
        if prompt_type == "box" and self.box is not None:
            x0, y0, x1, y1 = self.box
            return 2.0 * (abs(float(x1) - float(x0)) * float(width) + abs(float(y1) - float(y0)) * float(height))
        return max(1.0, _polyline_length_px(self.points, height, width))


@dataclass
class MaskProposal:
    class_id: int
    mask: np.ndarray
    score: Optional[float] = None
    source: str = ""
    candidate_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError("MaskProposal.mask must be a 2-D boolean array.")


@dataclass
class OracleSelection:
    class_id: int
    prompt: PromptPayload
    target_mask: np.ndarray
    allowed_mask: np.ndarray
    component_mask: np.ndarray

    def __post_init__(self) -> None:
        self.target_mask = np.asarray(self.target_mask, dtype=bool)
        self.allowed_mask = np.asarray(self.allowed_mask, dtype=bool)
        self.component_mask = np.asarray(self.component_mask, dtype=bool)


class MethodAdapter(ABC):
    method_id: str
    display_name: str
    prompt_type: str

    @abstractmethod
    def is_available(self) -> tuple[bool, Optional[str]]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image_rgb: np.ndarray,
        session_state: "SessionState",
        prompt: PromptPayload,
        work_dir: Path,
    ) -> list[MaskProposal]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        available, reason = self.is_available()
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "prompt_type": self.prompt_type,
            "available": bool(available),
            "unavailable_reason": reason,
        }


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return float(inter) / float(union)


def choose_best_proposal(
    proposals: list[MaskProposal],
    target_mask: np.ndarray,
) -> MaskProposal:
    if not proposals:
        raise ValueError("Expected at least one proposal.")
    if len(proposals) == 1:
        return proposals[0]

    best = None
    best_key: tuple[float, float, str] | None = None
    for proposal in proposals:
        iou = binary_iou(proposal.mask, target_mask)
        score = float(proposal.score) if proposal.score is not None and not math.isnan(float(proposal.score)) else float("-inf")
        candidate = (float(iou), score, str(proposal.candidate_id or ""))
        if best_key is None or candidate > best_key:
            best_key = candidate
            best = proposal
    assert best is not None
    return best
