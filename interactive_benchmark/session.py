from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .contracts import MaskProposal, PromptPayload


@dataclass
class SemanticCanvas:
    image_shape: tuple[int, int]
    labels: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.labels = np.full(self.image_shape, -1, dtype=np.int32)

    def rebuild(
        self,
        class_masks: dict[int, np.ndarray],
        class_update_order: dict[int, int],
    ) -> None:
        self.labels.fill(-1)
        ordered_classes = sorted(
            class_masks.keys(),
            key=lambda cid: (int(class_update_order.get(int(cid), -1)), int(cid)),
        )
        for class_id in ordered_classes:
            mask = np.asarray(class_masks[class_id], dtype=bool)
            self.labels[mask] = int(class_id)


@dataclass
class SessionState:
    image_shape: tuple[int, int]
    num_classes: int
    source_image_path: Optional[str] = None
    class_masks: dict[int, np.ndarray] = field(default_factory=dict)
    class_update_order: dict[int, int] = field(default_factory=dict)
    prompt_history: list[PromptPayload] = field(default_factory=list)
    prompts_by_class: dict[int, list[PromptPayload]] = field(default_factory=lambda: defaultdict(list))
    canvas: SemanticCanvas = field(init=False)

    def __post_init__(self) -> None:
        self.canvas = SemanticCanvas(self.image_shape)

    def apply_proposal(
        self,
        proposal: MaskProposal,
        *,
        interaction_id: int,
        prompt: Optional[PromptPayload] = None,
    ) -> None:
        class_id = int(proposal.class_id)
        if class_id < 0 or class_id >= int(self.num_classes):
            raise ValueError(f"Invalid class id: {class_id}")
        mask = np.asarray(proposal.mask, dtype=bool)
        if tuple(mask.shape) != tuple(self.image_shape):
            raise ValueError(
                f"Proposal mask shape mismatch: expected {self.image_shape}, got {mask.shape}"
            )
        self.class_masks[class_id] = mask.copy()
        self.class_update_order[class_id] = int(interaction_id)
        if prompt is not None:
            self.prompt_history.append(prompt)
            self.prompts_by_class[class_id].append(prompt)
        self.canvas.rebuild(self.class_masks, self.class_update_order)

    def to_runtime_payload(self) -> dict[str, object]:
        return {
            "image_shape": [int(self.image_shape[0]), int(self.image_shape[1])],
            "num_classes": int(self.num_classes),
            "source_image_path": self.source_image_path,
            "class_update_order": {str(k): int(v) for k, v in self.class_update_order.items()},
            "prompt_history": [item.to_dict() for item in self.prompt_history],
        }
