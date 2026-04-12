from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class StepMetrics:
    step: int
    n_scribbles: int
    total_ink_px: float
    mean_ink_px: float
    annotated_px: int
    correctly_annotated_px: int
    total_px: int
    coverage: float
    annotation_precision: float
    miou: float
    per_class_iou: list[float]
    per_class_n_scribbles: list[int]
    per_class_ink_px: list[float]

    @property
    def miou_per_scribble(self) -> float:
        return self.miou / max(1, self.n_scribbles)

    @property
    def miou_per_1kpx(self) -> float:
        return self.miou / max(1e-6, self.total_ink_px / 1000.0)

    @property
    def correct_px_per_scribble(self) -> float:
        return self.correctly_annotated_px / max(1, self.n_scribbles)

    @property
    def correct_px_per_px_ink(self) -> float:
        return self.correctly_annotated_px / max(1e-6, self.total_ink_px)

    @property
    def quality_x_coverage(self) -> float:
        return self.miou * self.coverage

    def as_flat_dict(self, class_names: list[str]) -> dict[str, object]:
        row: dict[str, object] = {
            "step": int(self.step),
            "n_scribbles": int(self.n_scribbles),
            "total_ink_px": round(float(self.total_ink_px), 2),
            "mean_ink_px": round(float(self.mean_ink_px), 2),
            "annotated_px": int(self.annotated_px),
            "correctly_annotated_px": int(self.correctly_annotated_px),
            "total_px": int(self.total_px),
            "coverage": round(float(self.coverage), 6),
            "annotation_precision": round(float(self.annotation_precision), 6),
            "miou": round(float(self.miou), 6),
            "miou_per_scribble": round(float(self.miou_per_scribble), 8),
            "miou_per_1kpx": round(float(self.miou_per_1kpx), 8),
            "correct_px_per_scribble": round(float(self.correct_px_per_scribble), 4),
            "correct_px_per_px_ink": round(float(self.correct_px_per_px_ink), 8),
            "quality_x_coverage": round(float(self.quality_x_coverage), 6),
        }
        for idx, name in enumerate(class_names):
            iou = self.per_class_iou[idx] if idx < len(self.per_class_iou) else float("nan")
            row[f"iou_{name}"] = round(float(iou), 6) if not math.isnan(float(iou)) else float("nan")
            row[f"n_scrib_{name}"] = int(self.per_class_n_scribbles[idx]) if idx < len(self.per_class_n_scribbles) else 0
            ink = self.per_class_ink_px[idx] if idx < len(self.per_class_ink_px) else 0.0
            row[f"ink_px_{name}"] = round(float(ink), 2)
        return row


def compute_ious(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    ious: list[float] = []
    for class_id in range(int(num_classes)):
        pred_c = pred == int(class_id)
        gt_c = gt == int(class_id)
        inter = int(np.logical_and(pred_c, gt_c).sum())
        union = int(np.logical_or(pred_c, gt_c).sum())
        ious.append(inter / union if union > 0 else float("nan"))
    valid = [item for item in ious if not math.isnan(float(item))]
    miou = float(np.mean(valid)) if valid else float("nan")
    return miou, ious


def build_step_metrics(
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    step: int,
    n_interactions: int,
    total_ink_px: float,
    per_class_n_interactions: list[int],
    per_class_ink_px: list[float],
) -> StepMetrics:
    annotated_mask = pred >= 0
    annotated_px = int(annotated_mask.sum())
    correct_mask = annotated_mask & (pred == gt)
    correctly_annotated_px = int(correct_mask.sum())
    total_px = int(pred.shape[0] * pred.shape[1])
    coverage = float(annotated_px) / float(max(1, total_px))
    annotation_precision = float(correctly_annotated_px) / float(max(1, annotated_px)) if annotated_px > 0 else 0.0
    miou, per_class_iou = compute_ious(pred, gt, int(num_classes))
    mean_ink = float(total_ink_px) / float(max(1, n_interactions)) if n_interactions > 0 else 0.0
    return StepMetrics(
        step=int(step),
        n_scribbles=int(n_interactions),
        total_ink_px=float(total_ink_px),
        mean_ink_px=float(mean_ink),
        annotated_px=int(annotated_px),
        correctly_annotated_px=int(correctly_annotated_px),
        total_px=int(total_px),
        coverage=float(coverage),
        annotation_precision=float(annotation_precision),
        miou=0.0 if math.isnan(float(miou)) else float(miou),
        per_class_iou=per_class_iou,
        per_class_n_scribbles=[int(v) for v in per_class_n_interactions],
        per_class_ink_px=[float(v) for v in per_class_ink_px],
    )


def save_batch_summary(
    summaries: dict[str, list[StepMetrics]],
    class_info: list[tuple[str, str]],
    out_path: Path,
) -> None:
    if not summaries:
        return
    rows: list[dict[str, object]] = []
    class_names = [item[0] for item in class_info]
    for image_name, history in summaries.items():
        if not history:
            continue
        row = {"image": image_name}
        row.update(history[-1].as_flat_dict(class_names))
        rows.append(row)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    means = {"image": "MEAN"}
    for key in fieldnames:
        if key == "image":
            continue
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float)) and not math.isnan(float(row[key]))
        ]
        means[key] = round(float(np.mean(values)), 6) if values else float("nan")
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writerow(means)


def plot_metrics(
    metrics_history: list[StepMetrics],
    class_info: list[tuple[str, str]],
    out_path: Path,
) -> None:
    if not metrics_history:
        return
    steps = [item.step for item in metrics_history]
    mious = [item.miou for item in metrics_history]
    coverages = [item.coverage for item in metrics_history]
    precisions = [item.annotation_precision for item in metrics_history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, mious, marker="o", label="mIoU")
    axes[0].plot(steps, coverages, marker="s", label="Coverage")
    axes[0].plot(steps, precisions, marker="^", label="Precision")
    axes[0].set_title("Quality vs interactions")
    axes[0].set_xlabel("Interaction")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(steps, [item.miou_per_scribble for item in metrics_history], marker="o", label="mIoU / interaction")
    axes[1].plot(steps, [item.miou_per_1kpx for item in metrics_history], marker="s", label="mIoU / 1kpx")
    axes[1].set_title("Efficiency")
    axes[1].set_xlabel("Interaction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    final = metrics_history[-1]
    class_names = [item[0] for item in class_info]
    bar_values = [
        0.0 if idx >= len(final.per_class_iou) or math.isnan(float(final.per_class_iou[idx])) else float(final.per_class_iou[idx])
        for idx in range(len(class_names))
    ]
    axes[2].bar(np.arange(len(class_names)), bar_values)
    axes[2].set_xticks(np.arange(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Final per-class IoU")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _group_mean(rows: list[dict[str, object]], key: str) -> float:
    values = [
        float(item[key])
        for item in rows
        if item.get(key) is not None and not math.isnan(float(item[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def plot_quality_vs_interactions(rows: list[dict[str, object]], out_path: Path) -> None:
    usable = [row for row in rows if row.get("status") == "ok"]
    if not usable:
        return
    grouped: dict[str, dict[int, list[dict[str, object]]]] = {}
    for row in usable:
        method = str(row["method_id"])
        budget = int(row["interaction_budget"])
        grouped.setdefault(method, {}).setdefault(budget, []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method_id, budget_rows in sorted(grouped.items()):
        budgets = sorted(budget_rows.keys())
        mean_miou = [_group_mean(budget_rows[budget], "miou") for budget in budgets]
        mean_cov = [_group_mean(budget_rows[budget], "coverage") for budget in budgets]
        mean_prec = [_group_mean(budget_rows[budget], "annotation_precision") for budget in budgets]
        axes[0].plot(budgets, mean_miou, marker="o", label=method_id)
        axes[1].plot(budgets, mean_cov, marker="o", label=method_id)
        axes[2].plot(budgets, mean_prec, marker="o", label=method_id)

    for axis, title in zip(axes, ("mIoU", "Coverage", "Precision")):
        axis.set_title(title)
        axis.set_xlabel("Interaction budget")
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
