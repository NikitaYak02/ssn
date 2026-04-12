#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluate_interactive_annotation import DEFAULT_CLASS_INFO, discover_image_pairs
from interactive_benchmark.baseline import run_current_pipeline
from interactive_benchmark.contracts import choose_best_proposal
from interactive_benchmark.oracle import InteractionOracle, mark_prompt_used_mask
from interactive_benchmark.registry import DEFAULT_EXTERNAL_METHODS, create_adapter
from interactive_benchmark.session import SessionState
from interactive_benchmark.shared import build_step_metrics, plot_quality_vs_interactions, write_json


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


def parse_json_value(raw: str | None) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    return json.loads(text)


def load_mask_as_ids(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.mode in ("P", "L", "I;16", "I"):
        arr = np.array(mask)
    else:
        arr = np.array(mask.convert("RGB"))[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {arr.shape}")
    return arr.astype(np.int32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified benchmark for current scribble pipeline and external interactive methods."
    )
    parser.add_argument("--image", default=None, help="Single RGB image.")
    parser.add_argument("--mask", default=None, help="Single GT mask.")
    parser.add_argument("--images", default=None, help="Batch image directory.")
    parser.add_argument("--masks", default=None, help="Batch mask directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--methods",
        default="current_pipeline," + ",".join(DEFAULT_EXTERNAL_METHODS),
        help="Comma-separated method ids.",
    )
    parser.add_argument(
        "--interaction-budgets",
        default="1,3,5,10,20,50,100",
        help="Comma-separated interaction budgets.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional batch limit.")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--border_margin", type=int, default=3)
    parser.add_argument("--no_overlap", action="store_true")
    parser.add_argument(
        "--region_selection_cycle",
        default="miou_gain,largest_error,unannotated",
    )
    parser.add_argument(
        "--current-pipeline-args",
        default=None,
        help="JSON string or path with extra args for evaluate_interactive_annotation.py.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Optional python executable for current_pipeline.",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="Fail instead of skipping unavailable external methods.",
    )
    return parser


def resolve_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.image and args.mask:
        return [(Path(args.image).resolve(), Path(args.mask).resolve())]
    if args.images and args.masks:
        pairs = discover_image_pairs(args.images, args.masks)
        if args.limit is not None:
            pairs = pairs[: int(args.limit)]
        return [(Path(image).resolve(), Path(mask).resolve()) for image, mask in pairs]
    raise ValueError("Provide either --image/--mask or --images/--masks.")


def resolve_num_classes(gt_mask: np.ndarray, num_classes_arg: int | None) -> int:
    if num_classes_arg is not None:
        return int(num_classes_arg)
    return int(np.max(gt_mask)) + 1


def resolve_class_info(num_classes: int) -> list[tuple[str, str]]:
    if num_classes <= len(DEFAULT_CLASS_INFO):
        return list(DEFAULT_CLASS_INFO[:num_classes])
    return list(DEFAULT_CLASS_INFO) + [
        (f"cls{idx}", "#aaaaaa") for idx in range(len(DEFAULT_CLASS_INFO), num_classes)
    ]


def make_result_row(
    *,
    image_name: str,
    method_id: str,
    display_name: str,
    prompt_type: str,
    interaction_budget: int,
    metrics,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "image": image_name,
        "method_id": method_id,
        "display_name": display_name,
        "prompt_type": prompt_type,
        "interaction_budget": int(interaction_budget),
        "status": status,
    }
    if metrics is not None:
        row.update(
            {
                "step": int(metrics.step),
                "n_interactions": int(metrics.n_scribbles),
                "miou": float(metrics.miou),
                "coverage": float(metrics.coverage),
                "annotation_precision": float(metrics.annotation_precision),
                "total_ink_px": float(metrics.total_ink_px),
                "annotated_px": int(metrics.annotated_px),
                "correctly_annotated_px": int(metrics.correctly_annotated_px),
            }
        )
    if extra:
        row.update(extra)
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method_id"]), int(row["interaction_budget"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method_id, interaction_budget), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ok_rows = [row for row in bucket if row.get("status") == "ok"]
        display_name = str(bucket[0].get("display_name") or method_id)
        prompt_type = str(bucket[0].get("prompt_type") or "")
        if ok_rows:
            summary_rows.append(
                {
                    "method_id": method_id,
                    "display_name": display_name,
                    "prompt_type": prompt_type,
                    "interaction_budget": int(interaction_budget),
                    "n_images_ok": int(len(ok_rows)),
                    "mean_miou": float(np.mean([float(row["miou"]) for row in ok_rows])),
                    "mean_coverage": float(np.mean([float(row["coverage"]) for row in ok_rows])),
                    "mean_annotation_precision": float(
                        np.mean([float(row["annotation_precision"]) for row in ok_rows])
                    ),
                    "mean_total_ink_px": float(np.mean([float(row["total_ink_px"]) for row in ok_rows])),
                    "status": "ok",
                }
            )
        else:
            summary_rows.append(
                {
                    "method_id": method_id,
                    "display_name": display_name,
                    "prompt_type": prompt_type,
                    "interaction_budget": int(interaction_budget),
                    "n_images_ok": 0,
                    "mean_miou": float("nan"),
                    "mean_coverage": float("nan"),
                    "mean_annotation_precision": float("nan"),
                    "mean_total_ink_px": float("nan"),
                    "status": bucket[0].get("status") or "failed",
                    "error": bucket[0].get("error"),
                }
            )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_leaderboard(summary_rows: list[dict[str, Any]], max_budget: int) -> str:
    finals = [
        row
        for row in summary_rows
        if int(row["interaction_budget"]) == int(max_budget) and row.get("status") == "ok"
    ]
    finals.sort(key=lambda item: float(item["mean_miou"]), reverse=True)
    lines = [
        "# Interactive Benchmark Leaderboard",
        "",
        f"Final ranking at interaction budget `{max_budget}`.",
        "",
        "| Rank | Method | Prompt | Mean mIoU | Mean Coverage | Mean Precision | Images |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(finals, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["display_name"]),
                    str(row["prompt_type"]),
                    f"{float(row['mean_miou']):.4f}",
                    f"{float(row['mean_coverage']):.4f}",
                    f"{float(row['mean_annotation_precision']):.4f}",
                    str(int(row["n_images_ok"])),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_adapter_on_pair(
    *,
    adapter,
    image_path: Path,
    mask_path: Path,
    interaction_budgets: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    image_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    gt_mask = load_mask_as_ids(mask_path)
    if gt_mask.shape[:2] != image_rgb.shape[:2]:
        gt_mask = np.array(
            Image.fromarray(gt_mask.astype(np.int32), mode="I").resize(
                (image_rgb.shape[1], image_rgb.shape[0]),
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.int32,
        )
    num_classes = resolve_num_classes(gt_mask, args.num_classes)
    session = SessionState(
        image_shape=(image_rgb.shape[0], image_rgb.shape[1]),
        num_classes=num_classes,
        source_image_path=str(image_path.resolve()),
    )
    oracle = InteractionOracle(
        gt_mask=gt_mask,
        num_classes=num_classes,
        prompt_type=adapter.prompt_type,
        seed=int(args.seed),
        margin=int(args.margin),
        border_margin=int(args.border_margin),
        no_overlap=bool(args.no_overlap),
        region_selection_cycle=str(args.region_selection_cycle),
    )
    used_mask = np.zeros(gt_mask.shape, dtype=bool)
    per_class_counts = [0] * num_classes
    per_class_ink = [0.0] * num_classes
    total_ink = 0.0
    budget_set = set(int(item) for item in interaction_budgets)
    max_budget = max(budget_set)
    rows: list[dict[str, Any]] = []

    for interaction_id in range(1, max_budget + 1):
        prev_pred = session.canvas.labels
        prev_annotated = int((prev_pred >= 0).sum())
        prev_correct = int(((prev_pred >= 0) & (prev_pred == gt_mask)).sum())

        selection = oracle.next_interaction(
            pred_mask=session.canvas.labels,
            used_mask=used_mask,
            class_interaction_counts=per_class_counts,
            interaction_id=interaction_id,
        )
        prompt = selection.prompt
        ink = float(prompt.ink_px(image_rgb.shape[0], image_rgb.shape[1]))
        total_ink += ink
        per_class_counts[int(selection.class_id)] += 1
        per_class_ink[int(selection.class_id)] += ink
        if args.no_overlap:
            mark_prompt_used_mask(used_mask, prompt)

        step_dir = output_dir / image_path.stem / adapter.method_id / f"step_{interaction_id:03d}"
        proposals = adapter.predict(image_rgb, session, prompt, step_dir)
        chosen = choose_best_proposal(proposals, selection.target_mask)
        session.apply_proposal(chosen, interaction_id=interaction_id, prompt=prompt)

        cur_pred = session.canvas.labels
        cur_annotated = int((cur_pred >= 0).sum())
        cur_correct = int(((cur_pred >= 0) & (cur_pred == gt_mask)).sum())
        oracle.report_result(cur_annotated > prev_annotated or cur_correct > prev_correct)

        metrics = build_step_metrics(
            pred=cur_pred,
            gt=gt_mask,
            num_classes=num_classes,
            step=interaction_id,
            n_interactions=interaction_id,
            total_ink_px=total_ink,
            per_class_n_interactions=per_class_counts,
            per_class_ink_px=per_class_ink,
        )
        prompt_path = step_dir / "prompt.json"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            prompt_path,
            {
                "prompt": prompt.to_dict(),
                "chosen_candidate": chosen.candidate_id,
                "num_candidates": len(proposals),
            },
        )
        if interaction_id in budget_set:
            rows.append(
                make_result_row(
                    image_name=image_path.stem,
                    method_id=adapter.method_id,
                    display_name=adapter.display_name,
                    prompt_type=adapter.prompt_type,
                    interaction_budget=interaction_id,
                    metrics=metrics,
                    status="ok",
                    extra={
                        "selected_candidate": chosen.candidate_id,
                        "num_candidates": len(proposals),
                    },
                )
            )
    return rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_csv_list(args.methods)
    interaction_budgets = sorted(set(parse_int_csv(args.interaction_budgets)))
    pairs = resolve_pairs(args)
    current_pipeline_args = parse_json_value(args.current_pipeline_args) or {}

    all_rows: list[dict[str, Any]] = []
    unavailable_methods: list[dict[str, Any]] = []

    for method_id in methods:
        if method_id == "current_pipeline":
            continue
        adapter = create_adapter(method_id)
        available, reason = adapter.is_available()
        if not available:
            if args.fail_on_unavailable:
                raise RuntimeError(f"{method_id} unavailable: {reason}")
            unavailable_methods.append(
                {
                    "method_id": method_id,
                    "display_name": adapter.display_name,
                    "prompt_type": adapter.prompt_type,
                    "reason": reason,
                }
            )

    for image_path, mask_path in pairs:
        if "current_pipeline" in methods:
            try:
                rows = run_current_pipeline(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_dir=output_dir / "runs" / image_path.stem / "current_pipeline",
                    interaction_budgets=interaction_budgets,
                    pipeline_args=current_pipeline_args,
                    python_bin=args.python_bin,
                )
                all_rows.extend(rows)
            except Exception as exc:
                for budget in interaction_budgets:
                    all_rows.append(
                        {
                            "image": image_path.stem,
                            "method_id": "current_pipeline",
                            "display_name": "Current Pipeline",
                            "prompt_type": "scribble",
                            "interaction_budget": int(budget),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

        for method_id in methods:
            if method_id == "current_pipeline":
                continue
            adapter = create_adapter(method_id)
            available, reason = adapter.is_available()
            if not available:
                for budget in interaction_budgets:
                    all_rows.append(
                        {
                            "image": image_path.stem,
                            "method_id": adapter.method_id,
                            "display_name": adapter.display_name,
                            "prompt_type": adapter.prompt_type,
                            "interaction_budget": int(budget),
                            "status": "skipped",
                            "error": reason,
                        }
                    )
                continue
            try:
                rows = run_adapter_on_pair(
                    adapter=adapter,
                    image_path=image_path,
                    mask_path=mask_path,
                    interaction_budgets=interaction_budgets,
                    args=args,
                    output_dir=output_dir / "runs",
                )
                all_rows.extend(rows)
            except Exception as exc:
                for budget in interaction_budgets:
                    all_rows.append(
                        {
                            "image": image_path.stem,
                            "method_id": adapter.method_id,
                            "display_name": adapter.display_name,
                            "prompt_type": adapter.prompt_type,
                            "interaction_budget": int(budget),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    per_step_csv = output_dir / "per_step.csv"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    leaderboard_md = output_dir / "leaderboard.md"
    plot_path = output_dir / "quality_vs_interactions.png"

    write_csv(per_step_csv, all_rows)
    summary_rows = summarize_rows(all_rows)
    write_csv(summary_csv, summary_rows)
    write_json(
        summary_json,
        {
            "interaction_budgets": interaction_budgets,
            "pairs": [[str(img), str(mask)] for img, mask in pairs],
            "summary": summary_rows,
            "unavailable_methods": unavailable_methods,
        },
    )
    plot_quality_vs_interactions(all_rows, plot_path)
    leaderboard_md.write_text(build_leaderboard(summary_rows, max(interaction_budgets)), encoding="utf-8")


if __name__ == "__main__":
    main()
