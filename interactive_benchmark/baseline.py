from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _coerce_value(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except Exception:
        return value


def load_metrics_csv(metrics_path: Path) -> list[dict[str, Any]]:
    with open(metrics_path, "r", encoding="utf-8", newline="") as fh:
        return [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(fh)]


def normalize_current_pipeline_metrics(
    rows: list[dict[str, Any]],
    *,
    interaction_budgets: list[int],
    image_name: str,
    method_id: str = "current_pipeline",
    display_name: str = "Current Pipeline",
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda item: int(float(item.get("step") or 0)))
    normalized: list[dict[str, Any]] = []
    for budget in sorted(int(item) for item in interaction_budgets):
        chosen = rows_sorted[0]
        for row in rows_sorted:
            step = int(float(row.get("step") or 0))
            if step <= int(budget):
                chosen = row
            else:
                break
        normalized.append(
            {
                "image": image_name,
                "method_id": method_id,
                "display_name": display_name,
                "interaction_budget": int(budget),
                "step": int(float(chosen.get("step") or 0)),
                "n_interactions": int(float(chosen.get("n_scribbles") or 0)),
                "coverage": float(chosen.get("coverage") or 0.0),
                "annotation_precision": float(chosen.get("annotation_precision") or 0.0),
                "miou": float(chosen.get("miou") or 0.0),
                "total_ink_px": float(chosen.get("total_ink_px") or 0.0),
                "status": "ok",
                "prompt_type": "scribble",
                "source": "evaluate_interactive_annotation.py",
            }
        )
    return normalized


def run_current_pipeline(
    *,
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    interaction_budgets: list[int],
    pipeline_args: dict[str, Any] | None = None,
    python_bin: str | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_args = dict(pipeline_args or {})
    cmd = [
        str(python_bin or sys.executable),
        "evaluate_interactive_annotation.py",
        "--image",
        str(image_path.resolve()),
        "--mask",
        str(mask_path.resolve()),
        "--out",
        str(output_dir.resolve()),
        "--save_every",
        "1",
        "--scribbles",
        str(max(int(item) for item in interaction_budgets)),
    ]
    for key, value in pipeline_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, (dict, list)):
            cmd.extend([flag, json.dumps(value, ensure_ascii=False)])
            continue
        cmd.extend([flag, str(value)])

    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"current_pipeline failed: {proc.returncode}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    metrics_rows = load_metrics_csv(output_dir / "metrics.csv")
    return normalize_current_pipeline_metrics(
        metrics_rows,
        interaction_budgets=interaction_budgets,
        image_name=image_path.stem,
    )
