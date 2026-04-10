#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from tools.repo_conventions import (
    ALLOWED_ROOT_NO_MOVE,
    CORE_TOP_LEVEL_NAMES,
    classify_top_level_name,
    is_run_dir,
    is_summary_file,
    should_ignore_walk_dir,
)


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan the repository and build run inventories.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--output-dir",
        default="reports/generated",
        help="Directory for generated inventory artifacts.",
    )
    return parser


def parse_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def directory_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for current_root, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if not should_ignore_walk_dir(name)]
        for filename in filenames:
            current_path = Path(current_root) / filename
            try:
                total += current_path.stat().st_size
            except OSError:
                continue
    return total


def count_matches(path: Path, predicate) -> int:
    total = 0
    for current_root, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if not should_ignore_walk_dir(name)]
        for filename in filenames:
            if predicate(filename):
                total += 1
    return total


def classify_status(path_text: str) -> str:
    lowered = path_text.lower()
    if "/uncategorized/" in lowered:
        return "uncategorized"
    if "/debug/" in lowered or "_tmp_eval_" in lowered or "tmp_replay_debug" in lowered:
        return "debug"
    if "smoke" in lowered:
        return "smoke"
    if "/training/" in lowered:
        return "main"
    if "/case_studies/" in lowered or "_quarter_run" in lowered or "_two_quarters" in lowered:
        return "case-study"
    if "/sweeps/" in lowered or "/refinement/" in lowered:
        return "archive"
    return "main"


def classify_family(path_text: str) -> str:
    lowered = path_text.lower()
    if "/interactive_runs/" in lowered:
        return "interactive-run"
    if "/sweeps/" in lowered:
        return "sweep"
    if "/refinement/" in lowered:
        return "refinement"
    if "/postprocessing/" in lowered:
        return "postprocessing"
    if "/case_studies/" in lowered:
        return "case-study"
    if "/training/" in lowered:
        return "training"
    if "/precomputed/" in lowered:
        return "precomputed"
    if "/debug/" in lowered:
        return "debug"
    return "uncategorized"


def summarize_metrics_csv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return {"row_count": 0}

    final_row = rows[-1]
    best_row = max(rows, key=lambda row: parse_float(row.get("miou")) or -1.0)
    return {
        "row_count": len(rows),
        "final_step": parse_float(final_row.get("step")),
        "final_miou": parse_float(final_row.get("miou")),
        "final_coverage": parse_float(final_row.get("coverage")),
        "final_annotation_precision": parse_float(final_row.get("annotation_precision")),
        "final_scribbles": parse_float(final_row.get("n_scribbles")),
        "final_total_ink_px": parse_float(final_row.get("total_ink_px")),
        "best_miou": parse_float(best_row.get("miou")),
        "best_step": parse_float(best_row.get("step")),
    }


def summarize_dynamic_metrics_csv(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return {"dynamic_row_count": 0}
    final_row = rows[-1]
    return {
        "dynamic_row_count": len(rows),
        "dynamic_final_miou": parse_float(final_row.get("miou") or final_row.get("dyn_miou")),
        "dynamic_final_coverage": parse_float(final_row.get("coverage") or final_row.get("cov")),
        "dynamic_final_precision": parse_float(
            final_row.get("annotation_precision") or final_row.get("prec")
        ),
    }


def summarize_run_log(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "log_no_progress_count": 0,
        "method": None,
        "image": None,
        "classes": None,
        "propagation": None,
        "start_time": None,
        "end_time": None,
    }
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return payload

    for line in lines:
        match = TIMESTAMP_RE.match(line)
        if match:
            stamp = match.group(1)
            payload["start_time"] = payload["start_time"] or stamp
            payload["end_time"] = stamp
        if "No progress after step" in line:
            payload["log_no_progress_count"] += 1
        if "SP method:" in line and payload["method"] is None:
            payload["method"] = line.split("SP method:", 1)[1].strip()
        if "Image:" in line and payload["image"] is None:
            payload["image"] = line.split("Image:", 1)[1].strip()
        if "Classes (" in line and payload["classes"] is None:
            payload["classes"] = line.split("INFO]", 1)[-1].strip()
        if "Propagation:" in line and payload["propagation"] is None:
            payload["propagation"] = line.split("Propagation:", 1)[1].strip()
    if payload["start_time"] and payload["end_time"]:
        try:
            start_dt = datetime.strptime(payload["start_time"], "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(payload["end_time"], "%Y-%m-%d %H:%M:%S")
            payload["duration_s"] = int((end_dt - start_dt).total_seconds())
        except ValueError:
            payload["duration_s"] = None
    else:
        payload["duration_s"] = None
    return payload


def summarize_summary_csv(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"summary_row_count": 0}, []

    score_key = "mean_miou" if "mean_miou" in rows[0] else "miou"
    valid_rows = [row for row in rows if parse_float(row.get(score_key)) is not None]
    best_row = max(valid_rows, key=lambda row: parse_float(row.get(score_key)) or -1.0) if valid_rows else rows[0]
    summary = {
        "summary_row_count": len(rows),
        "summary_metric_key": score_key,
        "best_label": best_row.get("label") or best_row.get("image") or best_row.get("method"),
        "best_score": parse_float(best_row.get(score_key)),
    }
    aggregates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        aggregates.append(
            {
                "source": str(path),
                "kind": "summary_row",
                "rank_hint": index,
                "label": row.get("label") or row.get("image") or row.get("method") or f"row_{index}",
                "method": row.get("method"),
                "metric_key": score_key,
                "metric_value": parse_float(row.get(score_key)),
                "coverage": parse_float(row.get("mean_coverage") or row.get("coverage")),
                "annotation_precision": parse_float(
                    row.get("mean_annotation_precision") or row.get("annotation_precision")
                ),
                "superpixels": parse_float(row.get("mean_superpixels") or row.get("superpixels")),
                "precompute_s": parse_float(row.get("precompute_s")),
                "eval_s": parse_float(row.get("eval_s")),
                "total_s": parse_float(row.get("total_s")),
            }
        )
    return summary, aggregates


def summarize_batch_csv(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"batch_row_count": 0}, []

    mean_row = next((row for row in rows if str(row.get("image")).upper() == "MEAN"), rows[-1])
    image_rows = [row for row in rows if str(row.get("image")).upper() != "MEAN"]
    best_image = max(image_rows, key=lambda row: parse_float(row.get("miou")) or -1.0) if image_rows else mean_row
    worst_image = min(image_rows, key=lambda row: parse_float(row.get("miou")) or 1.0) if image_rows else mean_row
    summary = {
        "batch_row_count": len(rows),
        "mean_miou": parse_float(mean_row.get("miou")),
        "mean_coverage": parse_float(mean_row.get("coverage")),
        "mean_annotation_precision": parse_float(mean_row.get("annotation_precision")),
        "best_image": best_image.get("image"),
        "best_image_miou": parse_float(best_image.get("miou")),
        "worst_image": worst_image.get("image"),
        "worst_image_miou": parse_float(worst_image.get("miou")),
    }
    aggregates = [
        {
            "source": str(path),
            "kind": "batch_mean",
            "label": mean_row.get("image") or "MEAN",
            "metric_key": "miou",
            "metric_value": parse_float(mean_row.get("miou")),
            "coverage": parse_float(mean_row.get("coverage")),
            "annotation_precision": parse_float(mean_row.get("annotation_precision")),
        }
    ]
    return summary, aggregates


def summarize_json_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"json_summary_error": True}, []

    if path.name == "comparison_summary.json":
        methods = payload.get("methods", {})
        best_method_name = None
        best_method_score = None
        aggregates: list[dict[str, Any]] = []
        for method_name, method_payload in methods.items():
            score = parse_float(method_payload.get("miou"))
            if best_method_score is None or (score is not None and score > best_method_score):
                best_method_name = method_name
                best_method_score = score
            aggregates.append(
                {
                    "source": str(path),
                    "kind": "comparison_method",
                    "label": method_name,
                    "metric_key": "miou",
                    "metric_value": score,
                    "delta_miou_vs_baseline": parse_float(method_payload.get("delta_miou_vs_baseline")),
                    "delta_acc_vs_baseline": parse_float(method_payload.get("delta_acc_vs_baseline")),
                }
            )
        return {
            "method_count": len(methods),
            "best_method": best_method_name,
            "best_method_miou": best_method_score,
            "n_images": payload.get("n_images"),
        }, aggregates

    before = payload.get("before", {})
    after = payload.get("after", {})
    aggregates = []
    if before:
        aggregates.append(
            {
                "source": str(path),
                "kind": "summary_before",
                "label": "before",
                "metric_key": "miou",
                "metric_value": parse_float(before.get("miou")),
                "pixel_accuracy": parse_float(before.get("pixel_accuracy")),
            }
        )
    if after:
        aggregates.append(
            {
                "source": str(path),
                "kind": "summary_after",
                "label": "after",
                "metric_key": "miou",
                "metric_value": parse_float(after.get("miou")),
                "pixel_accuracy": parse_float(after.get("pixel_accuracy")),
            }
        )
    return {
        "n_images": payload.get("n_images"),
        "before_miou": parse_float(before.get("miou")),
        "after_miou": parse_float(after.get("miou")),
        "delta_miou": parse_float(payload.get("delta", {}).get("miou")),
    }, aggregates


def top_level_inventory(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        classification = classify_top_level_name(child.name, child.is_dir())
        size_bytes = directory_size_bytes(child) if child.exists() else 0
        entries.append(
            {
                "name": child.name,
                "path": relpath(child, root),
                "is_dir": child.is_dir(),
                "classification": classification,
                "size_bytes": size_bytes,
                "tracked_zone": (
                    "core"
                    if child.name in CORE_TOP_LEVEL_NAMES or child.name in ALLOWED_ROOT_NO_MOVE
                    else classification
                ),
            }
        )
    return entries


def scan_runs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_entries: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not should_ignore_walk_dir(name)]
        current_path = Path(current_root)
        rel_current = relpath(current_path, root)

        if is_run_dir(current_path):
            metrics_path = current_path / "metrics.csv"
            dynamic_path = current_path / "dynamic_metrics.csv"
            log_path = current_path / "run.log"
            entry: dict[str, Any] = {
                "path": rel_current,
                "kind": "interactive_run",
                "family": classify_family(rel_current),
                "status": classify_status(rel_current),
                "size_bytes": directory_size_bytes(current_path),
                "frame_count": count_matches(current_path, lambda name: name.startswith("frame_") and name.endswith(".png")),
                "state_count": count_matches(current_path, lambda name: name.startswith("state_") and name.endswith(".json")),
                "spanno_count": count_matches(
                    current_path,
                    lambda name: name.endswith(".spanno.json") or name.endswith(".spanno.json.gz"),
                ),
                "video_count": count_matches(current_path, lambda name: name.endswith(".mp4")),
            }
            if metrics_path.exists():
                entry.update(summarize_metrics_csv(metrics_path))
                aggregates.append(
                    {
                        "source": rel_current,
                        "kind": "interactive_final",
                        "label": rel_current,
                        "metric_key": "miou",
                        "metric_value": entry.get("final_miou"),
                        "coverage": entry.get("final_coverage"),
                        "annotation_precision": entry.get("final_annotation_precision"),
                        "scribbles": entry.get("final_scribbles"),
                        "total_ink_px": entry.get("final_total_ink_px"),
                    }
                )
            if dynamic_path.exists():
                entry.update(summarize_dynamic_metrics_csv(dynamic_path))
            if log_path.exists():
                entry.update(summarize_run_log(log_path))
            if metrics_path.exists() and entry.get("final_miou") is None:
                anomalies.append({"path": rel_current, "issue": "metrics_without_final_miou"})
            run_entries.append(entry)
            continue

        for filename in filenames:
            file_path = current_path / filename
            if not is_summary_file(file_path):
                continue
            rel_file = relpath(file_path, root)
            family = classify_family(rel_file)
            status = classify_status(rel_file)
            entry = {
                "path": rel_file,
                "kind": filename,
                "family": family,
                "status": status,
                "size_bytes": file_path.stat().st_size,
            }
            summary_aggregates: list[dict[str, Any]] = []
            if filename == "summary.csv":
                summary, summary_aggregates = summarize_summary_csv(file_path)
                entry.update(summary)
            elif filename == "batch_summary.csv":
                summary, summary_aggregates = summarize_batch_csv(file_path)
                entry.update(summary)
            else:
                summary, summary_aggregates = summarize_json_summary(file_path)
                entry.update(summary)
            for aggregate in summary_aggregates:
                aggregate["source"] = rel_file
            run_entries.append(entry)
            aggregates.extend(summary_aggregates)
            if filename == "summary.csv" and entry.get("best_score") is None:
                anomalies.append({"path": rel_file, "issue": "summary_without_score"})
    return run_entries, aggregates, anomalies


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.repo_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    top_level = top_level_inventory(root)
    runs, aggregates, anomalies = scan_runs(root)

    summary = {
        "repo_root": str(root),
        "top_level_entry_count": len(top_level),
        "run_entry_count": len(runs),
        "aggregate_entry_count": len(aggregates),
        "anomaly_count": len(anomalies),
        "metrics_csv_count": sum(1 for item in runs if item.get("kind") == "interactive_run"),
        "dynamic_metrics_count": sum(1 for item in runs if item.get("dynamic_row_count")),
        "summary_like_count": sum(1 for item in runs if item.get("kind") != "interactive_run"),
    }

    write_json(output_dir / "repo_inventory.json", {"summary": summary, "top_level": top_level})
    write_csv(output_dir / "repo_inventory.csv", top_level)
    write_json(output_dir / "run_catalog.json", {"summary": summary, "runs": runs})
    write_csv(output_dir / "run_catalog.csv", runs)
    write_json(output_dir / "aggregate_catalog.json", {"summary": summary, "aggregates": aggregates})
    write_csv(output_dir / "aggregate_catalog.csv", aggregates)
    write_json(output_dir / "anomalies.json", {"summary": summary, "anomalies": anomalies})

    print(f"Inventory written to: {output_dir}")
    print(
        "Counts:"
        f" top_level={len(top_level)}"
        f" runs={len(runs)}"
        f" aggregates={len(aggregates)}"
        f" anomalies={len(anomalies)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
