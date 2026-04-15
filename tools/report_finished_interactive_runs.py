#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MethodStats:
    method: str
    n_ok: int
    n_failed: int
    miou_mean: float
    miou_variance: float
    miou_std: float


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_last_miou(metrics_csv_path: Path) -> float:
    if not metrics_csv_path.exists():
        return float("nan")
    try:
        with metrics_csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return float("nan")
        return float(rows[-1].get("miou", "nan"))
    except Exception:
        return float("nan")


def _compute_from_runs(run_dir: Path) -> list[MethodStats]:
    interactive_log = _load_json(run_dir / "interactive_log.json") or []
    failed_pairs = {
        (row.get("method"), str(row.get("image", "")).replace(".jpg", ""))
        for row in interactive_log
        if not row.get("ok", False)
    }

    method_dirs = sorted([p for p in (run_dir / "runs").glob("*") if p.is_dir()])
    stats: list[MethodStats] = []
    for method_dir in method_dirs:
        method = method_dir.name
        vals = []
        n_failed = 0
        for test_dir in sorted([p for p in method_dir.glob("test_*") if p.is_dir()]):
            if (method, test_dir.name) in failed_pairs:
                n_failed += 1
                continue
            miou = _read_last_miou(test_dir / "metrics.csv")
            if not math.isnan(miou):
                vals.append(miou)
        arr = np.array(vals, dtype=float)
        stats.append(
            MethodStats(
                method=method,
                n_ok=int(arr.size),
                n_failed=n_failed,
                miou_mean=float(np.mean(arr)) if arr.size else float("nan"),
                miou_variance=float(np.var(arr)) if arr.size else float("nan"),
                miou_std=float(np.std(arr)) if arr.size else float("nan"),
            )
        )
    return stats


def _is_finished(run_dir: Path) -> bool:
    if not (run_dir / "interactive_log.json").exists():
        return False
    # The orchestrator always writes this at the very end.
    return (run_dir / "summary_metrics.json").exists()


def _fmt_float(v: float) -> str:
    if v is None or math.isnan(v):
        return "NaN"
    return f"{v:.6f}"


def build_report(artifacts_dir: Path) -> tuple[list[dict[str, Any]], str]:
    runs = sorted([p for p in artifacts_dir.glob("s1v2_optimal4_*") if p.is_dir()])
    finished = [r for r in runs if _is_finished(r)]

    rows: list[dict[str, Any]] = []
    lines: list[str] = []
    lines.append("# Finished Interactive Runs Report")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Artifacts root: `{artifacts_dir}`")
    lines.append(f"- Finished runs found: **{len(finished)}**")
    lines.append("")

    for run_dir in finished:
        manifest = _load_json(run_dir / "manifest.json") or {}
        stats = _compute_from_runs(run_dir)
        rows.append(
            {
                "run_name": run_dir.name,
                "path": str(run_dir),
                "created_at": manifest.get("created_at"),
                "scribbles": manifest.get("scribbles"),
                "downscale": manifest.get("downscale"),
                "methods": [s.__dict__ for s in stats],
            }
        )
        lines.append(f"## `{run_dir.name}`")
        lines.append("")
        lines.append(f"- Path: `{run_dir}`")
        lines.append(f"- Created: `{manifest.get('created_at', 'unknown')}`")
        lines.append(
            f"- Params: `scribbles={manifest.get('scribbles', 'unknown')}`, "
            f"`downscale={manifest.get('downscale', 'unknown')}`"
        )
        lines.append("")
        lines.append("| Method | OK images | Failed images | mIoU mean | mIoU var | mIoU std |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for s in sorted(stats, key=lambda x: x.method):
            lines.append(
                f"| {s.method} | {s.n_ok} | {s.n_failed} | "
                f"{_fmt_float(s.miou_mean)} | {_fmt_float(s.miou_variance)} | {_fmt_float(s.miou_std)} |"
            )
        lines.append("")

    if not finished:
        lines.append("No finished runs found.")
        lines.append("")

    return rows, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build report for finished interactive runs.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/interactive_runs"),
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/experiments/interactive_runs_finished.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/experiments/interactive_runs_finished.json"),
        help="Output json report path.",
    )
    args = parser.parse_args()

    rows, report_md = build_report(args.artifacts_dir.resolve())
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report_md + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote markdown: {args.out_md}")
    print(f"Wrote json: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
