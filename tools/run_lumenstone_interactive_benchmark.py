#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interactive_benchmark.registry import create_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive benchmark on Lumenstone test_01 for cuda:0,cuda:2.")
    parser.add_argument(
        "--image",
        default="/home/n.yakovlev/datasets/lumenstone/S1_v2/imgs/test/test_01.jpg",
    )
    parser.add_argument(
        "--mask",
        default="/home/n.yakovlev/datasets/lumenstone/S1_v2/masks/test/test_01.png",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/n.yakovlev/datasets/lumenstone/S1_v2",
        help="Dataset root containing imgs/ and masks/ for optional training.",
    )
    parser.add_argument(
        "--python-bin",
        default=str(ROOT / ".venv" / "bin" / "python"),
    )
    parser.add_argument(
        "--methods",
        default="current_pipeline,interformer,segnext,seem,semantic_sam,iseg",
    )
    parser.add_argument(
        "--interaction-budgets",
        default="1,3,5,10,20",
    )
    parser.add_argument(
        "--gpus",
        default="0,2",
        help="Physical GPU ids.",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "interactive_benchmark"),
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force re-training of deep_slic checkpoint used by current_pipeline.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def ensure_deep_slic_checkpoint(args: argparse.Namespace, run_root: Path) -> Path:
    default_ckpt = ROOT / "artifacts" / "training" / "s1_v2_neural_parallel_20260412" / "deep_slic_n40" / "best_model.pth"
    if default_ckpt.exists() and not args.force_train:
        return default_ckpt

    train_out = run_root / "training_deep_slic"
    train_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python_bin),
        "train_neural_superpixels.py",
        "--img_dir",
        str(Path(args.dataset_root) / "imgs"),
        "--mask_dir",
        str(Path(args.dataset_root) / "masks"),
        "--method",
        "deep_slic",
        "--method_config",
        json.dumps(
            {
                "nspix": 80,
                "fdim": 20,
                "niter": 5,
                "backbone_width": 32,
                "compactness": 8.0,
                "color_scale": 0.26,
                "pos_scale": 2.5,
            }
        ),
        "--out_dir",
        str(train_out),
        "--train_iter",
        "500",
        "--test_interval",
        "100",
        "--print_interval",
        "50",
        "--max_ram_gb",
        str(args.memory_limit_gb),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to train deep_slic checkpoint.\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    trained = train_out / "best_model.pth"
    if not trained.exists():
        raise RuntimeError(f"Training succeeded but checkpoint not found: {trained}")
    return trained


def collect_method_availability(methods: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in methods:
        if method_id == "current_pipeline":
            rows.append(
                {
                    "method_id": method_id,
                    "available": True,
                    "reason": "",
                }
            )
            continue
        adapter = create_adapter(method_id)
        available, reason = adapter.is_available()
        rows.append(
            {
                "method_id": method_id,
                "display_name": adapter.display_name,
                "prompt_type": adapter.prompt_type,
                "available": bool(available),
                "reason": "" if reason is None else str(reason),
            }
        )
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_markdown_report(
    run_root: Path,
    *,
    methods: list[str],
    availability: list[dict[str, Any]],
    per_gpu_dirs: dict[str, Path],
) -> None:
    lines: list[str] = []
    lines.append("# Lumenstone test_01 interactive benchmark")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Methods requested: `{','.join(methods)}`")
    lines.append("- Devices: `cuda:0`, `cuda:2`")
    lines.append("")
    lines.append("## Availability before run")
    lines.append("")
    lines.append("| Method | Available | Reason |")
    lines.append("| --- | --- | --- |")
    for item in availability:
        lines.append(
            f"| {item.get('method_id','')} | {bool(item.get('available'))} | {str(item.get('reason','')).replace('|','/')} |"
        )
    lines.append("")

    for device_tag, out_dir in per_gpu_dirs.items():
        summary_csv = out_dir / "summary.csv"
        resource_csv = out_dir / "resource_summary.csv"
        lines.append(f"## Results for `{device_tag}`")
        lines.append("")
        if not summary_csv.exists():
            lines.append(f"- Missing `{summary_csv}`")
            lines.append("")
            continue
        quality = read_csv(summary_csv)
        lines.append("### Quality")
        lines.append("")
        lines.append("| Method | Budget | Mean mIoU | Mean Coverage | Mean Precision | Status |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in quality:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("method_id", "")),
                        str(row.get("interaction_budget", "")),
                        str(row.get("mean_miou", "")),
                        str(row.get("mean_coverage", "")),
                        str(row.get("mean_annotation_precision", "")),
                        str(row.get("status", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
        if resource_csv.exists():
            resources = read_csv(resource_csv)
            lines.append("### Resource usage")
            lines.append("")
            lines.append("| Method | Mean step time (s) | Max peak RSS (bytes) | Max peak GPU mem (MiB) | Status |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in resources:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("method_id", "")),
                            str(row.get("mean_wall_time_sec", "")),
                            str(row.get("max_peak_rss_bytes", "")),
                            str(row.get("max_peak_gpu_memory_mib", "")),
                            str(row.get("status", "")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    (run_root / "final_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = parse_csv_list(args.methods)
    gpu_ids = parse_csv_list(args.gpus)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root).resolve() / f"lumenstone_test01_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    checkpoint = ensure_deep_slic_checkpoint(args, run_root)
    availability = collect_method_availability(methods)
    (run_root / "method_availability.json").write_text(
        json.dumps(availability, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    current_pipeline_args = {
        "method": "deep_slic",
        "method_config": {
            "weight_path": str(checkpoint),
            "nspix": 80,
            "fdim": 20,
            "niter": 5,
            "backbone_width": 32,
            "compactness": 8.0,
            "color_scale": 0.26,
            "pos_scale": 2.5,
        },
        "sensitivity": 1.8,
        "seed": 0,
    }

    per_gpu_dirs: dict[str, Path] = {}
    for gpu in gpu_ids:
        device_tag = f"cuda:{gpu}"
        out_dir = run_root / f"run_{device_tag.replace(':', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        per_gpu_dirs[device_tag] = out_dir
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["INTERACTIVE_BENCHMARK_DEVICE"] = "cuda:0"
        cmd = [
            str(args.python_bin),
            "benchmark_interactive_methods.py",
            "--image",
            str(Path(args.image).resolve()),
            "--mask",
            str(Path(args.mask).resolve()),
            "--output-dir",
            str(out_dir),
            "--methods",
            ",".join(methods),
            "--interaction-budgets",
            args.interaction_budgets,
            "--python-bin",
            str(args.python_bin),
            "--current-pipeline-args",
            json.dumps(current_pipeline_args, ensure_ascii=False),
            "--memory-limit-gb",
            str(args.memory_limit_gb),
            "--device-tag",
            device_tag,
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        log_payload = {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
        (out_dir / "run_log.json").write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"benchmark failed for {device_tag}. See {out_dir / 'run_log.json'}")

    write_markdown_report(
        run_root=run_root,
        methods=methods,
        availability=availability,
        per_gpu_dirs=per_gpu_dirs,
    )
    print(str(run_root))


if __name__ == "__main__":
    main()
