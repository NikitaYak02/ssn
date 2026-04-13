#!/usr/bin/env python3
"""Optuna-based parameter search for interactive superpixel annotation.

This script reuses the existing sweep/evaluation pipeline:
1. resize input image/mask once,
2. sample method-specific hyperparameters with Optuna,
3. cache precomputed `.spanno` files per configuration,
4. run `evaluate_interactive_annotation.py`,
5. optimize the final mIoU for a fixed scribble budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
from PIL import Image

from sweep_interactive_superpixels import (
    SweepCase,
    ensure_resized_inputs,
    run_case_worker,
    sanitize_token,
    set_worker_thread_limits,
)

METHOD_ALIASES = {
    "fwb": "felzenszwalb",
    "felsewalb": "felzenszwalb",
    "ws": "watershed",
    "slicm": "slic",
}
SUPPORTED_METHODS = ("slic", "watershed", "felzenszwalb", "ssn")
DEFAULT_METHODS = ",".join(SUPPORTED_METHODS)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}


class MemoryLimitExceeded(RuntimeError):
    """Raised when total RSS usage is above the configured limit."""


def parse_csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def normalize_method_name(raw: str) -> str:
    method = str(raw).strip().lower()
    if not method:
        raise ValueError("Method name must be non-empty.")
    return METHOD_ALIASES.get(method, method)


def parse_method_list(raw: str) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for token in parse_csv_list(raw):
        method = normalize_method_name(token)
        if method in seen:
            continue
        seen.add(method)
        methods.append(method)
    return methods


def parse_ignore_codes(raw: str) -> set[int]:
    return {int(v) for v in parse_csv_list(raw)}


def _iter_files(path: Path, suffixes: Iterable[str]) -> list[Path]:
    return sorted(
        [
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
        ],
        key=lambda p: p.name.lower(),
    )


def find_mask_for_image(image_path: Path, mask_dir: Path) -> Path | None:
    direct = mask_dir / image_path.name
    if direct.exists() and direct.is_file():
        return direct
    for suffix in MASK_EXTS:
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def select_max_class_image_pair(
    dataset_root: Path,
    split: str,
    ignore_codes: set[int],
) -> tuple[Path, Path, int]:
    img_dir = dataset_root / "imgs" / split
    mask_dir = dataset_root / "masks" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    pairs: list[tuple[Path, Path]] = []
    for image_path in _iter_files(img_dir, IMAGE_EXTS):
        mask_path = find_mask_for_image(image_path, mask_dir)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found in {img_dir} and {mask_dir}")

    best_image: Path | None = None
    best_mask: Path | None = None
    best_count = -1
    for image_path, mask_path in pairs:
        with Image.open(mask_path) as mask_im:
            mask_codes = np.asarray(mask_im, dtype=np.int32)
        uniq = np.unique(mask_codes)
        if ignore_codes:
            uniq = uniq[~np.isin(uniq, list(ignore_codes))]
        class_count = int(uniq.size)
        if class_count > best_count:
            best_count = class_count
            best_image = image_path
            best_mask = mask_path

    if best_image is None or best_mask is None:
        raise RuntimeError("Failed to select image/mask pair with max class count.")
    return best_image.resolve(), best_mask.resolve(), int(best_count)


def read_proc_rss_bytes(pid: int) -> int:
    status_path = Path("/proc") / str(pid) / "status"
    try:
        text = status_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return 0


def get_descendant_pids(root_pid: int) -> set[int]:
    ppid_to_children: dict[int, list[int]] = {}
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        stat_path = entry / "stat"
        try:
            stat_text = stat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if ") " not in stat_text:
            continue
        after = stat_text.split(") ", 1)[1]
        parts = after.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(entry.name)
            ppid = int(parts[1])
        except Exception:
            continue
        ppid_to_children.setdefault(ppid, []).append(pid)

    descendants: set[int] = set()
    stack = [root_pid]
    while stack:
        current = stack.pop()
        for child in ppid_to_children.get(current, []):
            if child in descendants:
                continue
            descendants.add(child)
            stack.append(child)
    return descendants


def get_total_rss_bytes(root_pid: int | None = None) -> int:
    pid = int(root_pid or os.getpid())
    pids = {pid, *get_descendant_pids(pid)}
    return sum(read_proc_rss_bytes(item) for item in pids)


class MemoryGuard:
    def __init__(self, limit_bytes: int):
        self.limit_bytes = int(limit_bytes)

    def check(self, stage: str) -> int:
        rss_bytes = get_total_rss_bytes()
        if rss_bytes > self.limit_bytes:
            limit_gb = self.limit_bytes / (1024**3)
            rss_gb = rss_bytes / (1024**3)
            raise MemoryLimitExceeded(
                f"Memory limit exceeded at {stage}: RSS={rss_gb:.2f} GiB > {limit_gb:.2f} GiB"
            )
        return rss_bytes


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Optuna-based hyperparameter search for superpixel methods."
    )
    ap.add_argument("--image", default=None, help="Optional explicit image path.")
    ap.add_argument("--mask", default=None, help="Optional explicit mask path.")
    ap.add_argument(
        "--dataset-root",
        default=None,
        help="Dataset root with imgs/<split> and masks/<split> to auto-select max-class image.",
    )
    ap.add_argument("--split", default="train", help="Dataset split for --dataset-root mode.")
    ap.add_argument(
        "--ignore-codes",
        default="255",
        help="Comma-separated mask class codes ignored when selecting max-class image.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--resize-scale", type=float, default=0.5)
    ap.add_argument("--scribbles", type=int, default=150)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel Optuna trials for each method.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--study-prefix", default="optuna_superpixels")
    ap.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. You may use '{method}' placeholder.",
    )
    ap.add_argument(
        "--method",
        default=None,
        help="Single method mode. Supported: slic, watershed, felzenszwalb, ssn (+aliases).",
    )
    ap.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help=f"Comma-separated method list for multi-study mode. Default: {DEFAULT_METHODS}",
    )
    ap.add_argument(
        "--memory-limit-gb",
        type=float,
        default=100.0,
        help="Total RSS limit for this process tree (GiB).",
    )
    ap.add_argument(
        "--cpu-only",
        action="store_true",
        default=True,
        help="Force CPU-only execution (default on).",
    )
    ap.add_argument(
        "--allow-gpu",
        action="store_false",
        dest="cpu_only",
        help="Allow GPU execution (overrides --cpu-only).",
    )

    grp_sim = ap.add_argument_group("Interactive evaluation")
    grp_sim.add_argument("--margin", type=int, default=2)
    grp_sim.add_argument("--border_margin", type=int, default=3)
    grp_sim.add_argument("--no_overlap", action="store_true")
    grp_sim.add_argument("--max_no_progress", type=int, default=12)
    grp_sim.add_argument(
        "--region_selection_cycle",
        default="miou_gain,largest_error,unannotated",
    )
    grp_sim.add_argument("--sensitivity", type=float, default=1.8)
    grp_sim.add_argument("--emb_weights", default=None)
    grp_sim.add_argument("--emb_threshold", type=float, default=0.988)
    grp_sim.add_argument("--num_classes", type=int, default=None)

    grp_ssn = ap.add_argument_group("SSN")
    grp_ssn.add_argument("--ssn-weights", default=None)
    return ap


def resolve_methods(args: argparse.Namespace) -> list[str]:
    if args.method is not None and str(args.method).strip():
        methods = [normalize_method_name(args.method)]
    else:
        methods = parse_method_list(args.methods)
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    return methods


def resolve_image_and_mask(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    if args.image and args.mask:
        image = str(Path(args.image).resolve())
        mask = str(Path(args.mask).resolve())
        return image, mask, {"selection_mode": "explicit_paths"}

    if args.image or args.mask:
        raise ValueError("Either provide both --image and --mask, or provide --dataset-root.")
    if not args.dataset_root:
        raise ValueError(
            "No input pair provided. Use --image/--mask or --dataset-root with --split."
        )

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    ignore_codes = parse_ignore_codes(args.ignore_codes)
    image_path, mask_path, class_count = select_max_class_image_pair(
        dataset_root=dataset_root,
        split=str(args.split),
        ignore_codes=ignore_codes,
    )
    print(
        "Selected max-class sample: "
        f"image={image_path} mask={mask_path} classes={class_count}",
        flush=True,
    )
    meta = {
        "selection_mode": "max_class_from_dataset",
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "ignore_codes": sorted(ignore_codes),
        "selected_class_count": int(class_count),
    }
    return str(image_path), str(mask_path), meta


def enforce_cpu_only(cpu_only: bool) -> None:
    if not cpu_only:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"


def make_case(method: str, params: dict[str, Any], output_dir: Path) -> SweepCase:
    token_parts = [method] + [f"{k}_{sanitize_token(v)}" for k, v in sorted(params.items())]
    label = "__".join(token_parts)
    return SweepCase(
        method=method,
        label=label,
        params=params,
        spanno_path=str(output_dir / "spanno" / f"{label}.spanno.json.gz"),
        run_dir=str(output_dir / "runs" / label),
    )


def suggest_params(trial: optuna.Trial, method: str, args: argparse.Namespace) -> dict[str, Any]:
    if method == "felzenszwalb":
        return {
            "scale": float(trial.suggest_float("scale", 20.0, 4000.0, log=True)),
            "f_sigma": float(trial.suggest_float("f_sigma", 0.0, 3.0)),
            "min_size": int(trial.suggest_int("min_size", 5, 500, step=5)),
        }
    if method == "slic":
        return {
            "n_segments": int(trial.suggest_int("n_segments", 50, 5000, step=50)),
            "compactness": float(trial.suggest_float("compactness", 0.01, 120.0, log=True)),
            "sigma": float(trial.suggest_float("sigma", 0.0, 3.0)),
        }
    if method == "watershed":
        return {
            "ws_compactness": float(
                trial.suggest_float("ws_compactness", 1e-6, 2.0, log=True)
            ),
            "ws_components": int(trial.suggest_int("ws_components", 50, 6000, step=50)),
        }
    if method == "ssn":
        if not args.ssn_weights:
            raise ValueError("--ssn-weights is required for method=ssn")
        return {
            "ssn_weights": str(Path(args.ssn_weights).resolve()),
            "ssn_nspix": int(trial.suggest_int("ssn_nspix", 100, 3000, step=50)),
            "ssn_fdim": int(trial.suggest_categorical("ssn_fdim", [20])),
            "ssn_niter": int(trial.suggest_int("ssn_niter", 3, 20)),
            "ssn_color_scale": float(trial.suggest_float("ssn_color_scale", 0.05, 1.0)),
            "ssn_pos_scale": float(trial.suggest_float("ssn_pos_scale", 0.5, 12.0)),
        }
    raise ValueError(f"Unsupported method: {method}")


def write_study_summary(study: optuna.Study, output_dir: Path) -> dict[str, Any]:
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        rows.append(row)

    best = None
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None:
        best = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        }

    payload = {"best_trial": best, "trials": rows}
    with open(output_dir / "optuna_trials.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best or {}, fh, ensure_ascii=False, indent=2)
    return payload


def optimize_method(
    method: str,
    args: argparse.Namespace,
    image_path: str,
    mask_path: str,
    root_output_dir: Path,
    memory_guard: MemoryGuard,
) -> dict[str, Any]:
    out_dir = (root_output_dir / method).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resized_image, resized_mask = ensure_resized_inputs(
        image_path=str(Path(image_path).resolve()),
        mask_path=str(Path(mask_path).resolve()),
        output_dir=out_dir,
        resize_scale=float(args.resize_scale),
        overwrite=bool(args.overwrite),
    )

    if args.storage:
        storage = str(args.storage).format(method=method)
    else:
        storage = f"sqlite:///{(out_dir / 'optuna.db').resolve()}"
    study_name = (
        f"{args.study_prefix}_{method}_x{sanitize_token(args.resize_scale)}_"
        f"s{sanitize_token(args.scribbles)}"
    )
    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(3, min(10, int(args.trials) // 3))
    )

    python_bin_arg = Path(str(args.python_bin)).expanduser()
    if python_bin_arg.exists():
        # Keep symlink path as-is (do not resolve), so `.venv/bin/python` is preserved.
        python_bin = str(python_bin_arg.absolute())
    else:
        python_bin = str(args.python_bin)

    worker_args = {
        "image": resized_image,
        "mask": resized_mask,
        "output_dir": str(out_dir),
        "python_bin": python_bin,
        "scribbles": int(args.scribbles),
        "save_every": int(args.save_every),
        "seed": int(args.seed),
        "resize_scale": float(args.resize_scale),
        "margin": int(args.margin),
        "border_margin": int(args.border_margin),
        "no_overlap": bool(args.no_overlap),
        "max_no_progress": int(args.max_no_progress),
        "region_selection_cycle": str(args.region_selection_cycle),
        "sensitivity": float(args.sensitivity),
        "emb_weights": (
            None
            if args.emb_weights is None
            else str(Path(args.emb_weights).resolve())
        ),
        "emb_threshold": float(args.emb_threshold),
        "num_classes": args.num_classes,
        "overwrite": bool(args.overwrite),
    }

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        try:
            rss_before = memory_guard.check(stage=f"{method}:trial_{trial.number}:before")
        except MemoryLimitExceeded as exc:
            trial.set_user_attr("status", "memory_pruned")
            trial.set_user_attr("error", str(exc))
            raise optuna.TrialPruned()

        params = suggest_params(trial, method, args)
        case = make_case(method=method, params=params, output_dir=out_dir)
        row = run_case_worker({"case": asdict(case), "args": worker_args})
        if row.get("status") != "ok":
            trial.set_user_attr("status", row.get("status"))
            trial.set_user_attr("error", row.get("error"))
            raise optuna.TrialPruned()

        try:
            rss_after = memory_guard.check(stage=f"{method}:trial_{trial.number}:after")
        except MemoryLimitExceeded as exc:
            trial.set_user_attr("status", "memory_pruned")
            trial.set_user_attr("error", str(exc))
            raise optuna.TrialPruned()

        miou = float(row.get("miou") or 0.0)
        coverage = float(row.get("coverage") or 0.0)
        precision = float(row.get("annotation_precision") or 0.0)
        total_ink = float(row.get("total_ink_px") or 0.0)
        superpixels = int(row.get("superpixels") or 0)

        trial.set_user_attr("label", case.label)
        trial.set_user_attr("run_dir", case.run_dir)
        trial.set_user_attr("spanno_path", case.spanno_path)
        trial.set_user_attr("miou", miou)
        trial.set_user_attr("coverage", coverage)
        trial.set_user_attr("annotation_precision", precision)
        trial.set_user_attr("total_ink_px", total_ink)
        trial.set_user_attr("superpixels", superpixels)
        trial.set_user_attr("rss_before_bytes", int(rss_before))
        trial.set_user_attr("rss_after_bytes", int(rss_after))
        return miou

    print(
        f"Optuna study={study_name} method={method} trials={int(args.trials)} "
        f"jobs={int(args.jobs)} resize_scale={float(args.resize_scale):.4f} "
        f"scribbles={int(args.scribbles)}",
        flush=True,
    )
    print(f"Storage: {storage}", flush=True)
    print(f"Image: {resized_image}", flush=True)
    print(f"Mask:  {resized_mask}", flush=True)

    study.optimize(
        objective,
        n_trials=int(args.trials),
        n_jobs=int(args.jobs),
        show_progress_bar=False,
    )
    trials_payload = write_study_summary(study, output_dir=out_dir)

    best_payload = None
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None:
        best_payload = {
            "number": int(best_trial.number),
            "value": float(best_trial.value),
            "params": dict(best_trial.params),
            "user_attrs": dict(best_trial.user_attrs),
        }
        print(
            f"[{method}] Best trial #{best_trial.number}: "
            f"value={best_trial.value:.6f} params={best_trial.params}",
            flush=True,
        )
    else:
        print(f"[{method}] No completed trial found.", flush=True)

    return {
        "method": method,
        "study_name": study_name,
        "storage": storage,
        "output_dir": str(out_dir),
        "trials_requested": int(args.trials),
        "jobs": int(args.jobs),
        "best_trial": best_payload,
        "trial_count": len(trials_payload.get("trials", [])),
    }


def main() -> int:
    args = build_parser().parse_args()
    set_worker_thread_limits()
    enforce_cpu_only(bool(args.cpu_only))
    methods = resolve_methods(args)
    if "ssn" in methods and not args.ssn_weights:
        raise ValueError("--ssn-weights is required when ssn is requested.")

    image_path, mask_path, selection_meta = resolve_image_and_mask(args)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_limit_bytes = int(float(args.memory_limit_gb) * (1024**3))
    memory_guard = MemoryGuard(limit_bytes=memory_limit_bytes)
    _ = memory_guard.check(stage="startup")

    method_summaries: list[dict[str, Any]] = []
    for method in methods:
        summary = optimize_method(
            method=method,
            args=args,
            image_path=image_path,
            mask_path=mask_path,
            root_output_dir=out_dir,
            memory_guard=memory_guard,
        )
        method_summaries.append(summary)

    aggregate = {
        "input_image": image_path,
        "input_mask": mask_path,
        "selection_meta": selection_meta,
        "cpu_only": bool(args.cpu_only),
        "memory_limit_gb": float(args.memory_limit_gb),
        "scribbles": int(args.scribbles),
        "trials_per_method": int(args.trials),
        "methods": method_summaries,
    }
    with open(out_dir / "multi_method_summary.json", "w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, ensure_ascii=False, indent=2)

    print("Finished all requested studies.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
