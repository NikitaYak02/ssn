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
from typing import Any

import optuna

from sweep_interactive_superpixels import (
    SweepCase,
    ensure_resized_inputs,
    run_case_worker,
    sanitize_token,
    set_worker_thread_limits,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Optuna-based hyperparameter search for superpixel methods."
    )
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--method",
        required=True,
        choices=["felzenszwalb", "slic", "ssn"],
        help="Optimize one method per study.",
    )
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--resize-scale", type=float, default=0.5)
    ap.add_argument("--scribbles", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel Optuna trials. Keep 1 for SSN unless you know GPU memory is sufficient.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--study-name", default=None)
    ap.add_argument("--storage", default=None, help="Optuna storage URL. Defaults to local sqlite.")

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
            "scale": float(trial.suggest_float("scale", 80.0, 600.0, log=True)),
            "f_sigma": float(trial.suggest_float("f_sigma", 0.0, 1.5)),
            "min_size": int(trial.suggest_int("min_size", 10, 120, step=10)),
        }
    if method == "slic":
        return {
            "n_segments": int(trial.suggest_int("n_segments", 200, 1200, step=50)),
            "compactness": float(trial.suggest_float("compactness", 5.0, 40.0)),
            "sigma": float(trial.suggest_float("sigma", 0.0, 1.5)),
        }
    if method == "ssn":
        if not args.ssn_weights:
            raise ValueError("--ssn-weights is required for method=ssn")
        return {
            "ssn_weights": str(Path(args.ssn_weights).resolve()),
            "ssn_nspix": int(trial.suggest_int("ssn_nspix", 200, 1000, step=50)),
            "ssn_fdim": int(trial.suggest_categorical("ssn_fdim", [20])),
            "ssn_niter": int(trial.suggest_int("ssn_niter", 5, 10)),
            "ssn_color_scale": float(trial.suggest_float("ssn_color_scale", 0.12, 0.40)),
            "ssn_pos_scale": float(trial.suggest_float("ssn_pos_scale", 1.5, 4.0)),
        }
    raise ValueError(f"Unsupported method: {method}")


def write_study_summary(study: optuna.Study, output_dir: Path) -> None:
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
    if study.best_trial is not None:
        best = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": study.best_trial.user_attrs,
        }

    with open(output_dir / "optuna_trials.json", "w", encoding="utf-8") as fh:
        json.dump({"best_trial": best, "trials": rows}, fh, ensure_ascii=False, indent=2)

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best or {}, fh, ensure_ascii=False, indent=2)


def main() -> int:
    args = build_parser().parse_args()
    set_worker_thread_limits()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resized_image, resized_mask = ensure_resized_inputs(
        image_path=str(Path(args.image).resolve()),
        mask_path=str(Path(args.mask).resolve()),
        output_dir=out_dir,
        resize_scale=float(args.resize_scale),
        overwrite=bool(args.overwrite),
    )

    storage = args.storage or f"sqlite:///{(out_dir / 'optuna.db').resolve()}"
    study_name = args.study_name or (
        f"{args.method}_x{sanitize_token(args.resize_scale)}_{sanitize_token(args.scribbles)}"
    )
    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, min(8, int(args.trials) // 3)))

    worker_args = {
        "image": resized_image,
        "mask": resized_mask,
        "output_dir": str(out_dir),
        "python_bin": str(Path(args.python_bin).resolve() if Path(args.python_bin).exists() else args.python_bin),
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
        "emb_weights": (None if args.emb_weights is None else str(Path(args.emb_weights).resolve())),
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
        params = suggest_params(trial, args.method, args)
        case = make_case(method=args.method, params=params, output_dir=out_dir)
        row = run_case_worker({"case": asdict(case), "args": worker_args})
        if row.get("status") != "ok":
            trial.set_user_attr("status", row.get("status"))
            trial.set_user_attr("error", row.get("error"))
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
        return miou

    print(
        f"Optuna study={study_name} method={args.method} trials={int(args.trials)} "
        f"jobs={int(args.jobs)} resize_scale={float(args.resize_scale):.4f}",
        flush=True,
    )
    print(f"Storage: {storage}", flush=True)
    print(f"Image: {resized_image}", flush=True)
    print(f"Mask:  {resized_mask}", flush=True)

    study.optimize(objective, n_trials=int(args.trials), n_jobs=int(args.jobs), show_progress_bar=False)
    write_study_summary(study, output_dir=out_dir)

    best = study.best_trial
    print(
        f"Best trial #{best.number}: value={best.value:.6f} params={best.params}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
