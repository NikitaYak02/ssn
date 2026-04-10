#!/usr/bin/env python3
"""Local parameter refinement on a selected set of image/mask pairs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sweep_interactive_superpixels import (
    SweepCase,
    ensure_resized_inputs,
    run_case_worker,
    sanitize_token,
)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(v) for v in parse_csv_list(raw)]


def parse_float_csv(raw: str) -> list[float]:
    return [float(v) for v in parse_csv_list(raw)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Refine local superpixel parameters on a selected image set."
    )
    ap.add_argument("--pairs-json", required=True, help="JSON list of {name,image,mask}.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--python-bin", default=None)
    ap.add_argument("--method", required=True, choices=["slic", "ssn", "felzenszwalb"])
    ap.add_argument("--resize-scale", type=float, default=0.5)
    ap.add_argument("--scribbles", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")

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

    grp_f = ap.add_argument_group("Felzenszwalb local grid")
    grp_f.add_argument("--felz-scales", default="")
    grp_f.add_argument("--felz-sigmas", default="")
    grp_f.add_argument("--felz-min-sizes", default="")

    grp_s = ap.add_argument_group("SLIC local grid")
    grp_s.add_argument("--slic-n-segments", default="")
    grp_s.add_argument("--slic-compactnesses", default="")
    grp_s.add_argument("--slic-sigmas", default="")

    grp_ssn = ap.add_argument_group("SSN local grid")
    grp_ssn.add_argument("--ssn-weights", default=None)
    grp_ssn.add_argument("--ssn-nspix-list", default="")
    grp_ssn.add_argument("--ssn-fdim", type=int, default=20)
    grp_ssn.add_argument("--ssn-niter-list", default="")
    grp_ssn.add_argument("--ssn-color-scales", default="")
    grp_ssn.add_argument("--ssn-pos-scales", default="")
    return ap


def load_pairs(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        pairs = json.load(fh)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("--pairs-json must contain a non-empty list")
    out = []
    for item in pairs:
        out.append(
            {
                "name": str(item["name"]),
                "image": str(Path(item["image"]).resolve()),
                "mask": str(Path(item["mask"]).resolve()),
            }
        )
    return out


def make_case(method: str, params: dict[str, Any], base_dir: Path) -> SweepCase:
    token_parts = [method] + [f"{k}_{sanitize_token(v)}" for k, v in sorted(params.items())]
    label = "__".join(token_parts)
    return SweepCase(
        method=method,
        label=label,
        params=params,
        spanno_path=str(base_dir / "spanno" / f"{label}.spanno.json.gz"),
        run_dir=str(base_dir / "runs" / label),
    )


def build_cases(args: argparse.Namespace, out_dir: Path) -> list[SweepCase]:
    cases: list[SweepCase] = []
    if args.method == "slic":
        for n_segments, compactness, sigma in itertools.product(
            parse_int_csv(args.slic_n_segments),
            parse_float_csv(args.slic_compactnesses),
            parse_float_csv(args.slic_sigmas),
        ):
            cases.append(
                make_case(
                    "slic",
                    {
                        "n_segments": int(n_segments),
                        "compactness": float(compactness),
                        "sigma": float(sigma),
                    },
                    out_dir,
                )
            )
    elif args.method == "ssn":
        if not args.ssn_weights:
            raise ValueError("--ssn-weights required for method=ssn")
        for nspix, niter, color_scale, pos_scale in itertools.product(
            parse_int_csv(args.ssn_nspix_list),
            parse_int_csv(args.ssn_niter_list),
            parse_float_csv(args.ssn_color_scales),
            parse_float_csv(args.ssn_pos_scales),
        ):
            cases.append(
                make_case(
                    "ssn",
                    {
                        "ssn_weights": str(Path(args.ssn_weights).resolve()),
                        "ssn_nspix": int(nspix),
                        "ssn_fdim": int(args.ssn_fdim),
                        "ssn_niter": int(niter),
                        "ssn_color_scale": float(color_scale),
                        "ssn_pos_scale": float(pos_scale),
                    },
                    out_dir,
                )
            )
    elif args.method == "felzenszwalb":
        for scale, sigma, min_size in itertools.product(
            parse_float_csv(args.felz_scales),
            parse_float_csv(args.felz_sigmas),
            parse_int_csv(args.felz_min_sizes),
        ):
            cases.append(
                make_case(
                    "felzenszwalb",
                    {
                        "scale": float(scale),
                        "f_sigma": float(sigma),
                        "min_size": int(min_size),
                    },
                    out_dir,
                )
            )
    return cases


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def run_batch_case(payload: dict[str, Any]) -> dict[str, Any]:
    case = SweepCase(**payload["case"])
    args = payload["args"]
    pairs = payload["pairs"]
    output_dir = Path(payload["output_dir"])

    miou_values: list[float] = []
    coverage_values: list[float] = []
    precision_values: list[float] = []
    ink_values: list[float] = []
    sp_values: list[float] = []
    failed_pairs: list[str] = []

    for pair in pairs:
        pair_name = pair["name"]
        pair_out = output_dir / "per_pair" / case.label / sanitize_token(pair_name)
        resized_image, resized_mask = ensure_resized_inputs(
            image_path=pair["image"],
            mask_path=pair["mask"],
            output_dir=pair_out,
            resize_scale=float(args["resize_scale"]),
            overwrite=bool(args["overwrite"]),
        )
        pair_case = SweepCase(
            method=case.method,
            label=case.label,
            params=case.params,
            spanno_path=str(pair_out / "spanno" / Path(case.spanno_path).name),
            run_dir=str(pair_out / "run"),
        )
        pair_args = dict(args)
        pair_args["image"] = resized_image
        pair_args["mask"] = resized_mask
        row = run_case_worker({"case": pair_case.__dict__, "args": pair_args})
        if row.get("status") != "ok":
            failed_pairs.append(pair_name)
            continue
        miou_values.append(float(row.get("miou") or 0.0))
        coverage_values.append(float(row.get("coverage") or 0.0))
        precision_values.append(float(row.get("annotation_precision") or 0.0))
        ink_values.append(float(row.get("total_ink_px") or 0.0))
        if row.get("superpixels") is not None:
            sp_values.append(float(row["superpixels"]))

    return {
        "status": "ok" if not failed_pairs else ("failed" if not miou_values else "partial"),
        "method": case.method,
        "label": case.label,
        **{f"param_{k}": v for k, v in sorted(case.params.items())},
        "n_pairs": len(pairs),
        "failed_pairs": ",".join(failed_pairs),
        "mean_miou": mean_or_none(miou_values),
        "mean_coverage": mean_or_none(coverage_values),
        "mean_annotation_precision": mean_or_none(precision_values),
        "mean_total_ink_px": mean_or_none(ink_values),
        "mean_superpixels": mean_or_none(sp_values),
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.pairs_json)
    cases = build_cases(args, out_dir=output_dir)
    if not cases:
        raise RuntimeError("No refinement cases generated.")

    worker_args = {
        "python_bin": args.python_bin,
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
        "emb_weights": args.emb_weights,
        "emb_threshold": float(args.emb_threshold),
        "num_classes": args.num_classes,
        "overwrite": bool(args.overwrite),
    }

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(
                run_batch_case,
                {
                    "case": case.__dict__,
                    "args": worker_args,
                    "pairs": pairs,
                    "output_dir": str(output_dir),
                },
            ): case
            for case in cases
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{idx}/{len(cases)}] {row['label']} -> {row['status']} "
                f"(mean_mIoU={row.get('mean_miou')}, mean_cov={row.get('mean_coverage')})",
                flush=True,
            )

    rows.sort(
        key=lambda r: (
            -(r["mean_miou"] if isinstance(r.get("mean_miou"), (int, float)) and not math.isnan(r["mean_miou"]) else -1.0),
            -(r["mean_coverage"] if isinstance(r.get("mean_coverage"), (int, float)) and not math.isnan(r["mean_coverage"]) else -1.0),
            r["label"],
        )
    )
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
