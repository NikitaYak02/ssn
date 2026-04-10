#!/usr/bin/env python3
"""Parallel parameter sweep for interactive superpixel annotation runs.

The script compares one or more superpixel methods on the same image/mask
pair by:

1. building a parameter grid for each requested method,
2. precomputing one `.spanno.json.gz` per configuration,
3. running `evaluate_interactive_annotation.py` for a fixed number of scribbles,
4. collecting the final metrics into one CSV/JSON summary.

Example:
    superpixel_annotator/superpixel_annotator_venv/bin/python \
      sweep_interactive_superpixels.py \
      --image artifacts/case_studies/_quarter_run/input/train_01_q1.jpg \
      --mask artifacts/case_studies/_quarter_run/input/train_01_q1.png \
      --output-dir artifacts/sweeps/train01_100 \
      --methods felzenszwalb,slic,ssn \
      --ssn-weights models/checkpoints/best_model.pth \
      --scribbles 100 \
      --workers 4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image


METHOD_ALIASES = {
    "fwb": "felzenszwalb",
    "ws": "watershed",
}
NEURAL_METHODS = {
    "deep_slic",
    "cnn_rim",
    "sp_fcn",
    "sin",
    "rethink_unsup",
}
SIMPLE_METHODS = {
    "felzenszwalb",
    "slic",
    "watershed",
}
ACCELERATED_METHODS = {"ssn", *NEURAL_METHODS}
SUPPORTED_METHODS = SIMPLE_METHODS | ACCELERATED_METHODS


def parse_csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list.")
    return values


def parse_float_csv(raw: str) -> list[float]:
    return [float(v) for v in parse_csv_list(raw)]


def parse_int_csv(raw: str) -> list[int]:
    return [int(v) for v in parse_csv_list(raw)]


def normalize_method_name(value: str) -> str:
    method = str(value).strip().lower()
    if not method:
        raise ValueError("Expected a non-empty method name.")
    return METHOD_ALIASES.get(method, method)


def parse_method_list(raw: str) -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()
    for value in parse_csv_list(raw):
        method = normalize_method_name(value)
        if method not in seen:
            seen.add(method)
            methods.append(method)
    return methods


def parse_json_value(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8")
    return json.loads(text)


def parse_neural_method_configs(raw: Optional[str]) -> dict[str, list[dict[str, Any]]]:
    payload = parse_json_value(raw)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("--neural-method-configs must decode to a JSON object.")

    parsed: dict[str, list[dict[str, Any]]] = {}
    for method_name, configs in payload.items():
        method = normalize_method_name(str(method_name))
        if method not in NEURAL_METHODS:
            raise ValueError(
                f"--neural-method-configs only supports neural methods, got: {method_name!r}"
            )
        if isinstance(configs, dict):
            configs = [configs]
        if not isinstance(configs, list) or not configs:
            raise ValueError(
                f"Expected a non-empty config list for neural method {method!r}."
            )
        rows: list[dict[str, Any]] = []
        for config in configs:
            if not isinstance(config, dict):
                raise ValueError(
                    f"Neural config for {method!r} must be a JSON object, got: {type(config)!r}"
                )
            rows.append(dict(config))
        parsed[method] = rows
    return parsed


def serialize_method_config(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sanitize_token(value: Any) -> str:
    text = str(value)
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        elif ch == ".":
            safe.append("p")
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "x"


@dataclass(frozen=True)
class SweepCase:
    method: str
    label: str
    params: dict[str, Any]
    spanno_path: str
    run_dir: str


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Parallel parameter sweep for interactive superpixel annotation."
    )
    ap.add_argument("--image", required=True, help="RGB image for interactive evaluation.")
    ap.add_argument("--mask", required=True, help="GT mask with class ids.")
    ap.add_argument("--output-dir", required=True, help="Directory for sweep outputs.")
    ap.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used for evaluate_interactive_annotation.py subprocesses.",
    )
    ap.add_argument(
        "--methods",
        default="felzenszwalb,slic,ssn",
        help=(
            "Comma-separated methods subset. Supported: "
            "felzenszwalb/fwb, slic, watershed/ws, ssn, "
            "deep_slic, cnn_rim, sp_fcn, sin, rethink_unsup."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1))),
        help="Fallback parallelism when method-specific worker limits are not set.",
    )
    ap.add_argument(
        "--simple-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers for CPU-only methods "
            "(felzenszwalb, slic, watershed). Defaults to --workers."
        ),
    )
    ap.add_argument(
        "--ssn-workers",
        type=int,
        default=1,
        help=(
            "Parallel workers for SSN and neural-method cases. "
            "Keep low to avoid GPU/VRAM oversubscription."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cached spanno and rerun finished cases.",
    )

    grp_sim = ap.add_argument_group("Interactive evaluation")
    grp_sim.add_argument("--scribbles", type=int, default=100)
    grp_sim.add_argument("--save_every", type=int, default=20)
    grp_sim.add_argument("--seed", type=int, default=0)
    grp_sim.add_argument(
        "--resize-scale",
        type=float,
        default=1.0,
        help="Resize source image and mask before sweep (1.0 keeps original size).",
    )
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

    grp_f = ap.add_argument_group("Felzenszwalb grid")
    grp_f.add_argument("--felz-scales", default="200,400,800")
    grp_f.add_argument("--felz-sigmas", default="0.5,1.0")
    grp_f.add_argument("--felz-min-sizes", default="20,50,100")

    grp_s = ap.add_argument_group("SLIC grid")
    grp_s.add_argument("--slic-n-segments", default="500,1000,2000")
    grp_s.add_argument("--slic-compactnesses", default="10,20,30")
    grp_s.add_argument("--slic-sigmas", default="0.0,1.0")

    grp_w = ap.add_argument_group("Watershed grid")
    grp_w.add_argument("--ws-compactnesses", default="0.0001")
    grp_w.add_argument("--ws-components-list", default="500")

    grp_ssn = ap.add_argument_group("SSN grid")
    grp_ssn.add_argument("--ssn-weights", default=None, help="SSN checkpoint (.pth)")
    grp_ssn.add_argument("--ssn-nspix-list", default="200,500,800")
    grp_ssn.add_argument("--ssn-fdim", type=int, default=20)
    grp_ssn.add_argument("--ssn-niter-list", default="5")
    grp_ssn.add_argument("--ssn-color-scales", default="0.26")
    grp_ssn.add_argument("--ssn-pos-scales", default="2.5")

    grp_n = ap.add_argument_group("Neural method configs")
    grp_n.add_argument(
        "--neural-method-configs",
        default=None,
        help=(
            "JSON string or path to a JSON object mapping neural method ids "
            "to one or more config dicts. Each config may include method "
            "fields and optional weights/weight_path."
        ),
    )
    return ap


def iter_case_label_params(params: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in sorted(params.items()):
        if key in {"weights", "ssn_weights"} and value is not None:
            yield key, Path(str(value)).name
            continue
        if key == "method_config":
            try:
                decoded = json.loads(str(value))
            except Exception:
                decoded = None
            if isinstance(decoded, dict):
                for sub_key, sub_value in sorted(decoded.items()):
                    yield f"cfg_{sub_key}", sub_value
                continue
        yield key, value


def build_neural_case_params(config: dict[str, Any]) -> dict[str, Any]:
    raw_config = dict(config)
    weight_value = raw_config.pop("weights", None)
    if weight_value is None:
        weight_value = raw_config.pop("weight_path", None)

    params: dict[str, Any] = {}
    if raw_config:
        params["method_config"] = serialize_method_config(raw_config)
    if weight_value is not None and str(weight_value).strip():
        params["weights"] = str(Path(str(weight_value)).expanduser().resolve())
    return params


def build_cases(args: argparse.Namespace, output_dir: Path) -> list[SweepCase]:
    methods = parse_method_list(args.methods)
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported methods requested: {unknown}")
    neural_method_configs = parse_neural_method_configs(args.neural_method_configs)

    cases: list[SweepCase] = []
    spanno_dir = output_dir / "spanno"
    runs_dir = output_dir / "runs"

    def add_case(method: str, params: dict[str, Any]) -> None:
        token_parts = [method] + [
            f"{k}_{sanitize_token(v)}" for k, v in iter_case_label_params(params)
        ]
        label = "__".join(token_parts)
        cases.append(
            SweepCase(
                method=method,
                label=label,
                params=params,
                spanno_path=str(spanno_dir / f"{label}.spanno.json.gz"),
                run_dir=str(runs_dir / label),
            )
        )

    if "felzenszwalb" in methods:
        for scale, sigma, min_size in itertools.product(
            parse_float_csv(args.felz_scales),
            parse_float_csv(args.felz_sigmas),
            parse_int_csv(args.felz_min_sizes),
        ):
            add_case(
                "felzenszwalb",
                {
                    "scale": float(scale),
                    "f_sigma": float(sigma),
                    "min_size": int(min_size),
                },
            )

    if "slic" in methods:
        for n_segments, compactness, sigma in itertools.product(
            parse_int_csv(args.slic_n_segments),
            parse_float_csv(args.slic_compactnesses),
            parse_float_csv(args.slic_sigmas),
        ):
            add_case(
                "slic",
                {
                    "n_segments": int(n_segments),
                    "compactness": float(compactness),
                    "sigma": float(sigma),
                },
            )

    if "watershed" in methods:
        for compactness, n_components in itertools.product(
            parse_float_csv(args.ws_compactnesses),
            parse_int_csv(args.ws_components_list),
        ):
            add_case(
                "watershed",
                {
                    "ws_compactness": float(compactness),
                    "ws_components": int(n_components),
                },
            )

    if "ssn" in methods:
        if not args.ssn_weights:
            raise ValueError("--ssn-weights is required when ssn is included in --methods.")
        for nspix, niter, color_scale, pos_scale in itertools.product(
            parse_int_csv(args.ssn_nspix_list),
            parse_int_csv(args.ssn_niter_list),
            parse_float_csv(args.ssn_color_scales),
            parse_float_csv(args.ssn_pos_scales),
        ):
            add_case(
                "ssn",
                {
                    "ssn_weights": str(Path(args.ssn_weights).resolve()),
                    "ssn_nspix": int(nspix),
                    "ssn_fdim": int(args.ssn_fdim),
                    "ssn_niter": int(niter),
                    "ssn_color_scale": float(color_scale),
                    "ssn_pos_scale": float(pos_scale),
                },
            )

    for method in methods:
        if method not in NEURAL_METHODS:
            continue
        configs = neural_method_configs.get(method) or [{}]
        for config in configs:
            add_case(method, build_neural_case_params(config))

    return cases


def set_worker_thread_limits() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _resized_suffix(scale: float) -> str:
    return sanitize_token(f"{float(scale):.4f}")


def ensure_resized_inputs(
    image_path: str,
    mask_path: str,
    output_dir: Path,
    resize_scale: float,
    overwrite: bool,
) -> tuple[str, str]:
    if resize_scale <= 0.0:
        raise ValueError("--resize-scale must be positive.")
    if math.isclose(float(resize_scale), 1.0):
        return str(Path(image_path).resolve()), str(Path(mask_path).resolve())

    src_img = Path(image_path).resolve()
    src_mask = Path(mask_path).resolve()
    cache_dir = output_dir / "_resized_inputs" / f"scale_{_resized_suffix(resize_scale)}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_out = cache_dir / src_img.name
    mask_out = cache_dir / src_mask.name

    if overwrite or not img_out.exists():
        img = Image.open(src_img).convert("RGB")
        new_w = max(1, int(round(img.width * float(resize_scale))))
        new_h = max(1, int(round(img.height * float(resize_scale))))
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        img.save(img_out)

    if overwrite or not mask_out.exists():
        mask = Image.open(src_mask)
        if mask.mode in ("P", "L", "I;16", "I"):
            mask_img = mask
        else:
            mask_img = mask.convert("RGB")
        img_for_size = Image.open(img_out)
        resized_mask = mask_img.resize(img_for_size.size, resample=Image.Resampling.NEAREST)
        resized_mask.save(mask_out)

    return str(img_out), str(mask_out)


def ensure_spanno(case: SweepCase, image_path: str, overwrite: bool) -> tuple[int, float]:
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent / "superpixel_annotator"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import structs  # noqa: E402
    from precompute_superpixels import populate_full_image_superpixels  # noqa: E402

    out_path = Path(case.spanno_path)
    if out_path.exists() and not overwrite:
        try:
            algo = structs.SuperPixelAnnotationAlgo(
                downscale_coeff=1.0,
                superpixel_methods=[],
                image=Image.open(image_path).convert("RGB"),
                image_path=image_path,
            )
            algo.deserialize(str(out_path))
            method = next(iter(algo.superpixels.keys()))
            return len(algo.superpixels.get(method, [])), 0.0
        except Exception:
            pass

    img = Image.open(image_path).convert("RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=img,
        image_path=image_path,
        auto_propagation_sensitivity=0.0,
    )

    method_args = argparse.Namespace(
        method=case.method,
        method_config=None,
        weights=None,
        n_segments=3000,
        compactness=20.0,
        sigma=1.0,
        scale=400.0,
        f_sigma=1.0,
        min_size=50,
        ws_compactness=1e-4,
        ws_components=500,
        ssn_weights=None,
        ssn_nspix=100,
        ssn_fdim=20,
        ssn_niter=5,
        ssn_color_scale=0.26,
        ssn_pos_scale=2.5,
    )
    for key, value in case.params.items():
        setattr(method_args, key, value)
    sp_method = structs.build_superpixel_method_from_args(method_args)

    algo.add_superpixel_method(sp_method)
    n_superpixels = populate_full_image_superpixels(algo, sp_method)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    algo.serialize(str(out_path), pretty=False)
    return int(n_superpixels), float(getattr(sp_method, "nspix", 0.0))


def read_final_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"metrics.csv is empty: {metrics_path}")
    row = rows[-1]
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        if value in ("", None):
            parsed[key] = None
            continue
        try:
            parsed[key] = float(value)
        except Exception:
            parsed[key] = value
    return parsed


def build_eval_command(args: argparse.Namespace, case: SweepCase) -> list[str]:
    python_bin = getattr(args, "python_bin", None) or sys.executable
    cmd = [
        str(python_bin),
        "evaluate_interactive_annotation.py",
        "--image",
        str(Path(args.image).resolve()),
        "--mask",
        str(Path(args.mask).resolve()),
        "--out",
        case.run_dir,
        "--method",
        case.method,
        "--spanno",
        case.spanno_path,
        "--scribbles",
        str(int(args.scribbles)),
        "--save_every",
        str(int(args.save_every)),
        "--seed",
        str(int(args.seed)),
        "--margin",
        str(int(args.margin)),
        "--border_margin",
        str(int(args.border_margin)),
        "--max_no_progress",
        str(int(args.max_no_progress)),
        "--region_selection_cycle",
        str(args.region_selection_cycle),
        "--sensitivity",
        str(float(args.sensitivity)),
    ]
    if args.no_overlap:
        cmd.append("--no_overlap")
    if args.emb_weights:
        cmd.extend(["--emb_weights", str(Path(args.emb_weights).resolve())])
        cmd.extend(["--emb_threshold", str(float(args.emb_threshold))])
    if args.num_classes is not None:
        cmd.extend(["--num_classes", str(int(args.num_classes))])

    for key, value in sorted(case.params.items()):
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            value_text = str(value)
        cmd.extend([f"--{key}", value_text])
    return cmd


def run_case_worker(payload: dict[str, Any]) -> dict[str, Any]:
    set_worker_thread_limits()

    case = SweepCase(**payload["case"])
    args = payload["args"]
    image_path = str(Path(args["image"]).resolve())
    overwrite = bool(args["overwrite"])
    run_dir = Path(case.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"

    started = time.time()
    sp_count = None
    precompute_s = 0.0
    eval_s = 0.0
    status = "ok"
    error = None

    try:
        t0 = time.time()
        sp_count, _ = ensure_spanno(case, image_path=image_path, overwrite=overwrite)
        precompute_s = time.time() - t0

        if overwrite or not metrics_path.exists():
            ns_args = argparse.Namespace(**args)
            cmd = build_eval_command(ns_args, case)
            env = os.environ.copy()
            set_worker_thread_limits()
            env.pop("PYTHONHOME", None)
            t1 = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parent),
                capture_output=True,
                text=True,
                env=env,
            )
            eval_s = time.time() - t1
            if (
                result.returncode != 0
                and "No module named 'numpy'" in ((result.stderr or "") + (result.stdout or ""))
                and str(cmd[0]) != str(sys.executable)
            ):
                cmd[0] = str(sys.executable)
                t1 = time.time()
                result = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).resolve().parent),
                    capture_output=True,
                    text=True,
                    env=env,
                )
                eval_s = time.time() - t1
            if result.returncode != 0:
                status = "failed"
                error = (
                    f"CMD: {' '.join(cmd)}\n"
                    + (result.stderr[-4000:] or result.stdout[-4000:])
                )
                raise RuntimeError(error)
        metrics = read_final_metrics(metrics_path)
    except Exception as exc:
        status = "failed"
        if error is None:
            error = str(exc)
        metrics = {}

    total_s = time.time() - started
    out: dict[str, Any] = {
        "status": status,
        "method": case.method,
        "label": case.label,
        "spanno_path": case.spanno_path,
        "run_dir": case.run_dir,
        "resize_scale": args.get("resize_scale"),
        "superpixels": sp_count,
        "precompute_s": round(precompute_s, 4),
        "eval_s": round(eval_s, 4),
        "total_s": round(total_s, 4),
        "error": error,
    }
    out.update({f"param_{k}": v for k, v in sorted(case.params.items())})
    out.update(metrics)
    return out


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    all_keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    def sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
        miou = row.get("miou")
        coverage = row.get("coverage")
        return (
            -(float(miou) if isinstance(miou, (int, float)) and not math.isnan(miou) else -1.0),
            -(float(coverage) if isinstance(coverage, (int, float)) and not math.isnan(coverage) else -1.0),
            str(row.get("label", "")),
        )

    rows_sorted = sorted(rows, key=sort_key)
    with open(summary_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    best_by_method: dict[str, dict[str, Any]] = {}
    for row in rows_sorted:
        if row.get("status") != "ok":
            continue
        method = str(row.get("method"))
        best_by_method.setdefault(method, row)

    payload = {
        "rows": rows_sorted,
        "best_by_method": best_by_method,
    }
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def run_cases(
    cases: list[SweepCase],
    max_workers: int,
    worker_args: dict[str, Any],
    rows: list[dict[str, Any]],
    total_cases: int,
    completed_before: int,
    output_dir: Path,
) -> None:
    if not cases:
        return

    with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {
            pool.submit(run_case_worker, {"case": asdict(case), "args": worker_args}): case
            for case in cases
        }
        for stage_idx, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "status": "failed",
                    "method": case.method,
                    "label": case.label,
                    "spanno_path": case.spanno_path,
                    "run_dir": case.run_dir,
                    "error": str(exc),
                }
            rows.append(row)
            write_summary(rows, output_dir=output_dir)
            status = row.get("status", "unknown")
            miou = row.get("miou")
            coverage = row.get("coverage")
            print(
                f"[{completed_before + stage_idx}/{total_cases}] {case.label} -> {status} "
                f"(mIoU={miou}, coverage={coverage})",
                flush=True,
            )


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resized_image, resized_mask = ensure_resized_inputs(
        image_path=str(Path(args.image).resolve()),
        mask_path=str(Path(args.mask).resolve()),
        output_dir=output_dir,
        resize_scale=float(args.resize_scale),
        overwrite=bool(args.overwrite),
    )

    cases = build_cases(args, output_dir=output_dir)
    if not cases:
        raise RuntimeError("No sweep cases were generated.")

    simple_workers = int(args.simple_workers or args.workers)
    ssn_workers = int(args.ssn_workers or args.workers)

    print(f"Prepared {len(cases)} configurations", flush=True)
    print(
        f"Input resize scale: {float(args.resize_scale):.4f} | "
        f"image={resized_image} | mask={resized_mask}",
        flush=True,
    )
    print(
        f"Workers: simple={simple_workers}, ssn={ssn_workers}, fallback={int(args.workers)}",
        flush=True,
    )

    worker_args = {
        "image": resized_image,
        "mask": resized_mask,
        "output_dir": str(output_dir),
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

    rows: list[dict[str, Any]] = []
    simple_cases = [case for case in cases if case.method in SIMPLE_METHODS]
    accelerated_cases = [case for case in cases if case.method in ACCELERATED_METHODS]

    if simple_cases:
        print(
            f"Running {len(simple_cases)} CPU-oriented cases with {simple_workers} workers",
            flush=True,
        )
        run_cases(
            cases=simple_cases,
            max_workers=simple_workers,
            worker_args=worker_args,
            rows=rows,
            total_cases=len(cases),
            completed_before=0,
            output_dir=output_dir,
        )

    if accelerated_cases:
        print(
            f"Running {len(accelerated_cases)} SSN/neural cases with {ssn_workers} workers",
            flush=True,
        )
        run_cases(
            cases=accelerated_cases,
            max_workers=ssn_workers,
            worker_args=worker_args,
            rows=rows,
            total_cases=len(cases),
            completed_before=len(rows),
            output_dir=output_dir,
        )

    write_summary(rows, output_dir=output_dir)

    failures = sum(1 for row in rows if row.get("status") != "ok")
    print(f"Finished {len(rows)} runs with {failures} failures", flush=True)
    print(f"Summary: {output_dir / 'summary.csv'}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
