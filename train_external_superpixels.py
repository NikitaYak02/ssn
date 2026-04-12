#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from external_superpixels.spam import (
    bootstrap_spam_environment,
    build_spam_train_command,
    load_spam_manifest,
    prepare_bsd_like_dataset,
)


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Train external superpixel methods in isolated upstream environments. "
            "Currently supports the official Superpixel Anything Model (SPAM)."
        )
    )
    ap.add_argument("--method", default="spam", choices=["spam"])
    ap.add_argument("--out_dir", required=True, help="Directory for training outputs.")
    ap.add_argument(
        "--bsd_root",
        default=None,
        help=(
            "Existing BSD-style dataset root with images/{train,val,test} and "
            "groundTruth/{train,val,test}."
        ),
    )
    ap.add_argument("--img_dir", default=None, help="Flat directory with RGB images.")
    ap.add_argument("--mask_dir", default=None, help="Flat directory with 2-D masks.")
    ap.add_argument(
        "--prepared_dataset_dir",
        default=None,
        help="Where to materialize the BSD-style converted dataset when img/mask dirs are used.",
    )
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--copy_files",
        action="store_true",
        help="Copy source images instead of symlinking during BSD conversion.",
    )

    ap.add_argument("--bootstrap", action="store_true", help="Clone upstream repo and create isolated venv.")
    ap.add_argument("--repo_dir", default=None, help="Override upstream repo location.")
    ap.add_argument("--venv_dir", default=None, help="Override isolated venv location.")
    ap.add_argument("--python_bin", default=None, help="Explicit python binary for upstream training.")
    ap.add_argument("--bootstrap_python", default=sys.executable, help="Python used to create the isolated venv.")
    ap.add_argument("--dry_run", action="store_true", help="Resolve paths/command, write metadata, but do not execute training.")

    ap.add_argument("--batchsize", type=int, default=6)
    ap.add_argument("--nworkers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_iter", type=int, default=500000)
    ap.add_argument("--fdim", type=int, default=20)
    ap.add_argument("--niter", type=int, default=5)
    ap.add_argument("--nspix", type=int, default=100)
    ap.add_argument("--color_scale", type=float, default=0.26)
    ap.add_argument("--pos_scale", type=float, default=7.5)
    ap.add_argument("--compactness", type=float, default=1e-5)
    ap.add_argument("--test_interval", type=int, default=5000)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--loss_type", default="seg", choices=["seg", "contours", "seg_contours"])
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--use_sam", action="store_true")
    ap.add_argument(
        "--type_model",
        default="ssn",
        choices=["ssn", "resnet50", "resnet101", "mobilenetv3"],
        help=(
            "Backbone exposed by the official SPAM codepath. "
            "The upstream parser help mentions deeplabv3, but the current code supports mobilenetv3."
        ),
    )
    return ap


def resolve_dataset_root(args: argparse.Namespace, out_dir: Path) -> tuple[Path, dict[str, object] | None]:
    if args.bsd_root:
        return Path(args.bsd_root).resolve(), None
    if not args.img_dir or not args.mask_dir:
        raise ValueError("Either --bsd_root or both --img_dir and --mask_dir are required.")
    prepared_dir = Path(
        args.prepared_dataset_dir
        or (out_dir / "prepared_bsd_dataset")
    ).resolve()
    manifest = prepare_bsd_like_dataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        output_root=prepared_dir,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        use_symlinks=not bool(args.copy_files),
    )
    return prepared_dir, manifest


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_spam_manifest()
    dataset_root, dataset_manifest = resolve_dataset_root(args, out_dir)

    if args.bootstrap:
        bootstrap_spam_environment(
            manifest,
            repo_dir=args.repo_dir,
            venv_dir=args.venv_dir,
            bootstrap_python=args.bootstrap_python,
        )

    resolved = build_spam_train_command(
        manifest,
        dataset_root=dataset_root,
        out_dir=out_dir,
        repo_dir=args.repo_dir,
        venv_dir=args.venv_dir,
        python_bin=args.python_bin,
        batchsize=args.batchsize,
        nworkers=args.nworkers,
        lr=args.lr,
        train_iter=args.train_iter,
        fdim=args.fdim,
        niter=args.niter,
        nspix=args.nspix,
        color_scale=args.color_scale,
        pos_scale=args.pos_scale,
        compactness=args.compactness,
        test_interval=args.test_interval,
        normalize=args.normalize,
        use_sam=args.use_sam,
        type_model=args.type_model,
        loss_type=args.loss_type,
        weights=args.weights,
        model=args.model,
    )

    metadata = {
        "method": args.method,
        "dataset_root": str(dataset_root),
        "dataset_manifest": dataset_manifest,
        "resolved_command": resolved,
        "cli_args": vars(args),
    }
    with open(out_dir / "resolved_external_training.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    subprocess.run(
        resolved["cmd"],
        cwd=resolved["cwd"],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
