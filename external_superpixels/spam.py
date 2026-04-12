from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def load_spam_manifest(path: str | Path | None = None) -> dict[str, object]:
    manifest_path = Path(path or (ROOT / "external_superpixels" / "manifests" / "spam.json"))
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    manifest["_manifest_path"] = str(manifest_path.resolve())
    return manifest


def _expand(value: str, *, repo_root: Path, repo_dir: Path, venv_dir: Path) -> str:
    return (
        str(value)
        .replace("{repo_root}", str(repo_root.resolve()))
        .replace("{repo_dir}", str(repo_dir.resolve()))
        .replace("{venv_dir}", str(venv_dir.resolve()))
    )


def resolve_spam_paths(
    manifest: dict[str, object],
    *,
    repo_dir: str | Path | None = None,
    venv_dir: str | Path | None = None,
    python_bin: str | Path | None = None,
) -> dict[str, Path]:
    resolved_repo_dir = Path(
        repo_dir
        or _expand(
            str(manifest["default_repo_dir"]),
            repo_root=ROOT,
            repo_dir=ROOT / ".external_sources" / "spam",
            venv_dir=ROOT / ".method_envs" / "spam",
        )
    )
    resolved_venv_dir = Path(
        venv_dir
        or _expand(
            str(manifest["default_venv_dir"]),
            repo_root=ROOT,
            repo_dir=resolved_repo_dir,
            venv_dir=ROOT / ".method_envs" / "spam",
        )
    )
    resolved_python = Path(
        python_bin
        or (resolved_venv_dir / "bin" / "python")
    )
    train = dict(manifest.get("train") or {})
    return {
        "repo_dir": resolved_repo_dir,
        "venv_dir": resolved_venv_dir,
        "python_bin": resolved_python,
        "train_cwd": Path(
            _expand(
                str(train["cwd"]),
                repo_root=ROOT,
                repo_dir=resolved_repo_dir,
                venv_dir=resolved_venv_dir,
            )
        ),
        "train_entrypoint": Path(
            _expand(
                str(train["entrypoint"]),
                repo_root=ROOT,
                repo_dir=resolved_repo_dir,
                venv_dir=resolved_venv_dir,
            )
        ),
        "requirements": Path(
            _expand(
                str(train["requirements"]),
                repo_root=ROOT,
                repo_dir=resolved_repo_dir,
                venv_dir=resolved_venv_dir,
            )
        ),
    }


def _iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _match_image_mask_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[str, Path, Path]]:
    image_map: dict[str, Path] = {}
    mask_map: dict[str, Path] = {}
    for path in _iter_files(img_dir, IMAGE_EXTS):
        if path.stem in image_map:
            raise ValueError(f"Duplicate image stem found: {path.stem}")
        image_map[path.stem] = path
    for path in _iter_files(mask_dir, MASK_EXTS):
        if path.stem in mask_map:
            raise ValueError(f"Duplicate mask stem found: {path.stem}")
        mask_map[path.stem] = path
    missing = sorted(set(image_map) ^ set(mask_map))
    if missing:
        raise ValueError(
            "Image/mask stems do not match. First mismatches: "
            + ", ".join(missing[:10])
        )
    if not image_map:
        raise ValueError(f"No image/mask pairs found under {img_dir} and {mask_dir}")
    return [(stem, image_map[stem], mask_map[stem]) for stem in sorted(image_map)]


def _load_mask(mask_path: Path) -> np.ndarray:
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
    else:
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape} for {mask_path}")
    return np.asarray(mask)


def _segmentation_dtype(mask: np.ndarray):
    max_value = int(np.max(mask)) if mask.size else 0
    return np.uint16 if max_value <= np.iinfo(np.uint16).max else np.int32


def write_bsds_groundtruth_mat(path: Path, mask: np.ndarray) -> None:
    seg = np.asarray(mask, dtype=_segmentation_dtype(mask))
    ground_truth = np.empty((1, 1), dtype=object)
    ground_truth[0, 0] = np.array([(seg,)], dtype=[("Segmentation", "O")])
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"groundTruth": ground_truth}, do_compression=True)


def _copy_or_link(src: Path, dst: Path, *, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlinks:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def prepare_bsd_like_dataset(
    *,
    img_dir: str | Path,
    mask_dir: str | Path,
    output_root: str | Path,
    val_ratio: float = 0.1,
    seed: int = 0,
    use_symlinks: bool = True,
) -> dict[str, object]:
    img_root = Path(img_dir).resolve()
    mask_root = Path(mask_dir).resolve()
    out_root = Path(output_root).resolve()
    pairs = _match_image_mask_pairs(img_root, mask_root)

    rng = random.Random(int(seed))
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        train_pairs = list(shuffled)
        val_pairs = list(shuffled)
    else:
        n_val = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(val_ratio)))))
        val_pairs = sorted(shuffled[:n_val], key=lambda item: item[0])
        train_pairs = sorted(shuffled[n_val:], key=lambda item: item[0])

    split_map = {
        "train": train_pairs,
        "val": val_pairs,
        "test": val_pairs,
    }
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "groundTruth" / split).mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    for split, items in split_map.items():
        for stem, image_path, mask_path in items:
            image_dst = out_root / "images" / split / image_path.name
            mask_dst = out_root / "groundTruth" / split / f"{stem}.mat"
            _copy_or_link(image_path, image_dst, use_symlinks=use_symlinks)
            write_bsds_groundtruth_mat(mask_dst, _load_mask(mask_path))
            manifest_entries.append(
                {
                    "split": split,
                    "stem": stem,
                    "image_src": str(image_path),
                    "image_dst": str(image_dst),
                    "mask_src": str(mask_path),
                    "mask_dst": str(mask_dst),
                }
            )

    manifest = {
        "source_images": str(img_root),
        "source_masks": str(mask_root),
        "output_root": str(out_root),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "use_symlinks": bool(use_symlinks),
        "counts": {split: len(items) for split, items in split_map.items()},
        "entries": manifest_entries,
    }
    with open(out_root / "dataset_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest


def bootstrap_spam_environment(
    manifest: dict[str, object],
    *,
    repo_dir: str | Path | None = None,
    venv_dir: str | Path | None = None,
    bootstrap_python: str | Path | None = None,
) -> dict[str, Path]:
    paths = resolve_spam_paths(manifest, repo_dir=repo_dir, venv_dir=venv_dir)
    resolved_bootstrap_python = str(bootstrap_python or sys.executable)
    repo_url = str(manifest["repo_url"])
    commit = str(manifest["commit"])

    if not paths["repo_dir"].exists():
        paths["repo_dir"].parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", repo_url, str(paths["repo_dir"])],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(paths["repo_dir"]), "fetch", "--all", "--tags"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(paths["repo_dir"]), "checkout", commit],
        check=True,
    )

    if not paths["python_bin"].exists():
        paths["venv_dir"].parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [resolved_bootstrap_python, "-m", "venv", str(paths["venv_dir"])],
            check=True,
        )
    pip_bin = paths["venv_dir"] / "bin" / "pip"
    subprocess.run([str(pip_bin), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip_bin), "install", "-r", str(paths["requirements"])], check=True)
    return paths


def build_spam_train_command(
    manifest: dict[str, object],
    *,
    dataset_root: str | Path,
    out_dir: str | Path,
    repo_dir: str | Path | None = None,
    venv_dir: str | Path | None = None,
    python_bin: str | Path | None = None,
    batchsize: int = 6,
    nworkers: int = 4,
    lr: float = 1e-4,
    train_iter: int = 500000,
    fdim: int = 20,
    niter: int = 5,
    nspix: int = 100,
    color_scale: float = 0.26,
    pos_scale: float = 7.5,
    compactness: float = 1e-5,
    test_interval: int = 5000,
    normalize: bool = False,
    use_sam: bool = False,
    type_model: str = "ssn",
    loss_type: str = "seg",
    weights: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    paths = resolve_spam_paths(
        manifest,
        repo_dir=repo_dir,
        venv_dir=venv_dir,
        python_bin=python_bin,
    )
    cmd = [
        str(paths["python_bin"]),
        str(paths["train_entrypoint"]),
        "--root",
        str(Path(dataset_root).resolve()),
        "--out_dir",
        str(Path(out_dir).resolve()),
        "--batchsize",
        str(int(batchsize)),
        "--nworkers",
        str(int(nworkers)),
        "--lr",
        str(float(lr)),
        "--train_iter",
        str(int(train_iter)),
        "--fdim",
        str(int(fdim)),
        "--niter",
        str(int(niter)),
        "--nspix",
        str(int(nspix)),
        "--color_scale",
        str(float(color_scale)),
        "--pos_scale",
        str(float(pos_scale)),
        "--compactness",
        str(float(compactness)),
        "--test_interval",
        str(int(test_interval)),
        "--loss_type",
        str(loss_type),
        "--type_model",
        str(type_model),
    ]
    if normalize:
        cmd.append("--normalize")
    if use_sam:
        cmd.append("--use_sam")
    if weights:
        cmd.extend(["--weights", str(weights)])
    if model:
        cmd.extend(["--model", str(model)])

    return {
        "cmd": cmd,
        "cwd": str(paths["train_cwd"]),
        "paths": {key: str(value) for key, value in paths.items()},
    }
