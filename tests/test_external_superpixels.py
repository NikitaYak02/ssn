from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scipy.io
from PIL import Image
import numpy as np

import external_superpixels.spam as spam_module
from external_superpixels.paper_alignment import compute_superpixel_anything_overlap
from external_superpixels.spam import (
    apply_spam_runtime_patches,
    build_spam_train_command,
    load_spam_manifest,
    prepare_bsd_like_dataset,
    resolve_spam_device,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_toy_pair(img_path: Path, mask_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[:, :8] = 1
    mask[:, 8:16] = 2
    mask[:, 16:] = 3
    Image.fromarray(image, mode="RGB").save(img_path)
    Image.fromarray(mask, mode="L").save(mask_path)


def test_prepare_bsd_like_dataset_writes_mat_and_manifest(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for idx in range(3):
        _write_toy_pair(img_dir / f"sample_{idx}.png", mask_dir / f"sample_{idx}.png", seed=idx)

    manifest = prepare_bsd_like_dataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        output_root=tmp_path / "prepared",
        val_ratio=0.34,
        seed=7,
        use_symlinks=False,
    )

    assert manifest["counts"]["train"] >= 1
    assert manifest["counts"]["val"] >= 1
    mat_candidates = sorted((tmp_path / "prepared" / "groundTruth" / "train").glob("*.mat"))
    assert mat_candidates
    mat_path = mat_candidates[0]
    assert mat_path.exists()

    loaded = scipy.io.loadmat(mat_path)
    seg = loaded["groundTruth"][0][0][0][0][0]
    assert seg.shape == (24, 24)
    assert int(seg.max()) == 3

    manifest_json = json.loads((tmp_path / "prepared" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["counts"]["test"] == manifest_json["counts"]["val"]


def test_build_spam_train_command_uses_manifest_paths(tmp_path: Path) -> None:
    manifest = load_spam_manifest()
    repo_dir = tmp_path / "spam_repo"
    venv_dir = tmp_path / "spam_venv"
    cmd = build_spam_train_command(
        manifest,
        dataset_root=tmp_path / "dataset",
        out_dir=tmp_path / "out",
        repo_dir=repo_dir,
        venv_dir=venv_dir,
        python_bin=venv_dir / "bin" / "python",
        train_iter=123,
        type_model="resnet50",
        use_sam=True,
        device="cpu",
    )

    assert cmd["cmd"][0] == str(venv_dir / "bin" / "python")
    assert "--train_iter" in cmd["cmd"]
    assert "123" in cmd["cmd"]
    assert "--use_sam" in cmd["cmd"]
    assert cmd["cwd"] == str(repo_dir / "SPAM")
    assert cmd["device"]["selected"] == "cpu"
    assert cmd["runtime_env"]["env_overrides"]["SPAM_DEVICE"] == "cpu"
    assert cmd["runtime_env"]["env_overrides"]["SPAM_FORCE_CPU_PAIRWISE"] == "1"
    assert cmd["runtime_env"]["path_prefix"] == str((venv_dir / "bin").resolve())


def test_resolve_spam_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        spam_module,
        "_probe_torch_capabilities",
        lambda python_bin=None: {
            "probe_python": "/tmp/python",
            "capabilities": {"cuda": True, "mps": True},
        },
    )
    resolved = resolve_spam_device("auto")
    assert resolved["selected"] == "cuda"


def test_resolve_spam_device_auto_falls_back_to_mps(monkeypatch) -> None:
    monkeypatch.setattr(
        spam_module,
        "_probe_torch_capabilities",
        lambda python_bin=None: {
            "probe_python": "/tmp/python",
            "capabilities": {"cuda": False, "mps": True},
        },
    )
    resolved = resolve_spam_device("auto")
    assert resolved["selected"] == "mps"


def test_resolve_spam_device_raises_for_unavailable_mps(monkeypatch) -> None:
    monkeypatch.setattr(
        spam_module,
        "_probe_torch_capabilities",
        lambda python_bin=None: {
            "probe_python": "/tmp/python",
            "capabilities": {"cuda": False, "mps": False},
        },
    )
    try:
        resolve_spam_device("mps")
    except RuntimeError as exc:
        assert "MPS is not available" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unavailable mps device")


def test_superpixel_anything_overlap_flags_existing_methods() -> None:
    overlap = compute_superpixel_anything_overlap()
    assert "slic" in overlap["exact_overlap"]
    assert "ssn" in overlap["exact_overlap"]
    assert "sp_fcn" in overlap["lineage_overlap"]


def test_apply_spam_runtime_patches_is_idempotent(tmp_path: Path) -> None:
    repo_dir = tmp_path / "spam_repo"
    (repo_dir / "SPAM" / "ssn").mkdir(parents=True)
    (repo_dir / "SPAM" / "train.py").write_text(
        (
            "import os\n"
            "import torch\n\n"
            "def train(cfg):\n"
            "    if torch.cuda.is_available():\n"
            '        device = "cuda"\n'
            "    else:\n"
            '        device = "cpu"\n'
            "    return device\n"
        ),
        encoding="utf-8",
    )
    (repo_dir / "SPAM" / "ssn" / "model.py").write_text(
        (
            "import os\n"
            "import torch\n\n"
            "def create_model(*args, **kwargs):\n"
            '    device = "cuda" if torch.cuda.is_available() else "cpu"\n'
            "    return device\n"
        ),
        encoding="utf-8",
    )
    (repo_dir / "SPAM" / "ssn" / "pair_wise_distance.py").write_text(
        (
            "import os\n"
            "import torch\n"
            "from torch.utils.cpp_extension import load\n"
            "pair_wise_distance_cuda = load(name='pair_wise_distance_cuda', sources=['pair_wise_distance_cuda_source.cu'])\n"
        ),
        encoding="utf-8",
    )

    first = apply_spam_runtime_patches(repo_dir)
    assert first["changed"] is True
    assert len(first["changed_files"]) == 3
    assert "SPAM_DEVICE" in (repo_dir / "SPAM" / "train.py").read_text(encoding="utf-8")
    assert "SPAM_DEVICE" in (repo_dir / "SPAM" / "ssn" / "model.py").read_text(encoding="utf-8")
    assert "SPAM_FORCE_CPU_PAIRWISE" in (repo_dir / "SPAM" / "ssn" / "pair_wise_distance.py").read_text(encoding="utf-8")

    second = apply_spam_runtime_patches(repo_dir)
    assert second["changed"] is False
    assert second["changed_files"] == []


def test_train_external_superpixels_dry_run_writes_metadata(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for idx in range(2):
        _write_toy_pair(img_dir / f"sample_{idx}.png", mask_dir / f"sample_{idx}.png", seed=idx + 100)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train_external_superpixels.py"),
            "--method",
            "spam",
            "--img_dir",
            str(img_dir),
            "--mask_dir",
            str(mask_dir),
            "--out_dir",
            str(out_dir),
            "--repo_dir",
            str(tmp_path / "spam_repo"),
            "--venv_dir",
            str(tmp_path / "spam_venv"),
            "--python_bin",
            str(tmp_path / "spam_venv" / "bin" / "python"),
            "--dry_run",
            "--device",
            "cpu",
            "--train_iter",
            "11",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    resolved = json.loads((out_dir / "resolved_external_training.json").read_text(encoding="utf-8"))
    assert resolved["cli_args"]["train_iter"] == 11
    assert Path(resolved["dataset_root"]).exists()
