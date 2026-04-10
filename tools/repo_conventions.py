from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT_DOCS = [
    "ALBUMENTATIONS_INTEGRATION.md",
    "AUGMENTATION_STRATEGIES.md",
    "CHANGES_SUMMARY.md",
    "FIXES_APPLIED.md",
    "OPTIMIZATIONS_SUMMARY.md",
    "PROFILING.md",
    "PROFILING_BASELINE.md",
    "PROFILING_TROUBLESHOOT.md",
    "QUICK_START_ALBUMENTATIONS.md",
    "QUICK_START_OPTIMIZED.md",
    "README_OPTIMIZATION.md",
    "TESTING_ALBUMENTATIONS.md",
]

CHECKPOINT_FILES = [
    "best_model.pth",
    "S1v2_S2v2_x05.pth",
]

EXPLICIT_FILE_MOVES = {
    **{name: f"docs/optimization/{name}" for name in ROOT_DOCS},
    **{name: f"models/checkpoints/{name}" for name in CHECKPOINT_FILES},
    "selected_pairs_s1_v2_full_cover.json": "artifacts/refinement/selected_pairs_s1_v2_full_cover.json",
    "petroscope.log": "artifacts/debug/petroscope.log",
    "test_sp_postproc.ipynb": "docs/notebooks/test_sp_postproc.ipynb",
}

CORE_TOP_LEVEL_NAMES = {
    "README.md",
    ".gitignore",
    "requirements.txt",
    "benchmark_configs.py",
    "benchmark_simple_superpixel_methods.py",
    "compare.py",
    "compute_mask_class_percentages.py",
    "evaluate_interactive_annotation.py",
    "evaluate_ssn_scribble_batch.py",
    "evaluate_superpixel_postprocessing.py",
    "inference.py",
    "model.py",
    "optimize_superpixel_params_optuna.py",
    "plot_class_miou.py",
    "precompute_superpixels.py",
    "profile_minimal.py",
    "profile_one_batch.py",
    "refine_superpixel_on_pairs.py",
    "render_interactive_annotation_video.py",
    "run_superpixel_sweep.sh",
    "superpixel_refinement_strategies.py",
    "sweep_interactive_superpixels.py",
    "train.py",
    "train_neural_superpixels.py",
    "tune_hybrid_conservative.py",
    "tune_low_confidence_threshold.py",
    "lib",
    "superpixel_annotator",
    "tests",
    "docs",
    "reports",
    "models",
    "artifacts",
    "tools",
}

ALLOWED_ROOT_NO_MOVE = {
    ".git",
    ".claude",
    ".pytest_cache",
    "__pycache__",
    ".DS_Store",
}


@dataclass(frozen=True)
class MoveSpec:
    source: str
    target: str
    reason: str


def classify_top_level_name(name: str, is_dir: bool) -> str:
    if name in {"docs", "reports", "models", "artifacts", "tools"}:
        return name
    if name in EXPLICIT_FILE_MOVES:
        target = EXPLICIT_FILE_MOVES[name]
        return target.split("/", 1)[0]

    if not is_dir:
        if name == "README.md":
            return "docs"
        if name.endswith(".pth") or name.endswith(".pt") or name.endswith(".ckpt"):
            return "models"
        if name.endswith(".md"):
            return "docs"
        return "core"

    if name == "out":
        return "artifacts"
    if name == "log":
        return "artifacts"
    if name.startswith("results_") or name.startswith("sens_") or name.startswith("_eval_"):
        return "artifacts"
    if name in {"results_prop", "results_resume_from90_refineprobe", "results_resume_smoke_from100"}:
        return "artifacts"
    if name.startswith("_tmp_sweep_") or name == "_single_image_input":
        return "artifacts"
    if name.startswith("_refine_"):
        return "artifacts"
    if name.startswith("_tmp_safe_") or name.startswith("_tmp_novel"):
        return "artifacts"
    if name in {"_quarter_run", "_two_quarters"}:
        return "artifacts"
    if name.startswith("precomputed_ssn_") or name in {"_tmp_quarter_spanno", "_tmp_quarter_spanno_clean"}:
        return "artifacts"
    if name in {"_debug_candidates", "tmp_replay_debug"} or name.startswith("_tmp_eval_") or name.startswith("_tmp_quarter_"):
        return "artifacts"
    if name in CORE_TOP_LEVEL_NAMES or name in ALLOWED_ROOT_NO_MOVE:
        return "core"
    return "uncategorized"


def target_for_top_level_name(name: str, is_dir: bool) -> str | None:
    if name in CORE_TOP_LEVEL_NAMES or name in ALLOWED_ROOT_NO_MOVE:
        return None
    if name in EXPLICIT_FILE_MOVES:
        return EXPLICIT_FILE_MOVES[name]

    if not is_dir:
        if name.endswith(".pth") or name.endswith(".pt") or name.endswith(".ckpt"):
            return f"models/checkpoints/{name}"
        if name.endswith(".md"):
            return f"docs/misc/{name}"
        return None

    if name == "log":
        return "artifacts/debug/log"
    if name.startswith("results_") or name.startswith("sens_") or name.startswith("_eval_"):
        return f"artifacts/interactive_runs/{name}"
    if name in {"results_prop", "results_resume_from90_refineprobe", "results_resume_smoke_from100"}:
        return f"artifacts/interactive_runs/{name}"
    if name.startswith("_tmp_sweep_") or name == "_single_image_input":
        return f"artifacts/sweeps/{name}"
    if name.startswith("_refine_"):
        return f"artifacts/refinement/{name}"
    if name.startswith("_tmp_safe_") or name.startswith("_tmp_novel"):
        return f"artifacts/postprocessing/{name}"
    if name in {"_quarter_run", "_two_quarters"}:
        return f"artifacts/case_studies/{name}"
    if name.startswith("precomputed_ssn_") or name in {"_tmp_quarter_spanno", "_tmp_quarter_spanno_clean"}:
        return f"artifacts/precomputed/{name}"
    if name in {"_debug_candidates", "tmp_replay_debug"} or name.startswith("_tmp_eval_") or name.startswith("_tmp_quarter_"):
        return f"artifacts/debug/{name}"
    if name == "out":
        return None
    return f"artifacts/uncategorized/{name}"


def iter_out_subdir_moves(root: Path) -> Iterable[MoveSpec]:
    out_dir = root / "out"
    if not out_dir.is_dir():
        return []

    specs: list[MoveSpec] = []
    for child in sorted(out_dir.iterdir()):
        if child.name == "sp_postproc_eval" or child.name.startswith("vis_compare_"):
            target = Path("artifacts/postprocessing") / child.name
            reason = "out-postprocessing"
        else:
            target = Path("artifacts/case_studies") / child.name
            reason = "out-case-study"
        specs.append(MoveSpec(str(child.relative_to(root)), str(target), reason))
    return specs


def build_default_move_specs(root: Path) -> list[MoveSpec]:
    specs: list[MoveSpec] = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.name in {"out"}:
            continue
        target = target_for_top_level_name(child.name, child.is_dir())
        if not target:
            continue
        specs.append(MoveSpec(child.name, target, f"top-level:{classify_top_level_name(child.name, child.is_dir())}"))
    specs.extend(iter_out_subdir_moves(root))
    return specs


def is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).exists() for marker in ("metrics.csv", "dynamic_metrics.csv", "run.log"))


def is_summary_file(path: Path) -> bool:
    return path.name in {"summary.csv", "summary.json", "comparison_summary.json", "batch_summary.csv"}


def should_ignore_walk_dir(name: str) -> bool:
    return name in {".git", ".pytest_cache", "__pycache__", ".claude"} or name.endswith(".venv")
