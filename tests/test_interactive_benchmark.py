import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from interactive_benchmark.baseline import normalize_current_pipeline_metrics
from interactive_benchmark.oracle import InteractionOracle
from interactive_benchmark.session import SessionState
from interactive_benchmark.contracts import MaskProposal, PromptPayload


ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_pair(tmp_path: Path) -> tuple[Path, Path]:
    height, width = 48, 48
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image[..., 1] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    image[12:36, 12:36, 2] = 255

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[12:36, 12:36] = 1

    image_path = tmp_path / "demo.png"
    mask_path = tmp_path / "demo_mask.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    return image_path, mask_path


def test_normalize_current_pipeline_metrics_forward_fills_budget():
    rows = [
        {"step": 0.0, "n_scribbles": 0.0, "coverage": 0.0, "annotation_precision": 0.0, "miou": 0.0, "total_ink_px": 0.0},
        {"step": 1.0, "n_scribbles": 1.0, "coverage": 0.2, "annotation_precision": 0.8, "miou": 0.1, "total_ink_px": 10.0},
        {"step": 3.0, "n_scribbles": 3.0, "coverage": 0.5, "annotation_precision": 0.9, "miou": 0.4, "total_ink_px": 30.0},
    ]

    normalized = normalize_current_pipeline_metrics(
        rows,
        interaction_budgets=[1, 2, 3],
        image_name="demo",
    )

    assert [row["interaction_budget"] for row in normalized] == [1, 2, 3]
    assert normalized[1]["step"] == 1
    assert normalized[1]["miou"] == 0.1
    assert normalized[2]["step"] == 3
    assert normalized[2]["miou"] == 0.4


def test_session_state_uses_last_update_wins():
    session = SessionState(image_shape=(8, 8), num_classes=2)

    mask_a = np.zeros((8, 8), dtype=bool)
    mask_a[1:6, 1:6] = True
    session.apply_proposal(
        MaskProposal(class_id=0, mask=mask_a, source="test"),
        interaction_id=1,
        prompt=PromptPayload(prompt_type="point", class_id=0, interaction_id=1, points=[(0.2, 0.2)]),
    )

    mask_b = np.zeros((8, 8), dtype=bool)
    mask_b[4:7, 4:7] = True
    session.apply_proposal(
        MaskProposal(class_id=1, mask=mask_b, source="test"),
        interaction_id=2,
        prompt=PromptPayload(prompt_type="point", class_id=1, interaction_id=2, points=[(0.7, 0.7)]),
    )

    assert session.canvas.labels[2, 2] == 0
    assert session.canvas.labels[5, 5] == 1


def test_interaction_oracle_respects_prompt_profiles():
    gt = np.zeros((24, 24), dtype=np.int32)
    gt[6:18, 6:18] = 1
    pred = np.full_like(gt, -1)
    used = np.zeros_like(gt, dtype=bool)
    class_counts = [0, 0]

    point_oracle = InteractionOracle(gt_mask=gt, num_classes=2, prompt_type="point", seed=0)
    point_selection = point_oracle.next_interaction(pred, used, class_counts, interaction_id=1)
    assert point_selection.prompt.prompt_type == "point"
    assert len(point_selection.prompt.points) == 1

    line_oracle = InteractionOracle(gt_mask=gt, num_classes=2, prompt_type="line", seed=0)
    line_selection = line_oracle.next_interaction(pred, used, class_counts, interaction_id=1)
    assert line_selection.prompt.prompt_type == "line"
    assert 1 <= len(line_selection.prompt.points) <= 2

    scribble_oracle = InteractionOracle(gt_mask=gt, num_classes=2, prompt_type="scribble", seed=0)
    scribble_selection = scribble_oracle.next_interaction(pred, used, class_counts, interaction_id=1)
    assert scribble_selection.prompt.prompt_type == "scribble"
    assert len(scribble_selection.prompt.points) >= 1


def test_benchmark_cli_smoke_with_mock_and_real_current_pipeline(tmp_path):
    image_path, mask_path = _write_synthetic_pair(tmp_path)
    out_dir = tmp_path / "benchmark_out"
    current_args = {
        "method": "slic",
        "n_segments": 24,
        "compactness": 4.0,
        "sigma": 0.0,
        "sensitivity": 0.0,
        "max_no_progress": 2,
    }
    cmd = [
        sys.executable,
        str(ROOT / "benchmark_interactive_methods.py"),
        "--image",
        str(image_path),
        "--mask",
        str(mask_path),
        "--output-dir",
        str(out_dir),
        "--methods",
        "current_pipeline,mock_click",
        "--interaction-budgets",
        "1,2",
        "--current-pipeline-args",
        json.dumps(current_args),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    per_step_path = out_dir / "per_step.csv"
    summary_path = out_dir / "summary.csv"
    leaderboard_path = out_dir / "leaderboard.md"
    plot_path = out_dir / "quality_vs_interactions.png"

    assert per_step_path.exists()
    assert summary_path.exists()
    assert leaderboard_path.exists()
    assert plot_path.exists()

    with open(per_step_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    methods = {row["method_id"] for row in rows}
    assert "current_pipeline" in methods
    assert "mock_click" in methods
