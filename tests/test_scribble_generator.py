import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_interactive_annotation as interactive_eval  # noqa: E402


def test_generator_prefers_class_with_better_expected_miou_gain():
    gt = np.array(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 2, 2],
            [0, 0, 0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    pred = gt.copy()

    # Larger absolute error for class 1.
    pred[0, 0] = 0
    pred[0, 1] = 0
    pred[1, 0] = 0

    # Smaller component for class 2, but much larger expected IoU gain.
    pred[4, 4] = 0
    pred[4, 5] = 0

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=3,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 2
    assert pts01.shape == (2, 2)


def test_generator_switches_candidate_after_no_progress():
    gt = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2],
            [0, 0, 0, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    pred = gt.copy()

    # Class 1 is initially the best candidate.
    pred[0, 0] = 0
    pred[0, 1] = 0
    pred[1, 0] = 0
    pred[1, 1] = 0

    # Class 2 is the runner-up and should be tried after a failed step.
    pred[4, 3] = 0
    pred[4, 4] = 0

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=3,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    first_gt_id, _ = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))
    gen.report_last_result(progress=False)
    second_gt_id, _ = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert first_gt_id == 1
    assert second_gt_id == 2


def test_scribble_stays_away_from_bad_region_border():
    gt = np.zeros((14, 14), dtype=np.int32)
    gt[2:12, 2:12] = 1
    pred = np.zeros_like(gt)

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=2,
        seed=0,
        margin=0,
        border_margin=2,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 1

    bad_region = (gt == 1) & (pred != 1)
    safe_inner = distance_transform_edt(bad_region) > 2

    p0 = np.round(pts01[0] * np.array([gt.shape[1], gt.shape[0]])).astype(int)
    p1 = np.round(pts01[1] * np.array([gt.shape[1], gt.shape[0]])).astype(int)

    n = max(abs(int(p1[0] - p0[0])), abs(int(p1[1] - p0[1]))) + 1
    xs = np.linspace(p0[0], p1[0], n).round().astype(int)
    ys = np.linspace(p0[1], p1[1], n).round().astype(int)

    assert np.all(safe_inner[ys, xs])
