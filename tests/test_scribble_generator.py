import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_interactive_annotation as interactive_eval  # noqa: E402


def _edt_inside(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    dist = distance_transform_edt(padded)
    return dist[1:-1, 1:-1]


def _smooth_region_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask.copy()

    structure = np.ones((2 * int(radius) + 1, 2 * int(radius) + 1), dtype=bool)
    closed = ndimage.binary_erosion(
        ndimage.binary_dilation(mask, structure=structure, iterations=1),
        structure=structure,
        iterations=1,
    )
    opened = ndimage.binary_dilation(
        ndimage.binary_erosion(closed, structure=structure, iterations=1),
        structure=structure,
        iterations=1,
    )
    min_keep = max(1, int(0.20 * float(np.count_nonzero(mask))))
    if int(np.count_nonzero(opened)) >= min_keep:
        return opened.astype(bool, copy=False)
    if closed.any():
        return closed.astype(bool, copy=False)
    return mask.copy()


def _line_pixels(pts01: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    pts_px = np.round(np.asarray(pts01, dtype=np.float32) * np.array([w, h], dtype=np.float32)).astype(int)
    pts_px[:, 0] = np.clip(pts_px[:, 0], 0, w - 1)
    pts_px[:, 1] = np.clip(pts_px[:, 1], 0, h - 1)
    xs_parts = []
    ys_parts = []
    for i in range(max(0, pts_px.shape[0] - 1)):
        p0 = pts_px[i]
        p1 = pts_px[i + 1]
        n = max(abs(int(p1[0] - p0[0])), abs(int(p1[1] - p0[1]))) + 1
        xs = np.linspace(p0[0], p1[0], n).round().astype(int)
        ys = np.linspace(p0[1], p1[1], n).round().astype(int)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        if i > 0 and xs.size > 0:
            xs = xs[1:]
            ys = ys[1:]
        xs_parts.append(xs)
        ys_parts.append(ys)
    if not xs_parts:
        return pts_px[:, 0].copy(), pts_px[:, 1].copy()
    return np.concatenate(xs_parts), np.concatenate(ys_parts)


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
    assert pts01.ndim == 2
    assert pts01.shape[1] == 2
    assert pts01.shape[0] >= 2


def test_generator_balances_underrepresented_class():
    gt = np.array(
        [
            [1, 1, 1, 1, 0, 0, 2, 2],
            [1, 1, 1, 1, 0, 0, 2, 2],
            [1, 1, 1, 1, 0, 0, 2, 2],
            [1, 1, 1, 1, 0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    pred = gt.copy()
    pred[0:3, 0:3] = 0
    pred[0:2, 6:8] = 0

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=3,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, _ = gen.make_scribble(
        pred,
        np.zeros_like(gt, dtype=bool),
        class_scribble_counts=[0, 7, 0],
    )

    assert gt_id == 2


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


def test_generator_cycles_region_selection_modes_within_one_run():
    gt = np.array(
        [
            [1, 1, 1, 1, 0, 0, 2, 2, 0, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 0, 0, 2, 2, 0, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
        ],
        dtype=np.int32,
    )
    pred = gt.copy()

    pred[0, 0] = 0
    pred[0, 1] = 0
    pred[1, 0] = 0
    pred[1, 1] = 0
    pred[2, 0] = 0

    pred[0, 6] = 0
    pred[0, 7] = 0

    pred[0, 9] = -1
    pred[0, 10] = -1
    pred[1, 9] = -1
    pred[1, 10] = -1
    pred[2, 9] = -1
    pred[2, 10] = -1

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=4,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
        region_selection_cycle=["miou_gain", "largest_error", "unannotated"],
    )

    first_gt_id, _ = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))
    second_gt_id, _ = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))
    third_gt_id, _ = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert first_gt_id == 2
    assert second_gt_id == 1
    assert third_gt_id == 3


def test_parse_region_selection_cycle_from_cli_string():
    parsed = interactive_eval.parse_region_selection_cycle(
        "miou_gain, largest_error, unannotated"
    )

    assert parsed == ["miou_gain", "largest_error", "unannotated"]


def test_lookahead_prefers_long_axis_for_elongated_region():
    gt = np.zeros((10, 24), dtype=np.int32)
    gt[3:7, 3:21] = 1
    pred = np.zeros_like(gt)

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=2,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 1
    pts_px = np.round(pts01 * np.array([gt.shape[1], gt.shape[0]])).astype(int)
    dx = int(pts_px[:, 0].max() - pts_px[:, 0].min())
    dy = int(pts_px[:, 1].max() - pts_px[:, 1].min())

    assert dx >= 2 * max(1, dy)
    assert dx >= 10


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

    xs, ys = _line_pixels(pts01, gt.shape)

    assert np.all(safe_inner[ys, xs])


def test_scribble_follows_inner_core_of_bad_region():
    gt = np.zeros((24, 24), dtype=np.int32)
    gt[3:21, 3:21] = 1
    pred = np.zeros_like(gt)

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=2,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 1

    bad_region = (gt == 1) & (pred != 1)
    bad_dt = distance_transform_edt(bad_region)

    xs, ys = _line_pixels(pts01, gt.shape)

    line_dt = bad_dt[ys, xs]
    assert float(line_dt.max()) >= 8.0
    assert float(line_dt.mean()) >= 5.0


def test_scribble_stays_inside_dynamic_normalized_edt_core():
    for y0, y1, x0, x1 in [(3, 21, 3, 21), (8, 20, 9, 15)]:
        gt = np.zeros((28, 28), dtype=np.int32)
        gt[y0:y1, x0:x1] = 1
        pred = np.zeros_like(gt)

        gen = interactive_eval.LargestBadRegionGenerator(
            gt_mask=gt,
            num_classes=2,
            seed=0,
            margin=0,
            border_margin=0,
            no_overlap=False,
        )

        gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

        assert gt_id == 1

        bad_region = (gt == 1) & (pred != 1)
        analysis_mask = gen._analysis_region(bad_region, bad_region)
        center_dt = _edt_inside(analysis_mask if analysis_mask.any() else bad_region)
        corridor_union = np.zeros_like(bad_region, dtype=bool)
        for corridor in gen._edt_corridor_masks(analysis_mask if analysis_mask.any() else bad_region, center_dt):
            corridor_union |= corridor
        xs, ys = _line_pixels(pts01, gt.shape)

        assert np.all(corridor_union[ys, xs])


def test_edt_centerline_avoids_boundary_without_explicit_border_margin():
    gt = np.zeros((24, 24), dtype=np.int32)
    gt[3:21, 3:21] = 1
    pred = np.zeros_like(gt)

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=2,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 1

    bad_region = (gt == 1) & (pred != 1)
    bad_dt = _edt_inside(bad_region)
    xs, ys = _line_pixels(pts01, gt.shape)

    assert float(np.min(bad_dt[ys, xs])) > 1.0


def test_scribble_respects_image_frame_as_outer_border():
    gt = np.zeros((18, 18), dtype=np.int32)
    gt[2:16, 0:12] = 1
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
    safe_inner = _edt_inside(bad_region) > 2
    xs, ys = _line_pixels(pts01, gt.shape)

    assert np.all(safe_inner[ys, xs])


def test_scribble_midpoint_stays_on_smoothed_region_center():
    gt = np.zeros((28, 28), dtype=np.int32)
    gt[4:24, 4:24] = 1
    pred = np.zeros_like(gt)

    pred[4:8, 8:11] = 1
    pred[9:13, 4:7] = 1
    pred[15:19, 21:24] = 1
    pred[20:24, 13:16] = 1
    pred[11:15, 11:14] = 1

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=2,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 1

    bad_region = (gt == 1) & (pred != 1)
    smooth_bad = _smooth_region_mask(bad_region)
    smooth_dt = _edt_inside(smooth_bad)
    xs, ys = _line_pixels(pts01, gt.shape)

    assert float(smooth_dt[ys, xs].max()) >= 0.95 * float(smooth_dt.max())
    assert float(smooth_dt[ys, xs].mean()) >= 0.45 * float(smooth_dt.max())


def test_scribble_follows_curved_edt_centerline():
    gt = np.zeros((32, 32), dtype=np.int32)
    gt[4:26, 4:10] = 1
    gt[20:26, 4:26] = 1
    pred = np.zeros_like(gt)

    gen = interactive_eval.LargestBadRegionGenerator(
        gt_mask=gt,
        num_classes=2,
        seed=0,
        margin=0,
        border_margin=0,
        no_overlap=False,
    )

    gt_id, pts01 = gen.make_scribble(pred, np.zeros_like(gt, dtype=bool))

    assert gt_id == 1
    assert pts01.shape[0] > 2

    pts_px = np.round(pts01 * np.array([gt.shape[1], gt.shape[0]])).astype(int)
    dx = int(pts_px[:, 0].max() - pts_px[:, 0].min())
    dy = int(pts_px[:, 1].max() - pts_px[:, 1].min())
    step_dirs = np.diff(pts_px, axis=0)
    has_turn = np.any(
        (step_dirs[:-1, 0] * step_dirs[1:, 1] - step_dirs[:-1, 1] * step_dirs[1:, 0]) != 0
    )

    assert dx >= 10
    assert dy >= 10
    assert has_turn
