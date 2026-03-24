import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render_interactive_annotation_video as replay  # noqa: E402


def _rect(x0, y0, x1, y1):
    return [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ]


def _state_payload(step, scribbles, annotations, superpixels, width=64, height=48):
    method = "SLIC_TEST"
    return {
        "_meta": {
            "magic": "spanno",
            "version": 2,
            "image": {
                "path": "",
                "size_wh": [width, height],
                "lab_shape": [height, width, 3],
                "downscale_coeff": 1.0,
            },
            "methods": [method],
            "checks": {
                "n_scribbles": len(scribbles),
                "n_superpixels": {method: len(superpixels)},
                "n_annotations": {method: len(annotations)},
            },
        },
        "scribbles": scribbles,
        "superpixels": {
            method: superpixels,
        },
        "annotations": {
            method: annotations,
        },
        "bbox": [[0.0, 0.0], [1.0, 1.0]],
    }


def _write_state(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_metrics_csv(path: Path, rows):
    lines = [
        "step,n_scribbles,coverage,annotation_precision,miou,annotated_px",
    ]
    for row in rows:
        lines.append(
            f"{row['step']},{row['n_scribbles']},{row['coverage']},{row['annotation_precision']},"
            f"{row['miou']},{row['annotated_px']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_scribble(sid, code, p0, p1):
    return {
        "id": sid,
        "points": [list(p0), list(p1)],
        "params": {"radius": 1.0, "code": code},
        "creation_time": "2026-03-24T00:00:00",
    }


def _build_superpixel(sp_id, border):
    return {
        "id": sp_id,
        "method": "SLIC_TEST",
        "border": border,
        "parents": [],
        "props": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _build_annotation(anno_id, code, sp_id, border, scribble_ids, parent_intersect):
    return {
        "id": anno_id,
        "code": code,
        "border": border,
        "parent_superpixel": sp_id,
        "parent_scribble": list(scribble_ids),
        "parent_intersect": bool(parent_intersect),
    }


def test_build_events_marks_sparse_checkpoint_delta(tmp_path):
    run_dir = tmp_path / "run_sparse"
    run_dir.mkdir()

    sp1 = _build_superpixel(1, _rect(0.05, 0.10, 0.30, 0.40))
    sp2 = _build_superpixel(2, _rect(0.35, 0.10, 0.60, 0.40))
    sp3 = _build_superpixel(3, _rect(0.65, 0.10, 0.90, 0.40))

    state1 = _state_payload(
        step=1,
        scribbles=[_build_scribble(1, 1, (0.10, 0.20), (0.25, 0.25))],
        annotations=[_build_annotation(0, 1, 1, sp1["border"], [1], True)],
        superpixels=[sp1, sp2, sp3],
    )
    state3 = _state_payload(
        step=3,
        scribbles=[
            _build_scribble(1, 1, (0.10, 0.20), (0.25, 0.25)),
            _build_scribble(2, 2, (0.42, 0.24), (0.55, 0.28)),
            _build_scribble(3, 2, (0.70, 0.22), (0.82, 0.27)),
        ],
        annotations=[
            _build_annotation(0, 1, 1, sp1["border"], [1], True),
            _build_annotation(1, 2, 2, sp2["border"], [2], True),
            _build_annotation(2, 2, 3, sp3["border"], [3], False),
        ],
        superpixels=[sp1, sp2, sp3],
    )

    _write_state(run_dir / "state_000001.json", state1)
    _write_state(run_dir / "state_000003.json", state3)

    states = [replay.load_state(path, max_side=128, method=None) for path in replay.discover_state_files(run_dir)]
    events = replay.build_events(states)

    assert len(events) == 2
    assert events[0].step_delta == 1
    assert events[0].exact is True
    assert events[1].step_delta == 2
    assert events[1].exact is False
    assert events[1].direct_sp_ids == (2,)
    assert events[1].propagated_sp_ids == (3,)


def test_render_run_video_writes_mp4(tmp_path):
    run_dir = tmp_path / "run_exact"
    run_dir.mkdir()

    sp1 = _build_superpixel(1, _rect(0.05, 0.10, 0.30, 0.40))
    sp2 = _build_superpixel(2, _rect(0.35, 0.10, 0.60, 0.40))
    sp3 = _build_superpixel(3, _rect(0.65, 0.10, 0.90, 0.40))

    state1 = _state_payload(
        step=1,
        scribbles=[_build_scribble(1, 1, (0.10, 0.20), (0.25, 0.25))],
        annotations=[_build_annotation(0, 1, 1, sp1["border"], [1], True)],
        superpixels=[sp1, sp2, sp3],
    )
    state2 = _state_payload(
        step=2,
        scribbles=[
            _build_scribble(1, 1, (0.10, 0.20), (0.25, 0.25)),
            _build_scribble(2, 2, (0.42, 0.24), (0.55, 0.28)),
        ],
        annotations=[
            _build_annotation(0, 1, 1, sp1["border"], [1], True),
            _build_annotation(1, 2, 2, sp2["border"], [2], True),
            _build_annotation(2, 2, 3, sp3["border"], [2], False),
        ],
        superpixels=[sp1, sp2, sp3],
    )

    _write_state(run_dir / "state_000001.json", state1)
    _write_state(run_dir / "state_000002.json", state2)
    _write_metrics_csv(
        run_dir / "metrics.csv",
        [
            {"step": 1, "n_scribbles": 1, "coverage": 0.2, "annotation_precision": 1.0, "miou": 0.3, "annotated_px": 100},
            {"step": 2, "n_scribbles": 2, "coverage": 0.5, "annotation_precision": 0.9, "miou": 0.6, "annotated_px": 220},
        ],
    )

    out_path = tmp_path / "replay.mp4"
    logger = logging.getLogger("replay_test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    replay.render_run_video(
        run_dir=run_dir,
        out_path=out_path,
        image_path=None,
        fps=4.0,
        max_side=128,
        method=None,
        timing=replay.TimingConfig(
            intro_frames=1,
            pre_frames=1,
            direct_frames=1,
            prop_frames=1,
            final_frames=1,
            outro_frames=1,
        ),
        logger=logger,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
