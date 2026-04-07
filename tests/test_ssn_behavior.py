import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "superpixel_annotator"))
sys.path.insert(0, str(ROOT))

import structs  # noqa: E402
import evaluate_interactive_annotation as interactive_eval  # noqa: E402


def test_disconnected_regions_are_split_into_different_superpixels():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:3, 1:3] = 7
    labels[5:7, 5:7] = 7

    out = structs._postprocess_superpixel_labels(
        labels,
        nspix_hint=2,
        prune_small_thin=False,
    )

    assert out[1, 1] > 0
    assert out[5, 5] > 0
    assert out[1, 1] != out[5, 5]


def test_small_or_thin_regions_merge_into_neighbors_for_ssn_cleanup():
    labels = np.zeros((24, 24), dtype=np.int32)
    labels[3:20, 4:12] = 1       # left compact block
    labels[3:20, 12] = 2         # thin separator touching both sides
    labels[3:20, 13:20] = 3      # right compact block

    out = structs._postprocess_superpixel_labels(
        labels,
        nspix_hint=3,
        prune_small_thin=True,
    )

    line = out[3:20, 12]
    assert np.all(line > 0)
    assert np.all((line == out[3:20, 11]) | (line == out[3:20, 13]))


def test_ssn_method_enables_embedding_defaults_automatically():
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SSNSuperpixel(
        weight_path="relative/path/to/model.pth",
        nspix=25,
        fdim=20,
        niter=5,
        color_scale=0.26,
        pos_scale=2.5,
    )

    algo.add_superpixel_method(method)

    assert algo.embedding_weight_path == os.path.abspath("relative/path/to/model.pth")
    assert algo.embedding_fdim == 20
    assert algo.embedding_color_scale == 0.26
    assert algo.embedding_pos_scale == 2.5


def test_embedding_threshold_default_is_consistent():
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    parser = interactive_eval.build_parser()
    args = parser.parse_args(["--out", "tmp_out"])

    assert algo._embedding_threshold == 0.99
    assert args.emb_threshold == 0.99


def test_bbox_containment_uses_union_and_includes_boundaries():
    scribble = np.array([[0.20, 0.20], [0.50, 0.20]], dtype=np.float32)
    rectangles = [
        [0.10, 0.10, 0.35, 0.30],
        [0.35, 0.10, 0.60, 0.30],
    ]

    assert structs.check_bbox_contain_scribble(scribble, rectangles)


def test_propagation_can_reach_disconnected_neighbor_without_topology_gate():
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]

    seed_sp = structs.SuperPixel(
        id=0,
        method=method.short_string(),
        border=np.array(
            [[0.10, 0.10], [0.20, 0.10], [0.20, 0.20], [0.10, 0.20]],
            dtype=np.float32,
        ),
        parents=[],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    neigh_sp = structs.SuperPixel(
        id=1,
        method=method.short_string(),
        border=np.array(
            [[0.26, 0.10], [0.36, 0.10], [0.36, 0.20], [0.26, 0.20]],
            dtype=np.float32,
        ),
        parents=[],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    algo.superpixels[method] = [seed_sp, neigh_sp]
    algo._annotations[method] = structs.ImageAnnotation(
        annotations=[
            structs.AnnotationInstance(
                id=0,
                code=1,
                border=seed_sp.border.copy(),
                parent_superpixel=seed_sp.id,
                parent_scribble=[0],
                parent_intersect=True,
            )
        ]
    )
    algo._superpixel_ind[method] = 2
    algo._annotation_ind[method] = 1
    algo._mark_sp_index_dirty()

    scrib = structs.Scribble(
        id=0,
        points=np.array([[0.11, 0.15], [0.19, 0.15]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=1),
    )
    algo._scribbles = [scrib]

    algo.use_sensitivity_for_region(
        sp_idx=0,
        sens=2.0,
        scribble=scrib,
        radius_scale=1.0,
        property_scale=1.0,
    )

    annos = algo._annotations[method].annotations
    assert any(a.parent_superpixel == 1 for a in annos)


def test_propagation_can_relabel_previously_propagated_superpixels():
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SLICSuperpixel(n_clusters=3, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]

    seed_sp = structs.SuperPixel(
        id=0,
        method=method.short_string(),
        border=np.array(
            [[0.10, 0.10], [0.20, 0.10], [0.20, 0.20], [0.10, 0.20]],
            dtype=np.float32,
        ),
        parents=[],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    bridge_sp = structs.SuperPixel(
        id=1,
        method=method.short_string(),
        border=np.array(
            [[0.22, 0.10], [0.32, 0.10], [0.32, 0.20], [0.22, 0.20]],
            dtype=np.float32,
        ),
        parents=[],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    algo.superpixels[method] = [seed_sp, bridge_sp]
    algo._annotations[method] = structs.ImageAnnotation(
        annotations=[
            structs.AnnotationInstance(
                id=0,
                code=1,
                border=seed_sp.border.copy(),
                parent_superpixel=seed_sp.id,
                parent_scribble=[0],
                parent_intersect=True,
            ),
            structs.AnnotationInstance(
                id=1,
                code=2,
                border=bridge_sp.border.copy(),
                parent_superpixel=bridge_sp.id,
                parent_scribble=[0],
                parent_intersect=False,
            ),
        ]
    )
    algo._superpixel_ind[method] = 2
    algo._annotation_ind[method] = 2
    algo._mark_sp_index_dirty()

    scrib = structs.Scribble(
        id=1,
        points=np.array([[0.11, 0.15], [0.19, 0.15]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=1),
    )
    algo._scribbles = [scrib]

    algo.use_sensitivity_for_region(
        sp_idx=0,
        sens=2.0,
        scribble=scrib,
        radius_scale=1.0,
        property_scale=1.0,
    )

    ann_by_sp = {a.parent_superpixel: a for a in algo._annotations[method].annotations}
    assert ann_by_sp[1].code == 1
    assert ann_by_sp[1].parent_intersect is False
    assert 1 in ann_by_sp[1].parent_scribble


def test_split_descendant_is_not_split_twice():
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
        auto_propagation_sensitivity=0.0,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]

    child_sp = structs.SuperPixel(
        id=10,
        method=method.short_string(),
        border=np.array(
            [[0.20, 0.20], [0.60, 0.20], [0.60, 0.60], [0.20, 0.60]],
            dtype=np.float32,
        ),
        parents=[3],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    algo.superpixels[method] = [child_sp]
    algo._annotations[method] = structs.ImageAnnotation(
        annotations=[
            structs.AnnotationInstance(
                id=0,
                code=1,
                border=child_sp.border.copy(),
                parent_superpixel=child_sp.id,
                parent_scribble=[0],
                parent_intersect=True,
            )
        ]
    )
    algo._superpixel_ind[method] = 11
    algo._annotation_ind[method] = 1
    algo._mark_sp_index_dirty()

    scrib = structs.Scribble(
        id=1,
        points=np.array([[0.25, 0.40], [0.55, 0.40]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=2),
    )

    algo.add_scribble(scrib)

    annos = algo._annotations[method].annotations
    assert len(algo.superpixels[method]) == 1
    assert len(annos) == 1
    assert annos[0].parent_superpixel == child_sp.id
    assert annos[0].code == 2


def test_add_scribble_split_does_not_create_overlapping_superpixels():
    image = Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
        auto_propagation_sensitivity=0.0,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]

    parent_sp = structs.SuperPixel(
        id=0,
        method=method.short_string(),
        border=np.array(
            [[0.20, 0.20], [0.80, 0.20], [0.80, 0.80], [0.20, 0.80]],
            dtype=np.float32,
        ),
        parents=[],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    algo.superpixels[method] = [parent_sp]
    algo._annotations[method] = structs.ImageAnnotation(
        annotations=[
            structs.AnnotationInstance(
                id=0,
                code=1,
                border=parent_sp.border.copy(),
                parent_superpixel=parent_sp.id,
                parent_scribble=[0],
                parent_intersect=True,
            )
        ]
    )
    algo._superpixel_ind[method] = 1
    algo._annotation_ind[method] = 1
    algo._mark_sp_index_dirty()

    old_scribble = structs.Scribble(
        id=0,
        points=np.array([[0.26, 0.35], [0.74, 0.35]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=1),
    )
    algo._scribbles = [old_scribble]
    algo.scribbles_id_sequence = [0]
    algo.ind_scrible = 1

    new_scribble = structs.Scribble(
        id=-1,
        points=np.array([[0.26, 0.58], [0.74, 0.58]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=2),
    )

    algo.add_scribble(new_scribble)

    sps = algo.superpixels[method]
    assert len(sps) >= 2

    for i, sp_i in enumerate(sps):
        for sp_j in sps[i + 1:]:
            inter = sp_i.poly.intersection(sp_j.poly)
            assert float(getattr(inter, "area", 0.0)) <= 1e-8


def test_empty_local_superpixel_generation_does_not_save_bbox(monkeypatch):
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]
    algo.superpixels[method] = []
    algo._annotations[method] = structs.ImageAnnotation(annotations=[])
    algo._superpixel_ind[method] = 0
    algo._annotation_ind[method] = 0

    def fake_create(self, superpixel_method, image_roi, mask, bbox, scribble, forbidden_bboxes=None):
        return 0

    monkeypatch.setattr(structs.SuperPixelAnnotationAlgo, "_create_superpixel_for_mask", fake_create)

    scrib = structs.Scribble(
        id=0,
        points=np.array([[0.25, 0.25], [0.40, 0.25]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=1),
    )

    algo._create_superpixel_for_scribble(scrib, method)

    assert algo.annotated_bbox == []


def test_scribble_outside_existing_mask_creates_non_overlapping_superpixels(monkeypatch):
    image = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]

    old_sp = structs.SuperPixel(
        id=0,
        method=method.short_string(),
        border=np.array(
            [[0.05, 0.20], [0.35, 0.20], [0.35, 0.60], [0.05, 0.60]],
            dtype=np.float32,
        ),
        parents=[],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    algo.superpixels[method] = [old_sp]
    algo._annotations[method] = structs.ImageAnnotation(annotations=[])
    algo._superpixel_ind[method] = 1
    algo._annotation_ind[method] = 0
    algo.annotated_bbox = [[0.05, 0.20, 0.35, 0.60]]

    def fake_compute_superpixels(image_roi, method_obj, mask=None, embedding_guided_cleanup=False):
        out = np.zeros(mask.shape, dtype=np.int32)
        out[mask.astype(bool)] = 1
        return out

    monkeypatch.setattr(structs, "compute_superpixels", fake_compute_superpixels)

    scrib = structs.Scribble(
        id=0,
        points=np.array([[0.62, 0.42], [0.68, 0.42]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=2),
    )

    assert algo._exist_sp_mask(scrib) is False

    algo._create_superpixel_for_scribble(scrib, method)

    assert len(algo.superpixels[method]) >= 2
    new_sps = [sp for sp in algo.superpixels[method] if int(sp.id) != 0]
    assert new_sps

    for sp in new_sps:
        inter = old_sp.poly.intersection(sp.poly)
        assert float(getattr(inter, "area", 0.0)) <= 1e-8


def test_refine_mask_preserves_scribble_corridor_inside_bbox_union(monkeypatch):
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]
    algo.superpixels[method] = []
    algo._annotations[method] = structs.ImageAnnotation(annotations=[])
    algo._superpixel_ind[method] = 0
    algo._annotation_ind[method] = 0
    algo.annotated_bbox = [
        [0.10, 0.10, 0.35, 0.35],
        [0.35, 0.10, 0.60, 0.35],
    ]
    captured = {}

    def fake_create(self, superpixel_method, image_roi, mask, bbox, scribble, forbidden_bboxes=None):
        captured["mask"] = mask.copy()
        captured["bbox"] = list(bbox)
        return 1

    monkeypatch.setattr(structs.SuperPixelAnnotationAlgo, "_create_superpixel_for_mask", fake_create)

    scrib = structs.Scribble(
        id=0,
        points=np.array([[0.20, 0.20], [0.50, 0.20]], dtype=np.float32),
        params=structs.ScribbleParams(radius=1, code=1),
    )

    algo._create_superpixel_for_scribble(scrib, method)

    assert "mask" in captured
    ix0, iy0, ix1, iy1 = structs._bbox_to_pixel_rect(tuple(captured["bbox"]), 64, 64)
    scribble_mask = algo._rasterize_scribble_to_roi_mask(scrib, (ix0, iy0, ix1, iy1), width_px=3)
    assert scribble_mask is not None
    assert np.any(captured["mask"][scribble_mask])
    assert len(algo.annotated_bbox) == 3


def test_serialize_roundtrip_preserves_split_superpixels_and_bbox(tmp_path):
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
    algo = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    method = structs.SLICSuperpixel(n_clusters=2, compactness=10.0, sigma=0.0)
    algo.superpixel_methods = [method]

    child_sp = structs.SuperPixel(
        id=10,
        method=method.short_string(),
        border=np.array(
            [[0.20, 0.20], [0.60, 0.20], [0.60, 0.60], [0.20, 0.60]],
            dtype=np.float32,
        ),
        parents=[3, 7],
        props=np.array([1, 1, 1, 0, 0, 0], dtype=np.float32),
        emb=None,
    )
    algo.superpixels[method] = [child_sp]
    algo._annotations[method] = structs.ImageAnnotation(
        annotations=[
            structs.AnnotationInstance(
                id=0,
                code=2,
                border=child_sp.border.copy(),
                parent_superpixel=child_sp.id,
                parent_scribble=[5],
                parent_intersect=True,
            )
        ]
    )
    algo._superpixel_ind[method] = 11
    algo._annotation_ind[method] = 1
    algo._scribbles = [
        structs.Scribble(
            id=5,
            points=np.array([[0.25, 0.40], [0.55, 0.40]], dtype=np.float32),
            params=structs.ScribbleParams(radius=1, code=2),
        )
    ]
    algo.annotated_bbox = [[0.10, 0.10, 0.40, 0.40]]

    state_path = tmp_path / "roundtrip_state.json"
    algo.serialize(str(state_path), make_backup=False)

    loaded = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image=image,
    )
    loaded.deserialize(str(state_path))

    assert loaded.annotated_bbox == [[0.10, 0.10, 0.40, 0.40]]
    assert len(loaded.superpixel_methods) == 1
    loaded_method = loaded.superpixel_methods[0]
    loaded_sp = loaded.superpixels[loaded_method][0]
    loaded_anno = loaded._annotations[loaded_method].annotations[0]
    assert loaded_sp.parents == [3, 7]
    assert loaded_anno.parent_superpixel == 10
    assert loaded_anno.parent_scribble == [5]
