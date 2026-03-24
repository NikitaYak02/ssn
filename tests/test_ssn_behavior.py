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


def test_small_or_thin_regions_become_background_for_ssn_cleanup():
    labels = np.zeros((24, 24), dtype=np.int32)
    labels[3:20, 5] = 1          # thin line
    labels[8:16, 12:20] = 2      # compact block

    out = structs._postprocess_superpixel_labels(
        labels,
        nspix_hint=2,
        prune_small_thin=True,
    )

    assert np.all(out[3:20, 5] == 0)
    assert np.any(out[8:16, 12:20] > 0)


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
