import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "superpixel_annotator"))
sys.path.insert(0, str(ROOT))

import structs  # noqa: E402
import precompute_superpixels  # noqa: E402


def _synthetic_lab_image(height=32, width=32):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    lab = np.zeros((height, width, 3), dtype=np.float32)
    lab[..., 0] = 100.0 * xx
    lab[..., 1] = 40.0 * (yy - 0.5)
    lab[..., 2] = 30.0 * (xx - yy)
    lab[height // 4 : height // 2, width // 4 : width // 2, 0] += 30.0
    return lab


def _neural_methods():
    return [
        structs.DeepSLICSuperpixel(nspix=25, fdim=12, niter=3),
        structs.CNNRIMSuperpixel(nspix=25, fdim=12, niter=3, optim_steps=0),
        structs.SPFCNSuperpixel(nspix=25, fdim=12, refine_steps=1),
        structs.SINSuperpixel(nspix=25, fdim=12, interp_steps=2),
        structs.RethinkUnsupSuperpixel(nspix=25, fdim=12, niter=3, optim_steps=0),
    ]


def test_build_neural_method_from_args_with_config():
    args = argparse.Namespace(
        method="deep_slic",
        method_config='{"nspix": 49, "fdim": 16, "niter": 4, "backbone_width": 24}',
        weights="relative/weights.pth",
    )
    method = structs.build_superpixel_method_from_args(args)

    assert isinstance(method, structs.DeepSLICSuperpixel)
    assert method.nspix == 49
    assert method.fdim == 16
    assert method.niter == 4
    assert method.backbone_width == 24
    assert method.weight_path == "relative/weights.pth"


def test_parse_method_string_roundtrip_for_all_neural_methods():
    methods = [
        structs.DeepSLICSuperpixel(weight_path="foo.pth"),
        structs.CNNRIMSuperpixel(weight_path="bar.pth"),
        structs.SPFCNSuperpixel(weight_path="baz.pth"),
        structs.SINSuperpixel(weight_path="sin.pth"),
        structs.RethinkUnsupSuperpixel(weight_path="rethink.pth"),
    ]

    for method in methods:
        parsed = structs.SuperPixelAnnotationAlgo._parse_method_from_string(method.short_string())
        assert parsed.short_string() == method.short_string()


def test_compute_superpixels_for_neural_methods_supports_mask():
    image_lab = _synthetic_lab_image()
    mask = np.zeros(image_lab.shape[:2], dtype=bool)
    mask[6:26, 5:27] = True

    for method in _neural_methods():
        labels = structs.compute_superpixels(image_lab, method, mask=mask)
        assert labels.shape == image_lab.shape[:2]
        assert labels.dtype == np.int32
        assert np.all(labels[~mask] == 0)
        assert np.any(labels[mask] > 0)


def test_precompute_pipeline_serializes_neural_superpixel_state(tmp_path):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(0, 255, 40, dtype=np.uint8)[None, :]
    image[..., 1] = np.linspace(255, 0, 40, dtype=np.uint8)[:, None]

    img_path = tmp_path / "sample.png"
    out_path = tmp_path / "sample.spanno.json.gz"
    Image.fromarray(image, mode="RGB").save(img_path)

    method = structs.SINSuperpixel(nspix=36, fdim=10, interp_steps=2)
    ok = precompute_superpixels.process_one_image(
        image_path=img_path,
        out_path=out_path,
        sp_method=method,
        downscale=1.0,
        overwrite=True,
        logger=logging.getLogger("test_precompute_neural"),
    )

    assert ok is True
    assert out_path.exists()

    loaded = structs.SuperPixelAnnotationAlgo(
        downscale_coeff=1.0,
        superpixel_methods=[],
        image_path=img_path,
    )
    loaded.deserialize(str(out_path))
    assert len(loaded.superpixel_methods) == 1
    assert loaded.superpixel_methods[0] == method
    assert len(loaded.superpixels[loaded.superpixel_methods[0]]) > 0
