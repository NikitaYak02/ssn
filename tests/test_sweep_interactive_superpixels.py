import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sweep_interactive_superpixels as sweep  # noqa: E402


def _parse_args(tmp_path: Path, extra_args: list[str]):
    image_path = tmp_path / "dummy.png"
    mask_path = tmp_path / "dummy_mask.png"
    out_dir = tmp_path / "out"
    return sweep.build_parser().parse_args(
        [
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output-dir",
            str(out_dir),
            *extra_args,
        ]
    )


def _write_demo_image(path: Path) -> None:
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, 40, dtype=np.float32),
        np.linspace(0.0, 1.0, 40, dtype=np.float32),
        indexing="ij",
    )
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[..., 0] = np.clip(255.0 * xx, 0, 255).astype(np.uint8)
    image[..., 1] = np.clip(255.0 * yy, 0, 255).astype(np.uint8)
    image[..., 2] = np.clip(255.0 * (1.0 - xx), 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_build_cases_supports_watershed_alias(tmp_path):
    args = _parse_args(
        tmp_path,
        [
            "--methods",
            "slic,ws",
            "--slic-n-segments",
            "12",
            "--slic-compactnesses",
            "3.5",
            "--slic-sigmas",
            "0.0",
            "--ws-compactnesses",
            "0.002",
            "--ws-components-list",
            "9",
        ],
    )

    cases = sweep.build_cases(args, output_dir=Path(args.output_dir))

    assert [case.method for case in cases] == ["slic", "watershed"]
    assert cases[0].params == {
        "n_segments": 12,
        "compactness": 3.5,
        "sigma": 0.0,
    }
    assert cases[1].params == {
        "ws_compactness": 0.002,
        "ws_components": 9,
    }


def test_build_cases_supports_neural_method_configs(tmp_path):
    args = _parse_args(
        tmp_path,
        [
            "--methods",
            "sin,deep_slic",
            "--neural-method-configs",
            json.dumps(
                {
                    "sin": [{"nspix": 16, "fdim": 8, "interp_steps": 1}],
                    "deep_slic": [
                        {
                            "nspix": 9,
                            "fdim": 6,
                            "niter": 2,
                            "weights": str(tmp_path / "deep_slic_demo.pth"),
                        }
                    ],
                }
            ),
        ],
    )

    cases = sweep.build_cases(args, output_dir=Path(args.output_dir))

    assert [case.method for case in cases] == ["sin", "deep_slic"]

    sin_config = json.loads(cases[0].params["method_config"])
    assert sin_config == {"fdim": 8, "interp_steps": 1, "nspix": 16}

    deep_slic_config = json.loads(cases[1].params["method_config"])
    assert deep_slic_config == {"fdim": 6, "niter": 2, "nspix": 9}
    assert cases[1].params["weights"] == str((tmp_path / "deep_slic_demo.pth").resolve())


def test_build_eval_command_preserves_neural_config_flags(tmp_path):
    args = _parse_args(tmp_path, ["--methods", "sin"])
    case = sweep.SweepCase(
        method="sin",
        label="sin_demo",
        params={
            "method_config": json.dumps({"nspix": 16, "fdim": 8, "interp_steps": 1}),
            "weights": str(tmp_path / "sin_demo.pth"),
        },
        spanno_path=str(tmp_path / "demo.spanno.json.gz"),
        run_dir=str(tmp_path / "run"),
    )

    cmd = sweep.build_eval_command(args, case)

    assert "--method" in cmd
    assert "sin" in cmd
    assert "--method_config" in cmd
    assert json.loads(cmd[cmd.index("--method_config") + 1]) == {
        "nspix": 16,
        "fdim": 8,
        "interp_steps": 1,
    }
    assert cmd[cmd.index("--weights") + 1] == str(tmp_path / "sin_demo.pth")


def test_ensure_spanno_supports_slic_watershed_and_neural_methods(tmp_path):
    image_path = tmp_path / "demo.png"
    _write_demo_image(image_path)

    cases = [
        sweep.SweepCase(
            method="slic",
            label="slic_demo",
            params={"n_segments": 16, "compactness": 5.0, "sigma": 0.0},
            spanno_path=str(tmp_path / "slic.spanno.json.gz"),
            run_dir=str(tmp_path / "run_slic"),
        ),
        sweep.SweepCase(
            method="watershed",
            label="watershed_demo",
            params={"ws_compactness": 0.001, "ws_components": 16},
            spanno_path=str(tmp_path / "watershed.spanno.json.gz"),
            run_dir=str(tmp_path / "run_ws"),
        ),
        sweep.SweepCase(
            method="sin",
            label="sin_demo",
            params={
                "method_config": json.dumps({"nspix": 16, "fdim": 8, "interp_steps": 1})
            },
            spanno_path=str(tmp_path / "sin.spanno.json.gz"),
            run_dir=str(tmp_path / "run_sin"),
        ),
    ]

    for case in cases:
        count, _ = sweep.ensure_spanno(case, image_path=str(image_path), overwrite=True)
        assert count > 0
        assert Path(case.spanno_path).exists()
