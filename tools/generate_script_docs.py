#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


NON_RUNNABLE = {
    "model.py",
    "superpixel_refinement_strategies.py",
    "superpixel_annotator/structs.py",
    "superpixel_annotator/vizualize_utils.py",
}

CUSTOM_MINIMAL_COMMANDS = {
    "benchmark_simple_superpixel_methods.py": (
        "python3 benchmark_simple_superpixel_methods.py "
        "--images /path/to/images --masks /path/to/masks "
        "--checkpoint models/checkpoints/S1v2_S2v2_x05.pth "
        "--output-dir artifacts/postprocessing/benchmark_simple "
        "--suite simple --limit 5"
    ),
    "evaluate_interactive_annotation.py": (
        "python3 evaluate_interactive_annotation.py "
        "--image /path/to/image.png --mask /path/to/mask.png "
        "--out artifacts/interactive_runs/demo_single "
        "--method slic --n_segments 800 --compactness 20 "
        "--scribbles 100 --save_every 20"
    ),
    "evaluate_ssn_scribble_batch.py": (
        "python3 evaluate_ssn_scribble_batch.py "
        "--images /path/to/images --masks /path/to/masks "
        "--output-dir artifacts/interactive_runs/ssn_batch_demo "
        "--ssn_weights models/checkpoints/best_model.pth "
        "--scribbles 100 --save_every 20"
    ),
    "evaluate_superpixel_postprocessing.py": (
        "python3 evaluate_superpixel_postprocessing.py "
        "--images /path/to/images --masks /path/to/masks "
        "--checkpoint models/checkpoints/S1v2_S2v2_x05.pth "
        "--output-dir artifacts/postprocessing/sp_postproc_eval "
        "--device auto"
    ),
    "optimize_superpixel_params_optuna.py": (
        "python3 optimize_superpixel_params_optuna.py "
        "--image /path/to/image.png --mask /path/to/mask.png "
        "--output-dir artifacts/sweeps/optuna_demo "
        "--method ssn --ssn-weights models/checkpoints/best_model.pth "
        "--trials 20 --scribbles 100"
    ),
    "precompute_superpixels.py": (
        "python3 precompute_superpixels.py "
        "--img_dir /path/to/images "
        "--out_dir artifacts/precomputed/full_image_spanno "
        "--method slic --n_segments 800 --compactness 20"
    ),
    "refine_superpixel_on_pairs.py": (
        "python3 refine_superpixel_on_pairs.py "
        "--pairs-json artifacts/refinement/selected_pairs_s1_v2_full_cover.json "
        "--output-dir artifacts/refinement/local_refine "
        "--method ssn --ssn-weights models/checkpoints/best_model.pth "
        "--scribbles 100"
    ),
    "render_interactive_annotation_video.py": (
        "python3 render_interactive_annotation_video.py "
        "--input artifacts/case_studies/interactive_repro_train01_ssn "
        "--out artifacts/case_studies/videos/train01.mp4"
    ),
    "run_superpixel_sweep.sh": (
        "bash run_superpixel_sweep.sh "
        "artifacts/case_studies/_quarter_run/input/train_01_q1.jpg "
        "artifacts/case_studies/_quarter_run/input/train_01_q1.png "
        "artifacts/sweeps/default_100 "
        "models/checkpoints/best_model.pth"
    ),
    "sweep_interactive_superpixels.py": (
        "python3 sweep_interactive_superpixels.py "
        "--image artifacts/case_studies/_quarter_run/input/train_01_q1.jpg "
        "--mask artifacts/case_studies/_quarter_run/input/train_01_q1.png "
        "--output-dir artifacts/sweeps/train01_100 "
        "--methods felzenszwalb,slic,ssn "
        "--ssn-weights models/checkpoints/best_model.pth "
        "--scribbles 100"
    ),
    "train_neural_superpixels.py": (
        "python3 train_neural_superpixels.py "
        "--img_dir /path/to/images --mask_dir /path/to/masks "
        "--out_dir artifacts/training/neural_superpixels "
        "--method deep_slic --train_iter 5000"
    ),
    "tune_hybrid_conservative.py": (
        "python3 tune_hybrid_conservative.py "
        "--images /path/to/images --masks /path/to/masks "
        "--checkpoint models/checkpoints/S1v2_S2v2_x05.pth "
        "--output-dir artifacts/postprocessing/hybrid_tune "
        "--limit 5 --no-noise"
    ),
    "tune_low_confidence_threshold.py": (
        "python3 tune_low_confidence_threshold.py "
        "--cache-dir artifacts/postprocessing/hybrid_tune/logits_cache "
        "--output-dir artifacts/postprocessing/low_conf_tune "
        "--threshold-start 0.5 --threshold-stop 0.9 --threshold-step 0.05"
    ),
}

MANUAL_SUMMARY = {
    "train.py": "Обучение базовой SSN-модели сегментации.",
    "train_neural_superpixels.py": "Обучение или fine-tuning нейросетевых superpixel-методов на целевом домене.",
    "inference.py": "Инференс модели на одном изображении или наборе изображений.",
    "compare.py": "Сравнение SSN и SLIC по метрикам и визуализациям.",
    "benchmark_configs.py": "Сравнение разных конфигураций обучения или инференса.",
    "benchmark_simple_superpixel_methods.py": "Бенчмарк baseline и superpixel-постобработки на наборе изображений.",
    "compute_mask_class_percentages.py": "Подсчет распределения классов по маскам датасета.",
    "evaluate_interactive_annotation.py": "Оценка интерактивной аннотации со штрихами и подробными метриками.",
    "evaluate_ssn_scribble_batch.py": "Batch-запуск интерактивной оценки SSN на наборе изображений.",
    "evaluate_superpixel_postprocessing.py": "Оценка влияния superpixel-постобработки на segmentation baseline.",
    "optimize_superpixel_params_optuna.py": "Optuna-оптимизация параметров superpixel-метода.",
    "plot_class_miou.py": "Построение графиков per-class IoU и mIoU по metrics.csv.",
    "precompute_superpixels.py": "Предварительный расчет superpixel-аннотаций в формате spanno.",
    "profile_minimal.py": "Минимальный профилинг обучения без накладных расходов DataLoader workers.",
    "profile_one_batch.py": "Профилинг одного батча с полным тренировочным пайплайном.",
    "refine_superpixel_on_pairs.py": "Локальное доуточнение параметров superpixel-метода на выбранных парах.",
    "render_interactive_annotation_video.py": "Сборка MP4-видео из кадров интерактивной разметки.",
    "run_superpixel_sweep.sh": "Shell-обертка для запуска sweep_interactive_superpixels.py с дефолтными путями.",
    "superpixel_annotator/test_struct_biggest.py": "Legacy-инструмент интерактивной аннотации с упором на крупные структуры.",
    "superpixel_annotator/test_struct_res.py": "Legacy-инструмент интерактивной аннотации и просмотра результата.",
    "superpixel_annotator/tk_service.py": "Запуск Tk-сервиса или GUI-слоя для legacy annotator.",
    "superpixel_annotator/viz_spanno_annotations.py": "Визуализация state/spanno-аннотаций поверх изображения.",
    "sweep_interactive_superpixels.py": "Параллельный перебор параметров superpixel-методов на одном кейсе.",
    "tune_hybrid_conservative.py": "Подбор консервативных параметров hybrid postprocessing.",
    "tune_low_confidence_threshold.py": "Подбор порогов low-confidence postprocessing.",
}

SCENARIO_TYPE = {
    "train.py": "training",
    "train_neural_superpixels.py": "training",
    "inference.py": "inference",
    "compare.py": "comparison",
    "benchmark_configs.py": "benchmark",
    "benchmark_simple_superpixel_methods.py": "benchmark",
    "compute_mask_class_percentages.py": "dataset-analysis",
    "evaluate_interactive_annotation.py": "interactive-eval",
    "evaluate_ssn_scribble_batch.py": "interactive-eval",
    "evaluate_superpixel_postprocessing.py": "postprocessing-eval",
    "optimize_superpixel_params_optuna.py": "optimization",
    "plot_class_miou.py": "visualization",
    "precompute_superpixels.py": "precompute",
    "profile_minimal.py": "profiling",
    "profile_one_batch.py": "profiling",
    "refine_superpixel_on_pairs.py": "refinement",
    "render_interactive_annotation_video.py": "visualization",
    "run_superpixel_sweep.sh": "shell-wrapper",
    "superpixel_annotator/test_struct_biggest.py": "legacy-cli",
    "superpixel_annotator/test_struct_res.py": "legacy-cli",
    "superpixel_annotator/tk_service.py": "legacy-gui",
    "superpixel_annotator/viz_spanno_annotations.py": "visualization",
    "sweep_interactive_superpixels.py": "sweep",
    "tune_hybrid_conservative.py": "optimization",
    "tune_low_confidence_threshold.py": "optimization",
    "model.py": "library",
    "superpixel_refinement_strategies.py": "library",
    "superpixel_annotator/structs.py": "library",
    "superpixel_annotator/vizualize_utils.py": "library",
}

WHEN_USE = {
    "train.py": "Когда нужно обучить или дообучить основную SSN-модель на датасете сегментации.",
    "train_neural_superpixels.py": "Когда нужно обучить отдельный neural superpixel backend или адаптировать его под целевой домен.",
    "inference.py": "Когда нужно быстро получить предсказания модели без запуска обучения или длинных экспериментов.",
    "compare.py": "Когда нужно сравнить SSN и SLIC на одном и том же наборе изображений.",
    "benchmark_configs.py": "Когда нужно быстро сравнить несколько конфигураций и замерить скорость или качество.",
    "benchmark_simple_superpixel_methods.py": "Когда нужно сравнить baseline и несколько стратегий superpixel aggregation/postprocessing.",
    "compute_mask_class_percentages.py": "Когда нужно понять дисбаланс классов в масках и подготовить сводную статистику по датасету.",
    "evaluate_interactive_annotation.py": "Когда нужно измерить, как быстро и качественно растет разметка при добавлении штрихов.",
    "evaluate_ssn_scribble_batch.py": "Когда нужен пакетный интерактивный прогон по нескольким изображениям SSN-конфигурацией.",
    "evaluate_superpixel_postprocessing.py": "Когда нужно оценить улучшает ли postprocessing baseline или вносит регрессии.",
    "optimize_superpixel_params_optuna.py": "Когда нужен автоматический поиск параметров вместо ручного sweep.",
    "plot_class_miou.py": "Когда нужно визуализировать trajectory метрик после уже завершенного прогона.",
    "precompute_superpixels.py": "Когда хочется заранее закэшировать `spanno` и ускорить повторные интерактивные прогоны.",
    "profile_minimal.py": "Когда нужно понять базовую стоимость одного training step без шума от workers.",
    "profile_one_batch.py": "Когда нужно увидеть узкие места одного полного прохода через pipeline.",
    "refine_superpixel_on_pairs.py": "Когда после грубого sweep нужно локально доуточнить хорошие конфигурации на выбранных парах.",
    "render_interactive_annotation_video.py": "Когда нужно превратить state/frame артефакты в видео для демонстрации или анализа.",
    "run_superpixel_sweep.sh": "Когда нужен быстрый reproducible запуск sweep со встроенными дефолтами.",
    "superpixel_annotator/test_struct_biggest.py": "Когда нужен legacy CLI для ручной/полуручной аннотации и исследования superpixel-структур.",
    "superpixel_annotator/test_struct_res.py": "Когда нужен legacy CLI для аннотации и анализа результата на одном изображении.",
    "superpixel_annotator/tk_service.py": "Когда нужно поднять legacy GUI/Tk-сервис, а не пакетный скрипт.",
    "superpixel_annotator/viz_spanno_annotations.py": "Когда нужно отрисовать state/spanno поверх картинки и быстро проверить артефакт.",
    "sweep_interactive_superpixels.py": "Когда нужно параллельно перебрать параметры `slic`, `felzenszwalb` и `ssn` на одном кейсе.",
    "tune_hybrid_conservative.py": "Когда нужно подобрать безопасные параметры hybrid-conservative postprocessing.",
    "tune_low_confidence_threshold.py": "Когда нужно подобрать пороги low-confidence overwrite стратегии.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate script usage documentation and machine-readable catalog.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--docs-dir", default="docs/scripts", help="Directory for generated script docs.")
    parser.add_argument(
        "--catalog-dir",
        default="reports/generated",
        help="Directory for machine-readable script catalog outputs.",
    )
    return parser


def read_docstring(path: Path) -> str | None:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    doc = ast.get_docstring(module)
    if not doc:
        return None
    first = doc.strip().split("\n\n", 1)[0].strip()
    return first or None


def collect_scripts(root: Path) -> list[str]:
    scripts: list[str] = []
    for path in sorted(root.glob("*.py")):
        scripts.append(path.name)
    for path in sorted(root.glob("*.sh")):
        scripts.append(path.name)
    for path in sorted((root / "superpixel_annotator").glob("*.py")):
        scripts.append(str(path.relative_to(root)))
    return scripts


def help_runner(root: Path, script: str) -> list[str] | None:
    relative = Path(script)
    if script in NON_RUNNABLE:
        return None
    if relative.suffix == ".sh":
        return None
    python_bin = root / "superpixel_annotator" / "superpixel_annotator_venv" / "bin" / "python"
    if python_bin.exists():
        return [str(python_bin), script, "--help"]
    return ["python3", script, "--help"]


def run_help(root: Path, script: str) -> tuple[str | None, str | None]:
    command = help_runner(root, script)
    if not command:
        return None, None
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    text = (result.stdout or "").strip()
    if result.returncode == 0 and text:
        return text, None
    return None, (result.stderr or result.stdout or f"exit={result.returncode}").strip()


def parse_help_options(help_text: str) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_options = False
    for raw_line in help_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in {"options:", "optional arguments:"}:
            in_options = True
            continue
        if not in_options:
            continue
        if stripped.startswith("-"):
            if current:
                options.append(current)
            parts = stripped.split("  ", 1)
            option_name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            current = {"option": option_name, "description": description}
            continue
        if current and line.startswith(" "):
            if current["description"]:
                current["description"] += " "
            current["description"] += stripped
    if current:
        options.append(current)
    return options


def infer_inputs_outputs(script: str, options: list[str]) -> tuple[str, str]:
    if script == "precompute_superpixels.py":
        return (
            "директория изображений, опционально checkpoint для neural/ssn methods",
            "директория результатов, `.spanno.json.gz` файлы",
        )
    option_set = set(options)
    inputs: list[str] = []
    outputs: list[str] = []
    if {"--img_dir", "--mask_dir"} <= option_set:
        inputs.append("директория изображений")
        inputs.append("директория масок")
    elif "--img_dir" in option_set:
        inputs.append("директория изображений")
    if {"--images", "--masks"} <= option_set:
        inputs.append("директория изображений")
        inputs.append("директория масок")
    if {"--image", "--mask"} <= option_set:
        inputs.append("одно изображение")
        inputs.append("одна GT-маска")
    if "--input" in option_set:
        inputs.append("директория результата или одиночный run-каталог")
    if "--cache-dir" in option_set or "--cache_dir" in option_set:
        inputs.append("готовый on-disk cache")
    if "--masks" in option_set and "директория масок" not in inputs:
        inputs.append("директория масок")
    if "--checkpoint" in option_set or "--weights" in option_set or "--ssn-weights" in option_set:
        inputs.append("checkpoint / веса модели")
    if "--weight" in option_set:
        inputs.append("checkpoint / веса модели")
    if "--metrics" in option_set:
        inputs.append("metrics.csv завершенного прогона")
    if "--pairs-json" in option_set:
        inputs.append("JSON со списком пар image/mask")
    if "--spanno" in option_set:
        inputs.append("готовый spanno/state артефакт")
    if not inputs:
        inputs.append("аргументы командной строки и входные пути по конкретному сценарию")

    if "--out" in option_set or "--output-dir" in option_set or "--out_dir" in option_set:
        outputs.append("директория результатов")
    if "--output" in option_set:
        outputs.append("файл результата")
    if "--csv" in option_set:
        outputs.append("CSV-таблица")
    if "--vis_dir" in option_set:
        outputs.append("директория визуализаций")
    if script.endswith("plot_class_miou.py"):
        outputs.append("PNG-графики")
    if script.endswith("render_interactive_annotation_video.py"):
        outputs.append("MP4-видео")
    if script.endswith("precompute_superpixels.py"):
        outputs.append("`.spanno.json.gz` файлы")
    if script.endswith("compute_mask_class_percentages.py"):
        outputs.append("сводная статистика по классам")
    if not outputs:
        outputs.append("stdout, логи и/или артефакты в путях, переданных через CLI")
    return ", ".join(inputs), ", ".join(outputs)


def option_alias(option_names: list[str], *aliases: str) -> str | None:
    option_set = set(option_names)
    for alias in aliases:
        if alias in option_set:
            return alias
    return None


def generic_command(script: str, option_names: list[str]) -> str:
    cmd = f"python3 {script}"
    if script == "run_superpixel_sweep.sh":
        return CUSTOM_MINIMAL_COMMANDS[script]

    image_opt = option_alias(option_names, "--image", "--img", "--img_path")
    mask_opt = option_alias(option_names, "--mask", "--mask_path")
    images_opt = option_alias(option_names, "--images", "--img_dir")
    masks_opt = option_alias(option_names, "--masks", "--mask_dir")
    out_opt = option_alias(option_names, "--out", "--output-dir", "--out_dir")
    checkpoint_opt = option_alias(
        option_names,
        "--checkpoint",
        "--weights",
        "--weight",
        "--ssn-weights",
        "--ssn_weights",
    )
    method_opt = option_alias(option_names, "--method")

    if images_opt:
        cmd += f" {images_opt} /path/to/images"
    if masks_opt:
        cmd += f" {masks_opt} /path/to/masks"
    if image_opt:
        cmd += f" {image_opt} /path/to/image.png"
    if mask_opt:
        cmd += f" {mask_opt} /path/to/mask.png"
    if out_opt:
        if script == "plot_class_miou.py":
            cmd += f" {out_opt} artifacts/interactive_runs/demo/class_miou.png"
        else:
            cmd += f" {out_opt} artifacts/demo_output"
    if "--output" in option_names:
        cmd += " --output artifacts/demo_output/result.png"
    if "--csv" in option_names:
        cmd += " --csv reports/generated/comparison.csv"
    if "--vis_dir" in option_names:
        cmd += " --vis_dir artifacts/case_studies/compare_vis"
    if "--cache-dir" in option_names:
        cmd += " --cache-dir artifacts/postprocessing/hybrid_tune/logits_cache"
    if checkpoint_opt:
        checkpoint_path = (
            "models/checkpoints/S1v2_S2v2_x05.pth"
            if checkpoint_opt in {"--checkpoint", "--weight"}
            else "models/checkpoints/best_model.pth"
        )
        cmd += f" {checkpoint_opt} {checkpoint_path}"
    if method_opt and script not in {"train_neural_superpixels.py", "superpixel_annotator/tk_service.py"}:
        cmd += f" {method_opt} ssn"
    if "--pairs-json" in option_names:
        cmd += " --pairs-json artifacts/refinement/selected_pairs_s1_v2_full_cover.json"
    if "--metrics" in option_names:
        cmd += " --metrics artifacts/interactive_runs/demo/metrics.csv"
    if "--input" in option_names and script == "render_interactive_annotation_video.py":
        cmd += " --input artifacts/case_studies/interactive_repro_train01_ssn"
    if "--trials" in option_names:
        cmd += " --trials 20"
    if "--methods" in option_names:
        cmd += " --methods felzenszwalb,slic,ssn"
    if "--scribbles" in option_names:
        cmd += " --scribbles 100"
    if "--train_iter" in option_names:
        cmd += " --train_iter 500000"
    if "--method" in option_names and script == "train_neural_superpixels.py":
        cmd += " --method deep_slic"
    if "--slic-n-segments" in option_names:
        cmd += " --slic-n-segments 650,750"
    if "--slic-compactnesses" in option_names:
        cmd += " --slic-compactnesses 9.5,11.0"
    if "--slic-sigmas" in option_names:
        cmd += " --slic-sigmas 0.42,0.52"
    return cmd


def minimal_command(script: str, option_names: list[str]) -> str:
    if script in CUSTOM_MINIMAL_COMMANDS:
        return CUSTOM_MINIMAL_COMMANDS[script]
    cmd = generic_command(script, option_names)
    return cmd


def typical_scenarios(script: str, option_names: list[str]) -> list[tuple[str, str]]:
    if script in NON_RUNNABLE:
        return [("Библиотечный модуль", "Запускайте его только через импорт из других скриптов проекта.")]
    if script == "evaluate_interactive_annotation.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "SSN с заранее подготовленным spanno",
                "python3 evaluate_interactive_annotation.py "
                "--image /path/to/image.png --mask /path/to/mask.png "
                "--spanno /path/to/cache.spanno.json.gz "
                "--out artifacts/interactive_runs/demo_ssn "
                "--method ssn --ssn_weights models/checkpoints/best_model.pth "
                "--scribbles 100 --save_every 20",
            ),
        ]
    if script == "evaluate_ssn_scribble_batch.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Пакетный прогон с ресайзом и кэшем",
                "python3 evaluate_ssn_scribble_batch.py "
                "--images /path/to/images --masks /path/to/masks "
                "--output-dir artifacts/interactive_runs/ssn_batch_halfsize "
                "--work-dir artifacts/precomputed/ssn_batch_cache "
                "--resize-scale 0.5 --limit 10 "
                "--ssn_weights models/checkpoints/best_model.pth",
            ),
        ]
    if script == "evaluate_superpixel_postprocessing.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Без деградации checkpoint и с SLIC",
                "python3 evaluate_superpixel_postprocessing.py "
                "--images /path/to/images --masks /path/to/masks "
                "--checkpoint models/checkpoints/S1v2_S2v2_x05.pth "
                "--output-dir artifacts/postprocessing/sp_postproc_eval_slic "
                "--sp-method slic --n-segments 800 --compactness 20 --no-noise",
            ),
        ]
    if script == "optimize_superpixel_params_optuna.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Оптимизация только для SLIC",
                "python3 optimize_superpixel_params_optuna.py "
                "--image /path/to/image.png --mask /path/to/mask.png "
                "--output-dir artifacts/sweeps/optuna_slic_demo "
                "--method slic --trials 20 --scribbles 100",
            ),
        ]
    if script == "refine_superpixel_on_pairs.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Локальный refinement для SLIC",
                "python3 refine_superpixel_on_pairs.py "
                "--pairs-json artifacts/refinement/selected_pairs_s1_v2_full_cover.json "
                "--output-dir artifacts/refinement/local_refine_slic "
                "--method slic --scribbles 100 "
                "--slic-n-segments 650,750 --slic-compactnesses 9.5,11.0 "
                "--slic-sigmas 0.42,0.52",
            ),
        ]
    if script == "sweep_interactive_superpixels.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Только baseline-методы без SSN",
                "python3 sweep_interactive_superpixels.py "
                "--image artifacts/case_studies/_quarter_run/input/train_01_q1.jpg "
                "--mask artifacts/case_studies/_quarter_run/input/train_01_q1.png "
                "--output-dir artifacts/sweeps/train01_baselines "
                "--methods felzenszwalb,slic --scribbles 100",
            ),
        ]
    if script == "benchmark_simple_superpixel_methods.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Safe-suite без деградации checkpoint",
                "python3 benchmark_simple_superpixel_methods.py "
                "--images /path/to/images --masks /path/to/masks "
                "--checkpoint models/checkpoints/S1v2_S2v2_x05.pth "
                "--output-dir artifacts/postprocessing/benchmark_safe "
                "--suite safe --limit 5 --no-noise",
            ),
        ]
    if script == "precompute_superpixels.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "SSN-кэш для повторных интерактивных прогонов",
                "python3 precompute_superpixels.py "
                "--img_dir /path/to/images "
                "--out_dir artifacts/precomputed/ssn_cache "
                "--method ssn --ssn_weights models/checkpoints/best_model.pth "
                "--ssn_nspix 100",
            ),
        ]
    if script == "tune_hybrid_conservative.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Переиспользование уже собранного cache",
                "python3 tune_hybrid_conservative.py "
                "--images /path/to/images --masks /path/to/masks "
                "--checkpoint models/checkpoints/S1v2_S2v2_x05.pth "
                "--output-dir artifacts/postprocessing/hybrid_tune "
                "--cache-dir artifacts/postprocessing/hybrid_tune/logits_cache "
                "--limit 5 --no-noise",
            ),
        ]
    if script == "tune_low_confidence_threshold.py":
        return [
            ("Минимальный запуск", minimal_command(script, option_names)),
            (
                "Явный список threshold-ов",
                "python3 tune_low_confidence_threshold.py "
                "--cache-dir artifacts/postprocessing/hybrid_tune/logits_cache "
                "--output-dir artifacts/postprocessing/low_conf_tune_manual "
                "--thresholds 0.55,0.60,0.65,0.70,0.75",
            ),
        ]
    scenarios = [
        ("Минимальный запуск", minimal_command(script, option_names)),
    ]
    if script in {"optimize_superpixel_params_optuna.py"}:
        base = minimal_command(script, option_names)
        scenarios.append(
            (
                "Повторный запуск с кэшированными superpixels",
                (
                    base.replace("--method ssn --ssn-weights", "--spanno /path/to/cache.spanno.json.gz --method ssn --ssn-weights")
                    .replace("--method ssn --ssn_weights", "--spanno /path/to/cache.spanno.json.gz --method ssn --ssn_weights")
                ),
            )
        )
    if script == "render_interactive_annotation_video.py":
        scenarios.append(
            (
                "Пакетная сборка видео",
                "python3 render_interactive_annotation_video.py --input artifacts/case_studies/ssn_s1_v2_halfsize --out artifacts/case_studies/videos",
            )
        )
    if script == "precompute_superpixels.py":
        scenarios.append(
            (
                "Переиспользуемый кэш для интерактивной оценки",
                "python3 precompute_superpixels.py --img_dir /path/to/images --out_dir artifacts/precomputed/full_image_spanno --method slic --n_segments 800 --compactness 20",
            )
        )
    if script in {"tune_hybrid_conservative.py", "tune_low_confidence_threshold.py", "benchmark_simple_superpixel_methods.py"}:
        scenarios.append(
            (
                "Сравнение на тестовой выборке",
                minimal_command(script, option_names),
            )
        )
    return scenarios[:3]


def common_issues(script: str) -> list[str]:
    issues = [
        "Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.",
    ]
    checkpoint_sensitive = {
        "benchmark_simple_superpixel_methods.py",
        "compare.py",
        "evaluate_interactive_annotation.py",
        "evaluate_ssn_scribble_batch.py",
        "evaluate_superpixel_postprocessing.py",
        "inference.py",
        "optimize_superpixel_params_optuna.py",
        "precompute_superpixels.py",
        "refine_superpixel_on_pairs.py",
        "run_superpixel_sweep.sh",
        "sweep_interactive_superpixels.py",
        "train_neural_superpixels.py",
        "tune_hybrid_conservative.py",
    }
    if script in checkpoint_sensitive:
        issues.append("Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.")
    if script.endswith(".py") and script not in NON_RUNNABLE:
        issues.append("Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.")
    if script in {"evaluate_interactive_annotation.py", "precompute_superpixels.py", "sweep_interactive_superpixels.py", "refine_superpixel_on_pairs.py"}:
        issues.append("Для `spanno`-артефактов размер изображения и размер кэша должны совпадать, иначе будут ошибки совместимости.")
    if script == "superpixel_annotator/tk_service.py":
        issues.append("Этот скрипт ориентирован на GUI/Tk, поэтому может не стартовать в headless-окружении без дисплея.")
    if script == "tune_low_confidence_threshold.py":
        issues.append("Этот скрипт не строит cache сам: сначала подготовьте его через `tune_hybrid_conservative.py`.")
    return issues


def output_location_hint(script: str) -> str:
    if script in NON_RUNNABLE:
        return "Прямых файлов результата нет: модуль используется через импорт."
    if script == "run_superpixel_sweep.sh":
        return "Результаты лежат в директории, переданной третьим аргументом shell-скрипта."
    if script == "plot_class_miou.py":
        return "Ищите PNG-графики рядом с путем, указанным в `--out`."
    if script == "render_interactive_annotation_video.py":
        return "Ищите MP4 по пути из `--out`; в batch-режиме это директория с несколькими видео."
    if script == "precompute_superpixels.py":
        return "Ищите `.spanno.json.gz` файлы внутри `--out_dir`."
    return "Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария."


def help_block(help_text: str | None, error_text: str | None) -> str:
    if help_text:
        return f"```text\n{help_text}\n```"
    if error_text:
        return f"```text\nНе удалось автоматически получить --help:\n{error_text}\n```"
    return "```text\nУ этого скрипта нет стандартного `--help`, или он рассчитан на импорт/GUI.\n```"


def format_option_rows(options: list[dict[str, str]]) -> str:
    if not options:
        return "_Ключевые опции автоматически не извлечены; используйте `Raw --help` ниже._"
    rows = ["| Опция | Описание |", "| --- | --- |"]
    for item in options[:80]:
        option = item["option"].replace("|", "\\|")
        description = item["description"].replace("|", "\\|") or "-"
        rows.append(f"| `{option}` | {description} |")
    return "\n".join(rows)


def script_doc_path(script: str) -> str:
    return script.replace("/", "__").replace(".py", "").replace(".sh", "") + ".md"


def build_script_page(root: Path, script: str, summary: str, when_use: str, help_text: str | None, error_text: str | None) -> tuple[str, dict[str, Any]]:
    path = root / script
    runnable = script not in NON_RUNNABLE
    options = parse_help_options(help_text or "")
    option_names = [item["option"].split(",")[-1].strip().split()[0] for item in options if "--" in item["option"]]
    inputs_text, outputs_text = infer_inputs_outputs(script, option_names)
    scenarios = typical_scenarios(script, option_names)
    page_lines = [
        f"# {script}",
        "",
        f"- Тип сценария: `{SCENARIO_TYPE.get(script, 'misc')}`",
        f"- Прямой запуск: **{'да' if runnable else 'нет'}**",
        f"- Файл: `{script}`",
        "",
        "## Назначение",
        "",
        summary,
        "",
        "## Когда использовать",
        "",
        when_use,
        "",
        "## Обязательные зависимости",
        "",
        "- Python 3.12+ и зависимости проекта из `requirements.txt`.",
        "- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.",
        "- Подробная памятка по окружениям: [run_environments.md](run_environments.md).",
        "",
        "## Минимальный запуск",
        "",
        "```bash",
        minimal_command(script, option_names),
        "```",
        "",
        "## Типовые сценарии",
        "",
    ]
    for title, command in scenarios:
        page_lines.extend([f"### {title}", "", "```bash", command, "```", ""])
    page_lines.extend(
        [
            "## Входы / выходы",
            "",
            f"- Входы: {inputs_text}.",
            f"- Выходы: {outputs_text}.",
            "",
            "## Где искать результаты",
            "",
            output_location_hint(script),
            "",
            "## Ключевые аргументы",
            "",
            format_option_rows(options),
            "",
            "## Типичные ошибки",
            "",
        ]
    )
    for issue in common_issues(script):
        page_lines.append(f"- {issue}")
    page_lines.extend(
        [
            "",
            "## Raw `--help`",
            "",
            help_block(help_text, error_text),
        ]
    )

    catalog_entry = {
        "script": script,
        "doc_path": f"docs/scripts/{script_doc_path(script)}",
        "runnable": runnable,
        "scenario_type": SCENARIO_TYPE.get(script, "misc"),
        "summary": summary,
        "when_use": when_use,
        "inputs": inputs_text,
        "outputs": outputs_text,
        "has_help": bool(help_text),
    }
    return "\n".join(page_lines), catalog_entry


def build_index(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Скрипты проекта",
        "",
        "Этот раздел покрывает все реальные entrypoint-скрипты репозитория, а библиотечные модули отдельно помечает как non-runnable.",
        "",
        "## Матрица скриптов",
        "",
        "| Скрипт | Прямой запуск | Тип сценария | Входы | Выходы | Документация |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['script']}` | {'да' if entry['runnable'] else 'нет'} | `{entry['scenario_type']}` | {entry['inputs']} | {entry['outputs']} | [{Path(entry['doc_path']).name}]({Path(entry['doc_path']).name}) |"
        )
    lines.extend(
        [
            "",
            "## Дополнительно",
            "",
            "- [run_environments.md](run_environments.md) — как готовить окружение и какой Python использовать.",
        ]
    )
    return "\n".join(lines)


def build_run_environments_doc() -> str:
    return """# Окружения запуска

## Базовый вариант

- Для top-level скриптов используйте Python 3.12+.
- Создайте отдельное окружение проекта и установите зависимости из `requirements.txt`.
- Если нужен самый простой reproducible вариант для evaluation/legacy-сценариев, используйте bundled venv: `superpixel_annotator/superpixel_annotator_venv/bin/python`.

## Рекомендуемый порядок

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Когда использовать bundled venv

- Когда нужно быстро запустить evaluation/visualization/legacy CLI без отдельной ручной настройки.
- Когда нужно получить реальный `--help` и воспроизводимый parser для существующих скриптов.
- Когда проектовый `.venv` еще не подготовлен.

## Скрипты, которым обычно нужен checkpoint

- `train_neural_superpixels.py` — если идет fine-tuning через `--weights`
- `inference.py`
- `compare.py`
- `benchmark_simple_superpixel_methods.py`
- `evaluate_interactive_annotation.py` для `ssn`
- `evaluate_ssn_scribble_batch.py`
- `evaluate_superpixel_postprocessing.py`
- `optimize_superpixel_params_optuna.py` для `ssn`
- `sweep_interactive_superpixels.py` для `ssn`
- `tune_hybrid_conservative.py`
- `tune_low_confidence_threshold.py`

## Скрипты, которым нужен dataset

- `train.py`
- `train_neural_superpixels.py`
- `compare.py`
- `benchmark_configs.py`
- `benchmark_simple_superpixel_methods.py`
- `compute_mask_class_percentages.py`
- `evaluate_interactive_annotation.py` в batch-режиме
- `evaluate_ssn_scribble_batch.py`
- `evaluate_superpixel_postprocessing.py`
- `precompute_superpixels.py`
- `profile_minimal.py`
- `profile_one_batch.py`
- `refine_superpixel_on_pairs.py`
- `sweep_interactive_superpixels.py`
- `tune_hybrid_conservative.py`
- `tune_low_confidence_threshold.py`

## Скрипты, которым может понадобиться petroscope

- `benchmark_simple_superpixel_methods.py`
- `compute_mask_class_percentages.py`
- `evaluate_superpixel_postprocessing.py`
- `tune_hybrid_conservative.py`
- `tune_low_confidence_threshold.py`

## CPU / MPS / CUDA

- `train.py`, `train_neural_superpixels.py`, `inference.py`, `compare.py`, `evaluate_*`, `benchmark_*`, `tune_*` чувствительны к устройству и обычно выигрывают от `cuda` или `mps`.
- `profile_*` особенно полезны именно на том устройстве, на котором планируется работа.
- `plot_class_miou.py`, `render_interactive_annotation_video.py`, `compute_mask_class_percentages.py` и большая часть простых утилит могут запускаться на CPU.
- `superpixel_annotator/tk_service.py` зависит не столько от устройства, сколько от наличия GUI/дисплея.

## Куда складывать новые результаты

- Новые training/fine-tuning результаты удобно складывать в `artifacts/training/`.
- Интерактивные прогоны складывайте в `artifacts/interactive_runs/`.
- Sweep/refinement лучше держать в `artifacts/sweeps/` и `artifacts/refinement/`.
- Постобработку и сравнения держите в `artifacts/postprocessing/`.
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.repo_root).resolve()
    docs_dir = (root / args.docs_dir).resolve()
    catalog_dir = (root / args.catalog_dir).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for script in collect_scripts(root):
        summary = MANUAL_SUMMARY.get(script) or read_docstring(root / script) or "Описание пока не заполнено."
        when_use = WHEN_USE.get(script, "Используйте этот файл в соответствии с его parser'ом и назначением в кодовой базе.")
        help_text, error_text = run_help(root, script)
        page_text, catalog_entry = build_script_page(root, script, summary, when_use, help_text, error_text)
        write_text(docs_dir / script_doc_path(script), page_text)
        entries.append(catalog_entry)

    entries = sorted(entries, key=lambda item: item["script"])
    write_text(docs_dir / "index.md", build_index(entries))
    write_text(docs_dir / "run_environments.md", build_run_environments_doc())
    write_text(catalog_dir / "scripts_catalog.json", json.dumps({"scripts": entries}, indent=2, ensure_ascii=False))
    write_csv(catalog_dir / "scripts_catalog.csv", entries)
    print(f"Script docs written to: {docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
