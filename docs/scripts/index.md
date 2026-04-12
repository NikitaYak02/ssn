# Скрипты проекта

Этот раздел покрывает все реальные entrypoint-скрипты репозитория, а библиотечные модули отдельно помечает как non-runnable.

## Матрица скриптов

| Скрипт | Прямой запуск | Тип сценария | Входы | Выходы | Документация |
| --- | --- | --- | --- | --- | --- |
| `benchmark_configs.py` | да | `benchmark` | директория изображений, директория масок | stdout, логи и/или артефакты в путях, переданных через CLI | [benchmark_configs.md](benchmark_configs.md) |
| `benchmark_simple_superpixel_methods.py` | да | `benchmark` | директория изображений, директория масок, checkpoint / веса модели | директория результатов | [benchmark_simple_superpixel_methods.md](benchmark_simple_superpixel_methods.md) |
| `compare.py` | да | `comparison` | директория изображений, директория масок, checkpoint / веса модели | CSV-таблица, директория визуализаций | [compare.md](compare.md) |
| `compute_mask_class_percentages.py` | да | `dataset-analysis` | директория масок | сводная статистика по классам | [compute_mask_class_percentages.md](compute_mask_class_percentages.md) |
| `evaluate_interactive_annotation.py` | да | `interactive-eval` | директория изображений, директория масок, одно изображение, одна GT-маска, checkpoint / веса модели, готовый spanno/state артефакт | директория результатов | [evaluate_interactive_annotation.md](evaluate_interactive_annotation.md) |
| `evaluate_ssn_scribble_batch.py` | да | `interactive-eval` | директория изображений, директория масок | директория результатов | [evaluate_ssn_scribble_batch.md](evaluate_ssn_scribble_batch.md) |
| `evaluate_superpixel_postprocessing.py` | да | `postprocessing-eval` | директория изображений, директория масок, checkpoint / веса модели | директория результатов | [evaluate_superpixel_postprocessing.md](evaluate_superpixel_postprocessing.md) |
| `inference.py` | да | `inference` | checkpoint / веса модели | файл результата | [inference.md](inference.md) |
| `model.py` | нет | `library` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [model.md](model.md) |
| `optimize_superpixel_params_optuna.py` | да | `optimization` | одно изображение, одна GT-маска, checkpoint / веса модели | директория результатов | [optimize_superpixel_params_optuna.md](optimize_superpixel_params_optuna.md) |
| `plot_class_miou.py` | да | `visualization` | metrics.csv завершенного прогона | директория результатов, PNG-графики | [plot_class_miou.md](plot_class_miou.md) |
| `precompute_superpixels.py` | да | `precompute` | директория изображений, опционально checkpoint для neural/ssn methods | директория результатов, `.spanno.json.gz` файлы | [precompute_superpixels.md](precompute_superpixels.md) |
| `profile_minimal.py` | да | `profiling` | директория изображений, директория масок | stdout, логи и/или артефакты в путях, переданных через CLI | [profile_minimal.md](profile_minimal.md) |
| `profile_one_batch.py` | да | `profiling` | директория изображений, директория масок | stdout, логи и/или артефакты в путях, переданных через CLI | [profile_one_batch.md](profile_one_batch.md) |
| `refine_superpixel_on_pairs.py` | да | `refinement` | checkpoint / веса модели, JSON со списком пар image/mask | директория результатов | [refine_superpixel_on_pairs.md](refine_superpixel_on_pairs.md) |
| `render_interactive_annotation_video.py` | да | `visualization` | директория результата или одиночный run-каталог | директория результатов, MP4-видео | [render_interactive_annotation_video.md](render_interactive_annotation_video.md) |
| `report_superpixel_anything_overlap.py` | да | `report` | опционально путь для markdown-отчета | markdown-отчет | [report_superpixel_anything_overlap.md](report_superpixel_anything_overlap.md) |
| `run_superpixel_sweep.sh` | да | `shell-wrapper` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [run_superpixel_sweep.md](run_superpixel_sweep.md) |
| `superpixel_annotator/structs.py` | нет | `library` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_annotator__structs.md](superpixel_annotator__structs.md) |
| `superpixel_annotator/test_struct_biggest.py` | да | `legacy-cli` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_annotator__test_struct_biggest.md](superpixel_annotator__test_struct_biggest.md) |
| `superpixel_annotator/test_struct_res.py` | да | `legacy-cli` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_annotator__test_struct_res.md](superpixel_annotator__test_struct_res.md) |
| `superpixel_annotator/tk_service.py` | да | `legacy-gui` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_annotator__tk_service.md](superpixel_annotator__tk_service.md) |
| `superpixel_annotator/viz_spanno_annotations.py` | да | `visualization` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_annotator__viz_spanno_annotations.md](superpixel_annotator__viz_spanno_annotations.md) |
| `superpixel_annotator/vizualize_utils.py` | нет | `library` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_annotator__vizualize_utils.md](superpixel_annotator__vizualize_utils.md) |
| `superpixel_refinement_strategies.py` | нет | `library` | аргументы командной строки и входные пути по конкретному сценарию | stdout, логи и/или артефакты в путях, переданных через CLI | [superpixel_refinement_strategies.md](superpixel_refinement_strategies.md) |
| `sweep_interactive_superpixels.py` | да | `sweep` | одно изображение, одна GT-маска, checkpoint / веса модели | директория результатов | [sweep_interactive_superpixels.md](sweep_interactive_superpixels.md) |
| `train.py` | да | `training` | директория изображений, директория масок | директория результатов | [train.md](train.md) |
| `train_external_superpixels.py` | да | `training` | BSD-style root или директории изображений/масок | директория результатов | [train_external_superpixels.md](train_external_superpixels.md) |
| `train_neural_superpixels.py` | да | `training` | директория изображений, директория масок, checkpoint / веса модели | директория результатов | [train_neural_superpixels.md](train_neural_superpixels.md) |
| `tune_hybrid_conservative.py` | да | `optimization` | директория изображений, директория масок, готовый on-disk cache, checkpoint / веса модели | директория результатов | [tune_hybrid_conservative.md](tune_hybrid_conservative.md) |
| `tune_low_confidence_threshold.py` | да | `optimization` | готовый on-disk cache | директория результатов | [tune_low_confidence_threshold.md](tune_low_confidence_threshold.md) |

## Дополнительно

- [run_environments.md](run_environments.md) — как готовить окружение и какой Python использовать.
