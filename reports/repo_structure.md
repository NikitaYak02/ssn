# Структура репозитория

Репозиторий разделен на стабильные зоны:

- `docs/` — живая документация, включая разделы по оптимизациям и запуску скриптов.
- `reports/` — автоматически собираемые отчеты и машинно-читаемые инвентари.
- `models/` — контрольные точки и веса.
- `artifacts/` — экспериментальные прогоны и вспомогательные данные.
- `tools/` — утилиты инвентаризации, миграции и генерации документации.
- top-level Python entrypoint'ы, `lib/`, `superpixel_annotator/`, `tests/` — рабочая кодовая поверхность проекта.

Ниже показано фактическое содержимое корня после миграции.


| Путь                                   | Тип  | Классификация | Размер   |
| -------------------------------------- | ---- | ------------- | -------- |
| artifacts                              | dir  | artifacts     | 45.8 GB  |
| .git                                   | dir  | core          | 4.2 GB   |
| superpixel_annotator                   | dir  | core          | 902.2 MB |
| models                                 | dir  | models        | 140.6 MB |
| docs                                   | dir  | docs          | 3.0 MB   |
| reports                                | dir  | reports       | 1.4 MB   |
| __pycache__                            | dir  | core          | 880.7 KB |
| evaluate_interactive_annotation.py     | file | core          | 103.7 KB |
| tools                                  | dir  | tools         | 91.3 KB  |
| lib                                    | dir  | core          | 73.5 KB  |
| tests                                  | dir  | core          | 62.0 KB  |
| .DS_Store                              | file | core          | 60.0 KB  |
| evaluate_superpixel_postprocessing.py  | file | core          | 55.0 KB  |
| render_interactive_annotation_video.py | file | core          | 35.4 KB  |
| tune_hybrid_conservative.py            | file | core          | 24.2 KB  |
| sweep_interactive_superpixels.py       | file | core          | 23.3 KB  |
| compare.py                             | file | core          | 22.8 KB  |
| train.py                               | file | core          | 20.0 KB  |
| superpixel_refinement_strategies.py    | file | core          | 18.8 KB  |
| benchmark_simple_superpixel_methods.py | file | core          | 14.8 KB  |
| refine_superpixel_on_pairs.py          | file | core          | 11.1 KB  |
| train_neural_superpixels.py            | file | core          | 10.6 KB  |
| tune_low_confidence_threshold.py       | file | core          | 10.0 KB  |
| optimize_superpixel_params_optuna.py   | file | core          | 9.2 KB   |
| evaluate_ssn_scribble_batch.py         | file | core          | 9.0 KB   |
| precompute_superpixels.py              | file | core          | 8.5 KB   |
| compute_mask_class_percentages.py      | file | core          | 8.2 KB   |
| plot_class_miou.py                     | file | core          | 7.1 KB   |
| profile_minimal.py                     | file | core          | 7.0 KB   |
| profile_one_batch.py                   | file | core          | 6.5 KB   |
| .pytest_cache                          | dir  | core          | 6.0 KB   |
| benchmark_configs.py                   | file | core          | 4.2 KB   |
| inference.py                           | file | core          | 4.0 KB   |
| model.py                               | file | core          | 3.4 KB   |
| README.md                              | file | docs          | 2.6 KB   |
| run_superpixel_sweep.sh                | file | core          | 1.8 KB   |
| requirements.txt                       | file | core          | 660.0 B  |
| .claude                                | dir  | core          | 330.0 B  |
| .gitignore                             | file | core          | 235.0 B  |
