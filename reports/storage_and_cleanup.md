# Хранилище и очистка

## Самые тяжелые верхнеуровневые элементы

| Путь                                   | Классификация | Размер   |
| -------------------------------------- | ------------- | -------- |
| artifacts                              | artifacts     | 45.8 GB  |
| .git                                   | core          | 4.2 GB   |
| superpixel_annotator                   | core          | 902.2 MB |
| models                                 | models        | 140.6 MB |
| docs                                   | docs          | 3.0 MB   |
| reports                                | reports       | 1.4 MB   |
| __pycache__                            | core          | 880.7 KB |
| evaluate_interactive_annotation.py     | core          | 103.7 KB |
| tools                                  | tools         | 91.3 KB  |
| lib                                    | core          | 73.5 KB  |
| tests                                  | core          | 62.0 KB  |
| .DS_Store                              | core          | 60.0 KB  |
| evaluate_superpixel_postprocessing.py  | core          | 55.0 KB  |
| render_interactive_annotation_video.py | core          | 35.4 KB  |
| tune_hybrid_conservative.py            | core          | 24.2 KB  |
| sweep_interactive_superpixels.py       | core          | 23.3 KB  |
| compare.py                             | core          | 22.8 KB  |
| train.py                               | core          | 20.0 KB  |
| superpixel_refinement_strategies.py    | core          | 18.8 KB  |
| benchmark_simple_superpixel_methods.py | core          | 14.8 KB  |

## Самые тяжелые каталоги прогонов

| Путь                                                                      | Тип             | Семейство       | Размер   |
| ------------------------------------------------------------------------- | --------------- | --------------- | -------- |
| artifacts/interactive_runs/results_ssn_random_n1000_ds05                  | interactive_run | interactive-run | 5.2 GB   |
| artifacts/interactive_runs/results_ssn_random_n1000_ds05_1                | interactive_run | interactive-run | 4.5 GB   |
| artifacts/case_studies/interactive_repro_train01_ssn                      | interactive_run | case-study      | 4.1 GB   |
| artifacts/interactive_runs/results_slic                                   | interactive_run | interactive-run | 4.0 GB   |
| artifacts/case_studies/ssn_s1_v2_halfsize                                 | interactive_run | case-study      | 2.1 GB   |
| artifacts/case_studies/_two_quarters/top_left/eval                        | interactive_run | case-study      | 1.0 GB   |
| artifacts/case_studies/_two_quarters/center/eval_100                      | interactive_run | case-study      | 720.5 MB |
| artifacts/case_studies/_two_quarters/top_left/eval_100                    | interactive_run | case-study      | 568.6 MB |
| artifacts/interactive_runs/results_ssn_edt_clear                          | interactive_run | interactive-run | 410.5 MB |
| artifacts/interactive_runs/_eval_train01_prop_default_500                 | interactive_run | interactive-run | 318.3 MB |
| artifacts/interactive_runs/results_slic500_fresh300_probe_v3_unlocktarget | interactive_run | interactive-run | 311.5 MB |
| artifacts/interactive_runs/results_slic500_current100_v5                  | interactive_run | interactive-run | 254.9 MB |
| artifacts/interactive_runs/results_slic500_current100_border8             | interactive_run | interactive-run | 253.4 MB |
| artifacts/interactive_runs/results_slic500_fresh300_probe_v1              | interactive_run | interactive-run | 241.2 MB |
| artifacts/case_studies/ssn_s1_v2_halfsize/test_01                         | interactive_run | case-study      | 199.2 MB |
| artifacts/interactive_runs/results_slic500_current70_probe                | interactive_run | interactive-run | 180.9 MB |
| artifacts/interactive_runs/results_slic500_current100_v4                  | interactive_run | interactive-run | 167.7 MB |
| artifacts/interactive_runs/results_slic500_500scribbles_v1                | interactive_run | interactive-run | 136.6 MB |
| artifacts/interactive_runs/results_slic500_current100_border8_innercore   | interactive_run | interactive-run | 131.9 MB |
| artifacts/interactive_runs/results_edt_train01_ssn500_smoke5              | interactive_run | interactive-run | 125.8 MB |

## Кандидаты на ручную проверку

| Путь                                                        | Проблема              |
| ----------------------------------------------------------- | --------------------- |
| artifacts/case_studies/_two_quarters/diff_masks/summary.csv | summary_without_score |

## Практические рекомендации по очистке

- каталоги со статусом `debug` и `smoke` просмотреть первыми, если нужно освободить место;
- каталоги `artifacts/uncategorized/` держать коротким списком и разбирать вручную;
- большие case-study каталоги хранить, только если они нужны для воспроизведения или визуализации;
- precomputed/spanno артефакты можно архивировать отдельно от кода, если требуется облегчить рабочую копию.
