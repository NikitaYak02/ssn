# Семейства прогонов

В отчетах используется следующая интерпретация статусов:

- `main` — основной или содержательный результат.
- `smoke` — быстрый sanity-check.
- `debug` — отладочный запуск.
- `case-study` — демонстрационный или исследовательский сценарий.
- `archive` — полезная, но уже не первичная серия экспериментов.
- `uncategorized` — требует ручной проверки.

## Распределение по семействам

| Семейство       | Количество | Статусы                |
| --------------- | ---------- | ---------------------- |
| case-study      | 35         | case-study:30, smoke:5 |
| debug           | 6          | debug:6                |
| interactive-run | 79         | main:67, smoke:12      |
| postprocessing  | 8          | main:7, smoke:1        |
| refinement      | 165        | archive:165            |
| sweep           | 152        | archive:143, smoke:9   |

## Примеры семейств

| Семейство       | Пример пути                                                                                                                                | Комментарий                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| case-study      | artifacts/case_studies/_quarter_run/eval                                                                                                   | Отдельные экспериментальные сценарии и демонстрации.                                  |
| debug           | artifacts/debug/_tmp_eval_baseline                                                                                                         | Отладочные и временные каталоги, которые не стоит трактовать как основные результаты. |
| interactive-run | artifacts/interactive_runs/_eval_train01_no_prop_10                                                                                        | Одиночные и серийные интерактивные прогоны с `metrics.csv` и часто с `run.log`.       |
| postprocessing  | artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json                                                                       | Сравнение стратегий постобработки и их влияние на baseline.                           |
| refinement      | artifacts/refinement/_refine_slic_s1v2_fullcover_x05/per_pair/slic__compactness_11p0__n_segments_650__sigma_0p42/test_06/run               | Локальное доуточнение параметров на выбранных парах изображений.                      |
| sweep           | artifacts/sweeps/_single_image_input/optuna_felz_x05/runs/felzenszwalb__f_sigma_0p26625236339971226__min_size_50__scale_311p66123625738715 | Серии перебора параметров и компактные smoke-проверки.                                |
