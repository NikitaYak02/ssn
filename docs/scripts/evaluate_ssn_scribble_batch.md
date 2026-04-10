# evaluate_ssn_scribble_batch.py

- Тип сценария: `interactive-eval`
- Прямой запуск: **да**
- Файл: `evaluate_ssn_scribble_batch.py`

## Назначение

Batch-запуск интерактивной оценки SSN на наборе изображений.

## Когда использовать

Когда нужен пакетный интерактивный прогон по нескольким изображениям SSN-конфигурацией.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 evaluate_ssn_scribble_batch.py --images /path/to/images --masks /path/to/masks --output-dir artifacts/interactive_runs/ssn_batch_demo --ssn_weights models/checkpoints/best_model.pth --scribbles 100 --save_every 20
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 evaluate_ssn_scribble_batch.py --images /path/to/images --masks /path/to/masks --output-dir artifacts/interactive_runs/ssn_batch_demo --ssn_weights models/checkpoints/best_model.pth --scribbles 100 --save_every 20
```

### Пакетный прогон с ресайзом и кэшем

```bash
python3 evaluate_ssn_scribble_batch.py --images /path/to/images --masks /path/to/masks --output-dir artifacts/interactive_runs/ssn_batch_halfsize --work-dir artifacts/precomputed/ssn_batch_cache --resize-scale 0.5 --limit 10 --ssn_weights models/checkpoints/best_model.pth
```

## Входы / выходы

- Входы: директория изображений, директория масок.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--images IMAGES` | Directory with test images. |
| `--masks MASKS` | Directory with GT masks. |
| `--output-dir OUTPUT_DIR` | Directory where per-image runs and the summary CSV will be written. |
| `--work-dir WORK_DIR` | Optional directory for cached resized inputs and SSN spanno files. |
| `--resize-scale RESIZE_SCALE` | Resize images and masks by this factor before running evaluation. |
| `--limit LIMIT` | Optional cap on the number of image/mask pairs to process. |
| `--overwrite-spanno` | Recompute cached SSN spanno files even if they already exist. |
| `--scribbles SCRIBBLES` | - |
| `--save_every SAVE_EVERY` | - |
| `--seed SEED` | - |
| `--margin MARGIN` | - |
| `--border_margin BORDER_MARGIN` | - |
| `--no_overlap` | - |
| `--max_no_progress MAX_NO_PROGRESS` | - |
| `--region_selection_cycle REGION_SELECTION_CYCLE` | - |
| `--sensitivity SENSITIVITY` | - |
| `--emb_weights EMB_WEIGHTS` | - |
| `--emb_threshold EMB_THRESHOLD` | - |
| `--num_classes NUM_CLASSES` | - |
| `--method {ssn}` | - |
| `--ssn_weights SSN_WEIGHTS` | Path to SSN checkpoint (.pth). |
| `--ssn_nspix SSN_NSPIX` | - |
| `--ssn_fdim SSN_FDIM` | - |
| `--ssn_niter SSN_NITER` | - |
| `--ssn_color_scale SSN_COLOR_SCALE` | - |
| `--ssn_pos_scale SSN_POS_SCALE` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: evaluate_ssn_scribble_batch.py [-h] --images IMAGES --masks MASKS
                                      --output-dir OUTPUT_DIR
                                      [--work-dir WORK_DIR]
                                      [--resize-scale RESIZE_SCALE]
                                      [--limit LIMIT] [--overwrite-spanno]
                                      [--scribbles SCRIBBLES]
                                      [--save_every SAVE_EVERY] [--seed SEED]
                                      [--margin MARGIN]
                                      [--border_margin BORDER_MARGIN]
                                      [--no_overlap]
                                      [--max_no_progress MAX_NO_PROGRESS]
                                      [--region_selection_cycle REGION_SELECTION_CYCLE]
                                      [--sensitivity SENSITIVITY]
                                      [--emb_weights EMB_WEIGHTS]
                                      [--emb_threshold EMB_THRESHOLD]
                                      [--num_classes NUM_CLASSES]
                                      [--method {ssn}] --ssn_weights
                                      SSN_WEIGHTS [--ssn_nspix SSN_NSPIX]
                                      [--ssn_fdim SSN_FDIM]
                                      [--ssn_niter SSN_NITER]
                                      [--ssn_color_scale SSN_COLOR_SCALE]
                                      [--ssn_pos_scale SSN_POS_SCALE]

Run SSN-backed scribble evaluation on a whole test subset.

options:
  -h, --help            show this help message and exit
  --images IMAGES       Directory with test images.
  --masks MASKS         Directory with GT masks.
  --output-dir OUTPUT_DIR
                        Directory where per-image runs and the summary CSV
                        will be written.
  --work-dir WORK_DIR   Optional directory for cached resized inputs and SSN
                        spanno files.
  --resize-scale RESIZE_SCALE
                        Resize images and masks by this factor before running
                        evaluation.
  --limit LIMIT         Optional cap on the number of image/mask pairs to
                        process.
  --overwrite-spanno    Recompute cached SSN spanno files even if they already
                        exist.
  --scribbles SCRIBBLES
  --save_every SAVE_EVERY
  --seed SEED
  --margin MARGIN
  --border_margin BORDER_MARGIN
  --no_overlap
  --max_no_progress MAX_NO_PROGRESS
  --region_selection_cycle REGION_SELECTION_CYCLE
  --sensitivity SENSITIVITY
  --emb_weights EMB_WEIGHTS
  --emb_threshold EMB_THRESHOLD
  --num_classes NUM_CLASSES
  --method {ssn}
  --ssn_weights SSN_WEIGHTS
                        Path to SSN checkpoint (.pth).
  --ssn_nspix SSN_NSPIX
  --ssn_fdim SSN_FDIM
  --ssn_niter SSN_NITER
  --ssn_color_scale SSN_COLOR_SCALE
  --ssn_pos_scale SSN_POS_SCALE
```
