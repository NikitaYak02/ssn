# sweep_interactive_superpixels.py

- Тип сценария: `sweep`
- Прямой запуск: **да**
- Файл: `sweep_interactive_superpixels.py`

## Назначение

Параллельный перебор параметров superpixel-методов на одном кейсе.

## Когда использовать

Когда нужно параллельно перебрать параметры `slic`, `felzenszwalb` и `ssn` на одном кейсе.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 sweep_interactive_superpixels.py --image artifacts/case_studies/_quarter_run/input/train_01_q1.jpg --mask artifacts/case_studies/_quarter_run/input/train_01_q1.png --output-dir artifacts/sweeps/train01_100 --methods felzenszwalb,slic,ssn --ssn-weights models/checkpoints/best_model.pth --scribbles 100
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 sweep_interactive_superpixels.py --image artifacts/case_studies/_quarter_run/input/train_01_q1.jpg --mask artifacts/case_studies/_quarter_run/input/train_01_q1.png --output-dir artifacts/sweeps/train01_100 --methods felzenszwalb,slic,ssn --ssn-weights models/checkpoints/best_model.pth --scribbles 100
```

### Только baseline-методы без SSN

```bash
python3 sweep_interactive_superpixels.py --image artifacts/case_studies/_quarter_run/input/train_01_q1.jpg --mask artifacts/case_studies/_quarter_run/input/train_01_q1.png --output-dir artifacts/sweeps/train01_baselines --methods felzenszwalb,slic --scribbles 100
```

## Входы / выходы

- Входы: одно изображение, одна GT-маска, checkpoint / веса модели.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--image IMAGE` | RGB image for interactive evaluation. |
| `--mask MASK` | GT mask with class ids. |
| `--output-dir OUTPUT_DIR` | Directory for sweep outputs. |
| `--python-bin PYTHON_BIN` | Python interpreter used for evaluate_interactive_annotation.py subprocesses. |
| `--methods METHODS` | Comma-separated methods subset: felzenszwalb,slic,ssn |
| `--workers WORKERS` | Fallback parallelism when method-specific worker limits are not set. |
| `--simple-workers SIMPLE_WORKERS` | Parallel workers for CPU-only methods (felzenszwalb, slic). Defaults to --workers. |
| `--ssn-workers SSN_WORKERS` | Parallel workers for SSN cases. Keep low to avoid GPU/VRAM oversubscription. |
| `--overwrite` | Recompute cached spanno and rerun finished cases. |
| `--scribbles SCRIBBLES` | - |
| `--save_every SAVE_EVERY` | - |
| `--seed SEED` | - |
| `--resize-scale RESIZE_SCALE` | Resize source image and mask before sweep (1.0 keeps original size). |
| `--margin MARGIN` | - |
| `--border_margin BORDER_MARGIN` | - |
| `--no_overlap` | - |
| `--max_no_progress MAX_NO_PROGRESS` | - |
| `--region_selection_cycle REGION_SELECTION_CYCLE` | - |
| `--sensitivity SENSITIVITY` | - |
| `--emb_weights EMB_WEIGHTS` | - |
| `--emb_threshold EMB_THRESHOLD` | - |
| `--num_classes NUM_CLASSES` | - |
| `--felz-scales FELZ_SCALES` | - |
| `--felz-sigmas FELZ_SIGMAS` | - |
| `--felz-min-sizes FELZ_MIN_SIZES` | - |
| `--slic-n-segments SLIC_N_SEGMENTS` | - |
| `--slic-compactnesses SLIC_COMPACTNESSES` | - |
| `--slic-sigmas SLIC_SIGMAS` | - |
| `--ssn-weights SSN_WEIGHTS` | SSN checkpoint (.pth) |
| `--ssn-nspix-list SSN_NSPIX_LIST` | - |
| `--ssn-fdim SSN_FDIM` | - |
| `--ssn-niter-list SSN_NITER_LIST` | - |
| `--ssn-color-scales SSN_COLOR_SCALES` | - |
| `--ssn-pos-scales SSN_POS_SCALES` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.
- Для `spanno`-артефактов размер изображения и размер кэша должны совпадать, иначе будут ошибки совместимости.

## Raw `--help`

```text
usage: sweep_interactive_superpixels.py [-h] --image IMAGE --mask MASK
                                        --output-dir OUTPUT_DIR
                                        [--python-bin PYTHON_BIN]
                                        [--methods METHODS]
                                        [--workers WORKERS]
                                        [--simple-workers SIMPLE_WORKERS]
                                        [--ssn-workers SSN_WORKERS]
                                        [--overwrite] [--scribbles SCRIBBLES]
                                        [--save_every SAVE_EVERY]
                                        [--seed SEED]
                                        [--resize-scale RESIZE_SCALE]
                                        [--margin MARGIN]
                                        [--border_margin BORDER_MARGIN]
                                        [--no_overlap]
                                        [--max_no_progress MAX_NO_PROGRESS]
                                        [--region_selection_cycle REGION_SELECTION_CYCLE]
                                        [--sensitivity SENSITIVITY]
                                        [--emb_weights EMB_WEIGHTS]
                                        [--emb_threshold EMB_THRESHOLD]
                                        [--num_classes NUM_CLASSES]
                                        [--felz-scales FELZ_SCALES]
                                        [--felz-sigmas FELZ_SIGMAS]
                                        [--felz-min-sizes FELZ_MIN_SIZES]
                                        [--slic-n-segments SLIC_N_SEGMENTS]
                                        [--slic-compactnesses SLIC_COMPACTNESSES]
                                        [--slic-sigmas SLIC_SIGMAS]
                                        [--ssn-weights SSN_WEIGHTS]
                                        [--ssn-nspix-list SSN_NSPIX_LIST]
                                        [--ssn-fdim SSN_FDIM]
                                        [--ssn-niter-list SSN_NITER_LIST]
                                        [--ssn-color-scales SSN_COLOR_SCALES]
                                        [--ssn-pos-scales SSN_POS_SCALES]

Parallel parameter sweep for interactive superpixel annotation.

options:
  -h, --help            show this help message and exit
  --image IMAGE         RGB image for interactive evaluation.
  --mask MASK           GT mask with class ids.
  --output-dir OUTPUT_DIR
                        Directory for sweep outputs.
  --python-bin PYTHON_BIN
                        Python interpreter used for
                        evaluate_interactive_annotation.py subprocesses.
  --methods METHODS     Comma-separated methods subset: felzenszwalb,slic,ssn
  --workers WORKERS     Fallback parallelism when method-specific worker
                        limits are not set.
  --simple-workers SIMPLE_WORKERS
                        Parallel workers for CPU-only methods (felzenszwalb,
                        slic). Defaults to --workers.
  --ssn-workers SSN_WORKERS
                        Parallel workers for SSN cases. Keep low to avoid
                        GPU/VRAM oversubscription.
  --overwrite           Recompute cached spanno and rerun finished cases.

Interactive evaluation:
  --scribbles SCRIBBLES
  --save_every SAVE_EVERY
  --seed SEED
  --resize-scale RESIZE_SCALE
                        Resize source image and mask before sweep (1.0 keeps
                        original size).
  --margin MARGIN
  --border_margin BORDER_MARGIN
  --no_overlap
  --max_no_progress MAX_NO_PROGRESS
  --region_selection_cycle REGION_SELECTION_CYCLE
  --sensitivity SENSITIVITY
  --emb_weights EMB_WEIGHTS
  --emb_threshold EMB_THRESHOLD
  --num_classes NUM_CLASSES

Felzenszwalb grid:
  --felz-scales FELZ_SCALES
  --felz-sigmas FELZ_SIGMAS
  --felz-min-sizes FELZ_MIN_SIZES

SLIC grid:
  --slic-n-segments SLIC_N_SEGMENTS
  --slic-compactnesses SLIC_COMPACTNESSES
  --slic-sigmas SLIC_SIGMAS

SSN grid:
  --ssn-weights SSN_WEIGHTS
                        SSN checkpoint (.pth)
  --ssn-nspix-list SSN_NSPIX_LIST
  --ssn-fdim SSN_FDIM
  --ssn-niter-list SSN_NITER_LIST
  --ssn-color-scales SSN_COLOR_SCALES
  --ssn-pos-scales SSN_POS_SCALES
```
