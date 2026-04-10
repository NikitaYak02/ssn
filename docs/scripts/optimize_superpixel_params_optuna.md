# optimize_superpixel_params_optuna.py

- Тип сценария: `optimization`
- Прямой запуск: **да**
- Файл: `optimize_superpixel_params_optuna.py`

## Назначение

Optuna-оптимизация параметров superpixel-метода.

## Когда использовать

Когда нужен автоматический поиск параметров вместо ручного sweep.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 optimize_superpixel_params_optuna.py --image /path/to/image.png --mask /path/to/mask.png --output-dir artifacts/sweeps/optuna_demo --method ssn --ssn-weights models/checkpoints/best_model.pth --trials 20 --scribbles 100
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 optimize_superpixel_params_optuna.py --image /path/to/image.png --mask /path/to/mask.png --output-dir artifacts/sweeps/optuna_demo --method ssn --ssn-weights models/checkpoints/best_model.pth --trials 20 --scribbles 100
```

### Оптимизация только для SLIC

```bash
python3 optimize_superpixel_params_optuna.py --image /path/to/image.png --mask /path/to/mask.png --output-dir artifacts/sweeps/optuna_slic_demo --method slic --trials 20 --scribbles 100
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
| `--image IMAGE` | - |
| `--mask MASK` | - |
| `--output-dir OUTPUT_DIR` | - |
| `--method {felzenszwalb,slic,ssn}` | Optimize one method per study. |
| `--python-bin PYTHON_BIN` | - |
| `--resize-scale RESIZE_SCALE` | - |
| `--scribbles SCRIBBLES` | - |
| `--save_every SAVE_EVERY` | - |
| `--seed SEED` | - |
| `--trials TRIALS` | - |
| `--jobs JOBS` | Parallel Optuna trials. Keep 1 for SSN unless you know GPU memory is sufficient. |
| `--overwrite` | - |
| `--study-name STUDY_NAME` | - |
| `--storage STORAGE` | Optuna storage URL. Defaults to local sqlite. |
| `--margin MARGIN` | - |
| `--border_margin BORDER_MARGIN` | - |
| `--no_overlap` | - |
| `--max_no_progress MAX_NO_PROGRESS` | - |
| `--region_selection_cycle REGION_SELECTION_CYCLE` | - |
| `--sensitivity SENSITIVITY` | - |
| `--emb_weights EMB_WEIGHTS` | - |
| `--emb_threshold EMB_THRESHOLD` | - |
| `--num_classes NUM_CLASSES` | - |
| `--ssn-weights SSN_WEIGHTS` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: optimize_superpixel_params_optuna.py [-h] --image IMAGE --mask MASK
                                            --output-dir OUTPUT_DIR --method
                                            {felzenszwalb,slic,ssn}
                                            [--python-bin PYTHON_BIN]
                                            [--resize-scale RESIZE_SCALE]
                                            [--scribbles SCRIBBLES]
                                            [--save_every SAVE_EVERY]
                                            [--seed SEED] [--trials TRIALS]
                                            [--jobs JOBS] [--overwrite]
                                            [--study-name STUDY_NAME]
                                            [--storage STORAGE]
                                            [--margin MARGIN]
                                            [--border_margin BORDER_MARGIN]
                                            [--no_overlap]
                                            [--max_no_progress MAX_NO_PROGRESS]
                                            [--region_selection_cycle REGION_SELECTION_CYCLE]
                                            [--sensitivity SENSITIVITY]
                                            [--emb_weights EMB_WEIGHTS]
                                            [--emb_threshold EMB_THRESHOLD]
                                            [--num_classes NUM_CLASSES]
                                            [--ssn-weights SSN_WEIGHTS]

Optuna-based hyperparameter search for superpixel methods.

options:
  -h, --help            show this help message and exit
  --image IMAGE
  --mask MASK
  --output-dir OUTPUT_DIR
  --method {felzenszwalb,slic,ssn}
                        Optimize one method per study.
  --python-bin PYTHON_BIN
  --resize-scale RESIZE_SCALE
  --scribbles SCRIBBLES
  --save_every SAVE_EVERY
  --seed SEED
  --trials TRIALS
  --jobs JOBS           Parallel Optuna trials. Keep 1 for SSN unless you know
                        GPU memory is sufficient.
  --overwrite
  --study-name STUDY_NAME
  --storage STORAGE     Optuna storage URL. Defaults to local sqlite.

Interactive evaluation:
  --margin MARGIN
  --border_margin BORDER_MARGIN
  --no_overlap
  --max_no_progress MAX_NO_PROGRESS
  --region_selection_cycle REGION_SELECTION_CYCLE
  --sensitivity SENSITIVITY
  --emb_weights EMB_WEIGHTS
  --emb_threshold EMB_THRESHOLD
  --num_classes NUM_CLASSES

SSN:
  --ssn-weights SSN_WEIGHTS
```
