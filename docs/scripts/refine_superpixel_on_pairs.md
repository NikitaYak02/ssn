# refine_superpixel_on_pairs.py

- Тип сценария: `refinement`
- Прямой запуск: **да**
- Файл: `refine_superpixel_on_pairs.py`

## Назначение

Локальное доуточнение параметров superpixel-метода на выбранных парах.

## Когда использовать

Когда после грубого sweep нужно локально доуточнить хорошие конфигурации на выбранных парах.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 refine_superpixel_on_pairs.py --pairs-json artifacts/refinement/selected_pairs_s1_v2_full_cover.json --output-dir artifacts/refinement/local_refine --method ssn --ssn-weights models/checkpoints/best_model.pth --scribbles 100
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 refine_superpixel_on_pairs.py --pairs-json artifacts/refinement/selected_pairs_s1_v2_full_cover.json --output-dir artifacts/refinement/local_refine --method ssn --ssn-weights models/checkpoints/best_model.pth --scribbles 100
```

### Локальный refinement для SLIC

```bash
python3 refine_superpixel_on_pairs.py --pairs-json artifacts/refinement/selected_pairs_s1_v2_full_cover.json --output-dir artifacts/refinement/local_refine_slic --method slic --scribbles 100 --slic-n-segments 650,750 --slic-compactnesses 9.5,11.0 --slic-sigmas 0.42,0.52
```

## Входы / выходы

- Входы: checkpoint / веса модели, JSON со списком пар image/mask.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--pairs-json PAIRS_JSON` | JSON list of {name,image,mask}. |
| `--output-dir OUTPUT_DIR` | - |
| `--python-bin PYTHON_BIN` | - |
| `--method {slic,ssn,felzenszwalb}` | - |
| `--resize-scale RESIZE_SCALE` | - |
| `--scribbles SCRIBBLES` | - |
| `--save_every SAVE_EVERY` | - |
| `--seed SEED` | - |
| `--workers WORKERS` | - |
| `--overwrite` | - |
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
| `--ssn-weights SSN_WEIGHTS` | - |
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
usage: refine_superpixel_on_pairs.py [-h] --pairs-json PAIRS_JSON --output-dir
                                     OUTPUT_DIR [--python-bin PYTHON_BIN]
                                     --method {slic,ssn,felzenszwalb}
                                     [--resize-scale RESIZE_SCALE]
                                     [--scribbles SCRIBBLES]
                                     [--save_every SAVE_EVERY] [--seed SEED]
                                     [--workers WORKERS] [--overwrite]
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

Refine local superpixel parameters on a selected image set.

options:
  -h, --help            show this help message and exit
  --pairs-json PAIRS_JSON
                        JSON list of {name,image,mask}.
  --output-dir OUTPUT_DIR
  --python-bin PYTHON_BIN
  --method {slic,ssn,felzenszwalb}
  --resize-scale RESIZE_SCALE
  --scribbles SCRIBBLES
  --save_every SAVE_EVERY
  --seed SEED
  --workers WORKERS
  --overwrite

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

Felzenszwalb local grid:
  --felz-scales FELZ_SCALES
  --felz-sigmas FELZ_SIGMAS
  --felz-min-sizes FELZ_MIN_SIZES

SLIC local grid:
  --slic-n-segments SLIC_N_SEGMENTS
  --slic-compactnesses SLIC_COMPACTNESSES
  --slic-sigmas SLIC_SIGMAS

SSN local grid:
  --ssn-weights SSN_WEIGHTS
  --ssn-nspix-list SSN_NSPIX_LIST
  --ssn-fdim SSN_FDIM
  --ssn-niter-list SSN_NITER_LIST
  --ssn-color-scales SSN_COLOR_SCALES
  --ssn-pos-scales SSN_POS_SCALES
```
