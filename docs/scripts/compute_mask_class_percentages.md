# compute_mask_class_percentages.py

- Тип сценария: `dataset-analysis`
- Прямой запуск: **да**
- Файл: `compute_mask_class_percentages.py`

## Назначение

Подсчет распределения классов по маскам датасета.

## Когда использовать

Когда нужно понять дисбаланс классов в масках и подготовить сводную статистику по датасету.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 compute_mask_class_percentages.py --masks /path/to/masks
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 compute_mask_class_percentages.py --masks /path/to/masks
```

## Входы / выходы

- Входы: директория масок.
- Выходы: сводная статистика по классам.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--masks MASKS` | Directory with masks. |
| `--class-codes CLASS_CODES` | Optional comma-separated class-code order. If omitted, the script uses the sorted codes observed in masks. |
| `--ignore-codes IGNORE_CODES` | Optional comma-separated raw codes to ignore, for example 255. Ignored pixels are excluded from all percentages. |
| `--petroscope-root PETROSCOPE_ROOT` | Optional local petroscope checkout for recovering class names. |
| `--output-csv OUTPUT_CSV` | Optional path to save the class-percentage table as CSV. |
| `--output-json OUTPUT_JSON` | Optional path to save the summary as JSON. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: compute_mask_class_percentages.py [-h] --masks MASKS
                                         [--class-codes CLASS_CODES]
                                         [--ignore-codes IGNORE_CODES]
                                         [--petroscope-root PETROSCOPE_ROOT]
                                         [--output-csv OUTPUT_CSV]
                                         [--output-json OUTPUT_JSON]

Compute class-content percentages from segmentation masks.

options:
  -h, --help            show this help message and exit
  --masks MASKS         Directory with masks.
  --class-codes CLASS_CODES
                        Optional comma-separated class-code order. If omitted,
                        the script uses the sorted codes observed in masks.
  --ignore-codes IGNORE_CODES
                        Optional comma-separated raw codes to ignore, for
                        example 255. Ignored pixels are excluded from all
                        percentages.
  --petroscope-root PETROSCOPE_ROOT
                        Optional local petroscope checkout for recovering
                        class names.
  --output-csv OUTPUT_CSV
                        Optional path to save the class-percentage table as
                        CSV.
  --output-json OUTPUT_JSON
                        Optional path to save the summary as JSON.
```
