# plot_class_miou.py

- Тип сценария: `visualization`
- Прямой запуск: **да**
- Файл: `plot_class_miou.py`

## Назначение

Построение графиков per-class IoU и mIoU по metrics.csv.

## Когда использовать

Когда нужно визуализировать trajectory метрик после уже завершенного прогона.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 plot_class_miou.py --mask /path/to/mask.png --out_dir artifacts/interactive_runs/demo/class_miou.png --metrics artifacts/interactive_runs/demo/metrics.csv
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 plot_class_miou.py --mask /path/to/mask.png --out_dir artifacts/interactive_runs/demo/class_miou.png --metrics artifacts/interactive_runs/demo/metrics.csv
```

## Входы / выходы

- Входы: metrics.csv завершенного прогона.
- Выходы: директория результатов, PNG-графики.

## Где искать результаты

Ищите PNG-графики рядом с путем, указанным в `--out`.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--metrics METRICS` | Путь к metrics.csv |
| `--out_dir OUT_DIR` | Куда сохранить графики |
| `--mask MASK` | GT-маска изображения. Если указана, графики строятся только для классов, которые реально присутствуют на изображении. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: plot_class_miou.py [-h] --metrics METRICS --out_dir OUT_DIR
                          [--mask MASK]

plot_class_miou.py

Строит графики per-class IoU / mIoU по файлу metrics.csv, который создаёт
evaluate_interactive_annotation.py.

Сохраняет:
  - class_iou_over_time.png        — все классы на одном графике + mIoU
  - class_iou_grid.png             — отдельный subplot для каждого класса

Пример:
    python plot_class_miou.py         --metrics /path/to/metrics.csv         --out_dir /path/to/plots

options:
  -h, --help         show this help message and exit
  --metrics METRICS  Путь к metrics.csv
  --out_dir OUT_DIR  Куда сохранить графики
  --mask MASK        GT-маска изображения. Если указана, графики строятся
                     только для классов, которые реально присутствуют на
                     изображении.
```
