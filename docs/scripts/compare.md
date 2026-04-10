# compare.py

- Тип сценария: `comparison`
- Прямой запуск: **да**
- Файл: `compare.py`

## Назначение

Сравнение SSN и SLIC по метрикам и визуализациям.

## Когда использовать

Когда нужно сравнить SSN и SLIC на одном и том же наборе изображений.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 compare.py --img_dir /path/to/images --mask_dir /path/to/masks --csv reports/generated/comparison.csv --vis_dir artifacts/case_studies/compare_vis --weight models/checkpoints/S1v2_S2v2_x05.pth
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 compare.py --img_dir /path/to/images --mask_dir /path/to/masks --csv reports/generated/comparison.csv --vis_dir artifacts/case_studies/compare_vis --weight models/checkpoints/S1v2_S2v2_x05.pth
```

## Входы / выходы

- Входы: директория изображений, директория масок, checkpoint / веса модели.
- Выходы: CSV-таблица, директория визуализаций.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--img_dir IMG_DIR` | Directory with RGB images |
| `--mask_dir MASK_DIR` | Directory with grayscale masks |
| `--n_images N_IMAGES` | Number of images to evaluate (default: 50) |
| `--weight WEIGHT` | Path to SSN checkpoint (.pth). Omit to skip SSN. |
| `--fdim FDIM` | SSN feature dimension |
| `--niter NITER` | SSN SLIC iterations |
| `--color_scale COLOR_SCALE` | - |
| `--pos_scale POS_SCALE` | - |
| `--nspix NSPIX` | Target number of superpixels |
| `--enforce_connectivity` | Enforce superpixel connectivity (default: on) |
| `--no_enforce_connectivity` | - |
| `--slic_compactness SLIC_COMPACTNESS [SLIC_COMPACTNESS ...]` | SLIC compactness values to sweep (default: 0.1 1.0 10.0 30.0) |
| `--slic_sigma SLIC_SIGMA [SLIC_SIGMA ...]` | SLIC sigma values to sweep (default: 1.0) |
| `--csv CSV` | Output CSV path (default: comparison.csv) |
| `--vis_dir VIS_DIR` | Save visualisation grids to this directory. Omit to skip. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: compare.py [-h] --img_dir IMG_DIR --mask_dir MASK_DIR
                  [--n_images N_IMAGES] [--weight WEIGHT] [--fdim FDIM]
                  [--niter NITER] [--color_scale COLOR_SCALE]
                  [--pos_scale POS_SCALE] [--nspix NSPIX]
                  [--enforce_connectivity] [--no_enforce_connectivity]
                  [--slic_compactness SLIC_COMPACTNESS [SLIC_COMPACTNESS ...]]
                  [--slic_sigma SLIC_SIGMA [SLIC_SIGMA ...]] [--csv CSV]
                  [--vis_dir VIS_DIR]

Compare SSN vs SLIC on the first N dataset images

options:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Directory with RGB images
  --mask_dir MASK_DIR   Directory with grayscale masks
  --n_images N_IMAGES   Number of images to evaluate (default: 50)
  --weight WEIGHT       Path to SSN checkpoint (.pth). Omit to skip SSN.
  --fdim FDIM           SSN feature dimension
  --niter NITER         SSN SLIC iterations
  --color_scale COLOR_SCALE
  --pos_scale POS_SCALE
  --nspix NSPIX         Target number of superpixels
  --enforce_connectivity
                        Enforce superpixel connectivity (default: on)
  --no_enforce_connectivity
  --slic_compactness SLIC_COMPACTNESS [SLIC_COMPACTNESS ...]
                        SLIC compactness values to sweep (default: 0.1 1.0
                        10.0 30.0)
  --slic_sigma SLIC_SIGMA [SLIC_SIGMA ...]
                        SLIC sigma values to sweep (default: 1.0)
  --csv CSV             Output CSV path (default: comparison.csv)
  --vis_dir VIS_DIR     Save visualisation grids to this directory. Omit to
                        skip.
```
