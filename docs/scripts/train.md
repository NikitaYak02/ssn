# train.py

- Тип сценария: `training`
- Прямой запуск: **да**
- Файл: `train.py`

## Назначение

Обучение базовой SSN-модели сегментации.

## Когда использовать

Когда нужно обучить или дообучить основную SSN-модель на датасете сегментации.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 train.py --img_dir /path/to/images --mask_dir /path/to/masks --out_dir artifacts/demo_output --train_iter 500000
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 train.py --img_dir /path/to/images --mask_dir /path/to/masks --out_dir artifacts/demo_output --train_iter 500000
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
| `--img_dir IMG_DIR` | Directory with RGB images |
| `--mask_dir MASK_DIR` | Directory with grayscale masks |
| `--val_ratio VAL_RATIO` | - |
| `--max_classes MAX_CLASSES` | Max number of semantic classes in masks |
| `--crop_size CROP_SIZE` | Square crop size for training patches |
| `--out_dir OUT_DIR` | Checkpoint output directory |
| `--batchsize BATCHSIZE` | - |
| `--nworkers NWORKERS` | DataLoader workers (0 = main process only) |
| `--lr LR` | - |
| `--train_iter TRAIN_ITER` | - |
| `--fdim FDIM` | Feature embedding dimension |
| `--niter NITER` | Differentiable SLIC iterations |
| `--nspix NSPIX` | Number of superpixels |
| `--color_scale COLOR_SCALE` | - |
| `--pos_scale POS_SCALE` | - |
| `--compactness COMPACTNESS` | - |
| `--test_interval TEST_INTERVAL` | - |
| `--print_interval PRINT_INTERVAL` | - |
| `--n_viz_images N_VIZ_IMAGES` | Validation images to visualise at each test step |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: train.py [-h] --img_dir IMG_DIR --mask_dir MASK_DIR
                [--val_ratio VAL_RATIO] [--max_classes MAX_CLASSES]
                [--crop_size CROP_SIZE] [--out_dir OUT_DIR]
                [--batchsize BATCHSIZE] [--nworkers NWORKERS] [--lr LR]
                [--train_iter TRAIN_ITER] [--fdim FDIM] [--niter NITER]
                [--nspix NSPIX] [--color_scale COLOR_SCALE]
                [--pos_scale POS_SCALE] [--compactness COMPACTNESS]
                [--test_interval TEST_INTERVAL]
                [--print_interval PRINT_INTERVAL]
                [--n_viz_images N_VIZ_IMAGES]

options:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Directory with RGB images
  --mask_dir MASK_DIR   Directory with grayscale masks
  --val_ratio VAL_RATIO
  --max_classes MAX_CLASSES
                        Max number of semantic classes in masks
  --crop_size CROP_SIZE
                        Square crop size for training patches
  --out_dir OUT_DIR     Checkpoint output directory
  --batchsize BATCHSIZE
  --nworkers NWORKERS   DataLoader workers (0 = main process only)
  --lr LR
  --train_iter TRAIN_ITER
  --fdim FDIM           Feature embedding dimension
  --niter NITER         Differentiable SLIC iterations
  --nspix NSPIX         Number of superpixels
  --color_scale COLOR_SCALE
  --pos_scale POS_SCALE
  --compactness COMPACTNESS
  --test_interval TEST_INTERVAL
  --print_interval PRINT_INTERVAL
  --n_viz_images N_VIZ_IMAGES
                        Validation images to visualise at each test step
```
