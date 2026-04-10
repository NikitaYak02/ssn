# train_neural_superpixels.py

- Тип сценария: `training`
- Прямой запуск: **да**
- Файл: `train_neural_superpixels.py`

## Назначение

Обучение или fine-tuning нейросетевых superpixel-методов на целевом домене.

## Когда использовать

Когда нужно обучить отдельный neural superpixel backend или адаптировать его под целевой домен.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 train_neural_superpixels.py --img_dir /path/to/images --mask_dir /path/to/masks --out_dir artifacts/training/neural_superpixels --method deep_slic --train_iter 5000
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 train_neural_superpixels.py --img_dir /path/to/images --mask_dir /path/to/masks --out_dir artifacts/training/neural_superpixels --method deep_slic --train_iter 5000
```

## Входы / выходы

- Входы: директория изображений, директория масок, checkpoint / веса модели.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--img_dir IMG_DIR` | Directory with RGB images. |
| `--mask_dir MASK_DIR` | Directory with grayscale masks. |
| `--out_dir OUT_DIR` | Output directory. |
| `--method {cnn_rim,deep_slic,rethink_unsup,sin,sp_fcn}` | Neural superpixel method to train. |
| `--method_config METHOD_CONFIG` | JSON string or path to JSON config. |
| `--weights WEIGHTS` | Optional checkpoint to fine-tune from. |
| `--val_ratio VAL_RATIO` | - |
| `--max_classes MAX_CLASSES` | - |
| `--crop_size CROP_SIZE` | - |
| `--batchsize BATCHSIZE` | - |
| `--nworkers NWORKERS` | - |
| `--lr LR` | - |
| `--train_iter TRAIN_ITER` | - |
| `--compactness_weight COMPACTNESS_WEIGHT` | - |
| `--edge_reg_weight EDGE_REG_WEIGHT` | - |
| `--eval_downscale EVAL_DOWNSCALE` | - |
| `--print_interval PRINT_INTERVAL` | - |
| `--test_interval TEST_INTERVAL` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: train_neural_superpixels.py [-h] --img_dir IMG_DIR --mask_dir MASK_DIR
                                   [--out_dir OUT_DIR] --method
                                   {cnn_rim,deep_slic,rethink_unsup,sin,sp_fcn}
                                   [--method_config METHOD_CONFIG]
                                   [--weights WEIGHTS] [--val_ratio VAL_RATIO]
                                   [--max_classes MAX_CLASSES]
                                   [--crop_size CROP_SIZE]
                                   [--batchsize BATCHSIZE]
                                   [--nworkers NWORKERS] [--lr LR]
                                   [--train_iter TRAIN_ITER]
                                   [--compactness_weight COMPACTNESS_WEIGHT]
                                   [--edge_reg_weight EDGE_REG_WEIGHT]
                                   [--eval_downscale EVAL_DOWNSCALE]
                                   [--print_interval PRINT_INTERVAL]
                                   [--test_interval TEST_INTERVAL]

Train or fine-tune neural superpixel methods on the target domain.

options:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Directory with RGB images.
  --mask_dir MASK_DIR   Directory with grayscale masks.
  --out_dir OUT_DIR     Output directory.
  --method {cnn_rim,deep_slic,rethink_unsup,sin,sp_fcn}
                        Neural superpixel method to train.
  --method_config METHOD_CONFIG
                        JSON string or path to JSON config.
  --weights WEIGHTS     Optional checkpoint to fine-tune from.
  --val_ratio VAL_RATIO
  --max_classes MAX_CLASSES
  --crop_size CROP_SIZE
  --batchsize BATCHSIZE
  --nworkers NWORKERS
  --lr LR
  --train_iter TRAIN_ITER
  --compactness_weight COMPACTNESS_WEIGHT
  --edge_reg_weight EDGE_REG_WEIGHT
  --eval_downscale EVAL_DOWNSCALE
  --print_interval PRINT_INTERVAL
  --test_interval TEST_INTERVAL
```
