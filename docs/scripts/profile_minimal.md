# profile_minimal.py

- Тип сценария: `profiling`
- Прямой запуск: **да**
- Файл: `profile_minimal.py`

## Назначение

Минимальный профилинг обучения без накладных расходов DataLoader workers.

## Когда использовать

Когда нужно понять базовую стоимость одного training step без шума от workers.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 profile_minimal.py --img_dir /path/to/images --mask_dir /path/to/masks
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 profile_minimal.py --img_dir /path/to/images --mask_dir /path/to/masks
```

## Входы / выходы

- Входы: директория изображений, директория масок.
- Выходы: stdout, логи и/или артефакты в путях, переданных через CLI.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--img_dir IMG_DIR` | - |
| `--mask_dir MASK_DIR` | - |
| `--n_batches N_BATCHES` | - |
| `--batchsize BATCHSIZE` | - |
| `--crop_size CROP_SIZE` | - |
| `--max_classes MAX_CLASSES` | - |
| `--fdim FDIM` | - |
| `--nspix NSPIX` | - |
| `--niter NITER` | - |
| `--lr LR` | - |
| `--color_scale COLOR_SCALE` | - |
| `--pos_scale POS_SCALE` | - |
| `--compactness COMPACTNESS` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: profile_minimal.py [-h] --img_dir IMG_DIR --mask_dir MASK_DIR
                          [--n_batches N_BATCHES] [--batchsize BATCHSIZE]
                          [--crop_size CROP_SIZE] [--max_classes MAX_CLASSES]
                          [--fdim FDIM] [--nspix NSPIX] [--niter NITER]
                          [--lr LR] [--color_scale COLOR_SCALE]
                          [--pos_scale POS_SCALE] [--compactness COMPACTNESS]

Profile without DataLoader workers overhead

options:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR
  --mask_dir MASK_DIR
  --n_batches N_BATCHES
  --batchsize BATCHSIZE
  --crop_size CROP_SIZE
  --max_classes MAX_CLASSES
  --fdim FDIM
  --nspix NSPIX
  --niter NITER
  --lr LR
  --color_scale COLOR_SCALE
  --pos_scale POS_SCALE
  --compactness COMPACTNESS
```
