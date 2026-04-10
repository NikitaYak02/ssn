# inference.py

- Тип сценария: `inference`
- Прямой запуск: **да**
- Файл: `inference.py`

## Назначение

Инференс модели на одном изображении или наборе изображений.

## Когда использовать

Когда нужно быстро получить предсказания модели без запуска обучения или длинных экспериментов.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 inference.py --image /path/to/image.png --output artifacts/demo_output/result.png --weight models/checkpoints/S1v2_S2v2_x05.pth
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 inference.py --image /path/to/image.png --output artifacts/demo_output/result.png --weight models/checkpoints/S1v2_S2v2_x05.pth
```

## Входы / выходы

- Входы: checkpoint / веса модели.
- Выходы: файл результата.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--image IMAGE` | Path to input image |
| `--weight WEIGHT` | Path to pretrained weight |
| `--fdim FDIM` | Embedding dimension |
| `--niter NITER` | Number of SLIC iterations |
| `--nspix NSPIX` | Number of superpixels |
| `--color_scale COLOR_SCALE` | - |
| `--pos_scale POS_SCALE` | - |
| `--output OUTPUT` | Output file path |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: inference.py [-h] --image IMAGE [--weight WEIGHT] [--fdim FDIM]
                    [--niter NITER] [--nspix NSPIX]
                    [--color_scale COLOR_SCALE] [--pos_scale POS_SCALE]
                    [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --image IMAGE         Path to input image
  --weight WEIGHT       Path to pretrained weight
  --fdim FDIM           Embedding dimension
  --niter NITER         Number of SLIC iterations
  --nspix NSPIX         Number of superpixels
  --color_scale COLOR_SCALE
  --pos_scale POS_SCALE
  --output OUTPUT       Output file path
```
