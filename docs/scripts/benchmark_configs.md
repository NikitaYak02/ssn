# benchmark_configs.py

- Тип сценария: `benchmark`
- Прямой запуск: **да**
- Файл: `benchmark_configs.py`

## Назначение

Сравнение разных конфигураций обучения или инференса.

## Когда использовать

Когда нужно быстро сравнить несколько конфигураций и замерить скорость или качество.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 benchmark_configs.py --img_dir /path/to/images --mask_dir /path/to/masks
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 benchmark_configs.py --img_dir /path/to/images --mask_dir /path/to/masks
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
| `--repeats REPEATS` | Repeat each config N times |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: benchmark_configs.py [-h] --img_dir IMG_DIR --mask_dir MASK_DIR
                            [--repeats REPEATS]

Benchmark different training configurations

options:
  -h, --help           show this help message and exit
  --img_dir IMG_DIR
  --mask_dir MASK_DIR
  --repeats REPEATS    Repeat each config N times
```
