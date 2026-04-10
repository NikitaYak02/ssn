# superpixel_annotator/viz_spanno_annotations.py

- Тип сценария: `visualization`
- Прямой запуск: **да**
- Файл: `superpixel_annotator/viz_spanno_annotations.py`

## Назначение

Визуализация state/spanno-аннотаций поверх изображения.

## Когда использовать

Когда нужно отрисовать state/spanno поверх картинки и быстро проверить артефакт.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 superpixel_annotator/viz_spanno_annotations.py
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 superpixel_annotator/viz_spanno_annotations.py
```

## Входы / выходы

- Входы: аргументы командной строки и входные пути по конкретному сценарию.
- Выходы: stdout, логи и/или артефакты в путях, переданных через CLI.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

_Ключевые опции автоматически не извлечены; используйте `Raw --help` ниже._

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
Не удалось автоматически получить --help:
Traceback (most recent call last):
  File "/Users/nikitayakovlev/dev/diplom/ssn/superpixel_annotator/viz_spanno_annotations.py", line 52, in <module>
    import structs  # твой structs.py
    ^^^^^^^^^^^^^^
  File "/Users/nikitayakovlev/dev/diplom/ssn/superpixel_annotator/structs.py", line 32, in <module>
    from lib.utils.torch_device import get_torch_device, synchronize_device
ModuleNotFoundError: No module named 'lib'
```
