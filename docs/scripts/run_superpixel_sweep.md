# run_superpixel_sweep.sh

- Тип сценария: `shell-wrapper`
- Прямой запуск: **да**
- Файл: `run_superpixel_sweep.sh`

## Назначение

Shell-обертка для запуска sweep_interactive_superpixels.py с дефолтными путями.

## Когда использовать

Когда нужен быстрый reproducible запуск sweep со встроенными дефолтами.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
bash run_superpixel_sweep.sh artifacts/case_studies/_quarter_run/input/train_01_q1.jpg artifacts/case_studies/_quarter_run/input/train_01_q1.png artifacts/sweeps/default_100 models/checkpoints/best_model.pth
```

## Типовые сценарии

### Минимальный запуск

```bash
bash run_superpixel_sweep.sh artifacts/case_studies/_quarter_run/input/train_01_q1.jpg artifacts/case_studies/_quarter_run/input/train_01_q1.png artifacts/sweeps/default_100 models/checkpoints/best_model.pth
```

## Входы / выходы

- Входы: аргументы командной строки и входные пути по конкретному сценарию.
- Выходы: stdout, логи и/или артефакты в путях, переданных через CLI.

## Где искать результаты

Результаты лежат в директории, переданной третьим аргументом shell-скрипта.

## Ключевые аргументы

_Ключевые опции автоматически не извлечены; используйте `Raw --help` ниже._

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.

## Raw `--help`

```text
У этого скрипта нет стандартного `--help`, или он рассчитан на импорт/GUI.
```
