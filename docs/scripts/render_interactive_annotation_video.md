# render_interactive_annotation_video.py

- Тип сценария: `visualization`
- Прямой запуск: **да**
- Файл: `render_interactive_annotation_video.py`

## Назначение

Сборка MP4-видео из кадров интерактивной разметки.

## Когда использовать

Когда нужно превратить state/frame артефакты в видео для демонстрации или анализа.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 render_interactive_annotation_video.py --input artifacts/case_studies/interactive_repro_train01_ssn --out artifacts/case_studies/videos/train01.mp4
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 render_interactive_annotation_video.py --input artifacts/case_studies/interactive_repro_train01_ssn --out artifacts/case_studies/videos/train01.mp4
```

### Пакетная сборка видео

```bash
python3 render_interactive_annotation_video.py --input artifacts/case_studies/ssn_s1_v2_halfsize --out artifacts/case_studies/videos
```

## Входы / выходы

- Входы: директория результата или одиночный run-каталог.
- Выходы: директория результатов, MP4-видео.

## Где искать результаты

Ищите MP4 по пути из `--out`; в batch-режиме это директория с несколькими видео.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--input INPUT` | Single result directory or batch root with state_*.json files |
| `--out OUT` | Output .mp4 path for a single run or directory for batch mode |
| `--image IMAGE` | Optional original image for a single run |
| `--image_dir IMAGE_DIR` | Optional image directory for batch mode; matched by run directory stem |
| `--method METHOD` | Method key from the saved state; defaults to the first available one |
| `--fps FPS` | Output video FPS |
| `--max_side MAX_SIDE` | Resize the longest rendered side to at most this value |
| `--intro_seconds INTRO_SECONDS` | - |
| `--pre_seconds PRE_SECONDS` | - |
| `--direct_seconds DIRECT_SECONDS` | - |
| `--prop_seconds PROP_SECONDS` | - |
| `--final_seconds FINAL_SECONDS` | - |
| `--outro_seconds OUTRO_SECONDS` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: render_interactive_annotation_video.py [-h] --input INPUT [--out OUT]
                                              [--image IMAGE]
                                              [--image_dir IMAGE_DIR]
                                              [--method METHOD] [--fps FPS]
                                              [--max_side MAX_SIDE]
                                              [--intro_seconds INTRO_SECONDS]
                                              [--pre_seconds PRE_SECONDS]
                                              [--direct_seconds DIRECT_SECONDS]
                                              [--prop_seconds PROP_SECONDS]
                                              [--final_seconds FINAL_SECONDS]
                                              [--outro_seconds OUTRO_SECONDS]

render_interactive_annotation_video.py

Build an MP4 replay from `evaluate_interactive_annotation.py` outputs.

The script reads `state_*.json` / `state_*.json.gz` checkpoints and visualizes:
  1. current annotated superpixels,
  2. newly added scribbles,
  3. direct superpixel hits (`parent_intersect=True`),
  4. propagated labels (`parent_intersect=False`).

If checkpoints were saved sparsely (for example `--save_every 50`), the replay is
coarse between checkpoints because exact per-scribble intermediate states are no
longer available. For an exact step-by-step replay, run
`evaluate_interactive_annotation.py --save_every 1`.

Examples:
    superpixel_annotator/superpixel_annotator_venv/bin/python         render_interactive_annotation_video.py         --input artifacts/case_studies/interactive_repro_train01_ssn         --image /path/to/image.png

    superpixel_annotator/superpixel_annotator_venv/bin/python         render_interactive_annotation_video.py         --input artifacts/case_studies/ssn_s1_v2_halfsize         --image_dir /data/images         --fps 8

options:
  -h, --help            show this help message and exit
  --input INPUT         Single result directory or batch root with
                        state_*.json files
  --out OUT             Output .mp4 path for a single run or directory for
                        batch mode
  --image IMAGE         Optional original image for a single run
  --image_dir IMAGE_DIR
                        Optional image directory for batch mode; matched by
                        run directory stem
  --method METHOD       Method key from the saved state; defaults to the first
                        available one
  --fps FPS             Output video FPS
  --max_side MAX_SIDE   Resize the longest rendered side to at most this value
  --intro_seconds INTRO_SECONDS
  --pre_seconds PRE_SECONDS
  --direct_seconds DIRECT_SECONDS
  --prop_seconds PROP_SECONDS
  --final_seconds FINAL_SECONDS
  --outro_seconds OUTRO_SECONDS
```
