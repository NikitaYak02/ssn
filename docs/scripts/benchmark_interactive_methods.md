# benchmark_interactive_methods.py

- Тип сценария: `benchmark`
- Прямой запуск: **да**
- Файл: `benchmark_interactive_methods.py`

## Назначение

Единый benchmark для сравнения текущего scribble-pipeline и внешних интерактивных методов по фиксированным бюджетам взаимодействий.

## Когда использовать

Когда нужно:

- сравнить `current_pipeline` и внешние interactive methods на одном наборе изображений;
- получить единые `per_step.csv`, `summary.csv/json`, `quality_vs_interactions.png`, `leaderboard.md`;
- запускать современные методы через adapter/subprocess слой без смешивания их зависимостей с локальным `requirements.txt`.

## Минимальный запуск

```bash
python3 benchmark_interactive_methods.py \
  --image /path/to/image.png \
  --mask /path/to/mask.png \
  --output-dir artifacts/interactive_benchmark/demo \
  --methods current_pipeline,mock_click,mock_line,mock_scribble \
  --interaction-budgets 1,3,5
```

## Пример с batch-режимом

```bash
python3 benchmark_interactive_methods.py \
  --images /path/to/images \
  --masks /path/to/masks \
  --output-dir artifacts/interactive_benchmark/batch_demo \
  --methods current_pipeline,interformer,seem,semantic_sam,segnext,iseg \
  --current-pipeline-args '{"method":"slic","n_segments":64,"compactness":5.0,"sigma":0.0,"sensitivity":1.8}' \
  --interaction-budgets 1,3,5,10,20
```

## Внешние методы

- Манифесты лежат в `interactive_benchmark/manifests/`.
- По умолчанию реальные external methods будут пропущены, если в манифесте не задан runnable `entrypoint.command`.
- Для тестов и smoke-запусков доступны встроенные mock adapters:
  - `mock_click`
  - `mock_line`
  - `mock_scribble`

## Выходы

- `per_step.csv` — строки по `image × method × interaction_budget`
- `summary.csv` / `summary.json` — агрегированные средние по методам
- `quality_vs_interactions.png` — кривые `mIoU`, `coverage`, `precision`
- `leaderboard.md` — финальный ranking на максимальном interaction budget

## Важные детали

- `current_pipeline` запускается как black-box через `evaluate_interactive_annotation.py` с `--save_every 1`.
- Для внешних методов benchmark использует `last-update-wins` при overlap между class masks.
- Если метод возвращает несколько mask proposals за один interaction, benchmark выбирает proposal с максимальным IoU относительно target-component GT.
