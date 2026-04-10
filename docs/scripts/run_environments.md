# Окружения запуска

## Базовый вариант

- Для top-level скриптов используйте Python 3.12+.
- Создайте отдельное окружение проекта и установите зависимости из `requirements.txt`.
- Если нужен самый простой reproducible вариант для evaluation/legacy-сценариев, используйте bundled venv: `superpixel_annotator/superpixel_annotator_venv/bin/python`.

## Рекомендуемый порядок

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Когда использовать bundled venv

- Когда нужно быстро запустить evaluation/visualization/legacy CLI без отдельной ручной настройки.
- Когда нужно получить реальный `--help` и воспроизводимый parser для существующих скриптов.
- Когда проектовый `.venv` еще не подготовлен.

## Скрипты, которым обычно нужен checkpoint

- `train_neural_superpixels.py` — если идет fine-tuning через `--weights`
- `inference.py`
- `compare.py`
- `benchmark_simple_superpixel_methods.py`
- `evaluate_interactive_annotation.py` для `ssn`
- `evaluate_ssn_scribble_batch.py`
- `evaluate_superpixel_postprocessing.py`
- `optimize_superpixel_params_optuna.py` для `ssn`
- `sweep_interactive_superpixels.py` для `ssn`
- `tune_hybrid_conservative.py`
- `tune_low_confidence_threshold.py`

## Скрипты, которым нужен dataset

- `train.py`
- `train_neural_superpixels.py`
- `compare.py`
- `benchmark_configs.py`
- `benchmark_simple_superpixel_methods.py`
- `compute_mask_class_percentages.py`
- `evaluate_interactive_annotation.py` в batch-режиме
- `evaluate_ssn_scribble_batch.py`
- `evaluate_superpixel_postprocessing.py`
- `precompute_superpixels.py`
- `profile_minimal.py`
- `profile_one_batch.py`
- `refine_superpixel_on_pairs.py`
- `sweep_interactive_superpixels.py`
- `tune_hybrid_conservative.py`
- `tune_low_confidence_threshold.py`

## Скрипты, которым может понадобиться petroscope

- `benchmark_simple_superpixel_methods.py`
- `compute_mask_class_percentages.py`
- `evaluate_superpixel_postprocessing.py`
- `tune_hybrid_conservative.py`
- `tune_low_confidence_threshold.py`

## CPU / MPS / CUDA

- `train.py`, `train_neural_superpixels.py`, `inference.py`, `compare.py`, `evaluate_*`, `benchmark_*`, `tune_*` чувствительны к устройству и обычно выигрывают от `cuda` или `mps`.
- `profile_*` особенно полезны именно на том устройстве, на котором планируется работа.
- `plot_class_miou.py`, `render_interactive_annotation_video.py`, `compute_mask_class_percentages.py` и большая часть простых утилит могут запускаться на CPU.
- `superpixel_annotator/tk_service.py` зависит не столько от устройства, сколько от наличия GUI/дисплея.

## Куда складывать новые результаты

- Новые training/fine-tuning результаты удобно складывать в `artifacts/training/`.
- Интерактивные прогоны складывайте в `artifacts/interactive_runs/`.
- Sweep/refinement лучше держать в `artifacts/sweeps/` и `artifacts/refinement/`.
- Постобработку и сравнения держите в `artifacts/postprocessing/`.
