# tune_low_confidence_threshold.py

- Тип сценария: `optimization`
- Прямой запуск: **да**
- Файл: `tune_low_confidence_threshold.py`

## Назначение

Подбор порогов low-confidence postprocessing.

## Когда использовать

Когда нужно подобрать пороги low-confidence overwrite стратегии.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 tune_low_confidence_threshold.py --cache-dir artifacts/postprocessing/hybrid_tune/logits_cache --output-dir artifacts/postprocessing/low_conf_tune --threshold-start 0.5 --threshold-stop 0.9 --threshold-step 0.05
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 tune_low_confidence_threshold.py --cache-dir artifacts/postprocessing/hybrid_tune/logits_cache --output-dir artifacts/postprocessing/low_conf_tune --threshold-start 0.5 --threshold-stop 0.9 --threshold-step 0.05
```

### Явный список threshold-ов

```bash
python3 tune_low_confidence_threshold.py --cache-dir artifacts/postprocessing/hybrid_tune/logits_cache --output-dir artifacts/postprocessing/low_conf_tune_manual --thresholds 0.55,0.60,0.65,0.70,0.75
```

## Входы / выходы

- Входы: готовый on-disk cache.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--cache-dir CACHE_DIR` | Directory containing cache_manifest.json and per-image .npz files. |
| `--output-dir OUTPUT_DIR` | Directory where tuning CSV/JSON will be written. |
| `--thresholds THRESHOLDS` | Optional comma-separated explicit threshold list. If omitted, the script uses threshold-start/stop/step. |
| `--threshold-start THRESHOLD_START` | Start of the threshold sweep, inclusive. |
| `--threshold-stop THRESHOLD_STOP` | End of the threshold sweep, inclusive within rounding tolerance. |
| `--threshold-step THRESHOLD_STEP` | Step for the threshold sweep. |
| `--max-acc-drop MAX_ACC_DROP` | Max allowed pixel-accuracy drop for the 'safe' best threshold. |
| `--top-k TOP_K` | How many top thresholds to include in the summary JSON. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.
- Этот скрипт не строит cache сам: сначала подготовьте его через `tune_hybrid_conservative.py`.

## Raw `--help`

```text
usage: tune_low_confidence_threshold.py [-h] --cache-dir CACHE_DIR
                                        --output-dir OUTPUT_DIR
                                        [--thresholds THRESHOLDS]
                                        [--threshold-start THRESHOLD_START]
                                        [--threshold-stop THRESHOLD_STOP]
                                        [--threshold-step THRESHOLD_STEP]
                                        [--max-acc-drop MAX_ACC_DROP]
                                        [--top-k TOP_K]

Tune confidence_threshold for low_confidence_mean_proba using an existing on-
disk logits cache.

options:
  -h, --help            show this help message and exit
  --cache-dir CACHE_DIR
                        Directory containing cache_manifest.json and per-image
                        .npz files.
  --output-dir OUTPUT_DIR
                        Directory where tuning CSV/JSON will be written.
  --thresholds THRESHOLDS
                        Optional comma-separated explicit threshold list. If
                        omitted, the script uses threshold-start/stop/step.
  --threshold-start THRESHOLD_START
                        Start of the threshold sweep, inclusive.
  --threshold-stop THRESHOLD_STOP
                        End of the threshold sweep, inclusive within rounding
                        tolerance.
  --threshold-step THRESHOLD_STEP
                        Step for the threshold sweep.
  --max-acc-drop MAX_ACC_DROP
                        Max allowed pixel-accuracy drop for the 'safe' best
                        threshold.
  --top-k TOP_K         How many top thresholds to include in the summary
                        JSON.
```
