# tune_hybrid_conservative.py

- Тип сценария: `optimization`
- Прямой запуск: **да**
- Файл: `tune_hybrid_conservative.py`

## Назначение

Подбор консервативных параметров hybrid postprocessing.

## Когда использовать

Когда нужно подобрать безопасные параметры hybrid-conservative postprocessing.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 tune_hybrid_conservative.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/hybrid_tune --limit 5 --no-noise
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 tune_hybrid_conservative.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/hybrid_tune --limit 5 --no-noise
```

### Переиспользование уже собранного cache

```bash
python3 tune_hybrid_conservative.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/hybrid_tune --cache-dir artifacts/postprocessing/hybrid_tune/logits_cache --limit 5 --no-noise
```

## Входы / выходы

- Входы: директория изображений, директория масок, готовый on-disk cache, checkpoint / веса модели.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--images IMAGES` | Directory with images. |
| `--masks MASKS` | Directory with masks. |
| `--checkpoint CHECKPOINT` | Path to checkpoint. |
| `--output-dir OUTPUT_DIR` | Directory where tuning CSV/JSON will be written. |
| `--cache-dir CACHE_DIR` | Directory for cached logits/superpixels. Defaults to <output-dir>/logits_cache. |
| `--rebuild-cache` | Ignore existing cache files and rebuild them from the model. |
| `--petroscope-root PETROSCOPE_ROOT` | Optional local petroscope checkout. |
| `--device {auto,cpu,cuda,mps}` | Inference device. |
| `--limit LIMIT` | How many image/mask pairs to evaluate from the start of the test set. |
| `--noise-std NOISE_STD` | Std of Gaussian noise added to the selected tensor. |
| `--no-noise` | Disable checkpoint degradation entirely. Equivalent to running with the original weights regardless of |
| `--noise-std.` | - |
| `--noise-seed NOISE_SEED` | Seed for weight perturbation. |
| `--noise-weight-key NOISE_WEIGHT_KEY` | Checkpoint tensor key to perturb. |
| `--sp-method {felzenszwalb,slic}` | Superpixel algorithm. |
| `--n-segments N_SEGMENTS` | - |
| `--compactness COMPACTNESS` | - |
| `--slic-sigma SLIC_SIGMA` | - |
| `--felz-scale FELZ_SCALE` | - |
| `--felz-sigma FELZ_SIGMA` | - |
| `--felz-min-size FELZ_MIN_SIZE` | - |
| `--pad-align PAD_ALIGN` | - |
| `--patch-size-limit PATCH_SIZE_LIMIT` | - |
| `--patch-size PATCH_SIZE` | - |
| `--patch-stride PATCH_STRIDE` | - |
| `--class-codes CLASS_CODES` | Optional comma-separated class-code mapping. |
| `--unknown-label-policy {error,ignore}` | - |
| `--confidence-thresholds CONFIDENCE_THRESHOLDS` | Comma-separated confidence thresholds to test. |
| `--small-component-sizes SMALL_COMPONENT_SIZES` | Comma-separated component-size limits to test. |
| `--neighbor-ratios NEIGHBOR_RATIOS` | Comma-separated neighbor support ratios to test. |
| `--reference-confidence-threshold REFERENCE_CONFIDENCE_THRESHOLD` | Reference threshold for low_confidence_mean_proba comparisons. |
| `--max-acc-drop MAX_ACC_DROP` | Max allowed pixel-accuracy drop for the 'safe' best config. |
| `--top-k TOP_K` | How many top configs to include in the summary JSON. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: tune_hybrid_conservative.py [-h] --images IMAGES --masks MASKS
                                   --checkpoint CHECKPOINT --output-dir
                                   OUTPUT_DIR [--cache-dir CACHE_DIR]
                                   [--rebuild-cache]
                                   [--petroscope-root PETROSCOPE_ROOT]
                                   [--device {auto,cpu,cuda,mps}]
                                   [--limit LIMIT] [--noise-std NOISE_STD]
                                   [--no-noise] [--noise-seed NOISE_SEED]
                                   [--noise-weight-key NOISE_WEIGHT_KEY]
                                   [--sp-method {felzenszwalb,slic}]
                                   [--n-segments N_SEGMENTS]
                                   [--compactness COMPACTNESS]
                                   [--slic-sigma SLIC_SIGMA]
                                   [--felz-scale FELZ_SCALE]
                                   [--felz-sigma FELZ_SIGMA]
                                   [--felz-min-size FELZ_MIN_SIZE]
                                   [--pad-align PAD_ALIGN]
                                   [--patch-size-limit PATCH_SIZE_LIMIT]
                                   [--patch-size PATCH_SIZE]
                                   [--patch-stride PATCH_STRIDE]
                                   [--class-codes CLASS_CODES]
                                   [--unknown-label-policy {error,ignore}]
                                   [--confidence-thresholds CONFIDENCE_THRESHOLDS]
                                   [--small-component-sizes SMALL_COMPONENT_SIZES]
                                   [--neighbor-ratios NEIGHBOR_RATIOS]
                                   [--reference-confidence-threshold REFERENCE_CONFIDENCE_THRESHOLD]
                                   [--max-acc-drop MAX_ACC_DROP]
                                   [--top-k TOP_K]

Tune hybrid_conservative parameters on a subset while caching logits and
superpixels on disk.

options:
  -h, --help            show this help message and exit
  --images IMAGES       Directory with images.
  --masks MASKS         Directory with masks.
  --checkpoint CHECKPOINT
                        Path to checkpoint.
  --output-dir OUTPUT_DIR
                        Directory where tuning CSV/JSON will be written.
  --cache-dir CACHE_DIR
                        Directory for cached logits/superpixels. Defaults to
                        <output-dir>/logits_cache.
  --rebuild-cache       Ignore existing cache files and rebuild them from the
                        model.
  --petroscope-root PETROSCOPE_ROOT
                        Optional local petroscope checkout.
  --device {auto,cpu,cuda,mps}
                        Inference device.
  --limit LIMIT         How many image/mask pairs to evaluate from the start
                        of the test set.
  --noise-std NOISE_STD
                        Std of Gaussian noise added to the selected tensor.
  --no-noise            Disable checkpoint degradation entirely. Equivalent to
                        running with the original weights regardless of
                        --noise-std.
  --noise-seed NOISE_SEED
                        Seed for weight perturbation.
  --noise-weight-key NOISE_WEIGHT_KEY
                        Checkpoint tensor key to perturb.
  --sp-method {felzenszwalb,slic}
                        Superpixel algorithm.
  --n-segments N_SEGMENTS
  --compactness COMPACTNESS
  --slic-sigma SLIC_SIGMA
  --felz-scale FELZ_SCALE
  --felz-sigma FELZ_SIGMA
  --felz-min-size FELZ_MIN_SIZE
  --pad-align PAD_ALIGN
  --patch-size-limit PATCH_SIZE_LIMIT
  --patch-size PATCH_SIZE
  --patch-stride PATCH_STRIDE
  --class-codes CLASS_CODES
                        Optional comma-separated class-code mapping.
  --unknown-label-policy {error,ignore}
  --confidence-thresholds CONFIDENCE_THRESHOLDS
                        Comma-separated confidence thresholds to test.
  --small-component-sizes SMALL_COMPONENT_SIZES
                        Comma-separated component-size limits to test.
  --neighbor-ratios NEIGHBOR_RATIOS
                        Comma-separated neighbor support ratios to test.
  --reference-confidence-threshold REFERENCE_CONFIDENCE_THRESHOLD
                        Reference threshold for low_confidence_mean_proba
                        comparisons.
  --max-acc-drop MAX_ACC_DROP
                        Max allowed pixel-accuracy drop for the 'safe' best
                        config.
  --top-k TOP_K         How many top configs to include in the summary JSON.
```
