# benchmark_simple_superpixel_methods.py

- Тип сценария: `benchmark`
- Прямой запуск: **да**
- Файл: `benchmark_simple_superpixel_methods.py`

## Назначение

Бенчмарк baseline и superpixel-постобработки на наборе изображений.

## Когда использовать

Когда нужно сравнить baseline и несколько стратегий superpixel aggregation/postprocessing.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 benchmark_simple_superpixel_methods.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/benchmark_simple --suite simple --limit 5
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 benchmark_simple_superpixel_methods.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/benchmark_simple --suite simple --limit 5
```

### Safe-suite без деградации checkpoint

```bash
python3 benchmark_simple_superpixel_methods.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/benchmark_safe --suite safe --limit 5 --no-noise
```

## Входы / выходы

- Входы: директория изображений, директория масок, checkpoint / веса модели.
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
| `--output-dir OUTPUT_DIR` | Directory where comparison CSV/JSON will be written. |
| `--petroscope-root PETROSCOPE_ROOT` | Optional local petroscope checkout. |
| `--device {auto,cpu,cuda,mps}` | Inference device. |
| `--limit LIMIT` | How many image/mask pairs to evaluate from the start of the test set. |
| `--suite {simple,novel100,safe,all}` | Which refinement suite to benchmark. |
| `--strategy-limit STRATEGY_LIMIT` | How many generated novel strategies to include when suite is novel100/all. |
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
| `--confidence-threshold CONFIDENCE_THRESHOLD` | Threshold for confidence-gated and low-confidence methods. |
| `--prior-power PRIOR_POWER` | Exponent for prior correction. |
| `--small-component-superpixels SMALL_COMPONENT_SUPERPIXELS` | Max island size in superpixels for cleanup. |
| `--hybrid-neighbor-ratio HYBRID_NEIGHBOR_RATIO` | Minimum neighbor support ratio for hybrid_conservative cleanup. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: benchmark_simple_superpixel_methods.py [-h] --images IMAGES --masks
                                              MASKS --checkpoint CHECKPOINT
                                              --output-dir OUTPUT_DIR
                                              [--petroscope-root PETROSCOPE_ROOT]
                                              [--device {auto,cpu,cuda,mps}]
                                              [--limit LIMIT]
                                              [--suite {simple,novel100,safe,all}]
                                              [--strategy-limit STRATEGY_LIMIT]
                                              [--noise-std NOISE_STD]
                                              [--no-noise]
                                              [--noise-seed NOISE_SEED]
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
                                              [--confidence-threshold CONFIDENCE_THRESHOLD]
                                              [--prior-power PRIOR_POWER]
                                              [--small-component-superpixels SMALL_COMPONENT_SUPERPIXELS]
                                              [--hybrid-neighbor-ratio HYBRID_NEIGHBOR_RATIO]

Benchmark simple superpixel postprocessing methods.

options:
  -h, --help            show this help message and exit
  --images IMAGES       Directory with images.
  --masks MASKS         Directory with masks.
  --checkpoint CHECKPOINT
                        Path to checkpoint.
  --output-dir OUTPUT_DIR
                        Directory where comparison CSV/JSON will be written.
  --petroscope-root PETROSCOPE_ROOT
                        Optional local petroscope checkout.
  --device {auto,cpu,cuda,mps}
                        Inference device.
  --limit LIMIT         How many image/mask pairs to evaluate from the start
                        of the test set.
  --suite {simple,novel100,safe,all}
                        Which refinement suite to benchmark.
  --strategy-limit STRATEGY_LIMIT
                        How many generated novel strategies to include when
                        suite is novel100/all.
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
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Threshold for confidence-gated and low-confidence
                        methods.
  --prior-power PRIOR_POWER
                        Exponent for prior correction.
  --small-component-superpixels SMALL_COMPONENT_SUPERPIXELS
                        Max island size in superpixels for cleanup.
  --hybrid-neighbor-ratio HYBRID_NEIGHBOR_RATIO
                        Minimum neighbor support ratio for hybrid_conservative
                        cleanup.
```
