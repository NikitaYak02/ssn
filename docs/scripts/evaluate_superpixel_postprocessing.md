# evaluate_superpixel_postprocessing.py

- Тип сценария: `postprocessing-eval`
- Прямой запуск: **да**
- Файл: `evaluate_superpixel_postprocessing.py`

## Назначение

Оценка влияния superpixel-постобработки на segmentation baseline.

## Когда использовать

Когда нужно оценить улучшает ли postprocessing baseline или вносит регрессии.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 evaluate_superpixel_postprocessing.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/sp_postproc_eval --device auto
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 evaluate_superpixel_postprocessing.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/sp_postproc_eval --device auto
```

### Без деградации checkpoint и с SLIC

```bash
python3 evaluate_superpixel_postprocessing.py --images /path/to/images --masks /path/to/masks --checkpoint models/checkpoints/S1v2_S2v2_x05.pth --output-dir artifacts/postprocessing/sp_postproc_eval_slic --sp-method slic --n-segments 800 --compactness 20 --no-noise
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
| `--checkpoint CHECKPOINT` | Path to a petroscope HRNet checkpoint (.pth). |
| `--output-dir OUTPUT_DIR` | Where to save metrics and change visualizations. |
| `--petroscope-root PETROSCOPE_ROOT` | Optional root directory of the petroscope repository. Useful if the installed package does not contain HRNet. |
| `--device {auto,cpu,cuda,mps}` | Inference device. auto prefers mps, then cuda, then cpu. |
| `--noise-std NOISE_STD` | Std of Gaussian noise added to the selected tensor. |
| `--no-noise` | Disable checkpoint degradation entirely. Equivalent to running with the original weights regardless of |
| `--noise-std.` | - |
| `--noise-seed NOISE_SEED` | Seed for weight perturbation. |
| `--noise-weight-key NOISE_WEIGHT_KEY` | Checkpoint tensor key to perturb. |
| `--sp-method {felzenszwalb,slic}` | Superpixel algorithm. |
| `--vote-mode {mean_proba,majority_argmax,confidence_gated_mean_proba,low_confidence_mean_proba,prior_corrected_mean_proba,small_region_cleanup,hybrid_conservative}` | How to aggregate predictions inside a superpixel. |
| `--strategy-id STRATEGY_ID` | Optional named refinement strategy. Supports legacy aliases and the generated novel_XX_YY candidate pool. |
| `--list-strategies` | Print available strategy ids and exit. |
| `--confidence-threshold CONFIDENCE_THRESHOLD` | Threshold used by confidence-gated and low-confidence superpixel modes. |
| `--prior-power PRIOR_POWER` | Exponent for prior correction in prior_corrected_mean_proba. |
| `--small-component-superpixels SMALL_COMPONENT_SUPERPIXELS` | Maximum connected-component size in superpixels for small_region_cleanup. |
| `--hybrid-neighbor-ratio HYBRID_NEIGHBOR_RATIO` | Minimum fraction of neighboring boundary support needed to merge a small island in hybrid_conservative. |
| `--n-segments N_SEGMENTS` | SLIC n_segments. |
| `--compactness COMPACTNESS` | SLIC compactness. |
| `--slic-sigma SLIC_SIGMA` | SLIC sigma. |
| `--felz-scale FELZ_SCALE` | Felzenszwalb scale. |
| `--felz-sigma FELZ_SIGMA` | Felzenszwalb sigma. |
| `--felz-min-size FELZ_MIN_SIZE` | Felzenszwalb min_size. |
| `--pad-align PAD_ALIGN` | Pad images to a multiple of this value before inference. |
| `--patch-size-limit PATCH_SIZE_LIMIT` | If H or W is larger than this value, run tiled inference to reduce memory pressure. |
| `--patch-size PATCH_SIZE` | Tile size for large-image inference. |
| `--patch-stride PATCH_STRIDE` | Stride for tiled inference. |
| `--class-codes CLASS_CODES` | Comma-separated raw mask codes in model channel order. Example: 0,1,2,3,4,5,6,7,8,11 |
| `--unknown-label-policy {error,ignore}` | What to do with mask labels absent from class-codes. |
| `--save-predictions` | Also save baseline and postprocessed predictions. |
| `--limit LIMIT` | Only process the first N matched image/mask pairs. |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.

## Raw `--help`

```text
usage: evaluate_superpixel_postprocessing.py [-h] --images IMAGES --masks
                                             MASKS --checkpoint CHECKPOINT
                                             --output-dir OUTPUT_DIR
                                             [--petroscope-root PETROSCOPE_ROOT]
                                             [--device {auto,cpu,cuda,mps}]
                                             [--noise-std NOISE_STD]
                                             [--no-noise]
                                             [--noise-seed NOISE_SEED]
                                             [--noise-weight-key NOISE_WEIGHT_KEY]
                                             [--sp-method {felzenszwalb,slic}]
                                             [--vote-mode {mean_proba,majority_argmax,confidence_gated_mean_proba,low_confidence_mean_proba,prior_corrected_mean_proba,small_region_cleanup,hybrid_conservative}]
                                             [--strategy-id STRATEGY_ID]
                                             [--list-strategies]
                                             [--confidence-threshold CONFIDENCE_THRESHOLD]
                                             [--prior-power PRIOR_POWER]
                                             [--small-component-superpixels SMALL_COMPONENT_SUPERPIXELS]
                                             [--hybrid-neighbor-ratio HYBRID_NEIGHBOR_RATIO]
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
                                             [--save-predictions]
                                             [--limit LIMIT]

Evaluate segmentation quality before and after superpixel postprocessing.

options:
  -h, --help            show this help message and exit
  --images IMAGES       Directory with images.
  --masks MASKS         Directory with masks.
  --checkpoint CHECKPOINT
                        Path to a petroscope HRNet checkpoint (.pth).
  --output-dir OUTPUT_DIR
                        Where to save metrics and change visualizations.
  --petroscope-root PETROSCOPE_ROOT
                        Optional root directory of the petroscope repository.
                        Useful if the installed package does not contain
                        HRNet.
  --device {auto,cpu,cuda,mps}
                        Inference device. auto prefers mps, then cuda, then
                        cpu.
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
  --vote-mode {mean_proba,majority_argmax,confidence_gated_mean_proba,low_confidence_mean_proba,prior_corrected_mean_proba,small_region_cleanup,hybrid_conservative}
                        How to aggregate predictions inside a superpixel.
  --strategy-id STRATEGY_ID
                        Optional named refinement strategy. Supports legacy
                        aliases and the generated novel_XX_YY candidate pool.
  --list-strategies     Print available strategy ids and exit.
  --confidence-threshold CONFIDENCE_THRESHOLD
                        Threshold used by confidence-gated and low-confidence
                        superpixel modes.
  --prior-power PRIOR_POWER
                        Exponent for prior correction in
                        prior_corrected_mean_proba.
  --small-component-superpixels SMALL_COMPONENT_SUPERPIXELS
                        Maximum connected-component size in superpixels for
                        small_region_cleanup.
  --hybrid-neighbor-ratio HYBRID_NEIGHBOR_RATIO
                        Minimum fraction of neighboring boundary support
                        needed to merge a small island in hybrid_conservative.
  --n-segments N_SEGMENTS
                        SLIC n_segments.
  --compactness COMPACTNESS
                        SLIC compactness.
  --slic-sigma SLIC_SIGMA
                        SLIC sigma.
  --felz-scale FELZ_SCALE
                        Felzenszwalb scale.
  --felz-sigma FELZ_SIGMA
                        Felzenszwalb sigma.
  --felz-min-size FELZ_MIN_SIZE
                        Felzenszwalb min_size.
  --pad-align PAD_ALIGN
                        Pad images to a multiple of this value before
                        inference.
  --patch-size-limit PATCH_SIZE_LIMIT
                        If H or W is larger than this value, run tiled
                        inference to reduce memory pressure.
  --patch-size PATCH_SIZE
                        Tile size for large-image inference.
  --patch-stride PATCH_STRIDE
                        Stride for tiled inference.
  --class-codes CLASS_CODES
                        Comma-separated raw mask codes in model channel order.
                        Example: 0,1,2,3,4,5,6,7,8,11
  --unknown-label-policy {error,ignore}
                        What to do with mask labels absent from class-codes.
  --save-predictions    Also save baseline and postprocessed predictions.
  --limit LIMIT         Only process the first N matched image/mask pairs.
```
