# precompute_superpixels.py

- Тип сценария: `precompute`
- Прямой запуск: **да**
- Файл: `precompute_superpixels.py`

## Назначение

Предварительный расчет superpixel-аннотаций в формате spanno.

## Когда использовать

Когда хочется заранее закэшировать `spanno` и ускорить повторные интерактивные прогоны.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 precompute_superpixels.py --img_dir /path/to/images --out_dir artifacts/precomputed/full_image_spanno --method slic --n_segments 800 --compactness 20
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 precompute_superpixels.py --img_dir /path/to/images --out_dir artifacts/precomputed/full_image_spanno --method slic --n_segments 800 --compactness 20
```

### SSN-кэш для повторных интерактивных прогонов

```bash
python3 precompute_superpixels.py --img_dir /path/to/images --out_dir artifacts/precomputed/ssn_cache --method ssn --ssn_weights models/checkpoints/best_model.pth --ssn_nspix 100
```

## Входы / выходы

- Входы: директория изображений, опционально checkpoint для neural/ssn methods.
- Выходы: директория результатов, `.spanno.json.gz` файлы.

## Где искать результаты

Ищите `.spanno.json.gz` файлы внутри `--out_dir`.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--img_dir IMG_DIR` | Папка с изображениями |
| `--out_dir OUT_DIR` | Куда сохранять .spanno.json.gz |
| `--method {slic,felzenszwalb,fwb,watershed,ws,ssn,deep_slic,cnn_rim,sp_fcn,sin,rethink_unsup}` | - |
| `--method_config METHOD_CONFIG` | JSON string or path to JSON config for neural methods. |
| `--weights WEIGHTS` | Checkpoint for neural methods (and optional alias for ssn). |
| `--downscale DOWNSCALE` | downscale_coeff для SuperPixelAnnotationAlgo |
| `--overwrite` | Перезаписывать уже существующие .spanno.json.gz |
| `--no_recursive` | Не обходить вложенные директории |
| `--n_segments N_SEGMENTS` | - |
| `--compactness COMPACTNESS` | - |
| `--sigma SIGMA` | - |
| `--scale SCALE` | - |
| `--f_sigma F_SIGMA` | - |
| `--min_size MIN_SIZE` | - |
| `--ws_compactness WS_COMPACTNESS` | - |
| `--ws_components WS_COMPONENTS` | - |
| `--ssn_weights SSN_WEIGHTS` | Чекпоинт SSN (.pth) |
| `--ssn_nspix SSN_NSPIX` | - |
| `--ssn_fdim SSN_FDIM` | - |
| `--ssn_niter SSN_NITER` | - |
| `--ssn_color_scale SSN_COLOR_SCALE` | - |
| `--ssn_pos_scale SSN_POS_SCALE` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.
- Для `spanno`-артефактов размер изображения и размер кэша должны совпадать, иначе будут ошибки совместимости.

## Raw `--help`

```text
usage: precompute_superpixels.py [-h] --img_dir IMG_DIR --out_dir OUT_DIR
                                 [--method {slic,felzenszwalb,fwb,watershed,ws,ssn,deep_slic,cnn_rim,sp_fcn,sin,rethink_unsup}]
                                 [--method_config METHOD_CONFIG]
                                 [--weights WEIGHTS] [--downscale DOWNSCALE]
                                 [--overwrite] [--no_recursive]
                                 [--n_segments N_SEGMENTS]
                                 [--compactness COMPACTNESS] [--sigma SIGMA]
                                 [--scale SCALE] [--f_sigma F_SIGMA]
                                 [--min_size MIN_SIZE]
                                 [--ws_compactness WS_COMPACTNESS]
                                 [--ws_components WS_COMPONENTS]
                                 [--ssn_weights SSN_WEIGHTS]
                                 [--ssn_nspix SSN_NSPIX] [--ssn_fdim SSN_FDIM]
                                 [--ssn_niter SSN_NITER]
                                 [--ssn_color_scale SSN_COLOR_SCALE]
                                 [--ssn_pos_scale SSN_POS_SCALE]

precompute_superpixels.py

Пакетное предварительное разбиение изображений на суперпиксели с сохранением
результата в формат `.spanno.json.gz`.

Это полезно запускать до `evaluate_interactive_annotation.py`, чтобы оценка шла
по уже готовой автоматической сегментации изображения и не возникали странные
объединения разных регионов в один суперпиксель.

Пример:
    python precompute_superpixels.py         --img_dir /data/images         --out_dir /data/spanno         --method slic --n_segments 3000 --compactness 15 --sigma 1.0

options:
  -h, --help            show this help message and exit
  --img_dir IMG_DIR     Папка с изображениями
  --out_dir OUT_DIR     Куда сохранять .spanno.json.gz
  --method {slic,felzenszwalb,fwb,watershed,ws,ssn,deep_slic,cnn_rim,sp_fcn,sin,rethink_unsup}
  --method_config METHOD_CONFIG
                        JSON string or path to JSON config for neural methods.
  --weights WEIGHTS     Checkpoint for neural methods (and optional alias for
                        ssn).
  --downscale DOWNSCALE
                        downscale_coeff для SuperPixelAnnotationAlgo
  --overwrite           Перезаписывать уже существующие .spanno.json.gz
  --no_recursive        Не обходить вложенные директории

Superpixel method:
  --n_segments N_SEGMENTS
  --compactness COMPACTNESS
  --sigma SIGMA
  --scale SCALE
  --f_sigma F_SIGMA
  --min_size MIN_SIZE
  --ws_compactness WS_COMPACTNESS
  --ws_components WS_COMPONENTS
  --ssn_weights SSN_WEIGHTS
                        Чекпоинт SSN (.pth)
  --ssn_nspix SSN_NSPIX
  --ssn_fdim SSN_FDIM
  --ssn_niter SSN_NITER
  --ssn_color_scale SSN_COLOR_SCALE
  --ssn_pos_scale SSN_POS_SCALE
```
