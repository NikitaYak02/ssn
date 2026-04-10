# evaluate_interactive_annotation.py

- Тип сценария: `interactive-eval`
- Прямой запуск: **да**
- Файл: `evaluate_interactive_annotation.py`

## Назначение

Оценка интерактивной аннотации со штрихами и подробными метриками.

## Когда использовать

Когда нужно измерить, как быстро и качественно растет разметка при добавлении штрихов.

## Обязательные зависимости

- Python 3.12+ и зависимости проекта из `requirements.txt`.
- Для большинства evaluation/benchmark-скриптов удобнее использовать `superpixel_annotator/superpixel_annotator_venv/bin/python` или эквивалентное подготовленное окружение.
- Подробная памятка по окружениям: [run_environments.md](run_environments.md).

## Минимальный запуск

```bash
python3 evaluate_interactive_annotation.py --image /path/to/image.png --mask /path/to/mask.png --out artifacts/interactive_runs/demo_single --method slic --n_segments 800 --compactness 20 --scribbles 100 --save_every 20
```

## Типовые сценарии

### Минимальный запуск

```bash
python3 evaluate_interactive_annotation.py --image /path/to/image.png --mask /path/to/mask.png --out artifacts/interactive_runs/demo_single --method slic --n_segments 800 --compactness 20 --scribbles 100 --save_every 20
```

### SSN с заранее подготовленным spanno

```bash
python3 evaluate_interactive_annotation.py --image /path/to/image.png --mask /path/to/mask.png --spanno /path/to/cache.spanno.json.gz --out artifacts/interactive_runs/demo_ssn --method ssn --ssn_weights models/checkpoints/best_model.pth --scribbles 100 --save_every 20
```

## Входы / выходы

- Входы: директория изображений, директория масок, одно изображение, одна GT-маска, checkpoint / веса модели, готовый spanno/state артефакт.
- Выходы: директория результатов.

## Где искать результаты

Если есть `--out`, `--output-dir` или `--out_dir`, все ключевые артефакты окажутся там; иначе ориентируйтесь на stdout и соседние файлы входного сценария.

## Ключевые аргументы

| Опция | Описание |
| --- | --- |
| `-h, --help` | show this help message and exit |
| `--out OUT` | Директория для результатов |
| `--num_classes NUM_CLASSES` | Число классов (default — авто из DEFAULT_CLASS_INFO или GT уникальных) |
| `--image IMAGE` | Путь к одному RGB-изображению |
| `--mask MASK` | Путь к GT-маске (одно изображение) |
| `--img_dir IMG_DIR` | Директория с изображениями (пакетный режим) |
| `--mask_dir MASK_DIR` | Директория с масками (пакетный режим) |
| `--spanno SPANNO` | Заранее вычисленный .spanno.json[.gz] (одиночный режим). Рекомендуется: сначала автоматически сегментировать изображение суперпикселями, чтобы избежать странных объединений разных регионов. |
| `--downscale DOWNSCALE` | Коэффициент уменьшения изображения для algo (default 1.0) |
| `--scribbles SCRIBBLES` | Максимум штрихов |
| `--save_every SAVE_EVERY` | Checkpoint каждые N штрихов |
| `--seed SEED` | - |
| `--margin MARGIN` | Отступ от границы GT (пиксели) |
| `--border_margin BORDER_MARGIN` | Минимальный отступ штриха от границы bad-region (пиксели) |
| `--no_overlap` | Запрет перекрытия новых штрихов с ранее нанесёнными |
| `--max_no_progress MAX_NO_PROGRESS` | Ранний стоп после N штрихов подряд без прогресса |
| `--region_selection_cycle REGION_SELECTION_CYCLE` | Comma-separated cycle of region selection modes for new scribbles. Supported: miou_gain, largest_error, unannotated. |
| `--method {slic,felzenszwalb,fwb,watershed,ws,ssn,deep_slic,cnn_rim,sp_fcn,sin,rethink_unsup}` | - |
| `--method_config METHOD_CONFIG` | JSON string or path to JSON config for neural methods. |
| `--weights WEIGHTS` | Checkpoint for neural methods (and optional alias for ssn). |
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
| `--sensitivity SENSITIVITY` | Чувствительность BFS-распространения (0 = выкл., default 1.8) |
| `--emb_weights EMB_WEIGHTS` | Чекпоинт для эмбединг-пропагации (.pth). Если задан, использует cosine-similarity вместо LAB |
| `--emb_threshold EMB_THRESHOLD` | Порог косинусного сходства (default 0.988) |
| `--no_borders` | - |
| `--no_annos` | - |
| `--no_scribbles` | - |

## Типичные ошибки

- Проверьте, что запускаете скрипт из активированного окружения с установленными зависимостями.
- Если используется checkpoint, убедитесь, что путь к весам существует и соответствует ожидаемой архитектуре.
- Если аргументы не совпадают с документацией, сверяйтесь с блоком `Raw --help` ниже: он собран из реального parser'а.
- Для `spanno`-артефактов размер изображения и размер кэша должны совпадать, иначе будут ошибки совместимости.

## Raw `--help`

```text
usage: evaluate_interactive_annotation.py [-h] [--image IMAGE] [--mask MASK]
                                          [--img_dir IMG_DIR]
                                          [--mask_dir MASK_DIR]
                                          [--spanno SPANNO]
                                          [--downscale DOWNSCALE] --out OUT
                                          [--scribbles SCRIBBLES]
                                          [--save_every SAVE_EVERY]
                                          [--seed SEED] [--margin MARGIN]
                                          [--border_margin BORDER_MARGIN]
                                          [--no_overlap]
                                          [--max_no_progress MAX_NO_PROGRESS]
                                          [--region_selection_cycle REGION_SELECTION_CYCLE]
                                          [--method {slic,felzenszwalb,fwb,watershed,ws,ssn,deep_slic,cnn_rim,sp_fcn,sin,rethink_unsup}]
                                          [--method_config METHOD_CONFIG]
                                          [--weights WEIGHTS]
                                          [--n_segments N_SEGMENTS]
                                          [--compactness COMPACTNESS]
                                          [--sigma SIGMA] [--scale SCALE]
                                          [--f_sigma F_SIGMA]
                                          [--min_size MIN_SIZE]
                                          [--ws_compactness WS_COMPACTNESS]
                                          [--ws_components WS_COMPONENTS]
                                          [--ssn_weights SSN_WEIGHTS]
                                          [--ssn_nspix SSN_NSPIX]
                                          [--ssn_fdim SSN_FDIM]
                                          [--ssn_niter SSN_NITER]
                                          [--ssn_color_scale SSN_COLOR_SCALE]
                                          [--ssn_pos_scale SSN_POS_SCALE]
                                          [--sensitivity SENSITIVITY]
                                          [--emb_weights EMB_WEIGHTS]
                                          [--emb_threshold EMB_THRESHOLD]
                                          [--no_borders] [--no_annos]
                                          [--no_scribbles]
                                          [--num_classes NUM_CLASSES]

evaluate_interactive_annotation.py

Пайплайн автоматического проставления штрихов с детальными метриками
качества аннотации относительно стоимости взаимодействий пользователя.

Метрики качества:
    mIoU                    — средний IoU по классам
    per-class IoU           — IoU для каждого класса
    coverage                — доля размеченных пикселей от общего числа
    annotation_precision    — доля правильных пикселей среди размеченных

Метрики стоимости взаимодействия:
    n_scribbles             — число нанесённых штрихов
    total_ink_px            — суммарная длина всех штрихов (пиксели)
    mean_ink_px             — средняя длина одного штриха (пиксели)
    per_class_n_scribbles   — штрихов по каждому классу
    per_class_ink_px        — длина штрихов по каждому классу

Метрики эффективности (качество / стоимость):
    miou_per_scribble       — mIoU делённый на число штрихов
    miou_per_1kpx           — mIoU на 1 000 пикселей суммарной длины
    correct_px_per_scribble — правильно размеченных пикселей на штрих
    correct_px_per_px_ink   — правильно размеченных пикселей на пиксель длины
    quality_x_coverage      — mIoU × coverage (совместная метрика)

Пример запуска (одно изображение):
    python evaluate_interactive_annotation.py \
        --image img.png --mask gt.png --out results/ \
        --method slic --n_segments 3000 --compactness 15 \
        --sensitivity 2.0 --scribbles 500 --save_every 50

Пакетный режим (директория изображений):
    python evaluate_interactive_annotation.py \
        --img_dir /data/images --mask_dir /data/masks --out results/ \
        --method ssn --ssn_weights model.pth \
        --sensitivity 1.5 --scribbles 300 --save_every 50

Флаги пропагации:
    --sensitivity 0.0      — отключить распространение штрихов (только прямые аннотации)
    --sensitivity 1.8      — рекомендуемый консервативный режим для SSN+embeddings
    --emb_weights path.pth — использовать cosine-similarity по эмбедингам вместо LAB-цвета
    --emb_threshold 0.988  — строгий порог косинусного сходства для аккуратного propagation

Выбор региона для нового штриха:
    --region_selection_cycle miou_gain,largest_error,unannotated
                          — чередовать стратегии по шагам внутри одной разметки

options:
  -h, --help            show this help message and exit
  --out OUT             Директория для результатов
  --num_classes NUM_CLASSES
                        Число классов (default — авто из DEFAULT_CLASS_INFO
                        или GT уникальных)

Input:
  --image IMAGE         Путь к одному RGB-изображению
  --mask MASK           Путь к GT-маске (одно изображение)
  --img_dir IMG_DIR     Директория с изображениями (пакетный режим)
  --mask_dir MASK_DIR   Директория с масками (пакетный режим)
  --spanno SPANNO       Заранее вычисленный .spanno.json[.gz] (одиночный
                        режим). Рекомендуется: сначала автоматически
                        сегментировать изображение суперпикселями, чтобы
                        избежать странных объединений разных регионов.
  --downscale DOWNSCALE
                        Коэффициент уменьшения изображения для algo (default
                        1.0)

Simulation:
  --scribbles SCRIBBLES
                        Максимум штрихов
  --save_every SAVE_EVERY
                        Checkpoint каждые N штрихов
  --seed SEED
  --margin MARGIN       Отступ от границы GT (пиксели)
  --border_margin BORDER_MARGIN
                        Минимальный отступ штриха от границы bad-region
                        (пиксели)
  --no_overlap          Запрет перекрытия новых штрихов с ранее нанесёнными
  --max_no_progress MAX_NO_PROGRESS
                        Ранний стоп после N штрихов подряд без прогресса
  --region_selection_cycle REGION_SELECTION_CYCLE
                        Comma-separated cycle of region selection modes for
                        new scribbles. Supported: miou_gain, largest_error,
                        unannotated.

Superpixel method:
  --method {slic,felzenszwalb,fwb,watershed,ws,ssn,deep_slic,cnn_rim,sp_fcn,sin,rethink_unsup}
  --method_config METHOD_CONFIG
                        JSON string or path to JSON config for neural methods.
  --weights WEIGHTS     Checkpoint for neural methods (and optional alias for
                        ssn).
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

Propagation:
  --sensitivity SENSITIVITY
                        Чувствительность BFS-распространения (0 = выкл.,
                        default 1.8)
  --emb_weights EMB_WEIGHTS
                        Чекпоинт для эмбединг-пропагации (.pth). Если задан,
                        использует cosine-similarity вместо LAB
  --emb_threshold EMB_THRESHOLD
                        Порог косинусного сходства (default 0.988)

Visualization:
  --no_borders
  --no_annos
  --no_scribbles

Hint: before interactive evaluation it is preferable to pre-segment the image into superpixels (for example, save and pass --spanno). This reduces strange merging of different regions into the same superpixel and makes scribble-based evaluation more stable.
```
