# train_external_superpixels.py

- Тип сценария: `training`
- Прямой запуск: **да**
- Файл: `train_external_superpixels.py`

## Назначение

Обучение внешних superpixel-методов в изолированном upstream-окружении. Сейчас покрыт официальный `SPAM` из статьи *Superpixel Anything*.

## Когда использовать

Когда нужно:

- подтянуть официальный upstream `SPAM`,
- автоматически подготовить BSD-style датасет из пары `img_dir + mask_dir`,
- запустить training без смешивания зависимостей `SPAM` с локальным `requirements.txt`.

## Минимальный запуск

```bash
python3 train_external_superpixels.py \
  --method spam \
  --img_dir /path/to/images \
  --mask_dir /path/to/masks \
  --out_dir artifacts/training/spam \
  --bootstrap \
  --train_iter 5000 \
  --type_model ssn
```

## Что делает скрипт

1. При необходимости конвертирует локальный датасет в BSD-style layout с `groundTruth/*.mat`.
2. При `--bootstrap` клонирует официальный `SPAM` и поднимает отдельное venv-окружение.
3. Собирает и запускает официальный `SPAM/train.py`.
4. Сохраняет `resolved_external_training.json` с точной командой, путями и аргументами.

## Важные аргументы

| Опция | Описание |
| --- | --- |
| `--bsd_root BSD_ROOT` | Уже готовый BSD-style root. |
| `--img_dir IMG_DIR` | Папка с RGB изображениями, если BSD root еще не подготовлен. |
| `--mask_dir MASK_DIR` | Папка с 2-D масками, если BSD root еще не подготовлен. |
| `--prepared_dataset_dir PREPARED_DATASET_DIR` | Куда сохранить конвертированный BSD-style датасет. |
| `--bootstrap` | Клонировать upstream и создать отдельное окружение. |
| `--repo_dir REPO_DIR` | Явно указать путь к upstream `SPAM`. |
| `--venv_dir VENV_DIR` | Явно указать путь к isolated venv. |
| `--dry_run` | Ничего не обучать, только собрать команду и metadata. |
| `--type_model {ssn,resnet50,resnet101,mobilenetv3}` | Trainable variant из официального кода `SPAM`. |
| `--use_sam` | Включить SAM masks during training. |

## Примечания

- Upstream `SPAM` ждёт BSDS-style `.mat`, поэтому обычные PNG-маски скрипт сначала адаптирует.
- В текущем upstream help упоминается `deeplabv3`, но сам кодовый путь реально поддерживает `mobilenetv3`; wrapper экспонирует именно фактически доступные варианты.
- Внешние репозитории и их venv лучше держать в `.external_sources/` и `.method_envs/`.
