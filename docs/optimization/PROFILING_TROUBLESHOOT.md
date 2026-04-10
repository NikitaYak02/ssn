# Профайлинг: диагностика и исправление

Если `data_loading` > 20%, это проблема. Вот как это исправить.

## Диагностика: два уровня профайлинга

### Уровень 1: С DataLoader workers (может быть медленно)
```bash
python profile_one_batch.py --img_dir ... --mask_dir ... --nworkers 2
```

### Уровень 2: Чистая компутация (БЕЗ workers)
```bash
python profile_minimal.py --img_dir ... --mask_dir ...
```

## Интерпретация результатов

### Сценарий 1: profile_minimal.py медленный, profile_one_batch.py медленнее
```
profile_minimal.py:
  forward_pass: 45%
  backward_pass: 35%
  data_loading: 10%  ← НОРМАЛЬНО

profile_one_batch.py с --nworkers 2:
  forward_pass: 15%
  backward_pass: 11%
  data_loading: 70%  ← ПРОБЛЕМА!
```

**Вывод:** Проблема в DataLoader/workers, не в компутации.

**Решение:**
```bash
# Вариант 1: Отключить workers
python train.py --nworkers 0 ...

# Вариант 2: Использовать меньше workers
python train.py --nworkers 1 ...
```

### Сценарий 2: profile_minimal.py быстрый, profile_one_batch.py быстрый
```
profile_minimal.py:
  forward_pass: 45%
  backward_pass: 35%
  data_loading: 10%

profile_one_batch.py с --nworkers 2:
  forward_pass: 40%
  backward_pass: 32%
  data_loading: 15%  ← OK!
```

**Вывод:** Всё работает оптимально.

**Рекомендация:** Использовать `--nworkers 2` в train.py.

### Сценарий 3: Даже profile_minimal.py медленный
```
profile_minimal.py:
  forward_pass: 40%
  backward_pass: 30%
  data_loading: 20%  ← СЛИШКОМ МНОГО
```

**Вывод:** Проблема в augmentation-е (в основном `RandomScale` с `cv2.resize`).

**Решение:**

Вариант 1: Отключить RandomScale
```python
augment = augmentation.Compose([
    augmentation.RandomHorizontalFlip(),
    augmentation.RandomVerticalFlip(),
    # augmentation.RandomScale(),  ← ОТКЛЮЧИТЬ
    augmentation.RandomCrop(crop_size=(cfg.crop_size, cfg.crop_size)),
])
```

Вариант 2: Уменьшить crop_size
```bash
python train.py --crop_size 128 ...  # вместо 200
```

Вариант 3: Переместить RandomScale в более дешёвый формат
```python
# Перед: RGB uint8 → LAB float32 → Scale → Crop
# После: RGB uint8 → Scale → LAB float32 → Crop
```

## Пошаговый план диагностики

### Шаг 1: Запустить оба профайлера
```bash
echo "=== Profile minimal (no workers) ==="
python profile_minimal.py --img_dir ... --mask_dir ... --n_batches 5

echo "=== Profile with workers ==="
python profile_one_batch.py --img_dir ... --mask_dir ... --nworkers 2
```

### Шаг 2: Посмотреть результаты
Если `data_loading` в `profile_minimal.py` < 15%, а в `profile_one_batch.py` > 40%:
→ Проблема в DataLoader workers, используй `--nworkers 0`

Если `data_loading` в обоих > 20%:
→ Проблема в augmentation-е (RandomScale), отключи его

### Шаг 3: Переопробовать
```bash
# После изменений
python profile_minimal.py --img_dir ... --mask_dir ...
```

## Типичные проблемы и решения

### Проблема 1: DataLoader workers медленные
**Симптомы:**
```
profile_one_batch.py: data_loading 70%
profile_minimal.py: data_loading 10%
```

**Причины:**
- Spawning/forking overhead
- IPC (inter-process communication) overhead
- Python pickle/unpickle при передаче батчей

**Решение:**
```bash
# Вариант 1: Отключить workers совсем
train.py --nworkers 0

# Вариант 2: Использовать 1 worker (минимальный overhead)
train.py --nworkers 1
```

### Проблема 2: RandomScale медленный
**Симптомы:**
```
profile_minimal.py: data_loading 20-30%
```

**Причины:**
- `cv2.resize` на float32 LAB изображениях медленный
- Особенно если random scale фактор далеко от 1.0

**Решение:**

Вариант 1: Отключить
```python
augment = augmentation.Compose([
    augmentation.RandomHorizontalFlip(),
    augmentation.RandomVerticalFlip(),
    # augmentation.RandomScale(),  # удалить эту строку
    augmentation.RandomCrop(...),
])
```

Вариант 2: Ускорить (использовать меньший диапазон scale)
```python
augmentation.RandomScale(scale_range=(0.9, 1.1))  # вместо (0.75, 3.0)
```

Вариант 3: Переместить RandomScale перед LAB конвертацией
Отредактировать `custom_dataset.py`:
```python
def __getitem__(self, idx):
    img = self.images[idx]      # RGB uint8
    mask = self.masks[idx]       # uint8

    # geo_transforms (flip, scale, crop) - теперь на RGB uint8, быстрее!
    if self.geo_transforms is not None:
        img, mask = self.geo_transforms([img, mask])

    # LAB конверсия ПОСЛЕ augmentation - на маленьком cropped image
    img_lab = rgb2lab(img).astype(np.float32)
    ...
```

### Проблема 3: Batch fetch медленный
**Симптомы:**
```
profile_minimal.py: batch_fetch 30%
```

**Причины:**
- Dataset слишком медленно возвращает батч
- Проблема в `__getitem__` (может быть Image.open если датасет не в-памяти)

**Решение:**
Убедиться что используется `InMemorySegmentationDataset`, а не старый `SegmentationDataset`:
```python
# Правильно
train_dataset = InMemorySegmentationDataset(...)

# Неправильно
train_dataset = SegmentationDataset(...)  # это disk I/O!
```

## Быстрая чек-лист

Если training медленный:

```
□ Запустить profile_minimal.py
  └─ Если forward/backward < 70% → OK, проблема в data loading
  └─ Если forward + backward > 70% → проблема в GPU/модели

□ Запустить profile_one_batch.py с разными --nworkers
  └─ --nworkers 0:
  └─ --nworkers 1:
  └─ --nworkers 2:
  └─ --nworkers 4:
  Выбрать тот что дает лучший результат

□ Если data_loading > 30% даже в profile_minimal.py:
  □ Отключить RandomScale
  □ Уменьшить crop_size до 128
  □ Переместить Scale перед LAB конверсией

□ Переопробовать с оптимальной конфигурацией:
  python train.py \
    --img_dir ... --mask_dir ... \
    --nworkers 0 \  # или найденное значение
    --train_iter 500000
```

## Примеры реальных случаев

### Случай 1: GPU слабый, но можно оптимизировать (RTX 3060)
```
profile_minimal.py (batch=6):
  forward_pass: 2.5s (45%)
  backward_pass: 1.8s (32%)
  batch_fetch: 1.2s (22%)  ← слишком много!

Решение: Отключить RandomScale
→ batch_fetch: 0.3s (8%)
→ Общее время: 4.2s → 4.8 it/s (было ~3 it/s)
```

### Случай 2: Workers добавляют overhead (any GPU)
```
profile_minimal.py (nworkers=0):
  data_loading: 8%

profile_one_batch.py (nworkers=2):
  data_loading: 65%

Решение: Использовать --nworkers 0 в train.py
→ 3x ускорение для этой конфигурации
```

### Случай 3: Оптимальная конфигурация (RTX 4090)
```
profile_minimal.py (batch=24, nworkers=0):
  forward_pass: 18%
  backward_pass: 14%
  data_loading: 5%

profile_one_batch.py (batch=24, nworkers=4):
  forward_pass: 45%
  backward_pass: 35%
  data_loading: 12%

Результат: ~20-25 it/s (GPU fully utilized)
```

## Параметры для экспериментов

**Для быстрого prototyping (экономить время тренировки):**
```bash
python train.py \
    --crop_size 128 \         # меньший кроп
    --nworkers 0 \            # нет overhead
    --niter 3 \               # меньше SLIC итераций
    --fdim 10 \               # меньше features
    --batchsize 3             # маленький батч
```

**Для максимальной производительности:**
```bash
python benchmark_configs.py --img_dir ... --mask_dir ...
# Использовать рекомендуемую конфигурацию
```

## Debugging checklist

- [ ] Используется `InMemorySegmentationDataset` (не старый код)?
- [ ] Датасет предзагружается с `verbose=True` при инициализации?
- [ ] `profile_minimal.py` без workers даёт разумные цифры?
- [ ] Экспериментировал с `--nworkers 0`, 1, 2, 4?
- [ ] Проверил что RandomScale действительно медленный?
- [ ] Запустил `benchmark_configs.py` чтобы найти оптимум?

## Что дальше

1. Найти оптимальную конфигурацию через benchmarking
2. Запустить training с этой конфигурацией
3. Периодически смотреть на профайлер (выводится каждые 1000 итераций)
4. Если что-то изменилось (медленнее), переопробовать
