# Training Pipeline Profiling

Встроенный профайлер отслеживает время каждого этапа тренировки. Это помогает выявить реальные узкие места.

## Быстрый профайлинг (5 батчей)

Запустить профайлинг одного батча без полной тренировки:

```bash
python profile_one_batch.py \
    --img_dir /path/to/images \
    --mask_dir /path/to/masks \
    --batchsize 6 \
    --nworkers 2 \
    --crop_size 200
```

Результат покажет время каждого этапа (ms per batch):

```
PROFILER REPORT (seconds):
  forward_pass             5.42 s  (    5 calls)  45.2%
  backward_pass            4.21 s  (    5 calls)  35.1%
  cpu_to_gpu               1.08 s  (    5 calls)  9.0%
  loss_compute             0.87 s  (    5 calls)  7.2%
  data_loading             0.34 s  (    5 calls)  2.8%
  ...
```

## Профайлинг полной тренировки

При запуске `train.py` профайлер включен автоматически. Отчёты выводятся:
- **Каждые 1000 итераций** (по умолчанию `print_interval * 10`)
- **При завершении тренировки** (финальный отчёт всех этапов)

Пример вывода в процессе тренировки:

```
[iter 1000] Profiling (seconds):
  forward_pass             28.51 s  ( 1000 calls)  47.8%
  backward_pass            21.34 s  ( 1000 calls)  35.8%
  data_loading              5.82 s  ( 1000 calls)  9.8%
  loss_compute              2.15 s  ( 1000 calls)  3.6%
  TOTAL                    59.53 s
```

## Интерпретация результатов

### Норма:
- **Forward pass**: 40-50% времени (основное вычисление)
- **Backward pass**: 30-40% времени (обратное распространение)
- **Data loading**: < 10% времени (в-памятности датасет)
- **Loss compute**: < 10% времени (маленькие операции)

### Что-то не так, если:

#### `forward_pass` > 60% или очень медленный
- **Причины**: модель не скомпилирована, нет AMP, GPU busy с другими процессами
- **Решение**:
  ```bash
  # Проверить компиляцию
  torch.compile(model)  # PyTorch >= 2.0

  # Или увеличить batch size
  --batchsize 12  # вместо 6
  ```

#### `backward_pass` > 50%
- **Причины**: нет gradients accumulation, неэффективный backward
- **Решение**: обычное явление, backward дороже forward в 0.8-1.0x раз

#### `data_loading` > 20%
- **Причины**: недостаточно workers, слишком много I/O (но датасет в-памяти!)
- **Решение**:
  ```bash
  # Увеличить workers
  --nworkers 4

  # Или уменьшить (если слишком много overhead)
  --nworkers 0
  ```

#### `cpu_to_gpu` > 10%
- **Причины**: большой батч, отсутствует pin_memory
- **Решение**: обычно исправляется автоматически, проверить:
  ```python
  # В train.py
  pin_memory=(device == "cuda"),  # должно быть True
  ```

#### `loss_compute` > 15%
- **Причины**: слишком много уникальных классов в маске, неоптимальная loss
- **Решение**:
  ```bash
  # Уменьшить max_classes если не все используются
  --max_classes 30
  ```

## Параметры для оптимизации

### GPU утилизация
```bash
# Увеличить batch size (если хватает VRAM)
--batchsize 12

# Включить torch.compile (PyTorch >= 2.0)
# Автоматически включается в train.py
```

### Данные
```bash
# Уменьшить размер кропа для быстрой тренировки
--crop_size 128

# Увеличить workers для параллельной augmentation
--nworkers 4
```

### Модель
```bash
# Уменьшить размер embedding для быстрого prototyping
--fdim 10

# Уменьшить итерации SLIC
--niter 3
```

## Утилиты

### Запустить профайлинг и сохранить отчёт
```bash
python profile_one_batch.py \
    --img_dir ... --mask_dir ... \
    | tee profile_report.txt
```

### Сравнить разные конфигурации
```bash
# Конфиг 1
python profile_one_batch.py --batchsize 6 --nworkers 2

# Конфиг 2
python profile_one_batch.py --batchsize 12 --nworkers 0
```

## Примечания

- Первый батч может быть медленнее (компиляция CUDA kernel, инициализация GPU)
- С `torch.compile` первое выполнение долгое (граф компилируется), далее быстро
- Времена зависят от GPU и процессора
- На разных машинах результаты могут сильно отличаться

## Типичные значения (NVIDIA RTX 3060, batch=6, crop=200×200)

```
forward_pass:     ~27 ms  (CNN + SLIC)
backward_pass:    ~21 ms  (gradients)
loss_compute:     ~2 ms   (cross-entropy + mse)
data_loading:     ~5 ms   (augmentation in RAM)
cpu_to_gpu:       ~3 ms   (transfer)
────────────────
ИТОГО:            ~60 ms/batch → ~16 it/s
```

С оптимизацией (batch=12, torch.compile, num_workers=0):
```
ИТОГО:            ~110 ms/batch → ~9 it/s  (but 2x больше данных за итерацию)
```
