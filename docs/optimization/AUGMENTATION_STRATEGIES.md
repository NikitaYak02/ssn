# Augmentation Strategy Reference

Quick reference for different augmentation strategies depending on your profiling results.

## 1. Full Augmentation (Recommended Default)

Use when data_loading is < 20%.

```python
from lib.dataset import augmentation

augment = augmentation.get_train_augmentation(
    crop_size=200,
    scale_range=(0.75, 3.0)  # Random scale from 0.75x to 3.0x
)
```

**What it does:**
- Horizontal flip (50% chance)
- Vertical flip (50% chance)
- Random scale (0.75x to 3.0x)
- Random crop to 200x200
- Normalize (no-op, dummy)

**When to use:**
- Default for all training
- Provides good data augmentation diversity
- Albumentations makes it fast (~2-3x faster than manual)

**Training command:**
```bash
python train.py --crop_size 200 --nworkers 0 --train_iter 500000
```

---

## 2. Minimal Augmentation (Fast)

Use when data_loading is still > 20% even with full augmentation.

```python
from lib.dataset import augmentation

augment = augmentation.get_train_augmentation_v2(crop_size=200)
```

**What it does:**
- Horizontal flip (50% chance)
- Vertical flip (50% chance)
- Random crop to 200x200
- ~~Random scale~~ (removed)

**When to use:**
- If profiling shows batch_fetch > 20% even with Albumentations
- When speed is more important than augmentation diversity
- For quick prototyping/debugging

**Expected improvement:**
- Removes slowest augmentation (RandomScale with cv2.resize)
- Should drop batch_fetch from 12% to 8-10%

**Training command:**
```bash
python train.py --crop_size 128 --nworkers 0 --train_iter 100000
```

---

## 3. No Workers, Legacy Augmentation (Debug Only)

Use ONLY for debugging/comparison.

```python
from lib.dataset import augmentation

augment = augmentation.Compose([
    augmentation.RandomHorizontalFlip(),
    augmentation.RandomVerticalFlip(),
    augmentation.RandomScale(),
    augmentation.RandomCrop(crop_size=(200, 200)),
])
```

**What it does:**
- Same as full augmentation
- BUT uses manual OpenCV code instead of Albumentations
- ~2-3x slower

**When to use:**
- Comparing to old baseline
- Debugging augmentation issues
- Never for actual training (use full Albumentations instead)

**Training command:**
```bash
python train.py --crop_size 200 --nworkers 0 --train_iter 1000
```

---

## 4. Custom Augmentation

For advanced users: modify augmentation.py to add/remove specific transforms.

```python
def get_train_augmentation_custom(crop_size=200):
    """Your custom augmentation pipeline."""
    import albumentations as A

    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.2),  # Add noise
        A.GaussBlur(blur_limit=3, p=0.1),  # Slight blur
        # A.RandomScale(...),  # Uncomment if you want it
        A.RandomCrop(height=crop_size, width=crop_size),
    ])

    return augmentation.ComposeCustom(transforms)
```

---

## Choosing the Right Strategy

### Decision Tree

```
Does profiling show data_loading < 15%?
├─ YES → Use get_train_augmentation() (full, default)
└─ NO → Go to next question

Does profiling show data_loading < 20%?
├─ YES → Use get_train_augmentation() (still good)
└─ NO → Go to next question

Is data_loading 20-30%?
├─ YES → Might be GPU kernel issue, not augmentation
│        (Check forward_pass and backward_pass %)
└─ NO (> 30%) → Use get_train_augmentation_v2() (no scale)

Still slow with v2?
├─ YES → Problem is elsewhere (DataLoader workers?)
│        Try --nworkers 0
└─ NO → Problem solved!
```

---

## Profiling Results by Strategy

### Full Augmentation (Default)
```
With profile_minimal.py (num_workers=0):
  forward_pass:   48%
  backward_pass:  34%
  batch_fetch:    12%  ← Augmentation time
  cpu_to_gpu:     6%
```

### Minimal Augmentation (No Scale)
```
With profile_minimal.py (num_workers=0):
  forward_pass:   48%
  backward_pass:  34%
  batch_fetch:    9%   ← Faster!
  cpu_to_gpu:     9%
```

### With DataLoader Workers (comparison)
```
With profile_one_batch.py (num_workers=2):
  data_loading:   65%  ← Workers cause overhead!

With --nworkers 0:
  data_loading:   12%  ← No worker overhead
```

---

## Real-World Examples

### Example 1: RTX 3060, slow training

**Problem:** Only ~8 it/s

**Diagnosis:**
```bash
$ python profile_minimal.py --crop_size 200
batch_fetch: 20%  ← Too high!
```

**Solution:** Use minimal augmentation
```python
augment = augmentation.get_train_augmentation_v2(crop_size=200)
```

**Result:** 12-15 it/s ✓

---

### Example 2: RTX 4090, underutilized

**Problem:** GPU only 40% utilized

**Diagnosis:**
```bash
$ python profile_minimal.py --crop_size 200 --batchsize 24
batch_fetch: 8%   ← Good
forward_pass: 50% ← GPU kernel is slow
```

**Solution:** Increase batch size or model complexity
```bash
python train.py --batchsize 48 --nworkers 4 --crop_size 200
```

**Result:** 60-70 it/s, 90% GPU utilization ✓

---

### Example 3: Decent GPU, worker overhead

**Problem:** 10 it/s with workers, but should be faster

**Diagnosis:**
```bash
$ python profile_minimal.py --nworkers 0
data_loading: 10%

$ python profile_one_batch.py --nworkers 2
data_loading: 50%  ← Worker overhead!
```

**Solution:** Disable workers
```bash
python train.py --nworkers 0 --crop_size 200
```

**Result:** 18-22 it/s (worker overhead gone) ✓

---

## Summary Table

| Strategy | Speed | Diversity | When | Command |
|----------|-------|-----------|------|---------|
| Full (A.Compose) | ~12 it/s | Good | Default | `get_train_augmentation()` |
| Minimal (No scale) | ~15 it/s | OK | Slow GPU | `get_train_augmentation_v2()` |
| Legacy (manual) | ~4 it/s | Good | Debug only | `Compose([...])` |
| Custom | Varies | Best | Advanced | Edit augmentation.py |

---

## Next Steps

1. **Run profile_minimal.py** with your data to see current bottleneck
2. **Choose strategy** based on profiling results
3. **Update train.py** if needed (change augmentation function)
4. **Run training** and monitor it/s with profiler output every 1000 iterations
5. **Adjust hyperparameters** based on iteration speed and validation metrics

See `TESTING_ALBUMENTATIONS.md` for detailed testing instructions.
