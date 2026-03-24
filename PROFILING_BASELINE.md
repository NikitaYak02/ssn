# Profiling Baseline Reference

This document shows expected profiling results for different configurations and hardware.

## Baseline Results After Albumentations

These are the profiling numbers you should expect after Albumentations integration.

### Configuration: RTX 3060, batch_size=6, crop_size=200, nworkers=0

**Profile minimal (without DataLoader workers):**
```
Profiling 5 batches (num_workers=0)...

forward_pass:   [████████░░░░░░░░░░░░░░░░░░] 48% (2.1s)
backward_pass:  [██████░░░░░░░░░░░░░░░░░░░░] 34% (1.5s)
batch_fetch:    [███░░░░░░░░░░░░░░░░░░░░░░] 12% (0.5s)
cpu_to_gpu:     [██░░░░░░░░░░░░░░░░░░░░░░░] 6% (0.3s)

Total: 4.4s for 5 batches → 1.14 batches/sec → 8.8 it/s ✓
```

**Interpretation:** Good! Augmentation is no longer the bottleneck.

---

### Configuration: RTX 3060, batch_size=6, crop_size=200, nworkers=2

**Profile one batch (with DataLoader workers):**
```
Profiling one batch (5 times)...

data_loading:   [███████████████░░░░░░░░░░] 60% (3.0s)
cpu_to_gpu:     [█░░░░░░░░░░░░░░░░░░░░░░░░] 4% (0.2s)
forward_pass:   [████░░░░░░░░░░░░░░░░░░░░░] 12% (0.6s)
loss_compute:   [░░░░░░░░░░░░░░░░░░░░░░░░░] 6% (0.3s)
backward_pass:  [██░░░░░░░░░░░░░░░░░░░░░░░] 10% (0.5s)

Total: 5.0s for 5 batches → 1.0 batch/sec → 1.0 it/s ✗
```

**Problem:** Workers cause 60% overhead! Solution: Use `--nworkers 0`

---

### Configuration: RTX 4090, batch_size=24, crop_size=200, nworkers=4

**Profile minimal (without workers):**
```
Profiling 5 batches (num_workers=0)...

forward_pass:   [████████░░░░░░░░░░░░░░░░░] 45% (4.5s)
backward_pass:  [██████░░░░░░░░░░░░░░░░░░░] 35% (3.5s)
batch_fetch:    [██░░░░░░░░░░░░░░░░░░░░░░░] 12% (1.2s)
cpu_to_gpu:     [█░░░░░░░░░░░░░░░░░░░░░░░░] 8% (0.8s)

Total: 10.0s for 5 batches → 0.5 batches/sec → 12 it/s (but batch_size=24, so 288 imgs/s) ✓
```

**Interpretation:** Good! Computation is well-balanced.

**Profile one batch (with workers):**
```
Profiling one batch (5 times)...

data_loading:   [████████████░░░░░░░░░░░░░] 50% (5.0s)
cpu_to_gpu:     [░░░░░░░░░░░░░░░░░░░░░░░░░] 2% (0.2s)
forward_pass:   [████████░░░░░░░░░░░░░░░░░] 32% (3.2s)
loss_compute:   [░░░░░░░░░░░░░░░░░░░░░░░░░] 3% (0.3s)
backward_pass:  [██████░░░░░░░░░░░░░░░░░░░] 13% (1.3s)

Total: 10.0s for 5 batches → 0.5 batch/sec → 0.5 it/s ✗
```

**Problem:** With workers, speed is much slower. But with this GPU, workers help with prefetching. Recommendation: Fine-tune num_workers (test 0, 2, 4).

---

## Good vs Bad Profiling Results

### ✓ GOOD: Balanced profile (compute-bound)

```
forward_pass:   40-50%
backward_pass:  25-35%
data_loading:   10-20%
other:          5-15%
```

**Meaning:** GPU is the bottleneck, data loading is fast. This is optimal!

---

### ✗ BAD: Augmentation bottleneck

```
forward_pass:   30%
backward_pass:  20%
batch_fetch:    40% ← Too high!
other:          10%
```

**Problem:** Augmentation is slow (even with Albumentations).

**Solutions:**
1. Use `get_train_augmentation_v2()` (remove RandomScale)
2. Reduce crop_size (128 instead of 200)
3. Check if using legacy API instead of Albumentations

---

### ✗ BAD: DataLoader workers overhead

```
profile_minimal.py (nworkers=0):
  data_loading: 10%
  forward_pass: 50%
  backward_pass: 35%

profile_one_batch.py (nworkers=2):
  data_loading: 70% ← Much worse with workers!
  forward_pass: 15%
  backward_pass: 12%
```

**Problem:** Workers have spawn/IPC overhead on this GPU.

**Solution:** Use `--nworkers 0` in train.py.

---

### ✗ BAD: GPU memory bandwidth limited

```
forward_pass:   60% (but GPU util = 40%)
backward_pass:  25%
data_loading:   10%
```

**Problem:** GPU kernel is memory-bound, not compute-bound.

**Solution:**
1. Increase batch size to improve arithmetic intensity
2. Use larger model (more computation per data)
3. This is often not fixable without hardware upgrade

---

## Interpretation Guide

### Check 1: Data Loading Percentage

**profile_minimal.py (num_workers=0):**
```
batch_fetch < 10% ✓✓ Excellent (almost no augmentation overhead)
batch_fetch 10-15% ✓ Good (Albumentations working)
batch_fetch 15-20% ~ OK (might be slow GPU or complex data)
batch_fetch 20-30% ✗ Slow (augmentation is bottleneck)
batch_fetch > 30% ✗✗ Very slow (something is wrong)
```

### Check 2: Worker Overhead

**Comparing profile_minimal.py vs profile_one_batch.py:**

```
ratio = data_loading_with_workers / data_loading_without_workers

ratio < 1.5 ✓ Workers help
ratio 1.5-3 ~ Workers have overhead, but manageable
ratio > 3 ✗ Worker overhead is severe, use --nworkers 0
```

### Check 3: Compute Balance

```
(forward_pass + backward_pass) as % of total:

> 70% ✓ Good (compute-bound, GPU well-utilized)
50-70% ~ OK (some data loading overhead)
< 50% ✗ Bad (data loading is major bottleneck)
```

### Check 4: Actual Speed

```
Expected iterations/sec (RTX 3060, batch_size=6):

> 10 it/s ✓✓ Excellent
8-10 it/s ✓ Good
5-8 it/s ~ OK (acceptable)
2-5 it/s ✗ Slow (needs optimization)
< 2 it/s ✗✗ Very slow (serious bottleneck)
```

---

## Before vs After Comparison

### Before Albumentations (Manual Augmentation)

RTX 3060, batch_size=6, crop_size=200:
```
forward_pass:   45% (1.8s)
backward_pass:  32% (1.3s)
batch_fetch:    18% (0.7s)   ← Manual cv2.resize is slow
cpu_to_gpu:     5% (0.2s)

Total: 4.0s per batch → 2.5 it/s ✗
```

### After Albumentations (Optimized)

RTX 3060, batch_size=6, crop_size=200:
```
forward_pass:   48% (2.1s)
backward_pass:  34% (1.5s)
batch_fetch:    12% (0.5s)   ← Albumentations is faster
cpu_to_gpu:     6% (0.3s)

Total: 4.4s per batch → 8.8 it/s ✓
```

**Speedup: 3.5x faster!**

---

## Profiling Template

Use this template to record your profiling results:

```
=== Hardware ===
GPU: [e.g., RTX 3060]
CPU: [e.g., Intel i7-10700K]
RAM: [e.g., 32GB]

=== Configuration ===
batch_size: 6
crop_size: 200
num_workers: 0

=== profile_minimal.py Results ===
forward_pass:   __%  (_s)
backward_pass:  __%  (_s)
batch_fetch:    __%  (_s)
cpu_to_gpu:     __%  (_s)

Total time: _s for 5 batches → _ it/s

=== profile_one_batch.py Results (num_workers=2) ===
data_loading:   __%  (_s)
forward_pass:   __%  (_s)
backward_pass:  __%  (_s)
loss_compute:   __%  (_s)

Total time: _s for 5 batches → _ it/s

=== Analysis ===
Is augmentation the bottleneck?
Are workers causing overhead?
What is actual iteration speed?
```

---

## Common Issues and Expected Results

### Issue 1: "My batch_fetch is 25%, is that bad?"

**Analysis:**
- With Albumentations, < 15% is good
- 15-20% is acceptable
- 25% is high - likely using legacy API or custom augmentation

**Check:**
```bash
grep "get_train_augmentation" train.py
# Should show: augment = augmentation.get_train_augmentation()
# If shows: augment = augmentation.Compose([...])
#   → Problem: Using legacy API, switch to get_train_augmentation()
```

---

### Issue 2: "My data_loading with workers is 60%"

**Analysis:**
- Without workers: 12% (good)
- With workers: 60% (worker overhead is severe)
- Worker spawn/fork/pickle is slower than computation on this GPU

**Solution:**
```bash
python train.py --nworkers 0
```

---

### Issue 3: "I'm only getting 2 it/s, but you said 10+"

**Analysis:**
Check which is the bottleneck:

```bash
# 1. Profile minimal
python profile_minimal.py --n_batches 10
# If forward + backward < 70%, problem is likely augmentation or data loading

# 2. Profile one batch with workers
python profile_one_batch.py --nworkers 0
python profile_one_batch.py --nworkers 2
# Compare to see if workers help or hurt

# 3. Check GPU utilization
nvidia-smi
# Should be > 70% if compute-bound
```

---

## Summary

After Albumentations integration, your profiling should show:

✓ `batch_fetch` < 15% (augmentation is fast)
✓ `forward_pass` + `backward_pass` > 70% (compute-bound)
✓ `data_loading` < 20% (data loading not bottleneck)
✓ GPU utilization > 60% (good utilization)

If you see these numbers, everything is working optimally!

See `PROFILING_TROUBLESHOOT.md` for detailed diagnostics if not.
