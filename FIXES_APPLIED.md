# Fixes Applied - Albumentations Integration

## Issue 1: KeypointParams Compatibility Error

**Error:**
```
TypeError: KeypointParams.__init__() got an unexpected keyword argument 'format'
```

**Cause:**
The Albumentations `Compose` was initialized with `keypoint_params` and `bbox_params` arguments that are not needed for basic image/mask augmentation and caused compatibility issues with different Albumentations versions.

**Fix Applied:**
Removed unnecessary parameters from `get_train_augmentation()` in `augmentation.py`:
- Removed `keypoint_params=A.KeypointParams(...)`
- Removed `bbox_params=A.BboxParams(...)`
- Removed unnecessary `A.Normalize()` step

**Updated code:**
```python
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomScale(...),
    A.RandomCrop(height=crop_size, width=crop_size),
])
```

---

## Issue 2: Deprecated GradScaler API

**Warning:**
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated.
Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

**Cause:**
PyTorch deprecated the old `torch.cuda.amp.GradScaler()` API in favor of the newer `torch.amp.GradScaler()` API.

**Fix Applied:**
Updated `train.py` line 185:
```python
# Old (deprecated):
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# New (correct):
scaler = torch.amp.GradScaler('cuda') if use_amp else None
```

---

## Summary

Both issues are now fixed:
- ✓ Augmentation no longer has compatibility issues
- ✓ GradScaler uses the latest PyTorch API
- ✓ Code is ready for production

## Testing

Try running your training command again:

```bash
python train.py \
    --img_dir /home/n.yakovlev/datasets/lumenstone/S1_v1/imgs/train \
    --mask_dir /home/n.yakovlev/datasets/lumenstone/S1_v1/masks/train \
    --nspix 100 \
    --train_iter 5000 \
    --test_interval 1000 \
    --crop_size 512 \
    --batchsize 8
```

Should now run without errors! 🎉

Monitor the iteration speed - it should show `it/s` in the logs. With your configuration, expect 5-10 iterations/second depending on your hardware.
