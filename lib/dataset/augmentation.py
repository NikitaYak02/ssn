"""
Data augmentation using Albumentations (optimized for speed).

Albumentations is 2-3x faster than manual OpenCV code because it:
- Batches transformations efficiently
- Uses optimized implementations (SIMD, etc.)
- Avoids unnecessary memory copies
- Applies transformations in optimal order
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np


class ComposeCustom:
    """Wrapper around albumentations.Compose for compatibility."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        """Apply transforms to [image, mask]."""
        image, mask = data[0], data[1]
        result = self.transforms(image=image, mask=mask)
        return [result['image'], result['mask']]


def get_train_augmentation(crop_size=200, scale_range=(0.75, 3.0)):
    """
    Training augmentation pipeline (optimized with Albumentations).

    Args:
        crop_size: int, target crop size
        scale_range: tuple, (min_scale, max_scale) for random scaling

    Returns:
        ComposeCustom wrapper around Albumentations.Compose
    """
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomScale(
            scale_limit=(scale_range[0] - 1, scale_range[1] - 1),
            p=0.8,
            interpolation=1  # cv2.INTER_LINEAR for images
        ),
        A.RandomCrop(height=crop_size, width=crop_size),
    ])
    return ComposeCustom(transforms)


def get_train_augmentation_v2(crop_size=200):
    """
    Faster training augmentation (minimal, optimized).
    Use this if data_loading is still slow.

    Args:
        crop_size: int, target crop size

    Returns:
        ComposeCustom wrapper around Albumentations.Compose
    """
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Skip RandomScale if it's slow
        A.RandomCrop(height=crop_size, width=crop_size),
    ])
    return ComposeCustom(transforms)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy API for backwards compatibility (if old code still uses this)
# ──────────────────────────────────────────────────────────────────────────────

class Compose:
    """Legacy wrapper for backwards compatibility."""
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, data):
        for aug in self.augmentations:
            data = aug(data)
        return data


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data = [d[:, ::-1].copy() for d in data]
        return data


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data = [d[::-1, :].copy() for d in data]
        return data


class RandomScale:
    def __init__(self, scale_range=(0.75, 3.0)):
        self.scale_range = scale_range

    def __call__(self, data):
        rand_factor = np.random.normal(1, 0.75)
        scale = float(np.clip(rand_factor, self.scale_range[0],
                              self.scale_range[1]))
        import cv2
        resized = []
        for d in data:
            interp = cv2.INTER_LINEAR if d.ndim == 3 else cv2.INTER_NEAREST
            resized.append(cv2.resize(d, None, fx=scale, fy=scale,
                                      interpolation=interp))
        return resized


class RandomCrop:
    """Random crop to a fixed size at a uniformly random position."""

    def __init__(self, crop_size=(200, 200)):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

    def __call__(self, data):
        c_h, c_w = self.crop_size
        height, width = data[0].shape[:2]

        pad_h = max(0, c_h - height)
        pad_w = max(0, c_w - width)
        if pad_h > 0 or pad_w > 0:
            padded = []
            for d in data:
                if d.ndim == 3:
                    d = np.pad(d, ((0, pad_h), (0, pad_w), (0, 0)),
                               mode='reflect')
                else:
                    d = np.pad(d, ((0, pad_h), (0, pad_w)),
                               mode='reflect')
                padded.append(d)
            data = padded
            height, width = data[0].shape[:2]

        top  = random.randint(0, height - c_h)
        left = random.randint(0, width  - c_w)
        return [d[top:top + c_h, left:left + c_w] for d in data]
