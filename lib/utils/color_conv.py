"""
OpenCV-based rgb2lab / lab2rgb replacements.

Drop-in replacements for ``skimage.color.rgb2lab`` and ``skimage.color.lab2rgb``
that avoid the boolean-indexing bug triggered by NumPy ≥ 2 on Python 3.14.
"""

import numpy as np
import cv2


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float [0,1] (H,W,3) to LAB float (skimage convention).

    Returns L in [0, 100], a/b roughly in [-128, 127].
    """
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0      # L: [0,255] -> [0,100]
    lab[:, :, 1] -= 128.0              # a: [0,255] -> [-128,127]
    lab[:, :, 2] -= 128.0              # b: [0,255] -> [-128,127]
    return lab


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB float (skimage convention) to RGB float [0,1] (H,W,3).

    Expects L in [0, 100], a/b roughly in [-128, 127].
    """
    lab_cv = lab.copy().astype(np.float32)
    lab_cv[:, :, 0] *= 255.0 / 100.0   # L: [0,100] -> [0,255]
    lab_cv[:, :, 1] += 128.0           # a: [-128,127] -> [0,255]
    lab_cv[:, :, 2] += 128.0           # b: [-128,127] -> [0,255]
    lab_uint8 = np.clip(lab_cv, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return rgb
