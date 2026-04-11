import os
import glob
import numpy as np
import torch
from lib.utils.color_conv import rgb2lab
from PIL import Image


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def _find_mask(stem, mask_dir):
    for ext in IMAGE_EXTENSIONS:
        for candidate in (stem + ext, stem + ext.upper()):
            path = os.path.join(mask_dir, candidate)
            if os.path.exists(path):
                return path
    return None


def _collect_pairs(img_dir, mask_dir):
    img_files = []
    for ext in IMAGE_EXTENSIONS:
        img_files.extend(glob.glob(os.path.join(img_dir, f'*{ext}')))
        img_files.extend(glob.glob(os.path.join(img_dir, f'*{ext.upper()}')))
    img_files = sorted(set(img_files))

    pairs = []
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = _find_mask(stem, mask_dir)
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    return pairs


def convert_label(label, max_classes=50):
    """
    Convert (H, W) integer mask → (max_classes, H*W) uint8 one-hot.

    Fully vectorised via np.unique(return_inverse=True): no Python loop
    over classes.  Classes beyond max_classes are silently dropped.
    """
    _, inverse = np.unique(label.ravel(), return_inverse=True)  # (N,)
    N = label.size
    valid = inverse < max_classes
    onehot = np.zeros((max_classes, N), dtype=np.uint8)
    cols = np.where(valid)[0]
    onehot[inverse[cols], cols] = 1
    return onehot   # (max_classes, H*W)


class InMemorySegmentationDataset:
    """
    Loads the entire dataset into RAM at construction time so that every
    subsequent __getitem__ does only in-memory augmentation — zero disk I/O.

    Layout in RAM:
        self.images  – list of (H, W, 3) uint8  RGB arrays  (compact storage)
        self.masks   – list of (H, W)    uint8  label maps

    RGB→LAB conversion is deferred to __getitem__ and applied *after* the
    geometric crop so that the conversion runs on the small patch (e.g.
    200×200) rather than the full-resolution image, saving ~(scale²)× time.
    """

    def __init__(self, img_dir, mask_dir, split="train", val_ratio=0.1,
                 max_classes=50, geo_transforms=None, verbose=True,
                 image_downscale=1.0):
        self.max_classes = max_classes
        self.geo_transforms = geo_transforms
        self.image_downscale = float(image_downscale)
        if self.image_downscale <= 0:
            raise ValueError("image_downscale must be > 0")

        pairs = _collect_pairs(img_dir, mask_dir)
        if not pairs:
            raise RuntimeError(
                f"No image-mask pairs found.\n"
                f"  img_dir:  {img_dir}\n"
                f"  mask_dir: {mask_dir}")

        n_val = max(1, int(len(pairs) * val_ratio))
        if split == "train":
            pairs = pairs[n_val:]
        elif split == "val":
            pairs = pairs[:n_val]
        # split == "all" → keep everything

        progress_interval = max(1, len(pairs) // 10) if len(pairs) > 0 else 1
        if verbose:
            print(f"[{split}] preloading {len(pairs)} samples into RAM …", flush=True)

        self.images: list[np.ndarray] = []
        self.masks:  list[np.ndarray] = []

        for idx, (img_path, mask_path) in enumerate(pairs, start=1):
            img_pil = Image.open(img_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('L')
            if self.image_downscale < 1.0:
                new_w = max(1, int(round(img_pil.width * self.image_downscale)))
                new_h = max(1, int(round(img_pil.height * self.image_downscale)))
                img_pil = img_pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
                mask_pil = mask_pil.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
            img = np.array(img_pil)    # uint8
            mask = np.array(mask_pil)  # uint8
            self.images.append(img)
            self.masks.append(mask)
            if verbose and (idx % progress_interval == 0 or idx == len(pairs)):
                print(f"[{split}] preload progress: {idx}/{len(pairs)}", flush=True)

        if verbose:
            img_mb  = sum(x.nbytes for x in self.images)  / 1e6
            mask_mb = sum(x.nbytes for x in self.masks)   / 1e6
            print(f"[{split}] preload done ({img_mb + mask_mb:.0f} MB)")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = self.images[idx]   # (H, W, 3) uint8 — view, not copy
        mask = self.masks[idx]    # (H, W)    uint8

        # Geometric augmentation on the compact uint8 arrays
        if self.geo_transforms is not None:
            img, mask = self.geo_transforms([img, mask])
            # geo_transforms may return numpy views; ensure ownership
            img  = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)

        # LAB conversion on the small cropped patch (cheap)
        img_lab = rgb2lab(img).astype(np.float32)   # (H, W, 3) float32

        # One-hot label — vectorised, returned as uint8 (4× smaller than f32)
        onehot = convert_label(mask, self.max_classes)  # (max_classes, H*W)

        img_t = torch.from_numpy(img_lab).permute(2, 0, 1)  # (3, H, W)
        lbl_t = torch.from_numpy(onehot)                    # (C, H*W) uint8

        return img_t, lbl_t


# ---------------------------------------------------------------------------
# Legacy alias kept for backward compatibility with old scripts
# ---------------------------------------------------------------------------
class SegmentationDataset(InMemorySegmentationDataset):
    """Backward-compatible alias for InMemorySegmentationDataset."""
    def __init__(self, img_dir, mask_dir, split="train", val_ratio=0.1,
                 max_classes=50, color_transforms=None, geo_transforms=None,
                 verbose=True):
        # color_transforms are ignored (LAB conversion done internally)
        super().__init__(img_dir, mask_dir, split=split, val_ratio=val_ratio,
                         max_classes=max_classes, geo_transforms=geo_transforms,
                         verbose=verbose)
