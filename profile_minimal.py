#!/usr/bin/env python3
"""
Minimal profiling — без DataLoader workers overhead.
Предзагружает датасет, потом профилирует pure computation.

Usage:
    python profile_minimal.py \
        --img_dir /path/to/images \
        --mask_dir /path/to/masks \
        --n_batches 5
"""

import os
import sys
import math
import time
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.utils.profiler import BatchProfiler
from lib.utils.loss import reconstruct_loss_with_cross_entropy, reconstruct_loss_with_mse
from model import SSNModel
from lib.dataset.custom_dataset import InMemorySegmentationDataset
from lib.dataset import augmentation


def profile_minimal(cfg):
    """Profile without DataLoader workers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Batch size: {cfg.batchsize}")
    print(f"Crop size: {cfg.crop_size}\n")

    # ── load dataset once ────────────────────────────────────────────────────────
    print("Pre-loading dataset into RAM...")
    t0 = time.time()

    # Use Albumentations for faster augmentation
    augment = augmentation.get_train_augmentation(
        crop_size=cfg.crop_size,
        scale_range=(0.75, 3.0)
    )

    train_dataset = InMemorySegmentationDataset(
        cfg.img_dir, cfg.mask_dir, split="all",
        val_ratio=0.0, max_classes=cfg.max_classes,
        geo_transforms=augment, verbose=False)

    t_load = time.time() - t0
    print(f"Dataset pre-loaded in {t_load:.2f}s")

    # Warmup
    _ = train_dataset[0]

    # Create DataLoader WITHOUT workers (pure computation)
    train_loader = DataLoader(
        train_dataset, cfg.batchsize,
        shuffle=False, drop_last=True,
        num_workers=0,  # NO WORKERS!
        pin_memory=(device == "cuda"))

    # ── model ──────────────────────────────────────────────────────────────────
    print("Creating model...")
    model = SSNModel(cfg.fdim, cfg.nspix, cfg.niter).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ── profiling (no worker overhead) ──────────────────────────────────────────
    print(f"\nProfiling {cfg.n_batches} batches (num_workers=0)...\n")

    def _make_coords(height, width, device):
        return torch.stack(
            torch.meshgrid(torch.arange(height, device=device),
                           torch.arange(width,  device=device),
                           indexing='ij'), 0
        )[None].float()

    coords_cache = {}
    color_scale = cfg.color_scale
    pos_scale = cfg.pos_scale
    compactness_w = cfg.compactness

    profiler = BatchProfiler(enabled=True)
    batch_iter = iter(train_loader)

    for batch_num in range(cfg.n_batches):
        print(f"  Batch {batch_num + 1}/{cfg.n_batches} ...", end=" ", flush=True)

        # Get batch (no worker IPC)
        profiler.start('batch_fetch')
        inputs, labels_u8 = next(batch_iter)
        profiler.end()

        # Transfer + compute
        profiler.start('cpu_to_gpu')
        inputs = inputs.to(device, non_blocking=True)
        labels = labels_u8.to(device, non_blocking=True).float()
        profiler.end()

        height, width = inputs.shape[-2:]

        profiler.start('coords_cache')
        key = (height, width)
        if key not in coords_cache:
            coords_cache[key] = _make_coords(height, width, device)
        coords = coords_cache[key].expand(inputs.shape[0], -1, -1, -1)
        profiler.end()

        profiler.start('cat_input')
        nspix_per_axis = int(math.sqrt(model.nspix))
        ps = pos_scale * max(nspix_per_axis / height,
                             nspix_per_axis / width)
        model_input = torch.cat([color_scale * inputs, ps * coords], 1)
        profiler.end()

        profiler.start('forward_pass')
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            Q, H, _ = model(model_input)
        profiler.end()

        profiler.start('loss_compute')
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            recons_loss = reconstruct_loss_with_cross_entropy(Q, labels)
            compact_loss = reconstruct_loss_with_mse(
                Q, coords.reshape(*coords.shape[:2], -1), H)
            loss = recons_loss + compactness_w * compact_loss
        profiler.end()

        profiler.start('backward_pass')
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        profiler.end()

        print("done")

    # ── report ────────────────────────────────────────────────────────────────
    print("\n")
    profiler.report()
    print("\nPer-batch timing (without DataLoader overhead):")
    total_ms = sum(profiler.times.values())
    per_batch = total_ms / profiler.counts.get('batch_fetch', 1)
    it_s = 1000 / per_batch
    print(f"  {per_batch:.1f} ms/batch → {it_s:.1f} it/s\n")

    print("=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    print("This is the ACTUAL computation speed (no DataLoader workers).")
    print("If this is much faster than profile_one_batch.py, then")
    print("the bottleneck is in DataLoader/workers/augmentation.")
    print()
    print("Solutions:")
    print("  1. Use --nworkers 0 in train.py")
    print("  2. Reduce augmentation complexity (e.g., skip RandomScale)")
    print("  3. Profile with num_workers=0 to isolate worker overhead")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile without DataLoader workers overhead")
    parser.add_argument("--img_dir",  required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--n_batches",    default=5,    type=int)
    parser.add_argument("--batchsize",    default=6,    type=int)
    parser.add_argument("--crop_size",    default=200,  type=int)
    parser.add_argument("--max_classes",  default=50,   type=int)
    parser.add_argument("--fdim",         default=20,   type=int)
    parser.add_argument("--nspix",        default=100,  type=int)
    parser.add_argument("--niter",        default=5,    type=int)
    parser.add_argument("--lr",           default=1e-4, type=float)
    parser.add_argument("--color_scale",  default=0.26, type=float)
    parser.add_argument("--pos_scale",    default=2.5,  type=float)
    parser.add_argument("--compactness",  default=1e-5, type=float)

    args = parser.parse_args()
    profile_minimal(args)
