#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.dataset import augmentation
from lib.dataset.custom_dataset import InMemorySegmentationDataset
from lib.neural_sp.backends import (
    FEATURE_METHOD_IDS,
    build_model_input_from_lab_tensor,
    compute_assignment_for_model,
    create_model_for_method,
    load_checkpoint_into_model,
    save_neural_method_checkpoint,
)
from lib.utils.loss import reconstruct_loss_with_cross_entropy, reconstruct_loss_with_mse
from lib.utils.metrics import compute_all_metrics
from lib.utils.meter import Meter
from lib.utils.torch_device import get_torch_device


_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR / "superpixel_annotator"))
sys.path.insert(0, str(_SCRIPT_DIR))

import structs  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train or fine-tune neural superpixel methods on the target domain."
    )
    ap.add_argument("--img_dir", required=True, help="Directory with RGB images.")
    ap.add_argument("--mask_dir", required=True, help="Directory with grayscale masks.")
    ap.add_argument("--out_dir", default="./out/neural_superpixels", help="Output directory.")
    ap.add_argument(
        "--method",
        required=True,
        choices=sorted(structs.NEURAL_SUPERPIXEL_METHODS),
        help="Neural superpixel method to train.",
    )
    ap.add_argument("--method_config", default=None, help="JSON string or path to JSON config.")
    ap.add_argument("--weights", default=None, help="Optional checkpoint to fine-tune from.")

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_classes", type=int, default=50)
    ap.add_argument("--crop_size", type=int, default=160)
    ap.add_argument("--batchsize", type=int, default=4)
    ap.add_argument("--nworkers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_iter", type=int, default=5000)
    ap.add_argument("--compactness_weight", type=float, default=1e-5)
    ap.add_argument("--edge_reg_weight", type=float, default=5e-4)
    ap.add_argument("--eval_downscale", type=float, default=1.0)
    ap.add_argument("--print_interval", type=int, default=50)
    ap.add_argument("--test_interval", type=int, default=250)
    return ap.parse_args()


def build_method(args: argparse.Namespace) -> structs.SuperPixelMethod:
    return structs.build_superpixel_method_from_args(args)


def create_loaders(args: argparse.Namespace):
    augment = augmentation.get_train_augmentation(crop_size=args.crop_size, scale_range=(0.8, 2.0))
    train_dataset = InMemorySegmentationDataset(
        args.img_dir,
        args.mask_dir,
        split="train",
        val_ratio=args.val_ratio,
        max_classes=args.max_classes,
        geo_transforms=augment,
    )
    val_dataset = InMemorySegmentationDataset(
        args.img_dir,
        args.mask_dir,
        split="val",
        val_ratio=args.val_ratio,
        max_classes=args.max_classes,
        geo_transforms=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=args.nworkers,
        persistent_workers=(args.nworkers > 0),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, val_loader


def forward_assignment(
    inputs_lab: torch.Tensor,
    labels_u8: torch.Tensor,
    *,
    model: torch.nn.Module,
    method: structs.SuperPixelMethod,
    compactness_weight: float,
    edge_reg_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = labels_u8.float().to(inputs_lab.device, non_blocking=True)
    model_input = build_model_input_from_lab_tensor(
        inputs_lab,
        nspix=int(getattr(method, "nspix", 100)),
        color_scale=float(getattr(method, "color_scale", 0.26)),
        pos_scale=float(getattr(method, "pos_scale", 2.5)),
    )
    assignment, hard_labels, aux = compute_assignment_for_model(model, method, model_input)

    height, width = inputs_lab.shape[-2:]
    coords = model_input[:, 3:5].reshape(model_input.shape[0], 2, -1) / max(height, width)
    semantic_loss = reconstruct_loss_with_cross_entropy(assignment, labels, hard_labels)
    compactness_loss = reconstruct_loss_with_mse(assignment, coords, hard_labels)
    loss = semantic_loss + float(compactness_weight) * compactness_loss

    if "pixel_features" in aux and _method_id(method) in FEATURE_METHOD_IDS:
        pixel_features = aux["pixel_features"]
        feat_dx = pixel_features[:, :, :, 1:] - pixel_features[:, :, :, :-1]
        feat_dy = pixel_features[:, :, 1:, :] - pixel_features[:, :, :-1, :]
        loss = loss + float(edge_reg_weight) * (feat_dx.abs().mean() + feat_dy.abs().mean())

    stats = {
        "loss": float(loss.detach().item()),
        "semantic": float(semantic_loss.detach().item()),
        "compact": float(compactness_loss.detach().item()),
    }
    return loss, stats


def _method_id(method: structs.SuperPixelMethod) -> str:
    return str(getattr(method, "method_id", "")).lower()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    method: structs.SuperPixelMethod,
    val_loader: DataLoader,
    device: str,
    eval_downscale: float = 1.0,
) -> dict[str, float]:
    metrics = {key: [] for key in ("asa", "br", "ue", "compactness", "n_superpixels")}
    model.eval()
    for inputs_lab, labels_u8 in val_loader:
        inputs_lab = inputs_lab.to(device)
        gt = labels_u8.argmax(1).float()
        if float(eval_downscale) < 1.0:
            new_h = max(32, int(inputs_lab.shape[-2] * float(eval_downscale)))
            new_w = max(32, int(inputs_lab.shape[-1] * float(eval_downscale)))
            inputs_lab = torch.nn.functional.interpolate(
                inputs_lab,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            gt = torch.nn.functional.interpolate(
                gt.unsqueeze(1),
                size=(new_h, new_w),
                mode="nearest",
            ).squeeze(1)
        model_input = build_model_input_from_lab_tensor(
            inputs_lab,
            nspix=int(getattr(method, "nspix", 100)),
            color_scale=float(getattr(method, "color_scale", 0.26)),
            pos_scale=float(getattr(method, "pos_scale", 2.5)),
        )
        _, hard_labels, _ = compute_assignment_for_model(model, method, model_input)
        height, width = inputs_lab.shape[-2:]
        pred = hard_labels.reshape(height, width).cpu().numpy().astype(np.int32) + 1
        pred = structs._postprocess_superpixel_labels(
            pred,
            nspix_hint=int(getattr(method, "nspix", 100)),
            prune_small_thin=True,
        )
        gt_np = gt.reshape(height, width).cpu().numpy().astype(np.int32)
        cur = compute_all_metrics(pred, gt_np)
        for key, value in cur.items():
            metrics[key].append(float(value))
    model.train()
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    method = build_method(args)
    device = get_torch_device(torch)

    model = create_model_for_method(method, device=device)
    if args.weights:
        load_checkpoint_into_model(model, args.weights)

    train_loader, val_loader = create_loaders(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    meter = Meter()

    with open(os.path.join(args.out_dir, "method_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method_id": _method_id(method),
                "method": asdict(method),
                "train_args": vars(args),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    best_asa = -math.inf
    best_metrics: dict[str, float] = {}
    start_time = time.time()
    iterations = 0

    train_iter = iter(train_loader)
    while iterations < args.train_iter:
        try:
            inputs_lab, labels_u8 = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs_lab, labels_u8 = next(train_iter)

        model.train()
        inputs_lab = inputs_lab.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = forward_assignment(
            inputs_lab,
            labels_u8,
            model=model,
            method=method,
            compactness_weight=args.compactness_weight,
            edge_reg_weight=args.edge_reg_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        iterations += 1
        meter.add(stats)

        if iterations % args.print_interval == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(meter.state(f"[{iterations}/{args.train_iter}]", f"| {iterations / elapsed:.2f} it/s"))

        if iterations % args.test_interval == 0 or iterations == args.train_iter:
            val_metrics = evaluate_model(
                model,
                method,
                val_loader,
                device,
                eval_downscale=float(args.eval_downscale),
            )
            print(
                f"eval@{iterations} asa={val_metrics['asa']:.4f} br={val_metrics['br']:.4f} "
                f"ue={val_metrics['ue']:.4f} compactness={val_metrics['compactness']:.4f}"
            )
            if val_metrics["asa"] >= best_asa:
                best_asa = val_metrics["asa"]
                best_metrics = val_metrics
                save_neural_method_checkpoint(
                    os.path.join(args.out_dir, "best_model.pth"),
                    method=method,
                    model=model,
                    metrics=val_metrics,
                    extra_meta={"iteration": iterations},
                )

    save_neural_method_checkpoint(
        os.path.join(args.out_dir, "last_model.pth"),
        method=method,
        model=model,
        metrics=best_metrics,
        extra_meta={"iteration": iterations},
    )
    print("Best validation metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
