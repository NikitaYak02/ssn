import os
import math
import time
import colorsys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')   # headless — no display needed
import matplotlib.pyplot as plt

from lib.utils.meter import Meter
from lib.utils.profiler import BatchProfiler
from lib.utils.loss import (reconstruct_loss_with_cross_entropy,
                             reconstruct_loss_with_mse)
from lib.utils.metrics import compute_all_metrics
from lib.utils.torch_device import get_torch_device
from model import SSNModel
from lib.dataset.custom_dataset import InMemorySegmentationDataset
from lib.dataset import augmentation


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_coords(height, width, device):
    """Return (1, 2, H, W) coordinate grid on `device`."""
    return torch.stack(
        torch.meshgrid(torch.arange(height, device=device),
                       torch.arange(width,  device=device),
                       indexing='ij'), 0
    )[None].float()   # (1, 2, H, W)


# ── visualisation helpers ───────────────────────────────────────────────────────

def _spx_palette(n):
    """
    Generate N visually distinct RGB colours (uint8) using the golden-ratio
    HSV wheel so adjacent superpixels rarely share a similar hue.
    """
    phi = (1 + 5 ** 0.5) / 2   # golden ratio
    h = 0.0
    colors = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        h = (h + 1.0 / phi) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors   # (N, 3) uint8


def _spx_boundaries(labels_hw):
    """Return boolean (H, W) mask that is True on superpixel boundary pixels."""
    right = np.pad(labels_hw, ((0, 0), (0, 1)), mode='edge')[:, :-1]
    down  = np.pad(labels_hw, ((0, 1), (0, 0)), mode='edge')[:-1, :]
    return (labels_hw != right) | (labels_hw != down)


@torch.no_grad()
def save_superpixel_viz(model, val_loader, color_scale, pos_scale, device,
                        out_dir, step, n_images=4):
    """
    Save a PNG grid showing superpixel predictions for up to `n_images`
    validation samples.  Each row contains three panels:
        [input image | boundary overlay | random-colour superpixel map]

    The PNG is written to  <out_dir>/images/step_XXXXXXX.png.
    """
    model.eval()
    imgs_dir = os.path.join(out_dir, 'images')
    os.makedirs(imgs_dir, exist_ok=True)

    rows = []   # list of (img_display, img_boundary, img_spx)

    for inputs, _ in val_loader:
        if len(rows) >= n_images:
            break

        inputs_gpu = inputs.to(device, non_blocking=True)
        height, width = inputs_gpu.shape[-2:]
        nspix_per_axis = int(math.sqrt(model.nspix))
        ps = pos_scale * max(nspix_per_axis / height, nspix_per_axis / width)

        coords = _make_coords(height, width, device)
        coords = coords.expand(inputs_gpu.shape[0], -1, -1, -1)
        model_input = torch.cat([color_scale * inputs_gpu, ps * coords], 1)
        _, H, _ = model(model_input)

        for b in range(inputs_gpu.shape[0]):
            if len(rows) >= n_images:
                break

            # ── input image: LAB → RGB for display ───────────────────────────
            # inputs are in skimage LAB (L: 0..100, a/b: -128..127).
            # Naive min-max normalisation treats LAB channels as RGB, making
            # the image appear red (L dominates after normalisation).
            from lib.utils.color_conv import lab2rgb
            img_lab = inputs[b].permute(1, 2, 0).numpy().astype(np.float32)
            img = np.clip(lab2rgb(img_lab), 0.0, 1.0).astype(np.float32)

            # ── hard superpixel label map (H, W) int ─────────────────────────
            spx = H[b].reshape(height, width).cpu().numpy().astype(np.int32)

            # ── red boundary overlay ──────────────────────────────────────────
            boundary = _spx_boundaries(spx)
            img_bound = img.copy()
            img_bound[boundary] = [1.0, 0.0, 0.0]

            # ── random-colour superpixel map ──────────────────────────────────
            n_spx = int(spx.max()) + 1
            palette = _spx_palette(n_spx)
            img_spx = palette[spx].astype(np.float32) / 255.0   # (H, W, 3)

            rows.append((img, img_bound, img_spx))

    if not rows:
        model.train()
        return

    # ── build the composite figure ────────────────────────────────────────────
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n),
                             squeeze=False)   # always 2-D even for n=1

    col_titles = ['Input image', 'Superpixel boundaries', 'Superpixel map']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold')

    for row_idx, (img, img_bound, img_spx) in enumerate(rows):
        for col_idx, panel in enumerate([img, img_bound, img_spx]):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel)
            ax.axis('off')

    fig.suptitle(f'Superpixel predictions — step {step:,}',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(imgs_dir, f'step_{step:07d}.png')
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    model.train()
    print(f"  → visualization saved → {save_path}")


# ── evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, color_scale, pos_scale, device, profiler=None):
    """
    Evaluate on the validation set.
    Returns mean ASA / BR / UE / Compactness / N_superpixels.
    """
    model.eval()
    all_metrics = {k: [] for k in ('asa', 'br', 'ue', 'compactness',
                                   'n_superpixels')}

    for inputs, labels_u8 in loader:
        if profiler:
            profiler.start('eval_gpu_transfer')
        inputs = inputs.to(device, non_blocking=True)
        if profiler:
            profiler.end()

        height, width = inputs.shape[-2:]
        nspix_per_axis = int(math.sqrt(model.nspix))
        ps = pos_scale * max(nspix_per_axis / height,
                             nspix_per_axis / width)

        if profiler:
            profiler.start('eval_coords')
        coords = _make_coords(height, width, device)   # (1, 2, H, W)
        coords = coords.expand(inputs.shape[0], -1, -1, -1)
        if profiler:
            profiler.end()

        if profiler:
            profiler.start('eval_forward')
        model_input = torch.cat([color_scale * inputs, ps * coords], 1)
        Q, H, _ = model(model_input)
        if profiler:
            profiler.end()

        if profiler:
            profiler.start('eval_metrics')
        # H: (B, N) int — squeeze batch dim (val_loader uses batchsize=1)
        H_np = H.reshape(-1).cpu().numpy().reshape(height, width)
        # labels_u8: (B, C, N) uint8 — argmax(1) → (B, N) → squeeze → (H, W)
        gt_np = labels_u8.argmax(1).reshape(-1).numpy().reshape(height, width)

        m = compute_all_metrics(H_np, gt_np)
        for k, v in m.items():
            all_metrics[k].append(v)
        if profiler:
            profiler.end()

    model.train()
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ── single training step ────────────────────────────────────────────────────────

def update_param(inputs, labels_u8, coords_cache,
                 model, optimizer, scaler,
                 compactness_w, color_scale, pos_scale, device,
                 profiler=None):
    """
    One forward–backward–step.

    coords_cache : dict  (height, width) → (1, 2, H, W) coords tensor
                   shared across calls to avoid re-allocating the grid.
    labels_u8    : (B, C, N) uint8 on CPU — converted to float32 on GPU
                   (4× smaller CPU→GPU transfer than float32).
    scaler       : torch.cuda.amp.GradScaler or None
    profiler     : BatchProfiler or None for timing measurements
    """
    if profiler:
        profiler.start('cpu_to_gpu')
    inputs = inputs.to(device, non_blocking=True)
    # Convert labels to float on GPU (4× less data over PCIe)
    labels = labels_u8.to(device, non_blocking=True).float()   # (B, C, N)
    if profiler:
        profiler.end()

    height, width = inputs.shape[-2:]

    if profiler:
        profiler.start('coords_cache')
    # Coordinate grid — reuse cached tensor for this (H, W)
    key = (height, width)
    if key not in coords_cache:
        coords_cache[key] = _make_coords(height, width, device)
    coords = coords_cache[key].expand(inputs.shape[0], -1, -1, -1)
    if profiler:
        profiler.end()

    nspix_per_axis = int(math.sqrt(model.nspix))
    ps = pos_scale * max(nspix_per_axis / height,
                         nspix_per_axis / width)

    if profiler:
        profiler.start('cat_input')
    model_input = torch.cat([color_scale * inputs, ps * coords], 1)
    if profiler:
        profiler.end()

    if profiler:
        profiler.start('forward_pass')
    # AMP forward pass (float16 conv/BN for speed)
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        Q, H, _ = model(model_input)
    if profiler:
        profiler.end()

    if profiler:
        profiler.start('loss_compute')
    # ── Loss MUST run in float32 ──────────────────────────────────────────────
    # backward of log(x + eps) produces -1/(x+eps).  With eps=1e-6 and x≈0
    # this is ~1e6, which overflows float16 (max ~65504) → Inf gradient →
    # clip_grad_norm_ turns Inf×0 into NaN (IEEE 754) → scaler.step() skips
    # the optimizer update → weights never change → metrics frozen forever.
    Q_f32 = Q.float()       # detach dtype only; gradients still flow
    coords_flat = coords.reshape(*coords.shape[:2], -1)   # (B, 2, N)
    coords_norm = coords_flat / max(height, width)         # [0, 1]
    recons_loss = reconstruct_loss_with_cross_entropy(Q_f32, labels)
    compact_loss = reconstruct_loss_with_mse(Q_f32, coords_norm, H)
    loss = recons_loss + compactness_w * compact_loss
    if profiler:
        profiler.end()

    if profiler:
        profiler.start('backward_pass')
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Safety: nan_to_num_ before clip so Inf (if any remain) → finite
        for p in model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    if profiler:
        profiler.end()

    return {
        "loss":    loss.item(),
        "recons":  recons_loss.item(),
        "compact": compact_loss.item(),
    }


# ── main training loop ─────────────────────────────────────────────────────────

def train(cfg):
    if getattr(cfg, "device", None):
        device = torch.device(cfg.device)
    else:
        device = torch.device(get_torch_device(torch))
    use_amp = device.type == "cuda"

    model = SSNModel(cfg.fdim, cfg.nspix, cfg.niter).to(device)

    # torch.compile gives 10-30% extra speed on PyTorch ≥ 2.0 for free
    if device.type != "mps" and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            pass   # older GPU / driver — just continue without compile

    optimizer = optim.Adam(model.parameters(), cfg.lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── dataset ────────────────────────────────────────────────────────────────
    # Use Albumentations for 2-3x faster augmentation
    augment = augmentation.get_train_augmentation(
        crop_size=cfg.crop_size,
        scale_range=(0.75, 3.0)
    )

    train_dataset = InMemorySegmentationDataset(
        cfg.img_dir, cfg.mask_dir, split="train",
        val_ratio=cfg.val_ratio, max_classes=cfg.max_classes,
        geo_transforms=augment)

    val_dataset = InMemorySegmentationDataset(
        cfg.img_dir, cfg.mask_dir, split="val",
        val_ratio=cfg.val_ratio, max_classes=cfg.max_classes,
        geo_transforms=None)

    # With data in RAM the worker bottleneck is augmentation (cv2.resize),
    # not I/O.  persistent_workers avoids re-forking every epoch.
    _dl_kwargs = dict(
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.nworkers > 0),
    )
    train_loader = DataLoader(
        train_dataset, cfg.batchsize,
        shuffle=True, drop_last=True,
        num_workers=cfg.nworkers, **_dl_kwargs)

    val_loader = DataLoader(
        val_dataset, 1,
        shuffle=False, drop_last=False,
        num_workers=0)   # val is fast — no need for extra workers

    # ── info ───────────────────────────────────────────────────────────────────
    print(f"Train: {len(train_dataset)} samples  |  "
          f"Val: {len(val_dataset)} samples")
    print(f"Device: {str(device)}  |  AMP: {use_amp}  |  "
          f"Workers: {cfg.nworkers}")
    print(f"nspix={cfg.nspix}  fdim={cfg.fdim}  niter={cfg.niter}  "
          f"crop={cfg.crop_size}")
    print("-" * 70)

    meter = Meter()
    profiler = BatchProfiler(enabled=True)   # Detailed timing
    coords_cache = {}   # (H, W) → (1, 2, H, W) coords tensor on GPU
    iterations  = 0
    max_val_asa = 0.0
    best_metrics = {}
    t_start = time.time()

    while iterations < cfg.train_iter:
        profiler.start('data_loading')
        for inputs, labels_u8 in train_loader:
            profiler.end()
            iterations += 1
            metric = update_param(
                inputs, labels_u8, coords_cache,
                model, optimizer, scaler,
                cfg.compactness, cfg.color_scale, cfg.pos_scale, device,
                profiler=profiler)
            meter.add(metric)

            if iterations % cfg.print_interval == 0:
                elapsed = time.time() - t_start
                it_s = iterations / elapsed
                # AMP health: print scaler scale so we can verify optimizer steps
                scaler_info = ""
                if scaler is not None:
                    scaler_info = f"| scale={scaler.get_scale():.0f}"
                print(meter.state(f"[{iterations}/{cfg.train_iter}]",
                                  f"| {it_s:.1f} it/s {scaler_info}"))
                # Show profiling every N iterations
                if iterations % (cfg.print_interval * 10) == 0:
                    profiler.report(prefix=f"[iter {iterations}] ")
                    profiler.reset()

            profiler.start('data_loading')

            if iterations % cfg.test_interval == 0:
                print("-" * 70)
                print(f"Evaluating @ iter {iterations} …")
                val_m = evaluate(model, val_loader,
                                 cfg.color_scale, cfg.pos_scale, device,
                                 profiler=profiler)

                print(f"  ASA:           {val_m['asa']:.6f}")
                print(f"  Boundary Rec:  {val_m['br']:.6f}")
                print(f"  Underseg Err:  {val_m['ue']:.6f}")
                print(f"  Compactness:   {val_m['compactness']:.6f}")
                print(f"  N superpixels: {val_m['n_superpixels']:.0f}")

                if val_m['asa'] > max_val_asa:
                    max_val_asa = val_m['asa']
                    best_metrics = val_m.copy()
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.out_dir, "best_model.pth"))
                    print(f"  → new best model saved (ASA={max_val_asa:.4f})")

                # Save superpixel visualisation for up to n_viz_images val images
                save_superpixel_viz(
                    model, val_loader,
                    cfg.color_scale, cfg.pos_scale, device,
                    cfg.out_dir, iterations, n_images=cfg.n_viz_images)
                print("-" * 70)

            if iterations >= cfg.train_iter:
                break

    # Final checkpoint
    uid = str(int(time.time()))
    torch.save(model.state_dict(),
               os.path.join(cfg.out_dir, f"model_{uid}.pth"))

    print("=" * 70)
    print("Training complete!")
    print("\nBest validation metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoints saved to {cfg.out_dir}")

    # Final profiling summary if enabled
    if profiler.times:
        print("\nFinal profiling summary (full training run):")
        profiler.report()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--img_dir",  required=True,
                        help="Directory with RGB images")
    parser.add_argument("--mask_dir", required=True,
                        help="Directory with grayscale masks")
    parser.add_argument("--val_ratio",   default=0.1,  type=float)
    parser.add_argument("--max_classes", default=50,   type=int,
                        help="Max number of semantic classes in masks")
    parser.add_argument("--crop_size",   default=128,  type=int,
                        help="Square crop size for training patches")

    # Output
    parser.add_argument("--out_dir", default="./log",
                        help="Checkpoint output directory")

    # Device (default: cuda > mps > cpu via get_torch_device)
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device, e.g. "cuda:2". Default: auto-select.',
    )

    # Training
    parser.add_argument("--batchsize",   default=6,      type=int)
    parser.add_argument("--nworkers",    default=2,      type=int,
                        help="DataLoader workers (0 = main process only)")
    parser.add_argument("--lr",          default=1e-4,   type=float)
    parser.add_argument("--train_iter",  default=500000, type=int)

    # Model
    parser.add_argument("--fdim",  default=20, type=int,
                        help="Feature embedding dimension")
    parser.add_argument("--niter", default=5,  type=int,
                        help="Differentiable SLIC iterations")
    parser.add_argument("--nspix", default=100, type=int,
                        help="Number of superpixels")

    # Hyperparameters
    parser.add_argument("--color_scale",  default=0.26,  type=float)
    parser.add_argument("--pos_scale",    default=2.5,   type=float)
    parser.add_argument("--compactness",  default=1e-5,  type=float)

    # Logging
    parser.add_argument("--test_interval",  default=10000, type=int)
    parser.add_argument("--print_interval", default=100,   type=int)
    parser.add_argument("--n_viz_images",   default=4,     type=int,
                        help="Validation images to visualise at each test step")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
