"""
Compare SSN vs SLIC (with different parameters) on the first N images
of a dataset.

Outputs:
  - Console table with per-method mean metrics
  - CSV file with per-image results
  - (optional) side-by-side visualisation grid saved to --vis_dir
"""

import os
import math
import time
import glob
import argparse
import csv
import colorsys
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from lib.utils.color_conv import rgb2lab
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.utils.metrics import compute_all_metrics
from lib.utils.torch_device import get_torch_device, synchronize_device

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

# ── helpers ────────────────────────────────────────────────────────────────────

def collect_image_mask_pairs(img_dir, mask_dir):
    """Return sorted list of (img_path, mask_path) pairs."""
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


def _find_mask(stem, mask_dir):
    for ext in IMAGE_EXTENSIONS:
        for candidate in (stem + ext, stem + ext.upper()):
            path = os.path.join(mask_dir, candidate)
            if os.path.exists(path):
                return path
    return None


def load_image_rgb(path):
    """Load image as uint8 RGB numpy array."""
    img = np.array(Image.open(path).convert('RGB'))
    return img


def load_mask(path):
    """Load grayscale mask as int64 numpy array."""
    return np.array(Image.open(path).convert('L')).astype(np.int64)


# ── SLIC wrapper ───────────────────────────────────────────────────────────────

def run_slic(image_rgb, n_segments, compactness, sigma,
             enforce_connectivity=True):
    """
    Run skimage SLIC on an RGB image.

    Returns:
        labels: np.ndarray (H, W)
        elapsed: float, seconds
    """
    t0 = time.time()
    labels = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )
    elapsed = time.time() - t0

    if enforce_connectivity:
        H, W = labels.shape
        segment_size = H * W / n_segments
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels, elapsed


# ── SSN wrapper ────────────────────────────────────────────────────────────────

def build_ssn_model(weight_path, fdim, nspix, niter, device):
    from model import SSNModel
    model = SSNModel(fdim, nspix, niter).to(device)
    state = torch.load(weight_path, map_location="cpu")
    # torch.compile wraps the model in OptimizedModule, which prefixes all
    # state_dict keys with "_orig_mod.".  Strip the prefix so the checkpoint
    # loads correctly into a plain (uncompiled) SSNModel.
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_ssn(image_rgb, model, nspix, color_scale, pos_scale, device,
            enforce_connectivity=True):
    """
    Run SSN on an RGB image.

    Returns:
        labels: np.ndarray (H, W)
        elapsed: float, seconds
    """
    height, width = image_rgb.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    ps = pos_scale * max(nspix_per_axis / height,
                         nspix_per_axis / width)

    coords = torch.stack(torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'), 0)[None].float()

    image_lab = rgb2lab(image_rgb)
    image_t = (torch.from_numpy(image_lab.astype(np.float32))
               .permute(2, 0, 1)[None].to(device))

    inputs = torch.cat([color_scale * image_t, ps * coords], 1)

    t0 = time.time()
    _, H, _ = model(inputs)
    synchronize_device(torch, device)
    elapsed = time.time() - t0

    labels = H.reshape(height, width).cpu().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels, elapsed


# ── visualisation helpers ───────────────────────────────────────────────────────

def _make_spx_palette(n: int) -> np.ndarray:
    """Return (n, 3) float32 RGB array with visually distinct colors."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    n = max(n, 1)
    colors = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        h = (i / phi) % 1.0
        s = 0.55 + 0.35 * ((i % 3) / 2.0)
        v = 0.70 + 0.25 * (i % 2)
        colors[i] = colorsys.hsv_to_rgb(h, s, v)
    return colors


def _labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Convert int label map (H, W) → float32 RGB (H, W, 3)."""
    palette = _make_spx_palette(int(labels.max()) + 1)
    return palette[labels]


def _crop_boxes(H: int, W: int, n: int = 3, frac: float = 0.33) -> list:
    """
    Return n non-overlapping crop boxes (r0, r1, c0, c1):
      top-left corner, image center, bottom-right corner.
    """
    ch = max(int(H * frac), 48)
    cw = max(int(W * frac), 48)
    boxes = [
        (0,          ch,          0,          cw),
        ((H-ch)//2,  (H+ch)//2,  (W-cw)//2,  (W+cw)//2),
        (H-ch,       H,           W-cw,        W),
    ]
    return boxes[:n]


def _draw_crop_rect(ax, r0, r1, c0, c1, color='lime', lw=1.5):
    """Draw a rectangle on ax indicating the cropped region."""
    from matplotlib.patches import Rectangle
    rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                     linewidth=lw, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


# ── visualisation ──────────────────────────────────────────────────────────────

def save_vis_grid(image_rgb, mask, results, out_path):
    """
    Save a multi-row comparison grid.

    Layout (rows × cols = (2 + n_zoom) × (2 + n_methods)):

      Row 0  – full image: boundary overlays for each method
      Row 1  – full image: blended colored-segment maps
      Row 2  – zoom crop (top-left):     blended segments + thick boundaries
      Row 3  – zoom crop (center):       blended segments + thick boundaries
      Row 4  – zoom crop (bottom-right): blended segments + thick boundaries

    Columns: Original | GT mask | method_1 | method_2 | …

    Zoom boxes are shown as rectangles on the full-image rows.
    """
    N_ZOOM    = 3
    N_ROWS    = 2 + N_ZOOM
    N_COLS    = 2 + len(results)
    COL_W_IN  = 3.2
    ROW_H_IN  = 3.2

    fig = plt.figure(figsize=(COL_W_IN * N_COLS, ROW_H_IN * N_ROWS))
    gs  = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig,
                            wspace=0.03, hspace=0.18)

    H, W  = image_rgb.shape[:2]
    img_f = image_rgb.astype(np.float32) / 255.0
    gt_rgb = _labels_to_rgb(mask)
    boxes  = _crop_boxes(H, W, n=N_ZOOM)

    ZOOM_COLORS = ['lime', 'cyan', 'yellow']
    ROW_LABELS  = [
        "Boundaries\n(full)",
        "Segments\n(full)",
        "Zoom 1\n(top-left)",
        "Zoom 2\n(center)",
        "Zoom 3\n(bot-right)",
    ]

    # Pre-compute per-method colored segment maps (float32, H×W×3)
    seg_maps = [_labels_to_rgb(labels) for _, labels, _ in results]

    for row in range(N_ROWS):
        for col in range(N_COLS):
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')

            # ── column 0: original image (full or cropped) ─────────────────
            if col == 0:
                if row <= 1:
                    ax.imshow(image_rgb)
                    # draw zoom-box outlines on both full-image rows
                    for zi, (r0, r1, c0_, c1) in enumerate(boxes):
                        _draw_crop_rect(ax, r0, r1, c0_, c1,
                                        color=ZOOM_COLORS[zi], lw=1.5)
                else:
                    r0, r1, c0_, c1 = boxes[row - 2]
                    ax.imshow(image_rgb[r0:r1, c0_:c1])
                    # colored border matching the zoom indicator
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(ZOOM_COLORS[row - 2])
                        spine.set_linewidth(2.5)
                    ax.axis('on')
                    ax.set_xticks([])
                    ax.set_yticks([])

                # row label on the left
                ax.set_ylabel(ROW_LABELS[row], fontsize=7,
                              rotation=0, labelpad=55,
                              va='center', ha='right')
                if row == 0:
                    ax.set_title("Original", fontsize=7, pad=3)

            # ── column 1: ground-truth mask ─────────────────────────────────
            elif col == 1:
                if row <= 1:
                    ax.imshow(gt_rgb)
                else:
                    r0, r1, c0_, c1 = boxes[row - 2]
                    ax.imshow(gt_rgb[r0:r1, c0_:c1])
                if row == 0:
                    ax.set_title("GT mask", fontsize=7, pad=3)

            # ── columns 2+: per-method results ──────────────────────────────
            else:
                mi    = col - 2
                name, labels, m = results[mi]
                seg   = seg_maps[mi]

                if row == 0:
                    # full image – yellow boundary overlay on original
                    vis = mark_boundaries(img_f, labels,
                                         color=(1.0, 0.85, 0.0), mode='thick')
                    ax.imshow(np.clip(vis, 0, 1))
                    ax.set_title(
                        f"{name}\n"
                        f"ASA={m['asa']:.3f}  BR={m['br']:.3f}\n"
                        f"UE={m['ue']:.3f}  N={m['n_superpixels']}",
                        fontsize=5.5, pad=3)
                    # draw zoom-box outlines
                    for zi, (r0, r1, c0_, c1) in enumerate(boxes):
                        _draw_crop_rect(ax, r0, r1, c0_, c1,
                                        color=ZOOM_COLORS[zi], lw=1.5)

                elif row == 1:
                    # full image – blended segment colormap (55% seg, 45% orig)
                    blended = np.clip(0.55 * seg + 0.45 * img_f, 0, 1)
                    ax.imshow(blended)
                    # draw zoom-box outlines
                    for zi, (r0, r1, c0_, c1) in enumerate(boxes):
                        _draw_crop_rect(ax, r0, r1, c0_, c1,
                                        color=ZOOM_COLORS[zi], lw=1.5)

                else:
                    # zoom crop – segment blend + thick boundaries on top
                    zi  = row - 2
                    r0, r1, c0_, c1 = boxes[zi]
                    crop_img    = img_f[r0:r1, c0_:c1]
                    crop_labels = labels[r0:r1, c0_:c1]
                    crop_seg    = seg[r0:r1, c0_:c1]

                    base = np.clip(0.55 * crop_seg + 0.45 * crop_img, 0, 1)
                    vis  = mark_boundaries(base, crop_labels,
                                           color=(1.0, 0.85, 0.0), mode='thick')
                    ax.imshow(np.clip(vis, 0, 1))

                    # colored spine matching indicator
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(ZOOM_COLORS[zi])
                        spine.set_linewidth(2.5)
                    ax.axis('on')
                    ax.set_xticks([])
                    ax.set_yticks([])

    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


# ── reporting ──────────────────────────────────────────────────────────────────

METRIC_KEYS = ['asa', 'br', 'ue', 'compactness', 'n_superpixels', 'time_s']
HIGHER_BETTER = {'asa', 'br', 'compactness'}
LOWER_BETTER  = {'ue', 'time_s'}


def print_table(mean_results):
    """
    mean_results: dict  method_name -> {metric: mean_value}
    """
    col_w = 14
    methods = list(mean_results.keys())
    header = f"{'Metric':<18}" + "".join(f"{m:>{col_w}}" for m in methods)
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for key in METRIC_KEYS:
        label = key.upper() if key != 'time_s' else 'Time (s)'
        vals = {m: mean_results[m].get(key, float('nan')) for m in methods}

        # Find best
        valid = {m: v for m, v in vals.items() if not math.isnan(v)}
        if valid:
            if key in HIGHER_BETTER:
                best_val = max(valid.values())
                best_methods = {m for m, v in valid.items() if v == best_val}
            else:
                best_val = min(valid.values())
                best_methods = {m for m, v in valid.items() if v == best_val}
        else:
            best_methods = set()

        row = f"{label:<18}"
        for m in methods:
            v = vals[m]
            cell = f"{v:.4f}" if not math.isnan(v) else "  N/A"
            marker = " *" if m in best_methods else "  "
            row += f"{(marker + cell):>{col_w}}"
        print(row)

    print(sep)
    print("  * = best value for that metric")
    print()


def save_csv(rows, csv_path):
    """rows: list of dicts."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-image CSV saved to {csv_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = get_torch_device(torch)

    # ── collect pairs ──────────────────────────────────────────────────────────
    pairs = collect_image_mask_pairs(args.img_dir, args.mask_dir)
    if not pairs:
        raise RuntimeError("No image-mask pairs found.")

    pairs = pairs[:args.n_images]
    print(f"Evaluating on {len(pairs)} images  (device: {device})")

    # ── build SLIC configs ─────────────────────────────────────────────────────
    # Each entry: (display_name, n_segments, compactness, sigma)
    slic_configs = []
    for c in args.slic_compactness:
        for sig in args.slic_sigma:
            name = f"SLIC_c{c}_s{sig}"
            slic_configs.append((name, args.nspix, c, sig))

    # ── load SSN model (optional) ──────────────────────────────────────────────
    ssn_model = None
    if args.weight is not None:
        print(f"Loading SSN weights from {args.weight} ...")
        ssn_model = build_ssn_model(
            args.weight, args.fdim, args.nspix, args.niter, device)
        print("SSN model loaded.")

    # ── per-image accumulators ─────────────────────────────────────────────────
    method_names = []
    if ssn_model is not None:
        method_names.append("SSN")
    for name, *_ in slic_configs:
        method_names.append(name)

    accum = defaultdict(lambda: defaultdict(list))  # method -> metric -> [vals]
    csv_rows = []

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # ── iterate over images ────────────────────────────────────────────────────
    for img_idx, (img_path, mask_path) in enumerate(pairs):
        fname = os.path.basename(img_path)
        print(f"[{img_idx+1}/{len(pairs)}]  {fname}")

        image_rgb = load_image_rgb(img_path)
        mask = load_mask(mask_path)

        # ensure mask dtype is plain int
        mask = mask.astype(np.int64)

        vis_results = []
        row = {'image': fname}

        # ── SSN ────────────────────────────────────────────────────────────────
        if ssn_model is not None:
            labels, elapsed = run_ssn(
                image_rgb, ssn_model, args.nspix,
                args.color_scale, args.pos_scale, device,
                enforce_connectivity=args.enforce_connectivity)
            m = compute_all_metrics(labels, mask)
            m['time_s'] = elapsed

            for k, v in m.items():
                accum["SSN"][k].append(v)

            for k, v in m.items():
                row[f"SSN_{k}"] = v

            vis_results.append(("SSN", labels, m))
            print(f"    SSN:  ASA={m['asa']:.4f}  BR={m['br']:.4f}  "
                  f"UE={m['ue']:.4f}  N={m['n_superpixels']}  "
                  f"t={elapsed:.3f}s")

        # ── SLIC configs ───────────────────────────────────────────────────────
        for name, n_seg, comp, sigma in slic_configs:
            labels, elapsed = run_slic(
                image_rgb, n_seg, comp, sigma,
                enforce_connectivity=args.enforce_connectivity)
            m = compute_all_metrics(labels, mask)
            m['time_s'] = elapsed

            for k, v in m.items():
                accum[name][k].append(v)

            for k, v in m.items():
                row[f"{name}_{k}"] = v

            vis_results.append((name, labels, m))
            print(f"    {name}:  ASA={m['asa']:.4f}  BR={m['br']:.4f}  "
                  f"UE={m['ue']:.4f}  N={m['n_superpixels']}  "
                  f"t={elapsed:.3f}s")

        csv_rows.append(row)

        # ── visualisation ──────────────────────────────────────────────────────
        if args.vis_dir:
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(args.vis_dir, f"{stem}_compare.png")
            save_vis_grid(image_rgb, mask, vis_results, out_path)

    # ── aggregate & report ─────────────────────────────────────────────────────
    mean_results = {}
    for method in method_names:
        mean_results[method] = {
            k: float(np.mean(v)) for k, v in accum[method].items()
        }

    print("\n" + "=" * 70)
    print(f"MEAN RESULTS  (N={len(pairs)} images, target nspix={args.nspix})")
    print_table(mean_results)

    # ── save CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        save_csv(csv_rows, args.csv)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SSN vs SLIC on the first N dataset images")

    # Dataset
    parser.add_argument("--img_dir", required=True,
                        help="Directory with RGB images")
    parser.add_argument("--mask_dir", required=True,
                        help="Directory with grayscale masks")
    parser.add_argument("--n_images", type=int, default=50,
                        help="Number of images to evaluate (default: 50)")

    # SSN
    parser.add_argument("--weight", default=None,
                        help="Path to SSN checkpoint (.pth). "
                             "Omit to skip SSN.")
    parser.add_argument("--fdim", type=int, default=20,
                        help="SSN feature dimension")
    parser.add_argument("--niter", type=int, default=10,
                        help="SSN SLIC iterations")
    parser.add_argument("--color_scale", type=float, default=0.26)
    parser.add_argument("--pos_scale", type=float, default=2.5)

    # Shared
    parser.add_argument("--nspix", type=int, default=100,
                        help="Target number of superpixels")
    parser.add_argument("--enforce_connectivity", action="store_true",
                        default=True,
                        help="Enforce superpixel connectivity (default: on)")
    parser.add_argument("--no_enforce_connectivity",
                        dest="enforce_connectivity", action="store_false")

    # SLIC sweep
    parser.add_argument("--slic_compactness", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0, 30.0],
                        help="SLIC compactness values to sweep "
                             "(default: 0.1 1.0 10.0 30.0)")
    parser.add_argument("--slic_sigma", type=float, nargs="+",
                        default=[1.0],
                        help="SLIC sigma values to sweep (default: 1.0)")

    # Output
    parser.add_argument("--csv", default="comparison.csv",
                        help="Output CSV path (default: comparison.csv)")
    parser.add_argument("--vis_dir", default=None,
                        help="Save visualisation grids to this directory. "
                             "Omit to skip.")

    args = parser.parse_args()
    main(args)
