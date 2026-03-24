import math
import torch

from .pair_wise_distance import PairwiseDistFunction
from ..utils.sparse_utils import naive_sparse_bmm


def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    Calculate initial superpixel centroids and label map.

    Args:
        images: (B, C, H, W)
    Returns:
        centroids:      (B, C, S)   S = num_spixels_width * num_spixels_height
        init_label_map: (B, H*W)
    """
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(
        images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(
            1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(
            labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def _build_query_idx(init_label_map, num_spixels_width, num_spixels_height,
                     num_spixels):
    """
    Precompute the 9-neighbour superpixel indices for every pixel.
    Called once before the SLIC iteration loop — result is reused each iter.

    Returns:
        query_idx : (B, 9, N) long — clamped to [0, num_spixels-1]
        valid     : (B, 9, N) bool — False for out-of-grid neighbours
    """
    device = init_label_map.device
    k = torch.arange(9, device=device)
    off_x = (k % 3 - 1).long()   # [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    off_y = (k // 3 - 1).long()  # [-1,-1,-1,  0, 0, 0,  1, 1, 1]

    label = init_label_map.long()                       # (B, N)
    x_idx = label % num_spixels_width                   # (B, N)
    y_idx = label // num_spixels_width                  # (B, N)

    new_x = x_idx[:, None, :] + off_x[None, :, None]   # (B, 9, N)
    new_y = y_idx[:, None, :] + off_y[None, :, None]   # (B, 9, N)

    valid = (
        (new_x >= 0) & (new_x < num_spixels_width) &
        (new_y >= 0) & (new_y < num_spixels_height)
    )

    query_idx = (
        label[:, None, :]
        + off_x[None, :, None]
        + num_spixels_width * off_y[None, :, None]
    ).clamp(0, num_spixels - 1)

    return query_idx, valid


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    """Used only by sparse_ssn_iter (inference path)."""
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat(
        [r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = (torch.arange(n_pixel, device=device)[None, None]
                       .repeat(b, 9, 1).reshape(-1).long())
    abs_spix_indices = ((init_label_map[:, None]
                         + relative_spix_indices[None, :, None])
                        .reshape(-1).long())
    abs_batch_indices = (torch.arange(b, device=device)[:, None, None]
                         .repeat(1, 9, n_pixel).reshape(-1).long())

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width,
                        num_spixels_actual):
    """
    Compute hard superpixel assignment for each pixel.

    Border pixels can have all 9 neighbours in the affinity matrix, but some
    neighbours are outside the superpixel grid.  The argmax can therefore pick
    an out-of-grid neighbour, making the resulting label negative or ≥ S.
    We clamp to [0, S-1] so downstream indexing (sm[ha, :] in loss.py) is safe.
    """
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat(
        [r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.clamp(0, num_spixels_actual - 1).long()


# ---------------------------------------------------------------------------
# Inference path — sparse (no gradient needed, runs outside autograd)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sparse_ssn_iter(pixel_features, num_spixels, n_iter):
    """
    Sparse SLIC iteration for inference.
    Does NOT compute gradients.
    """
    height, width = pixel_features.shape[-2:]
    num_spixels_width  = int(math.sqrt(num_spixels * width  / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))
    num_spixels_actual = num_spixels_width * num_spixels_height

    spixel_features, init_label_map = calc_init_centroid(
        pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1)

    for _ in range(n_iter):
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features, spixel_features, init_label_map,
            num_spixels_width, num_spixels_height)

        affinity_matrix = (-dist_matrix).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)

        mask = (abs_indices[1] >= 0) & (abs_indices[1] < num_spixels_actual)
        sparse_abs_affinity = torch.sparse_coo_tensor(
            abs_indices[:, mask], reshaped_affinity_matrix[mask])
        spixel_features = (
            naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features)
            / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None]
               + 1e-6)
        )
        spixel_features = spixel_features.permute(0, 2, 1)

    hard_labels = get_hard_abs_labels(
        affinity_matrix, init_label_map, num_spixels_width, num_spixels_actual)

    return sparse_abs_affinity, hard_labels, spixel_features


# ---------------------------------------------------------------------------
# Training path — dense scatter_add (supports autograd)
# ---------------------------------------------------------------------------

def ssn_iter(pixel_features, num_spixels, n_iter):
    """
    Dense SLIC iteration for training.

    Key optimisation over the original:
    - Neighbour query indices (query_idx, valid) are computed ONCE before
      the loop and reused every iteration.
    - The sparse_coo_tensor → to_dense() round-trip is replaced by a single
      scatter_add_ into a pre-typed zeros buffer.  This removes:
        • get_abs_indices (B×9×N×3 index tensor)
        • sparse COO construction (Python dict / C++ hash-map)
        • to_dense() scatter kernel
      and replaces them with one scatter_add_ (a single fused CUDA kernel).
    """
    height, width = pixel_features.shape[-2:]
    num_spixels_width  = int(math.sqrt(num_spixels * width  / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))
    num_spixels_actual = num_spixels_width * num_spixels_height

    spixel_features, init_label_map = calc_init_centroid(
        pixel_features, num_spixels_width, num_spixels_height)

    B, C, H, W = pixel_features.shape
    pixel_features = pixel_features.reshape(B, C, -1)           # (B, C, N)
    N = pixel_features.shape[2]
    permuted = pixel_features.permute(0, 2, 1).contiguous()     # (B, N, C)

    # ── Precompute neighbour indices (fixed for this H×W / nspix combo) ──────
    query_idx, valid = _build_query_idx(
        init_label_map, num_spixels_width, num_spixels_height,
        num_spixels_actual)
    # query_idx: (B, 9, N) long,  valid: (B, 9, N) bool

    for _ in range(n_iter):
        dist_matrix = PairwiseDistFunction.apply(
            pixel_features, spixel_features, init_label_map,
            num_spixels_width, num_spixels_height)                # (B, 9, N)

        affinity_matrix = (-dist_matrix).softmax(1)              # (B, 9, N)

        # Zero-out affinities for out-of-grid neighbours, then scatter into
        # dense (B, S, N) assignment matrix — replaces sparse→dense round-trip
        src = affinity_matrix * valid                            # (B, 9, N)
        abs_affinity = pixel_features.new_zeros(
            B, num_spixels_actual, N)                            # (B, S, N)
        abs_affinity.scatter_add_(1, query_idx, src)

        # Update superpixel centroids
        spixel_features = (
            torch.bmm(abs_affinity, permuted)                    # (B, S, C)
            / (abs_affinity.sum(2, keepdim=True) + 1e-6)
        ).permute(0, 2, 1).contiguous()                          # (B, C, S)

    hard_labels = get_hard_abs_labels(
        affinity_matrix, init_label_map, num_spixels_width, num_spixels_actual)

    return abs_affinity, hard_labels, spixel_features
