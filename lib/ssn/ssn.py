import math
import torch

from .pair_wise_distance import PairwiseDistFunction
from ..utils.sparse_utils import naive_sparse_bmm


_INIT_LABEL_MAP_CACHE = {}


def _resolve_spixel_grid(height, width, num_spixels):
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))
    num_spixels_actual = num_spixels_width * num_spixels_height
    return num_spixels_width, num_spixels_height, num_spixels_actual


@torch.no_grad()
def _get_cached_init_label_map(batchsize, height, width, num_spixels_width,
                               num_spixels_height, device):
    """
    Return the regular-grid init label map reused across repeated calls with the
    same geometry. This avoids rebuilding the same interpolated grid every time.
    """
    key = (
        int(height),
        int(width),
        int(num_spixels_width),
        int(num_spixels_height),
        str(device),
    )
    cached = _INIT_LABEL_MAP_CACHE.get(key)
    if cached is None:
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(
            num_spixels, device=device, dtype=torch.float32
        ).reshape(1, 1, num_spixels_height, num_spixels_width)
        cached = torch.nn.functional.interpolate(
            labels, size=(height, width), mode="nearest"
        ).reshape(1, -1).long()
        _INIT_LABEL_MAP_CACHE[key] = cached

    if int(batchsize) == 1:
        return cached
    return cached.repeat(int(batchsize), 1)


def calc_init_centroid(images, num_spixels_width, num_spixels_height, init_label_map=None):
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

    if init_label_map is None:
        init_label_map = _get_cached_init_label_map(
            batchsize=batchsize,
            height=height,
            width=width,
            num_spixels_width=num_spixels_width,
            num_spixels_height=num_spixels_height,
            device=device,
        )
    else:
        init_label_map = init_label_map.reshape(batchsize, -1).to(device=device).long()

    num_spixels = num_spixels_width * num_spixels_height
    images_flat = images.reshape(batchsize, channels, -1)
    centroid_sums = images.new_zeros(batchsize, channels, num_spixels)
    centroid_counts = images.new_zeros(batchsize, 1, num_spixels)
    scatter_idx = init_label_map.unsqueeze(1).expand(-1, channels, -1)

    centroid_sums.scatter_add_(2, scatter_idx, images_flat)
    centroid_counts.scatter_add_(
        2,
        init_label_map.unsqueeze(1),
        images.new_ones(batchsize, 1, height * width),
    )
    centroids = centroid_sums / (centroid_counts + 1e-6)

    return centroids, init_label_map


@torch.no_grad()
def get_initial_label_map(batchsize, height, width, num_spixels, device):
    """
    Return the cached regular-grid init label map for a given geometry.
    """
    num_spixels_width, num_spixels_height, _ = _resolve_spixel_grid(
        height, width, num_spixels,
    )
    return _get_cached_init_label_map(
        batchsize=batchsize,
        height=height,
        width=width,
        num_spixels_width=num_spixels_width,
        num_spixels_height=num_spixels_height,
        device=device,
    )


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
    num_spixels_width, num_spixels_height, num_spixels_actual = _resolve_spixel_grid(
        height, width, num_spixels,
    )

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


@torch.no_grad()
def dense_ssn_iter_inference(pixel_features, num_spixels, n_iter,
                             init_spixel_features=None, init_label_map=None):
    """
    Dense inference path that stays on the active device.

    Unlike ``ssn_iter``, this function avoids materialising the full dense
    (B, S, N) assignment tensor. Instead it directly scatters weighted pixel
    features into centroid accumulators, which keeps memory bounded enough for
    inference on MPS/CUDA while still avoiding the sparse CPU fallback.

    Optional ``init_spixel_features`` and ``init_label_map`` provide a
    warm-start for repeated refinement of the same ROI.
    """
    height, width = pixel_features.shape[-2:]
    num_spixels_width, num_spixels_height, num_spixels_actual = _resolve_spixel_grid(
        height, width, num_spixels,
    )

    if init_label_map is None:
        init_label_map = get_initial_label_map(
            batchsize=pixel_features.shape[0],
            height=height,
            width=width,
            num_spixels=num_spixels,
            device=pixel_features.device,
        )

    if init_spixel_features is None:
        spixel_features, init_label_map = calc_init_centroid(
            pixel_features, num_spixels_width, num_spixels_height,
            init_label_map=init_label_map,
        )
    else:
        spixel_features = init_spixel_features.to(
            device=pixel_features.device, dtype=pixel_features.dtype
        )
        init_label_map = init_label_map.reshape(pixel_features.shape[0], -1).to(
            device=pixel_features.device
        ).long()

    B, C, _, _ = pixel_features.shape
    pixel_features = pixel_features.reshape(B, C, -1).contiguous()  # (B, C, N)
    N = pixel_features.shape[2]
    device_type = pixel_features.device.type

    query_idx, valid = _build_query_idx(
        init_label_map, num_spixels_width, num_spixels_height, num_spixels_actual
    )
    if device_type == "mps":
        chunk_size = min(N, 32768)
    elif device_type == "cuda":
        chunk_size = min(N, 262144)
    else:
        chunk_size = N

    hard_labels = None
    for iter_idx in range(n_iter):
        spixel_sums = pixel_features.new_zeros(B, C, num_spixels_actual)
        spixel_weights = pixel_features.new_zeros(B, 1, num_spixels_actual)
        if iter_idx == n_iter - 1:
            hard_labels = init_label_map.new_empty(B, N)

        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            pixel_chunk = pixel_features[:, :, start:end]
            init_chunk = init_label_map[:, start:end]
            query_chunk = query_idx[:, :, start:end]
            valid_chunk = valid[:, :, start:end].to(dtype=pixel_features.dtype)

            dist_matrix = PairwiseDistFunction.apply(
                pixel_chunk,
                spixel_features,
                init_chunk,
                num_spixels_width,
                num_spixels_height,
            )
            affinity_matrix = (-dist_matrix).softmax(1)  # (B, 9, chunk)
            src = affinity_matrix * valid_chunk

            for neighbor_idx in range(9):
                target_idx = query_chunk[:, neighbor_idx, :]          # (B, chunk)
                weights = src[:, neighbor_idx, :].unsqueeze(1)        # (B, 1, chunk)
                target_idx_feat = target_idx.unsqueeze(1).expand(-1, C, -1)

                spixel_sums.scatter_add_(2, target_idx_feat, pixel_chunk * weights)
                spixel_weights.scatter_add_(2, target_idx.unsqueeze(1), weights)

            if hard_labels is not None:
                hard_labels[:, start:end] = get_hard_abs_labels(
                    affinity_matrix,
                    init_chunk,
                    num_spixels_width,
                    num_spixels_actual,
                )

        spixel_features = spixel_sums / (spixel_weights + 1e-6)

    return None, hard_labels, spixel_features


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
    num_spixels_width, num_spixels_height, num_spixels_actual = _resolve_spixel_grid(
        height, width, num_spixels,
    )

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
