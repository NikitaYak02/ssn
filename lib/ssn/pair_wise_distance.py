"""
Pure-PyTorch drop-in replacement for the CUDA pairwise-distance kernel.

For every pixel p, the kernel computed squared Euclidean distances to the 9
superpixels in the 3×3 neighbourhood around p's initial centroid assignment:

    spixel_offset  0..8
    offset_x  = offset %  3 - 1   →  [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    offset_y  = offset // 3 - 1   →  [-1,-1,-1,  0, 0, 0,  1, 1, 1]
    query_idx = init_idx + offset_x + num_spixels_w * offset_y

Out-of-bounds neighbours receive distance 1e16 so softmax ignores them.

Gradients are handled automatically by PyTorch autograd — no custom backward
needed.  The math is identical to the CUDA backward:
    ∂dist/∂pix  =  2 (pix − sp)
    ∂dist/∂sp   = −2 (pix − sp)
"""

import torch


def _pairwise_dist(pixel_features, spixel_features, init_label_map,
                   num_spixels_width, num_spixels_height):
    """
    Args:
        pixel_features   : (B, C, N)   pixel feature vectors
        spixel_features  : (B, C, S)   superpixel centroid features
        init_label_map   : (B, N)      initial superpixel index per pixel
        num_spixels_width  : int
        num_spixels_height : int

    Returns:
        dist_matrix : (B, 9, N)   squared distances;
                      out-of-bounds entries are filled with 1e16
    """
    B, C, N = pixel_features.shape
    S = spixel_features.shape[2]
    device = pixel_features.device

    # ── neighbour offsets (match CUDA kernel indexing exactly) ────────────────
    # offset index k = 0..8:
    #   offset_x[k] = k % 3 - 1
    #   offset_y[k] = k // 3 - 1
    k = torch.arange(9, device=device)
    off_x = (k % 3 - 1).long()   # (9,)  [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    off_y = (k // 3 - 1).long()  # (9,)  [-1,-1,-1,  0, 0, 0,  1, 1, 1]

    # ── grid coordinates of each pixel's initial centroid ─────────────────────
    label = init_label_map.long()                      # (B, N)
    x_idx = (label % num_spixels_width)                # (B, N)
    y_idx = (label // num_spixels_width)               # (B, N)

    # ── neighbour grid coords  (B, 9, N) ─────────────────────────────────────
    new_x = x_idx[:, None, :] + off_x[None, :, None]  # (B, 9, N)
    new_y = y_idx[:, None, :] + off_y[None, :, None]  # (B, 9, N)

    # valid mask — False for out-of-bounds neighbours
    valid = (
        (new_x >= 0) & (new_x < num_spixels_width) &
        (new_y >= 0) & (new_y < num_spixels_height)
    )  # (B, 9, N)

    # ── superpixel index for each neighbour  (B, 9, N) ───────────────────────
    query_idx = (
        label[:, None, :]
        + off_x[None, :, None]
        + num_spixels_width * off_y[None, :, None]
    )  # (B, 9, N)
    # clamp to valid range so gather never accesses out-of-bounds memory;
    # those positions will be masked out after distance computation
    query_idx = query_idx.clamp(0, S - 1)

    # ── gather superpixel features for all 9 neighbours at once ──────────────
    # flatten neighbour dim: (B, 9*N)
    query_flat = query_idx.reshape(B, 9 * N)
    # expand channel dim for gather: (B, C, 9*N)
    query_flat_exp = query_flat[:, None, :].expand(B, C, 9 * N)
    # gather from (B, C, S) → (B, C, 9*N) → (B, C, 9, N)
    sp_feat = torch.gather(spixel_features, 2, query_flat_exp).reshape(B, C, 9, N)

    # ── squared Euclidean distance  (B, 9, N) ─────────────────────────────────
    # pixel_features: (B, C, N) → (B, C, 1, N) broadcasts over 9 neighbours
    dist = ((pixel_features[:, :, None, :] - sp_feat) ** 2).sum(dim=1)

    # ── mask out-of-bounds entries (gradient is 0 at masked positions) ────────
    # Use dtype-aware large value: 1e16 overflows float16 → use max finite val
    # so softmax sees a large negative for invalid neighbors (weight → 0).
    large_val = torch.finfo(dist.dtype).max / 2
    dist = dist.masked_fill(~valid, large_val)

    return dist  # (B, 9, N)


class PairwiseDistFunction:
    """
    Drop-in replacement for the old CUDA-backed autograd.Function.

    The interface is identical:
        PairwiseDistFunction.apply(pixel_features, spixel_features,
                                   init_label_map,
                                   num_spixels_width, num_spixels_height)

    Gradients are computed by PyTorch autograd — no custom backward required.
    """

    @staticmethod
    def apply(pixel_features, spixel_features, init_label_map,
              num_spixels_width, num_spixels_height):
        return _pairwise_dist(
            pixel_features, spixel_features, init_label_map,
            num_spixels_width, num_spixels_height,
        )
