import numpy as np
from skimage.segmentation import find_boundaries


def achievable_segmentation_accuracy(superpixel, label):
    """
    Achievable Segmentation Accuracy (ASA).
    ASA(S,G) = sum_j max_i |s_j & g_i| / sum_i |g_i|

    Args:
        superpixel: np.ndarray of shape (H, W) - superpixel label map
        label: np.ndarray of shape (H, W) - ground truth segmentation

    Returns:
        asa: float in [0, 1]
    """
    TP = 0
    for uid in np.unique(superpixel):
        mask = superpixel == uid
        label_hist = np.bincount(label[mask].ravel())
        TP += label_hist.max()
    return TP / label.size


def boundary_recall(superpixel, label, tolerance=2):
    """
    Boundary Recall (BR).
    Fraction of ground truth boundary pixels that lie within `tolerance`
    pixels of a superpixel boundary.

    Args:
        superpixel: np.ndarray (H, W)
        label: np.ndarray (H, W)
        tolerance: int, dilation radius for matching

    Returns:
        br: float in [0, 1]
    """
    gt_boundaries = find_boundaries(label, mode='thick').astype(np.uint8)
    sp_boundaries = find_boundaries(superpixel, mode='thick').astype(np.uint8)

    if gt_boundaries.sum() == 0:
        return 1.0

    # Dilate superpixel boundaries by tolerance
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1))
    dilated_sp = binary_dilation(sp_boundaries, structure=struct).astype(
        np.uint8)

    recalled = (gt_boundaries * dilated_sp).sum()
    return recalled / gt_boundaries.sum()


def undersegmentation_error(superpixel, label):
    """
    Undersegmentation Error (UE).
    UE = (1/N) * sum_i (sum_{s_j: s_j & g_i != 0} |s_j| - |g_i|)
    where N is total number of pixels.

    Args:
        superpixel: np.ndarray (H, W)
        label: np.ndarray (H, W)

    Returns:
        ue: float >= 0 (lower is better)
    """
    N = label.size
    total_leak = 0

    for gt_id in np.unique(label):
        gt_mask = label == gt_id
        gt_size = gt_mask.sum()

        # Find all superpixels that overlap with this GT region
        overlapping_sp_ids = np.unique(superpixel[gt_mask])
        sp_union_size = 0
        for sp_id in overlapping_sp_ids:
            sp_union_size += (superpixel == sp_id).sum()

        total_leak += sp_union_size - gt_size

    return total_leak / N


def compactness(superpixel):
    """
    Average compactness of superpixels.
    Compactness = 4 * pi * area / perimeter^2

    Args:
        superpixel: np.ndarray (H, W)

    Returns:
        mean_compactness: float in [0, 1]
    """
    compact_vals = []
    boundaries = find_boundaries(superpixel, mode='thick')

    for uid in np.unique(superpixel):
        mask = superpixel == uid
        area = mask.sum()
        perimeter = (mask & boundaries).sum()
        if perimeter > 0:
            c = 4 * np.pi * area / (perimeter ** 2)
            compact_vals.append(min(c, 1.0))

    return np.mean(compact_vals) if compact_vals else 0.0


def compute_all_metrics(superpixel, label, tolerance=2):
    """
    Compute all superpixel quality metrics.

    Args:
        superpixel: np.ndarray (H, W)
        label: np.ndarray (H, W)
        tolerance: int for boundary recall

    Returns:
        dict with keys: asa, br, ue, compactness, n_superpixels
    """
    return {
        'asa': achievable_segmentation_accuracy(superpixel, label),
        'br': boundary_recall(superpixel, label, tolerance),
        'ue': undersegmentation_error(superpixel, label),
        'compactness': compactness(superpixel),
        'n_superpixels': len(np.unique(superpixel)),
    }
