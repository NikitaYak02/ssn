import math
import numpy as np
import torch

from lib.utils.color_conv import rgb2lab
from lib.utils.torch_device import get_torch_device, synchronize_device
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26,
              pos_scale=2.5, weight=None, enforce_connectivity=True):
    """
    Generate superpixels for a single image.

    Args:
        image: numpy.ndarray of shape (H, W, 3), RGB uint8
        nspix: int, number of superpixels
        n_iter: int, number of iterations
        fdim: int, feature dimension (needed if using pretrained weight)
        color_scale: float
        pos_scale: float
        weight: str, path to pretrained weights
        enforce_connectivity: bool

    Returns:
        labels: numpy.ndarray of shape (H, W)
    """
    device = get_torch_device(torch)

    if weight is not None:
        from model import SSNModel
        model = SSNModel(fdim, nspix, n_iter).to(device)
        state = torch.load(weight, map_location="cpu")
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
    else:
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis / height,
                                nspix_per_axis / width)

    coords = torch.stack(torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'), 0)
    coords = coords[None].float()

    image_lab = rgb2lab(image)
    image_t = (torch.from_numpy(image_lab).permute(2, 0, 1)[None]
               .to(device).float())

    inputs = torch.cat([color_scale * image_t, pos_scale * coords], 1)

    _, H, _ = model(inputs)
    synchronize_device(torch, device)

    labels = H.reshape(height, width).cpu().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.06 * segment_size)
        max_size = int(3.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--weight", default=None, type=str,
                        help="Path to pretrained weight")
    parser.add_argument("--fdim", default=20, type=int,
                        help="Embedding dimension")
    parser.add_argument("--niter", default=10, type=int,
                        help="Number of SLIC iterations")
    parser.add_argument("--nspix", default=100, type=int,
                        help="Number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    parser.add_argument("--output", default="result.png", type=str,
                        help="Output file path")
    args = parser.parse_args()

    image = plt.imread(args.image)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    s = time.time()
    label = inference(image, args.nspix, args.niter, args.fdim,
                      args.color_scale, args.pos_scale, args.weight)
    print(f"Inference time: {time.time() - s:.2f}s")
    print(f"Number of superpixels: {len(np.unique(label))}")

    plt.imsave(args.output, mark_boundaries(image / 255.0, label))
    print(f"Result saved to {args.output}")
