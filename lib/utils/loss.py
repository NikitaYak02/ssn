import torch
from .sparse_utils import naive_sparse_bmm, sparse_permute


def reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction via dense matrix operations

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    spixel_mean = (torch.bmm(assignment, labels)
                   / (assignment.sum(2, keepdim=True) + 1e-6))
    if hard_assignment is None:
        permuted_assignment = assignment.permute(0, 2, 1).contiguous()
        reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
    else:
        reconstructed_labels = torch.stack(
            [sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruct_loss_with_cross_entropy(assignment, labels,
                                        hard_assignment=None):
    """
    reconstruction loss with cross entropy

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstructed_labels = reconstruction(assignment, labels, hard_assignment)
    reconstructed_labels = reconstructed_labels / (
        1e-6 + reconstructed_labels.sum(1, keepdim=True))
    mask = labels > 0
    return -(reconstructed_labels[mask] + 1e-6).log().mean()


def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstructed_labels = reconstruction(assignment, labels, hard_assignment)
    return torch.nn.functional.mse_loss(reconstructed_labels, labels)
