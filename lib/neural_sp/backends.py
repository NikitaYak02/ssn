from __future__ import annotations

import math
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ssn.ssn import (
    _build_query_idx,
    _resolve_spixel_grid,
    calc_init_centroid,
    get_hard_abs_labels,
    get_initial_label_map,
    ssn_iter,
)
from lib.utils.loss import reconstruct_loss_with_mse
from lib.utils.torch_device import get_torch_device, synchronize_device
from model import run_ssn_inference


FEATURE_METHOD_IDS = {"deep_slic", "cnn_rim", "rethink_unsup"}
ASSIGNMENT_METHOD_IDS = {"sp_fcn", "sin"}
_CHECKPOINT_STATE_CACHE: dict[str, dict[str, Any]] = {}


def _method_id(method: Any) -> str:
    return str(getattr(method, "method_id", "")).strip().lower()


def _weight_path(method: Any) -> str:
    value = str(getattr(method, "weight_path", "") or "").strip()
    return os.path.abspath(value) if value else ""


def _channels_for_method(method: Any) -> int:
    return max(8, int(getattr(method, "fdim", 20)))


def _backbone_width(method: Any) -> int:
    return max(8, int(getattr(method, "backbone_width", 32)))


def _build_coords(height: int, width: int, *, device: str, dtype, batch_size: int):
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        ),
        0,
    )
    return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)


def build_model_input_from_lab_tensor(
    image_lab: torch.Tensor,
    *,
    nspix: int,
    color_scale: float,
    pos_scale: float,
) -> torch.Tensor:
    if image_lab.ndim != 4 or image_lab.shape[1] != 3:
        raise ValueError("Expected LAB tensor with shape (B, 3, H, W)")
    batch_size, _, height, width = image_lab.shape
    coords = _build_coords(
        height,
        width,
        device=str(image_lab.device),
        dtype=image_lab.dtype,
        batch_size=batch_size,
    )
    nspix_per_axis = max(1, int(math.sqrt(max(2, int(nspix)))))
    ps = float(pos_scale) * max(nspix_per_axis / height, nspix_per_axis / width)
    return torch.cat([float(color_scale) * image_lab, ps * coords], dim=1)


def _build_model_input_from_numpy(image_lab: np.ndarray, method: Any) -> torch.Tensor:
    device = get_torch_device(torch)
    image_t = (
        torch.from_numpy(np.asarray(image_lab, dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    return build_model_input_from_lab_tensor(
        image_t,
        nspix=int(getattr(method, "nspix", 100)),
        color_scale=float(getattr(method, "color_scale", 0.26)),
        pos_scale=float(getattr(method, "pos_scale", 2.5)),
    )


def _fit_feature_dim(features: np.ndarray, out_dim: int) -> np.ndarray:
    if features.shape[-1] == out_dim:
        return features.astype(np.float32, copy=False)
    if features.shape[-1] > out_dim:
        return features[..., :out_dim].astype(np.float32, copy=False)

    pieces = [features]
    cur_dim = features.shape[-1]
    while cur_dim < out_dim:
        remaining = out_dim - cur_dim
        pieces.append(features[..., : min(features.shape[-1], remaining)])
        cur_dim += min(features.shape[-1], remaining)
    return np.concatenate(pieces, axis=-1).astype(np.float32, copy=False)


def _handcrafted_feature_map(image_lab: np.ndarray, out_dim: int) -> np.ndarray:
    image_lab = np.asarray(image_lab, dtype=np.float32)
    height, width = image_lab.shape[:2]
    lab = image_lab.copy()
    lab[..., 0] /= 100.0
    lab[..., 1:] /= 128.0
    grad_y, grad_x = np.gradient(lab[..., 0])
    ab_grad = np.sqrt(
        np.square(np.gradient(lab[..., 1])[0])
        + np.square(np.gradient(lab[..., 2])[1])
    )
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    features = np.stack(
        [
            lab[..., 0],
            lab[..., 1],
            lab[..., 2],
            grad_x.astype(np.float32),
            grad_y.astype(np.float32),
            ab_grad.astype(np.float32),
            xx,
            yy,
        ],
        axis=-1,
    )
    return _fit_feature_dim(features, out_dim)


def _handcrafted_feature_tensor(model_input: torch.Tensor, out_dim: int) -> torch.Tensor:
    lab = model_input[:, :3].detach().cpu().numpy().transpose(0, 2, 3, 1)
    out = [
        torch.from_numpy(_handcrafted_feature_map(sample, out_dim)).permute(2, 0, 1)
        for sample in lab
    ]
    return torch.stack(out, dim=0).to(device=model_input.device, dtype=model_input.dtype)


def _local_distance_logits(model_input: torch.Tensor, nspix: int) -> torch.Tensor:
    batch_size, _, height, width = model_input.shape
    device = model_input.device
    dtype = model_input.dtype
    num_spixels_width, num_spixels_height, num_spixels_actual = _resolve_spixel_grid(
        height, width, nspix
    )
    init_label_map = get_initial_label_map(batch_size, height, width, nspix, device)
    query_idx, valid = _build_query_idx(
        init_label_map, num_spixels_width, num_spixels_height, num_spixels_actual
    )
    lab_only = model_input[:, :3]
    spixel_lab, _ = calc_init_centroid(
        lab_only,
        num_spixels_width,
        num_spixels_height,
        init_label_map=init_label_map,
    )

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    y_grid = y_grid.reshape(1, 1, -1).expand(batch_size, -1, -1)
    x_grid = x_grid.reshape(1, 1, -1).expand(batch_size, -1, -1)

    label = init_label_map.long()
    x_idx = label % num_spixels_width
    y_idx = label // num_spixels_width
    offsets = torch.arange(9, device=device)
    off_x = (offsets % 3 - 1).long()
    off_y = (offsets // 3 - 1).long()
    new_x = (x_idx[:, None, :] + off_x[None, :, None]).clamp(0, num_spixels_width - 1)
    new_y = (y_idx[:, None, :] + off_y[None, :, None]).clamp(0, num_spixels_height - 1)

    step_x = float(width) / max(1, num_spixels_width)
    step_y = float(height) / max(1, num_spixels_height)
    center_x = (new_x.to(dtype) + 0.5) * step_x
    center_y = (new_y.to(dtype) + 0.5) * step_y
    spatial = (
        torch.square((x_grid - center_x) / max(step_x, 1.0))
        + torch.square((y_grid - center_y) / max(step_y, 1.0))
    )

    pixel_lab = lab_only.reshape(batch_size, 3, -1)[:, None, :, :]
    neighbor_lab = torch.gather(
        spixel_lab[:, None, :, :].expand(-1, 9, -1, -1),
        3,
        query_idx[:, :, None, :].expand(-1, -1, 3, -1),
    )
    color_dist = torch.square(pixel_lab - neighbor_lab).sum(2)
    logits = -(2.5 * spatial + 0.75 * color_dist)
    logits = torch.where(valid, logits, torch.full_like(logits, -1e4))
    return logits.reshape(batch_size, 9, height, width)


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        padding = dilation
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FeatureEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, out_ch: int, *, dilated: bool = False):
        super().__init__()
        d1, d2, d3 = (1, 2, 4) if dilated else (1, 1, 2)
        self.net = nn.Sequential(
            ConvBlock(in_ch, width, dilation=d1),
            ConvBlock(width, width, dilation=d2),
            ConvBlock(width, width, dilation=d3),
            nn.Conv2d(width, out_ch, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalAssignmentFCN(nn.Module):
    def __init__(self, in_ch: int, width: int, out_ch: int = 9):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, width),
            ConvBlock(width, width),
            ConvBlock(width, width),
            nn.Conv2d(width, out_ch, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class InterpolationSuperpixelNet(LocalAssignmentFCN):
    def __init__(self, in_ch: int, width: int, interp_steps: int):
        super().__init__(in_ch=in_ch, width=width, out_ch=9)
        self.interp_steps = max(1, int(interp_steps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x)
        for _ in range(self.interp_steps):
            logits = 0.65 * logits + 0.35 * F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
        return logits


def create_model_for_method(method: Any, device: str | torch.device | None = None) -> nn.Module:
    if device is None:
        device = get_torch_device(torch)
    method_id = _method_id(method)
    width = _backbone_width(method)
    out_dim = _channels_for_method(method)

    if method_id == "deep_slic":
        model = FeatureEncoder(5, width, out_dim, dilated=False)
    elif method_id == "cnn_rim":
        model = FeatureEncoder(5, width, out_dim, dilated=False)
    elif method_id == "rethink_unsup":
        model = FeatureEncoder(5, width, out_dim, dilated=True)
    elif method_id == "sp_fcn":
        model = LocalAssignmentFCN(5, width, out_ch=9)
    elif method_id == "sin":
        model = InterpolationSuperpixelNet(
            5,
            width,
            interp_steps=max(1, int(getattr(method, "interp_steps", 3))),
        )
    else:
        raise ValueError(f"Unsupported neural method id: {method_id!r}")
    model._codex_method_id = method_id
    return model.to(device)


def _extract_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
    if isinstance(checkpoint_obj, dict):
        if "model_state" in checkpoint_obj and isinstance(checkpoint_obj["model_state"], dict):
            return checkpoint_obj["model_state"]
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return checkpoint_obj["state_dict"]
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)!r}")
    return checkpoint_obj


def load_checkpoint_into_model(model: nn.Module, weight_path: str) -> dict[str, Any]:
    abs_path = os.path.abspath(weight_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)
    if abs_path not in _CHECKPOINT_STATE_CACHE:
        _CHECKPOINT_STATE_CACHE[abs_path] = torch.load(abs_path, map_location="cpu")
    checkpoint_obj = _CHECKPOINT_STATE_CACHE[abs_path]
    checkpoint_method_id = ""
    if isinstance(checkpoint_obj, dict):
        checkpoint_method_id = str(checkpoint_obj.get("method_id", "")).strip().lower()
    model_method_id = str(getattr(model, "_codex_method_id", "")).strip().lower()
    if checkpoint_method_id and model_method_id and checkpoint_method_id != model_method_id:
        raise ValueError(
            f"Checkpoint method_id={checkpoint_method_id!r} does not match model method_id={model_method_id!r}"
        )
    state_dict = _extract_state_dict(checkpoint_obj)
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return checkpoint_obj if isinstance(checkpoint_obj, dict) else {}


def save_neural_method_checkpoint(
    path: str,
    *,
    method: Any,
    model: nn.Module,
    metrics: Optional[dict[str, float]] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> None:
    payload = {
        "method_id": _method_id(method),
        "config": {
            key: value
            for key, value in vars(method).items()
            if not key.startswith("_")
        },
        "model_state": model.state_dict(),
        "metrics": metrics or {},
        "meta": extra_meta or {},
    }
    torch.save(payload, path)


def _local_logits_to_assignment(local_logits: torch.Tensor, nspix: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, height, width = local_logits.shape
    num_spixels_width, num_spixels_height, num_spixels_actual = _resolve_spixel_grid(
        height, width, nspix
    )
    init_label_map = get_initial_label_map(batch_size, height, width, nspix, local_logits.device)
    query_idx, valid = _build_query_idx(
        init_label_map,
        num_spixels_width,
        num_spixels_height,
        num_spixels_actual,
    )
    local_affinity = local_logits.reshape(batch_size, 9, -1).softmax(1)
    local_affinity = local_affinity * valid.to(dtype=local_affinity.dtype)
    abs_affinity = local_logits.new_zeros(batch_size, num_spixels_actual, height * width)
    abs_affinity.scatter_add_(1, query_idx, local_affinity)
    hard_labels = get_hard_abs_labels(
        local_affinity,
        init_label_map,
        num_spixels_width,
        num_spixels_actual,
    )
    return abs_affinity, hard_labels


def _weighted_tv_loss(feature_map: torch.Tensor, image_lab: torch.Tensor) -> torch.Tensor:
    feat_dx = feature_map[:, :, :, 1:] - feature_map[:, :, :, :-1]
    feat_dy = feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :]
    img_dx = image_lab[:, :, :, 1:] - image_lab[:, :, :, :-1]
    img_dy = image_lab[:, :, 1:, :] - image_lab[:, :, :-1, :]
    weight_x = torch.exp(-4.0 * img_dx.abs().mean(1, keepdim=True))
    weight_y = torch.exp(-4.0 * img_dy.abs().mean(1, keepdim=True))
    return (feat_dx.abs() * weight_x).mean() + (feat_dy.abs() * weight_y).mean()


def _diversity_rim_loss(assignment: torch.Tensor) -> torch.Tensor:
    local_entropy = -(assignment.clamp_min(1e-6) * assignment.clamp_min(1e-6).log()).sum(1).mean()
    usage = assignment.mean(2)
    global_entropy = -(usage.clamp_min(1e-6) * usage.clamp_min(1e-6).log()).sum(1).mean()
    return local_entropy - 0.2 * global_entropy


def _assignment_reconstruction_target(model_input: torch.Tensor) -> torch.Tensor:
    height, width = model_input.shape[-2:]
    coords = model_input[:, 3:5].reshape(model_input.shape[0], 2, -1) / max(height, width)
    colors = model_input[:, :3].reshape(model_input.shape[0], 3, -1)
    return torch.cat([colors, coords], dim=1)


def _optimize_feature_model_for_image(model: nn.Module, method: Any, model_input: torch.Tensor) -> nn.Module:
    optim_steps = int(getattr(method, "optim_steps", 0))
    if optim_steps <= 0:
        return model

    model = model.to(model_input.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(getattr(method, "lr", 5e-3)))
    recon_target = _assignment_reconstruction_target(model_input)
    image_lab = model_input[:, :3]
    edge_weight = float(getattr(method, "edge_weight", 0.25))
    rim_weight = float(getattr(method, "rim_weight", 1.0))
    soft_recon_weight = float(getattr(method, "soft_recon_weight", 0.35))

    for _ in range(optim_steps):
        optimizer.zero_grad(set_to_none=True)
        pixel_f = model(model_input)
        assignment, _, _ = ssn_iter(
            pixel_f,
            int(getattr(method, "nspix", 100)),
            int(getattr(method, "niter", 5)),
        )
        loss = reconstruct_loss_with_mse(assignment, recon_target)
        loss = loss + rim_weight * _diversity_rim_loss(assignment)
        loss = loss + edge_weight * _weighted_tv_loss(pixel_f, image_lab)
        if _method_id(method) == "rethink_unsup":
            loss = loss + soft_recon_weight * reconstruct_loss_with_mse(assignment, image_lab.reshape(image_lab.shape[0], 3, -1))
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def compute_assignment_for_model(
    model: nn.Module,
    method: Any,
    model_input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    method_id = _method_id(method)
    nspix = int(getattr(method, "nspix", 100))
    niter = int(getattr(method, "niter", 5))

    if method_id in FEATURE_METHOD_IDS:
        pixel_f = model(model_input)
        assignment, hard_labels, spixel_features = ssn_iter(pixel_f, nspix, niter)
        return assignment, hard_labels, {
            "pixel_features": pixel_f,
            "spixel_features": spixel_features,
        }

    local_logits = model(model_input)
    assignment, hard_labels = _local_logits_to_assignment(local_logits, nspix)
    return assignment, hard_labels, {"local_logits": local_logits}


def _predict_feature_method_labels(method: Any, model_input: torch.Tensor) -> np.ndarray:
    weight_path = _weight_path(method)
    if weight_path:
        model = create_model_for_method(method, model_input.device)
        load_checkpoint_into_model(model, weight_path)
        if _method_id(method) in {"cnn_rim", "rethink_unsup"} and int(getattr(method, "optim_steps", 0)) > 0:
            model = _optimize_feature_model_for_image(model, method, model_input)
        with torch.no_grad():
            pixel_f = model(model_input)
    else:
        pixel_f = _handcrafted_feature_tensor(model_input, _channels_for_method(method))

    with torch.no_grad():
        _, hard_labels, _ = run_ssn_inference(
            pixel_f,
            int(getattr(method, "nspix", 100)),
            int(getattr(method, "niter", 5)),
        )
    synchronize_device(torch, str(model_input.device).split(":")[0])
    height, width = model_input.shape[-2:]
    return hard_labels.reshape(height, width).detach().cpu().numpy().astype(np.int32)


def _predict_assignment_method_labels(method: Any, model_input: torch.Tensor) -> np.ndarray:
    weight_path = _weight_path(method)
    if weight_path:
        model = create_model_for_method(method, model_input.device)
        load_checkpoint_into_model(model, weight_path)
        model.eval()
        with torch.no_grad():
            logits = model(model_input)
    else:
        logits = _local_distance_logits(model_input, int(getattr(method, "nspix", 100)))
        if _method_id(method) == "sin":
            steps = max(1, int(getattr(method, "interp_steps", 3)))
            for _ in range(steps):
                logits = 0.6 * logits + 0.4 * F.avg_pool2d(logits, 3, 1, 1)

    with torch.no_grad():
        _, hard_labels = _local_logits_to_assignment(logits, int(getattr(method, "nspix", 100)))
    synchronize_device(torch, str(model_input.device).split(":")[0])
    height, width = model_input.shape[-2:]
    return hard_labels.reshape(height, width).detach().cpu().numpy().astype(np.int32)


def compute_neural_superpixels(
    image_lab: np.ndarray,
    method: Any,
    *,
    mask: Optional[np.ndarray] = None,
    postprocess_fn=None,
) -> np.ndarray:
    model_input = _build_model_input_from_numpy(image_lab, method)
    method_id = _method_id(method)
    if method_id in FEATURE_METHOD_IDS:
        labels = _predict_feature_method_labels(method, model_input)
    elif method_id in ASSIGNMENT_METHOD_IDS:
        labels = _predict_assignment_method_labels(method, model_input)
    else:
        raise ValueError(f"Unsupported neural superpixel method: {method_id!r}")

    labels = labels.astype(np.int32, copy=False) + 1
    if mask is not None:
        labels = labels.copy()
        labels[~mask.astype(bool)] = 0

    if postprocess_fn is not None:
        labels = postprocess_fn(
            labels,
            nspix_hint=int(getattr(method, "nspix", 100)),
            mask=mask.astype(bool) if mask is not None else None,
            prune_small_thin=True,
        )
    return labels.astype(np.int32, copy=False)
