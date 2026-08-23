"""Small, dependency-light helpers for GT camera-space normal evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def blender_camera_normal_to_runtime_view(normal: torch.Tensor) -> torch.Tensor:
    """Map Blender camera axes (+X right, +Y up, -Z forward) to runtime view.

    The rasterizer view basis is (+X right, +Y down, +Z forward), so the Y
    and Z components change sign. Input shape is [..., 3]. Only use this for a
    dataset explicitly exporting camera-local normals. LH Blender Normal-pass
    EXRs are world-space and must not pass through this conversion.
    """
    if normal.shape[-1] != 3:
        raise ValueError(f"normal must end in 3 channels, got {tuple(normal.shape)}")
    result = normal.clone()
    result[..., 1:] *= -1.0
    return F.normalize(result, dim=-1)


def normal_angular_error_degrees(
    rendered_normal: torch.Tensor,
    gt_normal: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-pixel unsigned angular error in degrees, zero outside mask."""
    if rendered_normal.shape != gt_normal.shape or rendered_normal.shape[0] != 3:
        raise ValueError("rendered_normal and gt_normal must both be [3,H,W].")
    if valid_mask.shape != rendered_normal.shape[1:]:
        raise ValueError("valid_mask must be [H,W].")
    rendered = F.normalize(rendered_normal, dim=0)
    target = F.normalize(gt_normal, dim=0)
    cosine = (rendered * target).sum(dim=0).clamp(-1.0, 1.0)
    errors = torch.rad2deg(torch.acos(cosine))
    return torch.where(valid_mask, errors, torch.zeros_like(errors))


def alpha_normalized_normal_map(
    encoded_normal_render: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """将 raster 累积的 [0,1] normal color 解码为单位 world normal map。"""
    if encoded_normal_render.ndim != 3 or encoded_normal_render.shape[0] != 3:
        raise ValueError("encoded_normal_render must be [3,H,W].")
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(0)
    if alpha.shape != encoded_normal_render[:1].shape:
        raise ValueError("alpha must be [1,H,W] or [H,W].")
    averaged_encoded = encoded_normal_render / alpha.clamp_min(1e-8)
    return F.normalize(averaged_encoded * 2.0 - 1.0, dim=0)


def masked_normal_cosine_loss(
    rendered_normal: torch.Tensor,
    gt_normal: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean 1-cosine loss over a boolean [H,W] mask."""
    if rendered_normal.shape != gt_normal.shape or rendered_normal.shape[0] != 3:
        raise ValueError("rendered_normal and gt_normal must both be [3,H,W].")
    if valid_mask.shape != rendered_normal.shape[1:]:
        raise ValueError("valid_mask must be [H,W].")
    if not bool(valid_mask.any()):
        raise ValueError("valid_mask contains no valid normal pixels.")
    rendered = F.normalize(rendered_normal, dim=0)
    target = F.normalize(gt_normal, dim=0)
    cosine = (rendered * target).sum(dim=0).clamp(-1.0, 1.0)
    return (1.0 - cosine)[valid_mask].mean()
