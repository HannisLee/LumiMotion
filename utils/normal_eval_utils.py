"""Small, dependency-light helpers for GT camera-space normal evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def blender_camera_normal_to_runtime_view(normal: torch.Tensor) -> torch.Tensor:
    """Map Blender camera axes (+X right, +Y up, -Z forward) to runtime view.

    The rasterizer view basis is (+X right, +Y down, +Z forward), so the Y
    and Z components change sign.  Input shape is [..., 3].
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
