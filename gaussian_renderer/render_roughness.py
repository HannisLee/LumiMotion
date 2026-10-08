"""只读光栅化原版 PBR 的 roughness 参数；零背景、单通道、无颜色变换。"""
import math

import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer


@torch.no_grad()
def render_roughness(camera, pc, deformation, compute_cov3D_python=False):
    if hasattr(pc, 'stage2_geometry'):
        raise ValueError('roughness 评测只支持原版 PBR 几何。')
    means = pc.get_xyz + deformation['d_xyz']
    opacity = pc.get_opacity
    if deformation['d_opacity'] is not None:
        opacity = opacity + deformation['d_opacity']
    scales = rotations = covariance = None
    if compute_cov3D_python:
        delta = deformation['d_rotation']
        covariance = pc.get_covariance(1.0, d_rotation=None if isinstance(delta, float) else delta)
    else:
        scales = pc.get_scaling + deformation['d_scaling']
        rotations = pc.get_rotation_bias(deformation['d_rotation'])
    roughness = pc.get_rough
    for value in (means, opacity, roughness, scales, rotations, covariance):
        if value is not None and not torch.isfinite(value).all():
            raise ValueError('roughness 光栅化输入存在 NaN/Inf。')
    if ((roughness < 0) | (roughness > 1)).any():
        raise ValueError('roughness 参数超出 [0,1]。')
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=torch.zeros(3, device=means.device, dtype=means.dtype), scale_modifier=1.0,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=False,
    )
    accumulated, _, allmap = GaussianRasterizer(settings)(
        means3D=means, means2D=torch.zeros_like(means), shs=None,
        colors_precomp=roughness.repeat(1, 3), opacities=opacity,
        scales=scales, rotations=rotations, cov3D_precomp=covariance,
    )
    alpha = allmap[1:2]
    if not torch.isfinite(accumulated).all() or not torch.isfinite(alpha).all():
        raise ValueError('roughness 光栅化输出存在 NaN/Inf。')
    accumulated = accumulated[:1]
    prediction = torch.where(alpha > 1e-6, accumulated / alpha.clamp_min(1e-6), 0.0)
    return {'accumulated': accumulated, 'alpha': alpha, 'roughness': prediction}
