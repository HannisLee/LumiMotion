"""从训练材质直接光栅化线性 albedo；不执行光照或 SH 外观计算。"""
import math
from types import SimpleNamespace

import torch
from scene.pbr_rasterizer import rasterizer_classes

from scene.stage2_photometric import srgb_to_linear


@torch.no_grad()
def render_albedo(camera, pc, albedo_srgb, deformation, compute_cov3D_python=False, rasterizer_backend='installed'):
    GaussianRasterizationSettings,GaussianRasterizer=rasterizer_classes(SimpleNamespace(
        pbr_rasterizer_backend=rasterizer_backend,compute_cov3D_python=compute_cov3D_python))
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
    if hasattr(pc, 'stage2_geometry'):
        if compute_cov3D_python:
            raise ValueError('几何残差仅支持直接scale/rotation光栅化。')
        means,rotations,scales,opacity = pc.stage2_geometry.prepare(means,rotations,scales,opacity)
        albedo_srgb = pc.stage2_geometry.expand_values(albedo_srgb)
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=torch.zeros(3, device=means.device, dtype=means.dtype), scale_modifier=1.0,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=False,
    )
    accumulated, _, allmap = GaussianRasterizer(settings)(
        means3D=means, means2D=torch.zeros_like(means), shs=None,
        colors_precomp=srgb_to_linear(albedo_srgb), opacities=opacity,
        scales=scales, rotations=rotations, cov3D_precomp=covariance,
    )
    alpha = allmap[1:2].clamp(0, 1)
    return {'accumulated_linear': accumulated, 'alpha': alpha}
