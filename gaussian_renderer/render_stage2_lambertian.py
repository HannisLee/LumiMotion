"""Stage2 屏幕空间直接 Lambertian；几何来自冻结的 SH Stage1。"""
import math
import torch
from torch.nn import functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb, directional_irradiance


def render_stage2_lambertian(camera, pc, material, deformation, background, direction=None,
                             intensity=1.0, hdr=None, depth_ratio=0.0):
    means = pc.get_xyz + deformation["d_xyz"]
    rotation = pc.get_rotation_bias(deformation["d_rotation"])
    scales = pc.get_scaling + deformation["d_scaling"]
    opacity = pc.get_opacity
    if deformation["d_opacity"] is not None:
        opacity = opacity + deformation["d_opacity"]
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=torch.zeros_like(background), scale_modifier=1.0,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=False)
    rasterizer = GaussianRasterizer(settings)
    accumulated, radii, allmap = rasterizer(
        means3D=means, means2D=torch.zeros_like(means), shs=None,
        colors_precomp=srgb_to_linear(material.albedo), opacities=opacity,
        scales=scales, rotations=rotation, cov3D_precomp=None)
    alpha = allmap[1:2].detach().clamp(0, 1)
    albedo_linear = accumulated / alpha.clamp_min(1e-6)
    normal_view = allmap[2:5].detach()
    normal = (normal_view.permute(1, 2, 0) @ camera.world_view_transform[:3, :3].T).permute(2, 0, 1)
    normal = F.normalize(normal, dim=0)
    normal = torch.where(alpha > 1e-6, normal, torch.zeros_like(normal))
    if hdr is not None:
        irradiance = hdr(normal.permute(1, 2, 0)).permute(2, 0, 1)
    elif direction is not None:
        irradiance = directional_irradiance(normal.permute(1, 2, 0), direction, intensity).permute(2, 0, 1)
    else:
        raise ValueError("必须指定方向光或 HDR。")
    foreground_linear = albedo_linear * irradiance / math.pi
    composed = foreground_linear * alpha + srgb_to_linear(background)[:, None, None] * (1 - alpha)
    expected = allmap[0:1].detach() / alpha.clamp_min(1e-6)
    median = torch.nan_to_num(allmap[5:6].detach())
    ratio = min(0.5, depth_ratio)
    return {
        "render": linear_to_srgb(composed), "render_linear": composed,
        "base_color": linear_to_srgb(albedo_linear).clamp(0, 1) * alpha + background[:, None, None] * (1-alpha),
        "base_color_linear": albedo_linear, "rend_alpha": alpha,
        "rend_normal": normal * alpha, "normal": normal,
        "shading": irradiance, "surf_depth": torch.nan_to_num(expected) * (1-ratio) + median * ratio,
        "mask": alpha[0] > 0.5, "radii": radii, "visibility_filter": radii > 0,
    }
