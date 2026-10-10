"""Stage2 屏幕空间直接 Lambertian；几何来自冻结的 SH Stage1。"""
import math
import torch
from torch.nn import functional as F
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb, directional_irradiance, area_irradiance


def render_stage2_lambertian(camera, pc, material, deformation, background, direction=None,
                             intensity=1.0, hdr=None, depth_ratio=0.0, light_position=None,
                             reference_distance=8.0, ambient_irradiance=0.0, direct_visibility=None,
                             light_sample_positions=None, light_emission_normal=None, light_influence_radius=None):
    means = pc.get_xyz + deformation["d_xyz"]
    rotation = pc.get_rotation_bias(deformation["d_rotation"])
    geometry = getattr(pc, 'stage2_geometry', None)
    scales = pc.get_scaling + deformation["d_scaling"]
    opacity = pc.get_opacity
    if deformation["d_opacity"] is not None:
        opacity = opacity + deformation["d_opacity"]
    colors = material.albedo
    if geometry is not None:
        means,rotation,scales,opacity = geometry.prepare(means,rotation,scales,opacity)
        colors = geometry.expand_values(colors)
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height, image_width=camera.image_width,
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=torch.zeros_like(background), scale_modifier=1.0,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=False)
    rasterizer_type = GaussianRasterizer
    if geometry is not None and getattr(geometry, 'distortion_enabled', False):
        from diff_surfel_rasterization_stable import GaussianRasterizer as rasterizer_type
    rasterizer = rasterizer_type(settings)
    accumulated, radii, allmap = rasterizer(
        means3D=means, means2D=torch.zeros_like(means), shs=None,
        colors_precomp=srgb_to_linear(colors), opacities=opacity,
        scales=scales, rotations=rotation, cov3D_precomp=None)
    maps = allmap if geometry is not None else allmap.detach()
    alpha = maps[1:2].clamp(0, 1)
    albedo_linear = accumulated / alpha.clamp_min(1e-6)
    normal_view = maps[2:5]
    normal = (normal_view.permute(1, 2, 0) @ camera.world_view_transform[:3, :3].T).permute(2, 0, 1)
    normal = F.normalize(normal, dim=0)
    normal = torch.where(alpha > 1e-6, normal, torch.zeros_like(normal))
    expected = maps[0:1] / alpha.clamp_min(1e-6)
    surfel_shading=geometry is not None and geometry.shading=='surfel'
    if surfel_shading:
        if hdr is None and light_position is None and direction is None:
            raise ValueError("必须指定方向光或 HDR。")
        irradiance=torch.zeros_like(alpha)
    elif hdr is not None:
        irradiance = hdr(normal.permute(1, 2, 0)).permute(2, 0, 1)
    elif light_position is not None:
        h,w=camera.image_height,camera.image_width
        y,x=torch.meshgrid(torch.arange(h,device=means.device,dtype=means.dtype)+.5,
                           torch.arange(w,device=means.device,dtype=means.dtype)+.5,indexing='ij')
        rays=torch.stack(((x-w/2)*2*math.tan(camera.FoVx/2)/w,
                          (y-h/2)*2*math.tan(camera.FoVy/2)/h,torch.ones_like(x)),-1)
        c2w=camera.world_view_transform.T.inverse()
        points=(rays@c2w[:3,:3].T)*expected.permute(1,2,0)+c2w[:3,3]
        to_light=light_position-points
        distance2=to_light.square().sum(-1,keepdim=True).clamp_min(1e-6)
        cosine=(normal.permute(1,2,0)*F.normalize(to_light,dim=-1)).sum(-1,keepdim=True).clamp_min(0)
        irradiance=(cosine*intensity*(reference_distance**2)/distance2).permute(2,0,1)
        if light_sample_positions is not None:
            irradiance=area_irradiance(points,normal.permute(1,2,0),light_sample_positions,
                light_emission_normal,intensity,reference_distance,light_influence_radius).permute(2,0,1)
        if direct_visibility is not None:irradiance=irradiance*direct_visibility.detach()
        irradiance=irradiance+ambient_irradiance
    elif direction is not None:
        irradiance = directional_irradiance(normal.permute(1, 2, 0), direction, intensity).permute(2, 0, 1)
        if direct_visibility is not None:irradiance=irradiance*direct_visibility.detach()
    else:
        raise ValueError("必须指定方向光或 HDR。")
    foreground_linear = albedo_linear * irradiance / math.pi
    if surfel_shading:
        # 前向着色：每个实际surfel先计算辐照度，再按可见度累积RGB。
        # 保留上面的独立材质累积，保证base_color仍表示albedo。
        from utils.general_utils import build_rotation
        surfel_normal=build_rotation(rotation)[:,:,2]
        raw_surfel_normal=surfel_normal
        facing=(surfel_normal*(camera.camera_center-means)).sum(-1,keepdim=True)
        surfel_normal=surfel_normal*torch.where(facing>=0,1.,-1.)
        if hdr is not None:
            surfel_irradiance=hdr(surfel_normal)
        elif light_position is not None:
            delta=light_position-means
            surfel_irradiance=(surfel_normal*F.normalize(delta,dim=-1)).sum(-1,keepdim=True).clamp_min(0)
            surfel_irradiance=surfel_irradiance*intensity*reference_distance**2/delta.square().sum(-1,keepdim=True).clamp_min(1e-6)
            if light_sample_positions is not None:
                surfel_irradiance=area_irradiance(means,surfel_normal,light_sample_positions,
                    light_emission_normal,intensity,reference_distance,light_influence_radius)
        else:
            surfel_irradiance=directional_irradiance(surfel_normal,direction,intensity)
        lit_colors = material.albedo
        if geometry is not None:lit_colors = geometry.expand_values(lit_colors)
        lit,_,_=rasterizer(means3D=means,means2D=torch.zeros_like(means),shs=None,
            colors_precomp=srgb_to_linear(lit_colors)*surfel_irradiance/math.pi,
            opacities=opacity,scales=scales,rotations=rotation,cov3D_precomp=None)
        foreground_linear=lit/alpha.clamp_min(1e-6)
        if direct_visibility is not None:foreground_linear=foreground_linear*direct_visibility.detach()
        if hdr is None and light_position is not None:
            foreground_linear=foreground_linear+albedo_linear*ambient_irradiance/math.pi
        irradiance=foreground_linear*math.pi/albedo_linear.clamp_min(1e-6)
    composed = foreground_linear * alpha + srgb_to_linear(background)[:, None, None] * (1 - alpha)
    display=linear_to_srgb(composed)
    if geometry is not None and geometry.tonemap=='filmic':
        from scene.stage2_geometry import apply_filmic_lut
        display=apply_filmic_lut(foreground_linear.permute(1,2,0),geometry.display_lut).permute(2,0,1)
        display=display*alpha+background[:,None,None]*(1-alpha)
    median = torch.nan_to_num(maps[5:6])
    ratio = min(0.5, depth_ratio)
    geometry_attributes = {}
    if geometry is not None and getattr(geometry, 'distortion_enabled', False):
        geometry_attributes['geometry_distortion'] = maps[6:7]
    if geometry is not None and (geometry.coherence_enabled or getattr(geometry,'rgb_normal_enabled',False)):
        if not surfel_shading:
            from utils.general_utils import build_rotation
            raw_surfel_normal = build_rotation(rotation)[:,:,2]
        geometry_attributes.update(geometry_means=means, geometry_normals=raw_surfel_normal)
    return {
        "render": display, "render_linear": composed,
        "base_color": linear_to_srgb(albedo_linear).clamp(0, 1) * alpha + background[:, None, None] * (1-alpha),
        "base_color_linear": albedo_linear, "rend_alpha": alpha,
        "rend_normal": normal * alpha, "normal": normal,
        "shading": irradiance, "surf_depth": torch.nan_to_num(expected) * (1-ratio) + median * ratio,
        "mask": alpha[0] > 0.5, "radii": radii, "visibility_filter": radii > 0,
        **geometry_attributes,
    }
