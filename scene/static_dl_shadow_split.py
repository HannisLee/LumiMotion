"""独立 Stage2 梯度分流；不改变既有静态 DL 渲染实现。

相机合成只传递颜色导数。接收路径更新真实 q、材质和光强；遮挡路径
仅更新光线上的 Gaussian XYZ。另用原生几何正则把真实 q 与深度关联。
"""
import math

import torch
import torch.nn.functional as F

from gaussian_renderer import render as render_native
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from utils.general_utils import build_rotation, build_scaling_rotation


@torch.no_grad()
def rebuild_bvh(pc):
    """只构建当前离散索引；避免低透明度点的 sqrt(log) 非有限值。"""
    xyz, opacity, scales, rotation = pc.get_xyz, pc.get_opacity, pc.get_scaling, pc.get_rotation
    scale3 = torch.cat((scales, torch.full_like(scales[:, :1], 1e-6)), -1)
    transform = build_scaling_rotation(scale3, rotation)
    radius = (2*(opacity/pc.alpha_min).log().clamp_min(0)).sqrt()[:, None]
    vertices = radius*(pc.unit_icosahedron_vertices[None]@transform.transpose(-1, -2))+xyz[:, None]
    faces = pc.unit_icosahedron_faces[None]+torch.arange(len(xyz), device=xyz.device)[:, None, None]*12
    ids = torch.arange(len(xyz), device=xyz.device)[:, None].expand(-1, faces.shape[1])
    pc.gaussian_tracer.build_bvh(vertices.reshape(-1, 3), faces.reshape(-1, 3), ids.reshape(-1))


def visibility_xyz_only(pc, directions, bias, receiver_positions=None):
    """同一 Gaussian 可接收也可遮挡；按此光线路径停止接收位置导数。"""
    xyz = pc.get_xyz
    receiver = xyz if receiver_positions is None else receiver_positions
    incident = directions.detach()[None].expand(len(receiver), -1, -1).contiguous()
    origin = receiver.detach()[:, None]+incident*bias
    traced = pc.trace(origin, incident, features=torch.zeros(len(xyz), 4, device=xyz.device),
                      xyz=xyz, scales=pc.get_scaling.detach(), rotation=pc.get_rotation.detach(),
                      opacity=pc.get_opacity.detach(), use_zeros=True)
    # 保持既有 pc.trace 的 .97 饱和语义，不能暗中换成未饱和版本。
    return (1-traced['alpha']).clamp(0, 1)


def render_split_photometric(camera, pc, pipe, background, directions, intensity, bias,
                             route='control', receiver_positions=None, rebuild=True,
                             detach_camera=True):
    """control/split 前向完全相同，split 只增加 V→遮挡 XYZ 导数。

occluder 是独立审计路径，不是正式第三组实验。detach_camera=False 仅用于
验证旁路存在的负对照，正式入口不提供此开关。方向始终是固定观测。
"""
    if route not in {'control', 'split', 'occluder'}:
        raise ValueError('未知梯度路径：'+route)
    if pipe.compute_cov3D_python:
        raise ValueError('分流入口固定原生 scale/rotation 路径，不支持 Python covariance。')
    if rebuild:
        rebuild_bvh(pc)
    visibility = visibility_xyz_only(pc, directions, bias, receiver_positions)
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    view = camera.camera_center-pc.get_xyz
    facing = torch.where((normal*view).sum(-1, keepdim=True) >= 0, 1., -1.).detach()
    cosine = ((normal*facing)@directions.detach().T).clamp_min(0)
    albedo = srgb_to_linear(pc.get_albedo)
    receiver_color = albedo*(cosine*visibility.detach()*intensity[None]).sum(-1, keepdim=True)/math.pi
    occluder_color = albedo.detach()*(cosine.detach()*visibility*intensity.detach()[None]).sum(-1, keepdim=True)/math.pi
    if route == 'control':
        colors = receiver_color
    elif route == 'occluder':
        colors = occluder_color
    else:
        # 括号内前向精确为零。只计算一次 RGB loss，q/材质/光强导数不加倍。
        colors = receiver_color+(occluder_color-occluder_color.detach())
    package = render_native(camera, pc, pipe, srgb_to_linear(background), 0., 0., 0.,
                            override_color=colors, detach_xyz=detach_camera, detach_rot=detach_camera,
                            detach_scale=True, detach_opacity=True)
    package['render_linear'] = package['render']
    package['render'] = linear_to_srgb(package['render'].clamp_min(0))
    package['override_colors'] = colors
    package['light_visibility'] = visibility.detach()
    package['mean_visibility'] = visibility.detach().mean()
    # 原生屏幕代理可收到颜色合成导数，但不得被用作增密统计或优化参数。
    package['unused_screen_proxy'] = package.pop('viewspace_points')
    return package


def geometry_regularization(camera, pc, pipe, background, original_rgba_alpha):
    """与 RGB 路径隔离的真实 XYZ/q 几何约束，两组完全相同。"""
    package = render_native(camera, pc, pipe, srgb_to_linear(background), 0., 0., 0.,
                            override_color=torch.zeros_like(pc.get_xyz),
                            detach_scale=True, detach_opacity=True)
    alpha = package['rend_alpha'].clamp(1e-6, 1-1e-6)
    alpha_loss = F.binary_cross_entropy(alpha, original_rgba_alpha.detach())
    normal = F.normalize(package['rend_normal'], dim=0)
    depth_normal = F.normalize(package['surf_normal'], dim=0)
    mask = (original_rgba_alpha.detach() > .99).float()
    mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
    mask = mask*(alpha.detach() > .5)
    normal_loss = ((1-(normal*depth_normal).sum(0, keepdim=True))*mask).sum()/mask.sum().clamp_min(1)
    distortion = package['rend_dist'].mean()
    return {'alpha': alpha_loss, 'normal': normal_loss, 'distortion': distortion}, package


def require_native_backend():
    """沿用正式原生光栅约束，禁止意外落到环境 pip 实现。"""
    import sys
    backend = sys.modules.get('diff_surfel_rasterization')
    provenance = getattr(backend, 'provenance', None)
    if (not isinstance(provenance, dict) or
            provenance.get('method') != 'repository_lumimotion_cuda_rasterizer' or
            not provenance.get('source_sha256') or not provenance.get('extension_sha256')):
        raise RuntimeError('必须通过 bootstrap_1011.py 绑定仓库原生 CUDA 光栅。')
    return provenance
