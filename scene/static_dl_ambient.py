"""T13续训的单标量间接底项：只由训练RGB学习，保留真实几何光度梯度。"""
from copy import copy
import hashlib
import math

import torch
import torch.nn.functional as F

from gaussian_renderer import render as render_native
from scene.static_dl_photometric import render_photometric as render_original
from scene.static_dl_censored import censored_photometric_loss
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from utils.general_utils import build_rotation


def diffuse_components(albedo, cosine, visibility, intensity, ambient_fraction):
    """a与所有灯的强度同尺度；底项不乘V，也没有空间/相机/时间参数。"""
    irradiance = (cosine * visibility * intensity[None]).sum(-1, keepdim=True)
    direct = albedo * irradiance / math.pi
    ambient = albedo * (ambient_fraction * intensity.sum()) / math.pi
    return direct, ambient


def render_photometric(camera, pc, pipe, background, lights, visibility=None,
                       iteration=0, ambient_fraction=0., component_images=False):
    directions, intensity = lights.for_camera(camera)
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    view = camera.camera_center - pc.get_xyz
    facing = torch.where((normal * view).sum(-1, keepdim=True) >= 0, 1., -1.).detach()
    cosine = ((normal * facing) @ directions.T).clamp_min(0)
    vis = torch.ones_like(cosine) if visibility is None else visibility(pc, directions, iteration)
    if vis.requires_grad:
        raise AssertionError('本配对的直接光可见度必须停止梯度。')
    direct, ambient = diffuse_components(srgb_to_linear(pc.get_albedo), cosine, vis,
                                         intensity, ambient_fraction)
    package = render_native(camera, pc, pipe, srgb_to_linear(background), 0., 0., 0.,
                            override_color=direct + ambient)
    package['render_linear'] = package['render']
    package['render'] = linear_to_srgb(package['render'].clamp_min(0))
    package.update(mean_visibility=vis.mean().detach(),
                   visibility_below_point_one=(vis < .1).float().mean().detach(),
                   visibility_above_point_nine=(vis > .9).float().mean().detach(),
                   light_intensity=intensity.detach(), light_visibility=vis.detach(),
                   ambient_fraction=torch.as_tensor(ambient_fraction).detach().clone())
    # 只在记录步分别无梯度合成直接/间接项，按像素贡献衡量底项。
    if component_images:
        with torch.no_grad():
            indirect = render_native(camera, pc, pipe, torch.zeros_like(background),
                                     0., 0., 0., override_color=ambient.detach())['render']
            package['ambient_linear'] = indirect
            direct_image = render_native(camera, pc, pipe, torch.zeros_like(background),
                                         0., 0., 0., override_color=direct.detach())['render']
            package['direct_linear'] = direct_image
            total = package['render_linear'].detach()
            if torch.count_nonzero(background) or not torch.allclose(total, direct_image + indirect, atol=1e-5, rtol=1e-5):
                raise AssertionError('此黑背景配对的两项真实线性合成不满足加和一致性。')
            package['linear_component_sum_max_abs'] = (total - direct_image - indirect).abs().max()
    return package


def geometry_losses(camera, pc, pipe, background, target_alpha, photo):
    """逐式保持T13 alpha、expected normal、distortion及原监督掩膜。"""
    alpha = photo['rend_alpha'].clamp(1e-6, 1 - 1e-6)
    alpha_loss = F.binary_cross_entropy(alpha, target_alpha)
    geometry_pipe = copy(pipe)
    geometry_pipe.depth_ratio = 0.
    geometry = render_native(camera, pc, geometry_pipe, srgb_to_linear(background),
                             0., 0., 0., override_color=torch.zeros_like(pc.get_xyz))
    if not torch.equal(geometry['radii'], photo['radii']) or not torch.equal(
            geometry['visibility_filter'], photo['visibility_filter']):
        raise AssertionError('expected副本不得改变可见点/半径。')
    normal = F.normalize(geometry['rend_normal'], dim=0)
    depth_normal = F.normalize(geometry['surf_normal'], dim=0)
    mask = (target_alpha > .99).float()
    mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
    mask = mask * (alpha.detach() > .5)
    normal_loss = ((1 - (normal * depth_normal).sum(0, keepdim=True)) * mask).sum() / mask.sum().clamp_min(1)
    return dict(alpha=alpha_loss, normal=normal_loss, distortion=photo['rend_dist'].mean()), geometry


@torch.no_grad()
def component_diagnostics(package, target, alpha, cap):
    foreground = (alpha > .5).expand_as(target)
    dark = foreground & (target.mean(0, keepdim=True) < .1).expand_as(target)
    direct, ambient = package['direct_linear'], package['ambient_linear']
    values = {}
    for label, mask in (('foreground', foreground), ('observed_dark', dark)):
        direct_sum, ambient_sum = direct[mask].sum(), ambient[mask].sum()
        values[label + '_direct_linear_sum'] = float(direct_sum)
        values[label + '_ambient_linear_sum'] = float(ambient_sum)
        values[label + '_ambient_energy_fraction'] = float(ambient_sum / (direct_sum + ambient_sum).clamp_min(1e-12))
        values[label + '_channels'] = int(mask.sum())
    values['linear_component_sum_max_abs'] = float(package['linear_component_sum_max_abs'])
    values['ambient_at_upper_bound'] = bool(package['ambient_fraction'] >= cap - 1e-8)
    return values


def parameter_digest(pc, lights):
    sha = hashlib.sha256()
    for name in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc', '_albedo_rest', 'feature'):
        sha.update(getattr(pc, name).detach().cpu().numpy().tobytes())
    sha.update(lights.log_intensity.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def audit_initial(camera, pc, pipe, background, lights, visibility, iteration):
    """零更新真实帧审计：a=0前向/光度梯度同义，且正底项不依赖直接可见度。"""
    before = parameter_digest(pc, lights)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
    old = render_original(camera, pc, pipe, background, lights, visibility, iteration)
    zero = torch.zeros((), device='cuda', requires_grad=True)
    new = render_photometric(camera, pc, pipe, background, lights, visibility, iteration, zero)
    checks = {}
    for key in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth', 'radii', 'light_intensity'):
        a, b = old[key], new[key]
        good = bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a, b, atol=1e-6, rtol=1e-5))
        checks['forward_' + key] = dict(passed=good, max_abs=float((a - b).abs().max()))
    params = [pc._xyz, pc._rotation, pc._scaling, pc._opacity, pc._albedo_dc, lights.log_intensity]
    names = ['xyz', 'rotation', 'scaling', 'opacity', 'albedo', 'light']
    options = (pipe.static_dl_shadow_weight, pipe.static_dl_highlight_weight, pipe.static_dl_residual_threshold)
    old_loss = censored_photometric_loss(old['render'], target, alpha, *options)[0]
    new_loss = censored_photometric_loss(new['render'], target, alpha, *options)[0]
    old_grads = torch.autograd.grad(old_loss, params)
    new_grads = torch.autograd.grad(new_loss, params + [zero])
    for name, a, b in zip(names, old_grads, new_grads):
        good = bool(torch.isfinite(b).all() and b.norm() > 0 and torch.allclose(a, b, atol=1e-6, rtol=1e-4))
        checks['photo_gradient_' + name] = dict(passed=good, max_abs=float((a - b).abs().max()), norm=float(b.norm()))
    checks['ambient_gradient'] = dict(passed=bool(torch.isfinite(new_grads[-1])), value=float(new_grads[-1]))
    # 独立的解析三点反例检查：V=0时底项仍非零且da导数不随V改变。
    albedo = torch.tensor([[.2, .3, .4], [.3, .4, .5], [.4, .5, .6]], device='cuda')
    cosine = torch.ones(3, 1, device='cuda')
    energy = torch.tensor([2.], device='cuda')
    a = torch.tensor(.05, device='cuda', requires_grad=True)
    direct0, ambient0 = diffuse_components(albedo, cosine, torch.zeros_like(cosine), energy, a)
    _, ambient1 = diffuse_components(albedo, cosine, torch.ones_like(cosine), energy, a)
    derivative = torch.autograd.grad(ambient0.sum(), a)[0]
    checks['ambient_not_direct_visibility_gated'] = dict(passed=bool(
        torch.count_nonzero(direct0) == 0 and (ambient0 > 0).all()
        and torch.equal(ambient0, ambient1)
        and torch.allclose(derivative, albedo.sum() * energy.sum() / math.pi)))
    checks['parameters_unchanged'] = dict(passed=before == parameter_digest(pc, lights))
    checks['rng_unchanged'] = dict(passed=bool(torch.equal(cpu_rng, torch.get_rng_state())
                                            and torch.equal(cuda_rng, torch.cuda.get_rng_state())))
    return dict(status='PASS' if all(x['passed'] for x in checks.values()) else 'FAILED',
                checks=checks, parameter_updates=0, camera=camera.image_name_train_light,
                forward_tolerance=dict(atol=1e-6, rtol=1e-5),
                gradient_tolerance=dict(atol=1e-6, rtol=1e-4),
                parameters_sha256=before)
