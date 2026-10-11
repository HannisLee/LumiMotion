"""只限制normal一致性对opacity的梯度，真实前向及其他梯度保持。"""
from copy import copy

import torch
import torch.nn.functional as F

from gaussian_renderer import render as render_native
from scene.stage2_photometric import srgb_to_linear
from scene.static_dl_ambient import geometry_losses as original_geometry_losses
from scene.static_dl_ambient import render_photometric, parameter_digest
from scene.static_dl_censored import censored_photometric_loss


def geometry_losses(camera, pc, pipe, background, target_alpha, photo, route):
    if route not in ('full', 'detach_opacity'):
        raise ValueError('未知normal opacity路线。')
    # 控制直接调用已冻结T01实现；候选只在normal独立副本的opacity处stop-gradient。
    if route == 'full':
        return original_geometry_losses(camera, pc, pipe, background, target_alpha, photo)
    alpha = photo['rend_alpha'].clamp(1e-6, 1 - 1e-6)
    alpha_loss = F.binary_cross_entropy(alpha, target_alpha)
    geometry_pipe = copy(pipe)
    geometry_pipe.depth_ratio = 0.
    geometry = render_native(camera, pc, geometry_pipe, srgb_to_linear(background),
                             0., 0., 0., override_color=torch.zeros_like(pc.get_xyz),
                             detach_opacity=True)
    if not torch.equal(geometry['radii'], photo['radii']) or not torch.equal(
            geometry['visibility_filter'], photo['visibility_filter']):
        raise AssertionError('normal候选不得改变可见点/半径。')
    normal = F.normalize(geometry['rend_normal'], dim=0)
    depth_normal = F.normalize(geometry['surf_normal'], dim=0)
    mask = (target_alpha > .99).float()
    mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
    mask = mask * (alpha.detach() > .5)
    normal_loss = ((1 - (normal * depth_normal).sum(0, keepdim=True)) * mask).sum() / mask.sum().clamp_min(1)
    # Alpha和distortion都仍来自完全可导的photo副本，未限制它们的opacity梯度。
    return dict(alpha=alpha_loss, normal=normal_loss, distortion=photo['rend_dist'].mean()), geometry


def audit_normal_opacity(camera, pc, pipe, background, lights, visibility, iteration):
    """真实起点零更新审计：相同值/相同其他导数，只删除normal的opacity导数。"""
    before = parameter_digest(pc, lights)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
    photo = render_photometric(camera, pc, pipe, background, lights, visibility, iteration, 0.)
    full, full_package = geometry_losses(camera, pc, pipe, background, alpha, photo, 'full')
    stop, stop_package = geometry_losses(camera, pc, pipe, background, alpha, photo, 'detach_opacity')
    checks = {}
    for key in ('normal', 'alpha', 'distortion'):
        a, b = full[key], stop[key]
        checks['loss_value_' + key] = dict(passed=bool(torch.equal(a, b) and torch.isfinite(a)),
                                           max_abs=float((a - b).abs()))
    for key in ('rend_normal', 'surf_normal', 'surf_depth', 'rend_alpha', 'radii'):
        a, b = full_package[key], stop_package[key]
        checks['geometry_forward_' + key] = dict(passed=bool(torch.equal(a, b) and torch.isfinite(a).all()),
                                                max_abs=float((a - b).abs().max()))
    params = [pc._xyz, pc._rotation, pc._scaling, pc._opacity]
    names = ['xyz', 'rotation', 'scaling', 'opacity']
    weight = pipe.static_dl_normal_weight
    full_grads = torch.autograd.grad(weight * full['normal'], params, retain_graph=True)
    stop_grads = torch.autograd.grad(weight * stop['normal'], params, allow_unused=True, retain_graph=True)
    for name, a, b in zip(names, full_grads, stop_grads):
        if name == 'opacity':
            is_zero = b is None or bool(torch.count_nonzero(b) == 0)
            checks['normal_opacity_removed'] = dict(passed=bool(torch.isfinite(a).all() and a.norm() > 0 and is_zero),
                full_norm=float(a.norm()), full_max_abs=float(a.abs().max()),
                full_positive_fraction=float((a > 0).float().mean()),
                full_negative_fraction=float((a < 0).float().mean()),
                detached_gradient_absent=b is None, detached_max_abs=0. if b is None else float(b.abs().max()))
        else:
            valid = b is not None and bool(torch.isfinite(b).all() and b.norm() > 0
                                          and torch.allclose(a, b, atol=1e-6, rtol=1e-4))
            checks['normal_gradient_' + name] = dict(passed=valid, full_norm=float(a.norm()),
                detached_norm=None if b is None else float(b.norm()),
                max_abs=None if b is None else float((a - b).abs().max()))
    options = (pipe.static_dl_shadow_weight, pipe.static_dl_highlight_weight, pipe.static_dl_residual_threshold)
    rgb = censored_photometric_loss(photo['render'], target, alpha, *options)[0]
    photo_grad = torch.autograd.grad(rgb, pc._opacity, retain_graph=True)[0]
    # full/detach共享同一photo图；因此以下检查直接验证其保留的路径非零且数值不变。
    checks['photo_opacity_preserved'] = dict(passed=bool(torch.isfinite(photo_grad).all() and photo_grad.norm() > 0),
                                           norm=float(photo_grad.norm()), same_photo_tensor=True)
    weights = dict(alpha=pipe.lambda_alpha_loss, distortion=pipe.lambda_dist)
    for key, w in weights.items():
        a = torch.autograd.grad(w * full[key], pc._opacity, retain_graph=True)[0]
        b = torch.autograd.grad(w * stop[key], pc._opacity, retain_graph=True)[0]
        checks[key + '_opacity_preserved'] = dict(passed=bool(torch.isfinite(a).all() and a.norm() > 0
                                                            and torch.allclose(a, b, atol=1e-6, rtol=1e-4)),
            norm=float(a.norm()), max_abs=float((a - b).abs().max()))
    # 总opacity余切必须恰好只减去normal项；避免误删photo/Alpha/dist任一支路。
    loss_full = rgb + pipe.lambda_alpha_loss * full['alpha'] + weight * full['normal'] + pipe.lambda_dist * full['distortion']
    loss_stop = rgb + pipe.lambda_alpha_loss * stop['alpha'] + weight * stop['normal'] + pipe.lambda_dist * stop['distortion']
    total_full = torch.autograd.grad(loss_full, pc._opacity, retain_graph=True)[0]
    total_stop = torch.autograd.grad(loss_stop, pc._opacity)[0]
    residual = total_full - total_stop - full_grads[-1]
    checks['total_opacity_removes_only_normal'] = dict(passed=bool(torch.allclose(
        total_full - total_stop, full_grads[-1], atol=1e-6, rtol=1e-4)), max_abs=float(residual.abs().max()))
    checks['parameters_unchanged'] = dict(passed=before == parameter_digest(pc, lights))
    checks['rng_unchanged'] = dict(passed=bool(torch.equal(cpu_rng, torch.get_rng_state())
                                            and torch.equal(cuda_rng, torch.cuda.get_rng_state())))
    return dict(status='PASS' if all(x['passed'] for x in checks.values()) else 'FAILED', checks=checks,
                camera=camera.image_name_train_light, parameter_updates=0, parameters_sha256=before,
                forward_tolerance='逐元素完全一致', gradient_tolerance=dict(atol=1e-6, rtol=1e-4),
                no_gt_region_used=True, route_difference='只删除normal->opacity')
