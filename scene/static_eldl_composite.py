"""1025第三轮：静态ELDL混合光的“sRGB域合成”前向（候选G8）。

与 scene/static_eldl_lighting.render_mixed 的唯一差别：逐Gaussian先把线性着色 C_lin=ρ_lin·[(1−m)S(n)+mV·max(n·l,0)]
编码为sRGB，再按alpha合成（黑背景）。即 render = Σ wᵢ·sRGB(cᵢ)，而不是 sRGB(Σ wᵢ·cᵢ)。
这样，部分覆盖像素的模型值为 â·sRGB(C)，与训练目标 a·sRGB(·) 的预乘结构处于同一合成域（见报告1025-02第五节）。
照明模型、幅度规范、可见度（no_grad）、法线翻面、方向表与 render_mixed 完全相同（直接复用其组件）。
本文件独立于 static_eldl_lighting.py，因此不改变G0–G5快照与评测包装器的SHA一致性检查。
"""
import torch

from gaussian_renderer import render as render_native
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from scene.static_eldl_lighting import oriented_normals, param_hashes, censored_loss

COMPOSITES = ('linear', 'srgb_per_gaussian')


def shade_linear(camera, pc, directions, lighting, visibility, iteration):
    d = directions.for_camera(camera)[None]
    n = oriented_normals(camera, pc)
    cosine = (n @ d.T).clamp_min(0)
    vis = visibility(pc, d, iteration)
    if vis.requires_grad:
        raise AssertionError('直射可见度必须停止梯度')
    m = lighting.direct_fraction()
    albedo = srgb_to_linear(pc.get_albedo)
    env = albedo * (1 - m) * lighting.environment(n)
    direct = albedo * m * vis * cosine
    return env, direct, vis, m


def render_mixed_srgb(camera, pc, pipe, background, directions, lighting, visibility, iteration, components=False):
    """前向只依赖相机、Gaussian、方向与照明参数；不接收目标图像。背景须为黑。"""
    if bool((background != 0).any()):
        raise ValueError('sRGB域合成只支持黑背景')
    env, direct, vis, m = shade_linear(camera, pc, directions, lighting, visibility, iteration)
    color = linear_to_srgb((env + direct).clamp_min(0))
    package = render_native(camera, pc, pipe, background, 0., 0., 0., override_color=color)
    package.update(mean_visibility=vis.mean().detach(), direct_fraction=m.detach(),
                   visibility_below_point_one=(vis < .1).float().mean().detach(), composite='srgb_per_gaussian')
    if components:
        with torch.no_grad():
            zero = torch.zeros_like(background)
            e_img = render_native(camera, pc, pipe, zero, 0., 0., 0., override_color=env.detach())['render']
            d_img = render_native(camera, pc, pipe, zero, 0., 0., 0., override_color=direct.detach())['render']
            lin = render_native(camera, pc, pipe, zero, 0., 0., 0., override_color=(env + direct).detach())['render']
            package['environment_linear'], package['direct_linear'] = e_img, d_img
            package['render_linear'] = lin  # 仅供分量记录；训练loss使用package['render']（sRGB域合成）
            package['component_sum_max_abs'] = (lin - e_img - d_img).abs().max()
    return package


def audit_composite(camera, pc, pipe, background, directions, lighting, visibility, iteration):
    """零更新审计：几何map与linear模式逐元素相同；render等于原生光栅以逐点sRGB颜色合成；背景为0；
    部分覆盖像素上sRGB域合成不比linear更亮；photo对opacity/xyz/albedo梯度有限非零；参数/RNG不变。"""
    from scene.static_eldl_lighting import render_mixed
    before = param_hashes(pc, lighting)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
    sat = camera.observed_saturation_mask.cuda()
    lin = render_mixed(camera, pc, pipe, background, directions, lighting, visibility, iteration)
    srgb = render_mixed_srgb(camera, pc, pipe, background, directions, lighting, visibility, iteration)
    checks = {}
    for k in ('rend_alpha', 'rend_normal', 'surf_depth', 'rend_dist', 'radii', 'visibility_filter'):
        checks['geometry_map_identical_' + k] = dict(passed=bool(torch.equal(lin[k], srgb[k])))
    with torch.no_grad():
        env, direct, _, _ = shade_linear(camera, pc, directions, lighting, visibility, iteration)
        ref = render_native(camera, pc, pipe, background, 0., 0., 0.,
                            override_color=linear_to_srgb((env + direct).clamp_min(0)))['render']
    checks['render_equals_native_srgb_override'] = dict(passed=bool(torch.equal(ref, srgb['render'].detach())))
    bgmask = (lin['rend_alpha'].detach() < 1e-6).expand_as(srgb['render'])
    checks['background_zero'] = dict(passed=bool(srgb['render'].detach()[bgmask].abs().max() < 1e-6) if bgmask.any() else True)
    part = ((alpha > 0) & (alpha < .99)).expand_as(srgb['render'])
    d_lin = float((lin['render'].detach() - target)[part].mean())
    d_srgb = float((srgb['render'].detach() - target)[part].mean())
    checks['partial_pixels_not_brighter_than_linear'] = dict(passed=d_srgb <= d_lin, residual_linear=d_lin,
                                                             residual_srgb=d_srgb)
    photo, _ = censored_loss(srgb['render'], target, alpha, sat)
    grads = torch.autograd.grad(photo, [pc._opacity, pc._xyz, pc._albedo_dc], allow_unused=True)
    for name, g in zip(('opacity', 'xyz', 'albedo_dc'), grads):
        checks['photo_gradient_' + name] = dict(passed=bool(g is not None and torch.isfinite(g).all() and g.norm() > 0),
                                                norm=None if g is None else float(g.norm()))
    checks['parameters_unchanged'] = dict(passed=before == param_hashes(pc, lighting))
    checks['rng_unchanged'] = dict(passed=bool(torch.equal(cpu_rng, torch.get_rng_state()) and
                                               torch.equal(cuda_rng, torch.cuda.get_rng_state())))
    return dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'FAILED', checks=checks,
                camera=camera.image_name_train_light, parameter_updates=0)
