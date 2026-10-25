"""静态ELDL-V3混合光物理前向：共享低阶normal-SH环境辐照 + GT方向Lambertian直射×预测可见度。

C_lin = rho_lin * [ (1-m) * S(n) + m * V * max(n·l,0) ]，其中
  S(n) = max(0, sum_k a_ck Y_k(n)) / L00，L00 = 亮度加权(0.2126,0.7152,0.0722)的DC项 a_c0*Y00；
  m = sigmoid(beta)。
幅度规范（固定、训练前登记）：环境平均亮度辐照 + 直射峰值辐照 ≡ pi（即上式乘pi/pi）。
因此 rho 与总光强之间的全局尺度不再自由；可学的只有环境/直射比例 m、SH色度与形状、反照率。
环境项不乘直射可见度V；V由预测Gaussian BVH在no_grad下计算（停止梯度）。
不读取GT光强、颜色、点光位置或环境参数；方向来自P0净化表（仅TRAIN帧单位向量）。
"""
import hashlib
import json
import math
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from gaussian_renderer import render as render_native
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb, frame_key
from utils.general_utils import build_rotation

LUMA = (0.2126, 0.7152, 0.0722)
Y00 = 0.28209479177387814


def sh9_basis(n):
    """实球谐 l<=2，n为单位向量 (N,3) -> (N,9)。"""
    x, y, z = n.unbind(-1)
    return torch.stack([
        torch.full_like(x, Y00),
        0.4886025119029199 * y, 0.4886025119029199 * z, 0.4886025119029199 * x,
        1.0925484305920792 * x * y, 1.0925484305920792 * y * z,
        0.31539156525252005 * (3 * z * z - 1),
        1.0925484305920792 * x * z, 0.5462742152960396 * (x * x - y * y)], -1)


class TrainDirections:
    """只读P0净化方向表；拒绝含光强/颜色/位置/环境字段的文件。"""
    FORBIDDEN = {'intensity', 'light_rgb', 'energy', 'power', 'color', 'light_pos_world', 'lights', 'environment',
                 'matrix_world', 'shadow_soft_size'}

    def __init__(self, path):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        assert data['split'] == 'TRAIN117' and data['forbidden_fields_included'] == []
        self.directions = {}
        for rec in data['records']:
            if self.FORBIDDEN & set(rec):
                raise ValueError('方向表含禁止字段')
            assert set(rec) == {'frame', 'image_name', 'direction'}
            v = torch.tensor(rec['direction'], dtype=torch.float32)
            if v.shape != (3,) or not torch.isfinite(v).all() or abs(float(v.norm()) - 1) > 1e-5:
                raise ValueError('方向必须为有限单位向量')
            self.directions[str(rec['frame']).zfill(4)] = v.cuda()
        self.frames = sorted(int(k) for k in self.directions)

    def for_camera(self, camera):
        return self.directions[frame_key(camera.image_name_train_light)]


class MixedLighting(torch.nn.Module):
    """共享normal-SH环境 + 单标量直射比例；全部无空间/逐帧自由度。"""

    def __init__(self, initial_direct_fraction=0.5):
        super().__init__()
        self.log_dc = torch.nn.Parameter(torch.zeros(3, device='cuda'))          # 环境DC色度（正）
        self.higher = torch.nn.Parameter(torch.zeros(3, 8, device='cuda'))       # l=1,2 形状
        p = float(initial_direct_fraction)
        self.beta = torch.nn.Parameter(torch.tensor(math.log(p / (1 - p)), device='cuda'))

    def coefficients(self):
        dc = self.log_dc.exp()[:, None]  # a_c0（Y00因子在基函数中）
        a = torch.cat([dc, self.higher], 1)
        l00 = sum(w * a[c, 0] for c, w in enumerate(LUMA)) * Y00
        return a / l00  # 归一化后环境平均亮度辐照为1

    def direct_fraction(self):
        return torch.sigmoid(self.beta)

    def environment(self, n):
        return (sh9_basis(n) @ self.coefficients().T).clamp_min(0)  # (N,3)

    def state(self):
        m = float(self.direct_fraction())
        return dict(log_dc=self.log_dc.detach().cpu().tolist(), higher=self.higher.detach().cpu().tolist(),
                    beta=float(self.beta), direct_fraction=m, environment_fraction=1 - m,
                    normalized_sh_coefficients=self.coefficients().detach().cpu().tolist(),
                    gauge='环境平均亮度辐照(1-m)·pi + 直射峰值辐照m·pi ≡ pi；C_lin=rho_lin·[(1-m)S(n)+mV·max(n·l,0)]',
                    implied_direct_irradiance=math.pi * m, implied_environment_mean_luminance_irradiance=math.pi * (1 - m))

    def load_state(self, s):
        with torch.no_grad():
            self.log_dc.copy_(torch.tensor(s['log_dc'], device='cuda'))
            self.higher.copy_(torch.tensor(s['higher'], device='cuda'))
            self.beta.copy_(torch.tensor(s['beta'], device='cuda'))


def oriented_normals(camera, pc):
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    view = camera.camera_center - pc.get_xyz
    facing = torch.where((normal * view).sum(-1, keepdim=True) >= 0, 1., -1.).detach()
    return normal * facing


def render_mixed(camera, pc, pipe, background, directions, lighting, visibility, iteration, components=False):
    """前向只依赖相机、Gaussian、方向与照明参数；不接收目标图像。"""
    d = directions.for_camera(camera)[None]  # (1,3)
    n = oriented_normals(camera, pc)
    cosine = (n @ d.T).clamp_min(0)  # (N,1)
    vis = visibility(pc, d, iteration)  # (N,1) no_grad
    if vis.requires_grad:
        raise AssertionError('直射可见度必须停止梯度')
    m = lighting.direct_fraction()
    albedo = srgb_to_linear(pc.get_albedo)
    env = albedo * (1 - m) * lighting.environment(n)
    direct = albedo * m * vis * cosine
    package = render_native(camera, pc, pipe, srgb_to_linear(background), 0., 0., 0., override_color=env + direct)
    package['render_linear'] = package['render']
    package['render'] = linear_to_srgb(package['render'].clamp_min(0))
    package.update(mean_visibility=vis.mean().detach(), direct_fraction=m.detach(),
                   visibility_below_point_one=(vis < .1).float().mean().detach())
    if components:
        with torch.no_grad():
            zero = torch.zeros_like(background)
            e_img = render_native(camera, pc, pipe, zero, 0., 0., 0., override_color=env.detach())['render']
            d_img = render_native(camera, pc, pipe, zero, 0., 0., 0., override_color=direct.detach())['render']
            package['environment_linear'], package['direct_linear'] = e_img, d_img
            package['component_sum_max_abs'] = (package['render_linear'].detach() - e_img - d_img).abs().max()
    return package


def partial_alpha_keep_mask(alpha, upper=.99):
    """1025补充：GT Alpha部分覆盖像素（0<a<upper）置0，其余为1。目标=a·sRGB(含原opaque世界背景的混合)，
    与线性域合成后再sRGB的物理前向在这些像素上不可比（见报告1025-01）。仅由训练Alpha决定。"""
    return (~((alpha.detach() > 0) & (alpha.detach() < upper))).float()


def censored_loss(image, target, alpha, saturation, shadow_weight=.25, highlight_weight=.5, threshold=.08,
                  pixel_keep=None):
    """沿用static_dl_censored的鲁棒权重；满码集合取原分辨率任一子像素满码（AREA下界）且Alpha>0.5。
    pixel_keep（1025补充，默认None=与G0逐式相同）：(1,H,W)的0/1像素掩膜，乘到权重上；分母仍为全部像素（mean不变）。"""
    residual = image - target
    signed = residual.detach().mean(0, keepdim=True)
    weight = torch.ones_like(signed)
    weight = torch.where(signed > threshold, shadow_weight, weight)
    weight = torch.where(signed < -threshold, highlight_weight, weight)
    weight = torch.where(alpha.detach() > .5, weight, torch.ones_like(weight))
    if pixel_keep is not None:
        weight = weight * pixel_keep.detach()
    censored = (saturation & (alpha.detach() > .5)).detach()
    residual = torch.where(censored, residual.clamp_max(0), residual)
    loss = (torch.sqrt(residual.square() + 1e-6) * weight).mean()
    diag = dict(shadow_downweight_fraction=float((signed > threshold).float().mean()),
                highlight_downweight_fraction=float((signed < -threshold).float().mean()),
                censored_channel_fraction=float(censored.float().mean()),
                exact_one_channel_fraction=float(((target == 1) & (alpha > .5)).float().mean()))
    if pixel_keep is not None:
        diag['photo_pixel_excluded_fraction'] = float(1 - pixel_keep.float().mean())
    return loss, diag


def geometry_losses(camera, pc, pipe, background, target_alpha, photo, route):
    """Alpha BCE + expected-depth法线一致性 + distortion；route只影响normal副本的opacity梯度。"""
    if route not in ('full', 'detach_opacity'):
        raise ValueError(route)
    alpha = photo['rend_alpha'].clamp(1e-6, 1 - 1e-6)
    alpha_loss = F.binary_cross_entropy(alpha, target_alpha)
    gpipe = copy(pipe)
    gpipe.depth_ratio = 0.
    geometry = render_native(camera, pc, gpipe, srgb_to_linear(background), 0., 0., 0.,
                             override_color=torch.zeros_like(pc.get_xyz), detach_opacity=route == 'detach_opacity')
    if not torch.equal(geometry['radii'], photo['radii']) or not torch.equal(
            geometry['visibility_filter'], photo['visibility_filter']):
        raise AssertionError('normal副本不得改变可见点/半径')
    normal = F.normalize(geometry['rend_normal'], dim=0)
    depth_normal = F.normalize(geometry['surf_normal'], dim=0)
    mask = (target_alpha > .99).float()
    mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
    mask = mask * (alpha.detach() > .5)
    normal_loss = ((1 - (normal * depth_normal).sum(0, keepdim=True)) * mask).sum() / mask.sum().clamp_min(1)
    return dict(alpha=alpha_loss, normal=normal_loss, distortion=photo['rend_dist'].mean()), geometry


@torch.no_grad()
def component_stats(package, alpha):
    fg = (alpha > .5).expand_as(package['environment_linear'])
    e, d = package['environment_linear'][fg].sum(), package['direct_linear'][fg].sum()
    return dict(foreground_environment_linear_sum=float(e), foreground_direct_linear_sum=float(d),
                foreground_direct_energy_fraction=float(d / (e + d).clamp_min(1e-12)),
                component_sum_max_abs=float(package['component_sum_max_abs']))


def tensor_digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def param_hashes(pc, lighting=None):
    out = {n: tensor_digest([getattr(pc, n)]) for n in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc',
                                                         '_albedo_rest', '_roughness', 'feature') if hasattr(pc, n)}
    if lighting is not None:
        out['lighting'] = tensor_digest([lighting.log_dc, lighting.higher, lighting.beta])
    return out


def audit_normal_routes(camera, pc, pipe, background, directions, lighting, visibility, iteration, weights):
    """零更新审计：full/detach前向一致；normal→opacity非零→零；normal→xyz/rot/scale保持；RGB/Alpha/dist→opacity保留。"""
    before = param_hashes(pc, lighting)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
    sat = camera.observed_saturation_mask.cuda()
    photo = render_mixed(camera, pc, pipe, background, directions, lighting, visibility, iteration)
    full, fpk = geometry_losses(camera, pc, pipe, background, alpha, photo, 'full')
    stop, spk = geometry_losses(camera, pc, pipe, background, alpha, photo, 'detach_opacity')
    checks = {}
    for k in ('normal', 'alpha', 'distortion'):
        checks['loss_value_' + k] = dict(passed=bool(torch.equal(full[k], stop[k]) and torch.isfinite(full[k])),
                                         value=float(full[k]))
    for k in ('rend_normal', 'surf_normal', 'surf_depth', 'rend_alpha', 'radii'):
        checks['geometry_forward_' + k] = dict(passed=bool(torch.equal(fpk[k], spk[k])),
                                               max_abs=float((fpk[k].float() - spk[k].float()).abs().max()))
    params = [pc._xyz, pc._rotation, pc._scaling, pc._opacity]
    names = ['xyz', 'rotation', 'scaling', 'opacity']
    wn = weights['normal']
    fg = torch.autograd.grad(wn * full['normal'], params, retain_graph=True, allow_unused=True)
    sg = torch.autograd.grad(wn * stop['normal'], params, retain_graph=True, allow_unused=True)
    for name, a, b in zip(names, fg, sg):
        if name == 'opacity':
            zero = b is None or bool(torch.count_nonzero(b) == 0)
            checks['normal_opacity_nonzero_to_zero'] = dict(
                passed=bool(a is not None and torch.isfinite(a).all() and a.norm() > 0 and zero),
                full_norm=float(a.norm()), detached_max_abs=0. if b is None else float(b.abs().max()))
        else:
            ok = a is not None and b is not None and bool(torch.isfinite(b).all() and b.norm() > 0 and
                                                         torch.allclose(a, b, atol=1e-6, rtol=1e-4))
            checks['normal_gradient_' + name + '_preserved'] = dict(
                passed=ok, full_norm=float(a.norm()), detached_norm=float(b.norm()), max_abs=float((a - b).abs().max()))
    rgb = censored_loss(photo['render'], target, alpha, sat)[0]
    pg = torch.autograd.grad(rgb, pc._opacity, retain_graph=True)[0]
    checks['rgb_opacity_preserved'] = dict(passed=bool(torch.isfinite(pg).all() and pg.norm() > 0), norm=float(pg.norm()))
    for k in ('alpha', 'distortion'):
        a = torch.autograd.grad(weights[k] * full[k], pc._opacity, retain_graph=True)[0]
        b = torch.autograd.grad(weights[k] * stop[k], pc._opacity, retain_graph=True)[0]
        checks[k + '_opacity_preserved'] = dict(passed=bool(torch.isfinite(a).all() and a.norm() > 0 and
                                                           torch.allclose(a, b, atol=1e-6, rtol=1e-4)),
                                               norm=float(a.norm()), max_abs=float((a - b).abs().max()))
    lf = rgb + weights['alpha'] * full['alpha'] + wn * full['normal'] + weights['distortion'] * full['distortion']
    ls = rgb + weights['alpha'] * stop['alpha'] + wn * stop['normal'] + weights['distortion'] * stop['distortion']
    tf = torch.autograd.grad(lf, params, retain_graph=True)
    ts = torch.autograd.grad(ls, params)
    checks['total_opacity_difference_equals_normal_term'] = dict(
        passed=bool(torch.allclose(tf[3] - ts[3], fg[3], atol=1e-6, rtol=1e-4)), max_abs=float((tf[3] - ts[3] - fg[3]).abs().max()))
    for i, name in enumerate(names[:3]):
        checks['total_' + name + '_identical'] = dict(passed=bool(torch.allclose(tf[i], ts[i], atol=1e-6, rtol=1e-4)),
                                                     max_abs=float((tf[i] - ts[i]).abs().max()))
    checks['parameters_unchanged'] = dict(passed=before == param_hashes(pc, lighting))
    checks['rng_unchanged'] = dict(passed=bool(torch.equal(cpu_rng, torch.get_rng_state()) and
                                               torch.equal(cuda_rng, torch.cuda.get_rng_state())))
    return dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'FAILED', checks=checks,
                camera=camera.image_name_train_light, parameter_updates=0,
                gradient_tolerance=dict(atol=1e-6, rtol=1e-4), forward_tolerance='逐元素完全一致')


@torch.no_grad()
def audit_physical_forward(camera, pc, pipe, background, directions, lighting, visibility, iteration):
    """分量加和、法线翻面、可见度缓存重建一致性及前向不读目标。"""
    pk = render_mixed(camera, pc, pipe, background, directions, lighting, visibility, iteration, components=True)
    n = oriented_normals(camera, pc)
    facing_ok = bool(((n * (camera.camera_center - pc.get_xyz)).sum(-1) >= 0).all())
    d = directions.for_camera(camera)[None]
    cached = visibility(pc, d, iteration)
    visibility.invalidate()
    rebuilt = visibility(pc, d, iteration)
    import inspect
    sig = list(inspect.signature(render_mixed).parameters)
    checks = dict(
        component_sum=dict(passed=float(pk['component_sum_max_abs']) < 1e-5, max_abs=float(pk['component_sum_max_abs'])),
        normals_face_camera=dict(passed=facing_ok),
        visibility_cache_equals_rebuild=dict(passed=bool(torch.equal(cached, rebuilt)),
                                             max_abs=float((cached - rebuilt).abs().max())),
        visibility_range=dict(passed=bool(((rebuilt >= 0) & (rebuilt <= 1)).all()), mean=float(rebuilt.mean())),
        forward_has_no_target_argument=dict(passed=not any('target' in s or 'image' in s or 'alpha' in s for s in sig),
                                            signature=sig),
        finite=dict(passed=bool(torch.isfinite(pk['render']).all())))
    return dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'FAILED', checks=checks,
                camera=camera.image_name_train_light, direct_fraction=float(lighting.direct_fraction()))
