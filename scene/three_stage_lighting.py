"""TRAIN共享法线SH与受约束方向光；所有混合均在线性颜色域完成。"""
import math

import torch
import torch.nn.functional as F

from scene.stage2_photometric import frame_key, srgb_to_linear, linear_to_srgb
from utils.general_utils import build_rotation


def normal_basis(n):
    """一阶实SH的等价归一基：共享DC固定，三个一阶项各RGB共享全场。"""
    return n


class SharedLighting(torch.nn.Module):
    def __init__(self, asset, direction_lr=0., device='cuda'):
        super().__init__()
        if asset['gt_read'] is not False or asset['source_split'] != 'TRAIN':
            raise ValueError('灯向只允许TRAIN估计资产。')
        self.ids = sorted(map(int, asset['frames']))
        if self.ids != [i for i in range(1, 151) if i % 10]:
            raise ValueError('必须完整135TRAIN方向，禁止TEST灯向。')
        initial = torch.tensor([asset['frames'][f'{i:04d}']['light_dir_world'] for i in self.ids], device=device)
        if not torch.isfinite(initial).all() or (initial.norm(dim=-1)-1).abs().max() > 1e-4:
            raise ValueError('非法方向资产。')
        self.register_buffer('initial', F.normalize(initial, dim=-1))
        self.raw_direction = torch.nn.Parameter(initial.clone(), requires_grad=direction_lr > 0)
        self.sh = torch.nn.Parameter(torch.zeros(3, 3, device=device))
        # 两分支使用独立于lambda的固定参考幅度，且不允许学习DC/光强抵消lambda。
        self.register_buffer('reference', torch.tensor(2*math.pi, device=device))

    def directions(self):
        return F.normalize(self.raw_direction, dim=-1)

    def for_camera(self, camera):
        number = int(frame_key(camera.image_name_train_light))
        ids = self.ids
        if number <= ids[0]:
            return self.directions()[0:1]
        if number >= ids[-1]:
            return self.directions()[-1:]
        right = next(i for i, n in enumerate(ids) if n >= number)
        if ids[right] == number:
            return self.directions()[right:right+1]
        left = right-1
        w = (number-ids[left])/(ids[right]-ids[left])
        return F.normalize(((1-w)*self.directions()[left]+w*self.directions()[right])[None], dim=-1)

    def environment(self, normals):
        # |n|=1, 每个RGB的一阶幅度总和至多.25；辐照度保持[.75,1.25]*2pi。
        coefficients = .25/math.sqrt(3)*self.sh.tanh()
        return self.reference*(1+normal_basis(normals)@coefficients)

    def direction_regularization(self):
        d = self.directions()
        anchor = (d-self.initial).square().sum(-1).mean()
        dt = d.new_tensor([b-a for a, b in zip(self.ids[:-1], self.ids[1:])])[:, None]
        # 保护原估计的时间变化，只平滑新增偏移，不把真实时变方向压成常量。
        delta = d-self.initial
        smooth = ((delta[1:]-delta[:-1])/dt).square().sum(-1).mean()
        return anchor, smooth

    @torch.no_grad()
    def project(self, max_degrees=.5):
        d = self.directions()
        dot = (d*self.initial).sum(-1, keepdim=True).clamp(-1, 1)
        delta = d-dot*self.initial
        angle = torch.atan2(delta.norm(dim=-1, keepdim=True), dot)
        tangent = F.normalize(delta, dim=-1)
        limit = math.radians(max_degrees)
        boundary = self.initial*math.cos(limit)+tangent*math.sin(limit)
        # 小角度不能用acos(dot)舍入为0再重置，否则1e-6更新会被全部吞掉。
        projected = torch.where(angle > limit, boundary, d)
        self.raw_direction.copy_(F.normalize(projected, dim=-1))


class EffectiveGaussians:
    """保留B00冻结网络产生的逐帧几何，不将其无声丢弃或重新训练。"""
    def __init__(self, pc, values):
        self.pc, self.values = pc, values

    def __getattr__(self, key):
        return getattr(self.pc, key)

    @property
    def get_xyz(self):
        return self.pc.get_xyz+self.values['d_xyz']

    @property
    def get_rotation(self):
        return self.pc.get_rotation_bias(self.values['d_rotation'])

    def get_rotation_bias(self, bias=None):
        if bias is None or (isinstance(bias, (int, float)) and bias == 0):
            return self.get_rotation
        return F.normalize(self.get_rotation+bias, dim=-1)

    @property
    def get_scaling(self):
        return self.pc.get_scaling+self.values['d_scaling']

    @property
    def get_opacity(self):
        value = self.values['d_opacity']
        return self.pc.get_opacity+(0. if value is None else value)


def surface_components(camera, pc, lights, visibility, step):
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    facing = torch.where((normal*(camera.camera_center-pc.get_xyz)).sum(-1, keepdim=True) >= 0, 1., -1.).detach()
    normal = normal*facing
    directions = lights.for_camera(camera)
    # BVH每次前向按有效几何重建。V无反向链路，方向只通过余弦得到近似梯度。
    visibility.invalidate()
    vis = visibility(pc, directions.detach(), step)
    rho = srgb_to_linear(pc.get_albedo)
    environment = rho*lights.environment(normal)/math.pi
    direct = rho*lights.reference*vis*(normal@directions.T).clamp_min(0)/math.pi
    return environment, direct, vis


def render_stage2(camera, pc, pipe, lights, visibility, step, lambda_mix, components=False):
    from gaussian_renderer import render
    zero = pc.get_xyz.new_zeros(3)
    env, direct, vis = surface_components(camera, pc, lights, visibility, step)
    weighted_env, weighted_direct = (1-lambda_mix)*env, lambda_mix*direct
    package = render(camera, pc, pipe, zero, 0., 0., 0., override_color=weighted_env+weighted_direct)
    package['render_linear'] = package['render']
    package['render'] = linear_to_srgb(package['render_linear'].clamp_min(0))
    # 训练残差需要EL；诊断额外输出DL，并核验同一光栅的线性加和。
    el = render(camera, pc, pipe, zero, 0., 0., 0., override_color=weighted_env)['render']
    package['base_linear'] = el
    package['residual_linear'] = package['render_linear']-el
    package['visibility_mean'] = vis.mean().detach()
    if components:
        dl = render(camera, pc, pipe, zero, 0., 0., 0., override_color=weighted_direct)['render']
        package['component_add_max_error'] = (el+dl-package['render_linear']).abs().max()
        package['residual_linear'] = dl
    return package


def censored_error(prediction, target, saturated):
    residual = prediction-target
    residual = torch.where(saturated, residual.clamp_max(0), residual)
    return (residual.square()+1e-6).sqrt()-1e-3


def brightness_weights(target_linear, stats):
    y = (target_linear*target_linear.new_tensor([.2126, .7152, .0722])[:, None, None]).sum(0, keepdim=True)
    width = stats['transition_width']
    base = torch.sigmoid((stats['q_base_value']-y)/width)
    bright = torch.sigmoid((y-stats['q_bright_value'])/width)
    return y, base, bright


def photometric_objective(package, target, alpha, saturated, stats, base_weight=.25, residual_weight=.1, support=None):
    observed = srgb_to_linear(target)
    y, base, bright = brightness_weights(observed, stats)
    fg = (alpha > .5).float()
    support = torch.ones_like(fg) if support is None else support.float()
    fg = fg*support
    denominator = (3*fg.sum()).clamp_min(1)
    error = censored_error(package['render_linear'], observed, saturated)
    total = (error*fg).sum()/denominator
    # 背景和Alpha继续监督完整重建，不能仅在亮区训练。
    total = total+.1*(error*(alpha <= .5)*support).sum()/(3*((alpha <= .5)*support).sum()).clamp_min(1)
    base_loss = (censored_error(package['base_linear'], observed, saturated)*base*fg).sum()/denominator
    positive = (observed-package['base_linear'].detach()).clamp_min(0)
    residual_loss = (censored_error(package['residual_linear'], positive, saturated)*bright*fg).sum()/denominator
    loss = total+base_weight*base_loss+residual_weight*residual_loss
    segments = {}
    for name, mask in [('dark', y < stats['q50']), ('body', y <= stats['q_base_value']), ('bright', y >= stats['q_bright_value'])]:
        mask = mask*fg
        segments[name+'_linear_mae'] = (error*mask).sum()/(3*mask.sum()).clamp_min(1)
    return loss, dict(full_linear=total, base=base_loss, residual=residual_loss, **segments)
