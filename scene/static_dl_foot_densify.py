"""只改变增密统计的可选支路下限；前向、实际参数梯度和拓扑算子保持。"""
from types import SimpleNamespace

import torch

from scene.gaussian_model import GaussianModel


@torch.no_grad()
def accumulate_screen_statistics(pc, photo, geometry, route, audit=False):
    if route not in ('original', 'non_normal_floor'):
        raise ValueError('未知增密统计路线。')
    other = photo['viewspace_points'].grad
    normal = geometry['viewspace_points'].grad
    if other is None or normal is None or other.shape != normal.shape:
        raise AssertionError('需要两支路真实屏幕梯度。')
    if not torch.isfinite(other).all() or not torch.isfinite(normal).all():
        raise FloatingPointError('屏幕梯度非有限。')
    selected = photo['visibility_filter']
    if not torch.equal(selected, geometry['visibility_filter']):
        raise AssertionError('两支路可见性必须完全相同。')
    non_normal_norm = torch.norm(other[:, :2], dim=-1, keepdim=True)
    before_accum, before_denom = (pc.xyz_gradient_accum.clone(), pc.denom.clone()) if audit else (None, None)
    parameter_grads = {name: value.grad.clone() for name, value in vars(pc).items()
                       if isinstance(value, torch.nn.Parameter) and value.grad is not None} if audit else {}
    # 与原训练相同，只相加一次屏幕余切；实际参数.grad此前已反传完成。
    other.add_(normal)
    original_norm = torch.norm(other[:, :2], dim=-1, keepdim=True)
    floor_norm = torch.maximum(original_norm, non_normal_norm)
    if route == 'original':
        pc.add_densification_stats(photo['viewspace_points'], selected)
    else:
        pc.xyz_gradient_accum[selected] += floor_norm[selected]
        pc.denom[selected] += 1
    result = dict(route=route, visible_points=int(selected.sum()),
                  floor_active_points=int(((floor_norm > original_norm).squeeze(-1) & selected).sum()),
                  original_norm_sum=float(original_norm[selected].sum()),
                  non_normal_norm_sum=float(non_normal_norm[selected].sum()),
                  floor_norm_sum=float(floor_norm[selected].sum()))
    if audit:
        reference = SimpleNamespace(xyz_gradient_accum=before_accum.clone(), denom=before_denom.clone())
        GaussianModel.add_densification_stats(reference, photo['viewspace_points'], selected)
        expected = before_accum.clone()
        expected[selected] += (original_norm if route == 'original' else floor_norm)[selected]
        checks = dict(
            original_matches_native=torch.equal(reference.xyz_gradient_accum[selected],
                (before_accum + original_norm)[selected]),
            original_invisible_unchanged=torch.equal(reference.xyz_gradient_accum[~selected],before_accum[~selected]),
            candidate_accumulation_exact=torch.equal(pc.xyz_gradient_accum, expected),
            denominator_same_as_native=torch.equal(pc.denom,reference.denom),
            floor_not_smaller=bool((floor_norm >= original_norm).all()),
            candidate_parameter_gradients_unchanged=all(torch.equal(getattr(pc,name).grad,g) for name,g in parameter_grads.items()),
            nonempty_parameter_gradient_audit=bool(parameter_grads),
        )
        if route == 'original':
            checks['original_full_stats_equal'] = torch.equal(pc.xyz_gradient_accum, reference.xyz_gradient_accum)
        result['audit'] = dict(status='PASS' if all(checks.values()) else 'FAILED', checks=checks,
            parameter_gradient_names=list(parameter_grads), actual_updates=0,
            scope='真实首步屏幕余切与原GaussianModel统计算子独立对照；本检查不执行参数更新。')
        if not all(checks.values()):
            raise AssertionError('增密统计路线审计失败：' + str(checks))
    return result
