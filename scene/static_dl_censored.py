"""只改变原训练 RGB 满码通道的光度约束；渲染前向与梯度路由不变。"""
import torch

from scene.static_dl_photometric import robust_photometric_loss


def saturated_channels(target, alpha):
    """固定观测掩膜：精确满码与原图自身前景，绝不依赖预测值。"""
    return ((target.detach() == 1.) & (alpha.detach() > .5)).detach()


def censored_photometric_loss(image, target, alpha, shadow_weight=.25, highlight_weight=.5, threshold=.08):
    residual = image-target
    # 权重沿用旧双边残差，避免同时改变鲁棒分组规则。
    signed = residual.detach().mean(0, keepdim=True)
    weight = torch.ones_like(signed)
    weight = torch.where(signed > threshold, shadow_weight, weight)
    weight = torch.where(signed < -threshold, highlight_weight, weight)
    weight = torch.where(alpha.detach() > .5, weight, torch.ones_like(weight))
    residual = torch.where(saturated_channels(target, alpha), residual.clamp_max(0), residual)
    loss = (torch.sqrt(residual.square()+1e-6)*weight).mean()
    return loss, {'shadow_downweight_fraction': (signed > threshold).float().mean().detach(),
                  'highlight_downweight_fraction': (signed < -threshold).float().mean().detach()}


def loss_for_mode(mode):
    if mode == 'two_sided':
        return robust_photometric_loss
    if mode == 'censored':
        return censored_photometric_loss
    raise ValueError('未知光度损失：'+str(mode))


@torch.no_grad()
def photometric_diagnostics(image, linear_image, target, alpha, pc, settings):
    """固定原观测集合上的诊断，不参与损失或参数更新。"""
    saturated = saturated_channels(target, alpha)
    foreground = (alpha > .5).expand_as(target)
    unsaturated = foreground & ~saturated
    residual = image-target
    clipped_count = saturated.sum().clamp_min(1)
    visible_linear = linear_image[foreground]
    albedo = pc.get_albedo
    old_loss = robust_photometric_loss(image, target, alpha, settings.shadow_weight,
                                       settings.highlight_weight, settings.residual_threshold)[0]
    def quantile(q):
        return float(torch.quantile(visible_linear, q)) if visible_linear.numel() else 0.
    return dict(photo_two_sided_reference=float(old_loss),
        foreground_saturated_channel_fraction=float(saturated.sum()/foreground.sum().clamp_min(1)),
        unsaturated_rgb_mae=float((residual.abs()*unsaturated).sum()/unsaturated.sum().clamp_min(1)),
        saturated_lowerbound_violation_fraction=float(((image<1.) & saturated).sum()/clipped_count),
        saturated_underprediction_mean=float(((1.-image).clamp_min(0)*saturated).sum()/clipped_count),
        prediction_srgb_above_one_fraction=float(((image>1.) & foreground).sum()/foreground.sum().clamp_min(1)),
        prediction_linear_p95=quantile(.95), prediction_linear_p99=quantile(.99),
        prediction_linear_max=float(visible_linear.max()) if visible_linear.numel() else 0.,
        albedo_mean=float(albedo.mean()), albedo_lower_fraction=float((albedo<=.0311).float().mean()),
        albedo_upper_fraction=float((albedo>=.9689).float().mean()))


def audit_real_residual(image, target, alpha, settings):
    """在真实训练 RGB 预测上只核验图像导数，不更新模型。"""
    prediction = image.detach().clone().requires_grad_(True)
    options = (settings.shadow_weight, settings.highlight_weight, settings.residual_threshold)
    old = robust_photometric_loss(prediction, target, alpha, *options)[0]
    new = censored_photometric_loss(prediction, target, alpha, *options)[0]
    old_grad = torch.autograd.grad(old, prediction)[0]
    new_grad = torch.autograd.grad(new, prediction)[0]
    mask = saturated_channels(target, alpha)
    satisfied = mask & (prediction.detach()>=1.)
    other = ~satisfied
    def maximum(value):
        return float(value.abs().max()) if value.numel() else 0.
    changed_correctly = maximum(new_grad[satisfied]) == 0
    preserved = maximum((new_grad-old_grad)[other]) == 0
    finite = bool(torch.isfinite(new_grad).all() and torch.isfinite(old_grad).all())
    return dict(status='PASS' if changed_correctly and preserved and finite else 'FAILED',
        satisfied_saturated_channels=int(satisfied.sum()), saturated_channels=int(mask.sum()),
        satisfied_saturated_gradient_max=maximum(new_grad[satisfied]),
        other_gradient_difference_max=maximum((new_grad-old_grad)[other]),
        old_loss=float(old.detach()), censored_loss=float(new.detach()),
        old_image_gradient_norm=float(old_grad.norm()), censored_image_gradient_norm=float(new_grad.norm()),
        finite=finite, parameter_updates=0)
