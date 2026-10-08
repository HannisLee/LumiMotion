"""可在 CPU 复核的 albedo 评测公式，输入均为 CHW 线性 RGB。"""
import numpy as np
from scipy.ndimage import binary_erosion


def srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= 0.04045, value / 12.92,
                    ((np.maximum(value, 0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= 0.0031308, 12.92 * value,
                    1.055 * np.maximum(value, 0.0031308) ** (1 / 2.4) - 0.055)


def normalize_accumulated(accumulated, alpha, epsilon=1e-6):
    accumulated = np.asarray(accumulated, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 2:
        alpha = alpha[None]
    if accumulated.shape != (3,) + alpha.shape[1:] or alpha.shape[0] != 1:
        raise ValueError('albedo 必须为 3HW，alpha 必须为 1HW 或 HW。')
    if not np.isfinite(accumulated).all() or not np.isfinite(alpha).all():
        raise ValueError('渲染中存在非有限值。')
    if (alpha < 0).any() or (alpha > 1 + 1e-6).any():
        raise ValueError('alpha 超出 [0,1]。')
    # 不按预测 alpha 删除评分像素；完全漏掉的 GT 前景记为零预测。
    return np.where(alpha > epsilon, accumulated / np.maximum(alpha, epsilon), 0.0)


def interior_mask(alpha, threshold=0.99, erosion_pixels=2):
    if not 0 <= threshold <= 1 or erosion_pixels < 0:
        raise ValueError('GT alpha 阈值或收缩像素非法。')
    result = np.asarray(alpha).squeeze(0) if np.asarray(alpha).ndim == 3 else np.asarray(alpha)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError('GT alpha 必须为有限 HW 数组。')
    result = result >= threshold
    if erosion_pixels:
        result = binary_erosion(result, structure=np.ones((3, 3)), iterations=erosion_pixels)
    return result


def material_masks(target, core, palette, tolerance_srgb=3 / 255, erosion_pixels=1):
    """仅使用 GT 颜色生成区域；容差内最近颜色归类，混合边界不归类。"""
    colors = np.stack([item['srgb8'] for item in palette]) / 255.0
    image = linear_to_srgb(target)
    distances = np.max(np.abs(image[None] - colors[:, :, None, None]), axis=1)
    nearest = distances.argmin(0)
    close = distances.min(0) <= tolerance_srgb
    masks = {}
    for index, item in enumerate(palette):
        mask = core & close & (nearest == index)
        if erosion_pixels:
            mask = binary_erosion(mask, structure=np.ones((3, 3)), iterations=erosion_pixels)
        masks[item['name']] = mask
    return masks


def summarize_errors(prediction, target, mask, alpha, coverage_threshold=0.9):
    mask = np.asarray(mask, dtype=bool)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != (3,) + mask.shape or target.shape != prediction.shape:
        raise ValueError('预测、真值和掩码形状不一致。')
    if not np.isfinite(prediction[:, mask]).all() or not np.isfinite(target[:, mask]).all():
        raise ValueError('评分区域存在非有限 albedo。')
    if not mask.any():
        return None
    pred = prediction[:, mask].T
    gt = target[:, mask].T
    delta = pred - gt
    errors = np.abs(delta).mean(1)
    covered = np.asarray(alpha).reshape(mask.shape)[mask] >= coverage_threshold
    luma = np.array([0.2126, 0.7152, 0.0722])
    norms = np.linalg.norm(pred, axis=1) * np.linalg.norm(gt, axis=1)
    angles = np.degrees(np.arccos(np.clip((pred * gt).sum(1) / np.maximum(norms, 1e-12), -1, 1)))
    return {
        'pixels': int(mask.sum()),
        'mae_linear': float(errors.mean()),
        'rmse_linear': float(np.sqrt(np.square(delta).mean())),
        'p95_pixel_mae_linear': float(np.percentile(errors, 95)),
        'below_0_05_pct': float((errors < 0.05).mean() * 100),
        'coverage_pct': float(covered.mean() * 100),
        'high_alpha_mae_linear_diagnostic': float(errors[covered].mean()) if covered.any() else None,
        'bias_rgb_linear': delta.mean(0).tolist(),
        'mean_prediction_rgb_linear': pred.mean(0).tolist(),
        'mean_gt_rgb_linear': gt.mean(0).tolist(),
        'prediction_std_rgb_linear': pred.std(0).tolist(),
        'chromaticity_angle_deg_diagnostic': float(angles.mean()),
        'relative_luminance_mae_diagnostic': float(np.abs(delta @ luma).sum() / max((gt @ luma).sum(), 1e-12)),
    }


class ScaleFit:
    """训练帧真值上的非负最小二乘；拟合结果只能作校准诊断。"""
    def __init__(self):
        self.xy = np.zeros(3, dtype=np.float64)
        self.xx = np.zeros(3, dtype=np.float64)
        self.pixels = 0

    def update(self, prediction, target, mask):
        pred, gt = prediction[:, mask], target[:, mask]
        self.xy += (pred * gt).sum(1)
        self.xx += np.square(pred).sum(1)
        self.pixels += int(mask.sum())

    def result(self):
        if self.pixels == 0 or (self.xx <= 0).any():
            raise ValueError('训练校准没有足够的非零预测。')
        return {
            'scalar': float(max(self.xy.sum() / self.xx.sum(), 0.0)),
            'rgb': np.maximum(self.xy / self.xx, 0.0).tolist(),
            'train_pixels': self.pixels,
        }
