"""CPU 可独立复算的单通道 roughness 条件 MAE 与覆盖率。"""
import numpy as np


def scalar_image(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('roughness/alpha 必须为有限 HW 或 1HW 数组。')
    return value


def normalize_roughness(accumulated, alpha):
    accumulated, alpha = scalar_image(accumulated), scalar_image(alpha)
    if accumulated.shape != alpha.shape or (alpha < 0).any() or (alpha > 1 + 1e-6).any():
        raise ValueError('累积值与 alpha 形状或范围非法。')
    result = np.where(alpha > 1e-6, accumulated / np.maximum(alpha, 1e-6), 0.0)
    if (result < -1e-6).any() or (result > 1 + 1e-5).any():
        raise ValueError('归一化 roughness 超出 [0,1]。')
    return result


def evaluate_region(prediction, target, mask, alpha, threshold=0.9):
    prediction, target, alpha = map(scalar_image, (prediction, target, alpha))
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != mask.shape or target.shape != mask.shape or alpha.shape != mask.shape:
        raise ValueError('预测、GT、mask、alpha 形状不匹配。')
    if not 0 < threshold <= 1 or (alpha < 0).any() or (alpha > 1 + 1e-6).any():
        raise ValueError('alpha/阈值超出合法范围。')
    if ((prediction < -1e-6) | (prediction > 1 + 1e-5)).any() or ((target < 0) | (target > 1)).any():
        raise ValueError('roughness 超出合法范围。')
    valid = mask & (alpha >= threshold)
    return counts_summary(int(mask.sum()), int(valid.sum()),
                          float(np.abs(prediction[valid] - target[valid]).sum(dtype=np.float64)))


def counts_summary(gt_pixels, valid_pixels, absolute_error_sum):
    return {'gt_pixels': gt_pixels, 'valid_pixels': valid_pixels,
            'absolute_error_sum': absolute_error_sum,
            'mae': absolute_error_sum / valid_pixels if valid_pixels else None,
            'coverage_pct': 100 * valid_pixels / gt_pixels if gt_pixels else None}


def aggregate(records):
    """只对全部前景的有效像素汇总，不使用材质或物体标签。"""
    values = [r['foreground'] for r in records]
    return {'foreground': counts_summary(sum(v['gt_pixels'] for v in values),
            sum(v['valid_pixels'] for v in values), sum(v['absolute_error_sum'] for v in values))}


def load_target(path, size):
    """NPZ为独立r/alpha；RGBA EXR为Blender预乘alpha的原始数值，无颜色变换。"""
    from pathlib import Path
    import cv2
    path = Path(path)
    metadata = {}
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            roughness = scalar_image(data['roughness'])
            alpha = scalar_image(data['alpha'])
            for key in ('time', 'camera_to_world'):
                if key in data:
                    metadata[key] = data[key].tolist()
    elif path.suffix.lower() == '.exr':
        import OpenEXR
        channels = OpenEXR.File(str(path), separate_channels=True).channels()
        if 'A' not in channels:
            raise ValueError('GT EXR 必须包含真实 alpha：' + str(path))
        alpha = scalar_image(channels['A'].pixels)
        if 'roughness' in channels:
            roughness = scalar_image(channels['roughness'].pixels)
            metadata['alpha_association'] = 'independent_scalar_channel'
        elif 'R' in channels:
            roughness = scalar_image(channels['R'].pixels)
            for name in ('G', 'B'):
                if name in channels and not np.allclose(roughness, channels[name].pixels, atol=1e-6, rtol=0):
                    raise ValueError('GT roughness EXR 的 RGB 必须为相同标量。')
            roughness = normalize_roughness(roughness, alpha)
            # 仅吸收EXR浮点舍入，拒绝超出 normalize_roughness 容差的数据。
            roughness = np.clip(roughness, 0, 1)
            metadata['alpha_association'] = 'premultiplied_RGB'
        else:
            raise ValueError('GT EXR 缺少 roughness/R 通道。')
    else:
        raise ValueError('GT roughness 仅接受原始浮点 NPZ/EXR，不接受展示 PNG。')
    if roughness.shape != alpha.shape or ((roughness < 0) | (roughness > 1)).any() or ((alpha < 0) | (alpha > 1)).any():
        raise ValueError('GT roughness/alpha 形状或 [0,1] 范围非法。')
    metadata['original_shape'] = list(roughness.shape)
    if roughness.shape != tuple(size):
        height, width = size
        method = cv2.INTER_AREA if height <= alpha.shape[0] and width <= alpha.shape[1] else cv2.INTER_LINEAR
        accumulated = cv2.resize(roughness * alpha, (width, height), interpolation=method)
        alpha = cv2.resize(alpha, (width, height), interpolation=method)
        roughness = normalize_roughness(accumulated, alpha)
    return roughness, alpha, metadata
