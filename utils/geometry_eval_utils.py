"""几何评测的 CPU 数值计算；法线为世界坐标，深度为正的 camera Z。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_number(name):
    match = re.search(r'(\d+)$', Path(name).stem)
    if match is None:
        raise ValueError(f'文件名没有帧号：{name}')
    return int(match.group(1))


def unit_normal(normal):
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal, axis=-1)
    valid = np.isfinite(normal).all(-1) & (norm > 1e-6)
    result = np.zeros_like(normal)
    np.divide(normal, norm[..., None], out=result, where=valid[..., None])
    return result, valid


def load_normal_exr(path, height, width):
    import OpenEXR
    channels = OpenEXR.File(str(path), separate_channels=True).channels()
    keys = ('Normal.X', 'Normal.Y', 'Normal.Z')
    if not all(k in channels for k in keys):
        keys = ('R', 'G', 'B')
    if not all(k in channels for k in keys):
        raise ValueError(f'不支持的法线通道：{list(channels)}')
    array = np.stack([channels[k].pixels for k in keys], axis=-1)
    if array.shape[:2] != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    return unit_normal(array)


def camera_rays(c2w, intrinsics, height, width):
    """像素中心 +0.5；Blender 的局部 -Z 前向。方向不单位化，t 即 Z 深度。"""
    y, x = np.mgrid[:height, :width]
    fx, fy, cx, cy = intrinsics
    local = np.stack(((x + .5 - cx) / fx, -(y + .5 - cy) / fy,
                      -np.ones_like(x)), axis=-1)
    direction = local @ np.asarray(c2w)[:3, :3].T
    origin = np.broadcast_to(np.asarray(c2w)[:3, 3], direction.shape)
    return np.concatenate((origin, direction), axis=-1).astype(np.float32)


def depth_normals(depth, c2w, intrinsics):
    """由中心差分得到 camera-facing 世界法线，边界和无效邻域不计分。"""
    depth = np.asarray(depth)
    rays = camera_rays(c2w, intrinsics, *depth.shape)
    points = rays[..., :3] + depth[..., None] * rays[..., 3:]
    normals = np.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normals[1:-1, 1:-1] = np.cross(dx, dy)
    normals, valid = unit_normal(normals)
    valid &= erode(np.isfinite(depth) & (depth > 0), 1)
    normals[~valid] = 0
    return normals, valid


def erode(mask, radius):
    if radius < 0:
        raise ValueError('腐蚀半径必须非负。')
    if radius == 0:
        return np.asarray(mask, dtype=bool).copy()
    size = 2 * radius + 1
    return cv2.erode(np.asarray(mask, dtype=np.uint8), np.ones((size, size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def normal_errors(predicted, target):
    predicted, pred_valid = unit_normal(predicted)
    target, gt_valid = unit_normal(target)
    dot = np.clip(np.sum(predicted * target, axis=-1), -1, 1)
    return dict(angular_deg=np.degrees(np.arccos(dot)),
                axis_deg=np.degrees(np.arccos(np.abs(dot))),
                component_l1=np.abs(predicted - target).mean(-1),
                cosine_loss=1-dot), pred_valid & gt_valid


def summarize(values, mask):
    samples = np.asarray(values, dtype=np.float64)[mask]
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return dict(count=0, mean=None, median=None, p95=None, sum=0.)
    return dict(count=len(samples), mean=float(samples.mean()),
                median=float(np.median(samples)), p95=float(np.quantile(samples, .95)),
                sum=float(samples.sum()))


def evaluate_region(prediction, gt_normal, gt_normal_valid, gt_depth, gt_mask,
                    alpha_threshold=.5):
    """误差按有效交集计算；GT 定义分母，并显式报告漏预测率。"""
    eligible = np.asarray(gt_mask, bool) & gt_normal_valid
    visible = np.isfinite(prediction['alpha']) & (prediction['alpha'] >= alpha_threshold)
    errors, normal_valid = normal_errors(prediction['normal'], gt_normal)
    normal_mask = eligible & visible & normal_valid
    report = dict(gt_pixels=int(eligible.sum()),
                  normal_pixels=int(normal_mask.sum()), normal={}, depth={})
    report['normal_coverage'] = float(normal_mask.sum() / max(1, eligible.sum()))
    for key, values in errors.items():
        report['normal'][key] = summarize(values, normal_mask)
    # 包含漏预测的角度版本：缺失/无效法线按最差 180°计入。
    penalized = np.where(visible & normal_valid, errors['angular_deg'], 180.)
    report['normal']['angular_with_missing_deg'] = summarize(penalized, eligible)
    if 'depth_normal' in prediction:
        derived, derived_valid = normal_errors(prediction['depth_normal'], gt_normal)
        derived_mask = erode(eligible & visible, 1) & derived_valid
        report['depth_normal_coverage'] = float(derived_mask.sum() / max(1, eligible.sum()))
        report['normal']['depth_derived_angular_deg'] = summarize(derived['angular_deg'], derived_mask)
        report['normal']['depth_derived_axis_deg'] = summarize(derived['axis_deg'], derived_mask)
    # 明确三类模型输出。旧 normal 字段保留，避免历史汇总脚本含义变化。
    method_specs = (
        ('gs_render', 'GS Render Normal MAE', 'normal', visible),
        ('depth_normal', 'Depth Normal MAE', 'depth_normal', erode(eligible & visible, 1)),
        ('mesh', 'Mesh Normal MAE', 'mesh_normal', prediction.get('mesh_valid', np.ones_like(eligible))),
    )
    report['normal_metrics'] = {}
    available = {}
    common = eligible.copy()
    for name, label, field, support in method_specs:
        item = dict(label=label, reference='GT unit world-space Normal EXR',
                    gt_pixels=int(eligible.sum()), valid_pixels=0, coverage=None,
                    status='not_available', angular_deg=summarize([], np.zeros(0, bool)))
        if field in prediction:
            error, valid = normal_errors(prediction[field], gt_normal)
            mask = eligible & np.asarray(support, bool) & valid
            item.update(status='evaluated', valid_pixels=int(mask.sum()),
                        coverage=float(mask.sum()/max(1, eligible.sum())),
                        angular_deg=summarize(error['angular_deg'], mask),
                        axis_deg=summarize(error['axis_deg'], mask),
                        angular_with_missing_deg=summarize(np.where(mask, error['angular_deg'], 180.), eligible))
            available[name] = error
            common &= mask
        report['normal_metrics'][name] = item
    report['common_normal_metrics'] = dict(methods=list(available), gt_pixels=int(eligible.sum()),
        valid_pixels=int(common.sum()) if available else 0,
        coverage=float(common.sum()/max(1, eligible.sum())) if available else None,
        angular_deg={name: summarize(error['angular_deg'], common) for name, error in available.items()})
    if gt_depth is not None:
        depth_gt_valid = np.asarray(gt_mask, bool) & np.isfinite(gt_depth) & (gt_depth > 0)
        report['depth_gt_pixels'] = int(depth_gt_valid.sum())
        for key in ('expected', 'median'):
            depth = prediction[key]
            mask = depth_gt_valid & visible & np.isfinite(depth) & (depth > 0)
            absolute = np.abs(depth - gt_depth)
            relative = absolute / np.maximum(gt_depth, 1e-12)
            report['depth'][key] = dict(absolute=summarize(absolute, mask),
                relative=summarize(relative, mask),
                coverage=float(mask.sum() / max(1, depth_gt_valid.sum())),
                bias=summarize(depth-gt_depth, mask)['mean'])
    return report


def aggregate_regions(frames):
    """明确区分逐帧等权 MAE 与所有有效像素加权 MAE。"""
    output = {}
    for region in frames[0]['regions']:
        records = [f['regions'][region] for f in frames]
        result = dict(frames=len(records), gt_pixels=sum(r['gt_pixels'] for r in records), normal={}, depth={})
        denominator = result['gt_pixels']
        result['normal_coverage'] = sum(r['normal_pixels'] for r in records) / max(1, denominator)
        for name in records[0]['normal']:
            stats = [r['normal'][name] for r in records]
            means = [s['mean'] for s in stats if s['count']]
            count = sum(s['count'] for s in stats)
            result['normal'][name] = dict(frame_mean=float(np.mean(means)) if means else None,
                pixel_mean=sum(s['sum'] for s in stats)/count if count else None,
                valid_frames=len(means), valid_pixels=count)
        for name in records[0]['depth']:
            stats = [r['depth'][name] for r in records]
            count = sum(s['absolute']['count'] for s in stats)
            means = [s['absolute']['mean'] for s in stats if s['absolute']['count']]
            result['depth'][name] = dict(mae_frame_mean=float(np.mean(means)) if means else None,
                mae_pixel_mean=sum(s['absolute']['sum'] for s in stats)/count if count else None,
                abs_rel_frame_mean=float(np.mean([s['relative']['mean'] for s in stats if s['relative']['count']])) if count else None,
                coverage=count/max(1, sum(r['depth_gt_pixels'] for r in records)),
                valid_frames=len(means), valid_pixels=count)
        if all('normal_metrics' in r for r in records):
            result['normal_metrics'] = {}
            for name, example in records[0]['normal_metrics'].items():
                items = [r['normal_metrics'][name] for r in records]
                stats = [i['angular_deg'] for i in items]
                means = [s['mean'] for s in stats if s['count']]
                count = sum(s['count'] for s in stats)
                present = any(i['status'] == 'evaluated' for i in items)
                result['normal_metrics'][name] = dict(label=example['label'], reference=example['reference'],
                    status='evaluated' if present else 'not_available',
                    mae_frame_mean_deg=float(np.mean(means)) if means else None,
                    mae_pixel_mean_deg=sum(s['sum'] for s in stats)/count if count else None,
                    coverage=count/max(1, sum(i['gt_pixels'] for i in items)) if present else None,
                    gt_pixels=sum(i['gt_pixels'] for i in items), valid_pixels=count, valid_frames=len(means))
            methods = list(records[0]['common_normal_metrics']['methods'])
            if all(r['common_normal_metrics']['methods'] == methods for r in records):
                common_count = sum(r['common_normal_metrics']['valid_pixels'] for r in records)
                result['common_normal_metrics'] = dict(methods=methods,
                    gt_pixels=sum(r['common_normal_metrics']['gt_pixels'] for r in records),
                    valid_pixels=common_count, coverage=common_count/max(1, denominator), angular_deg={})
                for name in methods:
                    stats = [r['common_normal_metrics']['angular_deg'][name] for r in records]
                    means = [s['mean'] for s in stats if s['count']]
                    result['common_normal_metrics']['angular_deg'][name] = dict(
                        mae_frame_mean_deg=float(np.mean(means)) if means else None,
                        mae_pixel_mean_deg=sum(s['sum'] for s in stats)/common_count if common_count else None)
        output[region] = result
    return output
