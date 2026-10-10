#!/usr/bin/env python3
"""从保存的数组独立复算全部可用 MAE，写入总报告读取的摘要。"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(sys.argv[1])
ITERATION = int(sys.argv[2])
expected_ids = list(range(10, 151, 10))
results = {}
errors = []
details = {}


def check_close(actual, expected, tolerance=2e-7):
    if expected is None or not np.isfinite(actual) or not np.isclose(actual, expected, atol=tolerance, rtol=1e-6):
        raise AssertionError(f'独立复算不匹配：{actual} vs {expected}')


def unit(array):
    array = np.asarray(array, dtype=np.float64)
    length = np.linalg.norm(array, axis=-1)
    valid = np.isfinite(array).all(-1) & (length > 1e-6)
    normalized = np.zeros_like(array)
    np.divide(array, length[..., None], out=normalized, where=valid[..., None])
    return normalized, valid


def geometry():
    metrics = json.loads((ROOT / 'normal_depth/metrics.json').read_text())
    if [r['source_frame'] for r in metrics['frames']] != expected_ids:
        raise AssertionError('几何评测必须覆盖15测试帧。')
    totals = {k: [0, 0., 0] for k in ('gs_render', 'depth_normal', 'mesh', 'expected', 'median')}
    common_total = {k: [0, 0.] for k in ('gs_render', 'depth_normal', 'mesh')}
    for row in metrics['frames']:
        frame = row['source_frame']
        with np.load(ROOT / 'normal_depth' / f'prediction_{frame:04d}.npz') as data:
            gt, valid_gt = unit(data['gt_normal'])
            eligible = data['gt_foreground'].astype(bool) & valid_gt
            visible = np.isfinite(data['alpha']) & (data['alpha'] >= .5)
            masks, angulars = {}, {}
            for key, field in [('gs_render', 'normal'), ('depth_normal', 'depth_normal'), ('mesh', 'mesh_normal')]:
                prediction, valid_prediction = unit(data[field])
                support = data['mesh_valid'] if key == 'mesh' else visible
                if key == 'depth_normal':
                    support = binary_erosion(eligible & visible, structure=np.ones((3, 3)), border_value=0)
                mask = eligible & support & valid_prediction
                error = np.degrees(np.arccos(np.clip((prediction * gt).sum(-1), -1, 1)))
                count = int(mask.sum())
                if count:
                    check_close(error[mask].mean(), row['regions']['foreground']['normal_metrics'][key]['angular_deg']['mean'], 1e-8)
                totals[key][0] += count
                totals[key][1] += float(error[mask].sum())
                totals[key][2] += int(eligible.sum())
                masks[key], angulars[key] = mask, error
            common = masks['gs_render'] & masks['depth_normal'] & masks['mesh']
            for key in common_total:
                common_total[key][0] += int(common.sum())
                common_total[key][1] += float(angulars[key][common].sum())
            gt_depth = data['gt_depth']
            depth_eligible = data['gt_foreground'].astype(bool) & np.isfinite(gt_depth) & (gt_depth > 0)
            for key in ('expected', 'median'):
                prediction = data[key]
                mask = depth_eligible & visible & np.isfinite(prediction) & (prediction > 0)
                totals[key][0] += int(mask.sum())
                totals[key][1] += float(np.abs(prediction[mask] - gt_depth[mask]).sum())
                totals[key][2] += int(depth_eligible.sum())
    summary = metrics['summary']['foreground']
    for key, (count, total, eligible) in totals.items():
        if not count or not eligible:
            raise AssertionError('缺少可评分像素，不能报告该MAE通过：' + key)
        score = total / count if count else None
        definition = summary['normal_metrics'][key] if key in common_total else summary['depth'][key]
        if count:
            check_close(score, definition['mae_pixel_mean_deg' if key in common_total else 'mae_pixel_mean'])
            check_close(count / eligible, definition['coverage'])
        results[key] = score
        details[key] = dict(valid_pixels=count, eligible_pixels=eligible,
                            coverage=count / eligible if eligible else 0,
                            common_mae_deg=common_total[key][1] / common_total[key][0]
                            if key in common_total and common_total[key][0] else None)
    meshes = json.loads((ROOT / 'predicted_meshes/manifest.json').read_text())
    if meshes['gt_geometry_used'] is not False or meshes['test_camera_used_for_reconstruction'] is not False:
        raise AssertionError('预测Mesh来源不合规。')
    train = json.loads((Path(meshes['source_path']) / 'transforms_train.json').read_text())['frames']
    train_ids = {r['source_frame'] for r in train}
    if len(train_ids) != 135 or not train_ids.isdisjoint(expected_ids):
        raise AssertionError('训练／测试划分不符。')
    for record in meshes['frames'].values():
        if len(record['integrations']) != 135 or {r['source_frame'] for r in record['integrations']} != train_ids:
            raise AssertionError('Mesh必须只融合全部135训练相机。')
    for path, digest in metrics['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise AssertionError('来源文件发生改变：' + path)


def albedo():
    metrics = json.loads((ROOT / f'albedo/iteration_{ITERATION}/metrics.json').read_text())
    if len(metrics['frames']) != 15:
        raise AssertionError('Albedo评测未覆盖15测试帧。')
    total, count = 0., 0
    for row in metrics['frames']:
        with np.load(ROOT / f'albedo/iteration_{ITERATION}' / (row['frame'] + '.npz')) as data:
            core = binary_erosion(data['gt_alpha'] >= .99, structure=np.ones((5, 5)), border_value=0)
            np.testing.assert_array_equal(core, data['gt_core'])
            prediction = data['albedo_linear'].astype(np.float64)
            gt = data['gt_albedo_linear'].astype(np.float64)
            total += float(np.abs(prediction[:, core] - gt[:, core]).mean(0).sum())
            count += int(core.sum())
    results['albedo'] = total / count
    check_close(results['albedo'], metrics['summaries']['raw']['foreground']['mae_linear'])
    details['albedo'] = metrics['summaries']['raw']['foreground']


for name, operation in [('geometry', geometry)] + ([('albedo', albedo)] if ITERATION == 55000 else []):
    try:
        operation()
    except Exception as error:
        errors.append(f'{name}：{type(error).__name__}：{error}')
roughness = ROOT / 'roughness/metrics.json'
if roughness.exists():
    metrics = json.loads(roughness.read_text())
    results['roughness'] = metrics['summaries']['foreground']['mae']
    details['roughness'] = metrics['summaries']['foreground']
    verification = list((ROOT / 'roughness').glob('*verif*.json'))
    if not verification:
        errors.append('Roughness缺少独立验证JSON。')
results.update(status='FAILED' if errors else 'PASS', iteration=ITERATION, test_frames=15,
               details=details, errors=errors)
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'mae_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
if errors:
    sys.exit(1)
