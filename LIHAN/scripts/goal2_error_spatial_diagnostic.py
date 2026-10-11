#!/usr/bin/env python3
"""只读CPU离线分区诊断；GT对象/深度不得进入训练，不改原主指标。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.geometry_eval_utils import unit_normal, camera_rays, file_sha256

FRAMES = list(range(10, 151, 10))
FIELDS = {'gs_render': 'normal', 'depth_normal': 'depth_normal', 'mesh': 'mesh_normal'}
FROZEN = {
    'scripts/eval_geometry_gt.py': '181bbf006b9c032b9728e193ec6e5094d061baad30be503182c32e2512775e9e',
    'scripts/extract_geometry_mesh_dynamic.py': '810c134e4e96151d943f619ee5fff7776d566b8b854ffc5dc779301e08d86da8',
    'utils/geometry_eval_utils.py': '4a8a373af98736d7780df41c446d137dcf3433e8e4ccacc50f24e2a48a344935',
    'utils/predicted_mesh.py': '1a6f367d30d752a74beec53021d581a41302c77201236237562f4aa6417c6a73',
    'LIHAN/scripts/audit_1011_mae.py': '60ce2db67b6a32f599c95658f6aa390d4256d20d53d4097594eea38c6471caa7',
    'LIHAN/scripts/bootstrap_1011.py': '693be49305394f8c07ba67421f1d74ee5cf9d4ec8ed425aebfdf0b6355558783',
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def describe(values):
    values = np.asarray(values)
    return {'count': int(values.size), 'mean': float(values.mean()),
            'p10': float(np.quantile(values, .1)), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9))} if values.size else {'count': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--gt_depth_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    evaluation, output = args.evaluation.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise ValueError('诊断输出必须位于output。')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'arrays').mkdir()
    frozen = {str(ROOT / path): file_sha256(ROOT / path) for path in FROZEN}
    for relative, sha in FROZEN.items():
        if frozen[str(ROOT / relative)] != sha:
            raise AssertionError('原六评测源码发生变化：' + relative)
    primary = json.loads((evaluation / 'goal_mae_summary.json').read_text())
    metrics = json.loads((evaluation / 'normal_depth/metrics.json').read_text())
    if primary['completeness'] != 'PASS' or [r['source_frame'] for r in metrics['frames']] != FRAMES:
        raise AssertionError('必须来自完整15帧原协议评测。')
    hashes = {str(evaluation / name): file_sha256(evaluation / name)
              for name in ('goal_mae_summary.json', 'normal_depth/metrics.json', 'protocol.json')}
    data, names = [], None
    for frame in FRAMES:
        pred_path = evaluation / 'normal_depth' / f'prediction_{frame:04d}.npz'
        gt_path = args.gt_depth_dir.resolve() / f'gt_{frame:04d}.npz'
        hashes.update({str(pred_path): file_sha256(pred_path), str(gt_path): file_sha256(gt_path)})
        with np.load(pred_path) as z:
            pred = {key: z[key] for key in z.files}
        with np.load(gt_path) as z:
            gtdata = {key: z[key] for key in z.files}
        if names is None:
            names = [str(v) for v in gtdata['object_names']]
        elif names != [str(v) for v in gtdata['object_names']]:
            raise AssertionError('GT对象编号跨帧不一致。')
        gt_normal, valid = unit_normal(pred['gt_normal'])
        eligible = pred['gt_foreground'].astype(bool) & valid
        visible = np.isfinite(pred['alpha']) & (pred['alpha'] >= .5)
        rays = camera_rays(gtdata['camera_to_world'], gtdata['intrinsics'], *eligible.shape).astype(np.float64)
        gtxyz = rays[..., :3] + gtdata['depth_z'][..., None] * rays[..., 3:]
        data.append((frame, pred, gtdata, eligible, visible, gt_normal, rays, gtxyz))
    # 完整8个对象构成互斥全范围分区，unknown也保留；复合组只作明确的重叠辅助汇总。
    if len(names) != 8 or names[4] != 'Character_Mesh_Ch02_Sneakers' or names[6] != 'Reference_Checkered_Platform':
        raise AssertionError('预定义语义对象与当前GT不一致。')
    partitions = {f'object_{i}': {'ids': [i], 'name': n} for i, n in enumerate(names)}
    partitions['unknown'] = {'ids': [-1], 'name': '未分配的原可评分像素'}
    partitions.update(all={'ids': list(range(8)) + [-1], 'name': '全部原可评分像素'},
                      human={'ids': list(range(6)), 'name': '人体全部6对象'},
                      ground={'ids': [6, 7], 'name': '平台及底座'},
                      footwear={'ids': [4, 5], 'name': '鞋与袜'})
    plane_values = np.concatenate([xyz[..., 2][eligible & (gt['object_id'] == 6)]
                                   for _, _, gt, eligible, _, _, _, xyz in data])
    plane = float(np.median(plane_values))
    definition = dict(scope='GT仅离线诊断，绝不生成训练监督。原eligible/normal-valid先算，随后与对象分区交集；不重新腐蚀分区边界。',
                      frames=FRAMES, partitions=partitions, exclusive_partition=list(partitions)[:9],
                      coordinate='原评测+0.5像素中心camera_rays，Blender世界Z向上；距离为数据集世界单位（m）',
                      plane_definition='所有15帧object6原eligible世界Z的中位数', plane_z=plane,
                      plane_max_abs=float(np.max(np.abs(plane_values-plane))),
                      gt_elevated_threshold=.02, predicted_plane_band=.01,
                      same_ray_point_distance_thresholds_m=[.01, .02, .05, .10],
                      angular_thresholds_deg=[2., 5., 10., 30., 60.],
                      invalid_pixels='原主MAE仍conditional-valid；另报coverage及missing180。几何保留率invalid计入分母且判未保留。')
    write(output / 'partition_definition.json', definition)
    totals = {key: dict(eligible=0, alpha_valid=0, normal={field: dict(count=0, error_sum=0., thresholds=[0]*5)
                          for field in FIELDS}, depth={field: dict(distance=[], height_delta=[], height=[], denominator=0,
                          count=0, close=[0]*4, elevated=0, plane_proxy=0, below_plane=0, above_plane=0)
                          for field in ('expected', 'median', 'mesh_depth')}) for key in partitions}
    rows = []
    for frame, pred, gtdata, eligible, visible, gt_normal, rays, gtxyz in data:
        masks, angles = {}, {}
        arrays = dict(object_id=gtdata['object_id'], eligible=eligible, alpha_visible=visible,
                      gt_world_height=gtxyz[..., 2]-plane)
        for method, field in FIELDS.items():
            normal, valid = unit_normal(pred[field])
            support = pred['mesh_valid'] if method == 'mesh' else visible
            if method == 'depth_normal':
                support = binary_erosion(eligible & visible, structure=np.ones((3, 3)), border_value=0)
            masks[method] = eligible & support & valid
            angles[method] = np.degrees(np.arccos(np.clip((normal*gt_normal).sum(-1), -1, 1)))
            arrays[method+'_valid'] = masks[method]
            arrays[method+'_angular_deg'] = np.where(masks[method], angles[method], np.nan)
        xyz_fields = {}
        for field in ('expected', 'median', 'mesh_depth'):
            xyz = rays[..., :3] + pred[field][..., None] * rays[..., 3:]
            support = pred['mesh_valid'] if field == 'mesh_depth' else visible
            valid = eligible & support & np.isfinite(pred[field]) & (pred[field] > 0) & np.isfinite(gtdata['depth_z']) & (gtdata['depth_z'] > 0)
            distance = np.linalg.norm(xyz-gtxyz, axis=-1)
            height_delta = xyz[..., 2]-gtxyz[..., 2]
            xyz_fields[field] = valid, distance, height_delta, xyz[..., 2]-plane
            arrays[field+'_point_valid'] = valid
            arrays[field+'_point_distance_m'] = np.where(valid, distance, np.nan)
            arrays[field+'_height_delta_m'] = np.where(valid, height_delta, np.nan)
            arrays[field+'_world_height_m'] = np.where(valid, xyz[..., 2]-plane, np.nan)
            arrays[field+'_plane_proxy'] = valid & (gtxyz[..., 2] > plane+.02) & (np.abs(xyz[..., 2]-plane) < .01)
        object_id = gtdata['object_id']
        if np.any(eligible & ~np.isin(object_id, list(range(8))+[-1])):
            raise AssertionError('存在未覆盖对象。')
        for key, part in partitions.items():
            region = eligible & np.isin(object_id, part['ids'])
            n, av = int(region.sum()), int((region & visible).sum())
            total = totals[key]
            total['eligible'] += n
            total['alpha_valid'] += av
            for method in FIELDS:
                selection = region & masks[method]
                errors = angles[method][selection]
                t = total['normal'][method]
                t['count'] += int(errors.size)
                t['error_sum'] += float(errors.sum())
                t['thresholds'] = [old + int((errors > threshold).sum()) for old, threshold in zip(t['thresholds'], definition['angular_thresholds_deg'])]
                rows.append(dict(frame=frame, partition=key, name=part['name'], method=method,
                                 eligible_pixels=n, alpha_valid_pixels=av, valid_pixels=int(errors.size),
                                 coverage=errors.size/n if n else None, angular_mae_deg=float(errors.mean()) if errors.size else None,
                                 angular_error_sum_deg=float(errors.sum())))
            depth_eligible = region & np.isfinite(gtdata['depth_z']) & (gtdata['depth_z'] > 0)
            for field, (valid, distance, height_delta, height) in xyz_fields.items():
                selection = region & valid
                t = total['depth'][field]
                t['denominator'] += int(depth_eligible.sum())
                t['count'] += int(selection.sum())
                t['distance'].append(distance[selection])
                t['height_delta'].append(height_delta[selection])
                t['height'].append(height[selection])
                t['close'] = [old + int((selection & (distance <= threshold)).sum()) for old, threshold in zip(t['close'], definition['same_ray_point_distance_thresholds_m'])]
                t['elevated'] += int((depth_eligible & (gtxyz[..., 2] > plane+.02)).sum())
                t['plane_proxy'] += int((selection & arrays[field+'_plane_proxy']).sum())
                t['below_plane'] += int((selection & (height < -.01)).sum())
                t['above_plane'] += int((selection & (height > .01)).sum())
        np.savez_compressed(output / 'arrays' / f'diagnostic_{frame:04d}.npz', **arrays)
    result = {}
    for key, t in totals.items():
        n = t['eligible']
        region = dict(**partitions[key], eligible_pixels=n, eligible_pixel_share=n/totals['all']['eligible'],
                      alpha_valid_pixels=t['alpha_valid'], alpha_coverage=t['alpha_valid']/n if n else None,
                      normal={}, point_geometry={})
        for method, normal in t['normal'].items():
            c, error = normal['count'], normal['error_sum']
            region['normal'][method] = dict(valid_pixels=c, coverage=c/n if n else None,
                mae_deg=error/c if c else None, error_sum_deg=error,
                missing180_mae_deg=(error+180*(n-c))/n if n else None,
                valid_pixel_share=c/totals['all']['normal'][method]['count'],
                error_contribution_share=error/totals['all']['normal'][method]['error_sum'],
                exceed_threshold_counts=dict(zip(map(str, definition['angular_thresholds_deg']), normal['thresholds'])))
        for field, point in t['depth'].items():
            region['point_geometry'][field] = dict(valid_pixels=point['count'], eligible_pixels=point['denominator'],
                coverage=point['count']/point['denominator'] if point['denominator'] else None,
                distance_m=describe(np.concatenate(point['distance'])),
                height_delta_m=describe(np.concatenate(point['height_delta'])),
                predicted_height_m=describe(np.concatenate(point['height'])),
                same_ray_close_counts=dict(zip(map(str, definition['same_ray_point_distance_thresholds_m']), point['close'])),
                same_ray_close_fraction=dict(zip(map(str, definition['same_ray_point_distance_thresholds_m']),
                    [v/point['denominator'] if point['denominator'] else None for v in point['close']])),
                gt_elevated_pixels=point['elevated'], predicted_near_plane_elevated_gt_pixels=point['plane_proxy'],
                plane_replacement_proxy_fraction=point['plane_proxy']/point['elevated'] if point['elevated'] else None,
                below_plane_pixels=point['below_plane'], above_plane_pixels=point['above_plane'])
        result[key] = region
    for method in FIELDS:
        np.testing.assert_allclose(result['all']['normal'][method]['mae_deg'], primary['values'][method], atol=1e-10, rtol=0)
        np.testing.assert_allclose(result['all']['normal'][method]['coverage'], primary['details'][method]['coverage'], atol=1e-12, rtol=0)
    assert sum(result[key]['eligible_pixels'] for key in definition['exclusive_partition']) == result['all']['eligible_pixels']
    summary = dict(status='PASS_DIAGNOSTIC_EXECUTION', geometry_completeness='REQUIRES_SEPARATE_REVIEW',
                   evaluation=str(evaluation), model=primary['model'], source=primary['source'], iteration=primary['iteration'],
                   original_primary=primary, measurement_protocol_sha256=primary['protocol_sha256'],
                   partition_definition=definition, regions=result, input_sha256=hashes,
                   original_six_source_sha256=frozen, diagnostic_source_sha256=file_sha256(Path(__file__)),
                   input_unchanged=all(file_sha256(path) == sha for path, sha in hashes.items()),
                   note='条件法线valid不保证预测深度占据GT鞋表面。近平台高度只是替代代理，无预测语义标签，不能把全部深度错误称地面。')
    if not summary['input_unchanged']:
        raise AssertionError('只读诊断期间输入变化。')
    write(output / 'summary.json', summary)
    with (output / 'per_frame_regions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.figures:
        figures(output, data, result, plane)
    print(json.dumps({'status': summary['status'], 'output': str(output), 'shoe': result['object_4']}, ensure_ascii=False))


def figures(output, data, regions, plane):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    (output / 'figures').mkdir()
    names = ['Body', 'Cloth', 'Eyelashes', 'Hair', 'Sneakers', 'Socks', 'Platform', 'Base']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    x = np.arange(8)
    for i, method in enumerate(FIELDS):
        axes[0].bar(x+(i-1)*.24, [regions[f'object_{k}']['normal'][method]['mae_deg'] or 0 for k in range(8)], .24, label=method)
        axes[1].bar(x+(i-1)*.24, [100*regions[f'object_{k}']['normal'][method]['error_contribution_share'] for k in range(8)], .24, label=method)
    for ax in axes:
        ax.set_xticks(x, names, rotation=35, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Conditional angular MAE (deg)')
    axes[1].set_ylabel('Contribution to total angular error (%)')
    fig.savefig(output / 'figures/object_errors_contributions.png', dpi=150)
    plt.close(fig)
    cmap = ListedColormap(['#999999', '#e41a1c', '#ff7f00', '#ffff33', '#a65628', '#377eb8', '#984ea3', '#4daf4a', '#76b482'])
    for frame, pred, gt, eligible, visible, _, rays, gtxyz in data:
        with np.load(output / 'arrays' / f'diagnostic_{frame:04d}.npz') as z:
            a = {k: z[k] for k in z.files}
        fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout='constrained')
        maps = [(gt['object_id']+1, 'GT semantic objects', cmap, 0, 8),
                (pred['alpha'], 'Prediction alpha', 'gray', 0, 1),
                (a['gs_render_angular_deg'], 'GS angular error', 'magma', 0, 90),
                (a['expected_height_delta_m'], 'Expected height - GT', 'coolwarm', -.25, .25),
                (a['gt_world_height'], 'GT world height', 'viridis', 0, .3),
                (a['expected_world_height_m'], 'Expected world height', 'viridis', 0, .3),
                (a['mesh_depth_world_height_m'], 'Mesh world height', 'viridis', 0, .3),
                (a['expected_point_distance_m'], 'Expected same-ray distance', 'magma', 0, .3)]
        footwear = np.isin(gt['object_id'], [4, 5]) & eligible
        for ax, (arr, title, cm, lo, hi) in zip(axes.ravel(), maps):
            ax.imshow(arr, cmap=cm, vmin=lo, vmax=hi)
            if footwear.any():
                ax.contour(footwear, levels=[.5], colors='cyan', linewidths=.6)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'Frame {frame}: cyan contour = all GT footwear (offline only)')
        fig.savefig(output / 'figures' / f'full_frame_{frame:04d}.png', dpi=110)
        if footwear.any():
            yy, xx = np.where(footwear)
            for ax in axes.ravel():
                ax.set_xlim(max(0, xx.min()-12), min(399, xx.max()+12))
                ax.set_ylim(min(399, yy.max()+12), max(0, yy.min()-12))
            fig.suptitle(f'Frame {frame}: entire GT footwear bbox; display crop only, no measurement change')
            fig.savefig(output / 'figures' / f'footwear_zoom_{frame:04d}.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
