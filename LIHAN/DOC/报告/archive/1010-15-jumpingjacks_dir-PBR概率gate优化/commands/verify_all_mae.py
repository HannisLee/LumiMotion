"""从保存浮点图独立CPU复算，不调用评测指标实现；核验三法线共同像素和来源。"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(array):
    array = np.asarray(array, dtype=np.float64)
    length = np.linalg.norm(array, axis=-1)
    valid = np.isfinite(array).all(-1) & (length > 1e-6)
    result = np.zeros_like(array)
    np.divide(array, length[..., None], out=result, where=valid[..., None])
    return result, valid


def main():
    geometry = json.loads((ROOT / 'normal_depth/metrics.json').read_text())
    albedo = json.loads((ROOT / 'albedo/iteration_55000/metrics.json').read_text())
    meshes = json.loads((ROOT / 'predicted_meshes/manifest.json').read_text())
    assert meshes['gt_geometry_used'] is False and meshes['static_scene'] is False
    assert meshes['test_camera_used_for_reconstruction'] is False
    assert geometry['iteration'] == albedo['iteration'] == meshes['iteration'] == 55000
    assert len(geometry['frames']) == len(albedo['frames']) == 15
    assert [f['source_frame'] for f in geometry['frames']] == list(range(10, 151, 10))
    train = json.loads((Path(meshes['source_path']) / 'transforms_train.json').read_text())['frames']
    train_ids = {f['source_frame'] for f in train}
    expected_ids = set(range(10, 151, 10))
    assert train_ids.isdisjoint(expected_ids) and len(train_ids) == 135
    for number, entry in meshes['frames'].items():
        assert int(number) in expected_ids and np.isclose(entry['time'], int(number) / 150)
        assert len(entry['integrations']) == 135
        assert {r['source_frame'] for r in entry['integrations']} == train_ids
        assert all(r['split'] == 'train' and np.isclose(r['render_time'], entry['time']) for r in entry['integrations'])
        assert sha(ROOT / 'predicted_meshes' / entry['mesh']) == entry['mesh_sha256']
    totals = {name: [0, 0, 0.] for name in ('gs_render', 'depth_normal', 'mesh')}
    depth_totals = {name: [0, 0, 0.] for name in ('expected', 'median')}
    common_totals = {name: 0. for name in totals}
    common_count = 0
    for row in geometry['frames']:
        n = row['source_frame']
        with np.load(ROOT / 'normal_depth' / ('prediction_%04d.npz' % n)) as data:
            gt, gt_valid = unit(data['gt_normal'])
            eligible = data['gt_foreground'].astype(bool) & gt_valid
            visible = np.isfinite(data['alpha']) & (data['alpha'] >= .5)
            masks, errors = {}, {}
            for name, field in (('gs_render', 'normal'), ('depth_normal', 'depth_normal'), ('mesh', 'mesh_normal')):
                pred, valid = unit(data[field])
                support = data['mesh_valid'] if name == 'mesh' else visible
                if name == 'depth_normal':
                    support = binary_erosion(eligible & visible, structure=np.ones((3, 3)), border_value=0)
                mask = eligible & support & valid
                error = np.degrees(np.arccos(np.clip((pred * gt).sum(-1), -1, 1)))
                count = int(mask.sum())
                np.testing.assert_allclose(error[mask].mean(), row['regions']['foreground']['normal_metrics'][name]['angular_deg']['mean'], atol=1e-9, rtol=1e-9)
                totals[name][0] += int(eligible.sum())
                totals[name][1] += count
                totals[name][2] += float(error[mask].sum())
                masks[name], errors[name] = mask, error
            common = masks['gs_render'] & masks['depth_normal'] & masks['mesh']
            common_count += int(common.sum())
            for name in totals:
                common_totals[name] += float(errors[name][common].sum())
            gt_depth = data['gt_depth']
            eligible_depth = data['gt_foreground'].astype(bool) & np.isfinite(gt_depth) & (gt_depth > 0)
            for name in depth_totals:
                pred = data[name]
                mask = eligible_depth & visible & np.isfinite(pred) & (pred > 0)
                depth_totals[name][0] += int(eligible_depth.sum())
                depth_totals[name][1] += int(mask.sum())
                depth_totals[name][2] += float(np.abs(pred[mask] - gt_depth[mask]).sum())
    normal_results = {}
    for name, (eligible, valid, error) in totals.items():
        original = geometry['summary']['foreground']['normal_metrics'][name]
        np.testing.assert_allclose([error / valid, valid / eligible],
                                  [original['mae_pixel_mean_deg'], original['coverage']], atol=1e-9, rtol=1e-9)
        common_mae = common_totals[name] / common_count
        np.testing.assert_allclose(common_mae, geometry['summary']['foreground']['common_normal_metrics']['angular_deg'][name]['mae_pixel_mean_deg'], atol=1e-9, rtol=1e-9)
        normal_results[name] = dict(mae_deg=error / valid, coverage=valid / eligible, common_mae_deg=common_mae)
    depth_results = {}
    for name, (eligible, valid, error) in depth_totals.items():
        original = geometry['summary']['foreground']['depth'][name]
        np.testing.assert_allclose([error / valid, valid / eligible],
                                  [original['mae_pixel_mean'], original['coverage']], atol=1e-7, rtol=1e-7)
        depth_results[name] = dict(mae=error / valid, coverage=valid / eligible)
    n_total, covered_total, error_total = 0, 0, 0.
    for row in albedo['frames']:
        with np.load(ROOT / 'albedo/iteration_55000' / (row['frame'] + '.npz')) as data:
            core = data['gt_alpha'] >= .99
            core = binary_erosion(core, structure=np.ones((5, 5)), border_value=0)
            np.testing.assert_array_equal(core, data['gt_core'])
            pred, gt = data['albedo_linear'].astype(np.float64), data['gt_albedo_linear'].astype(np.float64)
            error_total += float(np.abs(pred[:, core] - gt[:, core]).mean(0).sum())
            n_total += int(core.sum())
            covered_total += int((core & (data['alpha'] >= .9)).sum())
    albedo_result = albedo['summaries']['raw']['foreground']
    np.testing.assert_allclose(error_total / n_total, albedo_result['mae_linear'], atol=2e-7, rtol=1e-7)
    np.testing.assert_allclose(100 * covered_total / n_total, albedo_result['coverage_pct'], atol=1e-9, rtol=1e-9)
    source_hashes = dict(geometry['sources'])
    source_hashes.update(albedo['checkpoint_hashes'])
    source_hashes.update(albedo['gt_file_hashes'])
    for path, digest in source_hashes.items():
        assert sha(path) == digest, '来源改变：' + path
    assert not (ROOT / 'FAILED').exists()
    result = dict(status='PASS', iteration=55000, frames=15,
                  independently_recomputed_normal=normal_results,
                  independently_recomputed_depth=depth_results,
                  independently_recomputed_albedo=dict(mae_linear=error_total / n_total, coverage_pct=100 * covered_total / n_total),
                  source_hashes_verified=len(source_hashes), dynamic_meshes=15, train_poses_per_time=135,
                  common_normal_pixels=common_count,
                  roughness_mae=json.loads((ROOT/'roughness/metrics.json').read_text())['summaries']['foreground']['mae'])
    (ROOT / 'inspection/independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
