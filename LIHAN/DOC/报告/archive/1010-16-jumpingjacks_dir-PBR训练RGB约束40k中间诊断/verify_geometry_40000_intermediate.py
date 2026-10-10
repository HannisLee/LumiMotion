"""独立CPU复算40k全15中间GS/Depth/Z；不冒充55k七MAE。"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parent.parent
path = ROOT/'normal_depth_40000_intermediate'
record = json.loads((path/'metrics.json').read_text())
assert record['iteration'] == 40000 and len(record['frames']) == 15
assert [x['source_frame'] for x in record['frames']] == list(range(10, 151, 10))


def unit(array):
    array = np.asarray(array, dtype=np.float64)
    length = np.linalg.norm(array, axis=-1)
    valid = np.isfinite(array).all(-1) & (length > 1e-6)
    result = np.zeros_like(array)
    np.divide(array, length[..., None], out=result, where=valid[..., None])
    return result, valid


totals = {key: [0, 0, 0.] for key in ['gs_render', 'depth_normal', 'expected', 'median']}
missing = [0, 0.]
common = [0, 0., 0.]
for row in record['frames']:
    with np.load(path/('prediction_%04d.npz' % row['source_frame'])) as data:
        gt, valid_gt = unit(data['gt_normal'])
        eligible = data['gt_foreground'].astype(bool) & valid_gt
        visible = np.isfinite(data['alpha']) & (data['alpha'] >= .5)
        masks, errors = {}, {}
        for name, field in [('gs_render', 'normal'), ('depth_normal', 'depth_normal')]:
            pred, valid = unit(data[field])
            support = visible
            if name == 'depth_normal':
                support = binary_erosion(eligible & visible, structure=np.ones((3, 3)), border_value=0)
            mask = eligible & support & valid
            error = np.degrees(np.arccos(np.clip((pred*gt).sum(-1), -1, 1)))
            np.testing.assert_allclose(error[mask].mean(),
                                      row['regions']['foreground']['normal_metrics'][name]['angular_deg']['mean'],
                                      atol=1e-9, rtol=1e-9)
            totals[name][0] += int(eligible.sum())
            totals[name][1] += int(mask.sum())
            totals[name][2] += float(error[mask].sum())
            masks[name], errors[name] = mask, error
        missing[0] += int(eligible.sum())
        missing[1] += float(errors['gs_render'][masks['gs_render']].sum())+180*int((eligible & ~masks['gs_render']).sum())
        shared = masks['gs_render'] & masks['depth_normal']
        common[0] += int(shared.sum())
        for i, name in enumerate(['gs_render', 'depth_normal'], 1):
            common[i] += float(errors[name][shared].sum())
        depth_gt = data['gt_depth'].astype(np.float64)
        depth_eligible = data['gt_foreground'].astype(bool) & np.isfinite(depth_gt) & (depth_gt > 0)
        for name in ['expected', 'median']:
            pred = data[name].astype(np.float64)
            mask = depth_eligible & visible & np.isfinite(pred) & (pred > 0)
            totals[name][0] += int(depth_eligible.sum())
            totals[name][1] += int(mask.sum())
            totals[name][2] += float(np.abs(pred[mask]-depth_gt[mask]).sum())
fg = record['summary']['foreground']
results = {}
for name, (eligible, count, error) in totals.items():
    saved = fg['normal_metrics'][name] if name in ['gs_render', 'depth_normal'] else fg['depth'][name]
    key = 'mae_pixel_mean_deg' if name in ['gs_render', 'depth_normal'] else 'mae_pixel_mean'
    np.testing.assert_allclose([error/count, count/eligible], [saved[key], saved['coverage']],
                              atol=1e-7, rtol=1e-7)
    results[name] = dict(mae=error/count, coverage=count/eligible, valid_pixels=count, gt_pixels=eligible)
np.testing.assert_allclose(missing[1]/missing[0], fg['normal']['angular_with_missing_deg']['pixel_mean'],
                          atol=1e-9, rtol=1e-9)
for name, total in zip(['gs_render', 'depth_normal'], common[1:]):
    np.testing.assert_allclose(total/common[0], fg['common_normal_metrics']['angular_deg'][name]['mae_pixel_mean_deg'],
                              atol=1e-9, rtol=1e-9)
for p, digest in record['sources'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
out = dict(status='PASS_INDEPENDENT_INTERMEDIATE_ONLY', iteration=40000, frames=15,
           metrics=results, missing_180_normal_mae_deg=missing[1]/missing[0],
           common_GS_Depth_only_pixels=common[0],
           common_GS_Depth_only_mae_deg=[x/common[0] for x in common[1:]],
           source_hashes_verified=len(record['sources']), optimizer_steps=0,
           GT_labels_used_for_training=False, full_seven_MAE=False, mesh_normal='NOT_AVAILABLE',
           normal_7_goal='FAILED_INTERMEDIATE_NOT_FINAL')
p = ROOT/'inspection/geometry_40000_independent_audit.json'
assert not p.exists()
p.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
