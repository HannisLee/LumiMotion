"""只读B07离线评测误差分解；绝不生成训练mask/标签或替代原评分。"""
from pathlib import Path
import json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT.parent/'1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算'
sources = {}
record = json.loads((BASE/'model_mlp/pbr_plane_projection.json').read_text())
plane = np.asarray(record['source_plane_fit']['plane_normal'], dtype=np.float64)
plane /= np.linalg.norm(plane)
names = ['GT方向距来源正plane轴0至5度', 'GT方向距来源正plane轴5至30度', 'GT方向距来源正plane轴大于30度']
totals = {name: dict(gt_pixels=0, valid_pixels=0, error_sum=0.) for name in names}
tail_names = ['0至7度', '7至15度', '15至30度', '30至60度', '60至180度']
tails = {name: dict(valid_pixels=0, error_sum=0.) for name in tail_names}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1)
    valid = np.isfinite(x).all(-1) & (n > 1e-6)
    result = np.zeros_like(x)
    np.divide(x, n[..., None], out=result, where=valid[..., None])
    return result, valid


for frame in range(10, 151, 10):
    path = BASE/'normal_depth'/('prediction_%04d.npz' % frame)
    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        gt, gv = unit(data['gt_normal'])
        pred, pv = unit(data['normal'])
        eligible = data['gt_foreground'].astype(bool) & gv
        valid = eligible & pv & np.isfinite(data['alpha']) & (data['alpha'] >= .5)
        angles = np.degrees(np.arccos(np.clip((gt*pred).sum(-1), -1, 1)))
        gt_angle = np.degrees(np.arccos(np.clip(gt@plane, -1, 1)))
        bins = [gt_angle <= 5, (gt_angle > 5) & (gt_angle <= 30), gt_angle > 30]
        for name, selection in zip(names, bins):
            mask = valid & selection
            totals[name]['gt_pixels'] += int((eligible & selection).sum())
            totals[name]['valid_pixels'] += int(mask.sum())
            totals[name]['error_sum'] += float(angles[mask].sum())
        edges = [(0, 7), (7, 15), (15, 30), (30, 60), (60, 181)]
        for name, (lo, hi) in zip(tail_names, edges):
            mask = valid & (angles >= lo) & (angles < hi)
            tails[name]['valid_pixels'] += int(mask.sum())
            tails[name]['error_sum'] += float(angles[mask].sum())

count = sum(row['valid_pixels'] for row in totals.values())
gt_count = sum(row['gt_pixels'] for row in totals.values())
mae = sum(row['error_sum'] for row in totals.values())/count
original = json.loads((BASE/'metrics_summary.json').read_text())['metrics'][2]
np.testing.assert_allclose(mae, original['mae'], atol=1e-9, rtol=1e-9)
np.testing.assert_allclose(100*count/gt_count, original['coverage_pct'], atol=1e-9, rtol=1e-9)
assert sum(row['valid_pixels'] for row in tails.values()) == count
np.testing.assert_allclose(sum(row['error_sum'] for row in tails.values())/count, mae, atol=1e-9, rtol=1e-9)
for rows in [totals, tails]:
    for row in rows.values():
        row['mae_deg'] = row['error_sum']/row['valid_pixels'] if row['valid_pixels'] else None
        row['fraction_of_original_valid_pixels'] = row['valid_pixels']/count
        row['contribution_to_original_mae_deg'] = row['error_sum']/count
        if 'gt_pixels' in row:
            row['coverage'] = row['valid_pixels']/row['gt_pixels'] if row['gt_pixels'] else None

result = dict(status='PASS_OFFLINE_DIAGNOSTIC_ONLY', run='B07', iteration=55000, frames=15,
              original_mae_deg=mae, original_gt_pixels=gt_count, original_valid_pixels=count,
              original_scoring_unchanged=True, source_sha=sources, gt_direction_bins=totals,
              error_tail_bins=tails, optimizer_steps=0, training_inputs_generated=False,
              limitations='仅B07完整评测浮点数组的误差分解，不是B08结果；GT方向分箱不是物体分割；不筛帧/点/难像素，不输出任何训练监督或掩码。')
out = ROOT/'inspection/b07_offline_error_decomposition.json'
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
