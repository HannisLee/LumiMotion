"""只读预测来源：平面支持遗漏和全135训练时刻形变稳定性，0优化步。"""
from pathlib import Path
import sys, json, hashlib
ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parent.parent
sys.path.insert(0, str(CWD))
import torch
import numpy as np
from plyfile import PlyData
from utils.time_utils import DeformNetwork
from scene.pbr_geometry import quaternion_normal

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

record = json.loads((ROOT/'model_mlp/pbr_geometry_source.json').read_text())
assert all(sha(p) == h for p, h in record['source_files'].items())
config = json.loads((ROOT/'model_mlp/stage2_config.json').read_text())
source = Path(record['source_model'])
v = PlyData.read(source/'point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
xyz = torch.tensor(np.column_stack([v[f] for f in ['x','y','z']]), device='cuda')
q = torch.tensor(np.column_stack([v[f'rot_{i}'] for i in range(4)]), device='cuda')
feature = torch.sigmoid(torch.tensor(v['fea_0'].copy(), device='cuda')[:, None]*2)
opacity = torch.sigmoid(torch.tensor(v['opacity'].copy(), device='cuda'))
plane = xyz.new_tensor(record['plane_normal']); offset = record['plane_offset']
distance = (xyz@plane+offset).abs()
normal_cos = (quaternion_normal(q)@plane).abs()
static = feature[:, 0] < .5
support = static & (opacity > .05) & (distance <= record['band']) & (normal_cos >= .5)
assert int(support.sum()) == record['support_points']
# 仅用于诊断的未投影面积代理；不声称等同实际光栅覆盖/贡献。
area_mass = opacity*torch.exp(torch.tensor(v['scale_0'].copy()+v['scale_1'].copy(), device='cuda'))
model = DeformNetwork(is_blender=config['is_blender'], pred_color=config['pred_color']).cuda()
model.load_state_dict(torch.load(source/'deform/iteration_35000/deform.pth', map_location='cuda')); model.eval()
transforms = json.loads((Path(config['source_path'])/'transforms_train.json').read_text())
times = [float(f['time']) for f in transforms['frames']]
assert len(times) == 135
keys = [Path(f['file_path']).stem for f in transforms['frames']]
assert len(set(keys)) == 135 and all(int(k[-4:])%10 != 0 for k in keys)
lo = hi = None; max_plane_distance = torch.zeros(len(xyz), device='cuda')
with torch.no_grad():
    for i, time in enumerate(times):
        d = model(xyz, xyz.new_full((len(xyz),1), time), feature=feature, iteration=35000)
        value = d['d_xyz']
        lo = value.clone() if lo is None else torch.minimum(lo, value)
        hi = value.clone() if hi is None else torch.maximum(hi, value)
        max_plane_distance = torch.maximum(max_plane_distance, ((xyz+value)@plane+offset).abs())
        if (i+1)%30 == 0: print('source_training_times', i+1, flush=True)
span = (hi-lo).norm(dim=1)

def summary(mask):
    if not mask.any(): return dict(points=0)
    return dict(points=int(mask.sum()), opacity_sum=float(opacity[mask].sum()),
                unprojected_area_opacity_proxy=float(area_mass[mask].sum()),
                motion_axis_box_span_norm_p50_p90_p99_max=[float(x) for x in torch.quantile(span[mask], xyz.new_tensor([.5,.9,.99,1]))],
                all_train_max_plane_distance_p50_p90_p99_max=[float(x) for x in torch.quantile(max_plane_distance[mask], xyz.new_tensor([.5,.9,.99,1]))])

rows = {}
for band in [.08,.12,.2]:
    near = distance <= band
    for group, select in [('static',static),('dynamic',~static)]:
        rows[f'{group}_canonical_band_{band}'] = summary(near & select)
        rows[f'{group}_canonical_band_{band}_alpha_gt_005'] = summary(near & select & (opacity > .05))
        rows[f'{group}_canonical_band_{band}_alpha_le_005'] = summary(near & select & (opacity <= .05))
        rows[f'{group}_canonical_band_{band}_all_train_span_le_003'] = summary(near & select & (span <= .03))
        rows[f'{group}_canonical_band_{band}_all_train_plane_le_band'] = summary(near & select & (max_plane_distance <= band))
rows['current_support'] = summary(support)
rows['static_near_02_excluded_current_support'] = summary(static & (distance <= .2) & ~support)
rows['static_near_008_excluded_normal_cosine'] = summary(static & (distance <= .08) & (opacity > .05) & (normal_cos < .5))
assert all(sha(p) == h for p,h in record['source_files'].items())
result = dict(status='PASS_READ_ONLY_DIAGNOSTIC', source_iteration=35000, source_files=record['source_files'],
              points=len(xyz), source_plane=record, training_time_count=len(times), rows=rows,
              gt_inputs=False, optimizer_steps=0, current_training_modified=False, final_mae=False,
              caveat='opacity×scale0×scale1仅未投影面积代理，不等同光栅覆盖/遮挡贡献；box-span是坐标范围范数，不是精确最大两时刻距离。没有生成训练先验。')
(ROOT/'inspection/source_plane_coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
