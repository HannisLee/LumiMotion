"""只读来源35k及其full135time统计，诊断未支持点；不产生训练先验或改评分。"""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from plyfile import PlyData
import sys
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root.parent.parent))
from scene.pbr_geometry import quaternion_normal
model=root/'model_mlp';record=json.loads((model/'pbr_plane_projection.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(model/'pbr_plane_projection.npz')==record['asset_sha256']
assert all(sha(model/p)==h for p,h in record['source_snapshots'].items())
v=PlyData.read(model/'point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
xyz=np.column_stack([v[k] for k in ['x','y','z']]);q=np.column_stack([v[f'rot_{i}'] for i in range(4)])
feature=torch.sigmoid(torch.tensor(v['fea_0'].copy())*2).numpy();alpha=torch.sigmoid(torch.tensor(v['opacity'].copy())).numpy()
scale=np.exp(np.column_stack([v[f'scale_{i}'] for i in range(2)]));proxy=alpha*scale.prod(1)
normal=quaternion_normal(torch.from_numpy(q)).numpy()
with np.load(model/'pbr_plane_projection.npz',allow_pickle=False) as data:
 d={k:data[k].copy() for k in data.files}
distance=np.abs(xyz@d['plane']+d['offset']);cosine=np.abs(normal@d['plane'])
static=feature<.5;present=alpha>.05;near=distance<=.2;alltime_near=d['max_plane_distance']<=.2
axis=(cosine>=.5)&(d['min_axis_cosine']>=.5);stable=d['motion_span']<=.01
candidate=present&near&alltime_near&axis
support=d['support'];assert int(support.sum())==record['support_points']
groups={'projected_support':support,'all_unprojected':~support,
 'static_geometric_candidates_stable':candidate&static&stable,
 'static_geometric_candidates_unstable':candidate&static&~stable,
 'dynamic_geometric_candidates':candidate&~static,
 'static_near_alltime_but_bad_axis':present&static&near&alltime_near&~axis,
 'static_canonical_near_but_time_far':present&static&near&~alltime_near,
 'static_canonical_near_low_alpha':~present&static&near,
 'legacy_canonical_plane_support':static&present&(distance<=.08)&(cosine>=.5),
 'legacy_support_temporally_unstable':static&present&(distance<=.08)&(cosine>=.5)&~stable}
def summary(mask):
 ids=np.flatnonzero(mask);out=dict(points=len(ids),alpha_sum=float(alpha[mask].sum()),unprojected_alpha_scale_area_proxy=float(proxy[mask].sum()))
 for name,value in [('canonical_plane_distance',distance),('motion_span',d['motion_span']),('max_alltime_plane_distance',d['max_plane_distance']),('min_alltime_axis_cosine',d['min_axis_cosine']),('binary_feature',feature)]:
  out[name]={str(p):float(np.quantile(value[mask],p)) for p in [0,.5,.9,.99,1]} if len(ids) else None
 return out
report=dict(status='PASS_DIAGNOSTIC_ONLY',optimizer_steps=0,gt_files_read=False,new_training_prior_saved=False,scoring_changed=False,source_points=len(v),source_snapshots=record['source_snapshots'],training_cameras=record['training_cameras'],asset_sha256=record['asset_sha256'],groups={k:summary(mask) for k,mask in groups.items()},area_proxy_warning='未投影Alpha×scale0×scale1代理不是实际像素贡献；不能据此认定错误点属于平台或应删除/投影。')
assert np.array_equal(groups['static_geometric_candidates_stable'],support)
assert all(sha(model/p)==h for p,h in record['source_snapshots'].items())
(root/'inspection/unprojected_plane_groups.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
