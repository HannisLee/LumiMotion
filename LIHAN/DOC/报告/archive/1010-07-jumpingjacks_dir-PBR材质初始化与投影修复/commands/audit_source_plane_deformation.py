"""只读Stage1预测：核对canonical平面与原生实际形变表面的差异。"""
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

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

record = json.loads((ROOT/'model_mlp/pbr_geometry_source.json').read_text())
for p, digest in record['source_files'].items():
    assert sha(p) == digest
source = Path(record['source_model'])
v = PlyData.read(source/'point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
xyz = torch.tensor(np.column_stack([v[f] for f in ['x','y','z']]), device='cuda')
q = torch.tensor(np.column_stack([v['rot_'+str(i)] for i in range(4)]), device='cuda')
feature = torch.sigmoid(torch.tensor(v['fea_0'].copy(), device='cuda')[:,None]*2)
opacity = torch.sigmoid(torch.tensor(v['opacity'].copy(), device='cuda'))
plane = xyz.new_tensor(record['plane_normal'])
offset = record['plane_offset']
canonical_n = quaternion_normal(q)
support = ((feature[:,0] < .5) & (opacity > .05) &
           ((xyz@plane+offset).abs() <= record['band']) &
           ((canonical_n@plane).abs() >= .5))
assert int(support.sum()) == record['support_points']
model = DeformNetwork(is_blender=True, pred_color=True).cuda()
model.load_state_dict(torch.load(source/'deform/iteration_35000/deform.pth', map_location='cuda'))
model.eval()

def quantiles(value):
    return dict(zip(['min','p50','p90','p99','max'],
                    [float(x) for x in torch.quantile(value.float(), value.new_tensor([0,.5,.9,.99,1]))]))

rows=[]
with torch.no_grad():
    for frame in [1,75,149]:
        t = xyz.new_full((len(xyz),1), frame/150.)
        d = model(xyz, t, feature=feature, iteration=35000)
        effective_xyz = xyz+d['d_xyz']
        effective_n = quaternion_normal(q+d['d_rotation'])
        angular_shift = torch.rad2deg(torch.acos((effective_n*canonical_n).sum(1).abs().clamp(0,1)))
        rows.append(dict(frame=frame, source_time=frame/150.,
                         canonical_plane_distance=quantiles((xyz[support]@plane+offset).abs()),
                         effective_plane_distance=quantiles((effective_xyz[support]@plane+offset).abs()),
                         source_deformation_distance=quantiles(d['d_xyz'][support].norm(dim=1)),
                         source_deformation_axis_shift_deg=quantiles(angular_shift[support]),
                         effective_plane_axis_error_deg=quantiles(torch.rad2deg(torch.acos((effective_n[support]@plane).abs().clamp(0,1))))))
for p, digest in record['source_files'].items():
    assert sha(p) == digest
out=dict(status='PASS_READ_ONLY_DIAGNOSTIC', method='fixed_Stage1_prediction_canonical_vs_effective_plane',
         source_files=record['source_files'], source_iteration=35000, plane=record,
         support_feature_distribution=quantiles(feature[support,0]), rows=rows,
         gt_inputs=False, training_or_optimizer_steps=0, current_training_modified=False,
         final_mae=False)
(ROOT/'inspection/source_plane_deformation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
