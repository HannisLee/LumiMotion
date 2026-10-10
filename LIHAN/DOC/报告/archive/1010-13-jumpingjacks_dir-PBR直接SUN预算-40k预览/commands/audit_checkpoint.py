"""实际完整保存的中间checkpoint核验；不把检查点检查当最终MAE。"""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
from plyfile import PlyData
from scene.pbr_deformation import load_deformation_anchor
model=ROOT/'model_mlp';config=json.loads((model/'stage2_config.json').read_text());step=int(sys.argv[1])
assert step in [40000,50000,55000]
v=PlyData.read(model/f'point_cloud/iteration_{step}/point_cloud.ply')['vertex'].data
initial=PlyData.read(model/'point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
assert len(v)==len(initial)==99827 and all(np.isfinite(v[f]).all() for f in v.dtype.names)
frozen=[f for f in v.dtype.names if f in ['opacity','fea_0'] or f.startswith('scale_')]
assert all(np.array_equal(v[f],initial[f]) for f in frozen)
xyz=np.column_stack([v[f] for f in ['x','y','z']]);xyz0=np.column_stack([initial[f] for f in ['x','y','z']])
delta=np.linalg.norm(xyz-xyz0,axis=1);assert delta.max()<=config['pbr_geometry_max_displacement']+1e-5
q=np.column_stack([v[f'rot_{i}'] for i in range(4)]);q0=np.column_stack([initial[f'rot_{i}'] for i in range(4)])
q/=np.linalg.norm(q,axis=1,keepdims=True);q0/=np.linalg.norm(q0,axis=1,keepdims=True)
angle=2*np.degrees(np.arccos(np.clip(np.abs((q*q0).sum(1)),0,1)))
assert angle.max()<=config['pbr_geometry_max_angle']+.01
rgb=np.column_stack([v[f'albedo_dc_{i}'] for i in range(3)])*.282095+.5
assert rgb.min()>=.031-2e-6 and rgb.max()<=.969+2e-6
anchor=load_deformation_anchor(model,torch.as_tensor(xyz),step);assert torch.equal(anchor,torch.as_tensor(xyz0))
a=torch.load(model/'deform/iteration_35000/deform.pth',map_location='cpu')
b=torch.load(model/f'deform/iteration_{step}/deform.pth',map_location='cpu')
assert set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
source=json.loads((model/'stage1_source.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source.items())
lighting=json.loads((model/f'pbr_directional/iteration_{step}/lighting.json').read_text())
assert lighting['iteration']==step and lighting['lights_sha256']==hashlib.sha256((model/'pbr_gt_lights.json').read_bytes()).hexdigest()
assert lighting['deformation_anchor_sha256']==hashlib.sha256((model/'pbr_deformation_anchor.npz').read_bytes()).hexdigest()
# 仓库原生CUDA来源与checkpoint绑定，拒绝pip同名包或其它源码。
assert config['pbr_rasterizer_backend']=='lumimotion'
assert config['wo_indirect'] and config['wo_indirect_relight']
assert config['pbr_directional_actual_ray_budget']
from argparse import Namespace
from scene.pbr_rasterizer import validate_backend_model
native=validate_backend_model(model,Namespace(**config))
for iteration in [step]:
 assert json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())['rasterizer_source_sha256']==native['source_sha256']
# 平面支持/marker绑定；只审计实际几何，不改变评分。
plane_record=json.loads((model/'pbr_plane_projection.json').read_text())
assert config['pbr_plane_projection'] and config['pbr_deformation_anchor']
plane_sha=hashlib.sha256((model/'pbr_plane_projection.npz').read_bytes()).hexdigest()
assert plane_record['asset_sha256']==plane_sha
with np.load(model/'pbr_plane_projection.npz',allow_pickle=False) as asset:
 assert asset['support'].dtype==np.bool_ and len(asset['support'])==99827
 assert int(asset['support'].sum())==plane_record['support_points']
 assert (asset['motion_span'][asset['support']]<=config['pbr_plane_motion_span']).all()
 assert (asset['max_plane_distance'][asset['support']]<=config['pbr_plane_support_band']).all()
 assert (asset['min_axis_cosine'][asset['support']]>=.5).all()
for iteration in [step]:
 assert json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())['plane_projection_sha256']==plane_sha
record=dict(status='PASS_CHECKPOINT_ONLY',iteration=step,gaussians=len(v),all_finite=True,
            opacity_scale_feature_frozen=True,albedo_rgb_range=[float(rgb.min()),float(rgb.max())],
            clamped_albedo_channel_fraction=float(((rgb<=.03)|(rgb>=.97)).mean()),
            channels_at_projected_bound_fraction=float(((rgb<=.031001)|(rgb>=.968999)).mean()),
            max_displacement=float(delta.max()),max_rotation_deg=float(angle.max()),
            actual_light_scale=lighting['effective_intensity_scale'],deformation_state_tensors_unchanged=True,
            deformation_tensor_count=len(a),deformation_input_exactly_source_xyz=True,
            anchor_marker_bound=True,source_files_unchanged=True,final_mae=False)
path=ROOT/f'inspection/checkpoint_{step}_audit.json'
assert not path.exists(),'检查结果禁止覆盖'
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False))
