"""完整135时间来源平面的法向运动诊断；只读，不生成训练mask或GT先验。"""
from pathlib import Path
import json,hashlib,sys
import numpy as np,torch
from argparse import Namespace
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root.parent.parent))
from scripts.stage2_runtime import FrozenScene
from scene.pbr_geometry import quaternion_normal
model=root/'model_mlp';args=Namespace(**json.loads((model/'stage2_config.json').read_text()))
args.pbr_plane_projection=False # 只读原始来源形变，避免先验投影污染运动诊断。
ctx=FrozenScene(args,model,35000)
asset=model/'pbr_plane_projection.npz';record=json.loads((model/'pbr_plane_projection.json').read_text())
assert hashlib.sha256(asset.read_bytes()).hexdigest()==record['asset_sha256']
with np.load(asset,allow_pickle=False) as data:
 plane=torch.tensor(data['plane'],device='cuda');offset=torch.tensor(data['offset'],device='cuda')
 expected_span=data['motion_span'].copy();expected_max=data['max_plane_distance'].copy();expected_cos=data['min_axis_cosine'].copy();support=data['support'].copy()
xyz=ctx.pc.get_xyz;feature=ctx.pc.get_binary_feature();alpha=ctx.pc.get_opacity[:,0]
base=(xyz@plane+offset).abs();axis=(quaternion_normal(ctx.pc._rotation)@plane).abs()
lo=hi=xyzlo=xyzhi=None;maxdist=base.new_zeros(len(base));mincos=base.new_ones(len(base))
from scipy.spatial import cKDTree
cardinal=plane.new_zeros(3);cardinal[int(plane.abs().argmin())]=1
u=torch.nn.functional.normalize(torch.cross(plane,cardinal,dim=0),dim=0);v=torch.cross(plane,u,dim=0)
xy_support=torch.stack((xyz[support]@u,xyz[support]@v),1).detach().cpu().numpy();tree=cKDTree(xy_support)
max_nearest=base.new_zeros(len(base));candidate=((alpha>.05)&(base<=.2)&(axis>=.5)&torch.tensor(expected_max<=.2,device='cuda')&torch.tensor(expected_cos>=.5,device='cuda'))
ids=torch.where(candidate)[0];candidate_nearest=base.new_zeros(len(ids))
with torch.no_grad():
 for camera in sorted(ctx.train_cameras,key=lambda c:c.image_name_train_light):
  d=ctx.deformation(camera);world=xyz+d['d_xyz'];distance=world@plane+offset
  lo=distance.clone() if lo is None else torch.minimum(lo,distance)
  hi=distance.clone() if hi is None else torch.maximum(hi,distance)
  xyzlo=d['d_xyz'].clone() if xyzlo is None else torch.minimum(xyzlo,d['d_xyz'])
  xyzhi=d['d_xyz'].clone() if xyzhi is None else torch.maximum(xyzhi,d['d_xyz'])
  maxdist=torch.maximum(maxdist,distance.abs());mincos=torch.minimum(mincos,(quaternion_normal(ctx.pc._rotation+d['d_rotation'])@plane).abs())
  xy=torch.stack((world[ids]@u,world[ids]@v),1).cpu().numpy();near=tree.query(xy,k=1)[0]
  candidate_nearest=torch.maximum(candidate_nearest,torch.as_tensor(near,dtype=base.dtype,device='cuda'))
ctx.verify_frozen()
span=(xyzhi-xyzlo).norm(dim=1).cpu().numpy();maxdist=maxdist.cpu().numpy();mincos=mincos.cpu().numpy()
np.testing.assert_array_equal(span,expected_span);np.testing.assert_array_equal(maxdist,expected_max);np.testing.assert_array_equal(mincos,expected_cos)
normalspan=(hi-lo).cpu().numpy();signedlo=lo.cpu().numpy();signedhi=hi.cpu().numpy();static=(feature[:,0]<.5).cpu().numpy();candidate=candidate.cpu().numpy()
nearest=np.full(len(base),np.nan,dtype=np.float32);nearest[ids.cpu().numpy()]=candidate_nearest.cpu().numpy()
def summarize(mask):
 out=dict(points=int(mask.sum()))
 for name,value in [('normal_motion_span',normalspan),('world_box_motion_span',span),('max_plane_distance',maxdist),('min_signed_plane_distance',signedlo),('max_signed_plane_distance',signedhi),('max_alltime_nearest_static_plane_distance',nearest)]:
  a=value[mask];a=a[np.isfinite(a)];out[name]={str(p):float(np.quantile(a,p)) for p in [0,.5,.9,.99,1]} if len(a) else None
 return out
report=dict(status='PASS_SOURCE_DIAGNOSTIC_ONLY',optimizer_steps=0,frames=135,GT_labels_read=False,training_mask_saved=False,training_configuration_changed=False,plane_source='source_prediction_RANSAC_SVD',exact_recomputed_existing_fulltime_arrays=True,groups={k:summarize(mask) for k,mask in {'support':support,'static_geometric_candidates_unprojected':candidate&static&~support,'dynamic_geometric_candidates':candidate&~static}.items()},threshold_diagnostics={})
for height in [.01,.03,.05,.08,.1,.2]:
 for span_threshold in [.005,.01,.03]:
  for nearest_threshold in [.03,.05,.1]:
   mask=candidate&~static&(maxdist<=height)&(normalspan<=span_threshold)&(nearest<=nearest_threshold)
   report['threshold_diagnostics'][f'height{height}_normalspan{span_threshold}_nearest{nearest_threshold}']=summarize(mask)
(root/'inspection/source_plane_normal_motion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
