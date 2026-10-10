"""保存40k gate相对同checkpoint source gate的世界几何响应；0优化、无GT标签。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_deformation import deformation_input
from scene.pbr_geometry import quaternion_normal

def stats(value):
 value=value.detach().double()
 return dict(mean=float(value.mean()),median=float(value.median()),p90=float(torch.quantile(value,.9)),
             p99=float(torch.quantile(value,.99)),maximum=float(value.max()))

@torch.no_grad()
def main():
 assert json.loads((ROOT/'inspection/checkpoint_40000_audit.json').read_text())['status']=='PASS_CHECKPOINT_ONLY'
 run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context;pc=ctx.pc
 source=ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
 v=PlyData.read(source)['vertex'].data
 raw=torch.tensor(v['fea_0'].copy(),device='cuda')
 source_gate=torch.sigmoid(2*raw)
 current_raw=pc.feature[:,0];current_gate=pc.get_binary_feature()[:,0]
 xyz=torch.tensor(np.column_stack([v[k] for k in ['x','y','z']]),device='cuda')
 rotation=torch.tensor(np.column_stack([v[f'rot_{k}'] for k in range(4)]),device='cuda')
 source_alpha=torch.sigmoid(torch.tensor(v['opacity'].copy(),device='cuda'))
 with np.load(ROOT/'model_mlp/pbr_plane_projection.npz') as a:
  arrays={k:torch.tensor(a[k].copy(),device='cuda') for k in a.files}
 plane=arrays['plane'];offset=arrays['offset']
 dynamic=raw>=0
 near=(dynamic&(source_alpha>.05)&((xyz@plane+offset).abs()<=.2)&
       (arrays['max_plane_distance']<=.2)&((quaternion_normal(rotation)@plane).abs()>=.5)&
       (arrays['min_axis_cosine']>=.5))
 groups=dict(source_projected=arrays['support'],source_dynamic_near_plane=near,
             source_other_dynamic=dynamic&~near,source_static_unprojected=~dynamic&~arrays['support'])
 assert all(mask.any() for mask in groups.values())
 assert torch.equal(sum(mask.int() for mask in groups.values()),torch.ones_like(raw,dtype=torch.int32))
 parameter={}
 for name,mask in dict(all=torch.ones_like(dynamic),**groups).items():
  parameter[name]=dict(points=int(mask.sum()),raw_logit_delta=stats((current_raw-raw).abs()[mask]),
      probability_delta=stats((current_gate-source_gate).abs()[mask]),source_gate_mean=float(source_gate[mask].mean()),
      saved_gate_mean=float(current_gate[mask].mean()),raw_threshold_crossings=int(((raw[mask]>=0)!=(current_raw[mask]>=0)).sum()),
      changed_raw_points=int((current_raw[mask]!=raw[mask]).sum()))
 rows=[]
 for frame in ['0001','0075','0149']:
  cam=ctx.camera_by_key[frame];ctx.load_camera(cam)
  current=ctx.deformation(cam)
  reference=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),
      cam.fid.unsqueeze(0).expand(len(raw),-1),iteration=40000,feature=source_gate[:,None],camera_center=cam.camera_center)
  reference=ctx.pbr_plane_projection.apply(pc,reference)
  distance=(current['d_xyz']-reference['d_xyz']).norm(dim=-1)
  actual_axis=quaternion_normal((pc._rotation+current['d_rotation']).double())
  reference_axis=quaternion_normal((pc._rotation+reference['d_rotation']).double())
  angles=torch.rad2deg(torch.acos((actual_axis*reference_axis).sum(-1).abs().clamp(0,1)))
  assert torch.isfinite(distance).all() and torch.isfinite(angles).all()
  rows.append(dict(frame=frame,groups={name:dict(position_change_m=stats(distance[mask]),
       unoriented_axis_change_deg=stats(angles[mask])) for name,mask in dict(all=torch.ones_like(dynamic),**groups).items()}))
  ctx.unload_camera(cam)
 ctx.verify_frozen()
 manifest=json.loads((ROOT/'code_manifest.json').read_text())
 assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==sha for k,sha in manifest.items())
 result=dict(status='PASS_READ_ONLY_GATE_WORLD_RESPONSE',iteration=40000,source_iteration=35000,
    parameter_groups=parameter,rows=rows,source_PLY_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
    frozen_checkpoint_SHA=ctx.source_files,source_NN_frozen=True,optimizer_steps=0,GT_labels_read=False,
    training_prior_generated=False,scoring_mask_generated=False,
    limitations='同一40k canonical几何比较saved gate与source35k gate的只读输出响应，不是B07另训权重，也不是GT法线误差。source near-plane/静动态组不是GT物体分类，全部点完整分区；未强制静态/删点/调整评分。仅TRAIN三帧，不当全15正式MAE。')
 p=ROOT/'inspection/gate_world_effect_40000.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
