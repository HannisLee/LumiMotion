"""40k canonical更新在source加性MLP旋转后的实际响应，只读预测，不用GT。"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent;sys.path.insert(0,str(CWD))
import torch,numpy as np
import torch.nn.functional as F
from plyfile import PlyData
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_geometry import quaternion_normal

@torch.no_grad()
def main():
 run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context;pc=ctx.pc
 v=PlyData.read(ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
 q0=torch.tensor(np.column_stack([v[f'rot_{i}'] for i in range(4)]),device='cuda');q=pc._rotation
 groups={'projected_plane':ctx.pbr_plane_projection.support,'unprojected_dynamic':(pc.get_binary_feature()[:,0]>=.5)&~ctx.pbr_plane_projection.support,'unprojected_static':(pc.get_binary_feature()[:,0]<.5)&~ctx.pbr_plane_projection.support}
 levels=q.new_tensor([0.,.01,.5,.9,.99,1.])
 def stats(x,ids):
  return dict(quantile_levels=levels.tolist(),values=torch.quantile(x[ids],levels).tolist(),mean=float(x[ids].mean()))
 dot=(F.normalize(q,dim=1)*F.normalize(q0,dim=1)).sum(1).abs().clamp(0,1)
 canonical_angle=2*torch.acos(dot)*180/torch.pi
 summaries={k:dict(points=int(ids.sum()),source_canonical_raw_q_norm=stats(q0.norm(dim=1),ids),canonical_rotation_angle_deg=stats(canonical_angle,ids),points_at_angle_bound=int((canonical_angle[ids]>=59.9).sum())) for k,ids in groups.items()}
 rows=[]
 for key in ['0001','0038','0075','0076','0149']:
  cam=ctx.camera_by_key[key]
  # sourceXYZ+sourcefeature+fixed sourceNN；避免把plane额外投影误算为canonical响应。
  d=ctx.deform.step(ctx.pbr_deformation_xyz,cam.fid.unsqueeze(0).expand(len(q),-1),iteration=40000,feature=pc.get_binary_feature(),camera_center=cam.camera_center)
  dr=d['d_rotation'];world0=q0+dr;world=q+dr
  assert torch.isfinite(world).all() and torch.isfinite(world0).all()
  normal0=quaternion_normal(world0);normal=quaternion_normal(world)
  signed=(normal0*normal).sum(1).clamp(-1,1);angle=torch.acos(signed.abs())*180/torch.pi
  rows.append(dict(frame=key,groups={k:dict(source_world_raw_q_norm=stats(world0.norm(dim=1),ids),current_world_raw_q_norm=stats(world.norm(dim=1),ids),MLP_delta_norm_over_source_q_norm=stats(dr.norm(dim=1)/q0.norm(dim=1).clamp_min(1e-12),ids),actual_world_unoriented_axis_change_deg=stats(angle,ids),angle_bound_points_world_axis_change_deg=stats(angle,ids&(canonical_angle>=59.9)) if (ids&(canonical_angle>=59.9)).any() else None,near_source_world_cancellation_points=int((world0[ids].norm(dim=1)<.05*q0[ids].norm(dim=1)).sum())) for k,ids in groups.items()}))
 ctx.verify_frozen();manifest=json.loads((ROOT/'code_manifest.json').read_text());assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
 report=dict(status='PASS_READ_ONLY_ADDITIVE_ROTATION_RESPONSE',iteration=40000,summary=summaries,rows=rows,optimizer_steps=0,GT_geometry_normal_material_inputs=False,current_training_modified=False,source_files=ctx.source_files,limits='原始world轴由qcanonical+dMLP归一化，比较source和40k实际canonical更新的轴响应；projected组随后还有硬plane投影，此处不把它当最终光栅normal。没有GT正确性/最终MAE，不能据此直接选择参数或覆盖训练。')
 p=ROOT/'inspection/rotation_response.json';assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
