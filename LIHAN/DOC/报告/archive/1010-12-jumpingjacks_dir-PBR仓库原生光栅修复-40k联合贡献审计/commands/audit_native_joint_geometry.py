"""40k原生光栅联合贡献与source支持符号核验，零训练/不读GT标签。"""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch,numpy as np
import torch.nn.functional as F
from plyfile import PlyData
from scripts.render_stage2 import Stage2RenderRun,save_image
from scene.pbr_rasterizer import rasterizer_classes
from scene.pbr_geometry import quaternion_normal
from gaussian_renderer.render_ir import render_ir

@torch.no_grad()
def main():
 run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context;pc=ctx.pc
 settings_type,rasterizer_type=rasterizer_classes(run.args)
 print('RASTERIZER',settings_type.__module__,flush=True)
 plane=ctx.pbr_plane_projection.plane;offset=ctx.pbr_plane_projection.offset
 source=PlyData.read(ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
 xyz0=torch.tensor(np.column_stack([source[k] for k in ['x','y','z']]),device='cuda')
 q0=torch.tensor(np.column_stack([source[f'rot_{i}'] for i in range(4)]),device='cuda')
 feature=torch.sigmoid(torch.tensor(source['fea_0'].copy(),device='cuda')*2)
 opacity0=torch.sigmoid(torch.tensor(source['opacity'].copy(),device='cuda'))
 with np.load(ROOT/'model_mlp/pbr_plane_projection.npz') as a:
  maxdistance=torch.tensor(a['max_plane_distance'].copy(),device='cuda');mincos=torch.tensor(a['min_axis_cosine'].copy(),device='cuda')
 candidate=(feature>=.5)&(opacity0>.05)&((xyz0@plane+offset).abs()<=.2)&(maxdistance<=.2)&((quaternion_normal(q0)@plane).abs()>=.5)&(mincos>=.5)
 masks=[ctx.pbr_plane_projection.support,candidate,(feature>=.5)&~candidate,(feature<.5)&~ctx.pbr_plane_projection.support]
 assert torch.stack(masks).sum(0).eq(1).all()
 colors=torch.stack(masks[:3],dim=1).float();staticcolors=masks[3][:,None].float().expand(-1,3)
 dest=ROOT/'inspection/native_joint_geometry_40000';assert not dest.exists();dest.mkdir()
 rows=[]
 for frame in ['0001','0038','0075','0076','0149','0150']:
  cam=ctx.camera_by_key[frame];ctx.load_camera(cam);d=ctx.deformation(cam)
  xyz=pc.get_xyz+d['d_xyz'];q=pc.get_rotation_bias(d['d_rotation']);scale=pc.get_scaling+d['d_scaling']
  opacity=pc.get_opacity if d['d_opacity'] is None else pc.get_opacity+d['d_opacity']
  settings=settings_type(image_height=cam.image_height,image_width=cam.image_width,tanfovx=math.tan(cam.FoVx*.5),tanfovy=math.tan(cam.FoVy*.5),bg=ctx.background,scale_modifier=1.,viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,sh_degree=pc.active_sh_degree,campos=cam.camera_center,prefiltered=False,debug=False)
  fn=rasterizer_type(raster_settings=settings)
  common=dict(means3D=xyz,means2D=torch.zeros_like(xyz),shs=None,scales=scale,rotations=q,cov3D_precomp=None)
  weights,_,allmap=fn(colors_precomp=colors,opacities=opacity,**common)
  static,_,staticmap=fn(colors_precomp=staticcolors,opacities=opacity,**common)
  reference=render_ir(cam,pc,run.args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},opt=run.args,material_only=True)
  world=(allmap[2:5].permute(1,2,0)@cam.world_view_transform[:3,:3].T).permute(2,0,1)
  errors=dict(alpha=float((allmap[1:2]-reference['rend_alpha']).abs().max()),worldnormal=float((world-reference['rend_normal']).abs().max()),expectedZ=float((allmap[0:1]/allmap[1:2].clamp_min(1e-6)-reference['surf_depth']).abs().max()),group_sum_alpha=float((weights.sum(0)+static[0]-allmap[1]).abs().max()),second_pass_allmap=float((staticmap-allmap).abs().max()))
  assert max(errors.values())<2e-5
  _,_,projected=fn(colors_precomp=torch.ones_like(xyz),opacities=opacity*masks[0][:,None],**common)
  pnormal=(projected[2:5].permute(1,2,0)@cam.world_view_transform[:3,:3].T).permute(2,0,1)
  valid=(projected[1]>.5)&(pnormal.norm(dim=0)>1e-6)
  signed=F.normalize(pnormal,dim=0)[:,valid].T@plane;side=float(cam.camera_center@plane+offset)
  assert side>0 and signed.min()>.99999
  alpha=allmap[1];valid_joint=alpha>.5;contribution=torch.cat([weights,static[:1]],0)
  dominant=contribution.argmax(0)
  groups={name:dict(points=int(mask.sum()),joint_accumulated_weight_sum=float(contribution[i].sum()),dominant_alpha05_pixels=int(((dominant==i)&valid_joint).sum())) for i,(name,mask) in enumerate(zip(['projected_plane','source_dynamic_plane_candidates','other_dynamic','static_unprojected'],masks))}
  row=dict(frame=frame,camera_above_source_plane=side,buffer_max_errors=errors,projected_valid_pixels=int(valid.sum()),projected_signed_plane_dot_min=float(signed.min()),joint_groups=groups)
  rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  # 合成标注图：projected蓝、动态plane候选红、其它动态绿、静态非投影白。
  palette=torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.],[1.,1.,1.]],device='cuda')
  display=torch.einsum('ki,kxy->ixy',palette,contribution)
  save_image(dest/f'{frame}_joint_group_weight.png',display)
  save_image(dest/f'{frame}_normal.png',(F.normalize(world,dim=0)*.5+.5)*alpha)
  save_image(dest/f'{frame}_projected_normal.png',(F.normalize(pnormal,dim=0)*.5+.5)*projected[1])
  ctx.unload_camera(cam)
 ctx.verify_frozen();manifest=json.loads((ROOT/'code_manifest.json').read_text())
 assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
 report=dict(status='PASS_READ_ONLY_NATIVE_JOINT_GEOMETRY_AUDIT',iteration=40000,rows=rows,optimizer_steps=0,GT_geometry_normal_material_inputs=False,training_modified=False,final_mae=False,native_provenance=ctx.pbr_rasterizer_provenance,source_files=ctx.source_files,limits='joint颜色各pass保持全部Gaussian原opacity，权重是实际联合光栅贡献；projected独立pass仅核验符号。source候选不等于真实平台物体标签，颜色/计数不得用作评分mask或删点。')
 path=ROOT/'inspection/native_joint_geometry.json';assert not path.exists();path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
