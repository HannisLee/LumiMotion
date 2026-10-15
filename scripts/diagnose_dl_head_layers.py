"""位移头分层差分审计：原生输出VJP与双精度深度法线，无GT、零更新。"""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.stage2_photometric import srgb_to_linear,linear_to_srgb
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config,digest_tensor,save_image


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['model','source','lights','cache','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--compensated-head-difference',action='store_true')
 p.add_argument('--pixel-audit',action='store_true')
 p.add_argument('--fixed-coordinates',action='store_true')
 p.add_argument('--extended-scales',action='store_true')
 args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
 torch.manual_seed(1015)
 cfg=read_config(Path(args.model)/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
 ply=Path(args.model)/'point_cloud/iteration_35000/point_cloud.ply';pc.load_ply(str(ply))
 deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
 assert deform.load_weights(args.model,35000);deform.deform.eval().requires_grad_(False)
 for value in vars(pc).values():
  if isinstance(value,torch.Tensor):value.requires_grad_(False)
 initial=pc._xyz.detach().clone()
 head=deform.deform.gaussian_warp
 initial_head={k:v.detach().clone() for k,v in head.state_dict().items()}
 frozen_deform={k:digest_tensor(v) for k,v in deform.deform.state_dict().items() if not k.startswith('gaussian_warp.')}
 hidden_cache={}
 frozen={k:digest_tensor(v) for k,v in vars(pc).items() if isinstance(v,torch.Tensor) }
 light=json.loads(Path(args.lights).read_text());assert not light.get('gt_read',False) and not light.get('gt_direction_reference',False)
 tracks=torch.load(args.cache,map_location='cpu',weights_only=False)
 infos=readCamerasFromTransforms(args.source,'transforms_train.json',cfg.train_light_folder,False)
 assert [f['name'] for f in tracks['frames']]==light['frames']
 assert torch.equal(tracks['canonical_xyz'],initial[tracks['ids'].cuda()].cpu())
 directions=torch.tensor(light['directions'],device='cuda')
 train=light['train_indices'];support=tracks['valid'][train].sum(0)>=10
 lum=tracks['rgb'].mean(-1);reference=torch.nanquantile(torch.where(tracks['valid'][train],lum[train],torch.nan),.9,dim=0)
 bright=(lum/reference.clamp_min(.05)[None]>=.65)&torch.isfinite(reference)[None]&(reference[None]>.05)
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
 bg=torch.zeros(3,device='cuda');colors=torch.full_like(initial,.5);cached={};audit=[]
 for i in [0,60,134]:
  assert i in train
  camera=loadCam(cfg,i,infos[i],1.);camera.load2device()
  with torch.no_grad():
   ds=deform.step(initial,camera.fid.reshape(1,1).expand(len(initial),1),feature=pc.get_binary_feature(),camera_center=camera.camera_center)
   hidden_cache[i]=ds['hidden'].detach()
   ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
   base=render(camera,pc,pipe,bg,**ds,override_color=colors)
   normal=F.normalize(base['surf_normal'],dim=0)
   facing=(normal*(camera.camera_center[:,None,None]-base['surf_point'])).sum(0,keepdim=True)
   sign=torch.where(facing>=0,1.,-1.)
   observed=camera.gt_alpha_mask.cuda()
   valid=(observed>.99)&(base['rend_alpha']>.95)&torch.isfinite(base['surf_depth'])&(base['surf_depth']>0)
   valid=-F.max_pool2d(-valid.float()[None],3,stride=1,padding=1)[0]>.5
   chosen=torch.zeros_like(observed);good=tracks['valid'][i]&support&bright[i];uv=tracks['pixels'][i][good].cuda();chosen[0,uv[:,1],uv[:,0]]=1
   chosen=chosen*valid;assert chosen.sum()>0
   target=srgb_to_linear(camera.original_image_train_light*observed)
   # 饱和从与训练相同的原图阈值取得，保守池化至当前分辨率。
   from PIL import Image
   original=np.asarray(Image.open(camera.image_path_train_light).convert('RGBA'))
   saturated=torch.from_numpy((original[...,:3]>=254).astype(np.float32)).permute(2,0,1)[None].cuda()
   saturated=F.adaptive_max_pool2d(saturated,target.shape[-2:])[0].bool()&(observed>.99)
   assert float((base['render']-.5*base['rend_alpha']).abs().max())<1e-5
   cached[i]=(camera,ds,sign,chosen,target,saturated)
   audit.append(dict(frame=i,supported=int(chosen.sum()),facing_positive_fraction=float((facing[valid]>=0).float().mean()),premultiplied_error=float((base['render']-.5*base['rend_alpha']).abs().max())))
 head.requires_grad_(True)
 parameter=head.weight
 def forward(i,mode='full',fixed=None,head_perturb=None):
  camera,ds,sign,mask,target,saturated=cached[i]
  ds=dict(ds);ds['d_xyz']=head(hidden_cache[i])*pc.get_binary_feature()
  if head_perturb is not None:
   coordinate,delta=head_perturb;axis=coordinate//parameter.shape[1];column=coordinate%parameter.shape[1]
   displacement=cached[i][1]['d_xyz'].double().clone()
   displacement[:,axis]+=float(delta)*hidden_cache[i][:,column].double()*pc.get_binary_feature().flatten().double()
   ds['d_xyz']=displacement.float()
  package=render(camera,pc,pipe,bg,**ds,override_color=colors)
  normal=F.normalize(package['surf_normal'],dim=0)*sign
  shading=(normal*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)
  appearance=package['render']
  if mode=='normal':appearance=fixed[0]
  if mode=='projection':shading=fixed[1]
  predicted=appearance*shading
  residual=predicted-target;residual=torch.where(saturated,residual.clamp_max(0),residual)
  loss=(residual.double().square()*mask).sum()/(3*mask.sum())
  return loss,package,predicted,(appearance.detach(),shading.detach())

 def stable_loss(i,package):
  import math
  camera,ds,sign,mask,target,saturated=cached[i]
  depth=package['surf_depth'].double();h,w=depth.shape[-2:]
  yy,xx=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float64),torch.arange(w,device='cuda',dtype=torch.float64),indexing='ij')
  rays=torch.stack([(xx-w/2)/(w/(2*math.tan(camera.FoVx/2))),(yy-h/2)/(h/(2*math.tan(camera.FoVy/2))),torch.ones_like(xx)],-1)
  rotation=camera.world_view_transform.T.double().inverse()[:3,:3]
  points=depth.permute(1,2,0)*(rays@rotation.T)
  cross=torch.cross(points[2:,1:-1]-points[:-2,1:-1],points[1:-1,2:]-points[1:-1,:-2],dim=-1)
  normal=F.pad(F.normalize(cross,dim=-1).permute(2,0,1),(1,1,1,1))*sign.double()
  shade=(normal*directions[i,:,None,None].double()).sum(0,keepdim=True).clamp_min(0)
  residual=package['render'].double()*shade-target.double()
  residual=torch.where(saturated,residual.clamp_max(0),residual)
  loss=(residual.square()*mask).sum()/(3*mask.sum())
  return loss,normal
 def rel(a,b):return abs(a-b)/max(abs(a),abs(b),1e-9)
 rows=[]
 for i in cached:
  loss,base,_,_=forward(i)
  grad=torch.autograd.grad(loss,parameter,retain_graph=True)[0]
  gd,ga=torch.autograd.grad(loss,[base['surf_depth'],base['render']],retain_graph=True)
  double_loss,normal64=stable_loss(i,base)
  g64=torch.autograd.grad(double_loss,parameter,retain_graph=True)[0]
  gd64,ga64=torch.autograd.grad(double_loss,[base['surf_depth'],base['render']])
  coordinate=({0:162,60:196,134:29}[i] if args.fixed_coordinates else int(grad.abs().argmax()));analytic=float(grad.flatten()[coordinate]);analytic64=float(g64.flatten()[coordinate])
  saved_depth=base['surf_depth'].detach().double();saved_rgb=base['render'].detach().double()
  values=[]
  with torch.no_grad():
   flat=parameter.flatten();original=float(flat[coordinate])
   for eps in ([3e-4,1e-4,3e-5] if args.extended_scales else [])+[1e-5,3e-6,1e-6,3e-7,1e-7]:
    observations=[]
    for sign_delta in [1,-1]:
     flat[coordinate]=original+sign_delta*eps;actual=float(flat[coordinate])
     v,pkg,_,_=forward(i);v64,_=stable_loss(i,pkg)
     dd=pkg['surf_depth'].double()-saved_depth;da=pkg['render'].double()-saved_rgb
     linear=float((gd.double()*dd).sum()+(ga.double()*da).sum())
     linear64=float((gd64.double()*dd).sum()+(ga64.double()*da).sum())
     motion=float((head(hidden_cache[i])*pc.get_binary_feature()-cached[i][1]['d_xyz']).norm(dim=-1).max())
     observations.append((actual,float(v),float(v64),linear,linear64,motion))
    flat[coordinate]=original
    span=observations[0][0]-observations[1][0]
    numeric=[(observations[0][k]-observations[1][k])/span for k in range(1,5)]
    values.append(dict(epsilon=eps,max_motion=max(observations[0][5],observations[1][5]),native=numeric[0],stable=numeric[1],native_linear=numeric[2],stable_linear=numeric[3],native_error=rel(numeric[0],analytic),stable_error=rel(numeric[1],analytic64),native_linear_error=rel(numeric[2],analytic),stable_linear_error=rel(numeric[3],analytic64)))
   valid=cached[i][3].bool().expand_as(normal64)
   normal_error=float((normal64-F.normalize(base['surf_normal'],dim=0)*cached[i][2]).abs()[valid].max())
  row=dict(frame=i,coordinate=coordinate,analytic=analytic,analytic64=analytic64,loss=float(loss),loss64=float(double_loss),normal_max_difference=normal_error,differences=values)
  for kind in ['native','stable','native_linear','stable_linear']:
   original_values=values[-5:]
   row[kind+'_passed']=any(original_values[j][kind+'_error']<.2 and original_values[j+1][kind+'_error']<.2 for j in range(4))
   if args.extended_scales:
    row[kind+'_extended_passed']=any(values[j][kind+'_error']<.2 and values[j+1][kind+'_error']<.2 for j in range(3))
  if args.pixel_audit and i==0:
   # 固定原先失败的1e-6步长，按深度VJP数值贡献选择16个像素，仅用于诊断。
   fresh_loss,fresh,_,_=forward(i)
   fresh_gd=torch.autograd.grad(fresh_loss,fresh['surf_depth'],retain_graph=True)[0].detach()
   sampled={}
   with torch.no_grad():
    for eps in [1e-5,3e-6,1e-6,3e-7,1e-7]:
     pair=[]
     for delta in [eps,-eps]:
      parameter.flatten()[coordinate]=original+delta
      actual=float(parameter.flatten()[coordinate]);pkg=forward(i)[1]
      pair.append((actual,pkg['surf_depth'].detach().clone()))
     parameter.flatten()[coordinate]=original
     sampled[eps]=(pair[0][1]-pair[1][1]).double()/(pair[0][0]-pair[1][0])
    score=(fresh_gd.double()*sampled[1e-6]).abs().flatten()
    selected=score.topk(16).indices
   pixels=[]
   for pixel in selected.tolist():
    pixel_gradient=torch.autograd.grad(fresh['surf_depth'].flatten()[pixel],parameter,retain_graph=True)[0]
    derivative=float(pixel_gradient.flatten()[coordinate]);weight=float(fresh_gd.flatten()[pixel])
    pixels.append(dict(pixel=[pixel//fresh['surf_depth'].shape[-1],pixel%fresh['surf_depth'].shape[-1]],depth=float(fresh['surf_depth'].flatten()[pixel]),alpha=float(fresh['rend_alpha'].flatten()[pixel]),weight=weight,analytic=derivative,analytic_contribution=derivative*weight,differences=[dict(epsilon=eps,numerical=float(v.flatten()[pixel]),weighted=float(v.flatten()[pixel])*weight) for eps,v in sampled.items()]))
   row['depth_pixel_audit']=pixels
   row['top16_numeric_contribution']=sum(v['differences'][2]['weighted'] for v in pixels)
   row['top16_analytic_contribution']=sum(v['analytic_contribution'] for v in pixels)
   row['total_depth_numeric_contribution']=float((fresh_gd.double()*sampled[1e-6]).sum())
   row['total_depth_analytic_contribution']=float(torch.autograd.grad(fresh['surf_depth'],parameter,grad_outputs=fresh_gd)[0].flatten()[coordinate])
  rows.append(row);print(json.dumps(row),flush=True)
 frozen_ok=frozen=={k:digest_tensor(v) for k,v in vars(pc).items() if isinstance(v,torch.Tensor)}
 frozen_ok=frozen_ok and all(torch.equal(v,initial_head[k]) for k,v in head.state_dict().items())
 assert frozen_ok
 result=dict(scope='分层数值诊断，非梯度门槛放宽或几何验收',gt_read=False,updates=0,frozen_unchanged=frozen_ok,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),args=vars(args))
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
