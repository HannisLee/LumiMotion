"""冻结形变MLP的位置输入Jacobian与缓存近似审计；零更新、无GT。"""
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
 for key in ['model','source','lights','cache','checkpoint','output']:p.add_argument('--'+key,required=True)
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
 pc._xyz.requires_grad_(True);initial=pc._xyz.detach().clone()
 frozen={k:digest_tensor(v) for k,v in vars(pc).items() if isinstance(v,torch.Tensor) and k!='_xyz'}
 deform_hash={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
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
 def forward(i,mode='cached'):
  camera,ds,sign,mask,target,saturated=cached[i]
  if mode!='cached':
   query=deform.step(pc._xyz,camera.fid.reshape(1,1).expand(len(initial),1),feature=pc.get_binary_feature(),camera_center=camera.camera_center)
   ds={k:query[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
   if mode=='detached':ds={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in ds.items()}
  package=render(camera,pc,pipe,bg,**ds,override_color=colors)
  normal=F.normalize(package['surf_normal'],dim=0)*sign
  shading=(normal*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)
  appearance=package['render']
  predicted=appearance*shading
  residual=predicted-target;residual=torch.where(saturated,residual.clamp_max(0),residual)
  loss=(residual.double().square()*mask).sum()/(3*mask.sum())
  return loss,package,predicted,(appearance.detach(),shading.detach())

 rows=[];jacobians=[];gate=pc.get_binary_feature().detach().flatten();dynamic=gate>.5
 def stats(v):
  v=v.detach().float().flatten()
  return dict(mean=float(v.mean()),median=float(v.median()),p95=float(torch.quantile(v,.95)),max=float(v.max()))
 for i in cached:
  gradients={};predictions={};losses={}
  for mode in ['cached','detached','full']:
   loss,package,predicted,_=forward(i,mode)
   gradients[mode]=torch.autograd.grad(loss,pc._xyz)[0].detach()
   predictions[mode]=predicted.detach();losses[mode]=float(loss)
  difference=gradients['full']-gradients['detached']
  regions={}
  for name,mask in [('dynamic',dynamic),('static',~dynamic)]:
   a=gradients['full'][mask].flatten();b=gradients['detached'][mask].flatten()
   regions[name]=dict(points=int(mask.sum()),full_norm=float(a.norm()),detached_norm=float(b.norm()),difference_norm=float((a-b).norm()),cosine=float(F.cosine_similarity(a,b,dim=0)))
  coord=int(gradients['full'].abs().argmax());analytic=float(gradients['full'].flatten()[coord]);differences=[]
  with torch.no_grad():
   flat=pc._xyz.flatten();original=float(flat[coord])
   for eps in [1e-4,3e-5,1e-5,3e-6,1e-6]:
    flat[coord]=original+eps;xp=float(flat[coord]);plus=float(forward(i,'full')[0])
    flat[coord]=original-eps;xm=float(flat[coord]);minus=float(forward(i,'full')[0]);flat[coord]=original
    numerical=(plus-minus)/(xp-xm);differences.append(dict(epsilon=eps,numerical=numerical,relative_error=abs(numerical-analytic)/max(abs(numerical),abs(analytic),1e-9)))
  fd_ok=any(differences[j]['relative_error']<.2 and differences[j+1]['relative_error']<.2 for j in range(4))
  rows.append(dict(frame=i,losses=losses,forward_max_difference=float((predictions['full']-predictions['cached']).abs().max()),cached_detached_gradient_max=float((gradients['cached']-gradients['detached']).abs().max()),regions=regions,coordinate=coord,analytic=analytic,finite_differences=differences,gradient_passed=fd_ok))
  # 固定、均匀抽样预测动态身份；不依赖GT选择点。
  ids=torch.where(dynamic)[0];ids=ids[torch.linspace(0,len(ids)-1,512,device='cuda').long()]
  positions=initial[ids].clone().requires_grad_(True);camera=cached[i][0]
  query=deform.step(positions,camera.fid.reshape(1,1).expand(len(ids),1),feature=pc.get_binary_feature()[ids],camera_center=camera.camera_center)
  derivatives=torch.stack([torch.autograd.grad(query['d_xyz'][:,k].sum(),positions,retain_graph=True)[0] for k in range(3)],dim=1)
  jac=derivatives+torch.eye(3,device='cuda')[None];singular=torch.linalg.svdvals(jac)
  jacobians.append(dict(frame=i,points=len(ids),displacement_jacobian_norm=stats(derivatives.norm(dim=(1,2))),mapping_singular_min=stats(singular[:,-1]),mapping_singular_max=stats(singular[:,0]),negative_determinants=int((torch.linalg.det(jac)<0).sum())))
  print(json.dumps(rows[-1]),flush=True)
 endpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
 runtime=json.loads((Path(args.checkpoint).parent/'runtime.json').read_text())
 for path in [ply,Path(args.model)/'deform/iteration_35000/deform.pth',Path(args.model)/'cfg_args']:
  assert runtime['inputs'][str(path.resolve())]==hashlib.sha256(path.read_bytes()).hexdigest()
 assert all(torch.equal(getattr(pc,k).detach().cpu(),endpoint['geometry'][k]) for k in ['_rotation','_scaling'])
 endpoints=[]
 with torch.no_grad():
  pc._xyz.copy_(endpoint['geometry']['_xyz'].cuda())
  for i in cached:
   camera,ds,sign,mask,target,saturated=cached[i]
   current=deform.step(pc._xyz,camera.fid.reshape(1,1).expand(len(initial),1),feature=pc.get_binary_feature(),camera_center=camera.camera_center)
   delta=current['d_xyz']-ds['d_xyz'];pred={}
   for mode in ['cached','full']:
    loss,package,rgb,_=forward(i,mode);pred[mode]={k:package[k].clone() for k in ['surf_depth','rend_alpha']}
    prefix=out/f'{i:03d}_{mode}'
    save_image(str(prefix)+'_rgb.png',linear_to_srgb(rgb.clamp_min(0)));save_image(str(prefix)+'_alpha.png',package['rend_alpha'])
    save_image(str(prefix)+'_normal.png',(F.normalize(package['surf_normal'],dim=0)*sign*.5+.5)*package['rend_alpha'])
    ds_used=ds if mode=='cached' else {k:current[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
    binary=pc.get_binary_feature();sep=render(camera,pc,pipe,bg,**ds_used,override_color=torch.cat([binary,1-binary,torch.zeros_like(binary)],dim=1));save_image(str(prefix)+'_separation.png',sep['render'])
    torch.save({k:package[k].cpu() for k in ['surf_depth','surf_normal','rend_alpha']},str(prefix)+'_depth.pt')
   endpoints.append(dict(frame=i,dynamic_displacement_difference=stats(delta[dynamic].norm(dim=-1)),static_displacement_max=float(delta[~dynamic].abs().max()),depth_supported_difference=stats((pred['full']['surf_depth']-pred['cached']['surf_depth']).abs()[mask.bool()]),alpha_max_difference=float((pred['full']['rend_alpha']-pred['cached']['rend_alpha']).abs().max())))
  pc._xyz.copy_(initial)
 unchanged=all(torch.equal(pc._xyz,initial) for _ in [0]) and frozen=={k:digest_tensor(v) for k,v in vars(pc).items() if isinstance(v,torch.Tensor) and k!='_xyz'} and deform_hash=={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
 passed=unchanged and all(r['gradient_passed'] and r['forward_max_difference']<1e-5 and r['cached_detached_gradient_max']<1e-5 for r in rows)
 inputs=[Path(__file__),ply,Path(args.model)/'deform/iteration_35000/deform.pth',Path(args.model)/'cfg_args',Path(args.cache),Path(args.lights),Path(args.checkpoint)]
 result=dict(status='PASS' if passed else 'FAILED',gt_read=False,optimizer_updates=0,parameters_unchanged=unchanged,scope='形变输入Jacobian诊断，不是科学验收',audit=audit,initial=rows,jacobians=jacobians,endpoint=endpoints,inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(status=result['status'],jacobians=jacobians,endpoint=endpoints)),flush=True)
 if not passed:raise RuntimeError('形变审计失败；已保留证据')

if __name__=='__main__':main()
