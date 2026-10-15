"""位移头实测下降回溯探针；梯度差分与实际下降分别报告，无GT。"""
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
 differences=[];gradient_rows=[]
 for i in cached:
  full,_,_,fixed=forward(i);gfull=torch.autograd.grad(full,parameter)[0]
  projection=forward(i,'projection',fixed)[0];gproj=torch.autograd.grad(projection,parameter)[0]
  normal=forward(i,'normal',fixed)[0];gnormal=torch.autograd.grad(normal,parameter)[0]
  decomposition=float((gfull-gproj-gnormal).abs().max())
  gradient_rows.append(dict(frame=i,full_norm=float(gfull.norm()),projection_norm=float(gproj.norm()),normal_norm=float(gnormal.norm()),decomposition_max=decomposition))
  for mode,gradient in [('full',gfull),('normal',gnormal)]:
   coordinate=int(gradient.abs().argmax());analytic=float(gradient.flatten()[coordinate]);values=[]
   with torch.no_grad():
    flat=parameter.flatten();original=float(flat[coordinate])
    for eps in [1e-5,3e-6,1e-6,3e-7,1e-7]:
     if args.compensated_head_difference:
      xp=float(np.float32(original+eps));xm=float(np.float32(original-eps))
      plus=float(forward(i,mode,fixed,(coordinate,xp-original))[0]);minus=float(forward(i,mode,fixed,(coordinate,xm-original))[0])
     else:
      flat[coordinate]=original+eps;xp=float(flat[coordinate]);plus=float(forward(i,mode,fixed)[0])
      flat[coordinate]=original-eps;xm=float(flat[coordinate]);minus=float(forward(i,mode,fixed)[0]);flat[coordinate]=original
     num=(plus-minus)/(xp-xm);values.append(dict(epsilon=eps,numerical=num,relative_error=abs(num-analytic)/max(abs(num),abs(analytic),1e-9)))
   differences.append(dict(frame=i,mode=mode,coordinate=coordinate,analytic=analytic,differences=values,passed=any(values[j]['relative_error']<.2 and values[j+1]['relative_error']<.2 for j in range(len(values)-1))))
 # 差分状态独立记录；本入口以真实损失下降接受更新，不改判差分失败。
 passed=all(v['passed'] for v in differences) and all(v['normal_norm']>0 and v['decomposition_max']<1e-5*max(v['full_norm'],1) for v in gradient_rows)
 audit_passed=passed
 import copy
 history=[];projections=[];updates=0;opt=torch.optim.Adam(head.parameters(),lr=1e-7)
 with torch.no_grad():initial_loss={str(i):float(forward(i)[0]) for i in cached}
 if all(np.isfinite(g['full_norm']) and g['full_norm']>0 for g in gradient_rows):
  for step in range(12):
   start={k:v.detach().clone() for k,v in head.state_dict().items()};optimizer_start=copy.deepcopy(opt.state_dict())
   opt.zero_grad(set_to_none=True);loss=sum(forward(i)[0] for i in cached)/len(cached);baseline=float(loss);loss.backward()
   assert all(torch.isfinite(v.grad).all() for v in head.parameters());opt.step()
   with torch.no_grad():
    maximum=max(float((head(hidden_cache[i])*pc.get_binary_feature()-cached[i][1]['d_xyz']).norm(dim=-1).max()) for i in cached)
    fraction=min(1.,.001/max(maximum,1e-12))
    if fraction<1:
     for k,v in head.state_dict().items():v.copy_(torch.lerp(initial_head[k],v,fraction))
    proposal={k:v.detach().clone() for k,v in head.state_dict().items()};attempts=[];accepted=False
    for step_fraction in [1.,.5,.25,.125,.0625,.03125]:
     for k,v in head.state_dict().items():v.copy_(torch.lerp(start[k],proposal[k],step_fraction))
     trial=float(sum(forward(i)[0] for i in cached)/len(cached));accepted=np.isfinite(trial) and trial<baseline-1e-10
     attempts.append(dict(fraction=step_fraction,loss=trial,accepted=accepted))
     if accepted:break
    if not accepted:
     for k,v in head.state_dict().items():v.copy_(start[k])
     opt.load_state_dict(optimizer_start)
    else:updates+=1
    projections.append(dict(proposed_motion_max=maximum,bound_fraction=fraction,accepted=bool(accepted),attempts=attempts,baseline=baseline))
   history.append(float(loss));print(json.dumps(dict(step=step+1,**projections[-1])),flush=True)
 with torch.no_grad():
  final_loss={}
  for i in cached:
   loss,package,predicted,_=forward(i);final_loss[str(i)]=float(loss)
   save_image(out/f'{i:03d}_rgb.png',linear_to_srgb(predicted.clamp_min(0)));save_image(out/f'{i:03d}_alpha.png',package['rend_alpha'])
   save_image(out/f'{i:03d}_normal.png',(F.normalize(package['surf_normal'],dim=0)*cached[i][2]*.5+.5)*package['rend_alpha'])
   gate=pc.get_binary_feature();ds=dict(cached[i][1]);ds['d_xyz']=head(hidden_cache[i])*gate
   sep=render(cached[i][0],pc,pipe,bg,**ds,override_color=torch.cat([gate,1-gate,torch.zeros_like(gate)],dim=1));save_image(out/f'{i:03d}_separation.png',sep['render'])
   torch.save({k:package[k].cpu() for k in ['surf_depth','surf_normal','rend_alpha']},out/f'{i:03d}_depth.pt')
 unchanged=frozen=={k:digest_tensor(v) for k,v in vars(pc).items() if isinstance(v,torch.Tensor) }
 unchanged=unchanged and frozen_deform=={k:digest_tensor(v) for k,v in deform.deform.state_dict().items() if not k.startswith('gaussian_warp.')}
 delta=max(float((v-initial_head[k]).abs().max()) for k,v in head.state_dict().items())
 with torch.no_grad():motion_max=max(float((head(hidden_cache[i])*pc.get_binary_feature()-cached[i][1]['d_xyz']).norm(dim=-1).max()) for i in cached)
 actual_descent_passed=updates>0 and np.mean(list(final_loss.values()))<np.mean(list(initial_loss.values()))
 passed=actual_descent_passed and unchanged and delta>0 and 0<motion_max<=.00101
 torch.save(dict(geometry={k:getattr(pc,k).detach().cpu() for k in ['_xyz','_scaling','_rotation']},optimizer=opt.state_dict(),deform_state_dict={k:v.detach().cpu() for k,v in deform.deform.state_dict().items()},args=vars(args),geometry_updates=0,deformation_updates=updates,deformation_proposals=len(history)),out/'checkpoint.pt')
 inputs=[Path(__file__),ply,Path(args.model)/'deform/iteration_35000/deform.pth',Path(args.model)/'cfg_args',Path(args.cache),Path(args.lights),Path(args.source)/'transforms_train.json']
 result=dict(status='PASS' if passed else 'FAILED',gt_read=False,compensated_head_difference=args.compensated_head_difference,gradient_audit_passed=audit_passed,actual_descent_passed=bool(actual_descent_passed),scope='仅实测受限下降工程验收；梯度差分独立记录，非科学验收',inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},audit=audit,gradients=gradient_rows,finite_differences=differences,geometry_updates=0,deformation_updates=updates,deformation_proposals=len(history),projections=projections,motion_max=motion_max,head_parameter_delta=delta,frozen_unchanged=unchanged,initial_loss=initial_loss,final_loss=final_loss,history=history)
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','finite_differences','history']},ensure_ascii=False),flush=True)
 if not passed:raise RuntimeError('梯度或更新审计失败，保留结果与checkpoint')

if __name__=='__main__':main()
