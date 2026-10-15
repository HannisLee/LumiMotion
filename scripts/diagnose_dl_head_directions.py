"""按实际位移归一化的多方向导数检查；零更新、无GT。"""
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
 p.add_argument('--directions-from')
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

 loss=sum(forward(i)[0] for i in cached)/len(cached)
 gradients=torch.autograd.grad(loss,tuple(head.parameters()))
 gradient={k:g.detach() for (k,_),g in zip(head.named_parameters(),gradients)}
 baseline=float(loss)
 if args.directions_from:
  payload=torch.load(args.directions_from,map_location='cpu',weights_only=False)
  assert all(torch.equal(payload['initial_head'][k],v.cpu()) for k,v in initial_head.items())
  assert payload['frames']==list(cached)
  directions_to_test={name:{k:v.cuda() for k,v in direction.items()} for name,direction in payload['directions'].items()}
 else:
  generator=torch.Generator(device='cpu').manual_seed(20261015)
  directions_to_test={}
  for j in range(4):
   directions_to_test['random_'+str(j)]={k:(torch.randint(0,2,v.shape,generator=generator)*2-1).to(device='cuda',dtype=v.dtype) for k,v in initial_head.items()}
  directions_to_test['low_gradient_descent']={k:-g for k,g in gradient.items()}
  with torch.no_grad():
   for name,direction in directions_to_test.items():
    maximum=max(float((F.linear(hidden_cache[i],direction['weight'],direction['bias'])*pc.get_binary_feature()).norm(dim=-1).max()) for i in cached)
    assert maximum>0
    directions_to_test[name]={k:v/maximum for k,v in direction.items()}
 torch.save(dict(initial_head={k:v.cpu() for k,v in initial_head.items()},frames=list(cached),directions={name:{k:v.cpu() for k,v in direction.items()} for name,direction in directions_to_test.items()}),out/'directions.pt')
 rows=[]
 with torch.no_grad():
  for name,direction in directions_to_test.items():
   analytic=sum(float((gradient[k].double()*v.double()).sum()) for k,v in direction.items())
   measurements=[]
   for epsilon in [1e-4,3e-5,1e-5]:
    observations=[]
    for sign_delta in [1,-1]:
     for k,v in head.state_dict().items():v.copy_(initial_head[k]+sign_delta*epsilon*direction[k])
     losses={str(i):float(forward(i)[0]) for i in cached}
     motion=max(float((head(hidden_cache[i])*pc.get_binary_feature()-cached[i][1]['d_xyz']).norm(dim=-1).max()) for i in cached)
     linear=sum(float((gradient[k].double()*(v-initial_head[k]).double()).sum()) for k,v in head.state_dict().items())
     observations.append(dict(loss=sum(losses.values())/len(losses),frame_losses=losses,max_motion=motion,linear=linear))
    numerical=(observations[0]['loss']-observations[1]['loss'])/(2*epsilon)
    represented=(observations[0]['linear']-observations[1]['linear'])/(2*epsilon)
    error=abs(numerical-represented)/max(abs(numerical),abs(represented),1e-9)
    measurements.append(dict(epsilon=epsilon,analytic_represented=represented,numerical=numerical,relative_error=error,positive=observations[0],negative=observations[1]))
    for k,v in head.state_dict().items():v.copy_(initial_head[k])
   passed=any(measurements[j]['relative_error']<.2 and measurements[j+1]['relative_error']<.2 for j in range(2))
   rows.append(dict(direction=name,analytic=analytic,passed=passed,measurements=measurements))
   print(json.dumps(rows[-1]),flush=True)
 unchanged=all(torch.equal(v,initial_head[k]) for k,v in head.state_dict().items()) and frozen=={k:digest_tensor(v) for k,v in vars(pc).items() if isinstance(v,torch.Tensor)}
 unchanged=unchanged and frozen_deform=={k:digest_tensor(v) for k,v in deform.deform.state_dict().items() if not k.startswith('gaussian_warp.')}
 assert unchanged
 result=dict(status='PASS' if all(r['passed'] for r in rows) else 'FAILED',scope='五个方向的数值检查；非全参数梯度证明或几何验收',gt_read=False,updates=0,frozen_unchanged=unchanged,baseline=baseline,rows=rows,audit=audit,args=vars(args),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
