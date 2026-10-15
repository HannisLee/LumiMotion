"""隔离复制原生光栅源码，仅打印两个像素的贡献路径；零更新。"""
import os
import sys
import shutil
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
env=Path('/home/han.li/miniconda3/envs/lumimotion-ushas')
os.environ['PATH']=str(env/'bin')+':'+os.environ.get('PATH','')
os.environ['CPATH']=str(env/'targets/x86_64-linux/include')+':'+os.environ.get('CPATH','')
os.environ.setdefault('CUDA_HOME',str(env));os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6');os.environ.setdefault('MAX_JOBS','4')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
import scene.pbr_rasterizer as backend
copy_root=Path(sys.argv[sys.argv.index('--output')+1]).resolve().parent/'instrumented_source'
shutil.copytree(backend.SOURCE,copy_root,ignore=shutil.ignore_patterns('third_party','build','*.so','__pycache__'))
p=copy_root/'cuda_rasterizer/forward.cu';source=p.read_text()
needle='float alpha = min(0.99f, nor_o.w * exp(power));'
assert source.count(needle)==1
trace='\n            if (W == 200 && H == 200 && ((pix.x == 127 && pix.y == 66) || (pix.x == 125 && pix.y == 71)) && alpha >= 0.00001f) {\n                printf("DLTRACE %u %u %u %d %.9g %.9g %.9g %.9g %.9g %.9g\\n", pix.x, pix.y, contributor, collected_id[j], rho3d, rho2d, depth, alpha, T, T*(1-alpha));\n            }\n'
p.write_text(source.replace(needle,needle+trace))
backend.SOURCE=copy_root
module=backend.native_module();sys.modules['diff_surfel_rasterization']=module
PROVENANCE=module.provenance
for name in list(sys.modules):
 if name.split('.')[0] in {'scene','utils','arguments','gaussian_renderer'}:del sys.modules[name]
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


 i=0;coordinate=162;original=float(parameter.flatten()[coordinate]);results=[]
 torch.cuda.synchronize()
 for label,delta in [('base',0.),('plus',1e-6),('minus',-1e-6),('small_plus',1e-7),('small_minus',-1e-7)]:
  print('TRACE_BEGIN '+label,flush=True)
  with torch.no_grad():
   parameter.flatten()[coordinate]=original+delta
   loss,pkg,_,_=forward(i)
   values=[dict(y=y,x=x,depth=float(pkg['surf_depth'][0,y,x]),alpha=float(pkg['rend_alpha'][0,y,x])) for y,x in [(66,127),(71,125)]]
  torch.cuda.synchronize()
  results.append(dict(label=label,weight=float(parameter.flatten()[coordinate]),loss=float(loss),pixels=values))
  print('TRACE_END '+label,flush=True)
 with torch.no_grad():parameter.flatten()[coordinate]=original
 assert all(torch.equal(v,initial_head[k]) for k,v in head.state_dict().items())
 (out/'result.json').write_text(json.dumps(dict(gt_read=False,updates=0,rows=results,provenance=PROVENANCE,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
