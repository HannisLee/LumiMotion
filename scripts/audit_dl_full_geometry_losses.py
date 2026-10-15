"""固定T98未知光，核验动态真实几何的光度/法线/distortion梯度；零更新。"""
import argparse,hashlib,json,shlex,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
 (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
 fit=Path('output/smoke_test/1015-89-LH-dynamic-DL-T98固定新灯向重建材质/alternating_retry1');r=json.loads((fit/'result.json').read_text())
 for f,h in r['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 ck=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False);args=ck['args'];assert not args['gt_direction_reference']
 cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
 deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
 for v in vars(pc).values():
  if isinstance(v,torch.Tensor):v.requires_grad_(False)
 params=[pc._xyz,pc._rotation,pc._scaling];names=['xyz','rotation','scaling']
 for v in params:v.requires_grad_(True)
 originals=[v.detach().clone() for v in params]
 cameras=[loadCam(cfg,i,c,1.) for i,c in enumerate(readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False))]
 cache_path=Path('output/smoke_test/1015-75-LH-dynamic-DL-T83共享材质/fit/frame_cache.pt');cache=torch.load(cache_path,map_location='cpu',weights_only=False)
 D=ck['directions'].cuda();E=ck['energy'].cuda();albedo=ck['albedo'].cuda();pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda')
 from gaussian_renderer import render
 def forward(i):
  c=cameras[i];q=cache[i]
  # 冻结MLP权重，但保留canonical XYZ经形变MLP的导数；每次扰动重新查询。
  ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center)
  ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
  pack=render(c,pc,pipe,bg,**ds,override_color=albedo)
  n=F.normalize(pack['surf_normal'],dim=0);pred=pack['render']*E[i]*((n*q['sign'].cuda()*D[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1)
  mask=q['mask'].cuda();photo=((pred-q['target'].cuda()).double().square()*mask).sum()/(3*mask.sum())
  m=(c.gt_alpha_mask.cuda()>.99).float();m=-F.max_pool2d(-m[None],5,1,2)[0];m=m*(pack['rend_alpha'].detach()>.5)
  normal=((1-(F.normalize(pack['rend_normal'],dim=0)*n).sum(0,keepdim=True))*m).sum()/m.sum().clamp_min(1)
  return dict(photo=photo,normal=normal,distortion=pack['rend_dist'].mean()),pack,ds
 rows=[];saved={}
 for i in [2,53,126]:
  assert i in ck['train_frames'];c=cameras[i];c.load2device()
  with torch.enable_grad():
   losses,pack,ds=forward(i)
   for k,v in cache[i]['ds'].items():
    if isinstance(v,torch.Tensor):assert torch.equal(v,ds[k].detach().cpu()),k
   assert abs(float(losses['photo'])-r['final']['per_frame'][str(i)])<1e-9
   grads={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in losses.items()}
  base={k:float(v) for k,v in losses.items()};del losses,pack,ds
  for term,gg in grads.items():
   for name,param,g in zip(names,params,gg):
    if g is None:rows.append(dict(frame=i,loss=term,param=name,gradient_absent=True));continue
    assert torch.isfinite(g).all();direction=g/g.reshape(len(g),-1).norm(dim=-1).max().clamp_min(1e-30)
    analytic=float((g.double()*direction.double()).sum());original=param.clone();trials=[]
    for eps in [1e-4,1e-5]:
     param.copy_(original+eps*direction);plus=float(forward(i)[0][term])
     param.copy_(original-eps*direction);minus=float(forward(i)[0][term]);param.copy_(original)
     fd=(plus-minus)/(2*eps);rel=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12)
     trials.append(dict(eps=eps,plus=plus,minus=minus,finite_difference=fd,relative_error=rel,same_sign=bool(fd*analytic>0)))
    row=dict(frame=i,loss=term,param=name,baseline=base[term],gradient_norm=float(g.norm()),analytic=analytic,trials=trials);rows.append(row);print(json.dumps(row),flush=True)
   saved[f'{i}_{term}']={n:(v.cpu() if v is not None else None) for n,v in zip(names,gg)}
  for v,orig in zip(params,originals):assert torch.equal(v,orig)
  c.load2device('cpu')
 for k,v in ck['geometry'].items():assert torch.equal(v,vars(pc)[k].detach().cpu()),k
 for k,v in ck['deform_state_dict'].items():assert torch.equal(v,deform.deform.state_dict()[k].cpu()),k
 torch.save(saved,out/'gradients.pt')
 inputs={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),fit/'checkpoint.pt',fit/'result.json',cache_path]}
 result=dict(status='DIAGNOSTIC_COMPLETE',gt_read=False,parameter_updates=0,frames=[2,53,126],rows=rows,inputs=inputs,geometry_exact=True,elapsed_seconds=time.monotonic()-start,scope='完整canonical XYZ重新查询冻结形变MLP；光照材质固定，opacity冻结；有限差分不通过不得宣称梯度有效')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
