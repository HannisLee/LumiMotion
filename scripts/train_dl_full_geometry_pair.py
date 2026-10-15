"""固定未知光的真实几何短配对；每次更新由完整前向和动态位移限制验收。"""
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
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--group',choices=['photo','normal'],required=True);p.add_argument('--verify');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4)
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
 weight=.1 if a.group=='normal' else 0.
 train=ck['train_frames'];held=ck['held_frames']
 if a.verify:
  resumed=torch.load(Path(a.verify)/'checkpoint.pt',map_location='cpu',weights_only=False)
  assert resumed['group']==a.group and resumed['protocol']=='full_geometry_linesearch_v1'
  for name,param in zip(names,params):param.copy_(resumed['parameters'][name].cuda())
 def evaluate():
  values={}
  for i in train+held:
   cameras[i].load2device();ls,_,_=forward(i);values[i]={k:float(v) for k,v in ls.items()};cameras[i].load2device('cpu')
  return dict(train={k:float(np.mean([values[i][k] for i in train])) for k in ['photo','normal','distortion']},held={k:float(np.mean([values[i][k] for i in held])) for k in ['photo','normal','distortion']},per_frame=values)
 initial=evaluate()
 if a.verify:
  expected=json.loads((Path(a.verify)/'result.json').read_text())['final'];err=max(abs(v-expected['per_frame'][str(i)][k]) for i,ls in initial['per_frame'].items() for k,v in ls.items());assert err<1e-9
  for k,v in ck['deform_state_dict'].items():assert torch.equal(v,deform.deform.state_dict()[k].cpu())
  (out/'result.json').write_text(json.dumps(dict(status='PASS',frames=55,max_error=err,frozen_deform_exact=True),indent=2));return
 # 光度起点核验，不将T98未更新参数的缓存形变用于后续几何。
 assert max(abs(v['photo']-r['final']['per_frame'][str(i)]) for i,v in initial['per_frame'].items())<1e-9
 def worlds():
  values=[]
  for i in range(len(cameras)):
   c=cameras[i];c.load2device();ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center)
   values.append((pc.get_xyz+ds['d_xyz']).detach().cpu());c.load2device('cpu')
  return values
 initial_world=worlds();previous_world=initial_world;history=[]
 for step in range(10):
  sums=[torch.zeros_like(v) for v in params];before=0.
  for i in train:
   c=cameras[i];c.load2device()
   with torch.enable_grad():
    ls,_,_=forward(i);loss=ls['photo']+weight*ls['normal'];gg=torch.autograd.grad(loss,params)
   for g,total in zip(gg,sums):assert torch.isfinite(g).all();total.add_(g/len(train))
   before+=float(loss)/len(train);c.load2device('cpu')
  old=[v.clone() for v in params];deltas=[]
  for g,cap in zip(sums,[1e-5,1e-4,1e-4]):deltas.append(-g/g.reshape(len(g),-1).norm(dim=-1).max().clamp_min(1e-30)*cap)
  trials=[];accepted=False
  for attempt in range(8):
   fraction=.5**attempt
   for v,base,d in zip(params,old,deltas):v.copy_(base+fraction*d)
   after=0.
   for i in train:
    cameras[i].load2device();ls,_,_=forward(i);after+=float(ls['photo']+weight*ls['normal'])/len(train);cameras[i].load2device('cpu')
   current_world=worlds();max_step=max(float((x-y).norm(dim=-1).max()) for x,y in zip(current_world,previous_world));max_total=max(float((x-y).norm(dim=-1).max()) for x,y in zip(current_world,initial_world))
   ok=np.isfinite(after) and after<before and max_step<=1e-4 and max_total<=1e-3
   trials.append(dict(fraction=fraction,loss=after,max_world_step=max_step,max_world_total=max_total,accepted=bool(ok)))
   if ok:accepted=True;previous_world=current_world;break
  if not accepted:
   for v,base in zip(params,old):v.copy_(base)
  row=dict(step=step,before=before,gradient_norms=[float(g.norm()) for g in sums],accepted=accepted,trials=trials);history.append(row);print(json.dumps(row),flush=True)
  if not accepted:break
 final=evaluate()
 from scripts.probe_dynamic_dl_selfcal import save_image
 for i in [2,53,126,62]:
  c=cameras[i];c.load2device();_,pack,_=forward(i);q=cache[i];n=F.normalize(pack['surf_normal'],dim=0)*q['sign'].cuda();pred=pack['render']*E[i]*((n*D[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1)
  save_image(out/f'{i:03d}_rgb.png',pred);save_image(out/f'{i:03d}_alpha.png',pack['rend_alpha'].expand(3,-1,-1));save_image(out/f'{i:03d}_normal.png',(n+1)/2)
  ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center);ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
  gate=render(c,pc,pipe,bg,**ds,override_color=pc.get_binary_feature().expand(-1,3))['render'];save_image(out/f'{i:03d}_dynamic_gate.png',gate);c.load2device('cpu')
 for k,v in ck['deform_state_dict'].items():assert torch.equal(v,deform.deform.state_dict()[k].cpu())
 for k,v in ck['geometry'].items():
  if k not in ['_xyz','_rotation','_scaling']:assert torch.equal(v,vars(pc)[k].detach().cpu()),k
 changes={name:dict(norm=float((v-original).norm()),max_abs=float((v-original).abs().max())) for name,v,original in zip(names,params,originals)}
 inputs={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),fit/'checkpoint.pt',fit/'result.json',cache_path]}
 torch.save(dict(protocol='full_geometry_linesearch_v1',group=a.group,parameters={n:v.cpu() for n,v in zip(names,params)},geometry={k:v.detach().cpu() for k,v in vars(pc).items() if isinstance(v,torch.Tensor)},deform_state_dict=ck['deform_state_dict'],albedo=albedo.cpu(),directions=D.cpu(),energy=E.cpu(),args=args,config=vars(cfg),train_frames=train,held_frames=held,source_checkpoint=str(fit/'checkpoint.pt')),out/'checkpoint.pt')
 (out/'result.json').write_text(json.dumps(dict(protocol='full_geometry_linesearch_v1',group=a.group,gt_read=False,initial=initial,final=final,history=history,changes=changes,geometry_updates=sum(x['accepted'] for x in history),normal_weight=weight,inputs=inputs,elapsed_seconds=time.monotonic()-start,scope='完整canonical几何与冻结MLP输入导数，前向下降/135时刻位移验收；未做GT深度网格评测，不代表几何恢复'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
