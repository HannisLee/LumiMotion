"""冻结世界几何及未知光照的跨时间共享材质拟合；无GT。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import numpy as np,torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--constraint',choices=['bounded','nonnegative'],required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4)
 run=Path('output/smoke_test/1015-41-LH-dynamic-DL-T49全训练时刻预条件/preconditioned');rt=json.loads((run/'runtime.json').read_text());args=rt['args'];assert not args['gt_direction_reference'];inputs={}
 for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json']:
  sha=hashlib.sha256(f.read_bytes()).hexdigest();assert sha==rt['inputs'][str(f)];inputs[str(f.resolve())]=sha
 cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'));deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
 cams=[loadCam(cfg,i,c,1.) for i,c in enumerate(readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False))]
 base=Path('output/smoke_test/1015-68-LH-dynamic-DL-T76同源锚点');hr=Path('output/smoke_test/1015-67-LH-dynamic-DL-T75对应光度配对/fit/result.json');held=set(json.loads(hr.read_text())['held_material_frames']);meta=json.loads((base/'probe/result.json').read_text());edges=[];extra=[hr,base/'probe/result.json',base/'homography/result.json',run/'runtime.json',run/'xyz_metric.pt',Path(__file__)]
 for j,row in enumerate(meta['rows']):
  pp=base/'probe'/f'pair_{j:02d}.npz';hh=base/'homography'/f'residual_{j:02d}.npz';d=np.load(pp);h=np.load(hh);m=h['moving'];extra.extend([pp,hh])
  if not m.any():continue
  aa,bb=map(int,d['indices']);edges.append(dict(a=aa,b=bb,ids=torch.tensor(d['source_ids'][m],device='cuda'),flow=torch.tensor(d['pb'][m]-d['pa'][m],device='cuda'),train=aa not in held and bb not in held))
 assert sum(len(e['ids']) for e in edges)==449
 active=torch.unique(torch.cat([e['ids'] for e in edges if e['train']]));assert len(active)>0
 xyz0=pc.get_xyz.detach().clone();features=pc.get_binary_feature().detach();metric=torch.load(run/'xyz_metric.pt',map_location='cpu',weights_only=False)['diagonal'][active.cpu()].cuda();delta=torch.zeros((len(active),3),device='cuda')
 def forward(i,change):
  camera=cams[i];camera.load2device();xyz=xyz0;ds=deform.step(xyz,camera.fid.reshape(1,1).expand(len(xyz),1),feature=features,camera_center=camera.camera_center);world=xyz+ds['d_xyz'].index_add(0,active,change);clip=torch.cat([world,torch.ones_like(world[:,:1])],-1)@camera.full_proj_transform;uv=(clip[:,:2]/clip[:,3:].clamp_min(1e-8)+1)*200-.5
  return world,uv
 from types import SimpleNamespace
 import torch.nn.functional as F
 from PIL import Image
 from gaussian_renderer import render
 from scene.stage2_photometric import srgb_to_linear
 from scripts.probe_dynamic_dl_selfcal import save_image
 for v in vars(pc).values():
  if isinstance(v,torch.Tensor):v.requires_grad_(False)
 lp=Path('output/1015-15-LH-dynamic-DL-T05复核/seed_1015/checkpoint.pt');ep=Path('output/smoke_test/1015-49-LH-dynamic-DL-T57底项约定修正配对/supported/checkpoint.pt')
 directions=torch.load(lp,map_location='cpu',weights_only=False)['smooth'].cuda();energy=torch.load(ep,map_location='cpu',weights_only=False)['energy'].cuda()


 fit=Path('output/smoke_test/1015-75-LH-dynamic-DL-T83共享材质/fit')
 prior=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False);cached=torch.load(fit/'frame_cache.pt',map_location='cpu',weights_only=False);old=json.loads((fit/'result.json').read_text())
 for f,h in old['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 cached={i:{k:({kk:(vv.cuda() if isinstance(vv,torch.Tensor) else vv) for kk,vv in v.items()} if k=='ds' else v.cuda()) for k,v in q.items()} for i,q in cached.items()}
 train_used=prior['train_frames'];held_used=prior['held_frames'];assert torch.equal(prior['active_ids'],active.cpu())
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda')
 material=torch.full_like(xyz0,.5,requires_grad=True);opt=torch.optim.Adam([material],lr=.03)
 def project(x):return x.clamp(0,1) if a.constraint=='bounded' else x.clamp_min(0)
 def objective(i):
  c=cams[i];c.load2device();q=cached[i];pack=render(c,pc,pipe,bg,**q['ds'],override_color=material*energy[i]);pred=pack['render']*q['shading'];loss=((pred-q['target']).double().square()*q['mask']).sum()/(3*q['mask'].sum())
  return loss,pred
 def evaluate():
  values={i:float(objective(i)[0]) for i in cached}
  for i in cached:cams[i].load2device('cpu')
  return dict(per_frame=values,train=float(np.mean([values[i] for i in train_used])),held=float(np.mean([values[i] for i in held_used])))
 def gradient():
  opt.zero_grad(set_to_none=True);total=0.
  for i in train_used:
   with torch.enable_grad():loss,_=objective(i);(loss/len(train_used)).backward()
   total+=float(loss)/len(train_used);cams[i].load2device('cpu')
  assert torch.isfinite(material.grad).all()
  return total
 initial=evaluate();assert abs(initial['train']-old['initial']['train'])<1e-9;history=[]
 for epoch in range(120):
  total=gradient();pg=float((material-project(material-material.grad)).abs().max());opt.step();material.copy_(project(material))
  history.append(dict(epoch=epoch,before=total,projected_gradient_inf=pg))
  if epoch%10==0:print(json.dumps(history[-1]),flush=True)
 final=evaluate();gradient();pg=float((material-project(material-material.grad)).abs().max());albedo=material.detach()
 viz=old['visualized_frames']
 for i in viz:
  _,pred=objective(i);save_image(out/f'{i:03d}_prediction.png',pred);save_image(out/f'{i:03d}_target.png',cached[i]['target']);save_image(out/f'{i:03d}_mask.png',cached[i]['mask'].expand(3,-1,-1).float());cams[i].load2device('cpu')
 extra.extend([lp,ep,fit/'checkpoint.pt',fit/'frame_cache.pt',fit/'result.json'])
 inputs.update({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in extra})
 torch.save(dict(protocol='world_shared_material_projected_v1',constraint=a.constraint,albedo=albedo.cpu(),active_ids=active.cpu(),args=args,directions=directions.cpu(),energy=energy.cpu(),train_frames=train_used,held_frames=held_used,geometry_updates=0,material_updates=120,cache_source=str((fit/'frame_cache.pt').resolve())),out/'checkpoint.pt')
 torch.save(dict(final_gradient=material.grad.cpu(),initial_value=.5),out/'gradient.pt')
 result=dict(gt_read=False,constraint=a.constraint,geometry_updates=0,material_updates=120,initial=initial,final=final,history=history,final_projected_gradient_inf=pg,minimum=float(albedo.min()),maximum=float(albedo.max()),zero_channels=int((albedo==0).sum()),at_one_channels=int((albedo==1).sum()),above_one_channels=int((albedo>1).sum()),train_frames=train_used,held_frames=held_used,visualized_frames=viz,inputs=inputs,scope='同初值预算的直接材质系数投影Adam；非负系数不解释为绝对物理反射率；固定几何/未知灯光')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['history','inputs','initial','final']}),flush=True)
if __name__=='__main__':main()
