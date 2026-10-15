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
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--preconditioner',choices=['t49','none'],default='t49');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4)
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

 extra.extend([lp,ep]);selected=sorted({e[k] for e in edges for k in ['a','b']});train_frames=[i for i in selected if i not in held];held_frames=[i for i in selected if i in held]
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda');cached={};support=[]
 for i in selected:
  camera=cams[i];camera.load2device();ds=deform.step(xyz0,camera.fid.reshape(1,1).expand(len(xyz0),1),feature=features,camera_center=camera.camera_center);ds={k:(ds[k].detach() if isinstance(ds[k],torch.Tensor) else ds[k]) for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
  base_render=render(camera,pc,pipe,bg,**ds,override_color=torch.full_like(xyz0,.5));n=F.normalize(base_render['surf_normal'],dim=0);sign=torch.where((n*(camera.camera_center[:,None,None]-base_render['surf_point'])).sum(0,keepdim=True)>=0,1.,-1.)
  shading=((n*sign)*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1
  marker=torch.zeros_like(xyz0);marker[active]=1;coverage=render(camera,pc,pipe,bg,**ds,override_color=marker)['render'][:1]
  target=srgb_to_linear(camera.original_image_train_light.cuda())
  raw=np.asarray(Image.open(camera.image_path_train_light).convert('RGB'));sat=torch.tensor((raw>=254).any(-1).copy(),device='cuda').float()[None,None];sat=F.max_pool2d(F.adaptive_max_pool2d(sat,target.shape[-2:]),5,1,2)[0]>0
  valid=(camera.gt_alpha_mask.cuda()>.99)&(base_render['rend_alpha']>.99)&(base_render['surf_depth']>0)&torch.isfinite(base_render['surf_depth']);valid=-F.max_pool2d(-valid.float()[None],3,1,1)[0]>.5
  mask=valid&(~sat)&(coverage>.01)
  support.append(dict(frame=i,train=i in train_frames,pixels=int(mask.sum())))
  if mask.sum()==0:camera.load2device('cpu');continue
  cached[i]=dict(ds=ds,shading=shading,sign=sign,mask=mask,target=target)
  camera.load2device('cpu')
 train_used=[i for i in train_frames if i in cached];held_used=[i for i in held_frames if i in cached];assert train_used and held_used
 material=torch.zeros_like(xyz0,requires_grad=True);opt=torch.optim.Adam([material],lr=.03)
 def objective(i):
  c=cams[i];c.load2device();q=cached[i];pack=render(c,pc,pipe,bg,**q['ds'],override_color=material.sigmoid()*energy[i]);pred=pack['render']*q['shading'];loss=((pred-q['target']).double().square()*q['mask']).sum()/(3*q['mask'].sum())
  return loss,pred
 def evaluate():
  values={i:float(objective(i)[0]) for i in cached}
  for i in cached:cams[i].load2device('cpu')
  return dict(per_frame=values,train=float(np.mean([values[i] for i in train_used])),held=float(np.mean([values[i] for i in held_used])))
 initial=evaluate();history=[]
 for epoch in range(12):
  opt.zero_grad(set_to_none=True);total=0.
  for i in train_used:
   with torch.enable_grad():loss,_=objective(i);(loss/len(train_used)).backward()
   total+=float(loss)/len(train_used);cams[i].load2device('cpu')
  assert material.grad is not None and torch.isfinite(material.grad).all()
  opt.step();history.append(dict(epoch=epoch,before=total));print(json.dumps(history[-1]),flush=True)
 final=evaluate();albedo=material.sigmoid().detach();changed=(albedo-.5).abs().max(-1).values>0
 selected_viz=[train_used[k] for k in np.linspace(0,len(train_used)-1,3,dtype=int)]+[held_used[len(held_used)//2]]
 for i in selected_viz:
  loss,pred=objective(i);save_image(out/f'{i:03d}_prediction.png',pred);save_image(out/f'{i:03d}_target.png',cached[i]['target']);save_image(out/f'{i:03d}_mask.png',cached[i]['mask'].expand(3,-1,-1).float());cams[i].load2device('cpu')
 inputs.update({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in extra})
 torch.save(dict(protocol='world_shared_material_v1',albedo=albedo.cpu(),active_ids=active.cpu(),args=args,directions=directions.cpu(),energy=energy.cpu(),train_frames=train_used,held_frames=held_used,geometry_updates=0,material_updates=12),out/'checkpoint.pt')
 torch.save({i:{k:({kk:(vv.cpu() if isinstance(vv,torch.Tensor) else vv) for kk,vv in v.items()} if k=='ds' else v.cpu()) for k,v in q.items()} for i,q in cached.items()},out/'frame_cache.pt')
 result=dict(gt_read=False,geometry_updates=0,material_updates=12,initial=initial,final=final,history=history,support=support,train_frames=train_used,held_frames=held_used,visualized_frames=selected_viz,changed_material_points=int(changed.sum()),material_min=float(albedo.min()),material_max=float(albedo.max()),inputs=inputs,native=sys.modules['diff_surfel_rasterization'].provenance,scope='冻结几何和历史未知灯光的有界共享材质拟合；保留材质观测，非未见光照泛化；无GT')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ['initial','final','changed_material_points','material_min','material_max']}),flush=True)
if __name__=='__main__':main()
