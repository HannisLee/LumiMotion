"""固定30步预算的线性时间残差配对；保留三步入口。"""
import argparse,hashlib,json,shlex,sys
from pathlib import Path
import numpy as np,torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config
from scripts.dl_linear_world_residual import linear_residual,PROTOCOL

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry-support');p.add_argument('--material-fit',default='output/smoke_test/1015-75-LH-dynamic-DL-T83共享材质/fit');p.add_argument('--group',choices=['photo','joint'],required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4)
 run=Path('output/smoke_test/1015-41-LH-dynamic-DL-T49全训练时刻预条件/preconditioned');rt=json.loads((run/'runtime.json').read_text());args=rt['args'];assert not args['gt_direction_reference'];inputs={}
 for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json']:
  sha=hashlib.sha256(f.read_bytes()).hexdigest();assert sha==rt['inputs'][str(f)];inputs[str(f.resolve())]=sha
 cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'));deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
 cams=[loadCam(cfg,i,c,1.) for i,c in enumerate(readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False))]
 base=Path('output/smoke_test/1015-68-LH-dynamic-DL-T76同源锚点');hr=Path('output/smoke_test/1015-67-LH-dynamic-DL-T75对应光度配对/fit/result.json');held=set(json.loads(hr.read_text())['held_material_frames']);meta=json.loads((base/'probe/result.json').read_text());edges=[];extra=[hr,base/'probe/result.json',base/'homography/result.json',run/'runtime.json',run/'xyz_metric.pt',Path(__file__),Path('scripts/dl_linear_world_residual.py')]
 for j,row in enumerate(meta['rows']):
  pp=base/'probe'/f'pair_{j:02d}.npz';hh=base/'homography'/f'residual_{j:02d}.npz';d=np.load(pp);h=np.load(hh);m=h['moving'];extra.extend([pp,hh])
  if not m.any():continue
  aa,bb=map(int,d['indices']);edges.append(dict(a=aa,b=bb,ids=torch.tensor(d['source_ids'][m],device='cuda'),flow=torch.tensor(d['pb'][m]-d['pa'][m],device='cuda'),train=aa not in held and bb not in held))
 assert sum(len(e['ids']) for e in edges)==449
 active=torch.unique(torch.cat([e['ids'] for e in edges if e['train']]));assert len(active)>0
 correspondence_active=active.clone()
 if a.geometry_support:
  support_path=Path(a.geometry_support);support=torch.load(support_path,map_location='cpu',weights_only=False)
  assert torch.equal(support['original_ids'],correspondence_active.cpu())
  active=support['ids'].cuda();assert active.ndim==1 and len(active.unique())==len(active) and torch.isin(correspondence_active,active).all()
  assert int(active.min())>=0 and int(active.max())<len(pc.get_xyz)
  extra.append(support_path)

 xyz0=pc.get_xyz.detach().clone();features=pc.get_binary_feature().detach();metric=torch.load(run/'xyz_metric.pt',map_location='cpu',weights_only=False)['diagonal'][active.cpu()].cuda();delta=torch.zeros((len(active),2,3),device='cuda')
 def forward(i,change):
  camera=cams[i];camera.load2device();xyz=xyz0;ds=deform.step(xyz,camera.fid.reshape(1,1).expand(len(xyz),1),feature=features,camera_center=camera.camera_center);world=xyz+ds['d_xyz'].index_add(0,active,linear_residual(change,camera.fid));clip=torch.cat([world,torch.ones_like(world[:,:1])],-1)@camera.full_proj_transform;uv=(clip[:,:2]/clip[:,3:].clamp_min(1e-8)+1)*200-.5
  return world,uv
 initial_world=[];initial_uv=[]
 for i in range(135):
  w,u=forward(i,delta);initial_world.append(w.detach());initial_uv.append(u.detach());cams[i].load2device('cpu')
 count=sum(len(e['ids']) for e in edges if e['train'])
 def edge_loss(e,change):
  _,ua=forward(e['a'],change);_,ub=forward(e['b'],change);ids=e['ids'];res=(ub[ids]-ua[ids])-e['flow'];h=torch.nn.functional.huber_loss(res,torch.zeros_like(res),reduction='sum',delta=1.)/(2*count);anchor=.5*(ua[ids]-initial_uv[e['a']][ids]).square().sum()/(2*count)
  return h+anchor
 def loss(change):
  v=0.
  for e in edges:
   if e['train']:v+=float(edge_loss(e,change))
   cams[e['a']].load2device('cpu');cams[e['b']].load2device('cpu')
  return v
 def world_scan(change,previous):
  worlds=[];total=[];step=[]
  for i in range(135):
   w,_=forward(i,change);worlds.append(w.detach());total.append(float((w-initial_world[i]).norm(dim=-1).max()));step.append(float((w-previous[i]).norm(dim=-1).max()));cams[i].load2device('cpu')
  return worlds,max(total),max(step)
 def metrics(change):
  values={'train':[],'held':[]}
  for e in edges:
   _,pa=forward(e['a'],change);_,pb=forward(e['b'],change);r=(pb[e['ids']]-pa[e['ids']])-e['flow'];values['train' if e['train'] else 'held'].extend(r.norm(dim=-1).cpu().tolist());cams[e['a']].load2device('cpu');cams[e['b']].load2device('cpu')
  return {k:dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v))) for k,v in values.items()}
 from types import SimpleNamespace
 import torch.nn.functional as F
 from gaussian_renderer import render
 from scripts.probe_dynamic_dl_selfcal import save_image
 fit=Path(a.material_fit)
 material=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False);cache_path=Path(material.get('cache_source',str(fit/'frame_cache.pt')));cache=torch.load(cache_path,map_location='cpu',weights_only=False);mr=json.loads((fit/'result.json').read_text())
 for f,h in mr['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 assert torch.equal(correspondence_active.cpu(),material['active_ids'])
 extra.extend([fit/'checkpoint.pt',cache_path,fit/'result.json'])
 assert material['protocol'] in ['world_shared_material_v1','world_shared_material_projected_v1']
 for v in vars(pc).values():
  if isinstance(v,torch.Tensor):v.requires_grad_(False)
 albedo=material['albedo'].cuda();directions=material['directions'].cuda();energy=material['energy'].cuda();train_frames=material['train_frames'];held_frames=material['held_frames']
 cache={i:{k:({kk:(vv.cuda() if isinstance(vv,torch.Tensor) else vv) for kk,vv in v.items()} if k=='ds' else v.cuda()) for k,v in q.items()} for i,q in cache.items()}
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda')
 def photo(i,change):
  camera=cams[i];camera.load2device();q=cache[i];ds=dict(q['ds']);ds['d_xyz']=ds['d_xyz'].index_add(0,active,linear_residual(change,camera.fid))
  package=render(camera,pc,pipe,bg,**ds,override_color=albedo*energy[i])
  normal=F.normalize(package['surf_normal'],dim=0)*q['sign'];shading=(normal*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1;pred=package['render']*shading
  value=((pred-q['target']).double().square()*q['mask']).sum()/(3*q['mask'].sum())
  return value,package,pred
 def photo_metrics(change):
  vals={i:float(photo(i,change)[0]) for i in cache}
  for i in cache:cams[i].load2device('cpu')
  return dict(train=float(np.mean([vals[i] for i in train_frames])),held=float(np.mean([vals[i] for i in held_frames])),per_frame=vals)
 def gradients(change):
  gp=torch.zeros_like(change);gc=torch.zeros_like(change);v=change.clone().requires_grad_(True);pv=0.;cv=0.
  for i in train_frames:
   with torch.enable_grad():l,_,_=photo(i,v);g,=torch.autograd.grad(l,v)
   gp+=g/len(train_frames);pv+=float(l)/len(train_frames);cams[i].load2device('cpu')
  for e in edges:
   if not e['train']:continue
   with torch.enable_grad():l=edge_loss(e,v);g,=torch.autograd.grad(l,v)
   gc+=g;cv+=float(l);cams[e['a']].load2device('cpu');cams[e['b']].load2device('cpu')
  assert torch.isfinite(gp).all() and torch.isfinite(gc).all()
  return gp,gc,pv,cv
 gp,gc,pv,cv=gradients(delta);weight=float(gp.norm()/gc.norm().clamp_min(1e-20));used_weight=weight if a.group=='joint' else 0.
 initial_photo=photo_metrics(delta);initial_correspondence=metrics(delta)
 assert abs(initial_photo['train']-mr['final']['train'])<1e-9
 def objective(change):
  pv=sum(float(photo(i,change)[0])/len(train_frames) for i in train_frames)
  for i in train_frames:cams[i].load2device('cpu')
  cv=loss(change)
  return pv+used_weight*cv,pv,cv
 history=[];previous=initial_world;accepted=0;max_total=0.
 for step in range(30):
  if step:gp,gc,pv,cv=gradients(delta)
  g=gp+used_weight*gc;scale=max(float(linear_residual(g,t).norm(dim=-1).max()) for t in [0.,1.]);assert scale>0 and np.isfinite(scale);direction=-g/scale;before=pv+used_weight*cv;trials=[]
  for k in range(12):
   amount=1e-5*.5**k;candidate=delta+amount*direction;value,pnew,cnew=objective(candidate);worlds,total,movement=world_scan(candidate,previous);ok=value<before-1e-12 and total<=.00100002 and movement<=.00010002
   trials.append(dict(amount=amount,objective=value,photo=pnew,correspondence=cnew,world_total=total,world_step=movement,accepted=ok))
   if ok:delta=candidate;previous=worlds;max_total=total;accepted+=1;break
  history.append(dict(step=step,before=before,photo=pv,correspondence=cv,trials=trials));print(json.dumps(history[-1]),flush=True)
  torch.save(dict(world_residual=delta.cpu(),geometry_updates=accepted,requested_steps=30,history=history),out/'progress.pt')
  if not ok:break
 for f in extra:inputs[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
 folder=out/a.group;folder.mkdir()
 torch.save(dict(protocol=PROTOCOL,time_basis=dict(center=.5,minimum=0.,maximum=1.,kind='constant_plus_linear'),world_residual=delta.cpu(),geometry={'_xyz':xyz0.cpu()},deform_state_dict={k:v.cpu() for k,v in deform.deform.state_dict().items()},args=args,active_ids=active.cpu(),delta=delta.cpu(),geometry_updates=accepted,requested_steps=30,material_updates=0,fixed_material_source=str((fit/'checkpoint.pt').resolve()),fixed_material_sha256=inputs[str((fit/'checkpoint.pt').resolve())],objective_group=a.group,correspondence_active_ids=correspondence_active.cpu()),folder/'checkpoint.pt')
 (folder/'runtime.json').write_text(json.dumps(dict(inputs=inputs,gt_read=False,protocol=PROTOCOL),indent=2))
 final_photo=photo_metrics(delta);final_correspondence=metrics(delta)
 viz=[train_frames[0],train_frames[len(train_frames)//2],train_frames[-1],held_frames[len(held_frames)//2]]
 for i in viz:
  l,pack,pred=photo(i,delta);save_image(out/f'{i:03d}_rgb.png',pred);save_image(out/f'{i:03d}_alpha.png',pack['rend_alpha'].expand(3,-1,-1));save_image(out/f'{i:03d}_normal.png',(F.normalize(pack['surf_normal'],dim=0)*cache[i]['sign']+1)/2)
  ds=dict(cache[i]['ds']);ds['d_xyz']=ds['d_xyz'].index_add(0,active,linear_residual(delta,cams[i].fid));gate=render(cams[i],pc,pipe,bg,**ds,override_color=features.expand(-1,3))['render'];save_image(out/f'{i:03d}_dynamic_gate.png',gate);cams[i].load2device('cpu')
 result=dict(group=a.group,time_basis='offset+(t-0.5)*velocity',max_velocity=float(delta[:,1].norm(dim=-1).max()),active_points=len(active),correspondence_active_points=len(correspondence_active),gt_read=False,geometry_updates=accepted,requested_steps=30,material_updates=0,calibrated_weight=weight,used_weight=used_weight,initial_photo=initial_photo,final_photo=final_photo,initial_correspondence=initial_correspondence,final_correspondence=final_correspondence,history=history,max_world_displacement=max_total,visualized_frames=viz,scope='固定未知灯光与共同材质的光度/对应几何配对；不是灯光联合优化或完整多种子A/B/C')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
