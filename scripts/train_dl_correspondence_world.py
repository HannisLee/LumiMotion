"""固定对应驱动的跨时间共享世界平移残差；冻结canonical和形变，无GT。"""
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
 initial_loss=loss(delta);initial_metrics=metrics(delta);history=[];fd=[];previous=initial_world;accepted=0;max_total=0.
 for step in range(3):
  gradient=torch.zeros_like(delta);objective=0.;variable=delta.detach().requires_grad_(True)
  for e in edges:
   if not e['train']:continue
   with torch.enable_grad():v=edge_loss(e,variable);g,=torch.autograd.grad(v,variable)
   gradient+=g;objective+=float(v);cams[e['a']].load2device('cpu');cams[e['b']].load2device('cpu')
  assert torch.isfinite(gradient).all();direction=-gradient/(1+metric) if a.preconditioner=='t49' else -gradient;direction/=direction.norm(dim=-1).max().clamp_min(1e-20)
  if step==0:
   analytic=float((gradient*direction).sum())
   for eps in [1e-4,1e-5,1e-6]:
    numeric=(loss(delta+eps*direction)-loss(delta-eps*direction))/(2*eps);fd.append(dict(eps=eps,analytic=analytic,numeric=numeric,relative_error=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-12)))
  trials=[]
  for k in range(12):
   amount=1e-5*.5**k;candidate=delta+amount*direction;value=loss(candidate);worlds,total,movement=world_scan(candidate,previous);ok=value<objective-1e-12 and total<=.00100002 and movement<=.00010002;trials.append(dict(amount=amount,loss=value,world_total=total,world_step=movement,accepted=ok))
   if ok:delta=candidate;previous=worlds;max_total=total;accepted+=1;break
  history.append(dict(step=step,before=objective,gradient_norm=float(gradient.norm()),trials=trials));print(json.dumps(history[-1]),flush=True)
  if not ok:break
 for f in extra:inputs[str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest()
 # 无对应锚定目标在起点梯度严格为零；三次零更新控制。
 control=xyz0[active].clone().requires_grad_(True);control_history=[]
 for _ in range(3):
  with torch.enable_grad():v=(control-xyz0[active]).square().mean();g,=torch.autograd.grad(v,control)
  assert torch.count_nonzero(g)==0;control-=1e-5*g;control_history.append(float(v))
 assert torch.equal(control,xyz0[active])
 for name,change,num in [('control',torch.zeros_like(delta),0),('train',delta,accepted)]:
  folder=out/name;folder.mkdir();xyz=xyz0;torch.save(dict(world_residual=change.cpu(),protocol='correspondence_world_shared_v1',preconditioner=a.preconditioner,geometry={'_xyz':xyz.cpu()},deform_state_dict={k:v.cpu() for k,v in deform.deform.state_dict().items()},args=args,active_ids=active.cpu(),delta=change.cpu(),geometry_updates=num,material_updates=0),folder/'checkpoint.pt');(folder/'runtime.json').write_text(json.dumps(dict(inputs=inputs,gt_read=False,protocol='correspondence_world_shared_v1'),indent=2))
 final_metrics=metrics(delta);result=dict(preconditioner=a.preconditioner,gt_read=False,protocol='correspondence_world_shared_v1',geometry_updates=accepted,active_points=len(active),training_edges=sum(e['train'] for e in edges),held_edges=sum(not e['train'] for e in edges),initial_loss=initial_loss,final_loss=loss(delta),initial_metrics=initial_metrics,final_metrics=final_metrics,max_world_displacement=max_total,history=history,finite_difference=fd,control_history=control_history,scope='跨时间共享世界残差对应机制探针；无光度优化，不是新增DL几何收益；未离线几何验收');(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
