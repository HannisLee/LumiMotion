"""固定对应的跨边canonical梯度兼容性诊断；无GT，无参数更新。"""
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
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4)
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
  camera=cams[i];camera.load2device();xyz=xyz0.index_copy(0,active,xyz0[active]+change);ds=deform.step(xyz,camera.fid.reshape(1,1).expand(len(xyz),1),feature=features,camera_center=camera.camera_center);world=xyz+ds['d_xyz'];clip=torch.cat([world,torch.ones_like(world[:,:1])],-1)@camera.full_proj_transform;uv=(clip[:,:2]/clip[:,3:].clamp_min(1e-8)+1)*200-.5
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
 baseline_run=Path('output/smoke_test/1015-69-LH-dynamic-DL-T77对应几何更新/run');ck=torch.load(baseline_run/'train/checkpoint.pt',map_location='cpu',weights_only=False);assert torch.equal(active.cpu(),ck['active_ids']);actual=ck['delta'].cuda();assert torch.equal(xyz0.index_copy(0,active,xyz0[active]+actual).cpu(),ck['geometry']['_xyz'])
 assert all(torch.equal(v.cpu(),ck['deform_state_dict'][k]) for k,v in deform.deform.state_dict().items())
 rows=[];gradients=[];initial_total=0.;final_total=0.;records=[]
 for j,e in enumerate(edges):
  variable=delta.detach().requires_grad_(True)
  with torch.enable_grad():value=edge_loss(e,variable);g,=torch.autograd.grad(value,variable)
  final_value=float(edge_loss(e,actual));gradients.append(g.cpu());initial=float(value)
  if e['train']:initial_total+=initial;final_total+=final_value
  _,ua=forward(e['a'],delta);_,ub=forward(e['b'],delta);_,fa=forward(e['a'],actual);_,fb=forward(e['b'],actual)
  residual0=(ub[e['ids']]-ua[e['ids']])-e['flow'];residual1=(fb[e['ids']]-fa[e['ids']])-e['flow']
  for k,gid in enumerate(e['ids'].cpu().tolist()):records.append(dict(edge=j,a=e['a'],b=e['b'],gid=gid,train=e['train'],active=bool(torch.isin(e['ids'][k:k+1],active).item()),before=float(residual0[k].norm()),after=float(residual1[k].norm())))
  row=dict(edge=j,a=e['a'],b=e['b'],train=e['train'],count=len(e['ids']),initial_loss=initial,final_loss=final_value,linear_change=float((g*actual).sum()),actual_change=final_value-initial);rows.append(row);print(json.dumps(row),flush=True);cams[e['a']].load2device('cpu');cams[e['b']].load2device('cpu')
 original=json.loads((baseline_run/'result.json').read_text());assert abs(initial_total-original['initial_loss'])<1e-10;assert abs(final_total-original['final_loss'])<1e-10
 G=torch.stack(gradients).double();white=G/(1+metric.cpu().double()).sqrt()[None];train_edge=torch.tensor([e['train'] for e in edges]);pair_rows=[]
 for k,gid in enumerate(active.cpu().tolist()):
  relevant=torch.where(white[:,k].norm(dim=-1)>1e-12)[0].tolist()
  for q,i in enumerate(relevant):
   for j in relevant[q+1:]:
    kind='train_train' if train_edge[i] and train_edge[j] else 'held_held' if not train_edge[i] and not train_edge[j] else 'train_held';cos=float(torch.dot(white[i,k],white[j,k])/(white[i,k].norm()*white[j,k].norm()));pair_rows.append(dict(gid=gid,edge_i=i,edge_j=j,kind=kind,cosine=cos))
 summary={}
 for kind in ['train_train','train_held','held_held']:
  rr=[r for r in pair_rows if r['kind']==kind];v=np.array([r['cosine'] for r in rr]);summary[kind]=dict(edge_pairs=len(rr),identities=len({r['gid'] for r in rr}),negative=int((v<0).sum()),median=float(np.median(v)) if len(v) else None)
 groups={}
 for kind in ['train','held_active','held_inactive']:
  rr=[r for r in records if r['train']] if kind=='train' else [r for r in records if not r['train'] and r['active']==(kind=='held_active')]
  groups[kind]=dict(observations=len(rr),identities=len({r['gid'] for r in rr}),before=float(np.mean([r['before'] for r in rr])) if rr else None,after=float(np.mean([r['after'] for r in rr])) if rr else None,increased=sum(r['after']>r['before'] for r in rr),decreased=sum(r['after']<r['before'] for r in rr))
 # 合并梯度的一阶冲突，区别于逐边余弦。
 wt=white[train_edge].sum(0);wh=white[~train_edge].sum(0);joint=(wt*wh).sum(-1);supported=(wt.norm(dim=-1)>1e-12)&(wh.norm(dim=-1)>1e-12)
 overlap=dict(identities=int(supported.sum()),negative=int((joint[supported]<0).sum()),train_held_dot=float(joint.sum()),held_actual_change=sum(r['actual_change'] for r in rows if not r['train']),held_linear_change=sum(r['linear_change'] for r in rows if not r['train']))
 extra.extend([baseline_run/'result.json',baseline_run/'train/checkpoint.pt']);inputs.update({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in extra})
 torch.save(dict(active_ids=active.cpu(),edge_gradients=G,whitened_gradients=white,actual_delta=actual.cpu()),out/'gradients.pt');(out/'observations.json').write_text(json.dumps(records,indent=2));(out/'gradient_pairs.json').write_text(json.dumps(pair_rows,indent=2));result=dict(gt_read=False,geometry_updates=0,initial_loss_reproduction=initial_total,final_loss_reproduction=final_total,pairwise=summary,groups=groups,overlap=overlap,edges=rows,inputs=inputs,scope='局部目标兼容性与既有T77更新重放；非对应真值/非解冻形变收益证明');(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(dict(pairwise=summary,groups=groups,overlap=overlap)),flush=True)
if __name__=='__main__':main()
