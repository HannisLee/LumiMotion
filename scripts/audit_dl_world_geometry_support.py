"""固定共享材质和未知光照的世界残差光度/对应配对更新。"""
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
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--material-fit',default='output/smoke_test/1015-75-LH-dynamic-DL-T83共享材质/fit');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4)
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
 from gaussian_renderer import render
 from scripts.probe_dynamic_dl_selfcal import save_image
 fit=Path(a.material_fit)
 material=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False);cache_path=Path(material.get('cache_source',str(fit/'frame_cache.pt')));cache=torch.load(cache_path,map_location='cpu',weights_only=False);mr=json.loads((fit/'result.json').read_text())
 for f,h in mr['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 assert torch.equal(active.cpu(),material['active_ids'])
 extra.extend([fit/'checkpoint.pt',cache_path,fit/'result.json'])
 assert material['protocol'] in ['world_shared_material_v1','world_shared_material_projected_v1']
 for v in vars(pc).values():
  if isinstance(v,torch.Tensor):v.requires_grad_(False)
 albedo=material['albedo'].cuda();directions=material['directions'].cuda();energy=material['energy'].cuda();train_frames=material['train_frames'];held_frames=material['held_frames']
 cache={i:{k:({kk:(vv.cuda() if isinstance(vv,torch.Tensor) else vv) for kk,vv in v.items()} if k=='ds' else v.cuda()) for k,v in q.items()} for i,q in cache.items()}
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda')

 rows=[];all_gradients=[];samples={};training_support=torch.zeros(len(xyz0),device='cuda',dtype=torch.bool);marker=torch.zeros_like(xyz0);marker[active]=1
 fullzero=torch.zeros_like(xyz0)
 for i,q in cache.items():
  camera=cams[i];camera.load2device();marked=render(camera,pc,pipe,bg,**q['ds'],override_color=marker);mask=q['mask'][0]
  fraction=(marked['render'][0]/marked['rend_alpha'][0].clamp_min(1e-8))[mask]
  samples[i]=fraction.cpu();row=dict(frame=i,train=i in train_frames,pixels=len(fraction),active_mean=float(fraction.mean()),active_median=float(fraction.median()),below_half=int((fraction<.5).sum()),below_tenth=int((fraction<.1).sum()))
  if i in train_frames:
   variable=fullzero.clone().requires_grad_(True)
   with torch.enable_grad():
    ds=dict(q['ds']);ds['d_xyz']=ds['d_xyz']+variable;pack=render(camera,pc,pipe,bg,**ds,override_color=albedo*energy[i])
    normal=F.normalize(pack['surf_normal'],dim=0)*q['sign'];shading=(normal*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1;pred=pack['render']*shading
    loss=((pred-q['target']).double().square()*q['mask']).sum()/(3*q['mask'].sum());g,=torch.autograd.grad(loss,variable)
   assert torch.isfinite(g).all();assert abs(float(loss)-mr['final']['per_frame'][str(i)])<1e-9
   supported=g.norm(dim=-1)>1e-12;training_support|=supported;total=float(g.double().square().sum());active_energy=float(g[active].double().square().sum())
   row.update(loss=float(loss),gradient_squared_norm=total,active_gradient_squared_norm=active_energy,active_gradient_fraction=active_energy/total,nonzero_points=int(supported.sum()))
   all_gradients.append(g.cpu())
  rows.append(row);camera.load2device('cpu')
  if i in [2,62]:
   save_image(out/f'{i:03d}_active_fraction.png',(marked['render'][:1]/marked['rend_alpha'].clamp_min(1e-8)).expand(3,-1,-1));save_image(out/f'{i:03d}_mask.png',mask.expand(3,-1,-1).float())
  print(json.dumps(row),flush=True)
 G=torch.stack(all_gradients).double();aggregate=G.mean(0);squared=float(aggregate.square().sum());active_squared=float(aggregate[active.cpu()].square().sum())
 summary={}
 for kind in ['train','held']:
  rr=[r for r in rows if r['train']==(kind=='train')];values=torch.cat([samples[r['frame']] for r in rr]).double()
  summary[kind]=dict(frames=len(rr),pixels=len(values),mean=float(values.mean()),median=float(values.median()),p10=float(torch.quantile(values,.1)),p90=float(torch.quantile(values,.9)),below_half=float((values<.5).double().mean()),below_tenth=float((values<.1).double().mean()))
 summary['gradient']=dict(train_union=int(training_support.sum()),active_union=int(training_support[active].sum()),summed_frame_active_fraction=sum(r.get('active_gradient_squared_norm',0) for r in rows)/sum(r.get('gradient_squared_norm',0) for r in rows),aggregate_active_fraction=active_squared/squared)
 inputs.update({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in extra})
 torch.save(dict(active_ids=active.cpu(),frames=train_frames,gradients=G.float(),fractions=samples,train_support_ids=torch.where(training_support)[0].cpu()),out/'audit.pt')
 result=dict(gt_read=False,geometry_updates=0,material_fit=str(fit),rows=rows,summary=summary,inputs=inputs,scope='固定T87初始世界位置与掩码的贡献/真实光度梯度分配；非语义或对应真值；非扩大权限收益')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
