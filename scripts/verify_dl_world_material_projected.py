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
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--fit',required=True);p.add_argument('--preconditioner',choices=['t49','none'],default='t49');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4)
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


 fit=Path(a.fit)
 ck=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False);cache=torch.load(ck['cache_source'],map_location='cpu',weights_only=False);record=json.loads((fit/'result.json').read_text())
 for f,h in record['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 assert ck['geometry_updates']==0 and ck['material_updates']==120 and ck['protocol']=='world_shared_material_projected_v1'
 assert not set(ck['train_frames'])&set(ck['held_frames'])
 assert torch.equal(active.cpu(),ck['active_ids']);assert torch.all(ck['albedo']>=0)
 if ck['constraint']=='bounded':assert torch.all(ck['albedo']<=1)
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda');values={}
 for i,q in cache.items():
  c=cams[i];c.load2device();ds={k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in q['ds'].items()}
  # 独立重算形变，避免缓存自身掩盖几何错误。
  real=deform.step(xyz0,c.fid.reshape(1,1).expand(len(xyz0),1),feature=features,camera_center=c.camera_center)
  for k,v in ds.items():
   if isinstance(v,torch.Tensor):assert torch.equal(v,real[k])
   else:assert v==real[k]
  pred=render(c,pc,pipe,bg,**ds,override_color=ck['albedo'].cuda()*ck['energy'][i].cuda())['render']*q['shading'].cuda()
  value=float(((pred-q['target'].cuda()).double().square()*q['mask'].cuda()).sum()/(3*q['mask'].sum()))
  assert abs(value-record['final']['per_frame'][str(i)])<1e-9
  values[i]=value;c.load2device('cpu')
 grad=torch.load(fit/'gradient.pt',weights_only=False)['final_gradient'];material=ck['albedo'];proj=(material-grad).clamp(0,1) if ck['constraint']=='bounded' else (material-grad).clamp_min(0)
 pg=float((material-proj).abs().max());assert abs(pg-record['final_projected_gradient_inf'])<1e-8
 result=dict(projected_gradient_inf=pg,status='PASS',frames=len(values),geometry_cache_exact=True,per_frame=values,train=float(np.mean([values[i] for i in ck['train_frames']])),held=float(np.mean([values[i] for i in ck['held_frames']])),checkpoint_sha256=hashlib.sha256((fit/'checkpoint.pt').read_bytes()).hexdigest())
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
