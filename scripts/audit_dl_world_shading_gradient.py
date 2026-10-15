"""世界共享残差的深度法线光度梯度诊断；无GT，无更新。"""
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
 extra.extend([lp,ep]);pool=sorted({e['a'] for e in edges if e['train']});selected=[pool[k] for k in np.linspace(0,len(pool)-1,3,dtype=int)]
 pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda');rows=[];grads=[]
 for i in selected:
  camera=cams[i];camera.load2device();base_ds=deform.step(xyz0,camera.fid.reshape(1,1).expand(len(xyz0),1),feature=features,camera_center=camera.camera_center);base_ds={k:(base_ds[k].detach() if isinstance(base_ds[k],torch.Tensor) else base_ds[k]) for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
  def package(change,colors):
   ds=dict(base_ds);ds['d_xyz']=ds['d_xyz'].index_add(0,active,change)
   return render(camera,pc,pipe,bg,**ds,override_color=colors)
  colors=torch.full_like(xyz0,.5)*energy[i]
  initial=package(delta,colors);n0=F.normalize(initial['surf_normal'],dim=0);sign=torch.where((n0*(camera.camera_center[:,None,None]-initial['surf_point'])).sum(0,keepdim=True)>=0,1.,-1.)
  shading0=((n0*sign)*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1
  identity_colors=torch.zeros_like(xyz0);identity_colors[active]=1;coverage=package(delta,identity_colors)['render'][:1]
  target=srgb_to_linear(camera.original_image_train_light.cuda())
  raw=np.asarray(Image.open(camera.image_path_train_light).convert('RGB'));sat=torch.tensor((raw>=254).any(-1).copy(),device='cuda').float()[None,None];sat=F.adaptive_max_pool2d(sat,target.shape[-2:]);sat=F.max_pool2d(sat,5,1,2)[0]>0
  valid=(camera.gt_alpha_mask.cuda()>.99)&(initial['rend_alpha']>.99)&(initial['surf_depth']>0)&torch.isfinite(initial['surf_depth'])
  valid=-F.max_pool2d(-valid.float()[None],3,1,1)[0]>.5;mask=valid&(~sat)&(coverage>.01);assert mask.sum()>0
  def objective(change,mode):
   z=package(change,colors);normal=F.normalize(z['surf_normal'],dim=0)*sign;shading=(normal*directions[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1
   if mode=='stop':shading=shading.detach()
   if mode=='frozen':shading=shading0
   pred=z['render']*shading
   return ((pred-target).double().square()*mask).sum()/(3*mask.sum()),z,pred
  gradient={}
  for mode in ['full','stop']:
   v=delta.clone().requires_grad_(True)
   with torch.enable_grad():
    loss,_,_=objective(v,mode);g,=torch.autograd.grad(loss,v)
   gradient[mode]=g;assert torch.isfinite(g).all()
  assert float(objective(delta,'full')[0])==float(objective(delta,'frozen')[0])
  fd=[]
  for mode in ['full','stop']:
   direction=-gradient[mode];direction=direction/direction.norm(dim=-1).max().clamp_min(1e-20);analytic=float((gradient[mode]*direction).sum())
   for eps in [1e-4,1e-5,1e-6]:
    forward_mode='full' if mode=='full' else 'frozen'
    plus=float(objective(eps*direction,forward_mode)[0]);minus=float(objective(-eps*direction,forward_mode)[0]);numeric=(plus-minus)/(2*eps)
    fd.append(dict(mode=mode,eps=eps,analytic=analytic,numeric=numeric,relative_error=abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-12),forward_plus=plus,forward_minus=minus))
  full=gradient['full'];stop=gradient['stop'];normal_part=full-stop
  loss,z,pred=objective(delta,'full');save_image(out/f'{i:03d}_mask.png',mask.expand(3,-1,-1).float());save_image(out/f'{i:03d}_rgb_linear.png',pred);save_image(out/f'{i:03d}_normal.png',(n0*sign+1)/2)
  row=dict(frame=i,mask_pixels=int(mask.sum()),initial_loss=float(loss),full_norm=float(full.norm()),stop_norm=float(stop.norm()),normal_path_norm=float(normal_part.norm()),full_stop_dot=float((full*stop).sum()),finite_difference=fd)
  rows.append(row);grads.append({k:v.cpu() for k,v in gradient.items()});print(json.dumps(row),flush=True);camera.load2device('cpu')
 inputs.update({str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in extra})
 torch.save(dict(active_ids=active.cpu(),gradients=grads),out/'gradients.pt')
 result=dict(gt_read=False,geometry_updates=0,selected_frames=selected,rows=rows,inputs=inputs,native=sys.modules['diff_surfel_rasterization'].provenance,scope='固定0.5共享材质及未知T05/T57灯光的梯度机制诊断，非已拟合材质/几何收益；无阴影或镜面建模')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
