"""固定几何和形变主干，以有限面片一致性受限优化位移头；无GT。"""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from utils.general_utils import build_rotation
from scripts.probe_dynamic_dl_selfcal import read_config,save_image,digest_tensor

def axis(a,b):return torch.rad2deg(torch.acos((F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum(-1).abs().clamp(0,1)))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--trust-metric',choices=['parameter','actual'],default='parameter');p.add_argument('--center-weight',type=float,default=1.);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
 rt=json.loads((run/'runtime.json').read_text());args=rt['args'];assert not args['gt_direction_reference'];sources={}
 for f in [Path(args['model'])/'cfg_args',Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply',Path(args['model'])/'deform/iteration_35000/deform.pth',Path(args['source'])/'transforms_train.json',Path(args['cache'])]:
  sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();assert sources[str(f)]==rt['inputs'][str(f)]
 cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
 pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim);pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
 deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color);assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
 cache=torch.load(args['cache'],map_location='cpu',weights_only=False);ids=cache['ids'].cuda();train=rt['selected'];assert train==[i for i in range(135) if i%9!=4];assert torch.equal(cache['canonical_xyz'],pc.get_xyz[ids].detach().cpu())
 xyz=pc.get_xyz[ids].detach().clone().requires_grad_(True);gate=pc.get_binary_feature()[ids].detach();basis=build_rotation(pc.get_rotation)[ids].detach();pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.);bg=torch.zeros(3,device='cuda')
 infos=readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False);cameras=[loadCam(cfg,i,v,1.) for i,v in enumerate(infos)]
 support=cache['valid'][train].sum(0)>=10;lum=cache['rgb'].mean(-1);ref=torch.nanquantile(torch.where(cache['valid'][train],lum[train],torch.nan),.9,dim=0);bright=(lum/ref.clamp_min(.05)[None]>=.65)&torch.isfinite(ref)[None]&(ref[None]>.05)
 reference=torch.load(Path(a.reference)/'measurements.pt',map_location='cpu',weights_only=False);assert reference['train']==train and torch.equal(reference['ids'],ids.cpu())
 # 仅位移线性头更新，canonical几何和其他形变权重保持。
 initial={k:v.detach().clone() for k,v in deform.deform.state_dict().items()};head=deform.deform.gaussian_warp
 initial_head=torch.cat([head.weight.detach(),head.bias.detach()[:,None]],dim=1);theta=initial_head.clone().requires_grad_(True)
 scales=pc.get_scaling[ids].detach();records=[];original_world={};hidden_bound=0.;baseline_images={};preview=[4,67,130];full_hidden={}
 def render_preview(i,label):
  c=cameras[i];c.load2device();ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center);ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']};p=render(c,pc,pipe,bg,**ds);folder=out/label/f'frame_{i:03d}';folder.mkdir(parents=True)
  save_image(folder/'rgb.png',p['render']);save_image(folder/'alpha.png',p['rend_alpha']);save_image(folder/'normal.png',(F.normalize(p['rend_normal'],dim=0)*.5+.5)*p['rend_alpha']);g=pc.get_binary_feature();sep=render(c,pc,pipe,bg,**ds,override_color=torch.cat([g,1-g,torch.zeros_like(g)],-1));save_image(folder/'separation.png',sep['render']);c.load2device('cpu');return p['render'].detach().cpu()
 with torch.no_grad():
  for i,c in enumerate(cameras):
   c.load2device();full=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center);original_world[i]=(pc.get_xyz+full['d_xyz']).detach().cpu();hidden_bound=max(hidden_bound,float((torch.sqrt(full['hidden'].square().sum(-1)+1)*pc.get_binary_feature().flatten()).max()))
   if a.trust_metric=='actual':full_hidden[i]=full['hidden'].detach()
   if i in train:
    j=train.index(i);mask=(reference['arrays']['chosen'][j]&reference['dynamic']).cuda();xx=xyz[mask];gg=gate[mask];ss=scales[mask];bb=basis[mask];hs=[];chords=[];ti=c.fid.reshape(1,1).expand(len(xx),1)
    def hidden(x):return deform.step(x,ti,feature=gg,camera_center=c.camera_center)['hidden']
    center=torch.cat([hidden(xx),torch.ones((len(xx),1),device='cuda')],-1)*gg
    for k in range(2):
     offset=ss[:,k,None]*bb[:,:,k];hs.append((hidden(xx+offset)-hidden(xx-offset))*gg);chords.append(2*offset)
    records.append(dict(index=i,hdiff=torch.stack(hs,dim=1),chord=torch.stack(chords,dim=1),center=center,target=cache['normals'][i].cuda()[mask]))
   c.load2device('cpu')
   if i%30==0:print(json.dumps(dict(cached_frame=i)),flush=True)
  for i in preview:baseline_images[i]=render_preview(i,'initial')
 h=torch.cat([v['hdiff'] for v in records]);chord=torch.cat([v['chord'] for v in records]);center=torch.cat([v['center'] for v in records]);target=torch.cat([v['target'] for v in records]);frame_ids=torch.cat([torch.full((len(v['target']),),v['index']) for v in records])
 def terms(param,hh=h,cc=chord,cen=center,tar=target):
  edges=cc+hh@param[:,:-1].T;cross=torch.cross(edges[:,0],edges[:,1],dim=-1);length=edges.norm(dim=-1);sine=cross.norm(dim=-1)/(length[:,0]*length[:,1]).clamp_min(1e-20);normal=F.normalize(cross,dim=-1,eps=1e-12);alignment=1-(normal*tar).sum(-1).square();drift=cen@(param-initial_head.to(param)).T
  return alignment,sine,length,drift
 with torch.no_grad():
  v=terms(theta);valid=(v[2].min(-1).values>1e-7)&(v[1]>1e-4);assert valid.any();coverage=dict(total=len(valid),retained=int(valid.sum()),excluded=int((~valid).sum()))
 def objective(param):
  align,sine,length,drift=terms(param);return align[valid].mean()+a.center_weight*(drift.square().sum(-1).mean()/1e-8),align,sine,length
 # 仅缓存线性头目标的双精度方向差分，不涉及native渲染梯度。
 th=initial_head.double().requires_grad_(True);hd=h[:64].double();cd=chord[:64].double();ce=center[:64].double();ta=target[:64].double()
 def fd_loss(p):
  aa,_,_,dd=terms(p,hd,cd,ce,ta);return aa.mean()+a.center_weight*dd.square().sum(-1).mean()/1e-8
 torch.manual_seed(1015);direction=torch.randn_like(th);direction/=direction.norm();grad=torch.autograd.grad(fd_loss(th),th)[0];analytic=float((grad*direction).sum());eps=1e-6;numerical=float((fd_loss(th+eps*direction)-fd_loss(th-eps*direction))/(2*eps));fd_error=abs(analytic-numerical)/max(abs(analytic),abs(numerical),1e-12);assert fd_error<.01
 def summary():
  with torch.no_grad():
   aa,si,le,dr=terms(theta);angle=torch.rad2deg(torch.asin(aa.clamp(0,1).sqrt()));return dict(objective=float(objective(theta)[0]),alignment=float(aa[valid].mean()),axis_mean=float(angle[valid].mean()),center_rms=float(dr.square().sum(-1).mean().sqrt()),valid_count=int(valid.sum()))
 @torch.no_grad()
 def actual_bound(param,reference=None):
  largest=0.
  for hh in full_hidden.values():
   pp=F.linear(hh,param[:,:-1],param[:,-1])*pc.get_binary_feature()
   if reference is not None:
    previous=pc.get_xyz+F.linear(hh,reference[:,:-1],reference[:,-1])*pc.get_binary_feature();pp=pc.get_xyz+pp-previous
   largest=max(largest,float(pp.norm(dim=-1).max()))
  return largest
 before=summary();history=[]
 for step in range(20):
  loss=objective(theta)[0];g=torch.autograd.grad(loss,theta)[0];assert torch.isfinite(g).all();start=theta.detach().clone()
  direction_bound=actual_bound(g.detach()) if a.trust_metric=='actual' else hidden_bound*float(g.norm())
  proposal=start-g.detach()*(.99e-4/max(direction_bound,1e-30));difference=proposal-initial_head
  total_bound=actual_bound(proposal,initial_head) if a.trust_metric=='actual' else hidden_bound*float(difference.norm())
  proposal=initial_head+difference*min(1.,.99e-3/max(total_bound,1e-30));attempts=[]
  with torch.no_grad():
   for k in range(12):
    theta.copy_(torch.lerp(start,proposal,2.**(-k)));trial,aa,si,le=objective(theta);area_ok=bool(((si[valid]>1e-4)&(le[valid].min(-1).values>1e-7)).all())
    step_actual=actual_bound(theta,start) if a.trust_metric=='actual' else None;total_actual=actual_bound(theta,initial_head) if a.trust_metric=='actual' else None
    safe=a.trust_metric!='actual' or (step_actual<=1.0002e-4 and total_actual<=.00100002)
    ok=safe and area_ok and bool(torch.isfinite(trial)) and float(trial)<float(loss)-1e-10;attempts.append(dict(fraction=2.**(-k),loss=float(trial),area_valid=area_ok,accepted=ok,world_step=step_actual,world_total=total_actual))
    if ok:break
   if not ok:theta.copy_(start)
  history.append(dict(step=step+1,accepted=ok,attempts=attempts));print(json.dumps(history[-1]),flush=True)
 after=summary()
 with torch.no_grad():
  head.weight.copy_(theta[:,:-1]);head.bias.copy_(theta[:,-1]);torch.save(dict(deform_state_dict={k:v.cpu() for k,v in deform.deform.state_dict().items()},args=args,source_run=str(run),geometry={k:getattr(pc,k).detach().cpu() for k in ['_xyz','_rotation','_scaling','_opacity']}),out/'checkpoint.pt');loaded=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);deform.deform.load_state_dict(loaded['deform_state_dict']);world={};render_delta={}
  for i,c in enumerate(cameras):
   c.load2device();ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center);world[str(i)]=float((pc.get_xyz+ds['d_xyz']-original_world[i].cuda()).norm(dim=-1).max());c.load2device('cpu')
  for i in preview:
   image=render_preview(i,'final');render_delta[str(i)]=dict(mean_abs=float((image-baseline_images[i]).abs().mean()),max_abs=float((image-baseline_images[i]).abs().max()))
 unchanged=all(torch.equal(v,deform.deform.state_dict()[k]) for k,v in initial.items() if not k.startswith('gaussian_warp.'));accepted=sum(v['accepted'] for v in history);passed=unchanged and accepted>0 and max(world.values())<=.00101 and after['objective']<before['objective'];result=dict(status='PASS' if passed else 'FAILED',gt_read=False,experiment_args=vars(a),scope='缓存线性位移头有限面片一致性与轨迹保持；非几何质量验收，无光度训练',coverage=coverage,fd=dict(analytic=analytic,numerical=numerical,relative_error=fd_error),before=before,after=after,accepted=accepted,proposals=20,history=history,world=world,render_delta=render_delta,non_head_unchanged=unchanged,peak_memory_bytes=torch.cuda.max_memory_allocated(),hidden_bound=hidden_bound,head_delta=float((theta-initial_head).norm()),source_hashes=sources)
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));torch.save(dict(frame_ids=frame_ids,valid=valid.cpu()),out/'support.pt');print(json.dumps({k:v for k,v in result.items() if k not in ['world','history','source_hashes']},ensure_ascii=False),flush=True);assert passed
if __name__=='__main__':main()
