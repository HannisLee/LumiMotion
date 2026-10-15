"""默认形变切平面、旋转头及渲染深度法线一致性；只用训练资产，零更新。"""
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
from scripts.probe_dynamic_dl_selfcal import read_config,save_image

def axis(a,b):return torch.rad2deg(torch.acos((F.normalize(a,dim=-1)*F.normalize(b,dim=-1)).sum(-1).abs().clamp(0,1)))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
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
 arrays={k:[] for k in ['head_tangent','head_depth','tangent_depth','condition','det','tangent_sine','valid','chosen','depth_valid']};repro=[]
 for i in train:
  c=cameras[i];c.load2device();ds=deform.step(xyz,c.fid.reshape(1,1).expand(len(ids),1),feature=gate,camera_center=c.camera_center);world=xyz+ds['d_xyz'];J=torch.stack([torch.autograd.grad(world[:,k].sum(),xyz,retain_graph=k<2)[0].detach() for k in range(3)],dim=1)
  with torch.no_grad():
   tangent1=(J@basis[:,:,0,None]).squeeze(-1);tangent2=(J@basis[:,:,1,None]).squeeze(-1);cross=torch.cross(tangent1,tangent2,dim=-1);sine=cross.norm(dim=-1)/(tangent1.norm(dim=-1)*tangent2.norm(dim=-1)).clamp_min(1e-20);normal=F.normalize(cross,dim=-1)
   all_ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center);all_ds={k:all_ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']};head=build_rotation(pc.get_rotation_bias(all_ds['d_rotation']))[ids,:,2];repro.append(float(axis(head,cache['normals'][i].cuda()).max()))
   package=render(c,pc,pipe,bg,**all_ds,override_color=torch.full_like(pc.get_xyz,.5));u,v=cache['pixels'][i].T.cuda();depth=package['surf_normal'][:,v,u].T;depth_valid=torch.isfinite(depth).all(-1)&(depth.norm(dim=-1)>1e-6)
   alpha=c.gt_alpha_mask.to('cuda') if c.gt_alpha_mask is not None else torch.ones_like(package['rend_alpha']);valid_image=(alpha>.99)&(package['rend_alpha']>.95)&torch.isfinite(package['surf_depth'])&(package['surf_depth']>0);valid_image=-F.max_pool2d(-valid_image.float()[None],3,stride=1,padding=1)[0]>.5
   chosen=cache['valid'][i].cuda()&support.cuda()&bright[i].cuda()&valid_image[0,v,u]
   sv=torch.linalg.svdvals(J.double());record=dict(head_tangent=axis(head,normal),head_depth=axis(head,depth),tangent_depth=axis(normal,depth),condition=(sv[:,0]/sv[:,-1].clamp_min(1e-12)),det=torch.linalg.det(J.double()),tangent_sine=sine,valid=cache['valid'][i],chosen=chosen,depth_valid=depth_valid)
   for k,x in record.items():arrays[k].append(x.cpu())
   if i in [0,64,134]:
    folder=out/f'frame_{i:03d}';folder.mkdir();save_image(folder/'alpha.png',package['rend_alpha']);save_image(folder/'normal.png',(F.normalize(package['rend_normal'],dim=0)*.5+.5)*package['rend_alpha']);rgb=render(c,pc,pipe,bg,**all_ds);save_image(folder/'rgb.png',rgb['render']);g=pc.get_binary_feature();sep=render(c,pc,pipe,bg,**all_ds,override_color=torch.cat([g,1-g,torch.zeros_like(g)],-1));save_image(folder/'separation.png',sep['render'])
  c.load2device('cpu')
  if i%20==0:print(json.dumps(dict(frame=i)),flush=True)
 arrays={k:torch.stack(v) for k,v in arrays.items()};assert all(torch.isfinite(v).all() for v in arrays.values());assert max(repro)<.1
 stats={};dynamic=gate.flatten().cpu()>.5
 for region,mask in [('dynamic',dynamic),('static',~dynamic)]:
  for selection in ['all','valid','chosen']:
   m=mask[None].expand(len(train),-1)&(torch.ones_like(arrays['valid']) if selection=='all' else arrays[selection]);count=int(m.sum());row=dict(observations=count,points=int(m.any(0).sum()),negative_det_fraction=float((arrays['det'][m]<0).double().mean()),condition_median=float(arrays['condition'][m].median()),condition_above100_fraction=float((arrays['condition'][m]>100).double().mean()),tangent_sine_below001_fraction=float((arrays['tangent_sine'][m]<.01).double().mean()))
   for k in ['head_tangent','head_depth','tangent_depth']:
    used=m if k=='head_tangent' else m&arrays['depth_valid'];values=arrays[k][used];row[k]=dict(count=len(values),mean=float(values.mean()),median=float(values.median()),p90=float(torch.quantile(values,.9)))
   stats[region+'_'+selection]=row
 torch.save(dict(train=train,ids=ids.cpu(),dynamic=dynamic,arrays=arrays),out/'measurements.pt');result=dict(status='PASS',gt_read=False,updates=0,source_hashes=sources,frames=len(train),sampled_points=len(ids),cached_head_max_axis_deg=max(repro),statistics=stats,scope='默认起点采样点局部微分与固定像素深度比较；非GT准确率，cache valid含旋转头一致性预筛选，不能视为无偏物质对应')
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
