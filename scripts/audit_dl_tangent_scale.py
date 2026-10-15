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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
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
 scales=pc.get_scaling[ids].detach();results={};raw={}
 for factor in [.1,1.]:
  per_frame=[];normal_errors=[]
  with torch.no_grad():
   for i in train:
    c=cameras[i];c.load2device();time=c.fid.reshape(1,1).expand(len(ids),1)
    def world(x):return x+deform.step(x,time,feature=gate,camera_center=c.camera_center)['d_xyz']
    chords=[]
    for k in range(2):
     offset=factor*scales[:,k,None]*basis[:,:,k];chords.append(world(xyz+offset)-world(xyz-offset))
    tangent=F.normalize(torch.cross(chords[0],chords[1],dim=-1),dim=-1);ds=deform.step(xyz,time,feature=gate,camera_center=c.camera_center);head=build_rotation(F.normalize(pc._rotation[ids]+ds['d_rotation'],dim=-1))[:,:,2]
    normal_errors.append(float(axis(head,cache['normals'][i].cuda()).max()));per_frame.append(axis(head,tangent).cpu());c.load2device('cpu')
   values=torch.stack(per_frame);assert torch.isfinite(values).all() and max(normal_errors)<.1;raw[str(factor)]=values
   for region,region_mask in [('dynamic',reference['dynamic']),('static',~reference['dynamic'])]:
    mask=reference['arrays']['chosen']&region_mask[None];v=values[mask];results[str(factor)+'_'+region]=dict(observations=len(v),mean=float(v.mean()),median=float(v.median()),p90=float(torch.quantile(v,.9)),max_cached_head_axis_error=max(normal_errors))
 torch.save(raw,out/'angles.pt');data=dict(status='PASS',updates=0,gt_read=False,frames=len(train),statistics=results,scope='固定point feature和默认canonical切向量，在0.1与1倍Gaussian切向标准差上做中心弦；非真实对应或GT法线')
 (out/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print(json.dumps(data,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
