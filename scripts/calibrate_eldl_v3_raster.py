"""冻结无GT几何的生产全图光度校准；不导出光照，不更新几何。"""
from __future__ import annotations
import argparse
import math
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v


def load(a):
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from scripts.probe_dynamic_dl_selfcal import read_config
    v.install_guard(a)
    data,state,summary,result,hashes=v.load_inputs(a)
    cfg=read_config(a.model/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply'))
    for value in vars(pc).values():
        if isinstance(value,torch.Tensor):value.requires_grad_(False)
    weights=torch.load(a.model/f'deform/iteration_{a.iteration}/deform.pth',map_location='cpu',weights_only=True)
    for head in ('gaussian_warp','gaussian_rotation','gaussian_scaling'):
        for suffix in ('weight','bias'):assert torch.count_nonzero(weights[f'{head}.{suffix}'])==0
    cameras=[]
    for i,info in enumerate(readCamerasFromTransforms(str(a.source),'transforms_train.json','images',False)):
        camera=loadCam(cfg,i,info,1.);camera.load2device()
        frame=int(info.image_name_train_light.split('_')[-1]);assert frame==v.TRAIN[i]
        cameras.append(camera)
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    return pc,cameras,pipe,data,state,hashes


def render(camera,pc,pipe,colors):
    from gaussian_renderer import render as native
    return native(camera,pc,pipe,torch.zeros(3,device='cuda'),0.,0.,0.,override_color=colors)


def objective(pred,camera):
    from scene.stage2_photometric import srgb_to_linear
    target=srgb_to_linear(camera.original_image_train_light.cuda())
    mask=(camera.gt_alpha_mask.cuda()>.5).expand_as(target)
    saturated=camera.observed_saturation_mask.cuda().bool()
    residual=torch.where(saturated,(pred-target).clamp_max(0),pred-target)
    return residual[mask].square().mean()


@torch.no_grad()
def cache(a):
    from scene.static_dl_photometric import StaticVisibility
    from scene.eldl_v3_photometric import render_photometric
    from scene.stage2_photometric import linear_to_srgb
    from utils.general_utils import build_rotation
    pc,cameras,pipe,data,state,hashes=load(a)
    directions=state['directions'].cuda();normal=build_rotation(pc.get_rotation)[:,:,2]
    q=[];visibility=[];parity=[]
    rho=torch.full_like(pc.get_xyz,.5);environment=torch.tensor([math.pi/3,math.pi,3*math.pi],device='cuda')
    class Proxy:
        @property
        def get_albedo(self):return linear_to_srgb(rho)
        def __getattr__(self,key):return getattr(pc,key)
    for scale in (.5,1.,2.):
        trace=StaticVisibility(bias=float(data['shadow_bias'])*scale,update_interval=1000000)
        qs=[];vs=[]
        for i,camera in enumerate(cameras):
            light=directions[i:i+1];vis=trace(pc,light,0)
            facing=torch.where((normal*(camera.camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.)
            cosine=((normal*facing)@light.T).clamp_min(0)
            qi=cosine*vis;qs.append(qi.cpu());vs.append(vis.cpu())
            if v.TRAIN[i] in (1,81,149):
                lights=SimpleNamespace(ambient_rgb=environment,for_camera=lambda c,L=light:(L,L.new_tensor([math.pi])))
                production=render_photometric(camera,Proxy(),pipe,torch.zeros(3,device='cuda'),lights,lambda *unused:vis,0)
                local=render(camera,pc,pipe,rho*(environment/math.pi+qi))
                error=float((production['render_linear']-local['render']).abs().max())
                assert error<2e-6,error
                env=render(camera,pc,pipe,rho*environment/math.pi)['render']
                direct=render(camera,pc,pipe,rho*qi)['render']
                add_error=float((env+direct-local['render']).abs().max());assert add_error<2e-5,add_error
                parity.append(dict(frame=v.TRAIN[i],bias_scale=scale,production_max_error=error,additivity_max_error=add_error))
            if i%15==0:print(dict(stage='cache',scale=scale,frame=v.TRAIN[i]),flush=True)
        q.append(torch.stack(qs));visibility.append(torch.stack(vs))
    packed=dict(q=torch.stack(q),visibility=torch.stack(visibility),directions=state['directions'],frames=v.TRAIN,bias=float(data['shadow_bias']),scales=[.5,1.,2.])
    torch.save(packed,a.output/'raster_cache.pt')
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__))
    v.check_hashes(hashes)
    v.write(a.output/'summary.json',dict(status='PASS_FORWARD_CACHE_NOT_LIGHT_ACCEPTANCE',gt_read=False,internal_validation_passed=False,
        geometry_frozen=True,directions_frozen=True,frames=v.TRAIN,geometry_fit_frames=v.FIT,parity=parity,
        cache_sha256=v.sha(a.output/'raster_cache.pt'),input_sha256=hashes))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','fit','model','source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    p.add_argument('--mode',choices=['cache'],default='cache')
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():raise RuntimeError('需要ushas/lumimotion-ushas')
    a.output.mkdir(parents=True,exist_ok=False)
    cache(a)


if __name__=='__main__':main()
