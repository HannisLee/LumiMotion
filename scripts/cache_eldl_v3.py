"""由117-fit静态SH几何与TRAIN图像建立V3混合估光缓存，不读取GT。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.stage2_photometric import srgb_to_linear
from scene.static_dl_photometric import StaticVisibility
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config,save_image


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source,model=a.source.resolve(),a.model.resolve()
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([source,model,out])
    marker=json.loads((model/'eldl_v3_training.json').read_text())
    sm=json.loads((source/'no_gt_manifest.json').read_text())
    assert not marker['gt_supervision'] and not marker['init_weights_loaded'] and marker['baseline']=='static_sh'
    assert marker['train_frames']==sm['internal_fit_frames'] and not set(marker['train_frames'])&set(sm['internal_held_frames'])
    ply=model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply'
    deform=model/f'deform/iteration_{a.iteration}/deform.pth'
    weights=torch.load(deform,map_location='cpu',weights_only=True)
    for head in ('gaussian_warp','gaussian_rotation','gaussian_scaling'):
        for suffix in ('weight','bias'):assert torch.count_nonzero(weights[f'{head}.{suffix}'])==0
    cfg=read_config(model/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(ply));pc.active_sh_degree=min(a.iteration//1000,cfg.sh_degree)
    cloud=pc.get_xyz[pc.get_opacity[:,0]>.5]
    gen=torch.Generator(device='cuda').manual_seed(1019)
    if len(cloud)>20000:cloud=cloud[torch.randperm(len(cloud),generator=gen,device='cuda')[:20000]]
    assert len(cloud)>=1000,'拟合几何有效点不足'
    extent=float((torch.quantile(cloud,.99,dim=0)-torch.quantile(cloud,.01,dim=0)).norm())
    triples=cloud[torch.randint(len(cloud),(512,3),device='cuda',generator=gen)]
    candidates=F.normalize(torch.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0],dim=-1),dim=-1)
    distance=(cloud@candidates.T-(triples[:,0]*candidates).sum(-1)[None]).abs()
    scores=(distance<extent*.005).sum(0);scores[candidates.norm(dim=-1)<.9]=0
    inliers=distance[:,int(scores.argmax())]<extent*.005
    origin=cloud[inliers].mean(0)
    _,_,vectors=torch.linalg.svd(cloud[inliers]-origin,full_matrices=False)
    normal=vectors[-1]
    if ((cloud-origin)@normal).mean()<0:normal=-normal
    u=F.normalize(vectors[0]-(vectors[0]@normal)*normal,dim=0)
    basis=torch.stack([u,F.normalize(torch.cross(normal,u,dim=0),dim=0)])
    uv=(cloud[inliers]-origin)@basis.T
    lower,upper=torch.quantile(uv,.01,dim=0),torch.quantile(uv,.99,dim=0)
    yy,xx=torch.meshgrid(torch.linspace(0,1,64,device='cuda'),torch.linspace(0,1,64,device='cuda'),indexing='ij')
    plane_uv=lower+torch.stack([xx.flatten(),yy.flatten()],-1)*(upper-lower)
    points=origin+plane_uv@basis
    k=torch.arange(512,device='cuda');z=.05+.95*(k+.5)/512;phi=k*math.pi*(3-math.sqrt(5));rad=(1-z*z).sqrt()
    directions=(rad*phi.cos())[:,None]*basis[0]+(rad*phi.sin())[:,None]*basis[1]+z[:,None]*normal
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    bg=torch.zeros(3,device='cuda');records=[];frames=[]
    infos=readCamerasFromTransforms(str(source),'transforms_train.json','images',False)
    assert len(infos)==135
    homogeneous=torch.cat([points,torch.ones_like(points[:,:1])],-1)
    for index,info in enumerate(infos):
        camera=loadCam(cfg,index,info,1.);camera.load2device()
        package=render(camera,pc,pipe,bg,0.,0.,0.,override_color=torch.zeros_like(pc.get_xyz))
        clip=homogeneous@camera.full_proj_transform;ndc=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
        h,w=camera.image_height,camera.image_width
        uu=((ndc[:,0]+1)*w/2-.5).round().long();vv=((ndc[:,1]+1)*h/2-.5).round().long()
        inside=(clip[:,3]>0)&(uu>=1)&(uu<w-1)&(vv>=1)&(vv<h-1)
        uu,vv=uu.clamp(0,w-1),vv.clamp(0,h-1)
        zc=(homogeneous@camera.world_view_transform)[:,2]
        depth_error=(zc-package['surf_depth'][0,vv,uu]).abs()
        alpha=camera.gt_alpha_mask.cuda()[0]
        interior=-F.max_pool2d(-alpha[None,None],3,stride=1,padding=1)[0,0]
        valid=inside&(interior[vv,uu]>.99)&(package['rend_alpha'][0,vv,uu]>.9)&(depth_error<extent*.01)
        # 每像素只保留一个固定平面点，选择与预测深度最接近者。
        ids=torch.where(valid)[0].cpu().numpy();order=ids[np.argsort(depth_error[ids].cpu().numpy(),kind='stable')]
        pixels=(vv*w+uu).cpu().numpy();_,first=np.unique(pixels[order],return_index=True)
        valid.zero_();valid[torch.as_tensor(order[first],device='cuda')]=True
        rgb=srgb_to_linear(camera.original_image_train_light[:,vv,uu].T)
        sat=camera.observed_saturation_mask[:,vv.cpu(),uu.cpu()].T
        frame=int(camera.image_name_train_light.split('_')[-1]);fit=frame in sm['internal_fit_frames']
        records.append(dict(rgb=rgb.cpu(),saturated=sat.cpu(),valid=valid.cpu(),pixels=torch.stack([uu,vv],-1).cpu()))
        frames.append(dict(frame=frame,time=float(camera.fid.item()),fit=fit,valid=int(valid.sum())))
        if fit and frame in (1,21,41,81,101,121,141):
            preview=camera.original_image_train_light.clone();preview[:,vv[valid],uu[valid]]=preview.new_tensor([1.,0.,0.])[:,None]
            save_image(out/f'fit_{frame:04d}_samples.png',preview)
        print(json.dumps(frames[-1]),flush=True)
        del camera,package
    data={key:torch.stack([r[key] for r in records]) for key in records[0]}
    fit=torch.tensor([f['fit'] for f in frames]);support=data['valid'][fit].sum(0)>=10
    assert int(support.sum())>=100,'fit跨视角接收点支持不足'
    data['valid'] &= support[None]
    # 用相同预测Gaussian BVH在所有候选上计算连续透射率，完全不使用黑暗标签。
    bias=extent*.003
    visibility=StaticVisibility(bias=bias,update_interval=1)
    visibility(pc,directions[:1],a.iteration)
    vis=[]
    for start in range(0,len(directions),16):
        light=directions[start:start+16]
        rays_d=light[None].expand(len(points),-1,-1).contiguous()
        rays_o=points[:,None]+rays_d*bias
        traced=pc.trace(rays_o.contiguous(),rays_d,features=torch.zeros(len(pc.get_xyz),4,device='cuda'),
                        xyz=pc.get_xyz,scales=pc.get_scaling,rotation=pc.get_rotation,opacity=pc.get_opacity,use_zeros=True)
        vis.append((1-traced['alpha']).clamp(0,1).T.cpu())
    data.update(visibility=torch.cat(vis),directions=directions.cpu(),normal=normal.cpu(),origin=origin.cpu(),
                basis=basis.cpu(),points=points.cpu(),fit=fit,times=torch.tensor([f['time'] for f in frames]),
                frames=frames,extent=extent,shadow_bias=bias)
    torch.save(data,out/'cache.pt')
    hashes={str(p):sha(p) for p in (ply,deform,model/'eldl_v3_training.json',source/'no_gt_manifest.json',Path(__file__).resolve())}
    manifest=dict(status='PASS_CACHE_ENGINEERING',gt_read=False,source_split='TRAIN',
                  model=str(model),iteration=a.iteration,source=str(source),input_sha256=hashes,
                  cache_sha256=sha(out/'cache.pt'),fit_frames=sm['internal_fit_frames'],held_frames=sm['internal_held_frames'],
                  plane_inliers=int(inliers.sum()),plane_points=int(support.sum()),extent=extent,shadow_bias=bias,
                  valid_fit_observations=int(data['valid'][fit].sum()),valid_held_observations=int(data['valid'][~fit].sum()),
                  scope='几何由fit117训练；点集合/支持只由fit决定；held仅生成冻结预测评测缓存，不拟合',
                  visibility='真实预测Gaussian BVH连续透射率；共享4096×512缓存，无黑暗标签')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
