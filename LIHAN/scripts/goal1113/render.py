"""逐高斯使用训练同一公式，再进行冻结几何光栅合成。"""
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from . import inputs as I, physics as P
from .supplement import extract


def srgb(x):
    x=np.maximum(x,0.)
    return np.where(x<=.0031308,12.92*x,1.055*np.power(x,1/2.4)-.055)


@torch.no_grad()
def run(root, device):
    from LIHAN.scripts.static_rgb_light_extract import cameras, load
    from scene.deform_model import DeformModel
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.general_utils import build_rotation
    if str(device) not in ('cuda','cuda:0'): raise ValueError('原生光栅器需要当前可见 CUDA 设备')
    root=Path(root); status=json.loads((root/'fitting/status.json').read_text())
    stage=status.get('selected') or status.get('stage') or 'A'
    if stage not in ('A','B','C1','C2','C3'): raise ValueError('无可渲染阶段')
    path=(root/'fitting'/stage/'checkpoint.pt') if stage=='A' else sorted((root/'fitting'/stage).glob('checkpoint_*.pt'))[-1]
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    p=checkpoint['params']; config=json.loads((root/'inputs/config.json').read_text()); K=config['K']
    out=root/'render'; out.mkdir(exist_ok=False)
    I.whitelist(); source=I.load_observations()
    extract(out/'supplement.npz')
    with np.load(out/'supplement.npz') as q: data={k:q[k] for k in q.files}
    npts=len(data['rho_y']); ids=source['raw']['gaussian_ids']; fi=source['support']['body_fit']
    data['rho_y'][ids]=source['rho_y']; data['observed'][ids]=source['w'].sum(0)>0
    unbounded=data['rho_y']/K; rho=np.repeat(np.clip(unbounded,0,1)[:,None],3,axis=1)
    rho[~data['observed']]=0; rho[ids[fi]]=p['rho'].detach().cpu().numpy()
    material=torch.tensor(rho,dtype=torch.float64,device=device)
    signs=torch.tensor(data['sign'],dtype=torch.float64,device=device)
    env=p['env'].to(device=device,dtype=torch.float64); direction=P.direction(p['phi'],p['a']).to(device)
    cameras_all=cameras(I.RUNTIME,'all'); g=load(I.MODEL,35000)
    if len(g.get_xyz)!=npts: raise ValueError('高斯数量不匹配')
    deform=DeformModel(deform_type='mlp',is_blender=True,hyper_dim=1,pred_color=True)
    if not deform.load_weights(str(I.MODEL),35000): raise ValueError('缺少形变')
    deform.deform.eval(); video=None; coverage=[]
    for t,c in enumerate(cameras_all):
        c.load2device(); d=deform.step(g.get_xyz,c.fid.unsqueeze(0).expand(npts,-1),iteration=35000,
                 feature=g.get_binary_feature(),camera_center=c.camera_center)
        means=g.get_xyz+d['d_xyz']; rot=g.get_rotation_bias(d['d_rotation']); scales=g.get_scaling+d['d_scaling']
        opacity=g.get_opacity if d['d_opacity'] is None else g.get_opacity+d['d_opacity']
        n=build_rotation(rot)[:,:,2].to(torch.float64)*signs[:,None]
        view=F.normalize((c.camera_center[None]-means).to(torch.float64),dim=-1)
        direct=[]; static=[]
        for j in range(0,npts,512):
            sl=slice(j,min(j+512,npts))
            a,b=P.components(n[sl][None],view[sl][None],material[sl],env,direction[t:t+1],K)
            direct.append(a[0]); static.append(b[0])
        dc=torch.cat(direct).float(); sc=torch.cat(static).float()
        settings=GaussianRasterizationSettings(image_height=c.image_height,image_width=c.image_width,
            tanfovx=math.tan(c.FoVx/2),tanfovy=math.tan(c.FoVy/2),bg=torch.zeros(3,device=device),
            scale_modifier=1.,viewmatrix=c.world_view_transform,projmatrix=c.full_proj_transform,
            sh_degree=g.active_sh_degree,campos=c.camera_center,prefiltered=False,debug=False)
        raster=GaussianRasterizer(settings)
        def rasterize(colors):
            image,_,maps=raster(means3D=means,means2D=torch.zeros_like(means),shs=None,colors_precomp=colors,
                                opacities=opacity,scales=scales,rotations=rot,cov3D_precomp=None)
            return image,maps
        di,maps=rasterize(dc); st,_=rasterize(sc); composed,_=rasterize(dc+sc)
        if not torch.allclose(composed,di+st,atol=2e-5,rtol=2e-5): raise ValueError('光栅线性可加性不符')
        al,_=rasterize(material.float())
        normal,_=rasterize(((n+1)/2).float()); missing,_=rasterize(torch.tensor(~data['observed'],device=device,dtype=torch.float32)[:,None].expand(-1,3).contiguous())
        alpha=maps[1:2]; total=di+st
        arrays={k:v.permute(1,2,0).cpu().numpy() for k,v in dict(total=total,direct=di,static=st,albedo=al,normal=normal,alpha=alpha).items()}
        np.savez_compressed(out/f'linear_{t+1:04d}.npz',**arrays)
        frame=np.uint8(np.round(np.clip(srgb(arrays['total']),0,1)*255))
        if t in (0,37,74,112,149):
            for name,arr in arrays.items():
                shown=srgb(arr) if name in ('total','direct','static','albedo') else arr
                shown=np.uint8(np.round(np.clip(shown,0,1)*255))
                if shown.shape[-1]==1: shown=shown[...,0]
                else: shown=shown[...,::-1]
                cv2.imwrite(str(out/f'{name}_{t+1:04d}.png'),shown)
        if video is None:
            video=cv2.VideoWriter(str(out/'total.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(c.image_width,c.image_height))
            if not video.isOpened(): raise RuntimeError('视频编码器不可用')
        video.write(frame[...,::-1]); coverage.append(float(missing.sum()/alpha.sum().clamp_min(1e-12)/3))
        c.load2device('cpu'); print('光栅渲染',t+1,flush=True)
    video.release()
    reader=cv2.VideoCapture(str(out/'total.mp4')); count=0
    while True:
        ok,frame=reader.read()
        if not ok: break
        if frame.shape[:2]!=(c.image_height,c.image_width): raise ValueError('视频尺寸不符')
        count+=1
    reader.release()
    if count!=150: raise ValueError(f'视频解码帧数不符：{count}')
    for t in (1,38,75,113,150):
        for name in ('total','direct','static','albedo','normal','alpha'):
            image=cv2.imread(str(out/f'{name}_{t:04d}.png'),cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[:2]!=(c.image_height,c.image_width): raise ValueError('代表图读回失败')
    (out/'summary.json').write_text(json.dumps(dict(stage=stage,selected=status.get('selected'),unobserved=int((~data['observed']).sum()),
        unobserved_screen_fraction=coverage,clipped_nonfit=int((unbounded>1).sum()),
        clipped_max_error=float(np.maximum(unbounded-1,0).max()),indirect=0,video_decoded_frames=count,
        component_raster_additivity='PASS',representative_png_readback='PASS'),ensure_ascii=False,indent=2)+'\n')
