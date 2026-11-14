"""按冻结提取规则为完整高斯补采样，不读取光照真值。"""
import numpy as np
import torch
import torch.nn.functional as F
from . import inputs as I


@torch.no_grad()
def extract(out):
    from LIHAN.scripts.static_rgb_light_extract import cameras, load, sample, PIPE
    from LIHAN.scripts.static_rgb_light_fit_r3 import kkt
    from scene.deform_model import DeformModel
    from gaussian_renderer import render
    from utils.general_utils import build_rotation
    cams=cameras(I.RUNTIME,'all'); g=load(I.MODEL,35000)
    deform=DeformModel(deform_type='mlp',is_blender=True,hyper_dim=1,pred_color=True)
    if not deform.load_weights(str(I.MODEL),35000): raise ValueError('缺少形变权重')
    deform.deform.eval(); count=len(g.get_xyz)
    votes=torch.zeros(count,device='cuda'); mass=torch.zeros_like(votes)
    # 先记录未翻转法线与有效样本，再以跨帧固定符号筛选。
    normals=np.empty((150,count,3),np.float32); intens=np.zeros((150,count),np.float32)
    weights=np.zeros((150,count),np.float32); obs_sign=np.zeros((150,count),np.int8)
    for t,c in enumerate(cams):
        c.load2device()
        d=deform.step(g.get_xyz,c.fid.unsqueeze(0).expand(count,-1),iteration=35000,
                      feature=g.get_binary_feature(),camera_center=c.camera_center)
        xyz=g.get_xyz+d['d_xyz']; n=build_rotation(g.get_rotation_bias(d['d_rotation']))[:,:,2]
        normals[t]=n.cpu().numpy()
        # 颜色显式为零，仅取得几何深度/法线/alpha，不计算 SH 间接光。
        r=render(c,g,PIPE,torch.zeros(3,device='cuda'),d['d_xyz'],d['d_rotation'],d['d_scaling'],
                 d_opacity=d['d_opacity'],override_color=torch.zeros_like(xyz))
        hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],1); clip=hom@c.full_proj_transform
        grid=clip[:,:2]/clip[:,3:4]; z=(hom@c.world_view_transform)[:,2]
        val=sample(c.original_image_train_light,grid); depth=sample(r['surf_depth'],grid)[:,0]
        alpha=sample(r['rend_alpha'],grid)[:,0]
        rn=F.normalize(sample(r['rend_normal'],grid),dim=1); sn=F.normalize(sample(r['surf_normal'],grid),dim=1)
        fg=F.avg_pool2d(c.gt_alpha_mask.cuda()[None],7,1,3)[0]; fgval=sample(fg,grid)[:,0]
        ray=F.normalize(c.camera_center[None]-xyz,dim=1); facing=(n*ray).sum(1); sim=(n*rn).sum(1)
        ok=(grid.abs().max(1).values<.98)&(z>0)&(alpha>.95)&(fgval>.999)
        ok&=((z-depth).abs()<(.005*z+.005))&(sim.abs()>.9)&(facing.abs()>.15)
        ok&=((rn*sn).sum(1)>.8)&(val.max(1).values<.98)&(val.min(1).values>.002)
        confidence=(sim.abs()*facing.abs()).clamp(0,1); sig=sim.sign()
        votes+=torch.where(ok,sig*confidence,0.); mass+=torch.where(ok,confidence,0.)
        lin=torch.where(val<=.04045,val/12.92,((val+.055)/1.055).pow(2.4))
        intens[t]=(lin@lin.new_tensor([.2126,.7152,.0722])).cpu().numpy()
        weights[t]=torch.where(ok,confidence,0.).cpu().numpy()
        obs_sign[t]=torch.where(ok,sig,0.).to(torch.int8).cpu().numpy()
        c.load2device('cpu'); print('补采样',t+1,flush=True)
    sign=torch.where(votes>=0,1.,-1.).cpu().numpy(); consistency=(votes.abs()/mass.clamp_min(1e-8)).cpu().numpy()
    keep=(obs_sign==sign[None])&(consistency[None]>.8)
    weights*=keep; intens=np.where(weights>0,intens,0.).astype(np.float64)
    with np.load(I.OBS,allow_pickle=False) as old:
        ids=old['gaussian_ids']; reference=old['normals']
    # 原冻结观察中的高斯必须保持已有法线符号和采样，不由补采样覆盖。
    dot=(normals[0,ids]*reference[0]).sum(-1)
    if (np.abs(dot)<.99).any(): raise ValueError('补采样与冻结法线不一致')
    sign[ids]=np.where(dot>=0,1.,-1.)
    l=np.load(I.BASE/'predictions.npz',allow_pickle=False)['E4']
    n=torch.from_numpy(normals.astype(np.float64))*torch.from_numpy(sign.astype(np.float64))[None,:,None]
    x=torch.einsum('tgc,tc->tg',n,torch.from_numpy(l)).clamp_min(0)
    rho,b=kkt(x,torch.from_numpy(intens),torch.from_numpy(weights.astype(np.float64)))
    np.savez_compressed(out,sign=sign,rho_y=rho.numpy(),b=b.numpy(),observed=weights.sum(0)>0,
                        observations=(weights>0).sum(0),consistency=consistency)
    return count
