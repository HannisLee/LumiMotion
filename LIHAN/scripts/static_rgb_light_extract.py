"""纯RGB基线验证及Gaussian ID对应；不访问任何GT属性。"""
import argparse
import json
from argparse import Namespace
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scene import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import build_rotation
from gaussian_renderer import render
from piq import ssim, LPIPS
from LIHAN.scripts.static_rgb_light_prepare import sha, write


def cameras(source, split):
    source=source.resolve()
    info=readCamerasFromTransforms(str(source),f'transforms_{split}.json','images',False,no_bg=True)
    args=Namespace(resolution=2,data_device='cpu',load2gpu_on_the_fly=True)
    return cameraList_from_camInfos(info,1,args)


def load(model, iteration):
    g=GaussianModel(3,no_binary_separation=False,fea_dim=1)
    g.load_ply(str(model/f'point_cloud/iteration_{iteration}/point_cloud.ply'),og_number_points=100000)
    g.active_sh_degree=3
    return g


PIPE=Namespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=1.,debug=False)


def image(path, array):
    x=np.asarray(array)
    if x.ndim==2: x=np.repeat(x[...,None],3,axis=-1)
    cv2.imwrite(str(path),np.uint8(np.clip(x[...,::-1],0,1)*255))


@torch.no_grad()
def validate(source, model, iteration, out):
    out.mkdir(parents=True,exist_ok=False)
    g=load(model,iteration); metric=LPIPS().cuda()
    rows=[]
    for j,c in enumerate(cameras(source,'dev')):
        c.load2device(); r=render(c,g,PIPE,torch.zeros(3,device='cuda'),0.,0.,0.)
        pred=r['render'].clamp(0,1); target=c.original_image_train_light
        alpha=r['rend_alpha'][0]; fg=c.gt_alpha_mask[0].cuda()>.99
        rgb_foreground=c.gt_alpha_mask[0].cuda()>.5
        n=F.normalize(r['rend_normal'],dim=0); ns=F.normalize(r['surf_normal'],dim=0)
        interior=F.avg_pool2d(fg.float()[None,None],5,1,2)[0,0]>.999
        mask=interior&(alpha>.95)
        horizontal=mask&(n[2].abs()>.95)
        nonhorizontal=mask&~horizontal
        normal_agreement=(n*ns).sum(0)
        row=dict(frame=int(c.image_name_train_light.split('_')[-1]),
                 psnr=float(-10*torch.log10((pred-target).square().mean())),
                 foreground_psnr=float(-10*torch.log10((pred-target)[:,rgb_foreground].square().mean())),
                 ssim=float(ssim(pred[None],target[None],data_range=1.)),
                 lpips=float(metric(pred[None],target[None])),
                 predicted_alpha_coverage=float((alpha[fg]>.95).float().mean()),
                 normal_depth_cosine=float(normal_agreement[mask].mean()),
                 near_horizontal_model_fraction=float(horizontal.sum()/mask.sum().clamp_min(1)),
                 near_horizontal_normal_depth_cosine=float(normal_agreement[horizontal].mean()) if horizontal.any() else None,
                 other_model_normal_depth_cosine=float(normal_agreement[nonhorizontal].mean()) if nonhorizontal.any() else None,
                 normal_neighbor_cosine=float((n[:,:-1,:]*n[:,1:,:]).sum(0)[mask[:-1]&mask[1:]].mean()))
        rows.append(row)
        if j in [0,8,17]:
            cat=np.concatenate([target.permute(1,2,0).cpu(),pred.permute(1,2,0).cpu()],axis=1)
            image(out/f'rgb_{row["frame"]:04d}.png',cat)
            normal=(n.permute(1,2,0).cpu().numpy()+1)/2
            depthnormal=(ns.permute(1,2,0).cpu().numpy()+1)/2
            image(out/f'normal_{row["frame"]:04d}.png',np.concatenate([normal,depthnormal],axis=1))
            image(out/f'coverage_{row["frame"]:04d}.png',alpha.cpu().numpy())
        c.load2device('cpu')
    scales=g.get_scaling.detach().cpu().numpy(); xyz=g.get_xyz.detach().cpu().numpy()
    result=dict(iteration=iteration,model=str(model),gt_used=False,
                metric_scope=dict(psnr='全图黑背景RGB',foreground_psnr='固定RGB推断mask>0.5，不腐蚀、不依预测alpha裁分母',
                                  ssim='全图黑背景RGB',lpips='全图黑背景RGB',normal='RGB推断前景内缩5px且模型alpha>0.95'),
                metrics={key:float(np.mean([r[key] for r in rows if r[key] is not None])) for key in rows[0] if key!='frame'},
                geometry_partition='仅由模型单位法线abs(n_z)>0.95定义近水平面，不是GT平台标签；其余有效面另报法线cos',
                point_count=len(xyz),scale_quantiles=np.quantile(scales,[0,.5,.9,.99,1]).tolist(),
                nonfinite_points=int((~np.isfinite(xyz).all(1)).sum()),
                point_bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],per_frame=rows)
    write(out/'metrics.json',result)
    print(json.dumps(result['metrics']),flush=True)
    return result


def sample(value,grid):
    return F.grid_sample(value[None],grid[None,None],mode='bilinear',align_corners=False)[0,:,0].T


@torch.no_grad()
def extract(source, model, iteration, out, limit, deform_enabled=False):
    out.mkdir(parents=True,exist_ok=False)
    g=load(model,iteration); cams=cameras(source,'all')
    deformation=None
    if deform_enabled:
        from scene.deform_model import DeformModel
        deformation=DeformModel(deform_type='mlp',is_blender=True,hyper_dim=1,pred_color=True)
        assert deformation.load_weights(str(model),iteration), '缺少冻结形变权重'
        deformation.deform.eval()
    frame_ids=[int(c.image_name_train_light.split('_')[-1]) for c in cams]
    assert frame_ids==list(range(1,151)), '相机帧号必须逐个完整对应1..150'
    xyz=g.get_xyz; canonical=build_rotation(g.get_rotation)[:,:,2]
    rng=np.random.default_rng(1030)
    ids=np.sort(rng.choice(len(xyz),min(limit,len(xyz)),replace=False))
    ids=torch.as_tensor(ids,device='cuda'); xyz=xyz[ids]; n=canonical[ids]
    hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],dim=1)
    frame_normals=[];frame_xyz=[]
    obs=[]; diagnostics=[]; signvotes=torch.zeros(len(ids),device='cuda');signmass=torch.zeros_like(signvotes)
    for t,c in enumerate(cams):
        c.load2device()
        if deformation is not None:
            d=deformation.step(g.get_xyz,c.fid.unsqueeze(0).expand(len(g.get_xyz),-1),iteration=iteration,
                               feature=g.get_binary_feature(),camera_center=c.camera_center)
            r=render(c,g,PIPE,torch.zeros(3,device='cuda'),d['d_xyz'],d['d_rotation'],d['d_scaling'],
                     d_opacity=d['d_opacity'],d_color=d['d_color'])
            xyz=(g.get_xyz+d['d_xyz'])[ids]
            n=build_rotation(g.get_rotation_bias(d['d_rotation']))[ids,:,2]
            hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],dim=1)
        else:
            r=render(c,g,PIPE,torch.zeros(3,device='cuda'),0.,0.,0.)
        frame_normals.append(n.cpu().numpy());frame_xyz.append(xyz.cpu().numpy())
        clip=hom@c.full_proj_transform; grid=clip[:,:2]/clip[:,3:4]
        z=(hom@c.world_view_transform)[:,2]
        val=sample(c.original_image_train_light,grid)
        depth=sample(r['surf_depth'],grid)[:,0]
        alpha=sample(r['rend_alpha'],grid)[:,0]
        rn=F.normalize(sample(r['rend_normal'],grid),dim=1)
        sn=F.normalize(sample(r['surf_normal'],grid),dim=1)
        fg=c.gt_alpha_mask.cuda()
        interior=F.avg_pool2d(fg[None],7,1,3)[0]
        fgval=sample(interior,grid)[:,0]
        ray=F.normalize(c.camera_center[None]-xyz,dim=1)
        facing=(n*ray).sum(1); similarity=(n*rn).sum(1)
        depth_delta=(z-depth).abs()
        ok=(grid.abs().max(1).values<.98)&(z>0)&(alpha>.95)&(fgval>.999)
        ok&=(depth_delta<(.005*z+.005))&(similarity.abs()>.9)&(facing.abs()>.15)
        ok&=((rn*sn).sum(1)>.8)&(val.max(1).values<.98)&(val.min(1).values>.002)
        confidence=(similarity.abs()*facing.abs()).clamp(0,1)
        signvotes+=torch.where(ok,similarity.sign()*confidence,torch.zeros_like(confidence))
        signmass+=torch.where(ok,confidence,torch.zeros_like(confidence))
        index=torch.nonzero(ok).flatten()
        obs.append(dict(t=t,ids=index.cpu().numpy(),rgb=val[index].cpu().numpy(),
                        sign=similarity[index].sign().cpu().numpy(),weight=confidence[index].cpu().numpy(),
                        uv=grid[index].cpu().numpy()))
        diagnostics.append(dict(frame=t+1,reliable=int(ok.sum()),projected=int((grid.abs().max(1).values<1).sum()),
                                depth_median=float(depth_delta[ok].median()) if ok.any() else None,
                                normal_second_moment_eigenvalues=torch.linalg.eigvalsh(n[ok].T@n[ok]/ok.sum()).cpu().tolist() if ok.sum()>=3 else None))
        if t in [0,29,59,89,119,149]:
            pic=c.original_image_train_light.permute(1,2,0).cpu().numpy().copy()
            for uv in grid[index][::max(1,len(index)//1000)].cpu().numpy():
                xy=np.rint((uv+1)*np.array([c.image_width,c.image_height])/2-.5).astype(int)
                cv2.circle(pic,tuple(xy),1,(0,1,0),-1)
            image(out/f'correspondence_{t+1:04d}.png',pic)
            image(out/f'rgb_{t+1:04d}.png',np.concatenate([c.original_image_train_light.permute(1,2,0).cpu(),r['render'].clamp(0,1).permute(1,2,0).cpu()],axis=1))
            image(out/f'normal_{t+1:04d}.png',np.concatenate([(F.normalize(r['rend_normal'],dim=0).permute(1,2,0).cpu()+1)/2,
                 (F.normalize(r['surf_normal'],dim=0).permute(1,2,0).cpu()+1)/2],axis=1))
        c.load2device('cpu')
    signs=torch.where(signvotes>=0,1.,-1.).cpu().numpy()
    sign_consistency=(signvotes.abs()/signmass.clamp_min(1e-8)).cpu().numpy()
    intensity=np.full((150,len(ids)),np.nan,dtype=np.float32); weights=np.zeros_like(intensity)
    rgb=np.full((150,len(ids),3),np.nan,dtype=np.float32)
    projection_uv=np.full((150,len(ids),2),np.nan,dtype=np.float32)
    for o in obs:
        keep=(o['sign']==signs[o['ids']])&(sign_consistency[o['ids']]>.8)
        jj=o['ids'][keep]; rr=o['rgb'][keep]
        linear=np.where(rr<=.04045,rr/12.92,((rr+.055)/1.055)**2.4)
        intensity[o['t'],jj]=linear@np.array([.2126,.7152,.0722]); weights[o['t'],jj]=o['weight'][keep]
        rgb[o['t'],jj]=rr
        projection_uv[o['t'],jj]=o['uv'][keep]
    normals=(np.stack(frame_normals)*signs[None,:,None]) if deform_enabled else n.cpu().numpy()*signs[:,None]
    np.savez_compressed(out/'observations.npz',frames=np.asarray(frame_ids),gaussian_ids=ids.cpu().numpy(),
                        xyz=g.get_xyz[ids].cpu().numpy(),frame_xyz=np.stack(frame_xyz),normals=normals,intensity=intensity,weights=weights,rgb=rgb,
                        projection_uv=projection_uv,sign_consistency=sign_consistency)
    write(out/'extraction.json',dict(model=str(model),iteration=iteration,checkpoint_sha256=sha(model/f'point_cloud/iteration_{iteration}/point_cloud.ply'),
          direction_convention='surface_to_light',coordinate_system='world',gt_read=False,
          deform_enabled=deform_enabled,normal_layout='frame,gaussian,xyz' if deform_enabled else 'gaussian,xyz',
          stage1_frozen='PLY及MLP参数固定；若开启形变，逐帧确定性前向，不优化Stage1',
          sampling_seed=1030,max_gaussians=limit,gaussians=len(ids),observations=int(np.isfinite(intensity).sum()),
          rasterizer_provenance=__import__('sys').modules['diff_surfel_rasterization'].provenance,
          torch_version=torch.__version__,cuda_version=torch.version.cuda,
          quantile_assumption='仅原RGB逆sRGB变换；不访问导出灯光及材质配置',per_frame=diagnostics))
    print('对应提取完成',int(np.isfinite(intensity).sum()),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--action',choices=['validate','extract'],required=True); p.add_argument('--max-gaussians',type=int,default=30000)
    p.add_argument('--deform',action='store_true',help='使用冻结MLP逐帧几何和外观；观测亮度仍来自原RGB')
    a=p.parse_args()
    if a.action=='validate': validate(a.source,a.model,a.iteration,a.output)
    else: extract(a.source,a.model,a.iteration,a.output,a.max_gaussians,a.deform)


if __name__=='__main__': main()
