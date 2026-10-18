"""隔离CUDA副本记录TRAIN主相机median贡献Gaussian，不改生产库或接收选择。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','fit','model','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():raise RuntimeError('需要ushas环境')
    a.output.mkdir(parents=True,exist_ok=False)
    v.install_guard(a);data,state,fit,result,hashes=v.load_inputs(a)
    from scene import pbr_rasterizer as backend
    import gaussian_renderer as renderer
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from utils.general_utils import build_rotation
    from scripts.probe_dynamic_dl_selfcal import read_config
    original=backend.native_module();original_source=backend.SOURCE
    # 只复制本实验副本；生产文件和已加载原模块保持不动。
    copied=a.output/'native_source';shutil.copytree(original_source,copied,ignore=shutil.ignore_patterns('build','*.so','__pycache__','.git','third_party'))
    forward=copied/'cuda_rasterizer/forward.cu';text=forward.read_text()
    replacements=[('float median_contributor = {-1};','float median_contributor = {-1};\n\tfloat median_gaussian_id = {-1};'),
      ('median_contributor = contributor;','median_contributor = contributor;\n\t\t\t\tmedian_gaussian_id = collected_id[j];'),
      ('out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;','out_others[pix_id + DISTORTION_OFFSET * H * W] = median_gaussian_id;')]
    for old,new in replacements:
        assert text.count(old)==1,old;text=text.replace(old,new)
    forward.write_text(text)
    v.write(a.output/'instrumentation.json',dict(production_source_manifest=original.provenance,
      changes=replacements,scope='诊断副本distortion通道改为median Gaussian ID；只能no_grad forward，禁止训练/backward',original_forward_sha256=v.sha(original_source/'cuda_rasterizer/forward.cu'),diagnostic_forward_sha256=v.sha(forward)))
    backend.SOURCE=copied;backend.native_module.cache_clear();diagnostic=backend.native_module()
    v.write(a.output/'diagnostic_backend.json',diagnostic.provenance)
    cfg=read_config(a.model/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply'));pc.active_sh_degree=min(a.iteration//1000,cfg.sh_degree)
    normal=build_rotation(pc.get_rotation)[:,:,2];plane=data['normal'].cuda();origin=data['origin'].cuda()
    pool=torch.where((pc.get_opacity[:,0]>.5)&(((pc.get_xyz-origin)@plane).abs()<.005*data['extent'])&
                     ((normal@plane).abs()>math.cos(math.radians(10))))[0]
    points=pc.get_xyz[pool];homogeneous=torch.cat([points,torch.ones_like(points[:,:1])],-1)
    colors=torch.sigmoid(pc.get_xyz);bg=torch.zeros(3,device='cuda');pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=1.)
    records=[];frames=[];parity=[]
    for index,info in enumerate(readCamerasFromTransforms(str(a.source),'transforms_train.json','images',False)):
        frame=int(info.image_name_train_light.split('_')[-1])
        if frame not in v.FIT:continue
        camera=loadCam(cfg,index,info,1.);camera.load2device()
        renderer.GaussianRasterizationSettings=original.GaussianRasterizationSettings;renderer.GaussianRasterizer=original.GaussianRasterizer
        base=renderer.render(camera,pc,pipe,bg,0.,0.,0.,override_color=colors)
        renderer.GaussianRasterizationSettings=diagnostic.GaussianRasterizationSettings;renderer.GaussianRasterizer=diagnostic.GaussianRasterizer
        diag=renderer.render(camera,pc,pipe,bg,0.,0.,0.,override_color=colors)
        checks={k:float((base[k]-diag[k]).abs().max()) for k in ('render','rend_alpha','rend_normal','surf_depth','radii')}
        if frame in (1,81,149):
            pipe.depth_ratio=0.
            renderer.GaussianRasterizationSettings=original.GaussianRasterizationSettings;renderer.GaussianRasterizer=original.GaussianRasterizer
            expected=renderer.render(camera,pc,pipe,bg,0.,0.,0.,override_color=colors)['surf_depth']
            renderer.GaussianRasterizationSettings=diagnostic.GaussianRasterizationSettings;renderer.GaussianRasterizer=diagnostic.GaussianRasterizer
            expected_diag=renderer.render(camera,pc,pipe,bg,0.,0.,0.,override_color=colors)['surf_depth']
            checks['expected_depth']=float((expected-expected_diag).abs().max());pipe.depth_ratio=1.
        assert max(checks.values())==0.,checks
        ids=diag['rend_dist'][0].round().long();h,w=ids.shape
        assert float((diag['rend_dist'][0]-ids).abs().max())==0.
        alpha=camera.gt_alpha_mask.cuda()[0];interior=-F.max_pool2d(-alpha[None,None],3,stride=1,padding=1)[0,0]
        mask=(interior>.99)&(base['rend_alpha'][0]>.9)&(ids>=0)&(ids<len(pc.get_xyz))
        pixel_count=torch.bincount(ids[mask],minlength=len(pc.get_xyz))[pool]
        clip=homogeneous@camera.full_proj_transform;ndc=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
        u=((ndc[:,0]+1)*w/2-.5).round().long();vv=((ndc[:,1]+1)*h/2-.5).round().long()
        inside=(clip[:,3]>0)&(u>=1)&(u<w-1)&(vv>=1)&(vv<h-1)
        u,vv=u.clamp(1,w-2),vv.clamp(1,h-2)
        center_valid=inside&mask[vv,u];center_own=center_valid&(ids[vv,u]==pool)
        near_own=torch.zeros_like(center_own)
        for dy in (-1,0,1):
            for dx in (-1,0,1):near_own|=inside&mask[vv+dy,u+dx]&(ids[vv+dy,u+dx]==pool)
        z=(homogeneous@camera.world_view_transform)[:,2]
        record=dict(pixel_count=pixel_count.cpu(),center_valid=center_valid.cpu(),center_own=center_own.cpu(),near_own=near_own.cpu(),
                    center_median_id=ids[vv,u].cpu(),center_median_error=(z-base['surf_depth'][0,vv,u]).cpu())
        records.append(record);frames.append(frame);parity.append(dict(frame=frame,checks=checks))
        if frame in (1,81,149):torch.save(dict(ids=ids.cpu(),valid=mask.cpu(),median_depth=base['surf_depth'].cpu(),alpha=base['rend_alpha'].cpu()),a.output/f'fit_{frame:04d}_primary.pt')
        print(json.dumps(dict(frame=frame,pool=len(pool),primary_any=int((pixel_count>0).sum()),center_own=int(center_own.sum()),near_own=int(near_own.sum()),parity_max=max(checks.values()))),flush=True)
        del camera,base,diag
    tensors={key:torch.stack([r[key] for r in records]) for key in records[0]}
    tensors.update(pool_ids=pool.cpu(),frames=frames,points=points.cpu(),normal=normal[pool].cpu(),
                   point_opacity=pc.get_opacity[pool].cpu(),height=((points-origin)@plane).cpu())
    torch.save(tensors,a.output/'primary_evidence.pt')
    v.write(a.output/'render_parity.json',dict(status='PASS',frames=parity))
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__));hashes[str(original_source/'cuda_rasterizer/forward.cu')]=v.sha(original_source/'cuda_rasterizer/forward.cu')
    assert hashes[str(original_source/'cuda_rasterizer/forward.cu')]==v.read(a.output/'instrumentation.json')['original_forward_sha256']
    v.check_hashes(hashes)
    v.write(a.output/'summary.json',dict(status='PASS_PRIMARY_ID_DIAGNOSTIC_NOT_LIGHT_ACCEPTANCE',gt_read=False,external_test_read=False,
       fit_frames=frames,held_used=False,pool_count=len(pool),internal_validation_passed=False,primary_evidence_sha256=v.sha(a.output/'primary_evidence.pt'),
       any_primary_views_histogram=torch.bincount((tensors['pixel_count']>0).sum(0),minlength=118).tolist(),
       center_own_views_histogram=torch.bincount(tensors['center_own'].sum(0),minlength=118).tolist(),input_sha256=hashes,
       next_step='仅证据缓存；未挑选新接收集合、未重拟合或导出。'))


if __name__=='__main__':main()
