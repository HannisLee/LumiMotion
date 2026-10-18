"""隔离CUDA副本记录自身采样像素的实际alpha/T_before，不改生产库。"""
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
    old='\n#if RENDER_AXUTILITY\n\t\t\t// Render depth distortion map'
    inserted='''
            // 仅诊断：alpha阈值/提前终止已通过，只记实际进入合成的自身中心像素。
            int diagnostic_id = collected_id[j];
            if ((int)pix_id == (int)bg_color[4 + diagnostic_id]) {
                // 四组P元素的地址由总像素之外的独立输出通道承载。
                int diagnostic_p = (int)bg_color[3];
                out_others[8 * H * W + diagnostic_id] = T;
                out_others[8 * H * W + diagnostic_p + diagnostic_id] = alpha;
                out_others[8 * H * W + 2 * diagnostic_p + diagnostic_id] = T * alpha;
                out_others[8 * H * W + 3 * diagnostic_p + diagnostic_id] = pix_id + 1;
            }
'''+old
    replacements=[(old,inserted)]
    for old,new in replacements:
        assert text.count(old)==1,old;text=text.replace(old,new)
    forward.write_text(text)
    raster=copied/'rasterize_points.cu';cuda=raster.read_text()
    old_alloc='torch::Tensor out_others = torch::full({3+3+2, H, W}, 0.0, float_opts);'
    assert cuda.count(old_alloc)==1
    cuda=cuda.replace(old_alloc,'torch::Tensor out_others = torch::full({8 + (4 * P + H * W - 1) / (H * W), H, W}, 0.0, float_opts);')
    # Python传入bg前三项+P+逐Gaussian指定采样像素；原渲染仍只读取前三项。
    raster.write_text(cuda)
    v.write(a.output/'instrumentation.json',dict(production_source_manifest=original.provenance,
      changes=replacements,scope='独立新增输出通道记录实际T/alpha/weight/pixel；原8图不变，只能no_grad forward，禁止训练/backward',original_forward_sha256=v.sha(original_source/'cuda_rasterizer/forward.cu'),diagnostic_forward_sha256=v.sha(forward),diagnostic_rasterize_sha256=v.sha(raster)))
    backend.SOURCE=copied;backend.native_module.cache_clear();diagnostic=backend.native_module()
    v.write(a.output/'diagnostic_backend.json',diagnostic.provenance)
    cfg=read_config(a.model/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(a.model/f'point_cloud/iteration_{a.iteration}/point_cloud.ply'));pc.active_sh_degree=min(a.iteration//1000,cfg.sh_degree)
    latest={}
    class PrefixRasterizer(diagnostic.GaussianRasterizer):
        def forward(self,*args,**kwargs):
            values=super().forward(*args,**kwargs)
            latest['prefix']=values[2][8:].reshape(-1)[:4*len(pc.get_xyz)].reshape(4,-1)
            return values
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
        h,w=camera.image_height,camera.image_width
        clip=homogeneous@camera.full_proj_transform;ndc=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
        u=((ndc[:,0]+1)*w/2-.5).round().long();vv=((ndc[:,1]+1)*h/2-.5).round().long()
        inside=(clip[:,3]>0)&(u>=1)&(u<w-1)&(vv>=1)&(vv<h-1)
        u,vv=u.clamp(1,w-2),vv.clamp(1,h-2)
        query_pixels=torch.full((len(pc.get_xyz),),-1.,device='cuda')
        query_pixels[pool[inside]]=(vv*w+u)[inside].float()
        diagnostic_bg=torch.cat([bg,bg.new_tensor([len(pc.get_xyz)]),query_pixels])
        renderer.GaussianRasterizationSettings=original.GaussianRasterizationSettings;renderer.GaussianRasterizer=original.GaussianRasterizer
        base=renderer.render(camera,pc,pipe,bg,0.,0.,0.,override_color=colors)
        renderer.GaussianRasterizationSettings=diagnostic.GaussianRasterizationSettings;renderer.GaussianRasterizer=PrefixRasterizer
        diag=renderer.render(camera,pc,pipe,diagnostic_bg,0.,0.,0.,override_color=colors)
        prefix=latest['prefix'].clone()
        checks={k:float((base[k]-diag[k]).abs().max()) for k in ('render','rend_alpha','rend_normal','surf_depth','radii','rend_dist')}
        if frame in (1,81,149):
            pipe.depth_ratio=0.
            renderer.GaussianRasterizationSettings=original.GaussianRasterizationSettings;renderer.GaussianRasterizer=original.GaussianRasterizer
            expected=renderer.render(camera,pc,pipe,bg,0.,0.,0.,override_color=colors)['surf_depth']
            renderer.GaussianRasterizationSettings=diagnostic.GaussianRasterizationSettings;renderer.GaussianRasterizer=PrefixRasterizer
            expected_diag=renderer.render(camera,pc,pipe,diagnostic_bg,0.,0.,0.,override_color=colors)['surf_depth']
            checks['expected_depth']=float((expected-expected_diag).abs().max());pipe.depth_ratio=1.
        assert max(checks.values())==0.,checks
        h,w=base['rend_alpha'].shape[-2:]
        alpha=camera.gt_alpha_mask.cuda()[0];interior=-F.max_pool2d(-alpha[None,None],3,stride=1,padding=1)[0,0]
        mask=(interior>.99)&(base['rend_alpha'][0]>.9)
        clip=homogeneous@camera.full_proj_transform;ndc=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
        u=((ndc[:,0]+1)*w/2-.5).round().long();vv=((ndc[:,1]+1)*h/2-.5).round().long()
        inside=(clip[:,3]>0)&(u>=1)&(u<w-1)&(vv>=1)&(vv<h-1)
        u,vv=u.clamp(1,w-2),vv.clamp(1,h-2)
        center_valid=inside&mask[vv,u]
        actual=prefix[3,pool]>0;pixel_agree=prefix[3,pool]==(vv*w+u+1)
        assert not (actual&~pixel_agree).any(),'实际贡献与指定采样像素不一致'
        center_front=center_valid&actual&pixel_agree&(prefix[0,pool]>.5)&(prefix[1,pool]>=1/255)&(prefix[2,pool]>0)
        z=(homogeneous@camera.world_view_transform)[:,2]
        record=dict(center_valid=center_valid.cpu(),center_front=center_front.cpu(),actual_contribution=actual.cpu(),pixel_agreement=pixel_agree.cpu(),
                    T_before=prefix[0,pool].cpu(),alpha=prefix[1,pool].cpu(),weight=prefix[2,pool].cpu(),center_median_error=(z-base['surf_depth'][0,vv,u]).cpu())
        records.append(record);frames.append(frame);parity.append(dict(frame=frame,checks=checks))
        if frame in (1,81,149):torch.save(dict(prefix=prefix.cpu(),valid=mask.cpu(),median_depth=base['surf_depth'].cpu(),alpha=base['rend_alpha'].cpu()),a.output/f'fit_{frame:04d}_primary.pt')
        print(json.dumps(dict(frame=frame,pool=len(pool),actual=int(actual.sum()),center_front=int(center_front.sum()),pixel_disagree=int((actual&inside&~pixel_agree).sum()),parity_max=max(checks.values()))),flush=True)
        del camera,base,diag
    tensors={key:torch.stack([r[key] for r in records]) for key in records[0]}
    tensors.update(pool_ids=pool.cpu(),frames=frames,points=points.cpu(),normal=normal[pool].cpu(),
                   point_opacity=pc.get_opacity[pool].cpu(),height=((points-origin)@plane).cpu())
    torch.save(tensors,a.output/'primary_evidence.pt')
    v.write(a.output/'render_parity.json',dict(status='PASS',frames=parity))
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__));hashes[str(original_source/'cuda_rasterizer/forward.cu')]=v.sha(original_source/'cuda_rasterizer/forward.cu')
    assert hashes[str(original_source/'cuda_rasterizer/forward.cu')]==v.read(a.output/'instrumentation.json')['original_forward_sha256']
    v.check_hashes(hashes)
    v.write(a.output/'summary.json',dict(status='PASS_PRIMARY_PREFIX_DIAGNOSTIC_NOT_LIGHT_ACCEPTANCE',gt_read=False,external_test_read=False,
       fit_frames=frames,held_used=False,pool_count=len(pool),internal_validation_passed=False,primary_evidence_sha256=v.sha(a.output/'primary_evidence.pt'),
       center_front_views_histogram=torch.bincount(tensors['center_front'].sum(0),minlength=118).tolist(),input_sha256=hashes,
       next_step='仅证据缓存；未挑选新接收集合、未重拟合或导出。'))


if __name__=='__main__':main()
