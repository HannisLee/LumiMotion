"""仅TRAIN验证物理检查点独立重载、分量、真实几何梯度与配对初始前缀。"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel, load_train_cameras
from LIHAN.scripts.dynamic_v3_train import common_geometry, photo_objective


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_image(path, tensor, normal=False):
    value=tensor.detach().float().cpu().numpy()
    if value.ndim==3:value=value.transpose(1,2,0)
    if normal:value=(value+1)/2
    value=np.nan_to_num(value).clip(0,1)
    if value.ndim==3 and value.shape[-1]==3:value=value[...,::-1]
    cv2.imwrite(str(path),np.rint(value*255).astype(np.uint8))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--stats',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--peer',type=Path)
    parser.add_argument('--prefix-step',type=int,default=8)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert 'smoke_test' in out.parts,'本入口只做TRAIN工程冒烟'
    torch.set_num_threads(4)
    cameras=load_train_cameras(args.source,frame_ids=[1,81,149])
    stats=json.loads(args.stats.read_text())
    model=DynamicSurfaceModel.load(args.model,args.step)
    reference={};rows=[]
    for camera in cameras:
        camera.load2device()
        with torch.no_grad():
            package=model.render(camera,components=True)
            repeated=model.render(camera)
            assert torch.equal(package['render_linear'],repeated['render_linear'])
            assert float(package['component_sum_max_error'])<1e-5
            name=camera.image_name_train_light
            reference[name]={k:package[k].detach().cpu() for k in ['render_linear','rend_alpha','rend_normal','surf_depth','surf_normal']}
            for label,key in [('RGB','render'),('Alpha','rend_alpha'),('GS法线','rend_normal'),('Expected法线','surf_normal'),('EL','environment_linear'),('DL','direct_linear')]:
                save_image(out/f'{name}-{label}.png',package[key],normal='法线' in label)
            depth=package['surf_depth'];mask=package['rend_alpha']>.5
            depth_scale=float(torch.quantile(depth[mask],.99)) if mask.any() else 1.
            save_image(out/f'{name}-Expected深度.png',depth/max(depth_scale,1e-6))
            rows.append(dict(frame=name,repeat_bitwise=True,component_sum_max_error=float(package['component_sum_max_error']),
                alpha_coverage=float(mask.float().mean()),linear_rgb_finite=bool(torch.isfinite(package['render_linear']).all())))
        camera.load2device('cpu')
    del model,package,repeated
    torch.cuda.empty_cache()
    model=DynamicSurfaceModel.load(args.model,args.step)
    reload_errors={}
    for camera in cameras:
        camera.load2device()
        with torch.no_grad():
            package=model.render(camera)
            errors={k:float((package[k].cpu()-v).abs().max()) for k,v in reference[camera.image_name_train_light].items()}
            assert max(errors.values())==0.,errors
            reload_errors[camera.image_name_train_light]=errors
        camera.load2device('cpu')
    camera=cameras[0];camera.load2device()
    params=[model.pc._xyz,model.pc._rotation,model.pc._scaling,model.pc._opacity]
    names=['xyz','rotation','scale','opacity']
    for p in params:p.requires_grad_(True)
    gradients={}
    for protected in [False,True]:
        package=model.render(camera,protect_opacity=protected)
        consistency,_=common_geometry(package,camera)
        gs=torch.autograd.grad(consistency,params,retain_graph=True,allow_unused=True)
        gradients[str(protected)]={k:None if g is None else dict(norm=float(g.norm()),finite=bool(torch.isfinite(g).all())) for k,g in zip(names,gs)}
        if protected:
            assert gs[-1] is None or float(gs[-1].abs().max())==0.,'normal仍回传opacity'
            photo=photo_objective(package,camera,stats)
            g=torch.autograd.grad(photo,params[-1],allow_unused=True)[0]
            gradients['protected_photo_opacity']=dict(norm=float(g.norm()),finite=bool(torch.isfinite(g).all()))
            assert torch.isfinite(g).all() and g.norm()>0
    peer={}
    if args.peer:
        a=DynamicSurfaceModel.load(args.model,args.prefix_step)
        b=DynamicSurfaceModel.load(args.peer,args.prefix_step)
        for name in ['_xyz','_rotation','_scaling','_opacity','_features_dc','_features_rest','_feature']:
            av,bv=getattr(a.pc,name,None),getattr(b.pc,name,None)
            if isinstance(av,torch.Tensor):
                peer[name]=bool(torch.equal(av,bv));assert peer[name],name
        for name,av in a.deform.deform.state_dict().items():assert torch.equal(av,b.deform.deform.state_dict()[name]),name
        differences={name:float((av.double()-b.radiance.state_dict()[name].double()).abs().max())
                     for name,av in a.radiance.state_dict().items()}
        assert max(differences.values())<=1e-6,differences
        peer.update(deformation_bitwise=True,physical_state_bitwise=max(differences.values())==0.,
                    physical_state_max_errors=differences,physical_tolerance=1e-6,prefix_step=args.prefix_step)
    report=dict(status='PASS_ENGINEERING_ONLY',gt_read=False,test_read=False,model=str(args.model.resolve()),step=args.step,
        checkpoint_receipt_sha256=sha(args.model/f'dynamic_v3/iteration_{args.step}/receipt.json'),
        frames=rows,reload_max_errors=reload_errors,normal_opacity_gradients=gradients,paired_prefix=peer,
        scope='仅TRAIN重载/求导/分量一致性；不证明几何准确率或动态PS收益')
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
