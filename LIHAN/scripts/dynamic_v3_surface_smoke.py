"""短预算Stage1上的A/B物理接口门禁；合成灯向仅供工程检查。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import save_image
from utils.general_utils import build_rotation
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,native_normal_operator,load_train_cameras


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert 'smoke_test' in out.parts
    out.mkdir(parents=True,exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([args.source,args.model,out])
    torch.set_num_threads(4)
    cameras=load_train_cameras(args.source,{1,81,149})
    lighting=dict(times=torch.tensor([float(c.fid) for c in cameras]),
        directions=F.normalize(torch.tensor([[.3,.2,1.]]).repeat(len(cameras),1),dim=-1),intensity=math.pi)
    # 禁止导出此合成灯向为正式初始化；正式灯向仍须TRAIN阴影估计。
    stats=dict(mean_linear_luminance=.5,q_base=.5)
    rows=[];alphas={}
    for path in ('A','B'):
        model=DynamicSurfaceModel(args.model,20,lighting,stats,path,allow_smoke_parent=True)
        orientation=model.initialize_orientation(cameras)
        for p in (model.pc._xyz,model.pc._rotation,model.pc._scaling,model.pc._opacity):p.requires_grad_(True)
        model.radiance.lights.raw_directions.requires_grad_(True)
        camera=cameras[0];camera.load2device()
        package=model.render(camera,components=True,protect_opacity=True)
        assert float(package['component_sum_max_error'])<2e-5
        target=camera.observed_linear.cuda()
        loss=(package['render_linear']-target).square().mean()+.1*(package['rend_alpha']-camera.gt_alpha_mask.cuda()).square().mean()
        names=['xyz','rotation','scale','opacity','rho','D','light']
        params=[model.pc._xyz,model.pc._rotation,model.pc._scaling,model.pc._opacity,
                model.radiance.raw_rho,model.radiance.raw_intensity,model.radiance.lights.raw_directions]
        values=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        grads={name:None if grad is None else dict(norm=float(grad.norm()),finite=bool(torch.isfinite(grad).all())) for name,grad in zip(names,values)}
        assert all(value and value['finite'] and value['norm']>0 for value in grads.values()),grads
        protected=package['geometry_for_normal']
        normal_loss=(1-(F.normalize(protected['rend_normal'],dim=0)*F.normalize(protected['surf_normal'],dim=0)).sum(0)).mean()
        opacity_gradient=torch.autograd.grad(normal_loss,model.pc._opacity,allow_unused=True)[0]
        assert opacity_gradient is None or not opacity_gradient.any()
        alphas[path]=package['rend_alpha'].detach().clone()
        for name,value in [('rgb',package['render']),('alpha',package['rend_alpha']),
            ('gs_normal',(F.normalize(package['rend_normal'],dim=0)*.5+.5)*package['rend_alpha']),
            ('depth_normal',(F.normalize(package['surf_normal'],dim=0)*.5+.5)*package['rend_alpha']),
            ('EL',package['environment_linear']),('DL',package['direct_linear'])]:
            save_image(out/f'{path}_{name}.png',value)
        if path=='B':
            assert not package['environment_linear'].any()
            assert torch.equal(package['render_linear'],package['direct_linear'])
        # 固定几何时原生加性四元数算子必须严格复现实际surfel法线。
        with torch.no_grad():
            ids=torch.arange(min(64,len(model.pc.get_xyz)),device='cuda')
            deltas=[];actual=[]
            for c in cameras:
                c.load2device();geometry=model.effective(c)
                deltas.append(geometry.values['d_rotation'][ids])
                actual.append(build_rotation(geometry.get_rotation)[ids,:,2]*model.normal_sign[ids])
                model.visibility.build(geometry)
                c.load2device('cpu')
            op=native_normal_operator(model.pc._rotation[ids],torch.stack(deltas),model.normal_sign[ids])
            canonical=build_rotation(model.pc.get_rotation)[ids,:,2]*model.normal_sign[ids]
            operator_error=float((op(canonical)-torch.stack(actual)).abs().max())
            assert operator_error<2e-5
            saved=model.pc._opacity.detach().clone()
            model.pc._opacity.fill_(math.log(.005/.995))
            cameras[0].load2device()
            model.visibility.build(model.effective(cameras[0]))
            model.pc._opacity.copy_(saved)
            cameras[0].load2device('cpu')
        rows.append(dict(path=path,orientation=orientation,gradients=grads,
            component_sum_max_error=float(package['component_sum_max_error'].detach()),
            protected_normal_opacity_gradient_absent=True,native_normal_operator_max_error=operator_error,
            full_scene_build_count=model.visibility.build_count,low_opacity_bvh_finite=True))
    assert torch.equal(alphas['A'],alphas['B'])
    result=dict(status='PASS_DYNAMIC_SURFACE_ENGINEERING',rows=rows,gt_read=False,
        synthetic_lighting_only=True,formal_initialization_asset=False,alpha_identical_across_paths=True,
        scope='20步未拟合父模型上的物理接口/全动态BVH/梯度/透明度保护；非正式A0A1B0B1或PS收益')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
