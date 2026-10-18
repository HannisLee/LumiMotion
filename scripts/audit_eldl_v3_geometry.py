"""在保存模型上以真实参数中心差分验证混合光度梯度，固定DL可见度。"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.static_dl_photometric import StaticVisibility
from scene.eldl_v3_photometric import FixedELDLLights,render_photometric,censored_symmetric_loss
from utils.camera_utils import loadCam


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--iteration',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry-model',type=Path)
    p.add_argument('--geometry-iteration',type=int)
    p.add_argument('--image-vjp',action='store_true',help='一次封顶定位：固定基线线性图像余切，仅XYZ/scale')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    marker=json.loads((a.model/'eldl_v3_training.json').read_text());cfg=SimpleNamespace(**marker['args'])
    assert not marker['gt_supervision'] and marker['lighting_frozen']
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    geometry_model=a.geometry_model or a.model
    geometry_iteration=a.geometry_iteration or a.iteration
    if a.geometry_model:
        gm=json.loads((geometry_model/'eldl_v3_training.json').read_text())
        assert not gm['gt_supervision'] and not gm['init_weights_loaded'] and gm['baseline']=='static_sh'
    ply=geometry_model/f'point_cloud/iteration_{geometry_iteration}/point_cloud.ply';pc.load_ply(str(ply))
    lights=FixedELDLLights(cfg.eldl_lighting)
    info=readCamerasFromTransforms(cfg.source_path,'transforms_train.json','images',False)[0]
    cam=loadCam(cfg,0,info,1.);cam.load2device();bg=torch.zeros(3,device='cuda')
    tracer=StaticVisibility(cfg.static_dl_shadow_bias,1)
    with torch.no_grad():cached=tracer(pc,lights.for_camera(cam)[0],a.iteration).detach()
    def fixed_visibility(*_):return cached
    def forward():
        package=render_photometric(cam,pc,cfg,bg,lights,fixed_visibility,a.iteration)
        loss=censored_symmetric_loss(package['render'],cam.original_image_train_light,
                                     cam.gt_alpha_mask.cuda(),cam.observed_saturation_mask)[0]
        return package,loss
    def loss():return forward()[1]
    def double_loss(linear):
        from scene.stage2_photometric import linear_to_srgb
        srgb=linear_to_srgb(linear.double().clamp_min(0))
        return censored_symmetric_loss(srgb,cam.original_image_train_light.double(),
                                       cam.gt_alpha_mask.cuda(),cam.observed_saturation_mask)[0]
    def branches():
        from utils.general_utils import build_rotation
        n=build_rotation(pc.get_rotation)[:,:,2]
        facing=(n*(cam.camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0
        oriented=n*torch.where(facing,1.,-1.)
        return facing,(oriented@lights.for_camera(cam)[0].T)>0
    names=('xyz','scaling') if a.image_vjp else ('xyz','rotation','scaling')
    params=[pc._xyz,pc._scaling] if a.image_vjp else [pc._xyz,pc._rotation,pc._scaling]
    originals=[param.detach().clone() for param in params]
    base,value=forward()
    all_grads=torch.autograd.grad(value,params+[base['render_linear']],allow_unused=False)
    grads,image_cotangent=all_grads[:-1],all_grads[-1].detach().double()
    with torch.no_grad():base_branches=branches()
    records={}
    for name,param,original,grad in zip(names,params,originals,grads):
        assert torch.isfinite(grad).all() and grad.norm()>0
        direction=grad/grad.abs().max()
        derivative=float((grad*direction).sum())
        steps=[]
        for eps in (1e-3,3e-4,1e-4):
            with torch.no_grad():
                param.copy_(original+eps*direction);actual_plus=param.detach().clone();pp,pl=forward();plus=float(pl)
                bp=branches()
                param.copy_(original-eps*direction);actual_minus=param.detach().clone();pm,ml=forward();minus=float(ml)
                bm=branches()
                param.copy_(original)
            fd=(plus-minus)/(2*eps)
            steps.append(dict(step=eps,plus=plus,minus=minus,finite_difference=fd,
                              relative_error=abs(fd-derivative)/max(abs(fd),abs(derivative),1e-12),
                              gradient_descent_reduces_actual_loss=minus<float(value)))
            if a.image_vjp:
                image_vjp=float((image_cotangent*(pp['render_linear'].double()-pm['render_linear'].double())).sum()/(2*eps))
                scalar64=float((double_loss(pp['render_linear'])-double_loss(pm['render_linear']))/(2*eps))
                actual_vjp=float((grad.double()*(actual_plus.double()-actual_minus.double())).sum()/(2*eps))
                steps[-1].update(image_linear_vjp_float64=image_vjp,scalar_loss_finite_difference_float64=scalar64,
                    autograd_actual_perturbation_vjp=actual_vjp,
                    image_vs_scalar_relative=abs(image_vjp-scalar64)/max(abs(image_vjp),abs(scalar64),1e-12),
                    facing_changes_plus=int((bp[0]!=base_branches[0]).sum()),facing_changes_minus=int((bm[0]!=base_branches[0]).sum()),
                    cosine_branch_changes_plus=int((bp[1]!=base_branches[1]).sum()),cosine_branch_changes_minus=int((bm[1]!=base_branches[1]).sum()))
        passed=sum(r['relative_error']<.15 and r['gradient_descent_reduces_actual_loss'] for r in steps)>=2
        records[name]=dict(gradient_norm=float(grad.norm()),autograd_directional_derivative=derivative,
                           finite_difference_steps=steps,passed=passed)
    unchanged=all(torch.equal(p,v) for p,v in zip(params,originals))
    result=dict(status='PASS' if unchanged and all(r['passed'] for r in records.values()) else 'FAILED',
                camera=cam.image_name_train_light,base_loss=float(value),parameters=records,
                parameters_restored_exact=unchanged,model_file=str(ply.resolve()),
                model_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),updates=0,gt_read=False,
                geometry_model_override=a.geometry_model is not None,
                image_vjp_localization=a.image_vjp,
                visibility_frozen_for_finite_difference=True,
                scope='验证当前停止可见度梯度的光度子问题；不声称BVH边界/重新追踪遮挡梯度可微')
    (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
