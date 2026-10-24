"""从原生动态父模型继续几何优化；只允许净化的TRAIN GT方向，保持ELDL和动态形变。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from LIHAN.scripts.dynamic_v3_train import common_geometry,reset_opacity
from scene.stage2_photometric import linear_to_srgb
from scene.static_dl_censored import censored_photometric_loss
from utils.loss_utils import ssim
from utils.general_utils import get_expon_lr_func
from utils.no_gt_guard import install_no_gt_guard


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def objective(package,camera,recipe,normal_weight=.1):
    prediction=package['render']
    linear_target=camera.observed_linear.to(prediction.device).clamp_min(0)
    target=linear_to_srgb(linear_target)
    # 保留AREA后仍精确满码的通道，避免float32 gamma回转的1ulp破坏历史精确满码规则。
    target=torch.where(linear_target==1.,torch.ones_like(target),target)
    alpha=camera.gt_alpha_mask.to(prediction.device)
    if recipe=='historical':
        rgb,diagnostic=censored_photometric_loss(prediction,target,alpha,.25,.5,.08)
        alpha_loss=F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6,1-1e-6),alpha)
        l1=(prediction-target).abs().mean();dssim=prediction.new_zeros(())
    else:
        l1=(prediction-target).abs().mean();dssim=1-ssim(prediction,target)
        rgb=.8*l1+.2*dssim
        alpha_loss=(package['rend_alpha']-alpha).abs().mean();diagnostic={}
    normal,distortion=common_geometry(package,camera)
    loss=rgb+.1*alpha_loss+normal_weight*normal+1000*distortion
    return loss,dict(rgb=rgb,l1=l1,dssim=dssim,alpha=alpha_loss,normal=normal,distortion=distortion),diagnostic


def main(args):
    root=Path(os.environ.get('DYNAMIC_V3_REPO_ROOT',Path(__file__).resolve().parents[2])).resolve()
    source,parent,out=args.source.resolve(),args.parent.resolve(),args.output.resolve()
    assert out.is_relative_to(root/'output') and args.smoke==('smoke_test' in out.parts)
    out.mkdir(parents=True,exist_ok=True);model_dir=out/'model_mlp';assert not model_dir.exists()
    manifest=json.loads((source/'no_gt_manifest.json').read_text());lighting=json.loads(args.lighting.read_text());stats=json.loads(args.stats.read_text())
    assert lighting['source_split']=='TRAIN' and lighting['gt_direction_only']
    assert lighting['forbidden_gt_fields_included']==[] and lighting['evaluation_direction_source']=='heldout_GT_direction_evaluation_only'
    assert set(lighting)=={'format_version','source_split','gt_direction_only','forbidden_gt_fields_included','evaluation_direction_source','parent','parent_iteration','train_frames','times','directions','intensity','source_sha256','direction_audit'}
    assert args.iterations>args.adaptation_steps>=0
    assert np.allclose(np.linalg.norm(lighting['directions'],axis=1),1.,atol=1e-6)
    assert np.allclose(lighting['times'],(np.array(lighting['train_frames'])-1)/149,atol=1e-7)
    assert lighting['intensity']==math.pi
    assert lighting['train_frames']==manifest['train_frames'] and len(manifest['train_frames'])==117
    assert Path(lighting['parent']).resolve()==parent and lighting['parent_iteration']==35000
    for p,digest in lighting['source_sha256'].items():assert sha(p)==digest,p
    for rel,digest in manifest['files'].items():assert sha(source/rel)==digest,rel
    install_no_gt_guard([source,parent,out,args.lighting.resolve(),args.stats.resolve()])
    random.seed(1019);np.random.seed(1019);torch.manual_seed(1019);torch.cuda.manual_seed_all(1019);torch.set_num_threads(4)
    inputs=[source/'no_gt_manifest.json',source.parent/'input_audit_v1/radiometry_v1.json',args.lighting.resolve(),args.stats.resolve(),
            parent/'point_cloud/iteration_35000/point_cloud.ply',parent/'deform/iteration_35000/deform.pth',Path(__file__),
            Path(__file__).with_name('dynamic_v3_surface.py'),Path(__file__).with_name('dynamic_v3_train.py'),
            Path(__file__).resolve().parents[2]/'scene/static_dl_censored.py',Path(__file__).resolve().parents[2]/'utils/loss_utils.py']
    bound={str(p):sha(p) for p in inputs}
    model=DynamicSurfaceModel(parent,35000,dict(times=torch.tensor(lighting['times']),directions=torch.tensor(lighting['directions']),intensity=lighting['intensity']),stats,'A')
    parent_tensors={name:getattr(model.pc,name).detach().clone() for name in ['_xyz','_rotation','_scaling','_opacity']}
    parent_deform={name:value.detach().clone() for name,value in model.deform.deform.state_dict().items()}
    cameras=load_train_cameras(source);orientation=model.initialize_orientation(cameras)
    assert all(torch.equal(getattr(model.pc,name),value) for name,value in parent_tensors.items())
    assert all(torch.equal(model.deform.deform.state_dict()[name],value) for name,value in parent_deform.items())
    write(out/'parent_initialization_audit.json',dict(status='PASS',parent_iteration=35000,
        gaussian_geometry_bitwise_equal=True,deformation_state_bitwise_equal=True,
        orientation_modifies_geometry=False,optimizer_restored=False))
    del parent_tensors,parent_deform
    if args.smoke:cameras=[c for c in cameras if int(c.image_name_train_light.split('_')[-1]) in [1,21,41,81,101,121,141,149]]
    config=dict(iterations=20 if args.smoke else args.iterations,
        geometry_start=5 if args.smoke else args.adaptation_steps+1,
        deform_start=9 if args.smoke else args.adaptation_steps+1,
        reset_steps=[],protect_steps=0,
        saves=[0,4,12,20] if args.smoke else sorted(set([0,args.adaptation_steps,args.iterations//2,args.iterations])),recipe=args.recipe,
        normal_weight=args.normal_weight,distortion_weight=1000.,alpha_weight=.1,lambda_dssim=.2,
        rgb_loss='historical_srgb_censored_robust' if args.recipe=='historical' else 'srgb_0.8L1_0.2DSSIM',
        alpha_loss='BCE' if args.recipe=='historical' else 'L1',
        geometry_lrs=dict(xyz=args.xyz_lr,rotation=args.rotation_lr,scale=args.scale_lr,opacity=args.opacity_lr),
        normal_opacity_route='detach_from_first_step',ps_weight=0.,local_stability_weight=0.,direction_lr=0.,
        environment_sh_degree=2,native_sh_degree=model.cfg.sh_degree,depth_ratio=0.,seed=1019,denser_or_split=False,
        deform_lr=args.deform_lr,appearance_lr=args.appearance_lr,freeze_geometry=args.freeze_geometry,
        gt_direction_only=True,evaluation_direction_source='heldout_GT_direction_evaluation_only')
    model.config=config;model_dir.mkdir();(model_dir/'cfg_args').write_text((parent/'cfg_args').read_text())
    record=dict(status='RUNNING',path='A',ps_enabled=False,config=config,parent=str(parent),parent_iteration=35000,
        source=str(source),lighting=str(args.lighting.resolve()),stats=str(args.stats.resolve()),
        train_frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras],
        gt_supervision=['TRAIN_direction_only'],gt_geometry_supervision=False,external_test_read=False,parent_optimizer_loaded=False,orientation=orientation,
        smoke=args.smoke,actual_optimizer_updates=0,source_sha256=bound,initialization='同N00原生几何/形变；灰rho/零SH；冻结TRAIN GT方向，强度初始化pi后从TRAIN RGB学习',
        historical_reference='静态G2-T06只参考loss/路由，不加载其权重，动态形变保留')
    write(model_dir/'dynamic_v3_stage2.json',record);write(out/'plan.json',dict(config=config,argv=sys.argv))
    names=['xyz','rotation','scale','opacity','rho','D','SH']
    parameters=[model.pc._xyz,model.pc._rotation,model.pc._scaling,model.pc._opacity,model.radiance.raw_rho,model.radiance.raw_intensity,model.radiance.environment_coefficients]
    for p in parameters:p.requires_grad_(True)
    groups=[dict(params=[p],name=n,lr=args.appearance_lr if n in ['rho','D','SH'] else 0.) for n,p in zip(names,parameters)]
    deformation=[p for n,p in model.deform.deform.named_parameters() if not n.startswith('gaussian_color')]
    groups.append(dict(params=deformation,name='deformation',lr=0.))
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    initial_direction=model.radiance.lights.raw_directions.detach().clone()
    def checkpoint(step):
        receipt=model.checkpoint(model_dir,step,optimizer,config)
        receipt.update(gt_read=True,gt_fields=['TRAIN_direction_only'],gt_geometry_supervision=False)
        write(model_dir/f'dynamic_v3/iteration_{step}/receipt.json',receipt)
    checkpoint(0);stack=[];resets=[];start=time.monotonic()
    for step in range(1,config['iterations']+1):
        if step==config['deform_start']:
            for p in deformation:p.requires_grad_(True)
        if not stack:stack=cameras.copy()
        camera=stack.pop(random.randrange(len(stack)));camera.load2device()
        recovery=any(0<step-event<=config['protect_steps'] for event in config['reset_steps'])
        protect=True
        for group in optimizer.param_groups:
            name=group['name']
            if name in config['geometry_lrs']:
                progress=max(0,step-config['geometry_start'])/max(1,config['iterations']-config['geometry_start'])
                group['lr']=0. if args.freeze_geometry or step<config['geometry_start'] else config['geometry_lrs'][name]*(1-.9*progress)
                if name=='opacity' and recovery:group['lr']=.05
            elif name=='SH':group['lr']=args.appearance_lr
            elif name=='deformation':group['lr']=0. if args.freeze_geometry or step<config['deform_start'] else args.deform_lr
        direct_weight=.5
        optimizer.zero_grad(set_to_none=True)
        log_components=step in [1,config['geometry_start'],config['deform_start'],config['iterations']] or step%100==0 or step in config['saves']
        package=model.render(camera,direct_weight=direct_weight,protect_opacity=protect,components=log_components)
        loss,terms,diagnostic=objective(package,camera,args.recipe,args.normal_weight);assert torch.isfinite(loss)
        gradients={}
        audit=step in [1,config['geometry_start'],config['deform_start'],config['iterations']]+[s+1 for s in config['reset_steps']]
        if audit:
            for name,value in [('rgb',terms['rgb']),('alpha',terms['alpha']),('normal',terms['normal']),('distortion',terms['distortion'])]:
                gs=torch.autograd.grad(value,parameters[:4],retain_graph=True,allow_unused=True)
                gradients[name]={n:None if g is None else dict(norm=float(g.norm()),finite=bool(torch.isfinite(g).all())) for n,g in zip(names[:4],gs)}
                if protect and name=='normal':assert gs[-1] is None or float(gs[-1].abs().max())==0.
        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:assert p.grad is None or torch.isfinite(p.grad).all(),group['name']
        optimizer.step();assert torch.equal(initial_direction,model.radiance.lights.raw_directions)
        if step in config['reset_steps']:
            event=reset_opacity(model,optimizer);event['step']=step;resets.append(event);write(out/'opacity_resets.json',resets)
        if audit or step%100==0 or step in config['saves']:
            row=dict(step=step,actual_updates=step,elapsed_seconds=time.monotonic()-start,loss=float(loss.detach()),
                terms={k:float(v.detach()) for k,v in terms.items()},gradient_audit=gradients,
                normal_opacity_protected=protect,recovery=recovery,direct_weight=direct_weight,
                intensity=float(model.radiance.intensity.detach()),rho_median=float(model.radiance.rho.detach()[model.radiance.gauge_support].median()),
                gaussians=len(model.pc.get_xyz),lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                foreground_alpha_coverage=float(((camera.gt_alpha_mask.cuda()>.5)&(package['rend_alpha']>.5)).sum()/(camera.gt_alpha_mask.cuda()>.5).sum().clamp_min(1)),
                diagnostics={k:float(v) for k,v in diagnostic.items()},gpu_memory_peak_bytes=torch.cuda.max_memory_allocated(),
                environment_image_mean=float(package['environment_linear'].detach().mean()),
                weighted_direct_image_mean=float((direct_weight*package['direct_linear']).detach().mean()),
                environment_to_direct_ratio=float(package['environment_linear'].detach().mean()/(direct_weight*package['direct_linear'].detach().mean()).clamp_min(1e-12)),
                environment_coefficients_abs_max=float(model.radiance.environment_coefficients.detach().abs().max()),
                component_sum_max_error=float(package['component_sum_max_error']))
            with (out/'train_audit.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(row,ensure_ascii=False),flush=True)
        camera.load2device('cpu');del loss,terms,package
        if step in config['saves']:checkpoint(step)
    if args.smoke:
        reloaded=DynamicSurfaceModel.load(model_dir,config['iterations']);checks=[]
        with torch.no_grad():
            for camera in [c for c in cameras if int(c.image_name_train_light.split('_')[-1]) in [1,81,149]]:
                camera.load2device();a=model.render(camera);b=reloaded.render(camera)
                difference={k:float((a[k]-b[k]).abs().max()) for k in ['render_linear','rend_alpha','rend_normal','surf_depth']}
                assert all(v<=1e-5 for v in difference.values()),difference
                checks.append(dict(frame=camera.image_name_train_light,max_abs=difference));camera.load2device('cpu')
        write(out/'smoke_reload.json',dict(status='PASS_SAVED_FORWARD_RELOAD',checks=checks,optimizer_steps=0))
    for p,digest in bound.items():assert sha(p)==digest,p
    record.update(status='TRAINING_FINISHED_PENDING_EVALUATION',actual_optimizer_updates=config['iterations'],
        cumulative_actual_updates=34999+config['iterations'],reset_events=resets,
        checkpoint_sha256=json.loads((model_dir/f'dynamic_v3/iteration_{config["iterations"]}/receipt.json').read_text())['checkpoint_sha256'])
    write(model_dir/'dynamic_v3_stage2.json',record);write(model_dir/'training_finished.json',record);write(out/'training_finished.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','parent','lighting','stats','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recipe',choices=['image','historical'],default='historical');p.add_argument('--smoke',action='store_true')
    p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--adaptation-steps',type=int,default=500)
    p.add_argument('--normal-weight',type=float,default=.1)
    p.add_argument('--xyz-lr',type=float,default=1.6e-5)
    p.add_argument('--rotation-lr',type=float,default=.001)
    p.add_argument('--scale-lr',type=float,default=.001)
    p.add_argument('--opacity-lr',type=float,default=.0005)
    p.add_argument('--deform-lr',type=float,default=2e-6)
    p.add_argument('--appearance-lr',type=float,default=.005)
    p.add_argument('--freeze-geometry',action='store_true')
    main(p.parse_args())
