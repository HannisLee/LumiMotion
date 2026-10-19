"""同一原生Stage1起点的A0/A1/B0/B1固定预算训练，无TEST/GT选模。"""
import argparse
import copy
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
from LIHAN.scripts.dynamic_v3_tracks import pick_surface_ids,build_observations,build_ps_targets,ps_geometry_loss
from utils.no_gt_guard import install_no_gt_guard
from utils.general_utils import get_expon_lr_func


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def photo_objective(package,camera,stats):
    prediction=package['render_linear'];target=camera.observed_linear.to(prediction.device)
    alpha=camera.gt_alpha_mask.to(prediction.device)
    saturation=camera.observed_saturation_mask.to(prediction.device)
    difference=prediction-target
    difference=torch.where(saturation,difference.clamp_max(0),difference)
    error=(difference.square()+.02**2).sqrt()-.02
    foreground=(alpha>.5).float();background=1-foreground
    full=(error*foreground).sum()/(3*foreground.sum()).clamp_min(1)
    outside=(error*background).sum()/(3*background.sum()).clamp_min(1)
    luminance=(target*target.new_tensor([.2126,.7152,.0722])[:,None,None]).sum(0,keepdim=True)
    priority=torch.sigmoid((luminance-stats['q_bright'])/.02)
    bright=(error*foreground*priority).sum()/(3*foreground.sum()).clamp_min(1)
    alpha_loss=F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6,1-1e-6),alpha)
    return full+.1*outside+.1*bright+.1*alpha_loss


def common_geometry(package,camera):
    geometry=package.get('geometry_for_normal',package)
    normal=F.normalize(geometry['rend_normal'],dim=0)
    depth=F.normalize(geometry['surf_normal'],dim=0)
    alpha=camera.gt_alpha_mask.to(normal.device)
    interior=-F.max_pool2d(-(alpha>.99).float()[None],5,stride=1,padding=2)[0]
    mask=interior*(geometry['rend_alpha'].detach()>.5)
    consistency=((1-(normal*depth).sum(0,keepdim=True))*mask).sum()/mask.sum().clamp_min(1)
    distortion=package['rend_dist'].mean()
    return consistency,distortion


def reset_opacity(model,optimizer):
    with torch.no_grad():
        before=float(model.pc.get_opacity.median())
        opacity=model.pc.get_opacity.clamp(max=.01).clamp(1e-7,1-1e-7)
        model.pc._opacity.copy_(torch.logit(opacity))
        state=optimizer.state.get(model.pc._opacity,{})
        for key in ('exp_avg','exp_avg_sq'):
            if key in state:state[key].zero_()
        return dict(before_median=before,after_median=float(model.pc.get_opacity.median()),
                    moments_zeroed=True,step_counter_retained=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--parent-iteration',type=int,default=35000)
    parser.add_argument('--lighting',type=Path,required=True);parser.add_argument('--stats',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--path',choices=['A','B'],required=True)
    parser.add_argument('--ps',action='store_true');parser.add_argument('--direction-lr',type=float,choices=[0.,1e-5],default=0.)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--recovery-smoke',action='store_true',help='仅工程：完整200步opacity保护恢复期，固定220更新')
    args=parser.parse_args()
    assert not args.recovery_smoke or args.smoke
    root=Path(os.environ.get('DYNAMIC_V3_REPO_ROOT',Path(__file__).resolve().parents[2])).resolve()
    source,parent,out=args.source.resolve(),args.parent.resolve(),args.output.resolve()
    assert out.is_relative_to(root/'output') and args.smoke==('smoke_test' in out.parts)
    assert args.parent_iteration==35000 or args.smoke
    out.mkdir(parents=True,exist_ok=True);model_dir=out/'model_mlp'
    assert not model_dir.exists(),'禁止覆盖模型，失败重试必须新目录'
    manifest=json.loads((source/'no_gt_manifest.json').read_text())
    lighting=json.loads(args.lighting.read_text());stats=json.loads(args.stats.read_text())
    assert manifest['dataset']=='LH-dynamic-jumpingjacks-ELDL-V3'
    assert lighting['source_split']=='TRAIN' and not lighting['gt_read']
    assert lighting['smoke']==args.smoke and lighting['parent_iteration']==args.parent_iteration
    if not args.smoke:assert lighting['train_frames']==manifest['train_frames']
    for name,digest in lighting['source_sha256'].items():assert sha(name)==digest,name
    bound_inputs=[source/'no_gt_manifest.json',source.parent/'input_audit_v1/radiometry_v1.json',
        parent/f'point_cloud/iteration_{args.parent_iteration}/point_cloud.ply',
        parent/f'deform/iteration_{args.parent_iteration}/deform.pth',args.stats.resolve()]
    for path in bound_inputs:
        assert lighting['source_sha256'].get(str(path))==sha(path),'估光与训练入口资产不一致: '+str(path)
    assert Path(lighting['parent']).resolve()==parent and Path(lighting['source']).resolve()==source
    for name,digest in manifest['files'].items():assert sha(source/name)==digest,name
    install_no_gt_guard([source,parent,out,args.lighting.resolve(),args.stats.resolve()])
    random.seed(1019);np.random.seed(1019);torch.manual_seed(1019);torch.cuda.manual_seed_all(1019)
    torch.set_num_threads(4)
    light_input=dict(times=torch.tensor(lighting['times']),directions=torch.tensor(lighting['directions']),intensity=lighting['intensity'])
    model=DynamicSurfaceModel(parent,args.parent_iteration,light_input,stats,args.path,allow_smoke_parent=args.smoke)
    cameras=load_train_cameras(source)
    if args.smoke:cameras=[c for c in cameras if int(c.image_name_train_light.split('_')[-1]) in lighting['train_frames']]
    orientation=model.initialize_orientation(cameras)
    ids=pick_surface_ids(model,128 if args.smoke else 2048)
    model_dir.mkdir(exist_ok=False)
    (model_dir/'cfg_args').write_text((parent/'cfg_args').read_text())
    config=dict(iterations=20 if args.smoke else 8100,base_end=2 if args.smoke else 540,
        geometry_start=9 if args.smoke else 1351,deform_start=13 if args.smoke else 4051,
        refresh=[8,12,16] if args.smoke else [1350,4050,5400,6750],
        saves=[0,8,12,20] if args.smoke else [0,1350,4050,8100],
        reset_steps=[10] if args.smoke else [3000,6000],protect_steps=2 if args.smoke else 200,
        opacity_recovery_lr=.05,
        ps_weight=.02 if args.ps else 0.,ps_steps=50 if args.smoke else 200,
        lambda_dl=.5,normal_weight=.02,distortion_weight=1000.,alpha_weight=.1,
        geometry_lrs=dict(xyz=1.6e-6,rotation=1e-4,scale=2e-4,opacity=5e-4),
        appearance_lr=.001,deform_lr_fraction=.05,direction_lr=args.direction_lr,
        direction_anchor=.1,direction_smooth=.01,local_stability_weight=.001,
        environment_sh_degree=2,native_sh_degree=model.cfg.sh_degree,depth_ratio=0.,denser_or_split=False,seed=1019)
    if args.recovery_smoke:
        config.update(iterations=220,refresh=[8,208,216],saves=[0,8,208,220],protect_steps=200)
    model.config=config
    record=dict(status='RUNNING',path=args.path,ps_enabled=args.ps,config=config,parent=str(parent),
        parent_iteration=args.parent_iteration,source=str(source),lighting=str(args.lighting.resolve()),stats=str(args.stats.resolve()),
        train_frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras],
        gt_supervision=False,external_test_read=False,parent_optimizer_loaded=False,
        initialization='同一原生PLY/MLP；物理rho灰色、SH零系数；共同新TRAIN估计Lt/D；无旧数据权重',
        orientation=orientation,smoke=args.smoke,actual_optimizer_updates=0,
        source_sha256={str(p):sha(p) for p in [source/'no_gt_manifest.json',source.parent/'input_audit_v1/radiometry_v1.json',args.lighting,args.stats,Path(__file__),
            Path(__file__).with_name('dynamic_v3_surface.py'),Path(__file__).with_name('dynamic_v3_tracks.py'),Path(__file__).with_name('dynamic_v3_ps_math.py')]})
    write(model_dir/'dynamic_v3_stage2.json',record)
    write(out/'plan.json',dict(argv=sys.argv,config=config,command=shlex.join(sys.argv)))
    geometry_params=[('xyz',model.pc._xyz),('rotation',model.pc._rotation),('scale',model.pc._scaling),('opacity',model.pc._opacity)]
    for _,p in geometry_params:p.requires_grad_(True)
    groups=[dict(params=[p],lr=0.,name=name) for name,p in geometry_params]
    groups.extend([dict(params=[model.radiance.raw_rho],lr=.001,name='rho'),
                   dict(params=[model.radiance.raw_intensity],lr=.001,name='D')])
    if args.path=='A':groups.append(dict(params=[model.radiance.environment_coefficients],lr=.001,name='SH'))
    else:model.radiance.environment_coefficients.requires_grad_(False)
    deformation=[p for name,p in model.deform.deform.named_parameters() if not name.startswith('gaussian_color')]
    groups.append(dict(params=deformation,lr=0.,name='deformation'))
    model.radiance.lights.raw_directions.requires_grad_(args.direction_lr>0)
    if args.direction_lr:groups.append(dict(params=[model.radiance.lights.raw_directions],lr=0.,name='light'))
    optimizer=torch.optim.Adam(groups,eps=1e-15)
    original_direction=model.radiance.lights.raw_directions.detach().clone()
    xyz_anchor=model.pc.get_xyz.detach().clone()
    parent_deform=copy.deepcopy(model.deform.deform).eval().requires_grad_(False)
    extent=float((torch.quantile(xyz_anchor,.99,dim=0)-torch.quantile(xyz_anchor,.01,dim=0)).norm())
    from scipy.spatial import cKDTree
    neighbors=cKDTree(xyz_anchor.cpu().numpy()).query(xyz_anchor[ids].cpu().numpy(),k=5)[1][:,1:]
    neighbors=torch.as_tensor(neighbors,device=xyz_anchor.device)
    reference_edges=(xyz_anchor[neighbors]-xyz_anchor[ids,None]).norm(dim=-1).clamp_min(1e-5)
    native_deform_lr=get_expon_lr_func(lr_init=.0008,lr_final=.0000016,lr_delay_mult=.01,max_steps=40000)
    targets=None;ps_reference=None;stack=[];start=time.monotonic();reset_events=[]
    model.checkpoint(model_dir,0,optimizer,config)
    for step in range(1,config['iterations']+1):
        if step==config['deform_start']:
            for p in deformation:p.requires_grad_(True)
        if not stack:stack=cameras.copy()
        camera=stack.pop(random.randrange(len(stack)));camera.load2device()
        protect=any(0<step-event<=config['protect_steps'] for event in config['reset_steps'])
        for group in optimizer.param_groups:
            name=group['name']
            if name in config['geometry_lrs']:
                progress=max(0,step-config['geometry_start'])/max(1,config['iterations']-config['geometry_start'])
                group['lr']=0. if step<config['geometry_start'] else config['geometry_lrs'][name]*(1-.9*progress)
                if name=='opacity' and protect:group['lr']=config['opacity_recovery_lr']
            elif name=='SH':group['lr']=.001 if step<=config['base_end'] else 0.
            elif name=='deformation':group['lr']=0. if step<config['deform_start'] else .05*native_deform_lr(args.parent_iteration-1+step)
            elif name=='light':group['lr']=0. if step<config['geometry_start'] else args.direction_lr
        direct_weight=0. if args.path=='A' and step<=config['base_end'] else .5
        optimizer.zero_grad(set_to_none=True)
        package=model.render(camera,direct_weight=direct_weight,protect_opacity=protect)
        photo=photo_objective(package,camera,stats)
        normal,distortion=common_geometry(package,camera)
        edge=(model.pc.get_xyz[neighbors]-model.pc.get_xyz[ids,None]).norm(dim=-1).clamp_min(1e-5)
        local=(torch.log(edge/reference_edges).clamp(-2,2).square()).mean()
        local=local+((model.pc.get_xyz-xyz_anchor)/max(extent,1e-6)).square().mean()
        if step>=config['deform_start']:
            geom=package['surface_components']['geometry']
            with torch.no_grad():
                original=parent_deform(model.pc.get_xyz.detach(),camera.fid.reshape(1,1).expand(len(xyz_anchor),1),
                    feature=model.pc.get_binary_feature(),camera_center=camera.camera_center)
            local=local+(geom.values['d_xyz']-original['d_xyz']).square().mean()/max(extent**2,1e-8)
        ps=photo.new_zeros(());ps_info=dict(active_tracks=0,weight_sum=0.)
        if targets is not None and step>=config['geometry_start']:
            ps,ps_info=ps_geometry_loss(model,camera,package,targets)
        light_regular=photo.new_zeros(())
        if args.direction_lr:
            directions=F.normalize(model.radiance.lights.raw_directions,dim=-1)
            delta=directions-F.normalize(original_direction,dim=-1)
            light_regular=.1*delta.square().sum(-1).mean()+.01*(delta[1:]-delta[:-1]).square().sum(-1).mean()
        loss=photo+.02*normal+1000*distortion+.001*local+config['ps_weight']*ps+light_regular
        assert torch.isfinite(loss),'训练损失非有限'
        gradient_row={}
        audit=step in (1,config['geometry_start'],config['deform_start'],config['iterations']) or step%100==0
        audit |= any(step in (event+1,event+config['protect_steps'],event+config['protect_steps']+1)
                     for event in config['reset_steps'])
        if audit and args.ps and ps.requires_grad:
            gradients=torch.autograd.grad(ps,[p for _,p in geometry_params],retain_graph=True,allow_unused=True)
            gradient_row={name:None if g is None else dict(norm=float(g.norm()),finite=bool(torch.isfinite(g).all()))
                for (name,_),g in zip(geometry_params,gradients)}
        opacity_gradients={}
        if audit and protect:
            for name,objective in [('normal',normal),('photo',photo)]:
                g=torch.autograd.grad(objective,model.pc._opacity,retain_graph=True,allow_unused=True)[0]
                opacity_gradients[name]=None if g is None else dict(norm=float(g.norm()),finite=bool(torch.isfinite(g).all()))
                if name=='normal':assert g is None or float(g.abs().max())==0.
        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:
                assert p.grad is None or torch.isfinite(p.grad).all(),group['name']+'梯度非有限'
        optimizer.step()
        if args.direction_lr:
            with torch.no_grad():model.radiance.lights.raw_directions.copy_(F.normalize(model.radiance.lights.raw_directions,dim=-1))
        if step in config['reset_steps']:
            event=reset_opacity(model,optimizer);event['step']=step;reset_events.append(event)
            write(out/'opacity_resets.json',reset_events)
        if audit or step in config['refresh'] or step in config['reset_steps']:
            row=dict(step=step,actual_updates=step,cumulative_actual_updates=args.parent_iteration-1+step,elapsed_seconds=time.monotonic()-start,
                loss=float(loss.detach()),photo=float(photo.detach()),normal=float(normal.detach()),distortion=float(distortion.detach()),
                ps=float(ps.detach()),ps_weight=config['ps_weight'],ps_diagnostics=ps_info,ps_geometry_gradients=gradient_row,
                recovery_opacity_gradients=opacity_gradients,
                foreground_alpha_coverage=float(((camera.gt_alpha_mask.cuda()>=.5)&(package['rend_alpha']>.5)).sum()/(camera.gt_alpha_mask.cuda()>=.5).sum().clamp_min(1)),
                local_stability=float(local.detach()),normal_opacity_protected=protect,
                intensity=float(model.radiance.intensity.detach()),rho_median=float(model.radiance.rho.detach()[model.radiance.gauge_support].median()),
                gaussians=len(model.pc.get_xyz),lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                gpu_memory_peak_bytes=torch.cuda.max_memory_allocated())
            with (out/'train_audit.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(row,ensure_ascii=False),flush=True)
        camera.load2device('cpu')
        del package,loss,photo,normal,distortion,ps
        if step in config['refresh']:
            observations=build_observations(model,cameras,out/f'ps_refresh_{step}/observations',ids=ids,transport=True)
            targets=build_ps_targets(model,cameras,observations,out/f'ps_refresh_{step}/targets',steps=config['ps_steps'])
            ps_reference=str(out/f'ps_refresh_{step}/targets/targets.pt')
            write(out/f'ps_refresh_{step}/schedule_receipt.json',dict(step=step,PS_enabled=args.ps,
                same_schedule_when_disabled=True,reliable_tracks=int(targets['reliable'].sum())))
        if step in config['saves']:model.checkpoint(model_dir,step,optimizer,config,ps_reference)
    record.update(status='TRAINING_FINISHED_PENDING_DEV_AND_EVALUATION',actual_optimizer_updates=config['iterations'],
        cumulative_actual_updates=args.parent_iteration-1+config['iterations'],reset_events=reset_events,
        checkpoint_sha256=json.loads((model_dir/f'dynamic_v3/iteration_{config["iterations"]}/receipt.json').read_text())['checkpoint_sha256'])
    write(model_dir/'dynamic_v3_stage2.json',record)
    write(model_dir/'training_finished.json',record)
    write(out/'training_finished.json',record)


if __name__=='__main__':main()
