"""将静态T13→T01→T03→T06→F-T04训练路线适配到动态未知方向光。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from scene import GaussianModel, DeformModel
from scene.dataset_readers import fetchPly, readCamerasFromTransforms, getNerfppNorm
from scene.stage2_photometric import DirectionalLightTable, srgb_to_linear, linear_to_srgb
from scene.static_dl_censored import censored_photometric_loss
from scene.static_dl_normal_opacity import geometry_losses
from LIHAN.scripts.dynamic_v3_surface import EffectiveSurface, FullDynamicVisibility, load_train_cameras, sh2_basis
from gaussian_renderer import render as raster
from utils.general_utils import build_rotation
from utils.sh_utils import RGB2SH


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


class Lighting(torch.nn.Module):
    def __init__(self, data, environment, scale):
        super().__init__()
        self.table=DirectionalLightTable(torch.tensor(data['times'],device='cuda'),
            torch.tensor(data['directions'],device='cuda'),learned=False)
        self.log_intensity=torch.nn.Parameter(torch.tensor(math.log(math.pi),device='cuda'))
        self.coefficients=torch.nn.Parameter(torch.zeros(9,3,device='cuda'),requires_grad=environment)
        self.environment_enabled=environment
        self.register_buffer('environment_scale',torch.tensor(float(scale),device='cuda'))

    def environment(self,n):
        if not self.environment_enabled:return torch.zeros_like(n)
        field=sh2_basis(n)@(.5*self.coefficients.tanh())
        return self.environment_scale*F.softplus(field+math.log(math.expm1(1.)))


class DynamicSurfaceModel:
    """完整路线的独立可重载模型；几何与物理着色使用同一当前形变。"""
    def effective(self,camera):
        values=self.deform.step(self.pc.get_xyz,camera.fid.reshape(1,1).expand(len(self.pc.get_xyz),1),
            iteration=self.iteration,feature=self.pc.get_binary_feature(eval=True),
            camera_center=camera.camera_center,is_training=False)
        return EffectiveSurface(self.pc,values)

    def render(self,camera,components=False):
        geometry=self.effective(camera)
        direction=self.lighting.table(camera.fid)
        normal=build_rotation(geometry.get_rotation)[:,:,2]
        sign=torch.where((normal*(camera.camera_center-geometry.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.).detach()
        normal=normal*sign
        cosine=(normal*direction).sum(-1,keepdim=True).clamp_min(0)
        visible=self.visibility(geometry,direction[None]) if self.iteration>=5000 else torch.ones_like(cosine)
        albedo=srgb_to_linear(self.pc.get_albedo)
        direct=albedo/math.pi*self.lighting.log_intensity.exp()*visible*cosine
        environment=albedo/math.pi*self.lighting.environment(normal)
        zero=self.pc.get_xyz.new_zeros(3)
        package=raster(camera,geometry,self.pipe,zero,0.,0.,0.,override_color=direct+environment)
        package['render_linear']=package['render']
        package['render']=linear_to_srgb(package['render_linear'].clamp_min(0))
        package['effective_geometry']=geometry
        package['mean_visibility']=visible.mean().detach()
        if components:
            for key,color in (('direct_linear',direct),('environment_linear',environment)):
                package[key]=raster(camera,geometry,self.pipe,zero,0.,0.,0.,override_color=color)['render']
        return package

    @classmethod
    def load(cls,folder,step):
        folder=Path(folder)
        receipt=json.loads((folder/f'dynamic_full_route/iteration_{step}/receipt.json').read_text())
        for relative,digest in receipt['checkpoint_sha256'].items():assert sha(folder/relative)==digest
        state=torch.load(folder/f'dynamic_full_route/iteration_{step}/state.pt',map_location='cpu',weights_only=False)
        obj=cls();obj.iteration=step;obj.config=state['config']
        obj.pc=GaussianModel(3,no_binary_separation=True,fea_dim=1)
        obj.pc.load_ply(str(folder/f'point_cloud/iteration_{step}/point_cloud.ply'));obj.pc.active_sh_degree=3
        obj.deform=DeformModel('mlp',is_blender=True,hyper_dim=1,pred_color=False)
        assert obj.deform.load_weights(str(folder),step)
        obj.deform.deform.eval().requires_grad_(False)
        obj.lighting=Lighting(state['lighting_source'],state['environment_enabled'],state['environment_scale'])
        obj.lighting.load_state_dict(state['lighting']);obj.lighting.eval().requires_grad_(False)
        obj.visibility=FullDynamicVisibility(.03)
        obj.pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=1.)
        return obj


def train(args):
    from argparse import ArgumentParser
    from arguments import OptimizationParams
    from utils.loss_utils import ssim
    from utils.no_gt_guard import install_no_gt_guard
    import diff_surfel_rasterization as backend
    out=args.experiment.resolve();plan=json.loads((out/'plan.json').read_text())
    assert not (out/'train_audit.jsonl').exists()
    source=Path(plan['source']);stats_path=source.parent/'input_audit_v1/train_statistics.json'
    light_path=out/'lighting.json';light=json.loads(light_path.read_text())
    install_no_gt_guard([source,source.parent/'input_audit_v1',out])
    torch.set_num_threads(4);random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    stats=json.loads(stats_path.read_text());manifest=json.loads((source/'no_gt_manifest.json').read_text())
    cameras=load_train_cameras(source); assert len(cameras)==117
    infos=readCamerasFromTransforms(str(source),'transforms_train.json','images',False,no_bg=False)
    extent=float(getNerfppNorm(infos)['radius'])
    opt=OptimizationParams(ArgumentParser());opt.densify_until_iter=20000
    model=DynamicSurfaceModel();model.iteration=1
    model.pc=GaussianModel(3,no_binary_separation=True,fea_dim=1)
    model.pc.create_from_pcd(fetchPly(str(source/'points3d.ply')),extent)
    model.pc.camera_extent=extent
    # 动态适配：关闭时变颜色，保留位移/旋转及共享主干；不冻结形变几何。
    model.deform=DeformModel('mlp',is_blender=True,hyper_dim=1,pred_color=False)
    environment=plan['dataset']=='v3'
    scale=math.pi*min(stats['mean_linear_luminance'],stats['q_base'])/.5
    model.lighting=Lighting(light,environment,scale)
    model.visibility=FullDynamicVisibility(.03)
    model.pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=1.)
    cfg=dict(sh_degree=3,source_path=str(source),model_path=str(out/'model_mlp'),images='images',
        train_light_folder='images',test_light_folder='images',resolution=2,white_background=False,
        data_device='cpu',eval=True,load2gpu_on_the_fly=True,is_blender=True,deform_type='mlp',hyper_dim=1,
        pred_color=False,no_binary_separation=True,load_test_set_only=False,start_frame=0,end_frame=-1,depth_ratio=1.)
    folder=out/'model_mlp';folder.mkdir(exist_ok=False)
    (folder/'cfg_args').write_text('Namespace('+', '.join(k+'='+repr(v) for k,v in cfg.items())+')')
    model.config=cfg
    write(out/'implementation_binding.json',dict(gt_read=False,source=str(source),source_sha256=sha(source/'no_gt_manifest.json'),
        lighting_sha256=sha(light_path),initial_points=len(model.pc.get_xyz),camera_extent=extent,
        rasterizer=backend.provenance,route=plan,code_sha256={str(Path(__file__)):sha(__file__)}))
    optimizer=None;stack=[];started=time.time();events=[]
    def optimizers():
        model.pc.training_setup(opt);model.deform.train_setting(opt)
        return torch.optim.Adam([p for p in model.lighting.parameters() if p.requires_grad],lr=.001)
    optimizer=optimizers()
    def checkpoint(step):
        ply=folder/f'point_cloud/iteration_{step}/point_cloud.ply';model.pc.save_ply(str(ply))
        model.deform.save_weights(str(folder),step)
        state_dir=folder/f'dynamic_full_route/iteration_{step}';state_dir.mkdir(parents=True,exist_ok=False)
        torch.save(dict(config=cfg,iteration=step,actual_optimizer_updates=step,lighting=model.lighting.state_dict(),
            lighting_source=light,environment_enabled=environment,environment_scale=scale,
            gaussian_optimizer=model.pc.optimizer.state_dict(),deform_optimizer=model.deform.optimizer.state_dict(),
            light_optimizer=optimizer.state_dict(),xyz_gradient_accum=model.pc.xyz_gradient_accum,
            denom=model.pc.denom,max_radii2D=model.pc.max_radii2D,events=events,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),
            state_dir/'state.pt')
        paths=[ply,folder/f'deform/iteration_{step}/deform.pth',state_dir/'state.pt']
        write(state_dir/'receipt.json',dict(iteration=step,actual_optimizer_updates=step,gt_read=False,
            checkpoint_sha256={str(p.relative_to(folder)):sha(p) for p in paths}))
    try:
        for iteration in range(1,18001):
            model.iteration=iteration
            if iteration in (10001,12001,14001,16001):
                optimizer=optimizers();stack=[];random.seed(1012);torch.manual_seed(1012)
                events.append(dict(iteration=iteration,event='Adam及增密累计清零，保留模型参数'))
            model.pc.update_learning_rate(iteration)
            for group in model.deform.optimizer.param_groups:group['lr']=model.deform.deform_scheduler_args(iteration)
            if iteration%1000==0:model.pc.oneupSHdegree()
            if not stack:stack=list(range(len(cameras)))
            camera=cameras[stack.pop(random.randrange(len(stack)))];camera.load2device()
            zero=model.pc.get_xyz.new_zeros(3)
            alpha=camera.gt_alpha_mask.cuda();target=linear_to_srgb(camera.observed_linear.cuda())
            if iteration==1001:
                with torch.no_grad():model.pc._albedo_dc.fill_(float(RGB2SH(.5)));model.pc._albedo_rest.zero_()
                events.append(dict(iteration=iteration,event='切换物理着色，反照率重置0.5，D初始化pi'))
            if iteration<=1000:
                if iteration<1000:
                    package=raster(camera,model.pc,model.pipe,zero,0.,0.,0.)
                else:
                    geometry=model.effective(camera)
                    package=raster(camera,geometry,model.pipe,zero,0.,0.,0.)
                photo=.8*(package['render']-target).abs().mean()+.2*(1-ssim(package['render'],target))
                alpha_loss=F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6,1-1e-6),alpha)
                normal=dist=photo.new_zeros(());geometry_package=None
            else:
                package=model.render(camera)
                photo=censored_photometric_loss(package['render'],target,alpha)[0]
                route='detach_opacity' if iteration>=15001 else 'full'
                losses,geometry_package=geometry_losses(camera,package['effective_geometry'],model.pipe,zero,alpha,package,route)
                alpha_loss=losses['alpha'];normal=losses['normal'] if iteration>2000 else photo.new_zeros(())
                dist=losses['distortion'] if iteration>8000 else photo.new_zeros(())
                if iteration<=2000:geometry_package=None
            # 动态适配保留原生位移微弱正则；静态模型的该项恒为零。
            if iteration<1000:motion=photo.new_zeros(())
            else:
                current_geometry=package['effective_geometry'] if 'effective_geometry' in package else model.effective(camera)
                motion=.001*current_geometry.values['d_xyz'].square().mean()
            loss=photo+.1*alpha_loss+.1*normal+1000*dist+motion
            if not torch.isfinite(loss):raise RuntimeError('训练loss非有限')
            loss.backward()
            with torch.no_grad():
                selected=package['visibility_filter']
                model.pc.max_radii2D[selected]=torch.maximum(model.pc.max_radii2D[selected],package['radii'][selected])
                screen=package['viewspace_points']
                if geometry_package is not None and geometry_package['viewspace_points'].grad is not None:
                    screen.grad.add_(geometry_package['viewspace_points'].grad)
                model.pc.add_densification_stats(screen,selected)
                model.pc.optimizer.step();model.deform.optimizer.step();optimizer.step()
                model.pc.optimizer.zero_grad(set_to_none=True);model.deform.optimizer.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
                if iteration>1000:
                    model.pc._albedo_dc.clamp_(float(RGB2SH(.031)),float(RGB2SH(.969)))
                    model.lighting.log_intensity.clamp_(-5,5)
                before=len(model.pc.get_xyz)
                if iteration<18000 and iteration>500 and iteration%100==0:
                    model.pc.densify_and_prune(.0002,.01,extent,20 if iteration>3000 else None)
                    events.append(dict(iteration=iteration,event='增密裁剪',before=before,after=len(model.pc.get_xyz)))
                if iteration<=15000 and iteration%3000==0:
                    model.pc.reset_opacity();events.append(dict(iteration=iteration,event='opacity reset'))
                if iteration in (16000,16500,18000):checkpoint(iteration)
                if iteration==1 or iteration%100==0 or iteration in (1001,2001,5000,8001,10001,12001,14001,15001,16001):
                    row=dict(iteration=iteration,actual_optimizer_updates=iteration,photo=float(photo),alpha=float(alpha_loss),
                        normal=float(normal),distortion=float(dist),motion=float(motion),loss=float(loss),points=len(model.pc.get_xyz),
                        intensity=float(model.lighting.log_intensity.exp()),mean_visibility=float(package.get('mean_visibility',torch.tensor(1.))),
                        xyz_lr=model.pc.optimizer.param_groups[0]['lr'],elapsed_seconds=time.time()-started)
                    with (out/'train_audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
                    write(out/'progress.json',dict(status='TRAINING',**row))
            camera.load2device('cpu')
        write(out/'topology_reset_events.json',events)
        marker=dict(format_version=1,route='dynamic_full_route_v1',gt_supervision=False,external_test_read=False,
            train_frames=manifest['train_frames'],dataset=plan['dataset'],actual_optimizer_updates=18000,
            checkpoint_iterations=[16000,16500,18000],fresh_random_initialization=True,config=cfg,rasterizer=backend.provenance)
        write(folder/'dynamic_full_route.json',marker)
        write(out/'training_finished.json',dict(status='COMPLETED',actual_optimizer_updates=18000,elapsed_seconds=time.time()-started,
            checkpoint_iterations=[16000,16500,18000],gt_read=False))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAILED',iteration=model.iteration,error=repr(exc),all_outputs_retained=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True)
    train(parser.parse_args())
