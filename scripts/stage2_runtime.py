"""共享 Stage2 来源加载、完整状态、训练与渲染接口。"""
from __future__ import annotations
import copy
import hashlib
import json
import random
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from scene import Scene, GaussianModel, DeformModel
from scene.stage2_photometric import (LambertianMaterial, DirectionalLightTable, frame_key,
                                     read_gt_directions, srgb_to_linear, unit_vector)
from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian
from scripts.loss_stage2 import build_loss_preset


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(pc, deform):
    digest=hashlib.sha256()
    tensors={name:value for name,value in vars(pc).items() if torch.is_tensor(value)}
    tensors.update({'deform.'+name:value for name,value in deform.deform.state_dict().items()})
    for name,value in sorted(tensors.items()):
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_dir(path, deform_type):
    path=Path(path).resolve()
    if path.is_dir():
        return path
    candidate=Path(str(path)+'_'+deform_type)
    if candidate.is_dir():
        return candidate
    raise ValueError('模型目录不存在：'+str(path))


class FrozenScene:
    def __init__(self, args, source_model, source_iteration):
        self.args=args
        self.source=model_dir(source_model,args.deform_type)
        if source_iteration < 1:
            raise ValueError('新模式必须显式指定正的 Stage1 iteration。')
        ply=self.source/'point_cloud'/('iteration_'+str(source_iteration))/'point_cloud.ply'
        deform_file=self.source/'deform'/('iteration_'+str(source_iteration))/'deform.pth'
        if not ply.is_file() or not deform_file.is_file():
            raise ValueError('Stage1 必须同时存在 PLY 与 deform checkpoint。')
        self.source_files={str(p):file_sha256(p) for p in (ply,deform_file)}
        dataset=copy.copy(args); dataset.model_path=str(self.source)
        self.pc=GaussianModel(args.sh_degree,args.no_binary_separation,fea_dim=args.hyper_dim)
        self.scene=Scene(dataset,self.pc,load_iteration=source_iteration,shuffle=False)
        self.deform=DeformModel(deform_type=args.deform_type,is_blender=args.is_blender,
                               hyper_dim=args.hyper_dim,pred_color=args.pred_color)
        if not self.deform.load_weights(str(self.source),source_iteration):
            raise ValueError('形变检查点加载失败。')
        for value in vars(self.pc).values():
            if isinstance(value,torch.nn.Parameter):value.requires_grad_(False)
        self.deform.deform.requires_grad_(False);self.deform.deform.eval()
        self.iteration=source_iteration
        self.background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
        self.center=self.pc.get_xyz.detach().mean(0)
        self.fingerprint=tensor_fingerprint(self.pc,self.deform)
        self.train_cameras=self.scene.getTrainCameras()
        self.test_cameras=self.scene.getTestCameras()
        if not self.train_cameras:raise ValueError('训练相机为空。')
        self.camera_by_key={frame_key(c.image_name_train_light):c for c in self.train_cameras+self.test_cameras}
        self.training_keys={frame_key(c.image_name_train_light) for c in self.train_cameras}
        if self.training_keys & {frame_key(c.image_name_train_light) for c in self.test_cameras}:
            raise ValueError('训练与测试帧号重叠；请启用 --eval。')

    def load_camera(self,camera):
        if self.args.load2gpu_on_the_fly:camera.load2device()

    def unload_camera(self,camera):
        if self.args.load2gpu_on_the_fly:camera.load2device('cpu')

    @torch.no_grad()
    def deformation(self,camera,time_override=None):
        fid=camera.fid if time_override is None else torch.as_tensor([time_override],device='cuda')
        time_input=fid.to('cuda').unsqueeze(0).expand(len(self.pc.get_xyz),-1)
        return self.deform.step(self.pc.get_xyz.detach(),time_input,iteration=self.iteration,
                               feature=self.pc.get_binary_feature(),camera_center=camera.camera_center,
                               is_training=False)

    def verify_frozen(self):
        if tensor_fingerprint(self.pc,self.deform)!=self.fingerprint:
            raise AssertionError('冻结的 Stage1 参数发生改变。')
        if any(file_sha256(p)!=sha for p,sha in self.source_files.items()):
            raise AssertionError('Stage1 来源文件发生改变。')


def make_light(context,args):
    if args.photometric_light_mode=='learned_directional':
        grouped={}
        for c in context.train_cameras:
            grouped.setdefault(float(c.fid.item()),[]).append(c.camera_center.to('cuda')-context.center)
        times=sorted(grouped)
        dirs=torch.stack([unit_vector(torch.stack(grouped[t]).mean(0)) for t in times])
        return DirectionalLightTable(times,dirs),times
    keys=sorted(context.camera_by_key,key=lambda k:float(context.camera_by_key[k].fid.item()))
    dirs=read_gt_directions(args.photometric_gt_lights_path,context.center,keys,'cuda')
    times=[float(context.camera_by_key[k].fid.item()) for k in keys]
    return DirectionalLightTable(times,dirs,learned=False),times


def supervision(camera,background):
    target=camera.original_image_train_light.to('cuda')
    alpha=camera.gt_alpha_mask
    if alpha is None:raise ValueError('新模式需要真实 RGBA alpha 监督。')
    alpha=alpha.to('cuda')
    # reader 已经使用 no_bg 合成黑/白背景，不再对 RGB 乘 alpha。
    return target,(alpha>0.5).float()


@torch.no_grad()
def calibrate_intensity(context,material,path):
    if not path:raise ValueError('自动标定需要 GT 灯光路径；否则请显式给出固定光强。')
    keys=[frame_key(c.image_name_train_light) for c in context.train_cameras]
    directions=read_gt_directions(path,context.center,keys,'cuda')
    xy=0.0;xx=0.0;pixels=0
    for idx,(camera,direction) in enumerate(zip(context.train_cameras,directions)):
        context.load_camera(camera)
        pkg=render_stage2_lambertian(camera,context.pc,material,context.deformation(camera),context.background,
                                    direction=direction,intensity=1.0)
        target,mask=supervision(camera,context.background)
        valid=(mask[0]>0)&(pkg['rend_alpha'][0]>0.9)&(pkg['shading'][0]>0.1)&(target.max(0).values<0.98)
        prediction=(pkg['base_color_linear']*pkg['shading']/np.pi)[:,valid].double()
        target_linear=srgb_to_linear(target)[:,valid].double()
        xy+=float((prediction*target_linear).sum());xx+=float(prediction.square().sum());pixels+=int(valid.sum())
        context.unload_camera(camera)
        if (idx+1)%25==0:print('calibration {}/{}'.format(idx+1,len(keys)),flush=True)
    if xx<=0 or xy<=0:raise ValueError('标定没有有效正响应像素。')
    value=xy/xx
    return value,{'intensity':value,'train_keys':keys,'valid_pixels':pixels,'gt_lights_sha256':file_sha256(path),
                  'method':'训练帧RGB、SH初始材质和固定几何法线的非负标量最小二乘；非物理强度恢复'}


def checkpoint_path(output,iteration):
    return Path(output)/'stage2_photometric'/('iteration_'+str(iteration))/'checkpoint.pth'


def load_photometric_run(output,iteration):
    checkpoint=checkpoint_path(output,iteration)
    if not checkpoint.is_file():raise ValueError('完整 Stage2 checkpoint 不存在：'+str(checkpoint))
    state=torch.load(checkpoint,map_location='cuda')
    if state.get('format_version')!=1:raise ValueError('不支持的 Stage2 checkpoint 格式。')
    args=Namespace(**state['config'])
    context=FrozenScene(args,args.stage1_model_path,state['source_iteration'])
    if context.source_files!=state['source_files'] or context.fingerprint!=state['frozen_fingerprint']:
        raise ValueError('来源模型与 Stage2 checkpoint 不匹配。')
    material=LambertianMaterial(context.pc.get_albedo.detach())
    light,_=make_light(context,args)
    material.load_state_dict(state['material']);light.load_state_dict(state['light'])
    return context,material,light,args,state


class PhotometricTrainer:
    def __init__(self,args):
        self.args=args;self.output=Path(args.model_path)
        self.output.mkdir(parents=True,exist_ok=True)
        if args.resume_iteration is None:
            if (self.output/'stage2_config.json').exists() or (self.output/'stage2_photometric').exists():
                raise ValueError('已有实验禁止覆盖；请使用新目录或 --resume_iteration。')
            self.context=FrozenScene(args,args.stage1_model_path,args.load_iter)
            self.material=LambertianMaterial(self.context.pc.get_albedo.detach())
            self.light,_=make_light(self.context,args)
            if args.photometric_light_intensity<=0:
                value,record=calibrate_intensity(self.context,self.material,args.photometric_gt_lights_path)
                args.photometric_light_intensity=value
                (self.output/'calibration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            if not np.isfinite(args.photometric_light_intensity) or args.photometric_light_intensity<=0:
                raise ValueError('固定光强必须为正有限值。')
            self.completed=self.context.iteration;self.stack=[];state=None
        else:
            if args.stage1_model_path:raise ValueError('恢复与首次来源初始化互斥。')
            context,material,light,saved,state=load_photometric_run(self.output,args.resume_iteration)
            allowed={'iterations','test_iterations','save_iterations','resume_iteration','quiet','model_path','photometric_log_interval'}
            # 首次明确提供的参数可用于核对；默认值不会覆盖 checkpoint 配置。
            import sys
            explicit={t[2:].split('=')[0].replace('-','_') for t in sys.argv[1:] if t.startswith('--')}
            for key in explicit-allowed-{'stage1_model_path'}:
                if hasattr(saved,key) and getattr(args,key)!=getattr(saved,key):
                    raise ValueError('恢复参数不兼容：'+key)
            for key in allowed:setattr(saved,key,getattr(args,key))
            self.args=args=saved;self.context=context;self.material=material;self.light=light
            self.completed=state['iteration'];self.stack=state['camera_stack']
        if args.iterations<=self.completed:raise ValueError('目标 iteration 必须大于已完成 iteration。')
        self.optimizer=torch.optim.Adam([
            {'params':[self.material.raw_albedo],'lr':args.photometric_albedo_lr,'name':'albedo'},
            *([{'params':[self.light.raw_directions],'lr':args.photometric_light_lr,'name':'light'}]
              if self.light.raw_directions.requires_grad else [])])
        self.preset=build_loss_preset(args,'photometric_lambertian')
        if state is not None:
            self.optimizer.load_state_dict(state['optimizer'])
            random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
            torch.set_rng_state(state['rng_torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
        config=vars(args).copy()
        (self.output/('stage2_config.json' if state is None else 'resume_{}_config.json'.format(self.completed))).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        (self.output/'cfg_args').write_text(str(Namespace(**config)))
        self.log=open(self.output/'losses.jsonl','a')
        if state is None:self.save()

    def save(self):
        self.context.verify_frozen()
        path=checkpoint_path(self.output,self.completed)
        if path.exists():raise ValueError('禁止覆盖已有 checkpoint：'+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        state={'format_version':1,'iteration':self.completed,'source_iteration':self.context.iteration,
               'source_files':self.context.source_files,'frozen_fingerprint':self.context.fingerprint,
               'config':vars(self.args).copy(),'material':self.material.state_dict(),'light':self.light.state_dict(),
               'optimizer':self.optimizer.state_dict(),'camera_stack':self.stack,
               'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
               'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
        torch.save(state,path)
        print('checkpoint',self.completed,path,flush=True)

    def train(self):
        start=time.monotonic();last=time.monotonic()
        while self.completed<self.args.iterations:
            if not self.stack:self.stack=list(range(len(self.context.train_cameras)))
            index=self.stack.pop(random.randrange(len(self.stack)))
            camera=self.context.train_cameras[index];self.context.load_camera(camera)
            direction=self.light(camera.fid)
            pkg=render_stage2_lambertian(camera,self.context.pc,self.material,self.context.deformation(camera),
                                        self.context.background,direction=direction,intensity=self.args.photometric_light_intensity)
            target,mask=supervision(camera,self.context.background)
            result=self.preset.compute(pkg,target,mask,self.material,self.light)
            if not torch.isfinite(result.total):raise FloatingPointError('Stage2 loss 非有限。')
            self.optimizer.zero_grad(set_to_none=True);result.total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in self.optimizer.param_groups for p in g['params']):
                raise FloatingPointError('Stage2 梯度非有限。')
            self.optimizer.step();self.completed+=1
            self.context.unload_camera(camera)
            if self.completed%self.args.photometric_log_interval==0 or self.completed==self.args.iterations:
                record={'iteration':self.completed,'total':float(result.total),
                        **{k:float(v) for k,v in result.terms.items()},'elapsed_seconds':time.monotonic()-start}
                self.log.write(json.dumps(record)+'\n');self.log.flush()
                print(record,flush=True);last=time.monotonic()
            if self.completed in self.args.save_iterations or self.completed==self.args.iterations:self.save()
        self.context.verify_frozen();self.log.close()
        (self.output/'TRAIN_COMPLETED').write_text(str(self.completed)+'\n')
