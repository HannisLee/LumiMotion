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
from scene.stage2_photometric import (LambertianMaterial, DirectionalLightTable, PointLightTable, AreaLightTable, frame_key,
                                     read_gt_directions, srgb_to_linear, unit_vector)
from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian
from scripts.loss_stage2 import build_loss_preset, is_geometry_preset
from scene.stage2_photometric import make_material
from scene.stage2_geometry import (GeometryResidual, geometry_schedule, geometry_losses, load_coherence_source_mask,
    geometry_training_phase, geometry_group_lr, rgb_normal_prior_loss)


def backward_geometry_objective(total, normal_term, geometry, args, prior_only=False):
    """可选地只向XYZ分配预测深度一致性梯度；默认保持旧反传路径。"""
    accumulate = args.geometry_accumulate
    weight = args.lambda_geometry_normal
    if prior_only or getattr(geometry, 'normal_gradient_mode', 'joint') != 'position' or weight == 0:
        (total/accumulate).backward()
        return
    # 零值修正保持total的前向数值，取消一致性对其他参数的梯度。
    xyz_gradient = torch.autograd.grad(weight*normal_term/accumulate,
                                       geometry.translation, retain_graph=True, allow_unused=True)[0]
    other = total + weight*(normal_term.detach()-normal_term)
    (other/accumulate).backward()
    if xyz_gradient is not None:
        if geometry.translation.grad is None:
            geometry.translation.grad = xyz_gradient
        else:
            geometry.translation.grad.add_(xyz_gradient)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def restore_source_prediction_caches(reference, visibility, cameras, include_shading):
    """仅恢复冻结来源预测的缓存；拒绝验证/测试帧或不匹配的字段与像素形状。"""
    expected={'depth','alpha'} | ({'shading'} if include_shading else set())
    restored=[]
    for cache,fields,dtype in [(reference,expected,torch.float32),(visibility,None,torch.float16)]:
        if not isinstance(cache,dict) or not set(cache).issubset(cameras):
            raise ValueError('源预测缓存包含非优化相机或格式错误。')
        result={}
        for key,entry in cache.items():
            values=entry if fields is not None else {'visibility':entry}
            if not isinstance(values,dict) or (fields is not None and set(values)!=fields):
                raise ValueError('源预测缓存的字段不匹配。')
            shape=(1,cameras[key].image_height,cameras[key].image_width)
            for name,value in values.items():
                if (not torch.is_tensor(value) or value.dtype!=dtype or tuple(value.shape)!=shape or
                        not torch.isfinite(value).all()):
                    raise ValueError('源预测缓存的dtype、形状或有限性不匹配。')
            result[key]={name:value.detach().cpu() for name,value in values.items()} if fields is not None else entry.detach().cpu()
        restored.append(result)
    return restored


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
        from scene.stage2_source_deformation import find_static_anchor_camera
        self.static_anchor_camera = find_static_anchor_camera(args, self.train_cameras, frame_key)
        self.static_deformation_cache = None

    def load_camera(self,camera):
        if self.args.load2gpu_on_the_fly:camera.load2device()

    def unload_camera(self,camera):
        if self.args.load2gpu_on_the_fly:camera.load2device('cpu')

    @torch.no_grad()
    def deformation(self,camera,time_override=None):
        if getattr(self, 'static_anchor_camera', None) is not None:
            if self.static_deformation_cache is not None:
                return self.static_deformation_cache
            camera = self.static_anchor_camera
            time_override = None
        fid=camera.fid if time_override is None else torch.as_tensor([time_override],device='cuda')
        time_input=fid.to('cuda').unsqueeze(0).expand(len(self.pc.get_xyz),-1)
        prediction = self.deform.step(self.pc.get_xyz.detach(),time_input,iteration=self.iteration,
                                     feature=self.pc.get_binary_feature(),camera_center=camera.camera_center,
                                     is_training=False)
        if getattr(self, 'static_anchor_camera', None) is not None:
            self.static_deformation_cache = prediction
        return prediction

    def verify_frozen(self):
        if tensor_fingerprint(self.pc,self.deform)!=self.fingerprint:
            raise AssertionError('冻结的 Stage1 参数发生改变。')
        if any(file_sha256(p)!=sha for p,sha in self.source_files.items()):
            raise AssertionError('Stage1 来源文件发生改变。')

    def direct_visibility(self,camera,material,light):
        if not hasattr(self,'visibility_cache'):
            from scene.stage2_visibility import ReferenceVisibility
            self.visibility_cache=ReferenceVisibility(self,self.args)
        return self.visibility_cache.get(camera,material,light)


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
    if getattr(args,'geometry_point_light',False):
        records=json.loads(Path(args.photometric_gt_lights_path).read_text())
        positions=torch.tensor([records[k]['light_pos_world'] for k in keys],device='cuda',dtype=torch.float32)
        if getattr(args,'geometry_area_emitter',False):
            return AreaLightTable(times,positions,context.center,[records[k] for k in keys],args.geometry_light_distance,args.geometry_ambient,
                                  getattr(args,'geometry_area_power_convention','legacy')),times
        return PointLightTable(times,positions,context.center,args.geometry_light_distance,args.geometry_ambient),times
    return DirectionalLightTable(times,dirs,learned=False),times


def supervision(camera,background):
    target=camera.original_image_train_light.to('cuda')
    alpha=camera.gt_alpha_mask
    if alpha is None:raise ValueError('新模式需要真实 RGBA alpha 监督。')
    alpha=alpha.to('cuda')
    # reader 已经使用 no_bg 合成黑/白背景，不再对 RGB 乘 alpha。
    return target,(alpha>0.5).float()


@torch.no_grad()
def calibrate_intensity(context,material,path,light=None):
    if not path:raise ValueError('自动标定需要 GT 灯光路径；否则请显式给出固定光强。')
    keys=[frame_key(c.image_name_train_light) for c in context.train_cameras]
    directions=read_gt_directions(path,context.center,keys,'cuda')
    xy=0.0;xx=0.0;pixels=0
    for idx,(camera,direction) in enumerate(zip(context.train_cameras,directions)):
        context.load_camera(camera)
        pkg=render_stage2_lambertian(camera,context.pc,material,context.deformation(camera),context.background,
                                    direction=direction,intensity=1.0,
                                    **(light.render_kwargs(camera.fid) if isinstance(light,PointLightTable) else {}))
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
    if state.get('format_version') not in {1,2}:raise ValueError('不支持的 Stage2 checkpoint 格式。')
    if state.get('format_version')==2 and state.get('geometry') is None:
        raise ValueError('v2 checkpoint 缺少几何残差，禁止退回原Stage1几何。')
    if state['iteration']!=iteration:raise ValueError('checkpoint iteration 不匹配。')
    args=Namespace(**state['config'])
    context=FrozenScene(args,args.stage1_model_path,state['source_iteration'])
    if context.source_files!=state['source_files'] or context.fingerprint!=state['frozen_fingerprint']:
        raise ValueError('来源模型与 Stage2 checkpoint 不匹配。')
    material=make_material(context.pc.get_albedo.detach(),args,state['material'])
    light,_=make_light(context,args)
    light.load_state_dict(state['light'])
    if state.get('geometry') is not None:
        context.pc.stage2_geometry=GeometryResidual(context.pc.get_xyz,args)
        context.pc.stage2_geometry.load_state_dict(state['geometry'])
    return context,material,light,args,state


class PhotometricTrainer:
    def __init__(self,args):
        self.args=args;self.output=Path(args.model_path)
        if args.resume_iteration is None:
            source=model_dir(args.stage1_model_path,args.deform_type)
            if self.output.resolve()==source or source in self.output.resolve().parents:
                raise ValueError('Stage2 输出必须位于只读 Stage1 来源目录之外。')
            if self.output.exists() and any(self.output.iterdir()):
                raise ValueError('已有实验禁止覆盖；请使用新目录或 --resume_iteration。')
            self.output.mkdir(parents=True,exist_ok=True)
            self.context=FrozenScene(args,args.stage1_model_path,args.load_iter)
            self.material=make_material(self.context.pc.get_albedo.detach(),args)
            self.light,_=make_light(self.context,args)
            if args.photometric_light_intensity<=0 and isinstance(self.light,AreaLightTable):
                args.photometric_light_intensity=self.light.reference_intensity
                (self.output/'calibration.json').write_text(json.dumps({'method':'Known rectangle light power; four-point area quadrature',
                    'power_convention':self.light.power_convention,
                    'intensity':args.photometric_light_intensity,'metadata_sha256':file_sha256(args.photometric_gt_lights_path),
                    'gt_surface_used':False},indent=2)+'\n')
            if args.photometric_light_intensity<=0:
                if getattr(args,'geometry_tonemap','srgb')=='filmic' or getattr(args,'geometry_ambient',0)>0:
                    raise ValueError('Filmic/非零环境项需要显式固定光强；不使用旧线性最小二乘标定。')
                value,record=calibrate_intensity(self.context,self.material,args.photometric_gt_lights_path,self.light)
                args.photometric_light_intensity=value
                (self.output/'calibration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            if not np.isfinite(args.photometric_light_intensity) or args.photometric_light_intensity<=0:
                raise ValueError('固定光强必须为正有限值。')
            self.completed=self.context.iteration;self.stack=[];state=None
        else:
            if args.stage1_model_path:raise ValueError('恢复与首次来源初始化互斥。')
            context,material,light,saved,state=load_photometric_run(self.output,args.resume_iteration)
            if state.get('diagnostic'):raise ValueError('只读诊断checkpoint没有匹配优化器，不能续训。')
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
        self.geometry=None
        self.reference_cache={}
        self.geometry_schedule_origin=self.context.iteration
        initial_model=getattr(args,'geometry_initial_stage2_model','')
        if initial_model:
            if not is_geometry_preset(args.loss_preset):raise ValueError('Stage2几何初始化仅支持几何preset。')
            if args.geometry_initial_stage2_iteration<self.context.iteration:
                raise ValueError('Stage2初始化iteration不能早于Stage1来源。')
            self.geometry_schedule_origin=args.geometry_initial_stage2_iteration
            if state is not None and state.get('geometry_schedule_origin')!=self.geometry_schedule_origin:
                raise ValueError('Stage2初始化恢复缺少匹配的独立调度起点。')
        self.train_indices=list(range(len(self.context.train_cameras)))
        if is_geometry_preset(args.loss_preset):
            rgb_warmup = getattr(args, 'geometry_rgb_prior_warmup', 0)
            if rgb_warmup > 0 and not initial_model:
                raise ValueError('RGB预热需要真实Stage2初始化。')
            if args.geometry_accumulate<1 or args.geometry_validation_stride<2:
                raise ValueError('几何梯度累积至少1，验证划分间隔至少2。')
            if min(args.geometry_xyz_lr,args.geometry_rotation_lr)<0 or not 0<args.geometry_final_lr_ratio<=1:
                raise ValueError('几何学习率必须非负，衰减比必须在 (0,1]。')
            if getattr(args,'geometry_optimize_scale',False) and (
                not np.isfinite(args.geometry_scale_lr) or args.geometry_scale_lr<0 or args.geometry_scale_start<0):
                raise ValueError('尺寸学习率和起始步数必须非负，学习率必须有限。')
            if getattr(args,'geometry_optimize_opacity',False) and (
                not np.isfinite(args.geometry_opacity_lr) or args.geometry_opacity_lr<0 or
                args.geometry_opacity_start<0 or not np.isfinite(args.lambda_geometry_opacity_anchor) or
                args.lambda_geometry_opacity_anchor<0):
                raise ValueError('不透明度学习率、起始步数与锚定权重必须有限且非负。')
            if not 0<=getattr(args,'geometry_photo_visibility_threshold',0)<1 or getattr(args,'geometry_photo_min_shading',0)<0:
                raise ValueError('光度可见度阈值需在[0,1)，参考辐照度阈值需非负。')
            if getattr(args,'geometry_photo_visibility_threshold',0)>0 and not getattr(args,'geometry_shadows',False):
                raise ValueError('光度可见度筛选需要开启参考阴影。')
            if state is None:
                self.context.pc.stage2_geometry=GeometryResidual(self.context.pc.get_xyz,args)
            self.geometry=self.context.pc.stage2_geometry
            self.train_indices=[i for i in self.train_indices if i%args.geometry_validation_stride!=0]
            initialization_sha=None
            if state is None and initial_model:
                from scene.stage2_rgb_prior import validate_stage2_initialization, initialize_geometry_from_stage2
                init_path=checkpoint_path(model_dir(initial_model,args.deform_type),args.geometry_initial_stage2_iteration)
                init=torch.load(init_path,map_location='cpu')
                validate_stage2_initialization(init,args,self.context.source_files,self.context.fingerprint)
                if hasattr(self.material,'labels'):
                    torch.testing.assert_close(self.material.labels.cpu(),init['material']['labels'],rtol=0,atol=0)
                # 已知灯光必须精确相同，不能借初始化偷偷改变成像模型。
                if self.light.state_dict().keys()!=init['light'].keys():raise ValueError('初始化灯光类型不匹配。')
                for key,value in self.light.state_dict().items():
                    torch.testing.assert_close(value.cpu(),init['light'][key],rtol=0,atol=0)
                initialization_buffers=initialize_geometry_from_stage2(self.geometry,init,
                    refined_coherence=getattr(args,'geometry_refined_coherence',False) and self.geometry.coherence_enabled)
                if (self.geometry.translation.norm(dim=-1).max()>self.geometry.scene_extent*self.geometry.max_displacement+1e-6 or
                        self.geometry.axis_angle.norm(dim=-1).max()>self.geometry.max_angle+1e-6):
                    raise ValueError('初始化几何超出当前信赖域。')
                self.material.load_state_dict(init['material'])
                self.completed=args.geometry_initial_stage2_iteration
                if args.iterations<=self.completed:raise ValueError('目标iteration必须大于Stage2初始化。')
                initialization_sha=file_sha256(init_path)
                (self.output/'stage2_initialization.json').write_text(json.dumps(dict(
                    checkpoint=str(init_path.resolve()),sha256=initialization_sha,
                    iteration=self.completed,source_iteration=self.context.iteration,
                    fresh_optimizer=True,geometry_schedule_origin=self.geometry_schedule_origin,
                    diagnostic_source_rejected=True,**initialization_buffers),ensure_ascii=False,indent=2)+'\n')
                del init
            if state is None and self.geometry.rgb_normal_enabled:
                from scene.stage2_rgb_prior import load_rgb_normal_prior
                if not initialization_sha:raise ValueError('RGB法线先验需要匹配的已训练Stage2初始化。')
                keys=[frame_key(self.context.train_cameras[i].image_name_train_light) for i in self.train_indices]
                normal,valid,metadata=load_rgb_normal_prior(args.geometry_rgb_normal_prior_path,
                    len(self.geometry.translation),self.context.source_files,initialization_sha,keys)
                self.geometry.rgb_normal_targets.copy_(normal.to('cuda'))
                self.geometry.rgb_normal_valid.copy_(valid.to('cuda'))
                (self.output/'geometry_rgb_normal_prior.json').write_text(json.dumps(dict(
                    metadata=metadata,asset_sha256=file_sha256(args.geometry_rgb_normal_prior_path),
                    checkpoint_self_contained=True),ensure_ascii=False,indent=2)+'\n')
            if state is None and self.geometry.coherence_enabled:
                from utils.general_utils import build_rotation
                camera=self.context.train_cameras[self.train_indices[0]]
                self.context.load_camera(camera)
                deformation=self.context.deformation(camera)
                pc=self.context.pc
                opacity=pc.get_opacity
                if deformation['d_opacity'] is not None:opacity=opacity+deformation['d_opacity']
                with torch.no_grad():
                    means,rotations,scales,opacity=self.geometry.prepare(
                        pc.get_xyz+deformation['d_xyz'],pc.get_rotation_bias(deformation['d_rotation']),
                        pc.get_scaling+deformation['d_scaling'],opacity)
                    labels=self.geometry.expand_values(self.material.labels) if hasattr(self.material,'labels') else None
                    static=pc.get_binary_feature().reshape(-1)<.5
                    mask_path=getattr(args,'geometry_coherence_source_mask','')
                    if mask_path:
                        keys=[frame_key(self.context.train_cameras[i].image_name_train_light) for i in self.train_indices]
                        stable,metadata=load_coherence_source_mask(mask_path,self.context.source,
                            self.context.iteration,self.context.source_files,keys,len(pc.get_xyz))
                        self.geometry.coherence_source_static.copy_(stable.to(static.device))
                        static=static & self.geometry.coherence_source_static
                        (self.output/'geometry_coherence_source_mask.json').write_text(json.dumps({
                            'metadata':metadata,'asset_sha256':file_sha256(mask_path),
                            'stable_source_points':int(stable.sum()),'source_points':len(stable),
                            'checkpoint_self_contained':True},ensure_ascii=False,indent=2)+'\n')
                    static=self.geometry.expand_values(static)
                    self.geometry.initialize_coherence(means,build_rotation(rotations)[:,:,2],scales,opacity,labels,static)
                self.context.unload_camera(camera)
                if getattr(args,'geometry_refined_coherence',False):
                    (self.output/'geometry_coherence_initialization.json').write_text(json.dumps(dict(
                        method='real_stage2_refined_surface_static_tangent_graph',
                        stage2_initialization_sha256=initialization_sha,
                        stage2_initialization_iteration=self.geometry_schedule_origin,
                        reference_optimization_key=frame_key(camera.image_name_train_light),
                        rendered_points=len(self.geometry.translation),
                        graph_edges=int((self.geometry.coherence_weights>0).sum()),
                        gt_normal_inputs=False,gt_geometry_inputs=False,gt_material_inputs=False,
                        graph_checkpoint_self_contained=True),ensure_ascii=False,indent=2)+'\n')
            validation=[frame_key(c.image_name_train_light) for i,c in enumerate(self.context.train_cameras)
                        if i%args.geometry_validation_stride==0]
            (self.output/'validation_split.json').write_text(json.dumps({'validation_keys':validation,
                'optimization_keys':[frame_key(self.context.train_cameras[i].image_name_train_light) for i in self.train_indices],
                'protocol':'Stage1已使用的训练视角内部划分；Stage2不优化validation；原15测试视角保留最终评测'},ensure_ascii=False,indent=2)+'\n')
            if state is not None and getattr(args,'geometry_checkpoint_source_cache',False):
                if 'geometry_reference_cache' not in state or 'geometry_visibility_cache' not in state:
                    raise ValueError('已开启源缓存检查点的状态缺少缓存；禁止静默重算。')
                cameras={frame_key(self.context.train_cameras[i].image_name_train_light):self.context.train_cameras[i]
                         for i in self.train_indices}
                self.reference_cache,visibility=restore_source_prediction_caches(
                    state['geometry_reference_cache'],state['geometry_visibility_cache'],cameras,
                    getattr(args,'geometry_photo_min_shading',0)>0)
                from scene.stage2_visibility import ReferenceVisibility
                self.context.visibility_cache=ReferenceVisibility(self.context,args)
                self.context.visibility_cache.cache=visibility
        self.optimizer=torch.optim.Adam([
            {'params':[self.material.raw_albedo],'lr':args.photometric_albedo_lr,'name':'albedo'},
            *([{'params':[self.geometry.translation],'lr':args.geometry_xyz_lr,'name':'geometry_xyz'},
               {'params':[self.geometry.axis_angle],'lr':args.geometry_rotation_lr,'name':'geometry_rotation'}]
              if self.geometry is not None else []),
            *([{'params':[self.geometry.log_scale],'lr':args.geometry_scale_lr,'name':'geometry_scale'}]
              if self.geometry is not None and self.geometry.log_scale is not None else []),
            *([{'params':[self.geometry.log_opacity],'lr':args.geometry_opacity_lr,'name':'geometry_opacity'}]
              if self.geometry is not None and self.geometry.log_opacity is not None else []),
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
        state={'format_version':2 if self.geometry is not None else 1,'iteration':self.completed,'source_iteration':self.context.iteration,
               'source_files':self.context.source_files,'frozen_fingerprint':self.context.fingerprint,
               'config':vars(self.args).copy(),'material':self.material.state_dict(),'light':self.light.state_dict(),
               'optimizer':self.optimizer.state_dict(),'camera_stack':self.stack,
               'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
               'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
        if self.geometry is not None:state['geometry']=self.geometry.state_dict()
        if getattr(self.args,'geometry_initial_stage2_model',''):
            state['geometry_schedule_origin']=self.geometry_schedule_origin
        if self.geometry is not None and getattr(self.args,'geometry_checkpoint_source_cache',False):
            state['geometry_reference_cache']=self.reference_cache
            state['geometry_visibility_cache']=getattr(getattr(self.context,'visibility_cache',None),'cache',{})
        torch.save(state,path)
        print('checkpoint',self.completed,path,flush=True)

    def train(self):
        if self.geometry is not None:
            return self.train_geometry()
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

    @torch.no_grad()
    def geometry_reference(self, camera):
        key=frame_key(camera.image_name_train_light)
        if key not in self.reference_cache:
            pc=self.context.pc
            del pc.stage2_geometry
            try:
                package=render_stage2_lambertian(camera,pc,self.material,self.context.deformation(camera),
                    self.context.background,direction=self.light(camera.fid),intensity=self.args.photometric_light_intensity)
            finally:pc.stage2_geometry=self.geometry
            self.reference_cache[key]={'depth':package['surf_depth'].cpu(),'alpha':package['rend_alpha'].cpu()}
            if getattr(self.args,'geometry_photo_min_shading',0)>0:
                self.reference_cache[key]['shading']=package['shading'].cpu()
        return {k:v.to('cuda') for k,v in self.reference_cache[key].items()}

    def train_geometry(self):
        start=time.monotonic()
        while self.completed<self.args.iterations:
            local=self.completed-self.geometry_schedule_origin
            prior_only,phase_local=geometry_training_phase(local,self.args)
            for group in self.optimizer.param_groups:
                if prior_only or group['name'] in {'albedo','geometry_xyz','geometry_rotation','geometry_scale','geometry_opacity'}:
                    group['lr']=geometry_group_lr(group['name'],local,self.args)
            self.optimizer.zero_grad(set_to_none=True)
            record={}
            for _ in range(self.args.geometry_accumulate):
                if not self.stack:self.stack=self.train_indices.copy()
                index=self.stack.pop(random.randrange(len(self.stack)))
                camera=self.context.train_cameras[index];self.context.load_camera(camera)
                reference=self.geometry_reference(camera)
                lighting=self.light.render_kwargs(camera.fid) if isinstance(self.light,PointLightTable) else {}
                if getattr(self.args,'geometry_shadows',False):
                    lighting['direct_visibility']=self.context.direct_visibility(camera,self.material,self.light)
                pkg=render_stage2_lambertian(camera,self.context.pc,self.material,self.context.deformation(camera),
                    self.context.background,direction=self.light(camera.fid),intensity=self.args.photometric_light_intensity,
                    **lighting)
                target,mask=supervision(camera,self.context.background)
                if getattr(self.args,'geometry_photo_visibility_threshold',0)>0:
                    mask=mask*(lighting['direct_visibility']>=self.args.geometry_photo_visibility_threshold).float()
                if getattr(self.args,'geometry_photo_min_shading',0)>0:
                    mask=mask*(reference['shading']>=self.args.geometry_photo_min_shading).float()
                result=self.preset.compute(pkg,target,mask,self.material,self.light)
                normal_support = mask if getattr(self.args,'geometry_normal_unsupported_weight',1.) != 1. else None
                if prior_only:
                    terms={'geometry_rgb_normal_prior':rgb_normal_prior_loss(pkg['geometry_normals'],self.geometry)}
                    total=self.args.lambda_geometry_rgb_normal_prior*terms['geometry_rgb_normal_prior']
                else:
                    extra,terms=geometry_losses(camera,pkg,reference,camera.gt_alpha_mask.to('cuda'),self.geometry,self.args,normal_support)
                    total=result.total*self.args.geometry_photo_weight+extra
                terms['photo_valid_fraction']=mask.sum()/(camera.gt_alpha_mask.to('cuda')>.5).sum().clamp_min(1)
                if not torch.isfinite(total):raise FloatingPointError('几何微调损失非有限。')
                backward_geometry_objective(total,terms.get('geometry_normal'),self.geometry,self.args,prior_only)
                for key,value in {'total':total,**result.terms,**terms}.items():
                    record[key]=record.get(key,0.)+float(value.detach())/self.args.geometry_accumulate
                self.context.unload_camera(camera)
            for group in self.optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('几何微调梯度非有限。')
                    if group['lr']==0:parameter.grad=None
            torch.nn.utils.clip_grad_norm_(self.geometry.parameters(),1.)
            self.optimizer.step();self.geometry.project(self.optimizer,rotation_only=prior_only);self.completed+=1
            if self.completed%self.args.photometric_log_interval==0 or self.completed==self.args.iterations:
                if getattr(self.args,'geometry_rgb_prior_warmup',0)>0:
                    record.update(geometry_phase='rgb_prior_rotation' if prior_only else 'joint',
                        geometry_phase_step=phase_local+1,photo_terms_monitor_only=prior_only)
                if self.geometry.normal_gradient_mode == 'position':
                    record['geometry_normal_gradient_mode'] = 'position'
                record.update(iteration=self.completed,elapsed_seconds=time.monotonic()-start,
                    **self.geometry.diagnostics(),**{g['name']+'_lr':g['lr'] for g in self.optimizer.param_groups})
                self.log.write(json.dumps(record)+'\n');self.log.flush();print(record,flush=True)
            if self.completed in self.args.save_iterations or self.completed==self.args.iterations:self.save()
        self.context.verify_frozen();self.log.close()
        (self.output/'TRAIN_COMPLETED').write_text(str(self.completed)+'\n')
