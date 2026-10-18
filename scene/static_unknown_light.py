"""未知方向光静态训练：固定阴影估光，物理几何损失与重置恢复期透明度保护。"""
from copy import copy
import hashlib
import json
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
from gaussian_renderer import render as render_native
from scripts.train_static_dl import StaticDLTrainer
from scene.stage2_photometric import srgb_to_linear
from utils.sh_utils import RGB2SH


def regularization_schedule(iteration,normal_start,dist_start):
    return iteration>normal_start,iteration>dist_start


class UnknownLightTrainer(StaticDLTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.gaussians.preserve_screen_radii_event = bool(getattr(self.args,'preserve_screen_radii_event',False))
        if self.gaussians.preserve_screen_radii_event:
            original_densify = self.gaussians.densify_and_prune
            def audited_radius_event(*args,**kwargs):
                result = original_densify(*args,**kwargs)
                record = dict(self.gaussians._last_screen_radius_event,iteration=self.iteration)
                with (Path(self.args.model_path)/'screen_radius_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                return result
            self.gaussians.densify_and_prune = audited_radius_event
        expected = {str(i).zfill(4) for i in range(1,151) if i%10}
        if len(self.scene.getTrainCameras())!=135 or set(self.physical_lights.directions)!=expected:
            raise AssertionError('Stage1只允许原135训练相机与方向。')
        self.sparse_rgb_surface = None
        if getattr(self.args,'sparse_rgb_surface_weight',0.):
            from scene.static_sparse_rgb_surface import FixedSparseRGBSurface
            self.sparse_rgb_surface = FixedSparseRGBSurface(
                self.args.sparse_rgb_surface_file,self.scene.cameras_extent)
        model = Path(self.args.model_path)
        self.receiver_plane = None
        if self.args.plane_normal_weight:
            from scene.static_receiver_plane import FixedReceiverPlane
            self.receiver_plane = FixedReceiverPlane(self.args.source_path)
        static = json.loads((model/'static_dl_training.json').read_text())
        static.update(actual_entrypoint='scripts/train_static_unknown_light.py',loss_model='censored',
                      normal_depth_statistic='expected',normal_start=self.opt.start_normal_reg,dist_start=self.args.dist_start)
        (model/'static_dl_training.json').write_text(json.dumps(static,ensure_ascii=False,indent=2)+'\n')
        root = Path(__file__).resolve().parents[1]
        sources = ('scripts/train_static_unknown_light.py','scene/static_unknown_light.py',
                   'scripts/train_static_dl.py','scene/static_dl_censored.py','scene/static_dl_photometric.py',
                   'scene/gaussian_model.py')
        if self.receiver_plane is not None:
            sources += ('scene/static_receiver_plane.py',)
        record = {'actual_entrypoint':static['actual_entrypoint'],'args':vars(self.args),
                  'new_sources':{str(root/p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                  'loss_model':'censored','normal_depth_statistic':'expected','photo_depth_ratio':1.,
                  'normal_start':self.opt.start_normal_reg,'dist_start':self.args.dist_start,
                  'initial_point_count':len(self.gaussians.get_xyz),'densification':True,
                  'normal_screen_exclude_from':getattr(self.args,'normal_screen_exclude_from',0),
                  'preserve_screen_radii_event':self.gaussians.preserve_screen_radii_event,
                  'densification_screen_gradient':('photo_and_dist_proxy_only_from_normal_screen_exclude_from; original_combined_before' if getattr(self.args,'normal_screen_exclude_from',0) else 'photo_and_dist_proxy_plus_expected_normal_proxy_then_one_norm_and_one_denom'),
                  'initial_point_cloud_sha256':hashlib.sha256((Path(self.args.source_path)/'points3d.ply').read_bytes()).hexdigest(),
                  'smoke_schedule':self.args.smoke_schedule, 'gt_supervision':False,
                  'initialization':('原135 TRAIN Alpha/相机保守视觉外壳100k位置，原KNN自然scale，无模型权重/外壳法线旋转' if getattr(self.args,'train_alpha_hull',False) else '原135训练RGB、Alpha和相机之外仅确定性随机点云，无模型权重'),
                  'initialization_manifest_sha256':hashlib.sha256((Path(self.args.source_path)/'no_gt_manifest.json').read_bytes()).hexdigest(),
                  'optimizer_order':'backward→optimizer→非终点拓扑/reset→save',
                  'normal_detach_start':self.args.normal_detach_start,
                  'skip_opacity_reset_at':self.args.skip_opacity_reset_at,
                  'skip_additional_opacity_reset_at':getattr(self.args,'skip_additional_opacity_reset_at',0),
                  'dist_opacity_route':self.args.dist_opacity_route,
                  'plane_normal_weight':self.args.plane_normal_weight,'plane_start':self.args.plane_start,
                  'receiver_plane':None if self.receiver_plane is None else self.receiver_plane.metadata}
        if self.sparse_rgb_surface is not None:
            record['sparse_rgb_surface'] = self.sparse_rgb_surface.metadata()
            record['sparse_rgb_surface']['start'] = self.opt.start_normal_reg+1
            record['densification_screen_gradient'] = 'photo_dist_and_Median_anchor_proxy_plus_expected_normal_proxy_then_one_norm_and_one_denom'
            module = root/'scene/static_sparse_rgb_surface.py'
            record['new_sources'][str(module)] = hashlib.sha256(module.read_bytes()).hexdigest()
        (model/'unknown_light_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')

    def _save_physical(self):
        super()._save_physical()
        path = Path(self.args.model_path)/f'static_dl_lighting/iteration_{self.iteration}/lighting.json'
        value = json.loads(path.read_text())
        value.update(loss_model='censored',normal_depth_statistic='expected',
                     normal_start=self.opt.start_normal_reg,dist_start=self.args.dist_start,
                     smoke_schedule=self.args.smoke_schedule,actual_entrypoint='scripts/train_static_unknown_light.py')
        value.update(normal_opacity_route='detach_opacity' if self.iteration >= self.args.normal_detach_start else 'full',
                     skip_opacity_reset_at=self.args.skip_opacity_reset_at,
                     skip_additional_opacity_reset_at=getattr(self.args,'skip_additional_opacity_reset_at',0),
                     dist_opacity_route=self.args.dist_opacity_route,
                     plane_normal_weight=self.args.plane_normal_weight,plane_start=self.args.plane_start,gt_supervision=False)
        if getattr(self,'sparse_rgb_surface',None) is not None:
            value['sparse_rgb_surface'] = self.sparse_rgb_surface.metadata()
            value['sparse_rgb_surface']['start'] = self.opt.start_normal_reg+1
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        torch.save({'iteration':self.iteration,'gaussian_optimizer':self.gaussians.optimizer.state_dict(),
                    'light_optimizer':self.light_optimizer.state_dict(),
                    'python_rng':__import__('random').getstate(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state(),'camera_stack':[c.image_name_train_light for c in (self.viewpoint_stack or [])],
                    'numpy_rng':__import__('numpy').random.get_state(),
                    'densification':{name:getattr(self.gaussians,name).detach().clone()
                                      for name in ('xyz_gradient_accum','denom','max_radii2D')},
                    'active_sh_degree':self.gaussians.active_sh_degree,
                    'spatial_lr_scale':self.gaussians.spatial_lr_scale,
                    'checkpoint_state_version':2,
                    'note':'保存增密累计、NumPy及原优化器/RNG；尚需经验证的显式恢复入口，不声称已支持无缝续训'},
                   Path(self.args.model_path)/f'optimizer_{self.iteration}.pt')

    def _opacity_reset_boundary(self):
        """只改变预定reset调用；完整保留同一步优化与拓扑，审计其参数及Adam效应。"""
        policy = self.args.skip_opacity_reset_at
        extra = getattr(self.args,'skip_additional_opacity_reset_at',0)
        skipped_steps = {step for step in (policy,extra) if step}
        if not policy:
            self.gaussians.reset_opacity()
            self.visibility.invalidate()
            return
        import random
        import numpy as np
        optimizer = self.gaussians.optimizer
        opacity = self.gaussians._opacity
        before = opacity.detach().clone()
        old_state = optimizer.state.get(opacity, {})
        moments = {key: value.detach().clone() for key,value in old_state.items() if torch.is_tensor(value)}
        others = [(parameter,parameter._version,parameter.detach().clone(),
                   {key:value.detach().clone() for key,value in optimizer.state.get(parameter,{}).items()
                    if torch.is_tensor(value)})
                  for group in optimizer.param_groups if group['name'] != 'opacity' for parameter in group['params']]
        rng = (repr(random.getstate()),repr(np.random.get_state()),torch.get_rng_state(),torch.cuda.get_rng_state())
        skip = self.iteration in skipped_steps
        if not skip:
            self.gaussians.reset_opacity()
            self.visibility.invalidate()
        after = self.gaussians._opacity
        new_state = optimizer.state.get(after,{})
        other_unchanged = all(parameter._version == version and torch.equal(parameter,value)
                              and all(key in optimizer.state.get(parameter,{})
                                      and torch.equal(optimizer.state[parameter][key],moment)
                                      for key,moment in states.items())
                              for parameter,version,value,states in others)
        rng_unchanged = (repr(random.getstate()) == rng[0] and repr(np.random.get_state()) == rng[1]
                         and torch.equal(torch.get_rng_state(),rng[2]) and torch.equal(torch.cuda.get_rng_state(),rng[3]))
        expected = before.sigmoid().clamp_max(.01)
        if skip:
            opacity_ok = after is opacity and torch.equal(after,before)
            adam_ok = all(key in new_state and torch.equal(new_state[key],value) for key,value in moments.items())
        else:
            opacity_ok = bool(torch.allclose(after.sigmoid(),expected,atol=1e-7,rtol=1e-5))
            adam_ok = all(key in new_state and torch.count_nonzero(new_state[key]) == 0
                          for key in ('exp_avg','exp_avg_sq'))
            if 'step' in moments:
                adam_ok = adam_ok and torch.equal(new_state['step'],moments['step'])
        record = dict(iteration=self.iteration,action='SKIPPED' if skip else 'EXECUTED',
                      skip_opacity_reset_at=policy,after_optimizer_and_topology=True,
                      skip_additional_opacity_reset_at=extra,skip_opacity_reset_steps=sorted(skipped_steps),
                      point_count=len(after),opacity_value_check=bool(opacity_ok),
                      opacity_adam_check=bool(adam_ok),other_parameters_and_adam_unchanged=other_unchanged,
                      rng_unchanged=rng_unchanged,before_max=float(before.sigmoid().max()),
                      after_max=float(after.sigmoid().max()),before_mean=float(before.sigmoid().mean()),
                      after_mean=float(after.sigmoid().mean()))
        record['passed'] = bool(opacity_ok and adam_ok and other_unchanged and rng_unchanged)
        with (Path(self.args.model_path)/'opacity_reset_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print('[透明度重置边界] '+json.dumps(record,ensure_ascii=False),flush=True)
        if not record['passed']:
            raise RuntimeError('opacity reset边界参数/Adam/RNG审计失败，保留现场。')

    def _audit_dist_opacity(self, base_loss, package, geometry):
        """同图同参数核对dist支路；恢复retain_grad缓存，不影响增密累计。"""
        params=[self.gaussians._xyz,self.gaussians._rotation,self.gaussians._scaling,self.gaussians._opacity]
        screens=[package['viewspace_points'],geometry['viewspace_points']]
        saved=[None if x.grad is None else x.grad.detach().clone() for x in screens]
        versions=[x._version for x in params]
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        full=self.opt.lambda_dist*package['rend_dist'].mean()
        isolated=self.opt.lambda_dist*geometry['rend_dist'].mean()
        record={'iteration':self.iteration,'forward_exact':torch.equal(package['rend_dist'],geometry['rend_dist']),
                'scope':'只断开opacity参数；不等同CUDA DETACH_WEIGHT=1，权重对XYZ/scale的导数保留'}
        def compare(a,b):
            if a is None or b is None:
                return {'passed':a is None and b is None,'both_unused':a is None and b is None}
            difference=a-b
            relative=float(difference.norm()/torch.maximum(a.norm(),b.norm()).clamp_min(1e-12))
            return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all()) and relative<1e-3,
                    'relative_l2':relative,'max_abs':float(difference.abs().max()),'reference_norm':float(a.norm())}
        try:
            old=torch.autograd.grad(full,params+[screens[0]],retain_graph=True,allow_unused=True)
            new=torch.autograd.grad(isolated,params+[screens[1]],retain_graph=True,allow_unused=True)
            record['opacity_control_norm']=None if old[3] is None else float(old[3].norm())
            record['opacity_isolated_absent']=new[3] is None or bool(torch.count_nonzero(new[3])==0)
            record['geometry_and_screen_vjp']={name:compare(old[i],new[i]) for name,i in [('xyz',0),('rotation',1),('scaling',2),('screen',4)]}
            old_total=torch.autograd.grad(base_loss+full,params+screens,retain_graph=True,allow_unused=True)
            new_total=torch.autograd.grad(base_loss+isolated,params+screens,retain_graph=True,allow_unused=True)
            record['total_opacity_difference']=compare(old_total[3]-new_total[3],old[3])
            record['merged_total_screen_vjp']=compare(old_total[4]+old_total[5],new_total[4]+new_total[5])
        finally:
            for x,v in zip(screens,saved):
                x.grad=v
        record['screen_grad_cache_restored']=True
        record['parameter_versions_unchanged']=versions==[x._version for x in params]
        record['rng_unchanged']=torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        record['passed']=(record['forward_exact'] and record['opacity_isolated_absent'] and
                          record['opacity_control_norm'] is not None and record['opacity_control_norm']>0 and
                          all(v['passed'] for v in record['geometry_and_screen_vjp'].values()) and
                          record['total_opacity_difference']['passed'] and record['merged_total_screen_vjp']['passed'] and
                          record['parameter_versions_unchanged'] and record['rng_unchanged'])
        (Path(self.args.model_path)/'dist_opacity_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
        if not record['passed']:
            raise RuntimeError('dist opacity隔离的真实前向/余切审计失败；现场保留。')
        self.gradient_audited.add('dist_opacity_detached')

    def _render_photometric(self,camera,use_visibility):
        from scene.static_dl_photometric import render_photometric
        return render_photometric(camera,self.gaussians,self.pipe,self.background,
                                  self.physical_lights,self.visibility if use_visibility else None,self.iteration)

    def _photo_loss(self,package,target,gt_alpha,camera):
        from scene.static_dl_censored import censored_photometric_loss
        return censored_photometric_loss(package['render'],target,gt_alpha,self.args.static_dl_shadow_weight,
                                         self.args.static_dl_highlight_weight,self.args.static_dl_residual_threshold)

    def train_step(self):
        if self.args.static_dl_mode == 'sh' or self.iteration <= self.args.static_dl_physical_start:
            return super().train_step()
        if not self.physical_initialized:
            self._begin_physical()
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device()
        use_visibility = (self.args.static_dl_mode == 'lambertian_visibility' and
                          self.iteration >= self.args.static_dl_visibility_start)
        package = self._render_photometric(camera,use_visibility)
        target = camera.original_image_train_light.cuda()
        gt_alpha = camera.gt_alpha_mask.cuda()
        if self.dataset.white_background:
            target = gt_alpha*target+(1-gt_alpha)*self.background[:, None, None]
        photo, diagnostics = self._photo_loss(package,target,gt_alpha,camera)
        alpha = package['rend_alpha'].clamp(1e-6, 1-1e-6)
        alpha_loss = F.binary_cross_entropy(alpha, gt_alpha)
        loss = photo+self.opt.lambda_alpha_loss*alpha_loss
        normal_loss = loss.new_zeros(())
        dist_loss = loss.new_zeros(())
        plane_loss = loss.new_zeros(())
        plane_details = {}
        normal_active, dist_active = regularization_schedule(self.iteration,self.opt.start_normal_reg,self.args.dist_start)
        if normal_active:
            # RGB保持原pipe；仅独立真实几何副本采用expected。Stage1仍允许scale/opacity梯度。
            geometry_pipe = copy(self.pipe)
            geometry_pipe.depth_ratio = 0.
            geometry = render_native(camera,self.gaussians,geometry_pipe,srgb_to_linear(self.background),
                                     0.,0.,0.,override_color=torch.zeros_like(self.gaussians.get_xyz),
                                     detach_opacity=self.iteration >= self.args.normal_detach_start)
            if (not torch.equal(geometry['radii'],package['radii']) or
                    not torch.equal(geometry['visibility_filter'],package['visibility_filter'])):
                raise AssertionError('expected副本改变投影半径或可见性filter。')
            normal = F.normalize(geometry['rend_normal'], dim=0)
            depth_normal = F.normalize(geometry['surf_normal'], dim=0)
            mask = (gt_alpha > .99).float()
            mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
            mask = mask*(alpha.detach() > .5)
            normal_loss = ((1-(normal*depth_normal).sum(0, keepdim=True))*mask).sum()/mask.sum().clamp_min(1)
            loss = loss+self.args.static_dl_normal_weight*normal_loss
        if dist_active:
            if self.args.dist_opacity_route == 'detach':
                if not normal_active or self.iteration < self.args.normal_detach_start:
                    raise RuntimeError('dist隔离缺少已detach的几何副本。')
                if 'dist_opacity_detached' not in self.gradient_audited:
                    self._audit_dist_opacity(loss,package,geometry)
                dist_loss = geometry['rend_dist'].mean()
            else:
                dist_loss = package['rend_dist'].mean()
            loss = loss+self.opt.lambda_dist*dist_loss
        if self.receiver_plane is not None and self.iteration > self.args.plane_start:
            if not normal_active or self.iteration < self.args.normal_detach_start:
                raise RuntimeError('接收面先验缺少已detach的expected几何副本。')
            plane_loss,plane_details = self.receiver_plane(camera,geometry)
            weighted_plane = self.args.plane_normal_weight*plane_loss
            if 'receiver_plane' not in self.gradient_audited:
                from scene.static_receiver_plane import audit_plane_gradient
                audit_plane_gradient(weighted_plane,self.args.static_dl_normal_weight*normal_loss,
                                     self.gaussians,self.physical_lights,
                                     [package['viewspace_points'],geometry['viewspace_points']],
                                     Path(self.args.model_path)/'receiver_plane_gradient_audit.json',self.iteration)
                self.gradient_audited.add('receiver_plane')
            loss = loss+weighted_plane
        sparse_surface_loss = loss.new_zeros(())
        sparse_surface_diagnostics = {}
        if normal_active and self.sparse_rgb_surface is not None:
            if self.pipe.depth_ratio != 1.:
                raise AssertionError('固定像素三维表面项必须直接使用原photo native Median。')
            sparse_surface_loss,sparse_surface_diagnostics = self.sparse_rgb_surface(
                camera,package['surf_depth'])
            loss = loss+self.args.sparse_rgb_surface_weight*sparse_surface_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'静态物理loss非有限，iteration={self.iteration}。')
        if normal_active and self.iteration >= self.args.normal_detach_start and 'detach_opacity' not in self.gradient_audited:
            grads = torch.autograd.grad(normal_loss, [self.gaussians._xyz,self.gaussians._rotation,
                        self.gaussians._scaling,self.gaussians._opacity],retain_graph=True,allow_unused=True)
            audit = {'iteration':self.iteration,'opacity_gradient_absent':grads[3] is None or bool(torch.count_nonzero(grads[3])==0),
                     'geometry_gradient_norms':[None if g is None else float(g.norm()) for g in grads[:3]]}
            if not audit['opacity_gradient_absent'] or any(g is None or not torch.isfinite(g).all() or g.norm()==0 for g in grads[:3]):
                raise RuntimeError('normal透明度保护的真实梯度检查失败。')
            (Path(self.args.model_path)/'normal_opacity_audit.json').write_text(json.dumps(audit,indent=2))
            self.gradient_audited.add('detach_opacity')
        loss.backward()
        audit_key = ('visibility' if use_visibility else 'unoccluded')+f'_normal{int(normal_active)}_dist{int(dist_active)}'
        if audit_key not in self.gradient_audited:
            audit = {'iteration': self.iteration, 'mode': audit_key, 'gradients': {},
                     'normal_active':normal_active,'dist_active':dist_active,
                     'normal_start':self.opt.start_normal_reg,'dist_start':self.args.dist_start,
                     'normal_loss':float(normal_loss.detach()),'dist_loss':float(dist_loss.detach())}
            for name, param in (('xyz', self.gaussians._xyz), ('rotation', self.gaussians._rotation),
                                ('albedo_dc', self.gaussians._albedo_dc),
                                ('log_intensity', self.physical_lights.log_intensity)):
                if not param.requires_grad:
                    audit['gradients'][name] = {'frozen':True,'gradient_absent':param.grad is None}
                    if param.grad is not None:
                        raise AssertionError('固定光照不应有梯度')
                    continue
                grad = param.grad
                if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
                    raise FloatingPointError(f'静态物理{name}梯度缺失、非有限或全零。')
                audit['gradients'][name] = {'norm': float(grad.norm()), 'max_abs': float(grad.abs().max())}
            audit['visibility'] = {key: float(package[key]) for key in
                                   ('mean_visibility', 'visibility_below_point_one', 'visibility_above_point_nine')}
            audit['trace_rays'] = len(self.gaussians.get_xyz)*len(self.physical_lights.log_intensity) if use_visibility else 0
            folder = Path(self.args.model_path)/'static_dl_gradient_audit'
            folder.mkdir(exist_ok=True)
            (folder/f'{audit_key}.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
            self.gradient_audited.add(audit_key)
            print('[静态物理梯度] '+json.dumps(audit, ensure_ascii=False), flush=True)
        with torch.no_grad():
            visibility = package['visibility_filter']
            radii = package['radii']
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility] = torch.maximum(self.gaussians.max_radii2D[visibility], radii[visibility])
            self.ema_loss_for_log = .4*float(loss)+.6*self.ema_loss_for_log
            if self.iteration % 100 == 0:
                values = dict(iteration=self.iteration,camera=camera.image_name_train_light,
                              normal_active=normal_active,dist_active=dist_active,
                              normal_opacity_route='detach_opacity' if self.iteration>=self.args.normal_detach_start else 'full',
                              dist_opacity_route=self.args.dist_opacity_route,
                              photo=float(photo), alpha=float(alpha_loss),
                              normal=float(normal_loss), distortion=float(dist_loss),
                              plane_normal=float(plane_loss),plane_normal_weight=self.args.plane_normal_weight,
                              **plane_details,
                              visibility=float(package['mean_visibility']), points=len(self.gaussians.get_xyz),
                              light=package['light_intensity'].cpu().tolist(),
                              **{k: float(v) for k, v in diagnostics.items()})
                if self.sparse_rgb_surface is not None:
                    values['sparse_surface_loss'] = float(sparse_surface_loss.detach())
                    values['sparse_surface_weighted'] = self.args.sparse_rgb_surface_weight*float(sparse_surface_loss.detach())
                    values['sparse_surface_diagnostics'] = {k:float(v) for k,v in sparse_surface_diagnostics.items()}
                print('[静态物理] '+json.dumps(values, ensure_ascii=False), flush=True)
                with (Path(self.args.model_path)/'unknown_light_metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(values, ensure_ascii=False)+'\n')
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({'Loss': f'{self.ema_loss_for_log:.7f}'})
                self.progress_bar.update(10)
            # 在拓扑替换前使用本步真实参数梯度；终点也执行一次优化更新。
            self.gaussians.optimizer.step()
            self.light_optimizer.step()
            self.gaussians.update_learning_rate(self.iteration)
            self.gaussians._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            self.physical_lights.log_intensity.clamp_(-5, 5)
            if self.iteration < min(self.opt.densify_until_iter,self.opt.iterations):
                if normal_active:
                    screen_gradient = geometry['viewspace_points'].grad
                    if screen_gradient is None or not torch.isfinite(screen_gradient).all():
                        raise FloatingPointError('expected几何screen余切缺失或非有限。')
                    # 合并同一屏幕变量的向量余切后只统计一次，不能相加范数或重复denom。
                    # 屏幕代理不被optimizer更新；不改已经累计的pc._xyz.grad。
                    if not (0 < getattr(self.args,'normal_screen_exclude_from',0) <= self.iteration):
                        package['viewspace_points'].grad.add_(screen_gradient)
                self.gaussians.add_densification_stats(package['viewspace_points'], visibility)
                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    limit = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.min_opacity,
                                                     self.scene.cameras_extent, limit)
                    self.visibility.invalidate()
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    self._opacity_reset_boundary()
            if self.iteration in self.saving_iterations:
                self._save_physical()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.light_optimizer.zero_grad(set_to_none=True)
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device('cpu')
        if self.iteration == self.opt.iterations:
            self.progress_bar.close()
        self.iteration += 1

