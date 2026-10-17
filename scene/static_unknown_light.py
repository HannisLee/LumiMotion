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
        expected = {str(i).zfill(4) for i in range(1,151) if i%10}
        if len(self.scene.getTrainCameras())!=135 or set(self.physical_lights.directions)!=expected:
            raise AssertionError('Stage1只允许原135训练相机与方向。')
        model = Path(self.args.model_path)
        static = json.loads((model/'static_dl_training.json').read_text())
        static.update(actual_entrypoint='scripts/train_static_unknown_light.py',loss_model='censored',
                      normal_depth_statistic='expected',normal_start=self.opt.start_normal_reg,dist_start=self.args.dist_start)
        (model/'static_dl_training.json').write_text(json.dumps(static,ensure_ascii=False,indent=2)+'\n')
        root = Path(__file__).resolve().parents[1]
        sources = ('scripts/train_static_unknown_light.py','scene/static_unknown_light.py',
                   'scripts/train_static_dl.py','scene/static_dl_censored.py','scene/static_dl_photometric.py')
        record = {'actual_entrypoint':static['actual_entrypoint'],'args':vars(self.args),
                  'new_sources':{str(root/p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
                  'loss_model':'censored','normal_depth_statistic':'expected','photo_depth_ratio':1.,
                  'normal_start':self.opt.start_normal_reg,'dist_start':self.args.dist_start,
                  'initial_point_count':len(self.gaussians.get_xyz),'densification':True,
                  'densification_screen_gradient':'photo_and_dist_proxy_plus_expected_normal_proxy_then_one_norm_and_one_denom',
                  'initial_point_cloud_sha256':hashlib.sha256((Path(self.args.source_path)/'points3d.ply').read_bytes()).hexdigest(),
                  'smoke_schedule':self.args.smoke_schedule, 'gt_supervision':False,
                  'initialization':'原135训练RGB、Alpha和相机之外仅确定性随机点云，无模型权重',
                  'optimizer_order':'backward→optimizer→非终点拓扑/reset→save',
                  'normal_detach_start':self.args.normal_detach_start}
        (model/'unknown_light_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')

    def _save_physical(self):
        super()._save_physical()
        path = Path(self.args.model_path)/f'static_dl_lighting/iteration_{self.iteration}/lighting.json'
        value = json.loads(path.read_text())
        value.update(loss_model='censored',normal_depth_statistic='expected',
                     normal_start=self.opt.start_normal_reg,dist_start=self.args.dist_start,
                     smoke_schedule=self.args.smoke_schedule,actual_entrypoint='scripts/train_static_unknown_light.py')
        value.update(normal_opacity_route='detach_opacity' if self.iteration >= self.args.normal_detach_start else 'full',
                     gt_supervision=False)
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

    def train_step(self):
        if self.args.static_dl_mode == 'sh' or self.iteration <= self.args.static_dl_physical_start:
            return super().train_step()
        from scene.static_dl_photometric import render_photometric
        from scene.static_dl_censored import censored_photometric_loss
        if not self.physical_initialized:
            self._begin_physical()
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device()
        use_visibility = (self.args.static_dl_mode == 'lambertian_visibility' and
                          self.iteration >= self.args.static_dl_visibility_start)
        package = render_photometric(camera, self.gaussians, self.pipe, self.background,
                                     self.physical_lights, self.visibility if use_visibility else None,
                                     self.iteration)
        target = camera.original_image_train_light.cuda()
        gt_alpha = camera.gt_alpha_mask.cuda()
        if self.dataset.white_background:
            target = gt_alpha*target+(1-gt_alpha)*self.background[:, None, None]
        photo, diagnostics = censored_photometric_loss(
            package['render'], target, gt_alpha, self.args.static_dl_shadow_weight,
            self.args.static_dl_highlight_weight, self.args.static_dl_residual_threshold)
        alpha = package['rend_alpha'].clamp(1e-6, 1-1e-6)
        alpha_loss = F.binary_cross_entropy(alpha, gt_alpha)
        loss = photo+self.opt.lambda_alpha_loss*alpha_loss
        normal_loss = loss.new_zeros(())
        dist_loss = loss.new_zeros(())
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
            dist_loss = package['rend_dist'].mean()
            loss = loss+self.opt.lambda_dist*dist_loss
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
                              photo=float(photo), alpha=float(alpha_loss),
                              normal=float(normal_loss), distortion=float(dist_loss),
                              visibility=float(package['mean_visibility']), points=len(self.gaussians.get_xyz),
                              light=package['light_intensity'].cpu().tolist(),
                              **{k: float(v) for k, v in diagnostics.items()})
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
                    package['viewspace_points'].grad.add_(screen_gradient)
                self.gaussians.add_densification_stats(package['viewspace_points'], visibility)
                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    limit = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.min_opacity,
                                                     self.scene.cameras_extent, limit)
                    self.visibility.invalidate()
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()
                    self.visibility.invalidate()
            if self.iteration in self.saving_iterations:
                self._save_physical()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.light_optimizer.zero_grad(set_to_none=True)
        if self.dataset.load2gpu_on_the_fly:
            camera.load2device('cpu')
        if self.iteration == self.opt.iterations:
            self.progress_bar.close()
        self.iteration += 1

