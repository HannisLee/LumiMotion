#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# 派生自scripts/train_stage1.py，保留原Inria非商业研究许可证及LICENSE.md限制。
"""无GT静态SH预热：保留原native拓扑次序，只修真实终步与更新后保存。"""
import hashlib
import json
from pathlib import Path
import torch
from scripts.train_stage1 import Trainer,randint,render,l1_loss,ssim,F,training_report


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


class PostStepSHTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.completed_optimizer_calls=0
        self.audit_before={}
        self.last_audit=None

    def _before_optimizer_call(self,camera,loss):
        selected=self.args.smoke or self.iteration in self.saving_iterations or self.iteration==self.opt.warm_up-1
        self.audit_before={}
        if selected:
            for group in self.gaussians.optimizer.param_groups:
                params=group['params']
                self.audit_before[group['name']]=dict(parameter_ids=[id(p) for p in params],
                    with_grad=sum(p.grad is not None for p in params),
                    gradient_finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params),
                    gradient_norm=sum(float(p.grad.norm()) for p in params if p.grad is not None))
                if self.args.smoke:self.audit_before[group['name']]['_before']=[p.detach().clone() for p in params]
            assert all(v['gradient_finite'] for v in self.audit_before.values())
        if self.iteration%100==0 or selected:
            with (Path(self.args.model_path)/'sh_train_metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(iteration=self.iteration,camera=camera.image_name_train_light,
                    loss=float(loss),points=len(self.gaussians.get_xyz),topology_precedes_optimizer=True))+'\n')

    def _after_optimizer_call(self):
        self.completed_optimizer_calls+=1
        assert self.completed_optimizer_calls==self.iteration
        if not self.audit_before:return
        for group in self.gaussians.optimizer.param_groups:
            row=self.audit_before[group['name']]
            assert row['parameter_ids']==[id(p) for p in group['params']]
            row['adam_steps']=[int(self.gaussians.optimizer.state.get(p,{}).get('step',0)) for p in group['params']]
            if self.args.smoke:
                old=row.pop('_before');row['actual_max_change']=max(float((p-a).abs().max()) for p,a in zip(group['params'],old))
                assert all(torch.isfinite(p).all() for p in group['params'])
        self.last_audit=dict(iteration=self.iteration,completed_optimizer_calls=self.completed_optimizer_calls,
                            groups=self.audit_before,topology_precedes_optimizer=True)
        with (Path(self.args.model_path)/'sh_optimizer_audit.jsonl').open('a') as f:f.write(json.dumps(self.last_audit)+'\n')

    def _save_post_step(self):
        assert self.completed_optimizer_calls==self.iteration
        assert self.last_audit['iteration']==self.iteration
        self.scene.save(self.iteration);self.deform.save_weights(self.args.model_path,self.iteration)
        model=Path(self.args.model_path);folder=model/f'geometry_init/actual_{self.iteration}'
        folder.mkdir(parents=True,exist_ok=False)
        tensors={name:getattr(self.gaussians,name).detach().cpu().clone() for name in ('_xyz','_scaling','_rotation','_opacity')}
        assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in tensors.values())
        count=len(self.gaussians.get_xyz)
        assert count>0 and all(tensors[k].shape==(count,width) for k,width in [('_xyz',3),('_scaling',2),('_rotation',4),('_opacity',1)])
        assert (tensors['_rotation'].norm(dim=1)>0).all()
        scales=tensors['_scaling'].exp();assert torch.isfinite(scales).all() and (scales>0).all()
        assert all(torch.count_nonzero(getattr(self.deform.deform,name).weight)==0 and torch.count_nonzero(getattr(self.deform.deform,name).bias)==0 for name in ('gaussian_warp','gaussian_rotation','gaussian_scaling'))
        assert all(not p.requires_grad for name in ('gaussian_warp','gaussian_rotation','gaussian_scaling') for p in getattr(self.deform.deform,name).parameters())
        torch.save(tensors,folder/'geometry.pt')
        write(folder/'receipt.json',dict(gt_supervision=False,external_test_read=False,
            checkpoint_semantics='post_optimizer_post_native_topology',requested_step=self.iteration,
            completed_optimizer_calls=self.completed_optimizer_calls,points=len(self.gaussians.get_xyz),
            geometry_keys=list(tensors),geometry_sha256=sha(folder/'geometry.pt'),
            ply_sha256=sha(model/f'point_cloud/iteration_{self.iteration}/point_cloud.ply'),
            deform_sha256=sha(model/f'deform/iteration_{self.iteration}/deform.pth'),
            geometry_heads_zero=True,optimizer_audit=self.last_audit,
            note='实际optimizer调用数不等于每个参数有效更新数；拓扑前置原语义保留，不转移颜色或优化器。'))

    def train_step(self):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree 
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack

        time_interval = 1 / len(self.scene.all_timesteps)

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        #when start binarization
        if self.iteration>self.opt.binarization_warm_up and not self.dataset.no_binary_separation:
            self.gaussians.no_binary_separation = False
        else:
            self.gaussians.no_binary_separation = True


        if self.deform.name == 'mlp' or self.deform.name == 'static':
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
                d_values = self.deform.step(self.gaussians.get_xyz, time_input + ast_noise, iteration=self.iteration, 
                                            feature=self.gaussians.get_binary_feature(eval=False, T=self.T_current),
                                            camera_center=viewpoint_cam.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        else:
            raise NotImplemented

        # Render
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]



        lambda_normal = 0.02 if self.iteration > self.opt.start_normal_reg else 0.0
        lambda_dist = self.opt.lambda_dist if self.iteration > self.opt.start_normal_reg else 0.0
        rend_dist = render_pkg_re["rend_dist"]
        rend_normal  = render_pkg_re['rend_normal']
        surf_normal = render_pkg_re['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()



        # Loss
        gt_image = viewpoint_cam.original_image_train_light.cuda()
        if self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img + normal_loss + dist_loss

        #mask loss
        if self.opt.gt_alpha_mask_as_scene_mask and viewpoint_cam.gt_alpha_mask is not None:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            alpha_loss = F.binary_cross_entropy(render_pkg_re['rend_alpha'][:, None, None], gt_alpha_mask.unsqueeze(1).unsqueeze(1))
            loss += alpha_loss*self.opt.lambda_alpha_loss

        else:
            simulated_mask = torch.ones_like(render_pkg_re['rend_alpha'][:, None, None])
            alpha_loss = F.binary_cross_entropy(render_pkg_re['rend_alpha'][:, None, None], simulated_mask)
            loss += alpha_loss * 0.001


        if self.iteration > self.opt.warm_up:

            # d_xyz loss
            loss += ((d_xyz**2).mean())*self.opt.d_xyz_loss_weight 

            # d color loss
            d_color_reg_loss_weight = self.opt.d_color_reg_loss_weight
            if (d_color is not None and torch.is_tensor(d_color)):

                shadow_modulation = d_color[:, :3]

                d_color_reg_loss = (
                    shadow_modulation.pow(2.0).mean() * d_color_reg_loss_weight
                )

                loss += d_color_reg_loss

            if self.iteration > self.opt.binarization_warm_up and not self.dataset.no_binary_separation:

                # L1 for unsupervised bianrizationin the paper
                loss += (self.gaussians.get_binary_feature(eval=False, T=self.T_current)**1).mean()*self.opt.lambda_separation


        loss.backward()

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = training_report(self.tb_writer, self.iteration, Ll1, 
                                                                                         loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), 
                                                                                         self.testing_iterations, self.scene, render, 
                                                                                         (self.pipe, self.background), self.deform, 
                                                                                         self.dataset.load2gpu_on_the_fly, self.progress_bar)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    print("Gaussian numberBEFORE PRUNE", len(self.gaussians.get_xyz))
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.min_opacity, self.scene.cameras_extent, size_threshold)
                    print("Gaussian numberAFTER PRUNE", len(self.gaussians.get_xyz))

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            self._before_optimizer_call(viewpoint_cam, loss)
            self.gaussians.optimizer.step()
            self.gaussians.update_learning_rate(self.iteration)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.iteration)

            self._after_optimizer_call()

        self.deform.update(max(0, self.iteration - self.opt.warm_up))
        if self.iteration in self.saving_iterations or self.iteration == self.opt.warm_up-1:
            self._save_post_step()

        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1
