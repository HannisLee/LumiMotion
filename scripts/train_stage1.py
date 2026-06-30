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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.train_report_utils import training_report
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from scene.photometric_lambertian import PhotometricLambertianRenderer

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def normalize_render_mode(render_mode):
    if render_mode == "original":
        return "original_sh"
    return render_mode


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations, load_iter=None) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.photometric_stage = getattr(opt, "photometric_stage", "s1d_joint")
        self.render_mode = normalize_render_mode(getattr(pipe, "render_mode", "original_sh"))
        if self.photometric_stage == "s1a_original_warmup":
            self.render_mode = "original_sh"
        self.pipe.render_mode = self.render_mode
        if self.render_mode not in ["original_sh", "photometric_lambertian"]:
            raise ValueError(f"Unsupported render_mode '{self.render_mode}'.")
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.photometric_renderer = None
        self.photometric_checkpoint_loaded = False
        self.photometric_multistart_done = False

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, 
                                  hyper_dim=self.dataset.hyper_dim,
                                  pred_color=self.dataset.pred_color)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=load_iter) if load_iter is not None else False
        self.deform.train_setting(opt)

        gs_fea_dim = self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=self.dataset.no_binary_separation,
                                       fea_dim=gs_fea_dim)

        self.scene = Scene(dataset, self.gaussians, load_iteration=load_iter)
        if self.render_mode == "photometric_lambertian":
            self.gaussians.enable_photometric_albedo()
            self.photometric_renderer = PhotometricLambertianRenderer.from_args(self.scene.all_timesteps, opt, device="cuda")
            if load_iter is not None:
                photometric_path = os.path.join(
                    dataset.model_path, "photometric", f"iteration_{self.scene.loaded_iter}", "photometric.pth"
                )
                if os.path.isfile(photometric_path):
                    self.photometric_renderer.load_weights(dataset.model_path, self.scene.loaded_iter)
                    self.photometric_checkpoint_loaded = True
            self.photometric_renderer.training_setup(opt)
        print(f"Render mode: {self.render_mode}")
        print(f"Photometric stage: {self.photometric_stage}")
        self.gaussians.training_setup(opt)
        self.apply_photometric_stage_lrs()
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter + 1

        self.viewpoint_stack = None
        self.initial_photometric_saved = False
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        remaining_iterations = max(opt.iterations - self.iteration + 1, 0)
        self.progress_bar = tqdm.tqdm(range(remaining_iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)       
        self.T_current = 0.5

    # no gui mode
    def train(self, iters=5000):
        if iters >= self.iteration:
            self.run_photometric_multistart()
            self.save_initial_photometric_weights()
            while self.iteration <= iters:
                self.train_step()

    @staticmethod
    def _set_optimizer_requires_grad(optimizer, requires_grad: bool):
        if optimizer is None:
            return []
        states = []
        seen = set()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param is None or id(param) in seen:
                    continue
                seen.add(id(param))
                states.append((param, param.requires_grad))
                param.requires_grad_(requires_grad)
        return states

    @staticmethod
    def _restore_requires_grad(states):
        for param, requires_grad in states:
            param.requires_grad_(requires_grad)

    def _photometric_multistart_loss(self, viewpoint_cam, trial_iteration: int):
        time_interval = 1 / len(self.scene.all_timesteps)
        fid = viewpoint_cam.fid

        if self.deform.name == 'mlp' or self.deform.name == 'static':
            if trial_iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(trial_iteration)
                d_values = self.deform.step(
                    self.gaussians.get_xyz,
                    time_input + ast_noise,
                    iteration=trial_iteration,
                    feature=self.gaussians.get_binary_feature(eval=False, T=self.T_current),
                    camera_center=viewpoint_cam.camera_center,
                )
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = (
                    d_values['d_xyz'],
                    d_values['d_rotation'],
                    d_values['d_scaling'],
                    d_values['d_opacity'],
                    d_values['d_color'],
                )
        else:
            raise NotImplemented

        render_pkg_re = render(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            d_xyz,
            d_rotation,
            d_scaling,
            d_opacity=d_opacity,
            d_color=d_color,
            photometric_renderer=self.photometric_renderer,
        )
        image = render_pkg_re["render"]
        gt_image = viewpoint_cam.original_image_train_light.cuda()
        if self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        lambda_smooth1 = max(
            float(getattr(self.opt, "lambda_photometric_light_smooth1", 0.0)),
            float(getattr(self.opt, "lambda_photometric_light_smooth", 0.0)),
        )
        if lambda_smooth1 > 0:
            loss = loss + lambda_smooth1 * self.photometric_renderer.light_smoothness_loss(order=1)
        if self.opt.lambda_photometric_light_smooth2 > 0:
            loss = loss + self.opt.lambda_photometric_light_smooth2 * self.photometric_renderer.light_smoothness_loss(order=2)
        if self.opt.lambda_photometric_hemi > 0 and self.opt.photometric_use_hemi_prior:
            loss = loss + self.opt.lambda_photometric_hemi * self.photometric_renderer.hemisphere_loss(
                self.opt.photometric_hemi_axis, self.opt.photometric_hemi_margin
            )
        return loss

    def run_photometric_multistart(self):
        if self.photometric_multistart_done:
            return
        self.photometric_multistart_done = True
        if self.render_mode != "photometric_lambertian" or self.photometric_renderer is None:
            return
        if self.photometric_checkpoint_loaded:
            print("[photometric multistart] skip: loaded existing photometric checkpoint")
            return
        if not getattr(self.opt, "photometric_multistart_enabled", False):
            return

        train_cameras = self.scene.getTrainCameras()
        if len(train_cameras) == 0:
            print("[photometric multistart] skip: no training cameras")
            return

        num_phases = max(1, int(getattr(self.opt, "photometric_multistart_num_phases", 16)))
        short_iters = max(0, int(getattr(self.opt, "photometric_multistart_short_iters", 1000)))
        base_phase = float(getattr(self.opt, "photometric_init_phase", 0.0))
        base_sign = 1 if int(getattr(self.opt, "photometric_init_direction_sign", 1)) >= 0 else -1
        signs = [base_sign]
        if getattr(self.opt, "photometric_multistart_try_reverse_direction", False):
            signs.append(-base_sign)

        phases = [base_phase + 2.0 * np.pi * phase_idx / float(num_phases) for phase_idx in range(num_phases)]
        candidates = [(phase, sign) for sign in signs for phase in phases]
        sample_count = max(short_iters, 1)
        camera_indices = [idx % len(train_cameras) for idx in range(sample_count)]
        score_window = max(1, min(sample_count, max(short_iters // 10, 1)))

        print(
            f"[photometric multistart] candidates={len(candidates)}, "
            f"short_iters={short_iters}, score_window={score_window}"
        )

        freeze_states = []
        freeze_states.extend(self._set_optimizer_requires_grad(self.gaussians.optimizer, False))
        freeze_states.extend(self._set_optimizer_requires_grad(self.deform.optimizer, False))

        best_score = None
        best_state = None
        best_meta = None
        candidate_scores = []
        try:
            for candidate_idx, (phase, direction_sign) in enumerate(candidates):
                self.photometric_renderer.light_model.reset_circle_init(phase=phase, direction_sign=direction_sign)
                self.photometric_renderer.training_setup(self.opt)
                self.photometric_renderer.set_light_lr(getattr(self.opt, "photometric_s1c_light_lr", self.opt.photometric_light_lr))
                losses = []

                for local_iter, cam_idx in enumerate(camera_indices):
                    viewpoint_cam = train_cameras[cam_idx]
                    if self.dataset.load2gpu_on_the_fly:
                        viewpoint_cam.load2device()
                    loss = self._photometric_multistart_loss(viewpoint_cam, self.iteration + local_iter)
                    if short_iters > 0:
                        self.photometric_renderer.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.photometric_renderer.optimizer.step()
                    losses.append(float(loss.detach().item()))
                    if self.dataset.load2gpu_on_the_fly:
                        viewpoint_cam.load2device('cpu')

                score = float(np.mean(losses[-score_window:]))
                score_item = {
                    "candidate": candidate_idx,
                    "phase": float(phase),
                    "direction_sign": int(direction_sign),
                    "score": score,
                }
                candidate_scores.append(score_item)
                print(
                    "[photometric multistart] "
                    f"{candidate_idx + 1}/{len(candidates)} phase={phase:.6f} "
                    f"sign={direction_sign:+d} score={score:.6f}"
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.photometric_renderer.state_dict().items()
                    }
                    best_meta = score_item
        finally:
            self._restore_requires_grad(freeze_states)
            if self.gaussians.optimizer is not None:
                self.gaussians.optimizer.zero_grad(set_to_none=True)
            if self.deform.optimizer is not None:
                self.deform.optimizer.zero_grad(set_to_none=True)
            if self.photometric_renderer.optimizer is not None:
                self.photometric_renderer.optimizer.zero_grad(set_to_none=True)

        if best_state is None or best_meta is None:
            print("[photometric multistart] no valid candidate; keep current initialization")
            return

        self.photometric_renderer.load_state_dict(best_state, strict=True)
        self.photometric_renderer.light_model.init_phase = float(best_meta["phase"])
        self.photometric_renderer.light_model.init_direction_sign = int(best_meta["direction_sign"])
        self.photometric_renderer.multistart_metadata = {
            "enabled": True,
            "num_phases": num_phases,
            "try_reverse_direction": bool(getattr(self.opt, "photometric_multistart_try_reverse_direction", False)),
            "short_iters": short_iters,
            "score_window": score_window,
            "best": best_meta,
            "candidates": candidate_scores,
        }
        self.photometric_renderer.training_setup(self.opt)
        self.apply_photometric_stage_lrs()
        print(
            "[photometric multistart] selected "
            f"phase={best_meta['phase']:.6f} sign={best_meta['direction_sign']:+d} "
            f"score={best_meta['score']:.6f}"
        )
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("photometric/multistart_best_score", best_meta["score"], self.iteration)
            self.tb_writer.add_scalar("photometric/multistart_best_phase", best_meta["phase"], self.iteration)
            self.tb_writer.add_scalar("photometric/multistart_best_direction_sign", best_meta["direction_sign"], self.iteration)

    def save_initial_photometric_weights(self):
        if self.initial_photometric_saved or self.photometric_renderer is None:
            return
        print("\n[ITER {}] Saving initial photometric weights".format(self.iteration))
        self.photometric_renderer.save_weights(self.args.model_path, self.iteration)
        self.initial_photometric_saved = True
    
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
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling,
                               d_opacity=d_opacity, d_color=d_color, photometric_renderer=self.photometric_renderer)
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
        loss_light_smooth1 = torch.zeros((), dtype=loss.dtype, device=loss.device)
        loss_light_smooth2 = torch.zeros((), dtype=loss.dtype, device=loss.device)
        loss_hemi = torch.zeros((), dtype=loss.dtype, device=loss.device)
        loss_albedo_prior = torch.zeros((), dtype=loss.dtype, device=loss.device)

        if self.render_mode == "photometric_lambertian":
            lambda_smooth1 = max(
                float(getattr(self.opt, "lambda_photometric_light_smooth1", 0.0)),
                float(getattr(self.opt, "lambda_photometric_light_smooth", 0.0)),
            )
            lambda_albedo_prior = max(
                float(getattr(self.opt, "lambda_photometric_albedo_prior", 0.0)),
                float(getattr(self.opt, "lambda_photometric_albedo_reg", 0.0)),
            )
            if lambda_smooth1 > 0:
                loss_light_smooth1 = self.photometric_renderer.light_smoothness_loss(order=1)
                loss = loss + lambda_smooth1 * loss_light_smooth1
            if self.opt.lambda_photometric_light_smooth2 > 0:
                loss_light_smooth2 = self.photometric_renderer.light_smoothness_loss(order=2)
                loss = loss + self.opt.lambda_photometric_light_smooth2 * loss_light_smooth2
            if self.opt.lambda_photometric_hemi > 0 and self.opt.photometric_use_hemi_prior:
                loss_hemi = self.photometric_renderer.hemisphere_loss(
                    self.opt.photometric_hemi_axis, self.opt.photometric_hemi_margin
                )
                loss = loss + self.opt.lambda_photometric_hemi * loss_hemi
            if lambda_albedo_prior > 0:
                loss_albedo_prior = self.gaussians.photometric_albedo_reg_loss()
                loss = loss + lambda_albedo_prior * loss_albedo_prior

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

            # d_xyz loss is only meaningful when the deformation model returns tensor offsets.
            if self.opt.d_xyz_loss_weight > 0 and torch.is_tensor(d_xyz):
                loss += ((d_xyz**2).mean()) * self.opt.d_xyz_loss_weight
            
            # d color loss
            d_color_reg_loss_weight = self.opt.d_color_reg_loss_weight
            if self.render_mode != "photometric_lambertian" and (d_color is not None and torch.is_tensor(d_color)):
                
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
            self.log_photometric_stats(
                render_pkg_re,
                loss_light_smooth1,
                loss_light_smooth2,
                loss_hemi,
                loss_albedo_prior,
            )

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = training_report(self.tb_writer, self.iteration, Ll1, 
                                                                                         loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), 
                                                                                         self.testing_iterations, self.scene, render,
                                                                                         (self.pipe, self.background), self.deform,
                                                                                         self.dataset.load2gpu_on_the_fly, self.progress_bar,
                                                                                         photometric_renderer=self.photometric_renderer)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            if self.iteration in self.saving_iterations or self.iteration == self.best_iteration or self.iteration == self.opt.warm_up-1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
                if self.photometric_renderer is not None:
                    self.photometric_renderer.save_weights(self.args.model_path, self.iteration)

            # Densification
            allow_densification = not (
                self.render_mode == "photometric_lambertian"
                and self.photometric_stage == "s1c_light_calib"
            )
            if allow_densification and self.iteration < self.opt.densify_until_iter:
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
            if self.iteration < self.opt.iterations:
                self.apply_photometric_stage_lrs()
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)
                if self.photometric_renderer is not None:
                    self.apply_photometric_stage_lrs()
                    self.photometric_renderer.optimizer.step()
                    self.photometric_renderer.optimizer.zero_grad(set_to_none=True)
                self.apply_photometric_stage_lrs()
                
        self.deform.update(max(0, self.iteration - self.opt.warm_up))

        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1

    def apply_photometric_stage_lrs(self):
        if self.render_mode != "photometric_lambertian":
            return
        if self.gaussians.optimizer is None:
            return

        if self.photometric_stage == "s1c_light_calib":
            gaussian_lrs = {
                "xyz": 0.0,
                "albedo_dc": 0.0,
                "albedo_rest": 0.0,
                "opacity": 0.0,
                "roughness": 0.0,
                "scaling": 0.0,
                "rotation": 0.0,
                "feature": 0.0,
                "photometric_albedo": self.opt.photometric_s1c_albedo_lr,
            }
            deform_lr = 0.0
            light_lr = self.opt.photometric_s1c_light_lr
        elif self.photometric_stage == "s1d_joint":
            gaussian_lrs = {
                "xyz": self.opt.photometric_s1d_position_lr,
                "albedo_dc": 0.0,
                "albedo_rest": 0.0,
                "opacity": self.opt.photometric_s1d_opacity_lr,
                "roughness": 0.0,
                "scaling": self.opt.photometric_s1d_scaling_lr,
                "rotation": self.opt.photometric_s1d_rotation_lr,
                "feature": 0.0,
                "photometric_albedo": self.opt.photometric_s1d_albedo_lr,
            }
            deform_lr = self.opt.photometric_s1d_deformation_lr
            light_lr = self.opt.photometric_s1d_light_lr
        else:
            return

        for group in self.gaussians.optimizer.param_groups:
            if group["name"] in gaussian_lrs:
                group["lr"] = float(gaussian_lrs[group["name"]])
        if self.deform.optimizer is not None:
            for group in self.deform.optimizer.param_groups:
                group["lr"] = float(deform_lr)
        if self.photometric_renderer is not None:
            self.photometric_renderer.set_light_lr(light_lr)

    def log_photometric_stats(
        self,
        render_pkg,
        loss_light_smooth1,
        loss_light_smooth2,
        loss_hemi,
        loss_albedo_prior,
    ):
        if self.render_mode != "photometric_lambertian" or self.tb_writer is None:
            return
        if self.iteration == 1:
            self.tb_writer.add_text("photometric/render_mode", self.render_mode, self.iteration)
            self.tb_writer.add_text("photometric/stage", self.photometric_stage, self.iteration)
        if self.iteration % 10 != 0 and self.iteration not in self.testing_iterations:
            return

        light_dir = render_pkg["photometric_light_dir"].detach()
        all_light_dirs = self.photometric_renderer.get_all_light_dirs().detach()
        albedo = render_pkg["photometric_albedo"].detach()
        normal = render_pkg["photometric_normal"].detach()
        ndotl = render_pkg["photometric_ndotl"].detach()
        shading = render_pkg["photometric_shading"].detach()
        color = render_pkg["photometric_color"].detach()

        self.tb_writer.add_scalar("photometric/light_dir_mean", light_dir.mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_norm", light_dir.norm().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_norm_min", all_light_dirs.norm(dim=-1).min().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_norm_max", all_light_dirs.norm(dim=-1).max().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_norm_mean_all", all_light_dirs.norm(dim=-1).mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_x_mean", all_light_dirs[:, 0].mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_y_mean", all_light_dirs[:, 1].mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_dir_z_mean", all_light_dirs[:, 2].mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/albedo_mean", albedo.mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/albedo_min", albedo.min().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/albedo_max", albedo.max().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/normal_norm_mean", normal.norm(dim=-1).mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/ndotl_mean", ndotl.mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/ndotl_min", ndotl.min().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/ndotl_max", ndotl.max().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/shading_mean", shading.mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/shading_min", shading.min().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/shading_max", shading.max().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/color_mean", color.mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/color_min", color.min().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/color_max", color.max().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/loss_light_smooth1", loss_light_smooth1.detach().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/loss_light_smooth2", loss_light_smooth2.detach().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/loss_hemi", loss_hemi.detach().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/loss_albedo_prior", loss_albedo_prior.detach().item(), self.iteration)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def append_missing_iterations(iterations, required_iterations):
    out = list(iterations)
    for iteration in required_iterations:
        if iteration not in out:
            out.append(iteration)
    return out


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1, 1000, 3000, 5000, 10000, 20000] + list(range(10000, 100_0001, 10000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1000, 10000, 20000]+ list(range(10000, 100_0001, 10000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument("--load_iter", type=int, default=None)

    args = parser.parse_args(sys.argv[1:])
    # Always keep iteration 1 as a checkpoint. In photometric mode the
    # corresponding photometric.pth is the initialized light/albedo state before
    # any optimizer update, which is useful for light trajectory diagnostics.
    args.save_iterations = append_missing_iterations(args.save_iterations, [1, args.iterations])
    args.test_iterations = append_missing_iterations(args.test_iterations, [args.iterations])


    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
                      testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,
                      load_iter=args.load_iter)


    trainer.train(args.iterations)
    
    # All done
    print("\nTraining complete.")
