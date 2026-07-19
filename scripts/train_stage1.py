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


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations, load_iteration=None) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.requested_render_mode = getattr(pipe, "render_mode", "photometric_lambertian")
        if self.requested_render_mode == "original":
            self.requested_render_mode = "original_sh"
        if self.requested_render_mode not in {"original_sh", "photometric_lambertian"}:
            raise ValueError(f"Unsupported render mode: {self.requested_render_mode}")
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.photometric_renderer = None
        self.photometric_initialized = False

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, 
                                  hyper_dim=self.dataset.hyper_dim,
                                  pred_color=self.dataset.pred_color)
        if load_iteration is not None:
            deform_loaded = self.deform.load_weights(dataset.model_path, iteration=load_iteration)
            if not deform_loaded:
                raise FileNotFoundError(
                    f"Missing deformation checkpoint for iteration {load_iteration} in {dataset.model_path}"
                )
        self.deform.train_setting(opt)

        gs_fea_dim = self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=self.dataset.no_binary_separation,
                                       fea_dim=gs_fea_dim)

        self.scene = Scene(dataset, self.gaussians, load_iteration=load_iteration)
        if self.requested_render_mode == "photometric_lambertian":
            # Allocate the albedo group before optimizer construction so it also
            # follows any densification performed during the SH warm-up.
            if not self.gaussians.use_photometric_albedo:
                self.gaussians.enable_photometric_albedo()
            self.photometric_renderer = PhotometricLambertianRenderer.from_args(
                self.scene.all_timesteps, opt, device="cuda"
            )
            loaded_iter = self.scene.loaded_iter
            if loaded_iter is not None and loaded_iter >= self.opt.photometric_start_iter:
                self.photometric_renderer.load_weights(dataset.model_path, loaded_iter)
                self.photometric_initialized = True
            self.photometric_renderer.training_setup(opt)
        self.gaussians.training_setup(opt)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter + 1
        self.update_photometric_state()

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(
            range(max(opt.iterations - self.iteration + 1, 0)), desc="Training progress"
        )
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)       
        self.T_current = 0.5

    @property
    def photometric_active(self):
        return (
            self.requested_render_mode == "photometric_lambertian"
            and self.iteration >= self.opt.photometric_start_iter
        )

    def initialize_camera_back_ellipse(self):
        train_cameras = self.scene.getTrainCameras()
        if not train_cameras:
            raise RuntimeError("Camera-back light initialization requires a training camera.")
        camera = train_cameras[0]
        rays = camera.rays_d_hw
        center_y = camera.image_height // 2
        center_x = camera.image_width // 2
        forward = F.normalize(rays[center_y, center_x], dim=0)
        right_sample = rays[center_y, min(center_x + 1, camera.image_width - 1)]
        left_sample = rays[center_y, max(center_x - 1, 0)]
        right = right_sample - left_sample
        right = F.normalize(right - torch.dot(right, forward) * forward, dim=0)
        up = F.normalize(torch.cross(forward, right, dim=0), dim=0)
        self.photometric_renderer.initialize_camera_back_ellipse(
            right,
            up,
            forward,
            self.opt.photometric_camera_ellipse_horizontal,
            self.opt.photometric_camera_ellipse_vertical,
            self.opt.photometric_camera_ellipse_back,
            self.opt.photometric_camera_ellipse_phase,
            self.opt.photometric_camera_ellipse_direction_sign,
            self.opt.photometric_camera_ellipse_span,
        )
        print(
            "[photometric init] camera-back ellipse "
            f"at iteration {self.iteration}: "
            f"a={self.opt.photometric_camera_ellipse_horizontal}, "
            f"b={self.opt.photometric_camera_ellipse_vertical}, "
            f"back={self.opt.photometric_camera_ellipse_back}"
        )

    def update_photometric_state(self):
        active = self.photometric_active
        self.pipe.render_mode = "photometric_lambertian" if active else "original_sh"
        if self.photometric_renderer is None:
            return
        if active and not self.photometric_initialized:
            self.gaussians.reset_photometric_albedo_from_sh()
            self.initialize_camera_back_ellipse()
            self.photometric_initialized = True
        self.gaussians.set_photometric_albedo_lr(
            self.opt.photometric_albedo_lr if active else 0.0
        )
        self.photometric_renderer.set_light_lr(
            self.opt.photometric_light_lr if active else 0.0
        )
        # SH coefficients are no longer in the Lambertian RGB graph.
        if active:
            for group in self.gaussians.optimizer.param_groups:
                if group["name"] in {"albedo_dc", "albedo_rest", "albedo_dc_stage1"}:
                    group["lr"] = 0.0

    def log_photometric_stats(self, render_pkg, smooth_loss):
        if self.tb_writer is None or not self.photometric_active:
            return
        directions = self.photometric_renderer.get_all_light_dirs()
        self.tb_writer.add_scalar("photometric/light_smooth1", smooth_loss.item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_norm_mean", directions.norm(dim=-1).mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/albedo_mean", self.gaussians.get_photometric_albedo.mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/ndotl_mean", render_pkg["photometric_ndotl"].mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/shading_mean", render_pkg["photometric_shading"].mean().item(), self.iteration)

    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()
    
    def train_step(self):
        self.update_photometric_state()
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree 
        if not self.photometric_active and self.iteration % self.opt.oneupSHdegree_step == 0:
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
        render_pkg_re = render(
            viewpoint_cam, self.gaussians, self.pipe, self.background,
            d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color,
            photometric_renderer=self.photometric_renderer,
        )
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
        light_smooth_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if self.photometric_active:
            light_smooth_loss = self.photometric_renderer.light_smoothness_loss()
            loss = loss + self.opt.lambda_photometric_light_smooth1 * light_smooth_loss

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

            # d_xyz loss is absent during scalar warm-up/static deformation.
            if torch.is_tensor(d_xyz) and self.opt.d_xyz_loss_weight > 0:
                loss += ((d_xyz**2).mean())*self.opt.d_xyz_loss_weight
            
            # d color loss
            d_color_reg_loss_weight = self.opt.d_color_reg_loss_weight
            if not self.photometric_active and (d_color is not None and torch.is_tensor(d_color)):
                
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
            self.log_photometric_stats(render_pkg_re, light_smooth_loss)

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
                                                                                         photometric_renderer=(
                                                                                             self.photometric_renderer
                                                                                             if self.photometric_active else None
                                                                                         ))
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
                if self.photometric_initialized:
                    self.photometric_renderer.save_weights(self.args.model_path, self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    print("Gaussian numberBEFORE PRUNE", len(self.gaussians.get_xyz))
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    below_cap = self.opt.max_gaussians <= 0 or len(self.gaussians.get_xyz) < self.opt.max_gaussians
                    if below_cap:
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.min_opacity, self.scene.cameras_extent, size_threshold)
                    else:
                        pruned = self.gaussians.prune_only(self.opt.min_opacity, self.scene.cameras_extent, size_threshold)
                        print(f"Gaussian cap reached; skipped densification and pruned {pruned}")
                    print("Gaussian numberAFTER PRUNE", len(self.gaussians.get_xyz))

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity(self.opt.min_opacity)

            prune_only_active = (
                self.opt.prune_from_iter >= 0
                and self.iteration >= self.opt.prune_from_iter
                and (self.opt.prune_until_iter < 0 or self.iteration < self.opt.prune_until_iter)
                and self.iteration >= self.opt.densify_until_iter
                and self.iteration % self.opt.pruning_interval == 0
            )
            if prune_only_active:
                before_prune = len(self.gaussians.get_xyz)
                pruned = self.gaussians.prune_only(self.opt.min_opacity, self.scene.cameras_extent, 20)
                print(f"Gaussian prune-only {before_prune} -> {len(self.gaussians.get_xyz)} (pruned {pruned})")

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)
                if self.photometric_active:
                    self.photometric_renderer.optimizer.step()
                    self.photometric_renderer.optimizer.zero_grad(set_to_none=True)
                
        self.deform.update(max(0, self.iteration - self.opt.warm_up))

        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1

   

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


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # This feature branch opts Stage 1 into the scheduled photometric path;
    # other entrypoints retain PipelineParams' original_sh default.
    parser.set_defaults(render_mode="photometric_lambertian")
    

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[1, 1000, 3000, 5000, 10000, 20000] + list(range(10000, 100_0001, 10000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 1000, 10000, 20000]+ list(range(10000, 100_0001, 10000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument(
        "--load_iteration", type=int, default=None,
        help="Resume from a saved Gaussian/deformation iteration; omitted for a fresh run.",
    )

    args = parser.parse_args(sys.argv[1:])
    required_iterations = [args.iterations]
    if args.render_mode == "photometric_lambertian" and args.photometric_start_iter <= args.iterations:
        required_iterations.append(args.photometric_start_iter)
    args.save_iterations = sorted(set(
        iteration for iteration in args.save_iterations + required_iterations
        if iteration <= args.iterations
    ))
    args.test_iterations = sorted(set(
        iteration for iteration in args.test_iterations + required_iterations
        if iteration <= args.iterations
    ))


    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations, load_iteration=args.load_iteration)


    trainer.train(max(args.iterations - trainer.iteration + 1, 0))
    
    # All done
    print("\nTraining complete.")
