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

import csv
import math
import os
import torch
from pathlib import Path
from random import randint
from utils.loss_utils import l1_loss
from scripts.loss import apply_loss_preset, compute_stage1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
from utils.gt_normal_utils import (
    load_gt_normal,
    normal_paths,
    source_frame_by_image_name,
)
from utils.normal_eval_utils import (
    normal_angular_error_degrees,
)
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.train_report_utils import training_report
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from scene.photometric_lambertian import (
    PhotometricLambertianRenderer,
    get_gaussian_normal,
)
from scene.photometric_perlight_pbr import (
    PhotometricPerLightPBRRenderer,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def parse_photometric_light_lr_schedule(spec):
    """Parse a piecewise-constant start-iteration light-LR schedule."""
    text = str(spec or "").strip()
    if not text:
        return ()
    schedule = []
    for entry in text.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                "photometric_light_lr_schedule entries must use start_iter:lr, "
                f"got {entry!r}."
            )
        start_iteration = int(parts[0])
        learning_rate = float(parts[1])
        if start_iteration < 1:
            raise ValueError("Light-LR schedule start iterations must be >= 1.")
        if learning_rate < 0.0:
            raise ValueError("Light-LR schedule values must be non-negative.")
        schedule.append((start_iteration, learning_rate))
    if any(current[0] >= following[0] for current, following in zip(schedule, schedule[1:])):
        raise ValueError("Light-LR schedule start iterations must be strictly increasing.")
    return tuple(schedule)


def photometric_material_learning_rates(
    iteration,
    photometric_start_iter,
    normal_start_iter,
    albedo_freeze_iter,
    albedo_lr,
    normal_lr,
):
    """返回当前 iteration 的互斥/联合 albedo 与 normal 学习率。

    非正的 normal_start_iter 与 albedo_freeze_iter 保留历史行为。显式把
    两个边界设为同一 iteration，可得到先 albedo-only、再 normal-only 的
    可辨识性消融，且不需要改变已有 baseline 默认值。
    """
    if iteration < photometric_start_iter:
        return 0.0, 0.0
    resolved_normal_start = (
        photometric_start_iter if normal_start_iter <= 0 else normal_start_iter
    )
    albedo_active = albedo_freeze_iter <= 0 or iteration < albedo_freeze_iter
    normal_active = iteration >= resolved_normal_start
    return (
        float(albedo_lr) if albedo_active else 0.0,
        float(normal_lr) if normal_active else 0.0,
    )


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations, load_iteration=None) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.requested_render_mode = getattr(pipe, "render_mode", "photometric_lambertian")
        if self.requested_render_mode == "original":
            self.requested_render_mode = "original_sh"
        if self.requested_render_mode not in {
            "original_sh",
            "photometric_lambertian",
            "photometric_perlight_pbr",
        }:
            raise ValueError(f"Unsupported render mode: {self.requested_render_mode}")
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.photometric_renderer = None
        self.photometric_initialized = False
        self.photometric_light_init_version = str(
            getattr(opt, "photometric_light_init_version", "v2")
        ).strip().lower()
        if self.photometric_light_init_version not in {"v1", "v2"}:
            raise ValueError(
                "photometric_light_init_version must be 'v1' or 'v2', got "
                f"{self.photometric_light_init_version!r}."
            )
        self.photometric_light_lr_schedule = parse_photometric_light_lr_schedule(
            getattr(opt, "photometric_light_lr_schedule", "")
        )
        self.photometric_light_mode = str(
            getattr(opt, "photometric_light_mode", "learned_directional")
        ).strip().lower()
        if self.photometric_light_mode not in {
            "learned_directional",
            "gt_directional",
            "gt_point",
            "gt_point_direction_only",
        }:
            raise ValueError(
                "photometric_light_mode must be learned_directional, gt_directional, "
                "gt_point, or gt_point_direction_only, got "
                f"{self.photometric_light_mode!r}."
            )
        if (
            self.requested_render_mode == "photometric_perlight_pbr"
            and self.photometric_light_mode != "learned_directional"
        ):
            raise ValueError(
                "photometric_perlight_pbr is a learned per-light experiment; "
                "fixed GT lights are diagnostics only."
            )
        if (
            self.requested_render_mode == "photometric_perlight_pbr"
            and opt.photometric_start_iter < opt.densify_until_iter
        ):
            raise ValueError(
                "photometric_perlight_pbr requires photometric_start_iter >= "
                "densify_until_iter so its per-Gaussian normal residual keeps "
                "a stable checkpoint shape."
            )
        self.photometric_gt_lights_path = str(
            getattr(opt, "photometric_gt_lights_path", "")
        ).strip()
        self.photometric_staged_training = bool(
            getattr(opt, "photometric_staged_training", False)
        )
        self.photometric_normal_start_iter = int(
            getattr(opt, "photometric_normal_start_iter", -1)
        )
        self.photometric_albedo_freeze_iter = int(
            getattr(opt, "photometric_albedo_freeze_iter", -1)
        )
        self.photometric_rgb_loss_weight = float(
            getattr(opt, "photometric_rgb_loss_weight", 1.0)
        )
        self.lambda_photometric_gt_normal = float(
            getattr(opt, "lambda_photometric_gt_normal", 0.0)
        )
        self.photometric_gt_normal_alpha_threshold = float(
            getattr(opt, "photometric_gt_normal_alpha_threshold", 0.5)
        )
        self.photometric_gt_normal_log_interval = int(
            getattr(opt, "photometric_gt_normal_log_interval", 25)
        )
        if self.photometric_rgb_loss_weight < 0.0:
            raise ValueError("photometric_rgb_loss_weight must be non-negative.")
        if self.lambda_photometric_gt_normal < 0.0:
            raise ValueError("lambda_photometric_gt_normal must be non-negative.")
        if not 0.0 <= self.photometric_gt_normal_alpha_threshold <= 1.0:
            raise ValueError(
                "photometric_gt_normal_alpha_threshold must lie in [0, 1]."
            )
        if self.photometric_gt_normal_log_interval < 1:
            raise ValueError("photometric_gt_normal_log_interval must be >= 1.")
        self.lambda_photometric_normal_live = float(
            getattr(opt, "lambda_photometric_normal_live", 0.0)
        )
        self.photometric_normal_live_start_iter = int(
            getattr(opt, "photometric_normal_live_start_iter", 500)
        )
        self.photometric_normal_live_alpha_threshold = float(
            getattr(opt, "photometric_normal_live_alpha_threshold", 0.5)
        )
        self.lambda_photometric_normal_mv = float(
            getattr(opt, "lambda_photometric_normal_mv", 0.0)
        )
        self.photometric_normal_mv_start_iter = int(
            getattr(opt, "photometric_normal_mv_start_iter", 1000)
        )
        self.photometric_normal_mv_ramp_iters = int(
            getattr(opt, "photometric_normal_mv_ramp_iters", 2000)
        )
        self.photometric_normal_mv_alpha_threshold = float(
            getattr(opt, "photometric_normal_mv_alpha_threshold", 0.5)
        )
        self.photometric_normal_mv_depth_tol = float(
            getattr(opt, "photometric_normal_mv_depth_tol", 0.1)
        )
        self.photometric_normal_mv_interval = int(
            getattr(opt, "photometric_normal_mv_interval", 1)
        )
        if self.lambda_photometric_normal_live < 0.0:
            raise ValueError("lambda_photometric_normal_live must be non-negative.")
        if self.lambda_photometric_normal_mv < 0.0:
            raise ValueError("lambda_photometric_normal_mv must be non-negative.")
        if self.photometric_normal_live_start_iter < 1:
            raise ValueError("photometric_normal_live_start_iter must be >= 1.")
        if self.photometric_normal_mv_start_iter < 1:
            raise ValueError("photometric_normal_mv_start_iter must be >= 1.")
        if self.photometric_normal_mv_ramp_iters < 0:
            raise ValueError("photometric_normal_mv_ramp_iters must be non-negative.")
        if self.photometric_normal_mv_depth_tol <= 0.0:
            raise ValueError("photometric_normal_mv_depth_tol must be positive.")
        if self.photometric_normal_mv_interval < 1:
            raise ValueError("photometric_normal_mv_interval must be >= 1.")
        for label, value in (
            (
                "photometric_normal_live_alpha_threshold",
                self.photometric_normal_live_alpha_threshold,
            ),
            (
                "photometric_normal_mv_alpha_threshold",
                self.photometric_normal_mv_alpha_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1].")
        if (
            self.lambda_photometric_normal_live > 0.0
            or self.lambda_photometric_normal_mv > 0.0
        ) and self.requested_render_mode != "photometric_lambertian":
            raise ValueError(
                "photometric normal consistency losses are only supported by "
                "photometric_lambertian."
            )
        self.photometric_normal_live_enabled = self.lambda_photometric_normal_live > 0.0
        self.photometric_normal_mv_enabled = self.lambda_photometric_normal_mv > 0.0
        self.photometric_gt_normal_enabled = self.lambda_photometric_gt_normal > 0.0
        self.photometric_gt_normal_paths = {}
        self.photometric_source_frames = {}
        self.photometric_gt_normal_cache = {}
        if self.photometric_gt_normal_enabled:
            if self.requested_render_mode != "photometric_lambertian":
                raise ValueError(
                    "GT-normal oracle supervision is only supported by "
                    "photometric_lambertian."
                )
            gt_normal_dir = str(
                getattr(opt, "photometric_gt_normal_dir", "")
            ).strip()
            if not gt_normal_dir:
                raise ValueError(
                    "photometric_gt_normal_dir is required when "
                    "lambda_photometric_gt_normal > 0."
                )
            self.photometric_gt_normal_paths = normal_paths(Path(gt_normal_dir))
            self.photometric_source_frames = source_frame_by_image_name(
                Path(dataset.source_path)
            )
            missing_frames = sorted(
                set(self.photometric_source_frames.values())
                - set(self.photometric_gt_normal_paths)
            )
            if missing_frames:
                raise FileNotFoundError(
                    "Missing GT normal EXRs for source frames: "
                    f"{missing_frames[:12]}"
                )
        resolved_normal_start = (
            opt.photometric_start_iter
            if self.photometric_normal_start_iter <= 0
            else self.photometric_normal_start_iter
        )
        if resolved_normal_start < opt.photometric_start_iter:
            raise ValueError(
                "photometric_normal_start_iter must be <= 0 or greater than or "
                "equal to photometric_start_iter."
            )
        if (
            self.photometric_albedo_freeze_iter > 0
            and self.photometric_albedo_freeze_iter < opt.photometric_start_iter
        ):
            raise ValueError(
                "photometric_albedo_freeze_iter must be <= 0 or greater than or "
                "equal to photometric_start_iter."
            )
        self.photometric_deform_unfreeze_iter = int(
            getattr(opt, "photometric_deform_unfreeze_iter", 22_000)
        )
        self.photometric_rotation_unfreeze_iter = int(
            getattr(opt, "photometric_rotation_unfreeze_iter", 30_000)
        )
        self.photometric_deform_lr_scale_after_unfreeze = float(
            getattr(opt, "photometric_deform_lr_scale_after_unfreeze", 0.1)
        )
        self.photometric_rotation_lr_scale_after_unfreeze = float(
            getattr(opt, "photometric_rotation_lr_scale_after_unfreeze", 0.1)
        )
        if self.photometric_staged_training:
            if self.photometric_deform_unfreeze_iter < opt.photometric_start_iter:
                raise ValueError(
                    "photometric_deform_unfreeze_iter must be greater than or "
                    "equal to photometric_start_iter."
                )
            if (
                self.photometric_rotation_unfreeze_iter
                < self.photometric_deform_unfreeze_iter
            ):
                raise ValueError(
                    "photometric_rotation_unfreeze_iter must be greater than or "
                    "equal to photometric_deform_unfreeze_iter."
                )
            if self.photometric_deform_lr_scale_after_unfreeze < 0:
                raise ValueError(
                    "photometric_deform_lr_scale_after_unfreeze must be non-negative."
                )
            if self.photometric_rotation_lr_scale_after_unfreeze < 0:
                raise ValueError(
                    "photometric_rotation_lr_scale_after_unfreeze must be non-negative."
                )
        self._last_photometric_light_lr = None
        self._last_photometric_training_stage = None

        self.tb_writer = prepare_output_and_logger(dataset)
        self.gradient_audit_interval = int(
            getattr(args, "gradient_audit_interval", 0)
        )
        if self.gradient_audit_interval < 0:
            raise ValueError("gradient_audit_interval must be non-negative.")
        self.gradient_audit_path = os.path.join(
            self.args.model_path, "gradient_audit.csv"
        )
        if self.gradient_audit_interval > 0:
            mode = "a" if load_iteration is not None else "w"
            with open(self.gradient_audit_path, mode, newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if mode == "w" or handle.tell() == 0:
                    writer.writerow([
                        "iteration",
                        "frame_id",
                        "loss_term",
                        "parameter_group",
                        "loss_value",
                        "learning_rate",
                        "parameter_l2",
                        "gradient_l2",
                        "gradient_rms",
                        "gradient_mean_abs",
                        "gradient_max_abs",
                        "lr_times_gradient_l2",
                    ])
        self.gt_normal_oracle_path = os.path.join(
            self.args.model_path, "gt_normal_oracle.csv"
        )
        if self.photometric_gt_normal_enabled:
            with open(
                self.gt_normal_oracle_path,
                "a",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.writer(handle)
                if handle.tell() == 0:
                    writer.writerow([
                        "iteration",
                        "source_frame",
                        "valid_pixels",
                        "cosine_loss",
                        "mean_deg",
                        "median_deg",
                        "p95_deg",
                    ])
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
        if self.requested_render_mode in {
            "photometric_lambertian",
            "photometric_perlight_pbr",
        }:
            self.photometric_object_center = (
                self.gaussians.get_xyz.detach().mean(dim=0).clone()
            )
            # Allocate the albedo group before optimizer construction so it also
            # follows any densification performed during the SH warm-up.
            if not self.gaussians.use_photometric_albedo:
                self.gaussians.enable_photometric_albedo()
            if self.requested_render_mode == "photometric_perlight_pbr":
                self.photometric_renderer = PhotometricPerLightPBRRenderer.from_args(
                    self.scene.all_timesteps,
                    num_gaussians=self.gaussians.get_xyz.shape[0],
                    args=opt,
                    device="cuda",
                )
            else:
                self.photometric_renderer = PhotometricLambertianRenderer.from_args(
                    self.scene.all_timesteps, opt, device="cuda"
                )
                if not self.gaussians.use_photometric_normal:
                    self.gaussians.enable_photometric_normal(
                        get_gaussian_normal(
                            self.gaussians.get_rotation,
                            self.photometric_renderer.normal_axis,
                        ).detach()
                    )
            loaded_iter = self.scene.loaded_iter
            if loaded_iter is not None and loaded_iter >= self.opt.photometric_start_iter:
                self.photometric_renderer.load_weights(dataset.model_path, loaded_iter)
                self.photometric_initialized = True
            if (
                self.requested_render_mode == "photometric_perlight_pbr"
                and self.photometric_renderer.requires_local_visibility
            ):
                self.photometric_renderer.initialize_shadow_neighbors(
                    self.gaussians.get_xyz
                )
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
            self.requested_render_mode in {
                "photometric_lambertian",
                "photometric_perlight_pbr",
            }
            and self.iteration >= self.opt.photometric_start_iter
        )

    @property
    def pbr_active(self):
        return (
            self.requested_render_mode == "photometric_perlight_pbr"
            and self.photometric_active
        )

    def gt_normal_for_camera(self, viewpoint_cam):
        """按训练 camera 返回缓存的 world-space GT normal 和 mask。"""
        image_name = viewpoint_cam.image_name_train_light
        if image_name not in self.photometric_source_frames:
            raise KeyError(f"No source_frame mapping for camera {image_name!r}.")
        source_frame = self.photometric_source_frames[image_name]
        key = (source_frame, viewpoint_cam.image_height, viewpoint_cam.image_width)
        if key not in self.photometric_gt_normal_cache:
            self.photometric_gt_normal_cache[key] = load_gt_normal(
                self.photometric_gt_normal_paths[source_frame],
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
            )
        gt_normal, gt_valid = self.photometric_gt_normal_cache[key]
        return (
            source_frame,
            gt_normal.to(viewpoint_cam.fid.device, non_blocking=True),
            gt_valid.to(viewpoint_cam.fid.device, non_blocking=True),
        )

    def log_gt_normal_oracle(
        self,
        source_frame,
        predicted_normal,
        gt_normal,
        valid_mask,
        cosine_loss,
    ):
        should_log = (
            self.iteration == self.scene.loaded_iter + 1
            if self.scene.loaded_iter is not None
            else self.iteration == 1
        )
        should_log = (
            should_log
            or self.iteration == self.opt.iterations
            or self.iteration % self.photometric_gt_normal_log_interval == 0
        )
        if not should_log:
            return
        with torch.no_grad():
            errors = normal_angular_error_degrees(
                predicted_normal, gt_normal, valid_mask
            )[valid_mask]
            values = (
                int(errors.numel()),
                float(cosine_loss.detach().item()),
                float(errors.mean().item()),
                float(errors.median().item()),
                float(torch.quantile(errors, 0.95).item()),
            )
        with open(
            self.gt_normal_oracle_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            csv.writer(handle).writerow([
                self.iteration,
                source_frame,
                *values,
            ])
        print(
            "[gt normal oracle] "
            f"iteration {self.iteration}; frame={source_frame}; "
            f"valid={values[0]}; cosine_loss={values[1]:.7f}; "
            f"mean={values[2]:.4f}deg; median={values[3]:.4f}deg; "
            f"p95={values[4]:.4f}deg"
        )
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                "photometric_gt_normal/cosine_loss", values[1], self.iteration
            )
            self.tb_writer.add_scalar(
                "photometric_gt_normal/mean_deg", values[2], self.iteration
            )

    def photometric_light_lr(self):
        if self.photometric_light_mode in {
            "gt_directional",
            "gt_point",
            "gt_point_direction_only",
        }:
            return 0.0
        learning_rate = float(self.opt.photometric_light_lr)
        for start_iteration, scheduled_lr in self.photometric_light_lr_schedule:
            if self.iteration < start_iteration:
                break
            learning_rate = scheduled_lr
        return learning_rate

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
            "[photometric init] V1 camera-back ellipse "
            f"at iteration {self.iteration}: "
            f"a={self.opt.photometric_camera_ellipse_horizontal}, "
            f"b={self.opt.photometric_camera_ellipse_vertical}, "
            f"back={self.opt.photometric_camera_ellipse_back}"
        )

    def initialize_camera_pose_xz_ellipse(self):
        train_cameras = self.scene.getTrainCameras()
        if not train_cameras:
            raise RuntimeError("V2 light initialization requires a training camera.")
        camera_center = train_cameras[0].camera_center.to(
            device=self.photometric_object_center.device,
            dtype=self.photometric_object_center.dtype,
        )
        axis_ratio = float(self.opt.photometric_light_init_v2_axis_ratio)
        self.photometric_renderer.initialize_camera_pose_xz_ellipse(
            camera_center,
            self.photometric_object_center,
            axis_ratio,
        )
        metadata = self.photometric_renderer.initialization_metadata
        print(
            "[photometric init] V2 camera-pose world-XZ ellipse "
            f"at iteration {self.iteration}: "
            f"axis_ratio={axis_ratio}, "
            f"major_radius={metadata['major_radius']:.6g}, "
            f"minor_radius={metadata['minor_radius']:.6g}"
        )

    def initialize_photometric_light(self):
        if self.photometric_light_mode == "gt_directional":
            self.photometric_renderer.initialize_gt_directional_lights(
                self.photometric_gt_lights_path,
                self.photometric_object_center,
            )
            print(
                "[photometric init] fixed GT directional lights "
                f"at iteration {self.iteration}: "
                f"{os.path.abspath(self.photometric_gt_lights_path)}"
            )
        elif self.photometric_light_mode == "gt_point":
            self.photometric_renderer.initialize_gt_point_lights(
                self.photometric_gt_lights_path,
                self.photometric_object_center,
            )
            print(
                "[photometric init] fixed GT point lights "
                f"at iteration {self.iteration}: "
                f"{os.path.abspath(self.photometric_gt_lights_path)}"
            )
        elif self.photometric_light_mode == "gt_point_direction_only":
            self.photometric_renderer.initialize_gt_point_direction_only_lights(
                self.photometric_gt_lights_path,
                self.photometric_object_center,
            )
            print(
                "[photometric init] fixed GT point-position direction-only lights "
                f"at iteration {self.iteration}: "
                f"{os.path.abspath(self.photometric_gt_lights_path)}"
            )
        elif self.photometric_light_init_version == "v1":
            self.initialize_camera_back_ellipse()
        else:
            self.initialize_camera_pose_xz_ellipse()

    def update_photometric_state(self):
        active = self.photometric_active
        self.pipe.render_mode = self.requested_render_mode if active else "original_sh"
        if self.photometric_renderer is None:
            return
        if active and not self.photometric_initialized:
            self.gaussians.reset_photometric_albedo_from_sh()
            if self.requested_render_mode == "photometric_lambertian":
                self.gaussians.reset_photometric_normal_from_gs(
                    get_gaussian_normal(
                        self.gaussians.get_rotation,
                        self.photometric_renderer.normal_axis,
                    ).detach()
                )
            self.initialize_photometric_light()
            self.photometric_initialized = True
        material_albedo_lr, material_normal_lr = photometric_material_learning_rates(
            self.iteration,
            self.opt.photometric_start_iter,
            self.photometric_normal_start_iter,
            self.photometric_albedo_freeze_iter,
            self.opt.photometric_albedo_lr,
            self.opt.photometric_normal_lr,
        )
        if self.requested_render_mode != "photometric_lambertian":
            material_normal_lr = 0.0
        self.gaussians.set_photometric_albedo_lr(material_albedo_lr)
        self.gaussians.set_photometric_normal_lr(material_normal_lr)
        light_lr = self.photometric_light_lr() if active else 0.0
        if self.requested_render_mode == "photometric_perlight_pbr":
            self.photometric_renderer.set_learning_rates(
                self.opt, active=active, light_lr=light_lr
            )
        else:
            self.photometric_renderer.set_light_lr(light_lr)
        if light_lr != self._last_photometric_light_lr:
            print(f"[photometric light lr] iteration {self.iteration}: {light_lr:.10g}")
            self._last_photometric_light_lr = light_lr
        # SH coefficients are no longer in the Lambertian RGB graph.
        if active:
            for group in self.gaussians.optimizer.param_groups:
                if group["name"] in {"albedo_dc", "albedo_rest", "albedo_dc_stage1"}:
                    group["lr"] = 0.0
        if self.pbr_active:
            self.apply_pbr_training_schedule()
        else:
            self.apply_photometric_training_schedule()

    def apply_pbr_training_schedule(self):
        """Keep Stage1 geometry fixed; optimize only material and light terms."""
        for group in self.gaussians.optimizer.param_groups:
            if group["name"] == "photometric_albedo":
                group["lr"] = float(self.opt.photometric_albedo_lr)
            elif group["name"] == "roughness":
                group["lr"] = float(self.opt.photometric_pbr_roughness_lr)
            else:
                group["lr"] = 0.0
        for group in self.deform.optimizer.param_groups:
            group["lr"] = 0.0
        if self._last_photometric_training_stage != "pbr_fixed_geometry":
            print(
                "[photometric training stage] "
                f"iteration {self.iteration}: pbr_fixed_geometry; "
                f"albedo_lr={self.opt.photometric_albedo_lr:.10g}; "
                f"roughness_lr={self.opt.photometric_pbr_roughness_lr:.10g}"
            )
            self._last_photometric_training_stage = "pbr_fixed_geometry"

    def apply_photometric_training_schedule(self):
        if not self.photometric_active or not self.photometric_staged_training:
            return

        material_lrs = {
            group["name"]: float(group["lr"])
            for group in self.gaussians.optimizer.param_groups
            if group["name"] in {"photometric_albedo", "photometric_normal"}
        }
        material_names = [
            name.removeprefix("photometric_")
            for name in ("photometric_albedo", "photometric_normal")
            if material_lrs.get(name, 0.0) > 0.0
        ]
        material_stage = "_".join(material_names) if material_names else "material_frozen"

        if self.iteration < self.photometric_deform_unfreeze_iter:
            geometry_stage = "fixed_geometry"
        elif self.iteration < self.photometric_rotation_unfreeze_iter:
            geometry_stage = "deformation"
        else:
            geometry_stage = "deformation_rotation"
        stage = f"{material_stage}_{geometry_stage}"

        rotation_lr = 0.0
        if geometry_stage == "deformation_rotation":
            rotation_lr = (
                float(self.opt.rotation_lr)
                * self.photometric_rotation_lr_scale_after_unfreeze
            )

        for group in self.gaussians.optimizer.param_groups:
            name = group["name"]
            if name in {"photometric_albedo", "photometric_normal"}:
                continue
            group["lr"] = rotation_lr if name == "rotation" else 0.0

        deform_lr = 0.0
        if geometry_stage != "fixed_geometry":
            deform_lr = (
                float(self.deform.deform_scheduler_args(self.iteration))
                * self.photometric_deform_lr_scale_after_unfreeze
            )
        for group in self.deform.optimizer.param_groups:
            group["lr"] = deform_lr

        if stage != self._last_photometric_training_stage:
            print(
                "[photometric training stage] "
                f"iteration {self.iteration}: {stage}; "
                f"albedo_lr={material_lrs.get('photometric_albedo', 0.0):.10g}; "
                f"normal_lr={material_lrs.get('photometric_normal', 0.0):.10g}; "
                f"deform_lr={deform_lr:.10g}; rotation_lr={rotation_lr:.10g}"
            )
            self._last_photometric_training_stage = stage

    def log_photometric_stats(self, render_pkg, smooth_loss):
        if self.tb_writer is None or not self.photometric_active:
            return
        directions = self.photometric_renderer.get_all_light_dirs()
        self.tb_writer.add_scalar("photometric/light_smooth1", smooth_loss.item(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_lr", self.photometric_light_lr(), self.iteration)
        self.tb_writer.add_scalar("photometric/light_norm_mean", directions.norm(dim=-1).mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/albedo_mean", self.gaussians.get_photometric_albedo.mean().item(), self.iteration)
        if self.gaussians.use_photometric_normal:
            normal_cosine = (
                self.gaussians.get_photometric_normal
                * self.gaussians.get_photometric_normal_init
            ).sum(dim=-1).clamp(-1.0, 1.0)
            normal_drift = torch.rad2deg(torch.acos(normal_cosine))
            self.tb_writer.add_scalar(
                "photometric/normal_drift_mean_deg",
                normal_drift.mean().item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric/normal_drift_p95_deg",
                torch.quantile(normal_drift, 0.95).item(),
                self.iteration,
            )
        self.tb_writer.add_scalar("photometric/ndotl_mean", render_pkg["photometric_ndotl"].mean().item(), self.iteration)
        self.tb_writer.add_scalar("photometric/shading_mean", render_pkg["photometric_shading"].mean().item(), self.iteration)
        if self.photometric_light_mode in {"gt_point", "gt_point_direction_only"}:
            self.tb_writer.add_scalar(
                "photometric/gt_light_distance_mean",
                render_pkg["photometric_light_distance"].mean().item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric/gt_light_attenuation_mean",
                render_pkg["photometric_light_attenuation"].mean().item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric/gt_light_intensity",
                render_pkg["photometric_light_intensity"].item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric/gt_light_color_mean",
                render_pkg["photometric_light_color"].mean().item(),
                self.iteration,
            )
        if self.pbr_active:
            self.tb_writer.add_scalar(
                "photometric_pbr/visibility_mean",
                render_pkg["photometric_visibility"].mean().item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric_pbr/specular_mean",
                render_pkg["photometric_direct_specular_linear"].mean().item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric_pbr/environment_mean",
                render_pkg["photometric_environment_linear"].mean().item(),
                self.iteration,
            )
            self.tb_writer.add_scalar(
                "photometric_pbr/roughness_mean",
                render_pkg["photometric_roughness"].mean().item(),
                self.iteration,
            )

    def audit_loss_gradients(self, loss_terms, frame_id):
        """记录各损失对关键 canonical Gaussian 参数的加权梯度。"""
        interval = self.gradient_audit_interval
        if interval <= 0:
            return
        should_audit = (
            self.iteration == 1
            or self.iteration == self.opt.iterations
            or self.iteration % interval == 0
        )
        if not should_audit:
            return

        parameter_groups = {
            "position": self.gaussians._xyz,
            "rotation": self.gaussians._rotation,
            "scale": self.gaussians._scaling,
            "opacity": self.gaussians._opacity,
            "albedo": self.gaussians._photometric_albedo,
        }
        if self.gaussians.use_photometric_normal:
            parameter_groups["normal"] = self.gaussians._photometric_normal
        optimizer_names = {
            "position": "xyz",
            "rotation": "rotation",
            "scale": "scaling",
            "opacity": "opacity",
            "albedo": "photometric_albedo",
            "normal": "photometric_normal",
        }
        learning_rates = {
            group["name"]: float(group["lr"])
            for group in self.gaussians.optimizer.param_groups
        }
        differentiable_parameters = {
            name: parameter
            for name, parameter in parameter_groups.items()
            if parameter is not None and parameter.requires_grad
        }
        rows = []
        frame_value = float(frame_id.detach().reshape(-1)[0].item())

        for loss_name, loss_value in loss_terms.items():
            if torch.is_tensor(loss_value) and loss_value.requires_grad:
                gradients = torch.autograd.grad(
                    loss_value,
                    list(differentiable_parameters.values()),
                    retain_graph=True,
                    allow_unused=True,
                )
                gradients_by_name = dict(
                    zip(differentiable_parameters, gradients)
                )
                scalar_loss = float(loss_value.detach().item())
            else:
                gradients_by_name = {}
                scalar_loss = float(loss_value.detach().item()) if torch.is_tensor(loss_value) else float(loss_value)

            for parameter_name, parameter in parameter_groups.items():
                gradient = gradients_by_name.get(parameter_name)
                parameter_detached = parameter.detach()
                parameter_l2 = float(torch.linalg.vector_norm(parameter_detached).item())
                learning_rate = learning_rates.get(
                    optimizer_names[parameter_name], 0.0
                )
                if gradient is None:
                    gradient_l2 = 0.0
                    gradient_rms = 0.0
                    gradient_mean_abs = 0.0
                    gradient_max_abs = 0.0
                else:
                    gradient_detached = gradient.detach()
                    gradient_l2 = float(torch.linalg.vector_norm(gradient_detached).item())
                    gradient_rms = float(
                        torch.sqrt(gradient_detached.square().mean()).item()
                    )
                    gradient_mean_abs = float(gradient_detached.abs().mean().item())
                    gradient_max_abs = float(gradient_detached.abs().max().item())
                rows.append([
                    self.iteration,
                    frame_value,
                    loss_name,
                    parameter_name,
                    scalar_loss,
                    learning_rate,
                    parameter_l2,
                    gradient_l2,
                    gradient_rms,
                    gradient_mean_abs,
                    gradient_max_abs,
                    learning_rate * gradient_l2,
                ])

        with open(self.gradient_audit_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)

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
    


        # Loss：统一由 scripts/loss.py 组装（--loss_preset 选择组合）。
        gt_image = viewpoint_cam.original_image_train_light.cuda()
        loss_result = compute_stage1_loss(
            self,
            viewpoint_cam,
            render_pkg_re,
            image,
            gt_image,
            d_xyz,
            d_rotation,
            d_scaling,
            d_opacity,
            d_color,
        )
        loss = loss_result.loss
        Ll1 = loss_result.l1
        audit_loss_terms = loss_result.audit_terms
        pbr_loss_terms = loss_result.pbr_terms
        light_smooth_loss = loss_result.light_smooth
        if loss_result.gt_normal_oracle is not None:
            oracle = loss_result.gt_normal_oracle
            self.log_gt_normal_oracle(
                oracle["source_frame"],
                oracle["predicted_normal"],
                oracle["gt_normal"],
                oracle["valid_normal"],
                oracle["cosine_loss"],
            )

        audit_loss_terms["total"] = loss
        self.audit_loss_gradients(audit_loss_terms, fid)
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
            if self.tb_writer is not None and self.pbr_active:
                for name, value in pbr_loss_terms.items():
                    self.tb_writer.add_scalar(
                        f"photometric_pbr_loss/{name}",
                        value.item(),
                        self.iteration,
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
                if self.photometric_active and self.photometric_renderer.learns_light:
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
    parser.add_argument(
        "--gradient_audit_interval",
        type=int,
        default=0,
        help="Record per-loss gradients for key Gaussian parameter groups every N steps; 0 disables it.",
    )
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument(
        "--load_iteration", type=int, default=None,
        help="Resume from a saved Gaussian/deformation iteration; omitted for a fresh run.",
    )

    args = parser.parse_args(sys.argv[1:])
    # 按 --loss_preset 批量设定损失组合（仅覆盖未显式给出的参数）。
    apply_loss_preset(args)
    required_iterations = [args.iterations]
    if (
        args.render_mode in {
            "photometric_lambertian",
            "photometric_perlight_pbr",
        }
        and args.photometric_start_iter <= args.iterations
    ):
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
