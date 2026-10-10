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

from scene.light import EnvLight
from utils.loss_utils import l1_loss, ssim, first_order_edge_aware_loss, tv_loss
from gaussian_renderer.render_ir import render_ir
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments.stage2 import Stage2PipelineParams, Stage2OptimizationParams, add_stage2_io_arguments, normalize_render_mode
from scripts.loss_stage2 import apply_loss_preset, build_loss_preset
from utils.train_report_utils import training_report_relight_screen_space
import torch.nn.functional as F


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class Trainer:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations, load_iter) -> None:

        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.stage2_loss = build_loss_preset(opt, "original_ir")
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.n_samples = pipe.diffuse_sample_num

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(deform_type=self.dataset.deform_type,
                                  is_blender=self.dataset.is_blender,
                                  hyper_dim=self.dataset.hyper_dim,
                                  pred_color=self.dataset.pred_color)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=load_iter)
        self.deform.train_setting(opt)

        gs_fea_dim = self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                       fea_dim=gs_fea_dim)

        self.scene = Scene(dataset, self.gaussians, load_iteration=load_iter)
        self.pbr_deformation_xyz = None
        self.pbr_deformation_anchor_sha = None
        if getattr(opt, 'pbr_deformation_anchor', False):
            from scene.pbr_deformation import create_deformation_anchor
            self.pbr_deformation_xyz, anchor_record = create_deformation_anchor(
                args.model_path, self.gaussians.get_xyz, load_iter)
            self.pbr_deformation_anchor_sha = anchor_record['asset_sha256']

        if getattr(opt, 'pbr_albedo_interior_init', False):
            from scene.pbr_material import initialize_native_albedo
            from pathlib import Path
            import json
            initialization = initialize_native_albedo(self.gaussians, opt.pbr_albedo_init_margin)
            initialization.update(source_iteration=load_iter, source_model=args.stage1_model_path,
                source_files=json.loads((Path(args.model_path)/'stage1_source.json').read_text()))
            (Path(args.model_path)/'pbr_albedo_initialization.json').write_text(
                json.dumps(initialization, indent=2)+'\n')

        # lower lr for most of gaussian params
        lr_scale = 0.1
        opt.position_lr_init *= lr_scale
        opt.opacity_lr *= lr_scale
        opt.scaling_lr *= lr_scale
        opt.rotation_lr *= lr_scale
        opt.albedo_after_stage1_lr *= lr_scale
        opt.albedo_rest_lr *= lr_scale
        opt.feature_lr *= lr_scale

        self.gaussians.training_setup(opt)

        # 默认原生材质范围；独立几何preset才开放真实PLY的xyz/rotation。
        allowed = {"albedo_dc", "albedo_dc_stage1", "albedo_rest", "roughness", "opacity"}
        self.pbr_feature_gate = None
        if getattr(opt, "pbr_optimize_feature_gate", False):
            from scene.pbr_feature_gate import NativePBRFeatureGate
            self.pbr_feature_gate = NativePBRFeatureGate(args.model_path, self.gaussians, self.deform, args, load_iter)
            allowed.add("feature")
            for group in self.gaussians.optimizer.param_groups:
                if group['name'] == 'feature':
                    group['params'] = self.pbr_feature_gate.optimizer_parameters(self.gaussians)
        self.pbr_geometry = None
        if getattr(opt, 'pbr_optimize_geometry', False):
            from scene.pbr_geometry import NativePBRGeometry
            self.pbr_geometry = NativePBRGeometry(self.gaussians, opt)
            allowed.update({'xyz', 'rotation'})
            if opt.pbr_freeze_opacity:
                allowed.remove('opacity')
            import json
            from pathlib import Path
            record = dict(self.pbr_geometry.record, source_iteration=load_iter,
                          source_model=args.stage1_model_path, optimizer_groups=sorted(allowed),
                          source_files=json.loads((Path(args.model_path)/'stage1_source.json').read_text()))
            (Path(args.model_path)/'pbr_geometry_source.json').write_text(json.dumps(record, indent=2)+'\n')
        self.pbr_plane_projection = None
        self.pbr_plane_projection_sha = None
        if getattr(opt, 'pbr_plane_projection', False):
            from scene.pbr_plane_projection import create_plane_projection
            self.pbr_plane_projection, plane_record = create_plane_projection(
                args.model_path, self.gaussians, self.deform, self.scene.getTrainCameras(), args, load_iter)
            self.pbr_plane_projection_sha = plane_record['asset_sha256']
        self.pbr_rgb_normal = None
        if getattr(opt, 'pbr_source_local_rotation', False):
            from scene.pbr_rgb_normal import create_rgb_normal
            self.pbr_rgb_normal = create_rgb_normal(args.model_path, self.gaussians, self.deform,
                self.scene.getTrainCameras(), args, load_iter)
        remaining_params = [group for group in self.gaussians.optimizer.param_groups if group['name'] in allowed]
        self.gaussians.optimizer.param_groups.clear()
        self.gaussians.optimizer.param_groups.extend(remaining_params)
        self.gaussians.optimizer.state.clear()
        if self.pbr_geometry is not None:
            retained = {id(p) for g in remaining_params for p in g['params']}
            for value in vars(self.gaussians).values():
                if isinstance(value, torch.nn.Parameter):
                    value.requires_grad_(id(value) in retained)

        # optimize also mlp deform, but keep only color-related head
        before = sum(p.numel() for g in self.deform.optimizer.param_groups for p in g["params"])

        remaining_params = ([] if self.pbr_feature_gate is not None else
                            [group for group in self.deform.optimizer.param_groups if group["name"] in ["mlp_color"]])
        self.deform.optimizer.param_groups.clear()
        self.deform.optimizer.param_groups.extend(remaining_params)
        self.deform.optimizer.state.clear()
        for param_group in self.deform.optimizer.param_groups:
            param_group['lr'] = 0.000075
        
        after = sum(p.numel() for g in self.deform.optimizer.param_groups for p in g["params"])
        print(f"######## Optimizer params: before={before:,} | after={after:,}")

        self.original_train_cameras = self.scene.getTrainCameras()

        self.N = self.gaussians.get_xyz.shape[0]

        self.pbr_lights = None
        self.env_light = None
        self.log_pbr_intensity = None
        self.pbr_intensity_optimizer = None
        if getattr(args, 'pbr_learn_light_intensity', False):
            import math
            self.log_pbr_intensity = torch.nn.Parameter(torch.tensor(
                math.log(args.pbr_light_intensity_scale), device='cuda'))
            self.pbr_intensity_optimizer = torch.optim.Adam([self.log_pbr_intensity],
                lr=opt.pbr_light_intensity_lr)
            self.initial_log_pbr_intensity = self.log_pbr_intensity.detach().clone()
        if getattr(pipe, 'pbr_light_mode', 'envmap') == 'gt_directional':
            from scene.pbr_directional import load_gt_directional_lights
            self.pbr_lights = load_gt_directional_lights(pipe.pbr_gt_lights_path)
            for camera in self.original_train_cameras + self.scene.getTestCameras():
                self.pbr_lights.for_camera(camera, pipe.pbr_light_intensity_scale, device='cpu')
        else:
            self.env_light = EnvLight(path=None, device='cuda',
                                      resolution=[opt.envmap_resolution // 2, opt.envmap_resolution],
                                      max_res=opt.envmap_resolution, init_value=opt.envmap_init_value,
                                      activation=opt.envmap_activation)
            self.env_light.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter + 1

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")

        self.built_bvh = False
        self.edge_image_prepaired = False
        self.load_iter = load_iter

        self.iters_only_diffuse = 2000
        self.iters_only_diffuse_counter = 0


    def train(self, iters=5000):
        if (iters - self.scene.loaded_iter) > 0:
            for i in tqdm.trange(self.scene.loaded_iter, iters):
                self.train_step()

    def train_step(self):
        self.iter_start.record()
        geometry_active = (self.pbr_geometry is not None and
                           self.iteration > self.load_iter + self.opt.pbr_geometry_warmup)
        if self.pbr_geometry is not None:
            steps = self.iteration-self.load_iter-self.opt.pbr_geometry_warmup
            duration = self.opt.iterations-self.load_iter-self.opt.pbr_geometry_warmup
            ramp = max(0., min(steps/1000., 1.))
            decay = max(.1, 1-.9*max(steps, 0)/duration)
            for group in self.gaussians.optimizer.param_groups:
                if group['name'] == 'xyz':
                    group['lr'] = self.opt.pbr_geometry_xyz_lr*ramp*decay
                elif group['name'] == 'rotation':
                    group['lr'] = self.opt.pbr_geometry_rotation_lr*ramp*decay
                elif group['name'] == 'feature' and self.pbr_feature_gate is not None:
                    group['lr'] = self.opt.pbr_feature_gate_lr*ramp*decay

        if self.log_pbr_intensity is not None:
            self.pipe.pbr_light_intensity_scale = self.log_pbr_intensity.exp()

        # Pick a random Camera
        if not self.viewpoint_stack:
            viewpoint_stack = self.original_train_cameras.copy()
            self.viewpoint_stack = viewpoint_stack

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        time_input = fid.unsqueeze(0).expand(self.N, -1)

        # gate候选只对输出feature乘法可导；NN全部冻结，输入仍detach。
        with torch.set_grad_enabled(self.pbr_feature_gate is not None):
            deformation_xyz = self.gaussians.get_xyz.detach()
            if self.pbr_deformation_xyz is not None:
                from scene.pbr_deformation import deformation_input
                deformation_xyz = deformation_input(self.gaussians.get_xyz, self.pbr_deformation_xyz)
            d_values = self.deform.step(deformation_xyz, time_input, iteration=self.iteration,
                                        feature=(self.pbr_feature_gate.output(self.gaussians) if self.pbr_feature_gate is not None
                                                 else self.gaussians.get_binary_feature()),
                                        camera_center=viewpoint_cam.camera_center)
        if self.pbr_rgb_normal is not None:
            d_values = self.pbr_rgb_normal.apply(self.gaussians, d_values)
        if self.pbr_plane_projection is not None:
            d_values = self.pbr_plane_projection.apply(self.gaussians, d_values)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                           d_values['d_scaling'], d_values['d_opacity'], d_values[
                                                               'd_color']

        if not self.built_bvh:
            self.gaussians.build_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)
            self.built_bvh = True
        else:
            self.gaussians.update_bvh(d_rotation=d_rotation, d_xyz=d_xyz, d_scaling=d_scaling)

        # Render relight in screen space
        self.opt.train_ray = True
        render_pkg_re = render_ir(viewpoint_camera=viewpoint_cam,
                                  pc=self.gaussians, pipe=self.pipe,
                                  bg_color=self.background, d_xyz=d_xyz,
                                  d_rotation=d_rotation, d_scaling=d_scaling,
                                  d_opacity=d_opacity, d_color=d_color,
                                  relight=False, env_light=self.env_light, training=True, opt=self.opt)

        self.iters_only_diffuse_counter += 1
        if self.iters_only_diffuse_counter > self.iters_only_diffuse:
            image_non_masked = render_pkg_re["render"]
        else:
            image_non_masked = render_pkg_re["diffuse"]


        gt_image = viewpoint_cam.original_image_train_light.cuda()
        loss_result = self.stage2_loss.compute(self, render_pkg_re, gt_image, image_non_masked)
        Ll1, loss = loss_result.l1, loss_result.total
        if self.log_pbr_intensity is not None:
            intensity_anchor = (self.log_pbr_intensity-self.initial_log_pbr_intensity).square()
            loss = loss + self.opt.lambda_pbr_light_intensity_anchor*intensity_anchor
            if self.tb_writer and self.iteration % 100 == 0:
                self.tb_writer.add_scalar('pbr_light/intensity_scale', self.log_pbr_intensity.exp().item(), self.iteration)
        if geometry_active:
            extra, geometry_terms = self.pbr_geometry.losses(self.gaussians, render_pkg_re, viewpoint_cam)
            loss = loss + extra*ramp
            if self.pbr_feature_gate is not None:
                gate_anchor = self.pbr_feature_gate.loss(self.gaussians)
                loss = loss + self.opt.lambda_pbr_feature_gate_anchor*gate_anchor*ramp
                geometry_terms['pbr_feature_gate_anchor'] = gate_anchor
            if self.pbr_rgb_normal is not None:
                rgb_axis = self.pbr_rgb_normal.loss(self.gaussians)
                loss = loss + self.opt.lambda_pbr_rgb_normal_prior*rgb_axis*ramp
                geometry_terms['pbr_TRAIN_rgb_axis'] = rgb_axis
            if self.tb_writer and self.iteration % 100 == 0:
                for name, value in geometry_terms.items():
                    self.tb_writer.add_scalar('pbr_geometry/'+name, value.item(), self.iteration)
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
                        
            training_report_relight_screen_space(self.tb_writer, self.iteration, Ll1, loss,
                                                 self.iter_start.elapsed_time(self.iter_end),
                                                 self.testing_iterations, self.scene,
                                                 (self.pipe, self.background), self.deform,
                                                 self.dataset.load2gpu_on_the_fly,
                                                 env_light=self.env_light,
                                                 source_deformation_xyz=self.pbr_deformation_xyz,
                                                 plane_projection=self.pbr_plane_projection,
                                                 rgb_normal=self.pbr_rgb_normal)

            if self.iteration in self.saving_iterations or self.iteration == self.opt.warm_up - 1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
                if self.env_light is not None:
                    self.env_light.save_weights(self.args.model_path, self.iteration)
                else:
                    import json
                    from pathlib import Path
                    destination = Path(self.args.model_path)/'pbr_directional'/f'iteration_{self.iteration}'
                    destination.mkdir(parents=True, exist_ok=True)
                    lighting_record = dict(format_version=1, iteration=self.iteration,
                        lights_sha256=self.pbr_lights.sha256, intensity_scale=self.args.pbr_light_intensity_scale)
                    if self.log_pbr_intensity is not None:
                        lighting_record.update(effective_intensity_scale=float(self.log_pbr_intensity.exp()),
                            intensity_fitted_from_train_rgb=True)
                    if self.pbr_deformation_anchor_sha is not None:
                        lighting_record['deformation_anchor_sha256'] = self.pbr_deformation_anchor_sha
                    if self.pbr_plane_projection_sha is not None:
                        lighting_record['plane_projection_sha256'] = self.pbr_plane_projection_sha
                    if self.pbr_feature_gate is not None:
                        lighting_record['feature_gate_metadata_sha256'] = self.pbr_feature_gate.metadata_sha
                    if self.pbr_rgb_normal is not None:
                        lighting_record['rgb_normal_metadata_sha256'] = self.pbr_rgb_normal.metadata_sha
                    if getattr(self.args, 'pbr_rasterizer_backend', 'installed') == 'lumimotion':
                        from scene.pbr_rasterizer import validate_backend_model
                        native = validate_backend_model(self.args.model_path, self.args)
                        lighting_record['rasterizer_source_sha256'] = native['source_sha256']
                    (destination/'lighting.json').write_text(json.dumps(lighting_record, indent=2)+'\n')

            # Optimizer step
            if self.iteration < self.opt.iterations and self.iteration % 4 == 0:
                if geometry_active:
                    torch.nn.utils.clip_grad_norm_([self.gaussians._xyz, self.gaussians._rotation], 10.)
                self.gaussians.optimizer.step()
                if getattr(self.opt, 'pbr_albedo_project_after_step', False):
                    from scene.pbr_material import project_native_albedo
                    project_native_albedo(self.gaussians, self.opt.pbr_albedo_init_margin)
                if self.pbr_geometry is None:
                    self.gaussians.update_learning_rate(self.iteration)
                elif geometry_active:
                    self.pbr_geometry.project(self.gaussians)
                    if self.pbr_feature_gate is not None:
                        self.pbr_feature_gate.project(self.gaussians)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.pbr_intensity_optimizer is not None:
                    self.pbr_intensity_optimizer.step()
                    self.log_pbr_intensity.clamp_(min=-4.605170186, max=4.605170186)
                    self.pbr_intensity_optimizer.zero_grad(set_to_none=True)

                if self.env_light is not None:
                    self.env_light.optimizer.step()
                    self.env_light.optimizer.zero_grad(set_to_none=True)

                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad(set_to_none=True)

        self.progress_bar.set_description(
            "Stage2 material training | EMA loss={}".format('%.7f' % self.ema_loss_for_log))
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
    op = Stage2OptimizationParams(parser)
    pp = Stage2PipelineParams(parser)
    add_stage2_io_arguments(parser)

    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[4,35001, 1, 1000, 5000, 6000, 7_000] + list(range(8000, 300_0001, 2000)))
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[7_000, 10_000, 20_000, 30_000, 40000] + list(range(8000, 300_0001, 2000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = parser.parse_args(sys.argv[1:])
    args.render_mode = normalize_render_mode(args.render_mode)
    apply_loss_preset(args)
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)),
                                       os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.render_mode == "photometric_lambertian":
        from scripts.stage2_runtime import PhotometricTrainer
        args.source_path = os.path.abspath(args.source_path)
        if args.resume_iteration is None and not args.stage1_model_path:
            parser.error("Lambertian 首次运行必须提供 --stage1_model_path 和 --load_iter。")
        PhotometricTrainer(args).train()
        print("Stage2 Lambertian training complete.")
        sys.exit(0)
    if args.resume_iteration is not None:
        parser.error("完整状态恢复仅适用于 photometric_lambertian。")
    if args.pbr_light_mode not in {'envmap', 'gt_directional'}:
        parser.error('未知PBR光照模式。')
    if args.pbr_learn_light_intensity:
        import math
        if (args.pbr_light_mode != 'gt_directional' or not .01 <= args.pbr_light_intensity_scale <= 100 or
                not math.isfinite(args.pbr_light_intensity_lr) or args.pbr_light_intensity_lr <= 0 or
                not math.isfinite(args.lambda_pbr_light_intensity_anchor) or args.lambda_pbr_light_intensity_anchor < 0):
            parser.error('学习光强仅适用于GT方向光，初值在[.01,100]，学习率正且锚定权重非负。')
    if args.pbr_light_mode == 'gt_directional':
        from pathlib import Path
        import json, shutil, math
        from scene.pbr_directional import GTDirectionalLights
        if not args.stage1_model_path or args.load_iter <= 0:
            parser.error('GT方向光PBR首次训练需要独立 --stage1_model_path 和正的 --load_iter。')
        if not args.pbr_gt_lights_path:
            parser.error('GT方向光PBR需要 --pbr_gt_lights_path。')
        if not math.isfinite(args.pbr_light_intensity_scale) or args.pbr_light_intensity_scale <= 0:
            parser.error('PBR方向光强度倍率必须有限且为正。')
        if any(getattr(args, k) != 0 for k in ('d_lower_hemisphere_weight', 'lambda_light', 'lambda_light_smooth')):
            parser.error('已知方向光没有环境光可优化，请将环境光正则权重设为0。')
        if args.diffuse_sample_num <= 0 or args.light_sample_num != 0:
            parser.error('当前PBR方向光的原生SH间接光需要正的diffuse_sample_num和light_sample_num=0。')
        if args.iterations <= args.load_iter:
            parser.error('GT方向光PBR目标iteration必须大于Stage1来源iteration。')
        gt_lights = GTDirectionalLights(args.pbr_gt_lights_path)
        from scene.pbr_directional import validate_directional_training_inputs
        rgb_alpha_audit = validate_directional_training_inputs(args.source_path)
    from scene.pbr_ray_budget import validate_actual_ray_budget
    validate_actual_ray_budget(args)
    from scene.pbr_rasterizer import validate_backend, prepare_backend
    validate_backend(args)
    from scene.pbr_geometry import validate_pbr_geometry
    validate_pbr_geometry(args)
    from scene.pbr_material import validate_albedo_initialization
    validate_albedo_initialization(args)
    from scene.pbr_deformation import validate_deformation_anchor
    validate_deformation_anchor(args)
    from scene.pbr_plane_projection import validate_plane_projection
    validate_plane_projection(args)
    from scene.pbr_feature_gate import validate_feature_gate
    validate_feature_gate(args)
    from scene.pbr_rgb_normal import validate_rgb_normal
    validate_rgb_normal(args)
    if args.stage1_model_path:
        from scripts.stage2_runtime import model_dir, file_sha256
        from pathlib import Path
        import shutil, json
        source = model_dir(args.stage1_model_path, args.deform_type)
        output = Path(args.model_path)
        if output.exists() and any(output.iterdir()):
            parser.error("独立 Stage2 基线输出目录必须为空。")
        if args.load_iter <= 0:
            parser.error("独立 Stage2 基线必须显式指定正的 --load_iter。")
        output.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for relative in ("point_cloud/iteration_{}/point_cloud.ply", "deform/iteration_{}/deform.pth"):
            relative = relative.format(args.load_iter)
            original = source / relative
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            hashes[str(original)] = file_sha256(original)
        (output / "stage1_source.json").write_text(json.dumps(hashes, indent=2)+"\n")
        (output / "stage2_config.json").write_text(json.dumps(vars(args), indent=2)+"\n")
    if args.pbr_light_mode == 'gt_directional':
        (Path(args.model_path)/'rgb_alpha_provenance.json').write_text(json.dumps(rgb_alpha_audit, indent=2)+'\n')
        snapshot = Path(args.model_path)/'pbr_gt_lights.json'
        if gt_lights.path != snapshot.resolve():
            shutil.copy2(gt_lights.path, snapshot)
        args.pbr_gt_lights_path = str(snapshot.resolve())
        (Path(args.model_path)/'stage2_config.json').write_text(json.dumps(vars(args), indent=2)+'\n')
    prepare_backend(args.model_path, args)
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
              testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,
              load_iter=args.load_iter)

    trainer.train(args.iterations)

    # All done
    print("\nTraining complete.")

