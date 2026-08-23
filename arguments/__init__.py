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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            # if shorthand:
            #     if t == bool:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
            #     else:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            # else:
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):

        
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.train_light_folder = "chapel_day_4k_32x16_rot0"
        self.test_light_folder = "golden_bay_4k_32x16_rot330"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.deform_type = 'mlp'
        self.hyper_dim = 1 # its for static-dynamic learnable variable
        self.pred_color = True
        self.no_binary_separation = False
        self.load_test_set_only = False
        self.start_frame = 0 #Only used in DNA data
        self.end_frame = -1 #Only used in DNA data
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if not g.model_path.endswith(g.deform_type):
            g.model_path = os.path.join(os.path.dirname(os.path.normpath(g.model_path)), os.path.basename(os.path.normpath(g.model_path)) + f'_{g.deform_type}')
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # Shared default stays on the paper/baseline path. train_stage1 overrides
        # this parser default to photometric_lambertian on the perlight branch.
        self.render_mode = "original_sh"
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.depth_ratio = 1.0 #careful here, it can really breake relighting

        self.light_sample_num = 0 # we do not use it (its for importance sampling which is commented out right now)
        self.diffuse_sample_num = 256
        self.light_t_min = 0.1
        
        # Here options from IRGS ablation. We use full model, so all false.
        self.wo_indirect = False
        self.wo_indirect_relight = False
        self.detach_indirect = False
        self.wo_specular = False

        super().__init__(parser, "Pipeline Parameters")




class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 80_000
        self.warm_up = 1000 #1_000
        self.binarization_warm_up = 1000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.004 #feature is variable for static - dynamic separation #0.0025
        self.opacity_lr = 0.05
        self.roughness_lr = 0.002 #0.001
        self.scaling_lr = 0.002
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100 #100
        self.opacity_reset_interval = 3000 #3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000 #50_000
        self.densify_grad_threshold = 0.0002
        # Optional Stage1 guardrails. Negative values preserve the baseline
        # behavior, where pruning stops together with densification.
        self.prune_from_iter = -1
        self.prune_until_iter = -1
        self.pruning_interval = 1000
        self.max_gaussians = -1
        self.oneupSHdegree_step = 1000
        self.min_opacity = 0.01
        self.start_normal_reg = 8000
        self.lambda_dist = 1000

        self.deform_lr_scale = 1.
        self.gt_alpha_mask_as_scene_mask = False

        self.albedo_lr = 0.01
        self.albedo_after_stage1_lr = 0.01
        self.albedo_rest_lr = self.albedo_lr /20
        self.envmap_cubemap_lr = 0.1
        self.lambda_separation = 0.005 # for supervised  0.02 but tested long time ago
        self.d_color_reg_loss_weight = 0.01
        self.d_xyz_loss_weight = 0.001
        self.d_lower_hemisphere_weight = 0.00001 # mostly gives nicer looking envmap - no bright artifacts on lower hemisphere which is normally almost not supervised.
        self.lambda_alpha_loss = 0.1
        self.envmap_resolution = 32 #128
        self.envmap_init_value = 1.5
        self.envmap_activation = 'exp'

        self.train_ray = False
        self.trace_num_rays = (2**18)*1

        #its possible to test losses from irgs, but defaults to zero:
        self.lambda_roughness_smooth=0.0
        self.lambda_light=0.0
        self.lambda_light_smooth=0.0
        self.lambda_base_color_smooth=0.0

        # Stage 1 per-frame directional-light Lambertian rendering. A selected
        # photometric run keeps the original SH renderer through iteration
        # photometric_start_iter - 1, then switches in a single training process.
        self.photometric_start_iter = 10_001
        self.photometric_albedo_lr = 0.001
        # Independent Lambertian shading-normal parameter. It is initialized
        # from the GS normal at the photometric mode switch and does not affect
        # Gaussian covariance or raster coverage.
        self.photometric_normal_lr = 0.001
        # Optional material-identification schedule. A value <= 0 keeps the
        # historical behavior: normal starts with the photometric phase and
        # albedo remains trainable. Setting both boundaries to the same later
        # iteration produces an albedo-only calibration phase followed by a
        # normal-only phase without changing geometry or light.
        self.photometric_normal_start_iter = -1
        self.photometric_albedo_freeze_iter = -1
        # Trust-region prior around the independent normal initialization.
        # This is not GT-normal supervision; zero preserves the baseline.
        self.lambda_photometric_normal_init = 0.0
        # N0 oracle diagnostics. Defaults preserve every existing experiment.
        # The EXR normals are Blender world-space passes and supervise the
        # alpha-normalized rendered independent normal map directly.
        self.photometric_gt_normal_dir = ""
        self.lambda_photometric_gt_normal = 0.0
        self.photometric_gt_normal_alpha_threshold = 0.5
        self.photometric_gt_normal_log_interval = 25
        # Live consistency between the independent shading normal and the
        # depth-derived geometry normal of the same frame. Serves as the
        # geometry anchor when GT-normal supervision is not used.
        self.lambda_photometric_normal_live = 0.0
        self.photometric_normal_live_start_iter = 500
        self.photometric_normal_live_alpha_threshold = 0.5
        # Static-scene multi-view reprojection consistency for the
        # independent normal: a static surface point must keep the same
        # world normal when reprojected into another training camera.
        self.lambda_photometric_normal_mv = 0.0
        self.photometric_normal_mv_start_iter = 1000
        self.photometric_normal_mv_ramp_iters = 2000
        self.photometric_normal_mv_alpha_threshold = 0.5
        self.photometric_normal_mv_depth_tol = 0.1
        self.photometric_normal_mv_interval = 1
        # Set to zero only for the N0 representation/raster/evaluator oracle;
        # ordinary photometric training keeps the historical RGB weight 1.
        self.photometric_rgb_loss_weight = 1.0
        self.photometric_light_lr = 0.0001
        # Optional delayed-training ablation. Once photometric rendering starts,
        # keep every Gaussian group except photometric albedo frozen; unfreeze
        # deformation and Gaussian rotation at the absolute iterations below.
        self.photometric_staged_training = False
        self.photometric_deform_unfreeze_iter = 22_000
        self.photometric_rotation_unfreeze_iter = 30_000
        self.photometric_deform_lr_scale_after_unfreeze = 0.1
        self.photometric_rotation_lr_scale_after_unfreeze = 0.1
        # learned_directional preserves the original per-frame learned rays.
        # gt_directional converts GT positions to fixed per-frame directions at
        # the scene center, with no distance attenuation. gt_point shades from
        # fixed per-frame world-space light positions with inverse-square falloff.
        self.photometric_light_mode = "learned_directional"
        self.photometric_gt_lights_path = ""
        # Fixed Lambertian irradiance. The directional default is calibrated on
        # only_clothV3 world-space GT RGB/albedo/normal passes. GT point runs
        # must override it with a separately calibrated pre-attenuation source
        # intensity.
        self.photometric_gt_light_intensity = 5.5
        self.photometric_gt_light_color = "1.0,1.0,1.0"
        # Optional piecewise-constant schedule expressed as start_iter:lr
        # entries, for example "1:0.003,10001:0.0003,30001:0.0001".
        # An empty string preserves the constant photometric_light_lr path.
        self.photometric_light_lr_schedule = ""
        self.lambda_photometric_light_smooth1 = 0.001
        self.photometric_normal_axis = "+z"
        # V2 traces a virtual light-position ellipse from the first training
        # camera in world XZ. Pass v1 explicitly for the legacy initialization.
        self.photometric_light_init_version = "v2"
        self.photometric_light_init_v2_axis_ratio = 0.5
        self.photometric_camera_ellipse_horizontal = 0.7
        self.photometric_camera_ellipse_vertical = 0.35
        self.photometric_camera_ellipse_back = 1.0
        self.photometric_camera_ellipse_phase = 0.0
        self.photometric_camera_ellipse_direction_sign = 1
        self.photometric_camera_ellipse_span = 6.283185307179586

        # Constrained per-light PBR ablation. This is a separate render mode;
        # all defaults above continue to define photometric_lambertian.
        self.photometric_pbr_light_samples_train = 4
        self.photometric_pbr_light_samples_eval = 8
        self.photometric_pbr_light_residual_angle_deg = 10.0
        self.photometric_pbr_normal_residual_angle_deg = 10.0
        self.photometric_pbr_residual_log_scale = 0.2
        self.photometric_pbr_environment_init = 0.05
        self.photometric_pbr_angular_radius_init_deg = 2.0
        self.photometric_pbr_angular_radius_max_deg = 12.0
        # Integers make both features explicitly ablatable from the CLI.
        self.photometric_pbr_visibility = 1
        self.photometric_pbr_visibility_backend = "local_knn"
        self.photometric_pbr_shadow_neighbors = 16
        self.photometric_pbr_shadow_strength = 0.5
        self.photometric_pbr_shadow_distance_factor = 6.0
        self.photometric_pbr_residual = 1
        self.photometric_pbr_exposure_lr = 0.001
        self.photometric_pbr_environment_lr = 0.001
        self.photometric_pbr_normal_lr = 0.0001
        self.photometric_pbr_residual_lr = 0.0001
        self.photometric_pbr_roughness_lr = 0.0005
        self.photometric_pbr_loss_mse = 1.0
        self.photometric_pbr_loss_l1_fg = 0.2
        self.photometric_pbr_loss_dssim_fg = 0.1
        self.photometric_pbr_loss_log_linear = 0.05
        self.lambda_photometric_light_smooth2 = 0.01
        self.lambda_photometric_pbr_exposure = 0.001
        self.lambda_photometric_pbr_normal = 0.01
        self.lambda_photometric_pbr_roughness = 0.001
        self.lambda_photometric_pbr_environment = 0.0001
        self.lambda_photometric_pbr_residual = 0.01

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if not args_cmdline.model_path.endswith(args_cmdline.deform_type):
        args_cmdline.model_path = os.path.join(os.path.dirname(os.path.normpath(args_cmdline.model_path)), os.path.basename(os.path.normpath(args_cmdline.model_path)) + f'_{args_cmdline.deform_type}')

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

