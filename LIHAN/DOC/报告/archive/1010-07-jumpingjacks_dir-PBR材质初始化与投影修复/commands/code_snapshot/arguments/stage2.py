"""Stage2 专用参数；原始 Stage1 参数组保持不变。"""
from arguments import PipelineParams, OptimizationParams


class Stage2PipelineParams(PipelineParams):
    def __init__(self, parser):
        self.render_mode = "original_ir"
        self.pbr_light_mode = "envmap"
        self.pbr_gt_lights_path = ""
        self.pbr_light_intensity_scale = 1.0
        self.pbr_learn_light_intensity = False
        super().__init__(parser)


class Stage2OptimizationParams(OptimizationParams):
    def __init__(self, parser):
        self.loss_preset = "auto"
        self.pbr_optimize_geometry = False
        self.pbr_freeze_opacity = False
        self.pbr_albedo_interior_init = False
        self.pbr_albedo_project_after_step = False
        self.pbr_albedo_init_margin = 0.001
        self.pbr_light_intensity_lr = 0.001
        self.lambda_pbr_light_intensity_anchor = 0.001
        self.pbr_geometry_warmup = 2000
        self.pbr_geometry_xyz_lr = 0.00002
        self.pbr_geometry_rotation_lr = 0.001
        self.pbr_geometry_max_displacement = 0.08
        self.pbr_geometry_max_angle = 60.0
        self.pbr_plane_band = 0.08
        self.lambda_pbr_normal_depth = 0.05
        self.lambda_pbr_plane_normal = 0.5
        self.lambda_pbr_plane_position = 0.1
        self.lambda_pbr_xyz_anchor = 0.001
        self.lambda_pbr_rotation_anchor = 0.001
        self.lambda_pbr_silhouette = 0.1
        self.photometric_light_mode = "learned_directional"
        self.photometric_gt_lights_path = ""
        self.photometric_light_intensity = -1.0
        self.photometric_albedo_lr = 0.001
        self.photometric_material_mode = "per_gaussian"
        self.photometric_palette_size = 4
        self.photometric_palette_init = ""
        self.photometric_light_lr = 0.0001
        self.lambda_photometric_albedo_prior = 0.001
        self.lambda_photometric_light_smooth = 0.001
        self.photometric_log_interval = 100
        self.geometry_xyz_lr = 0.000001
        self.geometry_rotation_lr = 0.00001
        self.geometry_optimize_scale = False
        self.geometry_optimize_opacity = False
        self.geometry_opacity_lr = 0.0001
        self.geometry_opacity_start = 4000
        self.geometry_max_opacity_ratio = 2.0
        self.lambda_geometry_opacity_anchor = 0.001
        self.geometry_initial_stage2_model = ""
        self.geometry_initial_stage2_iteration = -1
        self.geometry_rgb_normal_prior_path = ""
        self.lambda_geometry_rgb_normal_prior = 0.0
        self.photometric_positive_residual_weight = 1.0
        self.photometric_positive_residual_threshold = 0.03
        self.geometry_rgb_prior_warmup = 0
        self.lambda_geometry_distortion = 0.0
        self.geometry_subdivide = False
        self.geometry_subdivision_side = 2
        self.lambda_geometry_coherence = 0.0
        self.geometry_coherence_neighbors = 6
        self.geometry_coherence_tangent_limit = 0.35
        self.geometry_coherence_max_angle = 45.0
        self.geometry_coherence_radius = 3.0
        self.geometry_coherence_source_mask = ""
        self.geometry_refined_coherence = False
        self.geometry_coherence_position_gradient_scale = 1.0
        self.geometry_checkpoint_source_cache = False
        self.geometry_scale_lr = 0.0001
        self.geometry_scale_start = 4000
        self.geometry_max_scale_ratio = 1.25
        self.lambda_geometry_scale_anchor = 0.001
        self.geometry_warmup = 2000
        self.geometry_xyz_start = 4000
        self.geometry_ramp = 1000
        self.geometry_schedule_steps = 20000
        self.geometry_final_lr_ratio = 0.1
        self.geometry_max_displacement = 0.002
        self.geometry_max_angle = 5.0
        self.geometry_accumulate = 4
        self.geometry_validation_stride = 10
        self.lambda_geometry_normal = 0.02
        self.geometry_normal_margin = 0.0
        self.geometry_normal_gradient_mode = "joint"
        self.geometry_source_deformation_mode = "dynamic"
        self.geometry_static_anchor_key = "0002"
        self.geometry_normal_unsupported_weight = 1.0
        self.lambda_geometry_alpha = 0.1
        self.lambda_geometry_anchor = 0.001
        self.lambda_geometry_rotation_anchor = 0.001
        self.lambda_geometry_depth = 0.001
        self.geometry_photo_weight = 1.0
        self.geometry_point_light = False
        self.geometry_area_emitter = False
        self.geometry_area_power_convention = "blender5"
        self.geometry_light_distance = 8.0
        self.geometry_ambient = 0.0
        self.geometry_tonemap = "srgb"
        self.geometry_shading = "screen"
        self.geometry_shadows = False
        self.geometry_shadow_bias = 0.1
        self.geometry_shadow_size = 1.25
        self.geometry_photo_visibility_threshold = 0.0
        self.geometry_photo_min_shading = 0.0
        super().__init__(parser)


def normalize_render_mode(mode):
    return {"photo_lambertian": "photometric_lambertian", "original": "original_ir"}.get(mode, mode)


def add_stage2_io_arguments(parser):
    parser.add_argument("--stage1_model_path", default="", help="只读 Stage1 来源模型目录。")
    parser.add_argument("--resume_iteration", type=int, default=None, help="恢复当前输出目录的完整 Stage2 状态。")
