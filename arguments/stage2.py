"""Stage2 专用参数；原始 Stage1 参数组保持不变。"""
from arguments import PipelineParams, OptimizationParams


class Stage2PipelineParams(PipelineParams):
    def __init__(self, parser):
        self.render_mode = "original_ir"
        super().__init__(parser)


class Stage2OptimizationParams(OptimizationParams):
    def __init__(self, parser):
        self.loss_preset = "auto"
        self.photometric_light_mode = "learned_directional"
        self.photometric_gt_lights_path = ""
        self.photometric_light_intensity = -1.0
        self.photometric_albedo_lr = 0.001
        self.photometric_light_lr = 0.0001
        self.lambda_photometric_albedo_prior = 0.001
        self.lambda_photometric_light_smooth = 0.001
        self.photometric_log_interval = 100
        super().__init__(parser)


def normalize_render_mode(mode):
    return {"photo_lambertian": "photometric_lambertian", "original": "original_ir"}.get(mode, mode)


def add_stage2_io_arguments(parser):
    parser.add_argument("--stage1_model_path", default="", help="只读 Stage1 来源模型目录。")
    parser.add_argument("--resume_iteration", type=int, default=None, help="恢复当前输出目录的完整 Stage2 状态。")
