# Stage1 训练损失的对象组合定义。
#
# 设计约定：
#   - 每一种原子损失都是独立对象，权重在构造时固定；
#   - 每一种 loss preset 都是独立类，在 __init__ 中显式列出损失组成；
#   - compute_stage1_loss() 保持历史调用签名，作为训练主循环兼容入口；
#   - LOSS_PRESETS / apply_loss_preset() 保持原命令行选择方式；
#   - PBR 损失路径已停用，旧实现保存在 loss_legacy_20260826.py。
#
# 数值兼容约定：非 PBR 路径保持原损失公式、门控条件与总损失累加顺序。
# 默认 --loss_preset auto 不覆盖任何参数，显式 CLI 参数始终优先。

import math
import sys
from dataclasses import dataclass, field
from random import randint

import torch
import torch.nn.functional as F

from gaussian_renderer import render
from scene.photometric_perlight_pbr import srgb_to_linear
from utils.loss_utils import l1_loss, ssim
from utils.normal_eval_utils import alpha_normalized_normal_map, masked_normal_cosine_loss
from utils.point_utils import depths_to_points


@dataclass
class Stage1LossResult:
    """compute_stage1_loss 的返回值。"""

    loss: torch.Tensor
    # 与训练日志/梯度审计对应的逐项加权损失。
    audit_terms: dict = field(default_factory=dict)
    # 保留旧接口；PBR 路径停用后正常为空。
    pbr_terms: dict = field(default_factory=dict)
    # 未加权 RGB L1，供 training_report 使用。
    l1: torch.Tensor | None = None
    # 未加权光照平滑项，供 log_photometric_stats 使用。
    light_smooth: torch.Tensor | None = None
    # GT-normal oracle 日志素材。
    gt_normal_oracle: dict | None = None


@dataclass
class Stage1LossContext:
    """一次 Stage1 损失计算所需的运行时上下文与只在本次使用的缓存。"""

    tr: object
    viewpoint_cam: object
    render_pkg: dict
    image: torch.Tensor
    gt_image: torch.Tensor
    d_xyz: object
    d_rotation: object
    d_scaling: object
    d_opacity: object
    d_color: object
    _world_normal_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def rendered_world_normal(self) -> tuple[torch.Tensor, torch.Tensor]:
        """渲染并缓存当前视角的独立 world-space 法线图与 alpha。"""
        if self._world_normal_cache is None:
            self._world_normal_cache = render_world_normal_map(
                self.tr,
                self.viewpoint_cam,
                self.render_pkg,
                self.d_xyz,
                self.d_rotation,
                self.d_scaling,
                d_opacity=self.d_opacity,
                d_color=self.d_color,
            )
        return self._world_normal_cache


def render_world_normal_map(
    tr,
    viewpoint_cam,
    render_pkg: dict,
    d_xyz,
    d_rotation,
    d_scaling,
    d_opacity=None,
    d_color=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """以 override color 渲染 world-space 独立法线。"""
    encoded_normal = render_pkg["photometric_normal_raw"] * 0.5 + 0.5
    normal_rendered = render(
        viewpoint_cam,
        tr.gaussians,
        tr.pipe,
        tr.background,
        d_xyz,
        d_rotation,
        d_scaling,
        d_opacity=d_opacity,
        d_color=d_color,
        override_color=encoded_normal,
    )
    predicted_normal = alpha_normalized_normal_map(
        normal_rendered["render"], normal_rendered["rend_alpha"]
    )
    return predicted_normal, normal_rendered["rend_alpha"]


class RGBL1Loss:
    """RGB L1；photometric 阶段可额外乘整体 RGB 权重。"""

    audit_name = "rgb_l1"

    def __init__(self, ratio=1.0, photometric_ratio=1.0):
        self.ratio = float(ratio)
        self.photometric_ratio = float(photometric_ratio)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        raw = l1_loss(context.image, context.gt_image)
        result.l1 = raw
        stage_ratio = (
            self.photometric_ratio if context.tr.photometric_active else 1.0
        )
        return stage_ratio * self.ratio * raw


class RGBDSSIMLoss:
    """RGB DSSIM；返回 ratio * (1 - SSIM)。"""

    audit_name = "rgb_dssim"

    def __init__(self, ratio=1.0, photometric_ratio=1.0):
        self.ratio = float(ratio)
        self.photometric_ratio = float(photometric_ratio)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        stage_ratio = (
            self.photometric_ratio if context.tr.photometric_active else 1.0
        )
        return (
            stage_ratio
            * self.ratio
            * (1.0 - ssim(context.image, context.gt_image))
        )


class GSNormalLoss:
    """GS raster normal 与深度导出表面法线的一致性。"""

    audit_name = "normal"

    def __init__(self, ratio=1.0, start_iter=0):
        self.ratio = float(ratio)
        self.start_iter = int(start_iter)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        active_ratio = (
            self.ratio
            if tr.iteration > self.start_iter and not tr.pbr_active
            else 0.0
        )
        rend_normal = context.render_pkg["rend_normal"]
        surf_normal = context.render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        return active_ratio * normal_error.mean()


class DistortionLoss:
    """2DGS distortion 正则。"""

    audit_name = "distortion"

    def __init__(self, ratio=1.0, start_iter=0):
        self.ratio = float(ratio)
        self.start_iter = int(start_iter)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        active_ratio = (
            self.ratio
            if tr.iteration > self.start_iter and not tr.pbr_active
            else 0.0
        )
        return active_ratio * context.render_pkg["rend_dist"].mean()


class GTNormalOracleLoss:
    """独立法线与外部 GT world normal 的余弦监督。"""

    audit_name = "photometric_gt_normal"

    def __init__(self, ratio=1.0, alpha_threshold=0.5):
        self.ratio = float(ratio)
        self.alpha_threshold = float(alpha_threshold)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        if not tr.photometric_active or self.ratio <= 0.0:
            return None

        encoded_normal = context.render_pkg["photometric_normal"] * 0.5 + 0.5
        normal_rendered = render(
            context.viewpoint_cam,
            tr.gaussians,
            tr.pipe,
            tr.background,
            context.d_xyz,
            context.d_rotation,
            context.d_scaling,
            d_opacity=context.d_opacity,
            d_color=context.d_color,
            override_color=encoded_normal,
        )
        predicted_normal = alpha_normalized_normal_map(
            normal_rendered["render"], normal_rendered["rend_alpha"]
        )
        source_frame, gt_normal, gt_normal_valid = tr.gt_normal_for_camera(
            context.viewpoint_cam
        )
        valid_normal = gt_normal_valid & (
            context.render_pkg["rend_alpha"][0].detach() >= self.alpha_threshold
        )
        cosine_loss = masked_normal_cosine_loss(
            predicted_normal, gt_normal, valid_normal
        )
        result.gt_normal_oracle = {
            "source_frame": source_frame,
            "predicted_normal": predicted_normal,
            "gt_normal": gt_normal,
            "valid_normal": valid_normal,
            "cosine_loss": cosine_loss,
        }
        return self.ratio * cosine_loss


class PhotometricNormalInitLoss:
    """独立法线向 GS 初始化方向的信任域先验。"""

    audit_name = "photometric_normal_init"

    def __init__(self, ratio=1.0):
        self.ratio = float(ratio)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        if not tr.photometric_active or not tr.gaussians.use_photometric_normal:
            return None
        cosine = (
            tr.gaussians.get_photometric_normal
            * tr.gaussians.get_photometric_normal_init
        ).sum(dim=-1).clamp(-1.0, 1.0)
        return self.ratio * (1.0 - cosine).mean()


class PhotometricNormalLiveLoss:
    """当前帧独立法线与深度导出几何法线的一致性。"""

    audit_name = "photometric_normal_live"

    def __init__(
        self,
        ratio=1.0,
        start_iter=1,
        alpha_threshold=0.5,
        log_tensorboard=True,
    ):
        self.ratio = float(ratio)
        self.start_iter = int(start_iter)
        self.alpha_threshold = float(alpha_threshold)
        self.log_tensorboard = bool(log_tensorboard)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        if (
            self.ratio <= 0.0
            or not tr.photometric_active
            or not tr.gaussians.use_photometric_normal
            or tr.iteration < self.start_iter
        ):
            return None

        predicted_normal, normal_alpha = context.rendered_world_normal()
        alpha_map = normal_alpha[0] if normal_alpha.ndim == 3 else normal_alpha
        surf_normal = context.render_pkg["surf_normal"].detach()
        valid = (
            alpha_map >= self.alpha_threshold
        ) & (surf_normal.norm(dim=0) > 1e-3)
        if not bool(valid.any()):
            return None

        target_normal = F.normalize(surf_normal, dim=0)
        cosine = (predicted_normal * target_normal).sum(dim=0).clamp(-1.0, 1.0)
        weighted = self.ratio * (1.0 - cosine)[valid].mean()
        if self.log_tensorboard and tr.tb_writer is not None:
            tr.tb_writer.add_scalar(
                f"photometric_normal_consistency/{self.audit_name}",
                float(weighted.detach().item()),
                tr.iteration,
            )
        return weighted


class PhotometricNormalMVLoss:
    """静态场景独立法线的多视角重投影一致性。"""

    audit_name = "photometric_normal_mv"

    def __init__(
        self,
        ratio=1.0,
        start_iter=1,
        interval=1,
        alpha_threshold=0.5,
        depth_tol=0.1,
        ramp_iters=0,
        log_tensorboard=True,
    ):
        self.ratio = float(ratio)
        self.start_iter = int(start_iter)
        self.interval = max(int(interval), 1)
        self.alpha_threshold = float(alpha_threshold)
        self.depth_tol = float(depth_tol)
        self.ramp_iters = int(ramp_iters)
        self.log_tensorboard = bool(log_tensorboard)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        if (
            self.ratio <= 0.0
            or not tr.photometric_active
            or not tr.gaussians.use_photometric_normal
            or tr.iteration < self.start_iter
            or tr.iteration % self.interval != 0
        ):
            return None

        train_cameras = tr.scene.getTrainCameras()
        if len(train_cameras) < 2:
            return None

        predicted_normal, normal_alpha = context.rendered_world_normal()
        alpha_map = normal_alpha[0] if normal_alpha.ndim == 3 else normal_alpha
        partner_cam = context.viewpoint_cam
        while partner_cam.uid == context.viewpoint_cam.uid:
            partner_cam = train_cameras[randint(0, len(train_cameras) - 1)]

        with torch.no_grad():
            num_points = tr.gaussians.get_xyz.shape[0]
            partner_time = partner_cam.fid.unsqueeze(0).expand(num_points, -1)
            partner_deform = tr.deform.step(
                tr.gaussians.get_xyz,
                partner_time,
                iteration=tr.iteration,
                feature=tr.gaussians.get_binary_feature(
                    eval=False, T=tr.T_current
                ),
                camera_center=partner_cam.camera_center,
            )
            partner_pkg = render(
                partner_cam,
                tr.gaussians,
                tr.pipe,
                tr.background,
                partner_deform["d_xyz"],
                partner_deform["d_rotation"],
                partner_deform["d_scaling"],
                d_opacity=partner_deform["d_opacity"],
                d_color=partner_deform["d_color"],
                photometric_renderer=tr.photometric_renderer,
            )
            partner_normal_map, partner_alpha_map = render_world_normal_map(
                tr,
                partner_cam,
                partner_pkg,
                partner_deform["d_xyz"],
                partner_deform["d_rotation"],
                partner_deform["d_scaling"],
                d_opacity=partner_deform["d_opacity"],
                d_color=partner_deform["d_color"],
            )
            partner_depth = partner_pkg["surf_depth"].detach()

        current_depth = context.render_pkg["surf_depth"].detach()
        height = context.viewpoint_cam.image_height
        width = context.viewpoint_cam.image_width
        world_points = depths_to_points(
            context.viewpoint_cam, current_depth
        ).reshape(height, width, 3)
        partner_w2c = partner_cam.world_view_transform
        camera_points = world_points @ partner_w2c[:3, :3] + partner_w2c[3, :3]
        partner_depth_projected = camera_points[..., 2]
        partner_w = partner_cam.image_width
        partner_h = partner_cam.image_height
        fx = partner_w / (2.0 * math.tan(partner_cam.FoVx / 2.0))
        fy = partner_h / (2.0 * math.tan(partner_cam.FoVy / 2.0))
        safe_depth = partner_depth_projected.clamp_min(1e-6)
        u_coord = fx * camera_points[..., 0] / safe_depth + partner_w / 2.0
        v_coord = fy * camera_points[..., 1] / safe_depth + partner_h / 2.0
        in_bounds = (
            (partner_depth_projected > 1e-4)
            & (u_coord >= 0.0)
            & (u_coord <= partner_w - 1)
            & (v_coord >= 0.0)
            & (v_coord <= partner_h - 1)
        )
        grid = torch.stack(
            (
                (u_coord + 0.5) / partner_w * 2.0 - 1.0,
                (v_coord + 0.5) / partner_h * 2.0 - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0)
        sampled_normal = F.grid_sample(
            partner_normal_map.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        sampled_alpha = F.grid_sample(
            partner_alpha_map.reshape(1, 1, partner_h, partner_w),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0]
        sampled_depth = F.grid_sample(
            partner_depth.reshape(1, 1, partner_h, partner_w),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0]
        valid = (
            in_bounds
            & (alpha_map >= self.alpha_threshold)
            & (sampled_alpha >= self.alpha_threshold)
            & (
                (partner_depth_projected - sampled_depth).abs()
                <= self.depth_tol
            )
            & (sampled_normal.norm(dim=0) > 1e-3)
        )
        if not bool(valid.any()):
            return None

        ramp = 1.0
        if self.ramp_iters > 0:
            ramp = min(
                1.0,
                (tr.iteration - self.start_iter + 1) / self.ramp_iters,
            )
        target_normal = F.normalize(sampled_normal, dim=0)
        cosine = (predicted_normal * target_normal).sum(dim=0).clamp(-1.0, 1.0)
        weighted = self.ratio * ramp * (1.0 - cosine)[valid].mean()
        if self.log_tensorboard and tr.tb_writer is not None:
            tr.tb_writer.add_scalar(
                f"photometric_normal_consistency/{self.audit_name}",
                float(weighted.detach().item()),
                tr.iteration,
            )
        return weighted


class LightSmoothnessLoss:
    """一阶光照轨迹平滑。"""

    audit_name = "light_smooth"

    def __init__(self, ratio=1.0):
        self.ratio = float(ratio)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        if not context.tr.photometric_active:
            return None
        raw = context.tr.photometric_renderer.light_smoothness_loss()
        result.light_smooth = raw
        return self.ratio * raw


class AlphaLoss:
    """渲染 alpha 的 GT-mask 或全一 simulated-mask BCE。"""

    audit_name = "alpha"

    def __init__(self, gt_ratio=1.0, simulated_ratio=0.001):
        self.gt_ratio = float(gt_ratio)
        self.simulated_ratio = float(simulated_ratio)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        if tr.pbr_active:
            return None
        if (
            tr.opt.gt_alpha_mask_as_scene_mask
            and context.viewpoint_cam.gt_alpha_mask is not None
        ):
            gt_alpha_mask = context.viewpoint_cam.gt_alpha_mask.cuda()
            raw = F.binary_cross_entropy(
                context.render_pkg["rend_alpha"][:, None, None],
                gt_alpha_mask.unsqueeze(1).unsqueeze(1),
            )
            return self.gt_ratio * raw

        simulated_mask = torch.ones_like(
            context.render_pkg["rend_alpha"][:, None, None]
        )
        raw = F.binary_cross_entropy(
            context.render_pkg["rend_alpha"][:, None, None],
            simulated_mask,
        )
        return self.simulated_ratio * raw


class DeformationXYZLoss:
    """形变位移 L2 正则。"""

    audit_name = "deformation_xyz"

    def __init__(self, ratio=1.0, warm_up=0):
        self.ratio = float(ratio)
        self.warm_up = int(warm_up)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        if (
            context.tr.pbr_active
            or context.tr.iteration <= self.warm_up
            or not torch.is_tensor(context.d_xyz)
            or self.ratio <= 0.0
        ):
            return None
        return self.ratio * (context.d_xyz**2).mean()


class DeformationColorLoss:
    """原始 SH 阶段 d_color 前三维的 L2 正则。"""

    audit_name = None

    def __init__(self, ratio=1.0, warm_up=0):
        self.ratio = float(ratio)
        self.warm_up = int(warm_up)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        if (
            context.tr.pbr_active
            or context.tr.photometric_active
            or context.tr.iteration <= self.warm_up
            or context.d_color is None
            or not torch.is_tensor(context.d_color)
        ):
            return None
        return self.ratio * context.d_color[:, :3].pow(2.0).mean()


class BinarySeparationLoss:
    """无监督二值化分离 L1。"""

    audit_name = None

    def __init__(self, ratio=1.0, warm_up=0, binarization_warm_up=0):
        self.ratio = float(ratio)
        self.warm_up = int(warm_up)
        self.binarization_warm_up = int(binarization_warm_up)

    def compute(self, context: Stage1LossContext, result: Stage1LossResult):
        tr = context.tr
        if (
            tr.pbr_active
            or tr.iteration <= self.warm_up
            or tr.iteration <= self.binarization_warm_up
            or tr.dataset.no_binary_separation
        ):
            return None
        feature = tr.gaussians.get_binary_feature(eval=False, T=tr.T_current)
        return self.ratio * (feature**1).mean()


def sum_stage1_losses(
    losses: tuple,
    context: Stage1LossContext,
) -> Stage1LossResult:
    """按 preset 声明顺序计算并相加已经加权的原子损失。"""
    zero = context.image.new_zeros(())
    result = Stage1LossResult(
        loss=zero,
        pbr_terms={},
        l1=None,
        light_smooth=zero,
        gt_normal_oracle=None,
    )
    total = None
    for loss_object in losses:
        value = loss_object.compute(context, result)
        if value is None:
            continue
        total = value if total is None else total + value
        audit_name = getattr(loss_object, "audit_name", None)
        if audit_name is not None:
            result.audit_terms[audit_name] = value
    result.loss = zero if total is None else total
    return result


def normal_consistency_terms(
    tr,
    viewpoint_cam,
    render_pkg_re: dict,
    d_xyz,
    d_rotation,
    d_scaling,
    d_opacity,
    d_color,
) -> dict[str, torch.Tensor]:
    """兼容旧工具入口，使用新的 live/MV loss 对象计算加权项。"""
    image = render_pkg_re["render"]
    context = Stage1LossContext(
        tr=tr,
        viewpoint_cam=viewpoint_cam,
        render_pkg=render_pkg_re,
        image=image,
        gt_image=image,
        d_xyz=d_xyz,
        d_rotation=d_rotation,
        d_scaling=d_scaling,
        d_opacity=d_opacity,
        d_color=d_color,
    )
    dummy = Stage1LossResult(
        loss=image.new_zeros(()),
        light_smooth=image.new_zeros(()),
    )
    losses = (
        PhotometricNormalLiveLoss(
            ratio=tr.lambda_photometric_normal_live,
            start_iter=tr.photometric_normal_live_start_iter,
            alpha_threshold=tr.photometric_normal_live_alpha_threshold,
            log_tensorboard=False,
        ),
        PhotometricNormalMVLoss(
            ratio=tr.lambda_photometric_normal_mv,
            start_iter=tr.photometric_normal_mv_start_iter,
            interval=tr.photometric_normal_mv_interval,
            alpha_threshold=tr.photometric_normal_mv_alpha_threshold,
            depth_tol=tr.photometric_normal_mv_depth_tol,
            ramp_iters=tr.photometric_normal_mv_ramp_iters,
            log_tensorboard=False,
        ),
    )
    terms = {}
    for loss_object in losses:
        value = loss_object.compute(context, dummy)
        if value is not None:
            terms[loss_object.audit_name] = value
    return terms


# ---------------------------------------------------------------------------
# 旧 PBR 公式仅作源码参考，不接入当前 preset。
# 完整可运行历史实现见 scripts/loss_legacy_20260826.py。
# ---------------------------------------------------------------------------


def _legacy_pbr_rgb_loss(context: Stage1LossContext):
    """停用的 PBR RGB 四项公式，保留用于历史对照。"""
    tr = context.tr
    opt = tr.opt
    foreground_mask = context.render_pkg["rend_alpha"].detach().clamp(0.0, 1.0)
    if foreground_mask.ndim == 2:
        foreground_mask = foreground_mask.unsqueeze(0)
    denominator = (foreground_mask.sum() * context.image.shape[0]).clamp_min(1.0)
    rgb_mse = F.mse_loss(context.image, context.gt_image)
    foreground_l1 = (
        (context.image - context.gt_image).abs() * foreground_mask
    ).sum() / denominator
    foreground_dssim = 1.0 - ssim(
        context.image * foreground_mask,
        context.gt_image * foreground_mask,
    )
    gt_linear = srgb_to_linear(context.gt_image)
    log_linear = torch.sqrt(
        (
            torch.log1p(context.render_pkg["render_linear"])
            - torch.log1p(gt_linear)
        ).pow(2.0)
        + 1e-6
    ).mean()
    terms = {
        "rgb_mse": rgb_mse,
        "foreground_l1": foreground_l1,
        "foreground_dssim": foreground_dssim,
        "log_linear": log_linear,
    }
    weighted = tr.photometric_rgb_loss_weight * (
        opt.photometric_pbr_loss_mse * rgb_mse
        + opt.photometric_pbr_loss_l1_fg * foreground_l1
        + opt.photometric_pbr_loss_dssim_fg * foreground_dssim
        + opt.photometric_pbr_loss_log_linear * log_linear
    )
    return weighted, terms


# ---------------------------------------------------------------------------
# 独立的 preset 大 loss。每个 __init__ 显式声明其组成，不使用 preset 继承。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossPreset:
    """保留旧公开 metadata 构造类型，具体 preset 不继承它。"""

    description: str
    render_modes: frozenset
    overrides: dict
    terms: tuple


def _opt(opt, name, default):
    return getattr(opt, name, default)


class SHBaselineLossPreset:
    description = (
        "original_sh 基线组合：L1+DSSIM RGB、几何法线/畸变正则"
        "（start_normal_reg 门控）、alpha、d_xyz/d_color、二值化分离。"
    )
    render_modes = frozenset({"original_sh"})
    overrides = {}
    terms = (
        "rgb_l1",
        "rgb_dssim",
        "normal",
        "distortion",
        "alpha",
        "deformation_xyz",
        "d_color_reg",
        "binary_separation",
    )

    def __init__(self, opt):
        self.losses = (
            RGBL1Loss(ratio=1.0 - _opt(opt, "lambda_dssim", 0.2)),
            RGBDSSIMLoss(ratio=_opt(opt, "lambda_dssim", 0.2)),
            GSNormalLoss(
                ratio=_opt(opt, "lambda_gs_normal", 0.02),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            DistortionLoss(
                ratio=_opt(opt, "lambda_dist", 1000.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            AlphaLoss(
                gt_ratio=_opt(opt, "lambda_alpha_loss", 0.1),
                simulated_ratio=0.001,
            ),
            DeformationXYZLoss(
                ratio=_opt(opt, "d_xyz_loss_weight", 0.001),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            DeformationColorLoss(
                ratio=_opt(opt, "d_color_reg_loss_weight", 0.01),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            BinarySeparationLoss(
                ratio=_opt(opt, "lambda_separation", 0.005),
                warm_up=_opt(opt, "warm_up", 1000),
                binarization_warm_up=_opt(opt, "binarization_warm_up", 1000),
            ),
        )

    def compute(self, context):
        return sum_stage1_losses(self.losses, context)


class LambertianDefaultLossPreset:
    description = (
        "photometric_lambertian 默认组合：photometric 加权 L1+DSSIM RGB、"
        "光源平滑、alpha、d_xyz、二值化分离；法线一致性默认关闭。"
    )
    render_modes = frozenset({"photometric_lambertian"})
    overrides = {}
    terms = (
        "rgb_l1",
        "rgb_dssim",
        "light_smooth",
        "alpha",
        "deformation_xyz",
        "binary_separation",
    )

    def __init__(self, opt):
        self.losses = (
            RGBL1Loss(
                ratio=1.0 - _opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            RGBDSSIMLoss(
                ratio=_opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            GSNormalLoss(
                ratio=_opt(opt, "lambda_gs_normal", 0.02),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            DistortionLoss(
                ratio=_opt(opt, "lambda_dist", 1000.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            GTNormalOracleLoss(
                ratio=_opt(opt, "lambda_photometric_gt_normal", 0.0),
                alpha_threshold=_opt(
                    opt, "photometric_gt_normal_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalInitLoss(
                ratio=_opt(opt, "lambda_photometric_normal_init", 0.0),
            ),
            PhotometricNormalLiveLoss(
                ratio=_opt(opt, "lambda_photometric_normal_live", 0.0),
                start_iter=_opt(opt, "photometric_normal_live_start_iter", 500),
                alpha_threshold=_opt(
                    opt, "photometric_normal_live_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalMVLoss(
                ratio=_opt(opt, "lambda_photometric_normal_mv", 0.0),
                start_iter=_opt(opt, "photometric_normal_mv_start_iter", 1000),
                interval=_opt(opt, "photometric_normal_mv_interval", 1),
                alpha_threshold=_opt(
                    opt, "photometric_normal_mv_alpha_threshold", 0.5
                ),
                depth_tol=_opt(opt, "photometric_normal_mv_depth_tol", 0.1),
                ramp_iters=_opt(opt, "photometric_normal_mv_ramp_iters", 2000),
            ),
            LightSmoothnessLoss(
                ratio=_opt(opt, "lambda_photometric_light_smooth1", 0.001),
            ),
            AlphaLoss(
                gt_ratio=_opt(opt, "lambda_alpha_loss", 0.1),
                simulated_ratio=0.001,
            ),
            DeformationXYZLoss(
                ratio=_opt(opt, "d_xyz_loss_weight", 0.001),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            DeformationColorLoss(
                ratio=_opt(opt, "d_color_reg_loss_weight", 0.01),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            BinarySeparationLoss(
                ratio=_opt(opt, "lambda_separation", 0.005),
                warm_up=_opt(opt, "warm_up", 1000),
                binarization_warm_up=_opt(opt, "binarization_warm_up", 1000),
            ),
        )

    def compute(self, context):
        return sum_stage1_losses(self.losses, context)


class LambertianNormal2LossPreset:
    description = (
        "0824-01 二法线一致性消融组合：lambertian_default + 独立法线 live/"
        "多视角一致性，去除 GS 几何自一致法线（lambda_gs_normal=0）；"
        "distortion 保持 start_normal_reg 门控。"
    )
    render_modes = frozenset({"photometric_lambertian"})
    overrides = {
        "lambda_photometric_normal_live": 0.01,
        "lambda_photometric_normal_mv": 0.02,
        "lambda_gs_normal": 0.0,
    }
    terms = (
        "rgb_l1",
        "rgb_dssim",
        "light_smooth",
        "photometric_normal_live",
        "photometric_normal_mv",
        "alpha",
        "deformation_xyz",
        "binary_separation",
    )

    def __init__(self, opt):
        self.losses = (
            RGBL1Loss(
                ratio=1.0 - _opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            RGBDSSIMLoss(
                ratio=_opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            GSNormalLoss(
                ratio=_opt(opt, "lambda_gs_normal", 0.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            DistortionLoss(
                ratio=_opt(opt, "lambda_dist", 1000.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            GTNormalOracleLoss(
                ratio=_opt(opt, "lambda_photometric_gt_normal", 0.0),
                alpha_threshold=_opt(
                    opt, "photometric_gt_normal_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalInitLoss(
                ratio=_opt(opt, "lambda_photometric_normal_init", 0.0),
            ),
            PhotometricNormalLiveLoss(
                ratio=_opt(opt, "lambda_photometric_normal_live", 0.01),
                start_iter=_opt(opt, "photometric_normal_live_start_iter", 500),
                alpha_threshold=_opt(
                    opt, "photometric_normal_live_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalMVLoss(
                ratio=_opt(opt, "lambda_photometric_normal_mv", 0.02),
                start_iter=_opt(opt, "photometric_normal_mv_start_iter", 1000),
                interval=_opt(opt, "photometric_normal_mv_interval", 1),
                alpha_threshold=_opt(
                    opt, "photometric_normal_mv_alpha_threshold", 0.5
                ),
                depth_tol=_opt(opt, "photometric_normal_mv_depth_tol", 0.1),
                ramp_iters=_opt(opt, "photometric_normal_mv_ramp_iters", 2000),
            ),
            LightSmoothnessLoss(
                ratio=_opt(opt, "lambda_photometric_light_smooth1", 0.001),
            ),
            AlphaLoss(
                gt_ratio=_opt(opt, "lambda_alpha_loss", 0.1),
                simulated_ratio=0.001,
            ),
            DeformationXYZLoss(
                ratio=_opt(opt, "d_xyz_loss_weight", 0.001),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            DeformationColorLoss(
                ratio=_opt(opt, "d_color_reg_loss_weight", 0.01),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            BinarySeparationLoss(
                ratio=_opt(opt, "lambda_separation", 0.005),
                warm_up=_opt(opt, "warm_up", 1000),
                binarization_warm_up=_opt(opt, "binarization_warm_up", 1000),
            ),
        )

    def compute(self, context):
        return sum_stage1_losses(self.losses, context)


class LambertianNormal3LossPreset:
    description = (
        "0823-01 三法线一致性组合：lambertian_default + 独立法线 live/"
        "多视角一致性（normal_init 保持 0，即不使用信任域先验）。"
    )
    render_modes = frozenset({"photometric_lambertian"})
    overrides = {
        "lambda_photometric_normal_live": 0.01,
        "lambda_photometric_normal_mv": 0.02,
    }
    terms = (
        "rgb_l1",
        "rgb_dssim",
        "light_smooth",
        "photometric_normal_live",
        "photometric_normal_mv",
        "alpha",
        "deformation_xyz",
        "binary_separation",
    )

    def __init__(self, opt):
        self.losses = (
            RGBL1Loss(
                ratio=1.0 - _opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            RGBDSSIMLoss(
                ratio=_opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            GSNormalLoss(
                ratio=_opt(opt, "lambda_gs_normal", 0.02),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            DistortionLoss(
                ratio=_opt(opt, "lambda_dist", 1000.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            GTNormalOracleLoss(
                ratio=_opt(opt, "lambda_photometric_gt_normal", 0.0),
                alpha_threshold=_opt(
                    opt, "photometric_gt_normal_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalInitLoss(
                ratio=_opt(opt, "lambda_photometric_normal_init", 0.0),
            ),
            PhotometricNormalLiveLoss(
                ratio=_opt(opt, "lambda_photometric_normal_live", 0.01),
                start_iter=_opt(opt, "photometric_normal_live_start_iter", 500),
                alpha_threshold=_opt(
                    opt, "photometric_normal_live_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalMVLoss(
                ratio=_opt(opt, "lambda_photometric_normal_mv", 0.02),
                start_iter=_opt(opt, "photometric_normal_mv_start_iter", 1000),
                interval=_opt(opt, "photometric_normal_mv_interval", 1),
                alpha_threshold=_opt(
                    opt, "photometric_normal_mv_alpha_threshold", 0.5
                ),
                depth_tol=_opt(opt, "photometric_normal_mv_depth_tol", 0.1),
                ramp_iters=_opt(opt, "photometric_normal_mv_ramp_iters", 2000),
            ),
            LightSmoothnessLoss(
                ratio=_opt(opt, "lambda_photometric_light_smooth1", 0.001),
            ),
            AlphaLoss(
                gt_ratio=_opt(opt, "lambda_alpha_loss", 0.1),
                simulated_ratio=0.001,
            ),
            DeformationXYZLoss(
                ratio=_opt(opt, "d_xyz_loss_weight", 0.001),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            DeformationColorLoss(
                ratio=_opt(opt, "d_color_reg_loss_weight", 0.01),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            BinarySeparationLoss(
                ratio=_opt(opt, "lambda_separation", 0.005),
                warm_up=_opt(opt, "warm_up", 1000),
                binarization_warm_up=_opt(opt, "binarization_warm_up", 1000),
            ),
        )

    def compute(self, context):
        return sum_stage1_losses(self.losses, context)


class GSWrappingLossPreset:
    description = (
        "Gaussian Wrapping（From Blobs to Spokes，arXiv:2604.07337，2026）"
        "论文 §4.1 损失权重方案映射：lambertian_default + "
        "L_DN 深度-法线一致性（lambda_gs_normal=0.05）+ "
        "L_N 独立法线/深度一致性（live=0.05）+ "
        "L_gc 多视角重投影一致性（mv=0.02）；论文 L_pc 多视角光度"
        "一致性本仓库无对应项。"
    )
    render_modes = frozenset({"photometric_lambertian"})
    overrides = {
        "lambda_gs_normal": 0.05,
        "lambda_photometric_normal_live": 0.05,
        "lambda_photometric_normal_mv": 0.02,
    }
    terms = (
        "rgb_l1",
        "rgb_dssim",
        "light_smooth",
        "normal",
        "photometric_normal_live",
        "photometric_normal_mv",
        "alpha",
        "deformation_xyz",
        "binary_separation",
    )

    def __init__(self, opt):
        self.losses = (
            RGBL1Loss(
                ratio=1.0 - _opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            RGBDSSIMLoss(
                ratio=_opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 1.0),
            ),
            GSNormalLoss(
                ratio=_opt(opt, "lambda_gs_normal", 0.05),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            DistortionLoss(
                ratio=_opt(opt, "lambda_dist", 1000.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            GTNormalOracleLoss(
                ratio=_opt(opt, "lambda_photometric_gt_normal", 0.0),
                alpha_threshold=_opt(
                    opt, "photometric_gt_normal_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalInitLoss(
                ratio=_opt(opt, "lambda_photometric_normal_init", 0.0),
            ),
            PhotometricNormalLiveLoss(
                ratio=_opt(opt, "lambda_photometric_normal_live", 0.05),
                start_iter=_opt(opt, "photometric_normal_live_start_iter", 500),
                alpha_threshold=_opt(
                    opt, "photometric_normal_live_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalMVLoss(
                ratio=_opt(opt, "lambda_photometric_normal_mv", 0.02),
                start_iter=_opt(opt, "photometric_normal_mv_start_iter", 1000),
                interval=_opt(opt, "photometric_normal_mv_interval", 1),
                alpha_threshold=_opt(
                    opt, "photometric_normal_mv_alpha_threshold", 0.5
                ),
                depth_tol=_opt(opt, "photometric_normal_mv_depth_tol", 0.1),
                ramp_iters=_opt(opt, "photometric_normal_mv_ramp_iters", 2000),
            ),
            LightSmoothnessLoss(
                ratio=_opt(opt, "lambda_photometric_light_smooth1", 0.001),
            ),
            AlphaLoss(
                gt_ratio=_opt(opt, "lambda_alpha_loss", 0.1),
                simulated_ratio=0.001,
            ),
            DeformationXYZLoss(
                ratio=_opt(opt, "d_xyz_loss_weight", 0.001),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            DeformationColorLoss(
                ratio=_opt(opt, "d_color_reg_loss_weight", 0.01),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            BinarySeparationLoss(
                ratio=_opt(opt, "lambda_separation", 0.005),
                warm_up=_opt(opt, "warm_up", 1000),
                binarization_warm_up=_opt(opt, "binarization_warm_up", 1000),
            ),
        )

    def compute(self, context):
        return sum_stage1_losses(self.losses, context)


class N0GTNormalOracleLossPreset:
    description = (
        "N0 oracle 消融：RGB 权重置 0，仅用 GT-normal 余弦监督"
        "（需同时提供 --photometric_gt_normal_dir）。"
    )
    render_modes = frozenset({"photometric_lambertian"})
    overrides = {
        "photometric_rgb_loss_weight": 0.0,
        "lambda_photometric_gt_normal": 1.0,
    }
    terms = ("photometric_gt_normal",)

    def __init__(self, opt):
        self.losses = (
            RGBL1Loss(
                ratio=1.0 - _opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 0.0),
            ),
            RGBDSSIMLoss(
                ratio=_opt(opt, "lambda_dssim", 0.2),
                photometric_ratio=_opt(opt, "photometric_rgb_loss_weight", 0.0),
            ),
            GSNormalLoss(
                ratio=_opt(opt, "lambda_gs_normal", 0.02),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            DistortionLoss(
                ratio=_opt(opt, "lambda_dist", 1000.0),
                start_iter=_opt(opt, "start_normal_reg", 8000),
            ),
            GTNormalOracleLoss(
                ratio=_opt(opt, "lambda_photometric_gt_normal", 1.0),
                alpha_threshold=_opt(
                    opt, "photometric_gt_normal_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalInitLoss(
                ratio=_opt(opt, "lambda_photometric_normal_init", 0.0),
            ),
            PhotometricNormalLiveLoss(
                ratio=_opt(opt, "lambda_photometric_normal_live", 0.0),
                start_iter=_opt(opt, "photometric_normal_live_start_iter", 500),
                alpha_threshold=_opt(
                    opt, "photometric_normal_live_alpha_threshold", 0.5
                ),
            ),
            PhotometricNormalMVLoss(
                ratio=_opt(opt, "lambda_photometric_normal_mv", 0.0),
                start_iter=_opt(opt, "photometric_normal_mv_start_iter", 1000),
                interval=_opt(opt, "photometric_normal_mv_interval", 1),
                alpha_threshold=_opt(
                    opt, "photometric_normal_mv_alpha_threshold", 0.5
                ),
                depth_tol=_opt(opt, "photometric_normal_mv_depth_tol", 0.1),
                ramp_iters=_opt(opt, "photometric_normal_mv_ramp_iters", 2000),
            ),
            LightSmoothnessLoss(
                ratio=_opt(opt, "lambda_photometric_light_smooth1", 0.001),
            ),
            AlphaLoss(
                gt_ratio=_opt(opt, "lambda_alpha_loss", 0.1),
                simulated_ratio=0.001,
            ),
            DeformationXYZLoss(
                ratio=_opt(opt, "d_xyz_loss_weight", 0.001),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            DeformationColorLoss(
                ratio=_opt(opt, "d_color_reg_loss_weight", 0.01),
                warm_up=_opt(opt, "warm_up", 1000),
            ),
            BinarySeparationLoss(
                ratio=_opt(opt, "lambda_separation", 0.005),
                warm_up=_opt(opt, "warm_up", 1000),
                binarization_warm_up=_opt(opt, "binarization_warm_up", 1000),
            ),
        )

    def compute(self, context):
        return sum_stage1_losses(self.losses, context)


class PBRDefaultLossPreset:
    description = "已停用的 photometric_perlight_pbr 损失组合。"
    render_modes = frozenset({"photometric_perlight_pbr"})
    overrides = {}
    terms = (
        "rgb_mse",
        "foreground_l1",
        "foreground_dssim",
        "log_linear",
        "light_smooth",
        "light_smooth2",
        "exposure",
        "normal_residual",
        "roughness_prior",
        "environment_energy",
        "residual",
    )

    def __init__(self, opt):
        self.losses = ()

    def compute(self, context):
        raise AssertionError(
            "photometric_perlight_pbr 已停用；当前 Stage1 loss 仅支持 "
            "original_sh 与 photometric_lambertian。"
        )


class AutoLossPreset:
    """auto 不覆盖参数，只按请求渲染模式组合现有默认 loss。"""

    def __init__(self, opt, render_mode):
        if render_mode == "original":
            render_mode = "original_sh"
        if render_mode == "original_sh":
            self.loss = SHBaselineLossPreset(opt)
        elif render_mode == "photometric_lambertian":
            self.loss = LambertianDefaultLossPreset(opt)
        elif render_mode == "photometric_perlight_pbr":
            raise AssertionError(
                "photometric_perlight_pbr 已停用；auto 不再提供 PBR loss。"
            )
        else:
            raise ValueError(f"Unsupported render mode: {render_mode!r}")

    @property
    def losses(self):
        return self.loss.losses

    def compute(self, context):
        return self.loss.compute(context)


LOSS_PRESETS: dict[str, type] = {
    "sh_baseline": SHBaselineLossPreset,
    "lambertian_default": LambertianDefaultLossPreset,
    "lambertian_normal2": LambertianNormal2LossPreset,
    "lambertian_normal3": LambertianNormal3LossPreset,
    "gs_wrapping": GSWrappingLossPreset,
    "pbr_default": PBRDefaultLossPreset,
    "n0_gt_normal_oracle": N0GTNormalOracleLossPreset,
}


def explicit_cli_keys(argv=None) -> set[str]:
    """返回命令行中显式出现过的参数名（下划线形式）。"""
    if argv is None:
        argv = sys.argv[1:]
    keys = set()
    for token in argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0]
        keys.add(name.replace("-", "_"))
    return keys


def _normalized_render_mode(value) -> str:
    mode = str(value)
    return "original_sh" if mode == "original" else mode


def apply_loss_preset(args, argv=None) -> list[str]:
    """应用 preset 默认参数；显式 CLI 参数优先，PBR 路径直接拒绝。"""
    name = str(getattr(args, "loss_preset", "auto") or "auto").strip().lower()
    render_mode = _normalized_render_mode(
        getattr(args, "render_mode", "original_sh")
    )
    if render_mode == "photometric_perlight_pbr":
        raise AssertionError(
            "photometric_perlight_pbr 已停用；当前 Stage1 loss 仅支持 "
            "original_sh 与 photometric_lambertian。"
        )
    if name == "auto":
        return []

    preset_type = LOSS_PRESETS.get(name)
    if preset_type is None:
        raise ValueError(
            f"Unknown loss_preset {name!r}; available presets: "
            f"{sorted(LOSS_PRESETS)} (or 'auto')."
        )
    if render_mode not in preset_type.render_modes:
        raise ValueError(
            f"loss_preset {name!r} requires render_mode in "
            f"{sorted(preset_type.render_modes)}, got {render_mode!r}."
        )
    explicit = explicit_cli_keys(argv)
    applied = []
    for key, value in preset_type.overrides.items():
        if key in explicit:
            continue
        setattr(args, key, value)
        applied.append(key)
    if applied:
        print(
            f"[loss_preset={name}] applied defaults: "
            + ", ".join(f"{key}={preset_type.overrides[key]}" for key in applied)
        )
    return applied


def build_loss_preset(opt, render_mode):
    """根据最终 opt 构造一次 preset 大 loss。"""
    render_mode = _normalized_render_mode(render_mode)
    if render_mode == "photometric_perlight_pbr":
        raise AssertionError(
            "photometric_perlight_pbr 已停用；无法构造 PBR loss preset。"
        )
    name = str(getattr(opt, "loss_preset", "auto") or "auto").strip().lower()
    if name == "auto":
        return AutoLossPreset(opt, render_mode)
    preset_type = LOSS_PRESETS.get(name)
    if preset_type is None:
        raise ValueError(
            f"Unknown loss_preset {name!r}; available presets: "
            f"{sorted(LOSS_PRESETS)} (or 'auto')."
        )
    if render_mode not in preset_type.render_modes:
        raise ValueError(
            f"loss_preset {name!r} requires render_mode in "
            f"{sorted(preset_type.render_modes)}, got {render_mode!r}."
        )
    return preset_type(opt)


def _prepare_gt_image(tr, viewpoint_cam, gt_image):
    if (
        not tr.pbr_active
        and tr.dataset.white_background
        and viewpoint_cam.gt_alpha_mask is not None
        and tr.opt.gt_alpha_mask_as_scene_mask
    ):
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        return (
            gt_alpha_mask * gt_image
            + (1 - gt_alpha_mask) * tr.background[:, None, None]
        )
    return gt_image


def compute_stage1_loss(
    tr,
    viewpoint_cam,
    render_pkg_re: dict,
    image: torch.Tensor,
    gt_image: torch.Tensor,
    d_xyz,
    d_rotation,
    d_scaling,
    d_opacity,
    d_color,
) -> Stage1LossResult:
    """保持历史签名，通过 Trainer 中预构造的 preset 计算 Stage1 总损失。"""
    render_mode = getattr(tr, "requested_render_mode", None)
    if render_mode is None:
        if tr.pbr_active:
            render_mode = "photometric_perlight_pbr"
        elif tr.photometric_active:
            render_mode = "photometric_lambertian"
        else:
            render_mode = "original_sh"

    preset = getattr(tr, "stage1_loss", None)
    if preset is None:
        # 兼容测试和外部最小 Trainer 对象；正式训练只在 __init__ 构造一次。
        preset = build_loss_preset(tr.opt, render_mode)

    context = Stage1LossContext(
        tr=tr,
        viewpoint_cam=viewpoint_cam,
        render_pkg=render_pkg_re,
        image=image,
        gt_image=_prepare_gt_image(tr, viewpoint_cam, gt_image),
        d_xyz=d_xyz,
        d_rotation=d_rotation,
        d_scaling=d_scaling,
        d_opacity=d_opacity,
        d_color=d_color,
    )
    return preset.compute(context)
