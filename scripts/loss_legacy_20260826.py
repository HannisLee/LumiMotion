# Stage1 训练损失的集中定义与组装。
#
# 本文件把原先散落在 scripts/train_stage1.py 的 train_step() 中的全部
# Stage1 损失项抽取到一处，方便统一修改与扩展：
#   - compute_stage1_loss(): 训练主循环唯一入口，按原有顺序组装总损失；
#   - render_world_normal_map() / normal_consistency_terms(): 独立法线
#     一致性相关工具与损失项（原 Trainer 方法，行为不变）；
#   - LOSS_PRESETS / apply_loss_preset(): 命名损失组合预设，通过
#     --loss_preset 传参选择（如 sh_baseline / lambertian_normal3）。
#
# 数值等价约定：compute_stage1_loss 的每一项计算与累加顺序与原
# train_step 内联实现完全一致，默认 --loss_preset auto 不改变任何
# 既有实验的行为。渲染器内部的 light_smoothness_loss() /
# regularization_losses() 仍保留在 scene/photometric_*.py，本文件仅调用。

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
    # 与训练日志/梯度审计对应的逐项损失（键名与顺序保持历史约定）。
    audit_terms: dict = field(default_factory=dict)
    # PBR 分支额外的未加权损失项（写入 TensorBoard photometric_pbr_loss/*）。
    pbr_terms: dict = field(default_factory=dict)
    # 未加权的渲染 RGB 与 GT 的 L1，供训练报告（training_report）使用。
    l1: torch.Tensor | None = None
    # 未加权的光滑项，供 log_photometric_stats 记录。
    light_smooth: torch.Tensor | None = None
    # GT-normal 分支的 oracle 日志素材；未启用时为 None。
    gt_normal_oracle: dict | None = None


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
    """以 override color 渲染 world-space 独立法线。

    返回解码后的单位 world 法线图 [3,H,W] 与渲染 alpha [1,H,W]。
    使用 photometric_normal_raw（world-space），不同相机的结果可直接比较。
    """
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
    """独立法线的免 GT 一致性项。

    - photometric_normal_live: 独立法线图与同帧深度导出几何法线的余弦。
    - photometric_normal_mv: 静态场景多视角重投影余弦。静态表面点在
      重投影到另一个训练相机时，其 world 法线应保持一致。
    """
    losses: dict[str, torch.Tensor] = {}
    live_active = (
        tr.photometric_normal_live_enabled
        and tr.iteration >= tr.photometric_normal_live_start_iter
    )
    mv_active = (
        tr.photometric_normal_mv_enabled
        and tr.iteration >= tr.photometric_normal_mv_start_iter
        and tr.iteration % max(tr.photometric_normal_mv_interval, 1) == 0
    )
    if not (live_active or mv_active):
        return losses

    predicted_normal, normal_alpha = render_world_normal_map(
        tr,
        viewpoint_cam,
        render_pkg_re,
        d_xyz,
        d_rotation,
        d_scaling,
        d_opacity=d_opacity,
        d_color=d_color,
    )
    alpha_map = normal_alpha[0] if normal_alpha.ndim == 3 else normal_alpha

    if live_active:
        surf_normal = render_pkg_re["surf_normal"].detach()
        valid_live = (
            alpha_map >= tr.photometric_normal_live_alpha_threshold
        ) & (surf_normal.norm(dim=0) > 1e-3)
        if bool(valid_live.any()):
            target_normal = F.normalize(surf_normal, dim=0)
            cosine = (predicted_normal * target_normal).sum(dim=0).clamp(-1.0, 1.0)
            losses["photometric_normal_live"] = (
                tr.lambda_photometric_normal_live
                * (1.0 - cosine)[valid_live].mean()
            )

    if mv_active:
        train_cameras = tr.scene.getTrainCameras()
        if len(train_cameras) >= 2:
            partner_cam = viewpoint_cam
            while partner_cam.uid == viewpoint_cam.uid:
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
            current_depth = render_pkg_re["surf_depth"].detach()
            height = viewpoint_cam.image_height
            width = viewpoint_cam.image_width
            world_points = depths_to_points(
                viewpoint_cam, current_depth
            ).reshape(height, width, 3)
            partner_w2c = partner_cam.world_view_transform
            camera_points = (
                world_points @ partner_w2c[:3, :3] + partner_w2c[3, :3]
            )
            partner_depth_projected = camera_points[..., 2]
            partner_w = partner_cam.image_width
            partner_h = partner_cam.image_height
            fx = partner_w / (2.0 * math.tan(partner_cam.FoVx / 2.0))
            fy = partner_h / (2.0 * math.tan(partner_cam.FoVy / 2.0))
            safe_depth = partner_depth_projected.clamp_min(1e-6)
            u_coord = (
                fx * camera_points[..., 0] / safe_depth + partner_w / 2.0
            )
            v_coord = (
                fy * camera_points[..., 1] / safe_depth + partner_h / 2.0
            )
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
            valid_mv = (
                in_bounds
                & (alpha_map >= tr.photometric_normal_mv_alpha_threshold)
                & (sampled_alpha >= tr.photometric_normal_mv_alpha_threshold)
                & (
                    (partner_depth_projected - sampled_depth).abs()
                    <= tr.photometric_normal_mv_depth_tol
                )
                & (sampled_normal.norm(dim=0) > 1e-3)
            )
            if bool(valid_mv.any()):
                ramp = 1.0
                if tr.photometric_normal_mv_ramp_iters > 0:
                    ramp = min(
                        1.0,
                        (
                            tr.iteration
                            - tr.photometric_normal_mv_start_iter
                            + 1
                        )
                        / tr.photometric_normal_mv_ramp_iters,
                    )
                target_normal = F.normalize(sampled_normal, dim=0)
                cosine = (predicted_normal * target_normal).sum(dim=0).clamp(
                    -1.0, 1.0
                )
                losses["photometric_normal_mv"] = (
                    tr.lambda_photometric_normal_mv
                    * ramp
                    * (1.0 - cosine)[valid_mv].mean()
                )
    return losses


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
    """按原 train_step 的顺序组装 Stage1 总损失。

    参数 tr 为 train_stage1.Trainer 实例，提供 opt/iteration/各启用标志
    与渲染器引用；本函数不改变其状态（GT-normal 的 oracle 日志素材通过
    返回值交回主循环记录）。
    """
    opt = tr.opt

    # GT 图像准备：白背景场景下用 GT alpha 合成背景。
    if (
        not tr.pbr_active
        and tr.dataset.white_background
        and viewpoint_cam.gt_alpha_mask is not None
        and opt.gt_alpha_mask_as_scene_mask
    ):
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * tr.background[:, None, None]

    # 几何法线一致性 / 畸变正则（PBR 阶段关闭）。
    lambda_normal = (
        opt.lambda_gs_normal
        if tr.iteration > opt.start_normal_reg and not tr.pbr_active
        else 0.0
    )
    lambda_dist = (
        opt.lambda_dist
        if tr.iteration > opt.start_normal_reg and not tr.pbr_active
        else 0.0
    )
    rend_dist = render_pkg_re["rend_dist"]
    rend_normal = render_pkg_re["rend_normal"]
    surf_normal = render_pkg_re["surf_normal"]
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    normal_loss = lambda_normal * (normal_error).mean()
    dist_loss = lambda_dist * (rend_dist).mean()

    # RGB 重建损失：PBR 四项组合 或 SH 的 L1+DSSIM。
    Ll1 = l1_loss(image, gt_image)
    pbr_loss_terms = {}
    if tr.pbr_active:
        # 几何已固定，其 detach 的渲染 alpha 是稳定的前景权重，
        # 不引入 GT mask 监督。
        foreground_mask = render_pkg_re["rend_alpha"].detach().clamp(0.0, 1.0)
        if foreground_mask.ndim == 2:
            foreground_mask = foreground_mask.unsqueeze(0)
        foreground_denominator = (
            foreground_mask.sum() * image.shape[0]
        ).clamp_min(1.0)
        rgb_mse = F.mse_loss(image, gt_image)
        foreground_l1 = (
            (image - gt_image).abs() * foreground_mask
        ).sum() / foreground_denominator
        foreground_dssim = 1.0 - ssim(
            image * foreground_mask,
            gt_image * foreground_mask,
        )
        gt_linear = srgb_to_linear(gt_image)
        log_linear = torch.sqrt(
            (
                torch.log1p(render_pkg_re["render_linear"])
                - torch.log1p(gt_linear)
            ).pow(2.0)
            + 1e-6
        ).mean()
        pbr_loss_terms.update(
            rgb_mse=rgb_mse,
            foreground_l1=foreground_l1,
            foreground_dssim=foreground_dssim,
            log_linear=log_linear,
        )
        loss_img = tr.photometric_rgb_loss_weight * (
            opt.photometric_pbr_loss_mse * rgb_mse
            + opt.photometric_pbr_loss_l1_fg * foreground_l1
            + opt.photometric_pbr_loss_dssim_fg * foreground_dssim
            + opt.photometric_pbr_loss_log_linear * log_linear
        )
        weighted_l1 = None
        weighted_dssim = None
    else:
        rgb_weight = (
            tr.photometric_rgb_loss_weight
            if tr.photometric_active
            else 1.0
        )
        weighted_l1 = rgb_weight * (1.0 - opt.lambda_dssim) * Ll1
        weighted_dssim = (
            rgb_weight
            * opt.lambda_dssim
            * (1.0 - ssim(image, gt_image))
        )
        loss_img = weighted_l1 + weighted_dssim
    loss = loss_img + normal_loss + dist_loss
    audit_loss_terms = {
        "rgb_l1": weighted_l1 if not tr.pbr_active else loss_img,
        "rgb_dssim": weighted_dssim if not tr.pbr_active else loss.new_zeros(()),
        "normal": normal_loss,
        "distortion": dist_loss,
    }

    # GT-normal oracle 监督（仅诊断/消融用，默认关闭）。
    gt_normal_oracle = None
    if tr.photometric_active and tr.photometric_gt_normal_enabled:
        encoded_normal = render_pkg_re["photometric_normal"] * 0.5 + 0.5
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
        source_frame, gt_normal, gt_normal_valid = tr.gt_normal_for_camera(
            viewpoint_cam
        )
        valid_normal = gt_normal_valid & (
            render_pkg_re["rend_alpha"][0].detach()
            >= tr.photometric_gt_normal_alpha_threshold
        )
        gt_normal_cosine_loss = masked_normal_cosine_loss(
            predicted_normal, gt_normal, valid_normal
        )
        weighted_gt_normal_loss = (
            tr.lambda_photometric_gt_normal * gt_normal_cosine_loss
        )
        loss = loss + weighted_gt_normal_loss
        audit_loss_terms["photometric_gt_normal"] = weighted_gt_normal_loss
        gt_normal_oracle = {
            "source_frame": source_frame,
            "predicted_normal": predicted_normal,
            "gt_normal": gt_normal,
            "valid_normal": valid_normal,
            "cosine_loss": gt_normal_cosine_loss,
        }

    # 独立法线向其 GS 初始方向的信任域先验。
    if tr.photometric_active and tr.gaussians.use_photometric_normal:
        normal_init_cosine = (
            tr.gaussians.get_photometric_normal
            * tr.gaussians.get_photometric_normal_init
        ).sum(dim=-1).clamp(-1.0, 1.0)
        photometric_normal_init_loss = (
            float(opt.lambda_photometric_normal_init)
            * (1.0 - normal_init_cosine).mean()
        )
        loss = loss + photometric_normal_init_loss
        audit_loss_terms["photometric_normal_init"] = photometric_normal_init_loss

    # 免 GT 的独立法线一致性项（live + 多视角重投影）。
    if (
        tr.photometric_active
        and tr.gaussians.use_photometric_normal
        and (
            tr.photometric_normal_live_enabled
            or tr.photometric_normal_mv_enabled
        )
    ):
        consistency_losses = normal_consistency_terms(
            tr,
            viewpoint_cam,
            render_pkg_re,
            d_xyz,
            d_rotation,
            d_scaling,
            d_opacity,
            d_color,
        )
        for name, term in consistency_losses.items():
            loss = loss + term
            audit_loss_terms[name] = term
            if tr.tb_writer is not None:
                tr.tb_writer.add_scalar(
                    f"photometric_normal_consistency/{name}",
                    float(term.detach().item()),
                    tr.iteration,
                )

    # 光源平滑正则。
    light_smooth_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
    if tr.photometric_active:
        light_smooth_loss = tr.photometric_renderer.light_smoothness_loss()
        weighted_light_smooth_loss = (
            opt.lambda_photometric_light_smooth1 * light_smooth_loss
        )
        loss = loss + weighted_light_smooth_loss
        audit_loss_terms["light_smooth"] = weighted_light_smooth_loss

    # PBR 正则项（曝光/法线残差/粗糙率/环境能量/残差光源等）。
    if tr.pbr_active:
        regularizers = tr.photometric_renderer.regularization_losses(
            tr.gaussians.get_rough
        )
        regularizer_weights = {
            "light_smooth2": opt.lambda_photometric_light_smooth2,
            "exposure": opt.lambda_photometric_pbr_exposure,
            "normal_residual": opt.lambda_photometric_pbr_normal,
            "roughness_prior": opt.lambda_photometric_pbr_roughness,
            "environment_energy": opt.lambda_photometric_pbr_environment,
            "residual": opt.lambda_photometric_pbr_residual,
        }
        for name, weight in regularizer_weights.items():
            loss = loss + float(weight) * regularizers[name]
        pbr_loss_terms.update(
            {f"regularizer_{name}": value for name, value in regularizers.items()}
        )

    # mask / alpha 损失。
    if (
        not tr.pbr_active
        and opt.gt_alpha_mask_as_scene_mask
        and viewpoint_cam.gt_alpha_mask is not None
    ):
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        alpha_loss = F.binary_cross_entropy(render_pkg_re['rend_alpha'][:, None, None], gt_alpha_mask.unsqueeze(1).unsqueeze(1))
        weighted_alpha_loss = alpha_loss * opt.lambda_alpha_loss
        loss += weighted_alpha_loss
        audit_loss_terms["alpha"] = weighted_alpha_loss
    elif not tr.pbr_active:
        simulated_mask = torch.ones_like(render_pkg_re['rend_alpha'][:, None, None])
        alpha_loss = F.binary_cross_entropy(render_pkg_re['rend_alpha'][:, None, None], simulated_mask)
        weighted_alpha_loss = alpha_loss * 0.001
        loss += weighted_alpha_loss
        audit_loss_terms["alpha"] = weighted_alpha_loss

    # 形变正则与二值化分离（仅 SH 阶段 warm-up 之后）。
    if tr.iteration > opt.warm_up and not tr.pbr_active:
        # 标量 warm-up / static 形变时 d_xyz 损失不存在。
        if torch.is_tensor(d_xyz) and opt.d_xyz_loss_weight > 0:
            weighted_d_xyz_loss = (
                (d_xyz**2).mean() * opt.d_xyz_loss_weight
            )
            loss += weighted_d_xyz_loss
            audit_loss_terms["deformation_xyz"] = weighted_d_xyz_loss

        # d_color 正则。
        d_color_reg_loss_weight = opt.d_color_reg_loss_weight
        if not tr.photometric_active and (d_color is not None and torch.is_tensor(d_color)):
            shadow_modulation = d_color[:, :3]
            d_color_reg_loss = (
                shadow_modulation.pow(2.0).mean() * d_color_reg_loss_weight
            )
            loss += d_color_reg_loss

        if tr.iteration > opt.binarization_warm_up and not tr.dataset.no_binary_separation:
            # 论文中无监督二值化的 L1。
            loss += (tr.gaussians.get_binary_feature(eval=False, T=tr.T_current)**1).mean() * opt.lambda_separation

    return Stage1LossResult(
        loss=loss,
        audit_terms=audit_loss_terms,
        pbr_terms=pbr_loss_terms,
        l1=Ll1,
        light_smooth=light_smooth_loss,
        gt_normal_oracle=gt_normal_oracle,
    )


# ---------------------------------------------------------------------------
# 损失组合预设（--loss_preset）。
#
# 预设为“命名组合 + 默认权重”的批量设定：仅覆盖用户未在命令行显式给出
# 的参数；显式 CLI 参数永远优先。auto（默认）完全不改变既有行为。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossPreset:
    """一个命名的 Stage1 损失组合。"""

    description: str
    render_modes: frozenset
    overrides: dict
    terms: tuple


LOSS_PRESETS: dict[str, LossPreset] = {
    "sh_baseline": LossPreset(
        description=(
            "original_sh 基线组合：L1+DSSIM RGB、几何法线/畸变正则"
            "（start_normal_reg 门控）、alpha、d_xyz/d_color、二值化分离。"
        ),
        render_modes=frozenset({"original_sh"}),
        overrides={},
        terms=(
            "rgb_l1",
            "rgb_dssim",
            "normal",
            "distortion",
            "alpha",
            "deformation_xyz",
            "d_color_reg",
            "binary_separation",
        ),
    ),
    "lambertian_default": LossPreset(
        description=(
            "photometric_lambertian 默认组合：photometric 加权 L1+DSSIM RGB、"
            "光源平滑、alpha、d_xyz、二值化分离；法线一致性默认关闭。"
        ),
        render_modes=frozenset({"photometric_lambertian"}),
        overrides={},
        terms=(
            "rgb_l1",
            "rgb_dssim",
            "light_smooth",
            "alpha",
            "deformation_xyz",
            "binary_separation",
        ),
    ),
    "lambertian_normal2": LossPreset(
        description=(
            "0824-01 二法线一致性消融组合：lambertian_default + 独立法线 live/"
            "多视角一致性，去除 GS 几何自一致法线（lambda_gs_normal=0）；"
            "distortion 保持 start_normal_reg 门控。"
        ),
        render_modes=frozenset({"photometric_lambertian"}),
        overrides={
            "lambda_photometric_normal_live": 0.01,
            "lambda_photometric_normal_mv": 0.02,
            "lambda_gs_normal": 0.0,
        },
        terms=(
            "rgb_l1",
            "rgb_dssim",
            "light_smooth",
            "photometric_normal_live",
            "photometric_normal_mv",
            "alpha",
            "deformation_xyz",
            "binary_separation",
        ),
    ),
    "lambertian_normal3": LossPreset(
        description=(
            "0823-01 三法线一致性组合：lambertian_default + 独立法线 live/"
            "多视角一致性（normal_init 保持 0，即不使用信任域先验）。"
        ),
        render_modes=frozenset({"photometric_lambertian"}),
        overrides={
            "lambda_photometric_normal_live": 0.01,
            "lambda_photometric_normal_mv": 0.02,
        },
        terms=(
            "rgb_l1",
            "rgb_dssim",
            "light_smooth",
            "photometric_normal_live",
            "photometric_normal_mv",
            "alpha",
            "deformation_xyz",
            "binary_separation",
        ),
    ),
    "gs_wrapping": LossPreset(
        description=(
            "Gaussian Wrapping（From Blobs to Spokes，arXiv:2604.07337，2026）"
            "论文 §4.1 损失权重方案映射：lambertian_default + "
            "L_DN 深度-法线一致性（lambda_gs_normal=0.05）+ "
            "L_N 独立法线/深度一致性（live=0.05）+ "
            "L_gc 多视角重投影一致性（mv=0.02）；论文 L_pc 多视角光度"
            "一致性本仓库无对应项。调度沿用本仓库：live@500、mv@1000+"
            "ramp、gs_normal 受 start_normal_reg 门控。"
        ),
        render_modes=frozenset({"photometric_lambertian"}),
        overrides={
            "lambda_gs_normal": 0.05,
            "lambda_photometric_normal_live": 0.05,
            "lambda_photometric_normal_mv": 0.02,
        },
        terms=(
            "rgb_l1",
            "rgb_dssim",
            "light_smooth",
            "normal",
            "photometric_normal_live",
            "photometric_normal_mv",
            "alpha",
            "deformation_xyz",
            "binary_separation",
        ),
    ),
    "pbr_default": LossPreset(
        description=(
            "photometric_perlight_pbr 默认组合：MSE+前景L1+前景DSSIM+"
            "log-linear RGB、光源平滑 1/2、曝光/法线残差/粗糙率/环境能量/"
            "残差光源正则。"
        ),
        render_modes=frozenset({"photometric_perlight_pbr"}),
        overrides={},
        terms=(
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
        ),
    ),
    "n0_gt_normal_oracle": LossPreset(
        description=(
            "N0 oracle 消融：RGB 权重置 0，仅用 GT-normal 余弦监督"
            "（需同时提供 --photometric_gt_normal_dir）。"
        ),
        render_modes=frozenset({"photometric_lambertian"}),
        overrides={
            "photometric_rgb_loss_weight": 0.0,
            "lambda_photometric_gt_normal": 1.0,
        },
        terms=("photometric_gt_normal",),
    ),
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


def apply_loss_preset(args, argv=None) -> list[str]:
    """按 --loss_preset 将预设默认值写入 args，返回被覆盖的参数名。

    - auto（默认）：不做任何修改，保持既有行为；
    - 预设仅覆盖未在命令行显式给出的参数，显式 CLI 参数永远优先；
    - 预设与 --render_mode 不兼容时抛出 ValueError。
    """
    name = str(getattr(args, "loss_preset", "auto") or "auto").strip().lower()
    if name == "auto":
        return []
    preset = LOSS_PRESETS.get(name)
    if preset is None:
        raise ValueError(
            f"Unknown loss_preset {name!r}; available presets: "
            f"{sorted(LOSS_PRESETS)} (or 'auto')."
        )
    render_mode = str(getattr(args, "render_mode", "original_sh"))
    if render_mode not in preset.render_modes:
        raise ValueError(
            f"loss_preset {name!r} requires render_mode in "
            f"{sorted(preset.render_modes)}, got {render_mode!r}."
        )
    explicit = explicit_cli_keys(argv)
    applied = []
    for key, value in preset.overrides.items():
        if key in explicit:
            continue
        setattr(args, key, value)
        applied.append(key)
    if applied:
        print(
            f"[loss_preset={name}] applied defaults: "
            + ", ".join(f"{k}={preset.overrides[k]}" for k in applied)
        )
    return applied
