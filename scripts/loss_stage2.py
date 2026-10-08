"""Stage2 原子损失与模式对应的 preset；显式 CLI 权重优先。"""
from __future__ import annotations
from dataclasses import dataclass
import sys
import math
import torch
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim, first_order_edge_aware_loss, tv_loss
from arguments.stage2 import normalize_render_mode


@dataclass
class Stage2LossResult:
    total: torch.Tensor
    terms: dict
    l1: torch.Tensor


def photometric_rgb_l1(image, target, mask, count, positive_weight=1., threshold=.03):
    """观测明显亮于预测的像素仅减弱L1；保持原有效像素分母及默认浮点顺序。"""
    if positive_weight == 1.:
        return ((image-target).abs()*mask).sum()/(count*3)
    positive = ((target-image).mean(0, keepdim=True).detach() > threshold)
    confidence = torch.where(positive, positive_weight, 1.)
    return ((image-target).abs()*mask*confidence).sum()/(count*3)


class LambertianLossPreset:
    overrides = {"lambda_dssim": 0.2, "lambda_photometric_albedo_prior": 0.001,
                 "lambda_photometric_light_smooth": 0.001}

    def __init__(self, opt):
        self.opt = opt
        weight = getattr(opt, 'photometric_positive_residual_weight', 1.)
        threshold = getattr(opt, 'photometric_positive_residual_threshold', .03)
        if not math.isfinite(weight) or not 0 <= weight <= 1 or not math.isfinite(threshold) or threshold < 0:
            raise ValueError('正光度残差权重须在[0,1]，阈值须有限且非负。')

    def compute(self, package, target, mask, material, light):
        mask = mask.detach().float()
        count = mask.sum()
        if count <= 0:
            raise ValueError("前景监督掩码为空。")
        image = package["render"]
        weight = getattr(self.opt, 'photometric_positive_residual_weight', 1.)
        threshold = getattr(self.opt, 'photometric_positive_residual_threshold', .03)
        rgb_l1 = photometric_rgb_l1(image, target, mask, count, weight, threshold)
        dssim = 1 - ssim(image * mask, target * mask)
        prior = material.prior()
        smooth = light.smoothness()
        total = (1-self.opt.lambda_dssim) * rgb_l1 + self.opt.lambda_dssim * dssim
        total = total + self.opt.lambda_photometric_albedo_prior * prior
        total = total + self.opt.lambda_photometric_light_smooth * smooth
        terms = {"rgb_l1": rgb_l1, "dssim": dssim, "albedo_prior": prior, "light_smooth": smooth}
        if weight != 1.:
            terms['photo_positive_residual_fraction'] = (((target-image).mean(0, keepdim=True).detach()>threshold)*mask).sum()/count
        return Stage2LossResult(total, terms, rgb_l1)


class LambertianGTLossPreset(LambertianLossPreset):
    overrides = dict(LambertianLossPreset.overrides, lambda_photometric_light_smooth=0.0)


class LambertianGeometryLossPreset(LambertianGTLossPreset):
    """几何附加项由 runtime 使用当前相机和只读参考深度计算。"""
    overrides = dict(LambertianGTLossPreset.overrides)


class LambertianPointGeometryLossPreset(LambertianGeometryLossPreset):
    overrides = dict(LambertianGeometryLossPreset.overrides, geometry_point_light=True)


class LambertianFilmicGeometryLossPreset(LambertianPointGeometryLossPreset):
    overrides = dict(LambertianPointGeometryLossPreset.overrides, geometry_tonemap="filmic")


class LambertianShadowGeometryLossPreset(LambertianFilmicGeometryLossPreset):
    overrides = dict(LambertianFilmicGeometryLossPreset.overrides, geometry_shadows=True)


class LambertianPaletteGeometryLossPreset(LambertianShadowGeometryLossPreset):
    overrides = dict(LambertianShadowGeometryLossPreset.overrides, photometric_material_mode="palette")


class LambertianSurfelGeometryLossPreset(LambertianFilmicGeometryLossPreset):
    overrides = dict(LambertianFilmicGeometryLossPreset.overrides, geometry_shading="surfel")


class LambertianLitGeometryLossPreset(LambertianSurfelGeometryLossPreset):
    overrides = dict(LambertianSurfelGeometryLossPreset.overrides, geometry_shadows=True,
                     photometric_material_mode="palette", geometry_photo_visibility_threshold=.98,
                     geometry_photo_min_shading=.2)


class LambertianAreaGeometryLossPreset(LambertianLitGeometryLossPreset):
    overrides = dict(LambertianLitGeometryLossPreset.overrides, geometry_area_emitter=True,
                     photometric_light_intensity=-1.0)


class LambertianAreaScaleGeometryLossPreset(LambertianAreaGeometryLossPreset):
    overrides = dict(LambertianAreaGeometryLossPreset.overrides, geometry_optimize_scale=True)


class LambertianAreaMarginGeometryLossPreset(LambertianAreaGeometryLossPreset):
    overrides = dict(LambertianAreaGeometryLossPreset.overrides, geometry_normal_margin=3.)


class LambertianAreaRefineGeometryLossPreset(LambertianAreaMarginGeometryLossPreset):
    overrides = dict(LambertianAreaMarginGeometryLossPreset.overrides,
                     geometry_subdivide=True, geometry_optimize_scale=True)


class LambertianAreaCoherentGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     lambda_geometry_coherence=0.05)


class LambertianAreaDenseGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     geometry_subdivision_side=4)


class LambertianAreaOpacityGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    """小学习率、有界衰减来源alpha；默认仍冻结opacity。"""
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     geometry_optimize_opacity=True)


class LambertianAreaRGBPriorGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     lambda_geometry_rgb_normal_prior=.2)


class LambertianAreaDistortionGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    """独立稳定raster核的预测深度畸变，未指定时所有旧分支仍关闭。"""
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     lambda_geometry_distortion=1e6)


class LambertianAreaDepthFollowGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    """仅一致性项固定预测GS方向，预测深度几何跟随；光度优化保持。"""
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     geometry_normal_gradient_mode='depth')


class LambertianAreaStaticCoherentGeometryLossPreset(LambertianAreaCoherentGeometryLossPreset):
    """新建图必须提供训练来源的时间稳定掩码；完整检查点恢复无需外部掩码。"""
    overrides = dict(LambertianAreaCoherentGeometryLossPreset.overrides,
                     geometry_checkpoint_source_cache=True)


class LambertianAreaRefinedCoherentGeometryLossPreset(LambertianAreaCoherentGeometryLossPreset):
    """从真实微调表面重新建立模型预测的冻结静态邻域。"""
    overrides = dict(LambertianAreaCoherentGeometryLossPreset.overrides,
                     geometry_refined_coherence=True, lambda_geometry_coherence=.01)


class LambertianAreaBalancedCoherentGeometryLossPreset(LambertianAreaRefinedCoherentGeometryLossPreset):
    """单独平衡邻域位置/法线的反向量级；原前向几何关系不改。"""
    overrides = dict(LambertianAreaRefinedCoherentGeometryLossPreset.overrides,
                     lambda_geometry_coherence=1., geometry_coherence_position_gradient_scale=.01)


class LambertianAreaPhotoSupportedGeometryLossPreset(LambertianAreaRefineGeometryLossPreset):
    """有效光度域外保留较弱预测一致性；默认旧分支仍为全权重。"""
    overrides = dict(LambertianAreaRefineGeometryLossPreset.overrides,
                     geometry_normal_unsupported_weight=.1)


class LambertianAreaRGBWarmupGeometryLossPreset(LambertianAreaPhotoSupportedGeometryLossPreset):
    """仅训练RGB先验旋转预热，随后回到同一Lambertian联合优化。"""
    overrides = dict(LambertianAreaPhotoSupportedGeometryLossPreset.overrides,
                     lambda_geometry_rgb_normal_prior=2., geometry_rgb_prior_warmup=2000)


class LambertianAreaRobustPhotoGeometryLossPreset(LambertianAreaPhotoSupportedGeometryLossPreset):
    """训练RGBA亮部异常仅减弱L1；DSSIM、原二值支持域和几何项不变。"""
    overrides = dict(LambertianAreaPhotoSupportedGeometryLossPreset.overrides,
                     photometric_positive_residual_weight=.25, photometric_positive_residual_threshold=.03)


class LambertianAreaPositionConsistencyGeometryLossPreset(LambertianAreaPhotoSupportedGeometryLossPreset):
    """预测深度一致性只更新XYZ；RGB及其他项仍联合优化表面方向。"""
    overrides = dict(LambertianAreaPhotoSupportedGeometryLossPreset.overrides,
                     geometry_normal_gradient_mode='position')


class LambertianAreaStaticAnchorGeometryLossPreset(LambertianAreaPhotoSupportedGeometryLossPreset):
    """固定训练来源相机的Stage1形变；Stage2几何仍联合优化。"""
    overrides = dict(LambertianAreaPhotoSupportedGeometryLossPreset.overrides,
                     geometry_source_deformation_mode='anchor_camera')


class IRGSBaselineLossPreset:
    overrides = {}

    def __init__(self, opt):
        self.opt = opt

    def compute(self, tr, pkg, gt, image):
        # 保持原 Stage2 的公式、门控和加法顺序，包括固定 SH DSSIM 权重。
        mask = pkg["mask"]
        rgb_l1 = F.l1_loss(image.permute(1,2,0)[mask], gt.permute(1,2,0)[mask])
        total = rgb_l1
        render_sh = pkg["render_sh"]
        mask2 = (pkg["rend_alpha"] > 0.9).float()
        if render_sh.shape[1] > 1:
            mask2 = mask2.expand_as(render_sh)
        masked_render, masked_gt = render_sh * mask2, gt * mask2
        loss_sh = 0.8 * l1_loss(masked_render, masked_gt) + 0.2 * (1-ssim(masked_render, masked_gt))
        total = total + loss_sh
        terms = {"rgb_l1": rgb_l1, "sh_rgb": loss_sh}
        if self.opt.d_lower_hemisphere_weight > 0:
            env = tr.env_light.render_env_map(H=64)["env2"].permute(2,0,1)
            penalty = (env[:,round(env.shape[1]*0.66):,:] ** 2).mean()
            total = total + self.opt.d_lower_hemisphere_weight * penalty
            terms["env_lower_hemisphere"] = penalty
        if self.opt.lambda_roughness_smooth > 0:
            value = first_order_edge_aware_loss(pkg["roughness"] * mask2, masked_gt)
            total = total + self.opt.lambda_roughness_smooth * value
            terms["roughness_smooth"] = value
        if self.opt.lambda_light > 0:
            direct = pkg["ray_light_direct"]
            value = F.l1_loss(direct, direct.mean(-1,keepdim=True).expand_as(direct))
            total = total + self.opt.lambda_light * value
            terms["light_white"] = value
        if self.opt.lambda_light_smooth > 0:
            value = tv_loss(pkg["env_only"])
            total = total + self.opt.lambda_light_smooth * value
            terms["light_smooth"] = value
        if self.opt.lambda_base_color_smooth > 0:
            value = first_order_edge_aware_loss(pkg["base_color_linear"] * mask2, gt * mask2)
            total = total + self.opt.lambda_base_color_smooth * value
            terms["base_color_smooth"] = value
        return Stage2LossResult(total, terms, rgb_l1)


LOSS_PRESETS = {"irgs_baseline": IRGSBaselineLossPreset,
                "lambertian_default": LambertianLossPreset,
                "lambertian_gt": LambertianGTLossPreset,
                "lambertian_geometry": LambertianGeometryLossPreset,
                "lambertian_geometry_point": LambertianPointGeometryLossPreset,
                "lambertian_geometry_filmic": LambertianFilmicGeometryLossPreset,
                "lambertian_geometry_shadow": LambertianShadowGeometryLossPreset,
                "lambertian_geometry_palette": LambertianPaletteGeometryLossPreset,
                "lambertian_geometry_surfel": LambertianSurfelGeometryLossPreset,
                "lambertian_geometry_lit": LambertianLitGeometryLossPreset,
                "lambertian_geometry_area": LambertianAreaGeometryLossPreset,
                "lambertian_geometry_area_scale": LambertianAreaScaleGeometryLossPreset,
                "lambertian_geometry_area_margin": LambertianAreaMarginGeometryLossPreset,
                "lambertian_geometry_area_photo_supported": LambertianAreaPhotoSupportedGeometryLossPreset,
                "lambertian_geometry_area_rgb_warmup": LambertianAreaRGBWarmupGeometryLossPreset,
                "lambertian_geometry_area_robust_photo": LambertianAreaRobustPhotoGeometryLossPreset,
                "lambertian_geometry_area_position_consistency": LambertianAreaPositionConsistencyGeometryLossPreset,
                "lambertian_geometry_area_static_anchor": LambertianAreaStaticAnchorGeometryLossPreset,
                "lambertian_geometry_area_refine": LambertianAreaRefineGeometryLossPreset,
                "lambertian_geometry_area_coherent": LambertianAreaCoherentGeometryLossPreset,
                "lambertian_geometry_area_static_coherent": LambertianAreaStaticCoherentGeometryLossPreset,
                "lambertian_geometry_area_refined_coherent": LambertianAreaRefinedCoherentGeometryLossPreset,
                "lambertian_geometry_area_balanced_coherent": LambertianAreaBalancedCoherentGeometryLossPreset,
                "lambertian_geometry_area_opacity": LambertianAreaOpacityGeometryLossPreset,
                "lambertian_geometry_area_rgb_prior": LambertianAreaRGBPriorGeometryLossPreset,
                "lambertian_geometry_area_distortion": LambertianAreaDistortionGeometryLossPreset,
                "lambertian_geometry_area_depth_follow": LambertianAreaDepthFollowGeometryLossPreset,
                "lambertian_geometry_area_dense": LambertianAreaDenseGeometryLossPreset}


def is_geometry_preset(name):
    return name in LOSS_PRESETS and issubclass(LOSS_PRESETS[name],LambertianGeometryLossPreset)


def resolve_preset(opt, mode):
    mode = normalize_render_mode(mode)
    name = opt.loss_preset
    if name == "auto":
        name = "irgs_baseline" if mode == "original_ir" else (
            "lambertian_gt" if opt.photometric_light_mode == "gt_directional" else "lambertian_default")
    if name not in LOSS_PRESETS:
        raise ValueError("未知 Stage2 loss preset：" + name)
    if mode not in {"original_ir", "photometric_lambertian"}:
        raise ValueError("未知 Stage2 render mode：" + mode)
    if (name == "irgs_baseline") != (mode == "original_ir"):
        raise ValueError("loss preset 与 render mode 不匹配。")
    if mode == "photometric_lambertian":
        if opt.photometric_light_mode not in {"learned_directional", "gt_directional"}:
            raise ValueError("未知光照模式。")
        if issubclass(LOSS_PRESETS[name],LambertianGTLossPreset) != (opt.photometric_light_mode == "gt_directional"):
            raise ValueError("loss preset 与 light mode 不匹配。")
    return name


def apply_loss_preset(args, argv=None):
    name = resolve_preset(args, args.render_mode)
    if args.loss_preset == "auto":
        return
    explicit = {t[2:].split('=')[0].replace('-','_') for t in (sys.argv[1:] if argv is None else argv) if t.startswith('--')}
    for key, value in LOSS_PRESETS[name].overrides.items():
        if key not in explicit:
            setattr(args, key, value)


def build_loss_preset(opt, mode):
    return LOSS_PRESETS[resolve_preset(opt, mode)](opt)
