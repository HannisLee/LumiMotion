"""Stage2 原子损失与模式对应的 preset；显式 CLI 权重优先。"""
from __future__ import annotations
from dataclasses import dataclass
import sys
import torch
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim, first_order_edge_aware_loss, tv_loss
from arguments.stage2 import normalize_render_mode


@dataclass
class Stage2LossResult:
    total: torch.Tensor
    terms: dict
    l1: torch.Tensor


class LambertianLossPreset:
    overrides = {"lambda_dssim": 0.2, "lambda_photometric_albedo_prior": 0.001,
                 "lambda_photometric_light_smooth": 0.001}

    def __init__(self, opt):
        self.opt = opt

    def compute(self, package, target, mask, material, light):
        mask = mask.detach().float()
        count = mask.sum()
        if count <= 0:
            raise ValueError("前景监督掩码为空。")
        image = package["render"]
        rgb_l1 = ((image - target).abs() * mask).sum() / (count * 3)
        dssim = 1 - ssim(image * mask, target * mask)
        prior = material.prior()
        smooth = light.smoothness()
        total = (1-self.opt.lambda_dssim) * rgb_l1 + self.opt.lambda_dssim * dssim
        total = total + self.opt.lambda_photometric_albedo_prior * prior
        total = total + self.opt.lambda_photometric_light_smooth * smooth
        return Stage2LossResult(total, {"rgb_l1": rgb_l1, "dssim": dssim,
                                      "albedo_prior": prior, "light_smooth": smooth}, rgb_l1)


class LambertianGTLossPreset(LambertianLossPreset):
    overrides = dict(LambertianLossPreset.overrides, lambda_photometric_light_smooth=0.0)


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
                "lambertian_gt": LambertianGTLossPreset}


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
        if (name == "lambertian_gt") != (opt.photometric_light_mode == "gt_directional"):
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
