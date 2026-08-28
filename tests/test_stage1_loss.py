"""scripts/loss.py 的单元测试：损失组装数值等价与 --loss_preset 预设。"""

import argparse
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scripts.loss import (
    AlphaLoss,
    BinarySeparationLoss,
    DeformationColorLoss,
    DeformationXYZLoss,
    DistortionLoss,
    GSNormalLoss,
    GTNormalOracleLoss,
    LightSmoothnessLoss,
    LOSS_PRESETS,
    PhotometricNormalInitLoss,
    PhotometricNormalLiveLoss,
    PhotometricNormalMVLoss,
    RGBDSSIMLoss,
    RGBL1Loss,
    Stage1LossContext,
    apply_loss_preset,
    build_loss_preset,
    compute_stage1_loss,
)
from utils.loss_utils import l1_loss, ssim


def _make_opt(**overrides):
    opt = dict(
        loss_preset="auto",
        start_normal_reg=0,
        lambda_dist=1000.0,
        lambda_gs_normal=0.02,
        gt_alpha_mask_as_scene_mask=False,
        lambda_dssim=0.0,
        lambda_alpha_loss=0.1,
        warm_up=0,
        d_xyz_loss_weight=0.001,
        d_color_reg_loss_weight=0.01,
        binarization_warm_up=0,
        lambda_separation=0.005,
        lambda_photometric_normal_init=0.0,
        lambda_photometric_normal_live=0.0,
        lambda_photometric_normal_mv=0.0,
        photometric_normal_live_start_iter=500,
        photometric_normal_live_alpha_threshold=0.5,
        photometric_normal_mv_start_iter=1000,
        photometric_normal_mv_interval=1,
        photometric_normal_mv_alpha_threshold=0.5,
        photometric_normal_mv_depth_tol=0.1,
        photometric_normal_mv_ramp_iters=2000,
        photometric_rgb_loss_weight=1.0,
        lambda_photometric_gt_normal=0.0,
        photometric_gt_normal_alpha_threshold=0.5,
        lambda_photometric_light_smooth1=0.001,
        lambda_photometric_light_smooth2=0.01,
        lambda_photometric_pbr_exposure=0.001,
        lambda_photometric_pbr_normal=0.01,
        lambda_photometric_pbr_roughness=0.001,
        lambda_photometric_pbr_environment=0.0001,
        lambda_photometric_pbr_residual=0.01,
        photometric_pbr_loss_mse=1.0,
        photometric_pbr_loss_l1_fg=0.2,
        photometric_pbr_loss_dssim_fg=0.1,
        photometric_pbr_loss_log_linear=0.05,
    )
    opt.update(overrides)
    return SimpleNamespace(**opt)


class _FakeGaussians:
    use_photometric_normal = False

    def __init__(self):
        self._binary = torch.full((4, 2), 0.25)
        self._rough = torch.full((4, 1), 0.3)

    def get_binary_feature(self, eval=False, T=0.5):
        return self._binary

    @property
    def get_rough(self):
        return self._rough


class _FakeRenderer:
    def light_smoothness_loss(self):
        return torch.tensor(0.5)

    def regularization_losses(self, roughness):
        return {
            "light_smooth2": torch.tensor(0.1),
            "exposure": torch.tensor(0.2),
            "normal_residual": torch.tensor(0.3),
            "roughness_prior": torch.tensor(0.4),
            "environment_energy": torch.tensor(0.5),
            "residual": torch.tensor(0.6),
        }


def _make_trainer(**overrides):
    tr = SimpleNamespace(
        opt=_make_opt(),
        dataset=SimpleNamespace(white_background=False, no_binary_separation=False),
        gaussians=_FakeGaussians(),
        iteration=5,
        pbr_active=False,
        photometric_active=False,
        photometric_gt_normal_enabled=False,
        photometric_normal_live_enabled=False,
        photometric_normal_mv_enabled=False,
        photometric_rgb_loss_weight=1.0,
        lambda_photometric_gt_normal=0.0,
        photometric_gt_normal_alpha_threshold=0.5,
        photometric_normal_live_start_iter=500,
        photometric_normal_live_alpha_threshold=0.5,
        photometric_normal_mv_start_iter=1000,
        photometric_normal_mv_interval=1,
        photometric_normal_mv_alpha_threshold=0.5,
        photometric_normal_mv_depth_tol=0.1,
        photometric_normal_mv_ramp_iters=2000,
        lambda_photometric_normal_live=0.0,
        lambda_photometric_normal_mv=0.0,
        T_current=0.5,
        tb_writer=None,
        background=torch.zeros(3),
        photometric_renderer=None,
        scene=None,
        deform=None,
        pipe=None,
    )
    for key, value in overrides.items():
        setattr(tr, key, value)
    return tr


def _render_pkg(height=4, width=4, pbr=False):
    torch.manual_seed(0)
    pkg = {
        "rend_dist": torch.rand(1, height, width),
        "rend_normal": F.normalize(torch.randn(3, height, width), dim=0),
        "surf_normal": F.normalize(torch.randn(3, height, width), dim=0),
        "rend_alpha": torch.full((1, height, width), 0.6),
    }
    if pbr:
        pkg["render_linear"] = torch.rand(3, height, width)
    return pkg


class ComputeStage1LossShPathTest(unittest.TestCase):
    """SH 基线路径：与手写参考公式逐项对比。"""

    def test_sh_path_total_matches_reference(self):
        torch.manual_seed(1)
        height = width = 4
        tr = _make_trainer()
        render_pkg = _render_pkg(height, width)
        image = torch.rand(3, height, width)
        gt_image = torch.rand(3, height, width)
        d_xyz = torch.randn(10, 3)
        d_color = torch.randn(10, 3)
        viewpoint_cam = SimpleNamespace(gt_alpha_mask=None)

        result = compute_stage1_loss(
            tr, viewpoint_cam, render_pkg, image, gt_image,
            d_xyz, None, None, None, d_color,
        )

        l1 = l1_loss(image, gt_image)
        dssim_term = 0.0  # lambda_dssim = 0
        normal_error = (1 - (render_pkg["rend_normal"] * render_pkg["surf_normal"]).sum(dim=0))[None]
        normal_loss = 0.02 * normal_error.mean()
        dist_loss = 1000.0 * render_pkg["rend_dist"].mean()
        alpha_loss = F.binary_cross_entropy(
            render_pkg["rend_alpha"][:, None, None],
            torch.ones_like(render_pkg["rend_alpha"][:, None, None]),
        ) * 0.001
        d_xyz_loss = (d_xyz ** 2).mean() * 0.001
        d_color_loss = d_color[:, :3].pow(2.0).mean() * 0.01
        separation_loss = tr.gaussians.get_binary_feature().mean() * 0.005
        expected = (
            l1 + dssim_term + normal_loss + dist_loss
            + alpha_loss + d_xyz_loss + d_color_loss + separation_loss
        )
        self.assertTrue(torch.allclose(result.loss, expected, atol=0.0, rtol=0.0))
        self.assertEqual(
            list(result.audit_terms),
            ["rgb_l1", "rgb_dssim", "normal", "distortion", "alpha", "deformation_xyz"],
        )
        self.assertTrue(torch.equal(result.audit_terms["rgb_l1"], l1))
        self.assertEqual(float(result.audit_terms["rgb_dssim"]), 0.0)
        self.assertEqual(result.gt_normal_oracle, None)
        self.assertEqual(float(result.light_smooth), 0.0)

    def test_sh_path_dssim_weighting(self):
        torch.manual_seed(2)
        tr = _make_trainer(opt=_make_opt(lambda_dssim=0.2, start_normal_reg=10**9,
                                         warm_up=10**9))
        render_pkg = _render_pkg()
        image = torch.rand(3, 4, 4)
        gt_image = torch.rand(3, 4, 4)
        result = compute_stage1_loss(
            tr, SimpleNamespace(gt_alpha_mask=None), render_pkg, image, gt_image,
            0.0, None, None, None, None,
        )
        expected = 0.8 * l1_loss(image, gt_image) + 0.2 * (1.0 - ssim(image, gt_image))
        alpha_loss = F.binary_cross_entropy(
            render_pkg["rend_alpha"][:, None, None],
            torch.ones_like(render_pkg["rend_alpha"][:, None, None]),
        ) * 0.001
        expected = expected + alpha_loss
        self.assertTrue(torch.allclose(result.loss, expected))


class ComputeStage1LossLambertianPathTest(unittest.TestCase):
    """Lambertian 路径：RGB 总权重与光照正则由对象内 ratio 控制。"""

    def test_lambertian_total_matches_reference(self):
        torch.manual_seed(4)
        opt = _make_opt(
            loss_preset="lambertian_default",
            lambda_dssim=0.2,
            photometric_rgb_loss_weight=0.5,
            start_normal_reg=10**9,
            warm_up=10**9,
        )
        tr = _make_trainer(
            opt=opt,
            requested_render_mode="photometric_lambertian",
            photometric_active=True,
            photometric_renderer=_FakeRenderer(),
        )
        render_pkg = _render_pkg()
        image = torch.rand(3, 4, 4)
        gt_image = torch.rand(3, 4, 4)
        result = compute_stage1_loss(
            tr, SimpleNamespace(gt_alpha_mask=None), render_pkg, image, gt_image,
            0.0, None, None, None, None,
        )
        expected = 0.5 * (
            0.8 * l1_loss(image, gt_image)
            + 0.2 * (1.0 - ssim(image, gt_image))
        )
        expected = expected + 0.001 * torch.tensor(0.5)
        expected = expected + 0.001 * F.binary_cross_entropy(
            render_pkg["rend_alpha"][:, None, None],
            torch.ones_like(render_pkg["rend_alpha"][:, None, None]),
        )
        self.assertTrue(torch.allclose(result.loss, expected))
        self.assertEqual(
            list(result.audit_terms),
            ["rgb_l1", "rgb_dssim", "normal", "distortion",
             "light_smooth", "alpha"],
        )
        self.assertEqual(float(result.light_smooth), 0.5)


class ComputeStage1LossPbrPathTest(unittest.TestCase):
    """PBR 路径已停用，必须在进入训练前明确失败。"""

    def test_pbr_compute_raises(self):
        tr = _make_trainer(pbr_active=True, photometric_active=True,
                           requested_render_mode="photometric_perlight_pbr",
                           photometric_renderer=_FakeRenderer())
        image = torch.rand(3, 4, 4)
        with self.assertRaisesRegex(AssertionError, "PBR|pbr"):
            compute_stage1_loss(
                tr, SimpleNamespace(gt_alpha_mask=None), _render_pkg(pbr=True),
                image, torch.rand_like(image), None, None, None, None, None,
            )

    def test_pbr_build_raises(self):
        with self.assertRaisesRegex(AssertionError, "PBR|pbr"):
            build_loss_preset(_make_opt(), "photometric_perlight_pbr")


class LossPresetTest(unittest.TestCase):
    def _make_args(self, **overrides):
        args = dict(
            loss_preset="auto",
            render_mode="photometric_lambertian",
            lambda_photometric_normal_live=0.0,
            lambda_photometric_normal_mv=0.0,
            photometric_rgb_loss_weight=1.0,
            lambda_photometric_gt_normal=0.0,
            lambda_gs_normal=0.02,
        )
        args.update(overrides)
        return argparse.Namespace(**args)

    def test_registry_complete(self):
        self.assertEqual(
            sorted(LOSS_PRESETS),
            ["gs_wrapping", "lambertian_default", "lambertian_normal2",
             "lambertian_normal3", "n0_gt_normal_oracle", "pbr_default",
             "sh_baseline"],
        )
        for name, preset in LOSS_PRESETS.items():
            self.assertIsInstance(preset, type, name)
            self.assertTrue(preset.render_modes, name)
            self.assertTrue(preset.description, name)
            self.assertTrue(preset.terms, name)

    def test_auto_is_noop(self):
        args = self._make_args()
        self.assertEqual(apply_loss_preset(args, argv=[]), [])
        self.assertEqual(args.lambda_photometric_normal_live, 0.0)
        self.assertEqual(args.lambda_photometric_normal_mv, 0.0)

    def test_unknown_preset_raises(self):
        args = self._make_args(loss_preset="does_not_exist")
        with self.assertRaises(ValueError) as ctx:
            apply_loss_preset(args, argv=[])
        self.assertIn("does_not_exist", str(ctx.exception))

    def test_render_mode_validation(self):
        args = self._make_args(loss_preset="lambertian_normal3",
                               render_mode="photometric_perlight_pbr")
        with self.assertRaises(AssertionError):
            apply_loss_preset(args, argv=[])
        args = self._make_args(loss_preset="pbr_default",
                               render_mode="photometric_lambertian")
        with self.assertRaises(ValueError):
            apply_loss_preset(args, argv=[])

    def test_lambertian_normal3_applies_weights(self):
        args = self._make_args(loss_preset="lambertian_normal3")
        applied = apply_loss_preset(args, argv=["--loss_preset", "lambertian_normal3"])
        self.assertEqual(applied, ["lambda_photometric_normal_live",
                                   "lambda_photometric_normal_mv"])
        self.assertEqual(args.lambda_photometric_normal_live, 0.01)
        self.assertEqual(args.lambda_photometric_normal_mv, 0.02)

    def test_lambertian_normal2_applies_weights(self):
        args = self._make_args(loss_preset="lambertian_normal2")
        applied = apply_loss_preset(args, argv=["--loss_preset", "lambertian_normal2"])
        self.assertEqual(applied, ["lambda_photometric_normal_live",
                                   "lambda_photometric_normal_mv",
                                   "lambda_gs_normal"])
        self.assertEqual(args.lambda_photometric_normal_live, 0.01)
        self.assertEqual(args.lambda_photometric_normal_mv, 0.02)
        self.assertEqual(args.lambda_gs_normal, 0.0)

    def test_lambertian_normal2_requires_lambertian_mode(self):
        args = self._make_args(loss_preset="lambertian_normal2",
                               render_mode="original_sh")
        with self.assertRaises(ValueError):
            apply_loss_preset(args, argv=[])

    def test_gs_wrapping_applies_weights(self):
        args = self._make_args(loss_preset="gs_wrapping")
        applied = apply_loss_preset(args, argv=["--loss_preset", "gs_wrapping"])
        self.assertEqual(applied, ["lambda_gs_normal",
                                   "lambda_photometric_normal_live",
                                   "lambda_photometric_normal_mv"])
        self.assertEqual(args.lambda_gs_normal, 0.05)
        self.assertEqual(args.lambda_photometric_normal_live, 0.05)
        self.assertEqual(args.lambda_photometric_normal_mv, 0.02)

    def test_gs_wrapping_requires_lambertian_mode(self):
        args = self._make_args(loss_preset="gs_wrapping",
                               render_mode="original_sh")
        with self.assertRaises(ValueError):
            apply_loss_preset(args, argv=[])

    def test_explicit_cli_wins_over_preset(self):
        args = self._make_args(
            loss_preset="lambertian_normal3",
            lambda_photometric_normal_mv=0.05,
        )
        applied = apply_loss_preset(
            args,
            argv=["--loss_preset", "lambertian_normal3",
                  "--lambda_photometric_normal_mv", "0.05"],
        )
        self.assertEqual(applied, ["lambda_photometric_normal_live"])
        self.assertEqual(args.lambda_photometric_normal_mv, 0.05)
        self.assertEqual(args.lambda_photometric_normal_live, 0.01)

    def test_n0_oracle_preset(self):
        args = self._make_args(loss_preset="n0_gt_normal_oracle")
        applied = apply_loss_preset(args, argv=["--loss_preset", "n0_gt_normal_oracle"])
        self.assertEqual(set(applied),
                         {"photometric_rgb_loss_weight", "lambda_photometric_gt_normal"})
        self.assertEqual(args.photometric_rgb_loss_weight, 0.0)
        self.assertEqual(args.lambda_photometric_gt_normal, 1.0)


class PresetCompositionTest(unittest.TestCase):
    """Preset 是独立大 loss 类，__init__ 明确保存原子 loss 顺序。"""

    def test_preset_classes_do_not_inherit_from_each_other(self):
        preset_types = list(LOSS_PRESETS.values())
        self.assertEqual(len(preset_types), len(set(preset_types)))
        for preset_type in preset_types:
            self.assertEqual(preset_type.__bases__, (object,), preset_type.__name__)

    def test_sh_composition_order(self):
        preset = build_loss_preset(
            _make_opt(loss_preset="sh_baseline"), "original_sh"
        )
        self.assertEqual(
            [type(loss) for loss in preset.losses],
            [RGBL1Loss, RGBDSSIMLoss, GSNormalLoss, DistortionLoss,
             AlphaLoss, DeformationXYZLoss, DeformationColorLoss,
             BinarySeparationLoss],
        )

    def test_lambertian_composition_order(self):
        preset = build_loss_preset(
            _make_opt(loss_preset="lambertian_normal3"),
            "photometric_lambertian",
        )
        self.assertEqual(
            [type(loss) for loss in preset.losses],
            [RGBL1Loss, RGBDSSIMLoss, GSNormalLoss, DistortionLoss,
             GTNormalOracleLoss, PhotometricNormalInitLoss,
             PhotometricNormalLiveLoss, PhotometricNormalMVLoss,
             LightSmoothnessLoss, AlphaLoss, DeformationXYZLoss,
             DeformationColorLoss, BinarySeparationLoss],
        )

    def test_loss_ratio_is_fixed_after_construction(self):
        opt = _make_opt(loss_preset="sh_baseline", lambda_dssim=0.2)
        preset = build_loss_preset(opt, "original_sh")
        opt.lambda_dssim = 0.8
        self.assertEqual(preset.losses[0].ratio, 0.8)
        self.assertEqual(preset.losses[1].ratio, 0.2)

    def test_atomic_loss_default_ratio_is_one(self):
        self.assertEqual(RGBL1Loss().ratio, 1.0)
        self.assertEqual(GSNormalLoss().ratio, 1.0)

    def test_world_normal_render_is_cached_per_context(self):
        normal = torch.ones(3, 2, 2)
        alpha = torch.ones(1, 2, 2)
        context = Stage1LossContext(
            tr=SimpleNamespace(),
            viewpoint_cam=SimpleNamespace(),
            render_pkg={},
            image=torch.zeros(3, 2, 2),
            gt_image=torch.zeros(3, 2, 2),
            d_xyz=None,
            d_rotation=None,
            d_scaling=None,
            d_opacity=None,
            d_color=None,
        )
        with patch(
            "scripts.loss.render_world_normal_map",
            return_value=(normal, alpha),
        ) as mocked_render:
            first = context.rendered_world_normal()
            second = context.rendered_world_normal()
            self.assertIs(first[0], normal)
            self.assertIs(first[1], alpha)
            self.assertIs(second[0], normal)
            self.assertIs(second[1], alpha)
        mocked_render.assert_called_once()

    def test_legacy_backup_checksum(self):
        backup = Path(__file__).resolve().parents[1] / "scripts" / "loss_legacy_20260826.py"
        digest = hashlib.sha256(backup.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "3f6cfb6f33499536ef775f946979b5f6726a62ec43746692e0db2925ee70d0f9",
        )


if __name__ == "__main__":
    unittest.main()
