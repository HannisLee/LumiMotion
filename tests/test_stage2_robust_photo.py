"""正RGB残差减权的默认精确兼容、分母/符号/梯度和显式关闭。"""
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scripts.loss_stage2 import photometric_rgb_l1, LambertianLossPreset, apply_loss_preset


class RobustPhotoTest(unittest.TestCase):
    def test_default_formula_and_gradients_bit_exact(self):
        image = torch.randn(3, 2, 2, requires_grad=True); target = torch.randn_like(image)
        mask = torch.tensor([[[1., 1.], [1., 0.]]]); count = mask.sum()
        expected = ((image-target).abs()*mask).sum()/(count*3)
        actual = photometric_rgb_l1(image, target, mask, count)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, image)[0], torch.autograd.grad(expected, image)[0], rtol=0, atol=0)

    def test_positive_outlier_only_downweighted_original_denominator_kept(self):
        image = torch.tensor([[[0., .1, 0.]]]*3, requires_grad=True)
        target = torch.tensor([[[.2, 0., .5]]]*3)
        mask = torch.tensor([[[1., 1., 0.]]])
        loss = photometric_rgb_l1(image, target, mask, mask.sum(), .25, .03)
        self.assertAlmostEqual(float(loss), .075, places=7)
        gradient = torch.autograd.grad(loss, image)[0]
        torch.testing.assert_close(gradient, torch.tensor([[[-.25/6, 1/6, 0.]]]*3))

    def test_small_positive_error_and_zero_outlier_weight_are_finite(self):
        image = torch.zeros(3, 1, 2, requires_grad=True)
        target = torch.tensor([[[.02, .2]]]*3); mask = torch.ones(1, 1, 2)
        loss = photometric_rgb_l1(image, target, mask, mask.sum(), 0., .03)
        self.assertAlmostEqual(float(loss), .01, places=7)
        torch.testing.assert_close(torch.autograd.grad(loss, image)[0], torch.tensor([[[-1/6, 0.]]]*3))

    def test_invalid_weight_and_threshold_rejected(self):
        for weight in [-.1, 1.1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError): LambertianLossPreset(SimpleNamespace(photometric_positive_residual_weight=weight))
        for threshold in [-.1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError): LambertianLossPreset(SimpleNamespace(photometric_positive_residual_threshold=threshold))
        LambertianLossPreset(SimpleNamespace())

    def test_preset_opt_in_and_explicit_disable(self):
        parser = ArgumentParser(); Stage2OptimizationParams(parser); Stage2PipelineParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.photometric_positive_residual_weight, 1.)
        args.render_mode='photometric_lambertian'; args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_robust_photo'; apply_loss_preset(args, [])
        self.assertEqual(args.photometric_positive_residual_weight, .25)
        self.assertEqual(args.photometric_positive_residual_threshold, .03)
        args.photometric_positive_residual_weight=1.
        apply_loss_preset(args, ['--photometric_positive_residual_weight','1'])
        self.assertEqual(args.photometric_positive_residual_weight, 1.)


if __name__=='__main__': unittest.main()
