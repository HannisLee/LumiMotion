"""畸变新preset的关闭兼容、参数拒绝及预测深度损失入口。"""
import unittest
from argparse import ArgumentParser
from unittest.mock import patch

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual, geometry_losses
from scripts.loss_stage2 import apply_loss_preset


class DistortionTest(unittest.TestCase):
    def args(self):
        parser = ArgumentParser()
        Stage2OptimizationParams(parser)
        Stage2PipelineParams(parser)
        return parser.parse_args([])

    def test_old_default_and_explicit_disable_preserve_state(self):
        args = self.args()
        xyz = torch.randn(8, 3)
        self.assertFalse(GeometryResidual(xyz, args).distortion_enabled)
        args.render_mode = 'photometric_lambertian'
        args.photometric_light_mode = 'gt_directional'
        args.loss_preset = 'lambertian_geometry_area_distortion'
        apply_loss_preset(args, [])
        self.assertEqual(args.lambda_geometry_distortion, 1e6)
        new = GeometryResidual(xyz, args)
        self.assertTrue(new.distortion_enabled)
        args.lambda_geometry_distortion = 0.
        apply_loss_preset(args, ['--lambda_geometry_distortion', '0'])
        old = GeometryResidual(xyz, args)
        self.assertFalse(old.distortion_enabled)
        self.assertEqual(set(new.state_dict()), set(old.state_dict()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)

    def test_reject_nonfinite_or_negative_weight(self):
        for weight in [-1., float('inf'), float('nan')]:
            args = self.args()
            args.lambda_geometry_distortion = weight
            with self.assertRaises(ValueError):
                GeometryResidual(torch.randn(8, 3), args)

    def test_loss_uses_full_predicted_map_without_normal_target(self):
        args = self.args()
        args.lambda_geometry_distortion = 1e6
        for key in ['normal', 'alpha', 'depth', 'anchor', 'rotation_anchor']:
            setattr(args, 'lambda_geometry_'+key, 0.)
        geometry = GeometryResidual(torch.randn(8, 3), args)
        depth = torch.ones(1, 8, 8)
        normal = torch.zeros(3, 8, 8)
        normal[2] = 1.
        distortion = torch.linspace(0, 1e-8, 64).reshape(1, 8, 8).requires_grad_()
        package = dict(surf_depth=depth, normal=normal, rend_alpha=depth,
                       geometry_distortion=distortion)
        reference = dict(depth=depth, alpha=depth)
        with patch('utils.point_utils.depth_to_normal', return_value=(normal.permute(1,2,0), None)):
            loss, terms = geometry_losses(None, package, reference, depth, geometry, args)
        torch.testing.assert_close(loss, distortion.mean()*1e6, atol=0, rtol=0)
        self.assertIn('geometry_distortion', terms)
        loss.backward()
        torch.testing.assert_close(distortion.grad, torch.full_like(distortion, 1e6/64), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
