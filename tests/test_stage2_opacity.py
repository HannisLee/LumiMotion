"""有界opacity残差：旧状态兼容、源alpha梯度、信赖域及细分重载。"""
import math
import unittest
from argparse import ArgumentParser

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual
from scripts.loss_stage2 import apply_loss_preset


class OpacityTest(unittest.TestCase):
    def args(self):
        parser = ArgumentParser()
        Stage2OptimizationParams(parser)
        Stage2PipelineParams(parser)
        return parser.parse_args([])

    def test_default_identity_and_preset_explicit_priority(self):
        args = self.args()
        xyz = torch.randn(8, 3)
        opacity = torch.tensor([[0.], [.1], [.9], [1.]]).repeat(2, 1)
        old = GeometryResidual(xyz, args)
        self.assertIs(old.apply_opacity(opacity), opacity)
        self.assertNotIn('log_opacity', old.state_dict())
        args.loss_preset = 'lambertian_geometry_area_opacity'
        args.render_mode = 'photometric_lambertian'
        args.photometric_light_mode = 'gt_directional'
        apply_loss_preset(args, [])
        self.assertTrue(args.geometry_optimize_opacity)
        self.assertTrue(args.geometry_optimize_scale)
        new = GeometryResidual(xyz, args)
        expanded = new.expand_values(opacity)
        torch.testing.assert_close(new.apply_opacity(expanded), expanded, rtol=0, atol=0)
        args.geometry_optimize_opacity = False
        apply_loss_preset(args, ['--geometry_optimize_opacity'])
        self.assertFalse(args.geometry_optimize_opacity)

    def test_source_alpha_gradient_including_saturated_source(self):
        args = self.args()
        args.geometry_optimize_opacity = True
        g = GeometryResidual(torch.randn(8, 3), args)
        opacity = torch.tensor([[0.], [.1], [.9], [1.]]).repeat(2, 1)
        g.apply_opacity(opacity).sum().backward()
        torch.testing.assert_close(g.log_opacity.grad, opacity, rtol=0, atol=0)
        with torch.no_grad():
            g.log_opacity.fill_(-math.log(2))
        torch.testing.assert_close(g.apply_opacity(opacity), opacity*.5, rtol=0, atol=0)

    def test_projection_clears_only_clipped_adam_moments(self):
        args = self.args()
        args.geometry_optimize_opacity = True
        g = GeometryResidual(torch.randn(8, 3), args)
        opt = torch.optim.Adam([g.log_opacity], lr=.01)
        g.log_opacity.sum().backward()
        opt.step()
        with torch.no_grad():
            g.log_opacity[0] = -10
            g.log_opacity[1] = 10
        g.project(opt)
        torch.testing.assert_close(g.log_opacity[:2, 0], torch.tensor([-math.log(2), 0.]))
        for key in ['exp_avg', 'exp_avg_sq']:
            self.assertEqual(float(opt.state[g.log_opacity][key][:2].abs().sum()), 0.)
            self.assertGreater(float(opt.state[g.log_opacity][key][2:].abs().sum()), 0.)
        d = g.diagnostics()
        self.assertEqual(d['opacity_ratio_min'], .5)
        self.assertEqual(d['opacity_ratio_max'], 1.)
        for ratio in [1., 0., float('nan'), float('inf')]:
            args.geometry_max_opacity_ratio = ratio
            with self.assertRaises(ValueError):
                GeometryResidual(torch.randn(8, 3), args)

    def test_subdivision_has_independent_children_and_exact_reload(self):
        args = self.args()
        args.geometry_subdivide = True
        args.geometry_optimize_opacity = True
        xyz = torch.randn(8, 3)
        g = GeometryResidual(xyz, args)
        q = torch.tensor([1., 0., 0., 0.]).expand(8, -1)
        scales, opacity = torch.ones(8, 2), torch.ones(8, 1)
        with torch.no_grad():
            g.log_opacity[0] = -.2
        alpha = g.prepare(xyz, q, scales, opacity)[3]
        self.assertLess(float(alpha[0]), float(alpha[1]))
        alpha.sum().backward()
        torch.testing.assert_close(g.log_opacity.grad, alpha)
        clone = GeometryResidual(xyz, args)
        clone.load_state_dict(g.state_dict())
        for a, b in zip(g.prepare(xyz, q, scales, opacity), clone.prepare(xyz, q, scales, opacity)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
