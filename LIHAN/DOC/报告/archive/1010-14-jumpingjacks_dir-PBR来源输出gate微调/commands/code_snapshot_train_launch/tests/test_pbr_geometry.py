"""PBR可选几何的来源预测拟合、信赖域、独立preset和梯度。"""
import math
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import torch
from arguments.stage2 import Stage2PipelineParams, Stage2OptimizationParams
from scene.pbr_geometry import fit_source_plane, quaternion_normal, project_geometry, NativePBRGeometry, validate_pbr_geometry
from scripts.loss_stage2 import apply_loss_preset


class PBRGeometryTest(unittest.TestCase):
    def test_plane_derived_from_tilted_source_positions(self):
        torch.manual_seed(1010)
        xy = torch.rand((500, 2), dtype=torch.float64)*2-1
        xyz = torch.cat((xy, (.03*xy[:, :1]-.02*xy[:, 1:2]+.1)), 1)
        outliers = torch.rand((50, 3), dtype=torch.float64)
        outliers[:, 2] += 1
        xyz = torch.cat((xyz, outliers), 0)
        q = torch.zeros((550, 4), dtype=torch.float64); q[:, 0] = 1
        n, d, support, record = fit_source_plane(xyz, torch.ones(550, dtype=torch.bool),
                                               torch.ones((550, 1)), q)
        target = torch.tensor([-.03, .02, 1.], dtype=torch.float64)
        target /= target.norm()
        torch.testing.assert_close(n, target, atol=1e-6, rtol=1e-6)
        self.assertLess(float((xyz[:500]@n+d).abs().max()), 1e-6)
        self.assertTrue(support[:500].all())
        self.assertFalse(record['gt_normal_inputs'])
        self.assertFalse(record['gt_geometry_inputs'])

    def test_source_without_static_support_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_source_plane(torch.rand(40, 3), torch.zeros(40, dtype=torch.bool),
                             torch.ones((40, 1)), torch.ones((40, 4)))

    def test_projection_shortest_arc_and_world_distance(self):
        source = torch.zeros((3, 3)); xyz = torch.tensor([[1., 0., 0.], [.01, 0., 0.], [0., 0., 0.]])
        q0 = torch.tensor([[2., 0., 0., 0.]]).repeat(3, 1)
        q = torch.tensor([[0., 0., 2., 0.], [-2., 0., 0., 0.], [2., 0., 0., 0.]])
        project_geometry(xyz, q, source, q0, .08, 60.)
        torch.testing.assert_close(xyz.norm(dim=-1), torch.tensor([.08, .01, 0.]))
        torch.testing.assert_close(q.norm(dim=-1), q0.norm(dim=-1))
        dot = (torch.nn.functional.normalize(q, dim=-1)*torch.nn.functional.normalize(q0, dim=-1)).sum(-1).abs()
        angle = 2*torch.acos(dot.clamp(-1, 1))*180/math.pi
        self.assertLess(float(angle.max()), 60.0001)
        self.assertTrue(torch.isfinite(q).all())

    def test_geometry_preset_opt_in_and_validation(self):
        parser=ArgumentParser(); Stage2PipelineParams(parser); Stage2OptimizationParams(parser)
        default=parser.parse_args([])
        self.assertFalse(default.pbr_optimize_geometry)
        args=parser.parse_args(['--loss_preset','pbr_gt_directional_geometry'])
        args.load_iter=35000; args.iterations=55000
        apply_loss_preset(args,['--loss_preset','pbr_gt_directional_geometry'])
        self.assertTrue(args.pbr_optimize_geometry)
        self.assertTrue(args.pbr_freeze_opacity)
        self.assertEqual(args.pbr_light_mode,'gt_directional')
        validate_pbr_geometry(args)
        args.lambda_pbr_plane_normal=float('nan')
        with self.assertRaises(ValueError): validate_pbr_geometry(args)

    def test_quaternion_normal_has_finite_rotation_gradient(self):
        q=torch.tensor([[1., .1, .3, .02]],requires_grad=True)
        n=quaternion_normal(q)
        (1-n[:,2].square()).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.norm()),0.)


if __name__=='__main__': unittest.main()
