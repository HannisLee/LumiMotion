"""RGB旋转预热的阶段边界、旧调度精确兼容与冻结Adam状态。"""
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import (GeometryResidual, geometry_schedule, geometry_group_lr,
    geometry_training_phase, rgb_normal_prior_loss)
from scripts.loss_stage2 import apply_loss_preset


class RGBWarmupTest(unittest.TestCase):
    def args(self):
        p = ArgumentParser(); Stage2OptimizationParams(p); Stage2PipelineParams(p)
        return p.parse_args([])

    def test_disabled_and_missing_option_preserve_legacy_schedule_exactly(self):
        args = self.args()
        for missing in [False, True]:
            if missing: del args.geometry_rgb_prior_warmup
            for step in [0, 1999, 2000, 3999, 4000, 20000, 30000]:
                self.assertEqual(geometry_training_phase(step, args), (False, step))
                for name, lr, start in [('xyz', args.geometry_xyz_lr, args.geometry_xyz_start),
                        ('rotation', args.geometry_rotation_lr, args.geometry_warmup),
                        ('scale', args.geometry_scale_lr, args.geometry_scale_start),
                        ('opacity', args.geometry_opacity_lr, args.geometry_opacity_start)]:
                    self.assertEqual(geometry_group_lr('geometry_'+name, step, args), lr*geometry_schedule(step, start, args))
                self.assertEqual(geometry_group_lr('albedo', step, args), args.photometric_albedo_lr*(1. if step < args.geometry_warmup else .3))

    def test_boundary_resets_joint_schedule_and_keeps_frozen_groups_zero(self):
        args = self.args(); args.geometry_rgb_prior_warmup = 20; args.geometry_ramp = 5
        args.geometry_warmup = 0; args.geometry_xyz_start = 7; args.geometry_scale_start = 7
        for step in [0, 4, 19]:
            self.assertEqual(geometry_training_phase(step, args), (True, step))
            self.assertEqual(geometry_group_lr('geometry_rotation', step, args), args.geometry_rotation_lr*min(1., (step+1)/5))
            for name in ['albedo', 'geometry_xyz', 'geometry_scale', 'geometry_opacity', 'light']:
                self.assertEqual(geometry_group_lr(name, step, args), 0.)
        self.assertEqual(geometry_training_phase(20, args), (False, 0))
        self.assertEqual(geometry_group_lr('geometry_rotation', 20, args), args.geometry_rotation_lr*geometry_schedule(0, 0, args))
        self.assertEqual(geometry_group_lr('geometry_xyz', 26, args), 0.)
        self.assertGreater(geometry_group_lr('geometry_xyz', 27, args), 0.)

    def test_rgb_loss_preserves_formula_and_eligibility_gradients(self):
        normals = torch.tensor([[.9, .1, 0.], [.1, .8, 0.], [.3, .3, 0.]], requires_grad=True)
        g = SimpleNamespace(rgb_normal_targets=torch.tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]),
            rgb_normal_valid=torch.tensor([True, True, False]))
        expected = ((1-(normals*g.rgb_normal_targets).sum(-1).clamp(-1, 1))*g.rgb_normal_valid).sum()/g.rgb_normal_valid.sum().clamp_min(1)
        actual = rgb_normal_prior_loss(normals, g)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.autograd.grad(actual, normals)[0]
        torch.testing.assert_close(gradient, torch.tensor([[-.5, 0., 0.], [0., -.5, 0.], [0., 0., 0.]]), rtol=0, atol=0)
        g.rgb_normal_valid[:] = False
        self.assertEqual(float(rgb_normal_prior_loss(normals, g)), 0.)

    def test_zero_lr_with_grad_none_preserves_existing_adam_momentum(self):
        p = torch.nn.Parameter(torch.tensor([.2])); q = torch.nn.Parameter(torch.tensor([.1]))
        optimizer = torch.optim.Adam([{'params': [p], 'lr': .01}, {'params': [q], 'lr': .01}])
        (p.square()+q.square()).sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        before = p.detach().clone(); state = {k: v.clone() for k, v in optimizer.state[p].items()}
        optimizer.param_groups[0]['lr'] = 0.
        (p.square()+q.square()).sum().backward(); p.grad = None; optimizer.step()
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        for key, value in state.items(): torch.testing.assert_close(optimizer.state[p][key], value, rtol=0, atol=0)

    def test_opt_in_preset_invalid_prior_and_legacy_state_layout(self):
        args = self.args(); args.render_mode = 'photometric_lambertian'; args.photometric_light_mode = 'gt_directional'
        self.assertEqual(args.geometry_rgb_prior_warmup, 0)
        for count in [-1, .5, True]:
            args.geometry_rgb_prior_warmup = count
            with self.assertRaises(ValueError): GeometryResidual(torch.randn(10, 3), args)
        args.geometry_rgb_prior_warmup = 20
        with self.assertRaises(ValueError): GeometryResidual(torch.randn(10, 3), args)
        args.loss_preset = 'lambertian_geometry_area_rgb_warmup'; apply_loss_preset(args, [])
        self.assertEqual(args.geometry_rgb_prior_warmup, 2000)
        args.geometry_rgb_normal_prior_path = 'source-verified-training-rgb.npz'
        a = GeometryResidual(torch.randn(10, 3), args)
        args.geometry_rgb_prior_warmup = 0
        b = GeometryResidual(torch.randn(10, 3), args)
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        a.load_state_dict(b.state_dict(), strict=True)
        apply_loss_preset(args, ['--geometry_rgb_prior_warmup', '0'])
        self.assertEqual(args.geometry_rgb_prior_warmup, 0)

    def test_rotation_only_projection_keeps_frozen_parameters_and_momentum_bit_exact(self):
        args = self.args(); args.geometry_optimize_scale = True
        g = GeometryResidual(torch.randn(10, 3), args)
        with torch.no_grad():
            g.translation[:, 0] = g.scene_extent*g.max_displacement*1.0000001
            g.axis_angle[:, 0] = g.max_angle*2
            g.log_scale[:] = .01
        optimizer = torch.optim.Adam(g.parameters())
        for p in g.parameters(): optimizer.state[p]['exp_avg'] = torch.ones_like(p)
        frozen = {key: g.state_dict()[key].clone() for key in ['translation', 'log_scale']}
        xyz_momentum = optimizer.state[g.translation]['exp_avg'].clone()
        g.project(optimizer, rotation_only=True)
        for key, value in frozen.items(): torch.testing.assert_close(g.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[g.translation]['exp_avg'], xyz_momentum, rtol=0, atol=0)
        self.assertLessEqual(float(g.axis_angle.norm(dim=-1).max()), g.max_angle+1e-7)
        self.assertEqual(float(optimizer.state[g.axis_angle]['exp_avg'].abs().max()), 0.)


if __name__ == '__main__': unittest.main()
