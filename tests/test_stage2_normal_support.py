"""光度支持区域减权的语义、梯度及旧检查点兼容。"""
import unittest
from argparse import ArgumentParser

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual, supported_normal_loss
from scripts.loss_stage2 import apply_loss_preset


class NormalSupportTest(unittest.TestCase):
    def args(self):
        p=ArgumentParser();Stage2OptimizationParams(p);Stage2PipelineParams(p)
        return p.parse_args([])

    def test_default_forward_backward_bit_exact_and_ignores_support(self):
        values=torch.tensor([.3,.7,.9],requires_grad=True);valid=torch.tensor([1.,1.,0.])
        old=(values*valid).sum()/valid.sum().clamp_min(1)
        new=supported_normal_loss(values,valid)
        torch.testing.assert_close(new,old,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(new,values)[0],torch.autograd.grad(old,values)[0],rtol=0,atol=0)
        torch.testing.assert_close(supported_normal_loss(values,valid,1.,torch.tensor(float('nan'))),old,rtol=0,atol=0)

    def test_dark_regions_are_downweighted_without_renormalizing_and_mask_detached(self):
        values=torch.tensor([2.,4.,100.],requires_grad=True);valid=torch.tensor([1.,1.,0.])
        support=torch.tensor([1.,0.,1.],requires_grad=True)
        loss=supported_normal_loss(values,valid,.1,support)
        self.assertAlmostEqual(float(loss),1.2,places=6)
        loss.backward()
        torch.testing.assert_close(values.grad,torch.tensor([.5,.05,0.]))
        self.assertIsNone(support.grad)
        self.assertAlmostEqual(float(supported_normal_loss(values,valid,.1,torch.zeros(3))),.3,places=6)
        self.assertEqual(float(supported_normal_loss(values,valid,0.,torch.zeros(3))),0.)

    def test_empty_valid_domain_finite_and_missing_or_invalid_support_rejected(self):
        self.assertEqual(float(supported_normal_loss(torch.ones(3),torch.zeros(3),.1,torch.ones(3))),0.)
        for support in [None,torch.ones(2),torch.tensor([1.,.5,0.]),torch.tensor([1.,float('nan'),0.])]:
            with self.assertRaises(ValueError):supported_normal_loss(torch.ones(3),torch.ones(3),.1,support)
        for weight in [-.1,1.1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):supported_normal_loss(torch.ones(3),torch.ones(3),weight,torch.ones(3))
            args=self.args();args.geometry_normal_unsupported_weight=weight
            with self.assertRaises(ValueError):GeometryResidual(torch.randn(10,3),args)

    def test_preset_explicit_disable_and_old_state_layout(self):
        args=self.args()
        self.assertEqual(args.geometry_normal_unsupported_weight,1.)
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_photo_supported';apply_loss_preset(args,[])
        self.assertEqual(args.geometry_normal_unsupported_weight,.1)
        args.geometry_subdivide=False;args.geometry_optimize_scale=False
        new=GeometryResidual(torch.randn(10,3),args)
        args.geometry_normal_unsupported_weight=1.
        old=GeometryResidual(torch.randn(10,3),args)
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        new.load_state_dict(old.state_dict(),strict=True)
        args.geometry_normal_unsupported_weight=1.;apply_loss_preset(args,['--geometry_normal_unsupported_weight','1'])
        self.assertEqual(args.geometry_normal_unsupported_weight,1.)
        expected_legacy=GeometryResidual(torch.randn(10,3),args)
        del args.geometry_normal_unsupported_weight
        legacy=GeometryResidual(torch.randn(10,3),args)
        self.assertEqual(set(legacy.state_dict()),set(expected_legacy.state_dict()))


if __name__=='__main__':unittest.main()
