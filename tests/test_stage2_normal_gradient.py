"""预测法线目标的可选梯度方向；保持旧loss/state并验证深度梯度。"""
import unittest
from argparse import ArgumentParser
from unittest.mock import patch

import torch
from torch.nn import functional as F
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual, geometry_losses
from scripts.loss_stage2 import apply_loss_preset


class NormalGradientTest(unittest.TestCase):
    def args(self):
        p = ArgumentParser()
        Stage2OptimizationParams(p); Stage2PipelineParams(p)
        args = p.parse_args([])
        for key in ['alpha','depth','anchor','rotation_anchor']:
            setattr(args,'lambda_geometry_'+key,0.)
        args.lambda_geometry_normal = 1.
        return args

    def test_legacy_missing_mode_and_joint_state_are_identical(self):
        args = self.args(); xyz = torch.randn(8,3)
        joint = GeometryResidual(xyz,args)
        del args.geometry_normal_gradient_mode
        legacy = GeometryResidual(xyz,args)
        self.assertEqual(joint.normal_gradient_mode,'joint')
        self.assertEqual(legacy.normal_gradient_mode,'joint')
        self.assertEqual(joint.state_dict().keys(),legacy.state_dict().keys())
        for k in joint.state_dict():
            torch.testing.assert_close(joint.state_dict()[k],legacy.state_dict()[k],rtol=0,atol=0)

    def test_mode_rejects_unknown_values(self):
        args = self.args()
        for mode in ['depth_only','gt','',None]:
            args.geometry_normal_gradient_mode = mode
            with self.assertRaises(ValueError):
                GeometryResidual(torch.randn(8,3),args)

    def test_preset_and_explicit_joint_override(self):
        args = self.args();args.render_mode='photometric_lambertian'
        args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_depth_follow'
        apply_loss_preset(args,[])
        self.assertEqual(args.geometry_normal_gradient_mode,'depth')
        args.geometry_normal_gradient_mode='joint'
        apply_loss_preset(args,['--geometry_normal_gradient_mode','joint'])
        self.assertEqual(args.geometry_normal_gradient_mode,'joint')

    def test_same_forward_loss_blocks_only_render_target_gradient(self):
        xyz = torch.randn(8,3)
        values, gradients = {}, {}
        for mode in ['joint','depth']:
            args = self.args();args.geometry_normal_gradient_mode=mode
            geometry = GeometryResidual(xyz,args)
            depth = torch.ones(1,8,8,requires_grad=True)
            target = torch.tensor([.5,0.,1.]).view(3,1,1).expand(3,8,8).clone().requires_grad_()
            normal = F.normalize(target,dim=0)
            alpha = torch.ones_like(depth)
            def predicted_depth_normal(camera,z):
                n = torch.cat((z*.2,z*0,z*0+1),dim=0)
                return F.normalize(n,dim=0).permute(1,2,0),None
            package = dict(normal=normal,surf_depth=depth,rend_alpha=alpha)
            reference = dict(depth=depth.detach(),alpha=alpha)
            with patch('utils.point_utils.depth_to_normal',side_effect=predicted_depth_normal):
                loss,terms = geometry_losses(None,package,reference,alpha,geometry,args)
            values[mode] = loss.detach()
            loss.backward()
            gradients[mode] = dict(target=target.grad,depth=depth.grad)
        torch.testing.assert_close(values['joint'],values['depth'],atol=0,rtol=0)
        self.assertGreater(float(gradients['joint']['target'].norm()),0)
        self.assertIsNone(gradients['depth']['target'])
        self.assertTrue(torch.isfinite(gradients['depth']['depth']).all())
        self.assertGreater(float(gradients['depth']['depth'].norm()),0)
        torch.testing.assert_close(gradients['joint']['depth'],gradients['depth']['depth'],atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
