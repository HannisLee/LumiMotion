"""邻域梯度分配的数值前向、双端法线及默认关闭协议。"""
import math
import unittest
from argparse import ArgumentParser

import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual
from scripts.loss_stage2 import apply_loss_preset


class CoherencePreconditionTest(unittest.TestCase):
    def args(self, **kwargs):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser)
        args=parser.parse_args([])
        for k,v in kwargs.items():setattr(args,k,v)
        return args

    def circle(self):
        u=torch.arange(40,dtype=torch.float64)*2*math.pi/40
        return torch.stack((u.cos(),u.sin(),u*0),-1)

    def test_position_gradient_scales_normal_gradient_and_value_unchanged(self):
        xyz=self.circle();g=GeometryResidual(xyz,self.args(lambda_geometry_coherence=.01))
        g.initialize_coherence(xyz,xyz,torch.full((40,2),.3,dtype=torch.float64),torch.ones(40,1,dtype=torch.float64))
        means=xyz.clone();means[0]+=means.new_tensor([.02,.01,.01]);means.requires_grad_()
        normals=xyz.clone();normals[1]+=normals.new_tensor([.01,.02,.1]);normals.requires_grad_()
        reference=g.coherence_loss(means,normals)
        p,n=torch.autograd.grad(reference,[means,normals])
        self.assertGreater(float(p.norm()),0);self.assertGreater(float(n.norm()),0)
        g.coherence_position_gradient_scale=.01
        balanced=g.coherence_loss(means,normals)
        pb,nb=torch.autograd.grad(balanced,[means,normals])
        torch.testing.assert_close(reference,balanced,rtol=0,atol=0)
        torch.testing.assert_close(pb,p*.01,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(nb,n,rtol=0,atol=0)
        # λ从0.01→1时，位置保持旧加权量级，法线100倍。
        torch.testing.assert_close(pb,p*.01,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(nb,(n*.01)*100,rtol=1e-12,atol=1e-14)

    def test_curve_zero_and_saved_graph_default_remains_compatible(self):
        xyz=self.circle();default=GeometryResidual(xyz,self.args(lambda_geometry_coherence=.01))
        default.initialize_coherence(xyz,xyz,torch.full((40,2),.3,dtype=torch.float64),torch.ones(40,1,dtype=torch.float64))
        balanced=GeometryResidual(xyz,self.args(lambda_geometry_coherence=1.,geometry_coherence_position_gradient_scale=.01))
        balanced.load_state_dict(default.state_dict())
        self.assertEqual(default.state_dict().keys(),balanced.state_dict().keys())
        self.assertLess(float(balanced.coherence_loss(xyz,xyz)),1e-10)
        torch.testing.assert_close(default.coherence_loss(xyz,xyz),balanced.coherence_loss(xyz,xyz),rtol=0,atol=0)

    def test_invalid_scale_and_explicit_disabled_preset(self):
        xyz=self.circle()
        for scale in [0.,-1.,1.001,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):GeometryResidual(xyz,self.args(geometry_coherence_position_gradient_scale=scale))
        args=self.args();self.assertEqual(args.geometry_coherence_position_gradient_scale,1.)
        args.loss_preset='lambertian_geometry_area_balanced_coherent'
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.lambda_geometry_coherence=0.
        apply_loss_preset(args,['--lambda_geometry_coherence','0'])
        self.assertEqual(args.geometry_coherence_position_gradient_scale,.01)
        self.assertTrue(args.geometry_refined_coherence)
        g=GeometryResidual(xyz,args);self.assertFalse(g.coherence_enabled)
        self.assertFalse(any(k.startswith('coherence_') for k in g.state_dict()))


if __name__=='__main__':unittest.main()
