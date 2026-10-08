"""预测一致性仅XYZ的可选反传：保留前向、默认梯度与梯度累积。"""
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.nn import functional as F
from arguments.stage2 import Stage2OptimizationParams,Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual,geometry_losses
from scripts.loss_stage2 import apply_loss_preset
from scripts.stage2_runtime import backward_geometry_objective

class PositionConsistencyTest(unittest.TestCase):
    def graph(self,mode):
        x=torch.tensor([.3,.2,.1],requires_grad=True);rotation=torch.tensor([.2,.1,.3],requires_grad=True);scale=torch.tensor([.1,.3,.2],requires_grad=True)
        geometry=SimpleNamespace(translation=x,axis_angle=rotation,log_scale=scale,normal_gradient_mode=mode)
        normal=(x.square()+x*rotation+scale.square()).sum();photo=(x+3*rotation+2*scale).square().sum()
        return geometry,normal,photo
    def test_default_backward_matches_old_bit_exact(self):
        a,n,p=self.graph('joint');b,m,q=self.graph('joint')
        args=SimpleNamespace(geometry_accumulate=2,lambda_geometry_normal=.7)
        backward_geometry_objective(p+.7*n,n,a,args);((q+.7*m)/2).backward()
        for field in ['translation','axis_angle','log_scale']:torch.testing.assert_close(getattr(a,field).grad,getattr(b,field).grad,rtol=0,atol=0)
    def test_only_xyz_gets_consistency_other_parameters_keep_photo(self):
        a,n,p=self.graph('position');b,m,q=self.graph('joint')
        expected=torch.autograd.grad(q+.7*m,b.translation,retain_graph=True)[0]
        other=torch.autograd.grad(q,[b.axis_angle,b.log_scale])
        backward_geometry_objective(p+.7*n,n,a,SimpleNamespace(geometry_accumulate=1,lambda_geometry_normal=.7))
        torch.testing.assert_close(a.translation.grad,expected)
        for actual,wanted in zip([a.axis_angle.grad,a.log_scale.grad],other):torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
    def test_two_view_accumulation_preserves_average(self):
        a,n,p=self.graph('position');b,m,q=self.graph('position')
        args=SimpleNamespace(geometry_accumulate=2,lambda_geometry_normal=.7)
        expected=torch.autograd.grad(q+.7*m,b.translation,retain_graph=True)[0]
        backward_geometry_objective(p+.7*n,n,a,args)
        n2=(a.translation.square()+a.translation*a.axis_angle+a.log_scale.square()).sum()
        p2=(a.translation+3*a.axis_angle+2*a.log_scale).square().sum()
        backward_geometry_objective(p2+.7*n2,n2,a,args)
        torch.testing.assert_close(a.translation.grad,expected)
    def test_rgb_warmup_bypasses_selective_geometry_backward(self):
        a,n,p=self.graph('position');b,m,q=self.graph('position')
        args=SimpleNamespace(geometry_accumulate=2,lambda_geometry_normal=.7)
        backward_geometry_objective(p,None,a,args,prior_only=True);(q/2).backward()
        for field in ['translation','axis_angle','log_scale']:torch.testing.assert_close(getattr(a,field).grad,getattr(b,field).grad,rtol=0,atol=0)
    def test_preset_default_off_and_explicit_override(self):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser);args=parser.parse_args([])
        self.assertEqual(args.geometry_normal_gradient_mode,'joint');args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_position_consistency';apply_loss_preset(args,[])
        self.assertEqual(args.geometry_normal_gradient_mode,'position')
        args.geometry_normal_gradient_mode='joint';apply_loss_preset(args,['--geometry_normal_gradient_mode','joint'])
        self.assertEqual(args.geometry_normal_gradient_mode,'joint')
    def test_same_forward_normal_penalty_and_detached_model_target(self):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser);args=parser.parse_args([])
        for key in ['alpha','depth','anchor','rotation_anchor']:setattr(args,'lambda_geometry_'+key,0.)
        args.lambda_geometry_normal=1.;results={}
        for mode in ['joint','position']:
            args.geometry_normal_gradient_mode=mode;g=GeometryResidual(torch.randn(8,3),args)
            depth=torch.ones(1,8,8,requires_grad=True);target=torch.tensor([.5,0.,1.]).view(3,1,1).expand(3,8,8).clone().requires_grad_();alpha=torch.ones_like(depth)
            def normals(camera,z):return F.normalize(torch.cat((z*.2,z*0,z*0+1),dim=0),dim=0).permute(1,2,0),None
            with patch('utils.point_utils.depth_to_normal',side_effect=normals):
                total,terms=geometry_losses(None,dict(normal=F.normalize(target,dim=0),surf_depth=depth,rend_alpha=alpha),dict(depth=depth.detach(),alpha=alpha),alpha,g,args)
            total.backward();results[mode]=(terms['geometry_normal'].detach(),target.grad,depth.grad)
        torch.testing.assert_close(results['joint'][0],results['position'][0],rtol=0,atol=0)
        self.assertIsNone(results['position'][1]);self.assertGreater(float(results['position'][2].norm()),0)
if __name__=='__main__':unittest.main()
