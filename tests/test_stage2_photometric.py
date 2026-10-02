"""数值、光照泄漏、preset 与原始 Stage2 回归检查。"""
import copy
import math
import subprocess
import textwrap
import unittest
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from scene.stage2_photometric import (diffuse_color, srgb_to_linear, linear_to_srgb,
    DirectionalLightTable, LambertianMaterial, HDRIrradiance, unit_vector, world_direction_to_hdr_uv)
from scripts.loss_stage2 import build_loss_preset, apply_loss_preset
from arguments import ModelParams
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from utils.loss_utils import l1_loss, ssim, first_order_edge_aware_loss, tv_loss


def options(**kwargs):
    parser=ArgumentParser(); Stage2OptimizationParams(parser); Stage2PipelineParams(parser)
    opt=parser.parse_args([])
    for key,value in kwargs.items():setattr(opt,key,value)
    return opt


class Stage2PhotometricTest(unittest.TestCase):
    def test_plane_brightness_and_back_light(self):
        n=torch.tensor([[0.,0.,1.]])
        rho=torch.tensor([[0.4,0.2,0.8]])
        for d,scale in [([0.,0.,1.],1),([math.sqrt(3)/2,0.,0.5],0.5),([1.,0.,0.],0),([0.,0.,-1.],0)]:
            torch.testing.assert_close(diffuse_color(rho,n,torch.tensor(d),math.pi),rho*scale)

    def test_color_roundtrip_and_gradients(self):
        x=torch.linspace(0,1,101,requires_grad=True)
        y=linear_to_srgb(srgb_to_linear(x));torch.testing.assert_close(y,x,atol=1e-6,rtol=1e-6)
        y.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_heldout_direction_has_no_parameter(self):
        light=DirectionalLightTable([0.,1.],torch.tensor([[1.,0.,1.],[-1.,0.,1.]]))
        self.assertEqual(tuple(light.raw_directions.shape),(2,3))
        torch.testing.assert_close(light(0.5),torch.tensor([0.,0.,1.]))
        torch.testing.assert_close(light(-1),unit_vector(torch.tensor([1.,0.,1.])))
        torch.testing.assert_close(light(2),unit_vector(torch.tensor([-1.,0.,1.])))
        light(0.3)[0].backward();self.assertGreater(float(light.raw_directions.grad.abs().sum()),0)

    def test_opposite_direction_interpolation_is_finite(self):
        light=DirectionalLightTable([0.,1.],torch.tensor([[1.,0.,0.],[-1.,0.,0.]]))
        torch.testing.assert_close(light(0.5),torch.tensor([1.,0.,0.]))

    def test_gt_has_no_trainable_light(self):
        light=DirectionalLightTable([0.],torch.tensor([[0.,0.,1.]]),learned=False)
        self.assertFalse(light.raw_directions.requires_grad)
        self.assertEqual(float(light.smoothness()),0)

    def test_invalid_direction_and_time(self):
        with self.assertRaises(ValueError):unit_vector(torch.zeros(3))
        with self.assertRaises(ValueError):DirectionalLightTable([1.,0.],torch.ones(2,3))

    def test_material_roundtrip(self):
        material=LambertianMaterial(torch.tensor([[0.2,0.4,0.6]]))
        clone=LambertianMaterial(torch.ones(1,3)*0.5);clone.load_state_dict(material.state_dict())
        torch.testing.assert_close(clone.albedo,material.albedo)
        self.assertEqual(float(clone.prior()),0)

    def test_white_furnace_hdr(self):
        hdr=HDRIrradiance(torch.ones(8,16,3)*2,samples=2048)
        normals=unit_vector(torch.tensor([[0.,0.,1.],[1.,0.,0.],[0.1,0.4,-0.9]]))
        torch.testing.assert_close(hdr(normals)/math.pi,torch.full_like(normals,2),atol=0.002,rtol=0.002)

    def test_hdr_axes_match_original_transform(self):
        directions=unit_vector(torch.tensor([[1.,2.,3.],[-2.,1.,-3.],[1.,-3.,2.],[-1.,-2.,3.]]))
        transform=torch.tensor([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
        env=directions@transform.T
        expected=torch.stack((torch.atan2(env[:,0],-env[:,2])/(2*math.pi)+0.5,
                              torch.acos(env[:,1])/math.pi),-1)
        torch.testing.assert_close(world_direction_to_hdr_uv(directions),expected)
        # 非均匀经度图：u=0.25 的亮区应照亮世界 +Y，而不是 -Y。
        u=(torch.arange(128)+0.5)/128
        radiance=(1+torch.cos(2*math.pi*(u-0.25))).clamp_min(0)
        hdr=HDRIrradiance(radiance[None,:,None].expand(64,128,3).clone())
        response=hdr(torch.tensor([[0.,1.,0.],[0.,-1.,0.]]))
        self.assertGreater(float(response[0,0]),float(response[1,0])*2)

    def test_cli_explicit_wins_and_mode_validation(self):
        opt=options(render_mode='photometric_lambertian',loss_preset='lambertian_default',lambda_dssim=0.5)
        apply_loss_preset(opt,['--lambda_dssim=0.5']);self.assertEqual(opt.lambda_dssim,0.5)
        opt.loss_preset='irgs_baseline'
        with self.assertRaises(ValueError):build_loss_preset(opt,opt.render_mode)

    def test_auto_does_not_override(self):
        opt=options(render_mode='photometric_lambertian',lambda_dssim=0.4)
        apply_loss_preset(opt,[]);self.assertEqual(opt.lambda_dssim,0.4)

    def test_lambertian_gradients_and_mask(self):
        opt=options(render_mode='photometric_lambertian')
        material=LambertianMaterial(torch.full((2,3),0.5))
        light=DirectionalLightTable([0.,1.],torch.tensor([[0.,0.,1.],[0.2,0.,1.]]))
        image=(material.albedo.mean(0)*light(0.5)[2])[:,None,None].expand(3,12,12)
        result=build_loss_preset(opt,opt.render_mode).compute({'render':image},torch.zeros_like(image),
                    torch.ones(1,12,12),material,light)
        result.total.backward()
        self.assertGreater(float(material.raw_albedo.grad.abs().sum()),0)
        self.assertGreater(float(light.raw_directions.grad.abs().sum()),0)
        with self.assertRaises(ValueError):build_loss_preset(opt,opt.render_mode).compute({'render':image},image,
                    torch.zeros(1,12,12),material,light)

    def test_irgs_loss_matches_initial_commit_with_all_terms(self):
        torch.manual_seed(3)
        opt=options(d_lower_hemisphere_weight=0.1,lambda_roughness_smooth=0.02,lambda_light=0.01,
                    lambda_light_smooth=0.03,lambda_base_color_smooth=0.04)
        image=torch.rand(3,16,16,requires_grad=True)
        gt=torch.rand(3,16,16)
        env=torch.rand(12,24,3,requires_grad=True)
        pkg={'mask':torch.ones(16,16,dtype=torch.bool),'render_sh':torch.rand(3,16,16,requires_grad=True),
             'rend_alpha':torch.ones(1,16,16),'roughness':torch.rand(3,16,16,requires_grad=True),
             'ray_light_direct':torch.rand(32,3,requires_grad=True),'env_only':torch.rand(3,16,16,requires_grad=True),
             'base_color_linear':torch.rand(3,16,16,requires_grad=True)}
        tr=SimpleNamespace(opt=opt,env_light=SimpleNamespace(render_env_map=lambda H:{'env1':env,'env2':env}))
        original=subprocess.check_output(['git','show','9cd834f:scripts/train_stage2.py']).decode().replace('\r\n','\n')
        start=original.index('        if self.opt.train_ray:',original.index('        gt_image = viewpoint_cam.original_image_train_light.cuda()'))
        end=original.index('        loss.backward()',start)
        scope={'self':tr,'render_pkg_re':pkg,'gt_image':gt,'image_non_masked':image,'F':F,
               'l1_loss':l1_loss,'ssim':ssim,'first_order_edge_aware_loss':first_order_edge_aware_loss,'tv_loss':tv_loss}
        tr.opt.train_ray=True
        exec(textwrap.dedent(original[start:end]),scope)
        result=build_loss_preset(opt,'original_ir').compute(tr,pkg,gt,image)
        torch.testing.assert_close(result.total,scope['loss'],rtol=0,atol=0)
        tensors=[image,env,pkg['render_sh'],pkg['roughness'],pkg['ray_light_direct'],pkg['env_only'],pkg['base_color_linear']]
        actual=torch.autograd.grad(result.total,tensors,retain_graph=True)
        expected=torch.autograd.grad(scope['loss'],tensors)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
