"""预测DC内部初始化的梯度、SH来源隔离、显式preset和默认兼容性。"""
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace
import torch
from arguments.stage2 import Stage2PipelineParams, Stage2OptimizationParams
from scripts.loss_stage2 import apply_loss_preset
from scene.pbr_material import initialize_native_albedo, validate_albedo_initialization, project_native_albedo
from utils.sh_utils import RGB2SH, SH2RGB


class PBRMaterialTest(unittest.TestCase):
    def test_source_projection_restores_gradient_and_keeps_sh_source(self):
        rgb=torch.tensor([[[-1.,.2,2.]],[[.01,.5,1.2]]])
        dc=torch.nn.Parameter(RGB2SH(rgb))
        sh=torch.nn.Parameter(dc.detach().clone())
        pc=SimpleNamespace(_albedo_dc=dc,_albedo_dc_stage1=sh)
        before=SH2RGB(dc).clamp(.03,.97).detach().clone()
        SH2RGB(dc).clamp(.03,.97).sum().backward()
        self.assertEqual(int((dc.grad==0).sum()),4)
        dc.grad=None;source_sh=sh.detach().clone()
        record=initialize_native_albedo(pc)
        after=SH2RGB(dc).clamp(.03,.97)
        self.assertLessEqual(float((after-before).abs().max()),.0010001)
        after.sum().backward()
        self.assertTrue(torch.isfinite(dc.grad).all())
        self.assertTrue((dc.grad>0).all())
        torch.testing.assert_close(sh,source_sh,atol=0,rtol=0)
        self.assertFalse(record['gt_material_inputs'])
        self.assertEqual(record['after_clamped_channels_fraction'],0.)

    def test_nonfinite_shape_margin_and_shared_source_rejected(self):
        pc=SimpleNamespace(_albedo_dc=torch.nn.Parameter(torch.zeros((2,1,3))))
        for margin in [0.,-.1,.47,float('nan')]:
            with self.assertRaises(ValueError):initialize_native_albedo(pc,margin)
        pc._albedo_dc.data[0,0,0]=float('nan')
        with self.assertRaises(ValueError):initialize_native_albedo(pc)
        pc._albedo_dc=torch.nn.Parameter(torch.zeros((2,3)))
        with self.assertRaises(ValueError):initialize_native_albedo(pc)
        pc._albedo_dc=torch.nn.Parameter(torch.zeros((2,1,3)))
        pc._albedo_dc_stage1=pc._albedo_dc
        with self.assertRaises(ValueError):initialize_native_albedo(pc)

    def test_refine_preset_only_enables_new_initializer(self):
        parser=ArgumentParser();Stage2PipelineParams(parser);Stage2OptimizationParams(parser)
        for preset in ['auto','pbr_gt_directional','pbr_gt_directional_geometry']:
            args=parser.parse_args(['--loss_preset',preset]);apply_loss_preset(args,['--loss_preset',preset])
            self.assertFalse(args.pbr_albedo_interior_init)
            self.assertFalse(args.pbr_albedo_project_after_step)
        args=parser.parse_args(['--loss_preset','pbr_gt_directional_refine'])
        apply_loss_preset(args,['--loss_preset','pbr_gt_directional_refine'])
        self.assertTrue(args.pbr_albedo_interior_init)
        self.assertTrue(args.pbr_albedo_project_after_step)
        self.assertTrue(args.pbr_optimize_geometry)
        validate_albedo_initialization(args)
        args.pbr_albedo_init_margin=float('nan')
        with self.assertRaises(ValueError):validate_albedo_initialization(args)
        args.render_mode='photometric_lambertian'
        with self.assertRaises(ValueError):validate_albedo_initialization(args)

    def test_auto_mode_selects_refine_for_explicit_initializer(self):
        parser=ArgumentParser();Stage2PipelineParams(parser);Stage2OptimizationParams(parser)
        argv=['--pbr_light_mode','gt_directional','--pbr_albedo_interior_init']
        args=parser.parse_args(argv);apply_loss_preset(args,argv)
        self.assertEqual(args.loss_preset,'pbr_gt_directional_refine')

    def test_projection_keeps_gradient_after_adam_crosses_native_boundary(self):
        dc=torch.nn.Parameter(RGB2SH(torch.full((2,1,3),.969)))
        pc=SimpleNamespace(_albedo_dc=dc)
        optimizer=torch.optim.Adam([dc],lr=.01)
        for _ in range(8):
            optimizer.zero_grad()
            loss=(SH2RGB(dc).clamp(.03,.97)-2).square().sum()
            loss.backward()
            self.assertTrue((dc.grad!=0).all())
            optimizer.step()
            self.assertGreater(float(SH2RGB(dc).max()),.97)
            project_native_albedo(pc)
            self.assertLessEqual(float(SH2RGB(dc).max()),.9690001)
            self.assertGreaterEqual(float(SH2RGB(dc).min()),.0309999)

    def test_projection_keeps_sh_source_and_validates_margin(self):
        dc=torch.nn.Parameter(RGB2SH(torch.tensor([[[-1.,2.,.5]]])))
        sh=dc.detach().clone();pc=SimpleNamespace(_albedo_dc=dc,_albedo_dc_stage1=sh)
        before=sh.clone();project_native_albedo(pc)
        torch.testing.assert_close(sh,before,atol=0,rtol=0)
        for margin in [0.,.47,float('inf')]:
            with self.assertRaises(ValueError):project_native_albedo(pc,margin)
        pc._albedo_dc_stage1=pc._albedo_dc
        with self.assertRaises(ValueError):project_native_albedo(pc)
        del pc._albedo_dc_stage1
        pc._albedo_dc.data[0,0,0]=float('nan')
        with self.assertRaises(ValueError):project_native_albedo(pc)
