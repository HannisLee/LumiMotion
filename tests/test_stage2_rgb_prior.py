"""RGB弱先验协议、关闭兼容及真实Stage2初始化拒绝条件。"""
import json
import tempfile
import unittest
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual
from scene.stage2_rgb_prior import load_rgb_normal_prior, validate_stage2_initialization
from scripts.loss_stage2 import apply_loss_preset


class RGBPriorTest(unittest.TestCase):
    def test_prior_protocol_and_source_rejections(self):
        meta = dict(format_version=1, method='training_rgb_refined_area_ps', gt_normal_inputs=False,
            gt_depth_inputs=False, gt_material_inputs=False, original_test_used=False,
            optimization_keys=['0002'], source_files={'source':'sha'}, stage2_checkpoint_sha256='initsha')
        normal = np.tile(np.array([0.,0.,1.],dtype=np.float32),(8,1))
        valid = np.array([True]*4+[False]*4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'prior.npz'
            def write(m=meta, n=normal, v=valid):
                np.savez(p, normal=n, valid=v, metadata=json.dumps(m))
            write()
            n, v, _ = load_rgb_normal_prior(p, 8, {'source':'sha'}, 'initsha', ['0002'])
            torch.testing.assert_close(n,torch.from_numpy(normal),rtol=0,atol=0)
            self.assertTrue(torch.equal(v,torch.from_numpy(valid)))
            for field, value in [('gt_normal_inputs',True),('gt_depth_inputs',True),
                ('gt_material_inputs',True),('original_test_used',True),('source_files',{}),
                ('optimization_keys',['0001']),('stage2_checkpoint_sha256','other')]:
                write(dict(meta,**{field:value}))
                with self.assertRaises(ValueError):
                    load_rgb_normal_prior(p,8,{'source':'sha'},'initsha',['0002'])
            for n,v in [(normal.astype(np.float64),valid),(normal*2,valid),(normal,valid.astype(np.uint8)),(normal,np.zeros(8,dtype=bool))]:
                write(n=n,v=v)
                with self.assertRaises(ValueError):
                    load_rgb_normal_prior(p,8,{'source':'sha'},'initsha',['0002'])

    def test_closed_default_and_explicit_disable(self):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser)
        args=parser.parse_args([]);xyz=torch.randn(8,3)
        g=GeometryResidual(xyz,args)
        self.assertFalse(g.rgb_normal_enabled)
        self.assertFalse(any(k.startswith('rgb_normal_') for k in g.state_dict()))
        args.render_mode='photometric_lambertian';args.photometric_light_mode='gt_directional'
        args.loss_preset='lambertian_geometry_area_rgb_prior';args.geometry_rgb_normal_prior_path='unused.npz'
        apply_loss_preset(args,[])
        self.assertEqual(args.lambda_geometry_rgb_normal_prior,.2)
        new=GeometryResidual(xyz,args)
        self.assertEqual(new.rgb_normal_targets.shape,(32,3))
        args.lambda_geometry_rgb_normal_prior=0
        apply_loss_preset(args,['--lambda_geometry_rgb_normal_prior','0'])
        self.assertFalse(GeometryResidual(xyz,args).rgb_normal_enabled)

    def test_initialization_requires_real_matching_checkpoint(self):
        args=Namespace(geometry_initial_stage2_iteration=43000,load_iter=35000)
        state=dict(format_version=2,iteration=43000,source_iteration=35000,
            source_files={'source':'sha'},frozen_fingerprint='fingerprint',config={},
            geometry={'translation':torch.zeros(8,3)},material={},light={})
        validate_stage2_initialization(state,args,{'source':'sha'},'fingerprint')
        for field,value in [('diagnostic','RGB proposal'),('source_iteration',34000),
            ('iteration',42000),('source_files',{}),('frozen_fingerprint','other')]:
            with self.assertRaises(ValueError):
                validate_stage2_initialization(dict(state,**{field:value}),args,{'source':'sha'},'fingerprint')
        with self.assertRaises(ValueError):
            validate_stage2_initialization(dict(state,config={'geometry_subdivision_side':4}),args,{'source':'sha'},'fingerprint')

    def test_prior_buffers_reload_without_asset(self):
        parser=ArgumentParser();Stage2OptimizationParams(parser);Stage2PipelineParams(parser)
        args=parser.parse_args([]);args.lambda_geometry_rgb_normal_prior=.2
        args.geometry_rgb_normal_prior_path='missing_asset.npz'
        xyz=torch.randn(8,3);g=GeometryResidual(xyz,args)
        g.rgb_normal_targets[:,2]=1.;g.rgb_normal_valid[:4]=True
        clone=GeometryResidual(xyz,args);clone.load_state_dict(g.state_dict())
        for k,v in g.state_dict().items():
            torch.testing.assert_close(clone.state_dict()[k],v,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
