"""原生gate默认隔离、真实来源绑定、冻结NN和PLY logit信赖域。"""
import hashlib
import json
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scripts.loss_stage2 import apply_loss_preset, resolve_preset
from scene.pbr_feature_gate import (NativePBRFeatureGate, validate_feature_gate,
    validate_feature_gate_model, gate_anchor, project_gate, METADATA)


class FeatureGateTest(unittest.TestCase):
    def args(self):
        p = ArgumentParser(); Stage2PipelineParams(p); Stage2OptimizationParams(p)
        a = p.parse_args([]); a.load_iter = 35000; a.iterations = 55000
        a.loss_preset = 'pbr_gt_directional_plane_native_direct_gate'
        apply_loss_preset(a, [])
        return a

    def fixture(self, path):
        model = Path(path); ply = model/'point_cloud/iteration_35000/point_cloud.ply'
        pth = model/'deform/iteration_35000/deform.pth'
        ply.parent.mkdir(parents=True); pth.parent.mkdir(parents=True)
        v = np.zeros(3, dtype=[(k, '<f4') for k in ['x', 'y', 'z', 'fea_0']])
        v['fea_0'] = [-1., 0., 1.]
        PlyData([PlyElement.describe(v, 'vertex')]).write(str(ply))
        deform = SimpleNamespace(deform=torch.nn.Linear(3, 4))
        torch.save(deform.deform.state_dict(), pth)
        source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ply, pth]}
        (model/'stage1_source.json').write_text(json.dumps(source))
        pc = SimpleNamespace(feature=torch.nn.Parameter(torch.tensor([[-1.], [0.], [1.]])))
        return model, pc, deform

    def test_default_presets_and_auto(self):
        a = self.args()
        self.assertTrue(a.pbr_optimize_feature_gate)
        a.loss_preset = 'auto'
        self.assertEqual(resolve_preset(a, 'original_ir'), 'pbr_gt_directional_plane_native_direct_gate')
        for name in ['pbr_gt_directional', 'pbr_gt_directional_plane_native_direct', 'irgs_baseline']:
            p=ArgumentParser(); Stage2PipelineParams(p); Stage2OptimizationParams(p)
            a=p.parse_args([]); a.loss_preset=name; apply_loss_preset(a, [])
            self.assertFalse(a.pbr_optimize_feature_gate)

    def test_invalid_configuration(self):
        for key, value in [('render_mode','photometric_lambertian'), ('pbr_light_mode','envmap'),
            ('pbr_deformation_anchor',False), ('pbr_optimize_geometry',False), ('no_binary_separation',True),
            ('hyper_dim',2), ('deform_type','static'), ('pbr_feature_gate_lr',0.),
            ('pbr_feature_gate_max_logit_delta',float('nan')), ('lambda_pbr_feature_gate_anchor',-1.)]:
            a=self.args();setattr(a,key,value)
            with self.assertRaises(ValueError):validate_feature_gate(a)

    def test_projection_and_anchor_gradient(self):
        source=torch.tensor([[-2.], [1.]])
        feature=torch.nn.Parameter(torch.tensor([[7.], [-8.]]))
        gate_anchor(feature,source,4.).backward()
        self.assertTrue(torch.isfinite(feature.grad).all());self.assertGreater(float(feature.grad.norm()),0)
        project_gate(feature,source,4.)
        torch.testing.assert_close(feature,torch.tensor([[2.],[-3.]]),atol=0,rtol=0)

    def test_source_initialization_and_frozen_nn_gate_gradient(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path);a=self.args()
            before={k:v.clone() for k,v in deform.deform.state_dict().items()}
            gate=NativePBRFeatureGate(model,pc,deform,a,35000)
            self.assertFalse(any(p.requires_grad for p in deform.deform.parameters()))
            self.assertEqual(float(gate.loss(pc)),0.)
            hidden=deform.deform(torch.ones(3,3))
            self.assertFalse(hidden.requires_grad)
            (hidden*torch.sigmoid(pc.feature*2)).sum().backward()
            self.assertGreater(float(pc.feature.grad.norm()),0)
            self.assertTrue(all(p.grad is None for p in deform.deform.parameters()))
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in deform.deform.state_dict().items()))
            validate_feature_gate_model(model,pc,deform,a,35000)
            with self.assertRaises(FileExistsError):NativePBRFeatureGate(model,pc,deform,a,35000)

    def test_source_changed_feature_or_nn_rejected(self):
        for change in ['feature','nn']:
            with tempfile.TemporaryDirectory() as path:
                model,pc,deform=self.fixture(path)
                with torch.no_grad():
                    if change=='feature':pc.feature.add_(.01)
                    else:deform.deform.weight.add_(.01)
                with self.assertRaises(ValueError):NativePBRFeatureGate(model,pc,deform,self.args(),35000)

    def test_checkpoint_marker_and_trust_bound(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path);a=self.args()
            gate=NativePBRFeatureGate(model,pc,deform,a,35000)
            marker=model/'pbr_directional/iteration_55000/lighting.json';marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps(dict(iteration=55000,feature_gate_metadata_sha256=gate.metadata_sha)))
            with torch.no_grad():pc.feature.add_(.4)
            validate_feature_gate_model(model,pc,deform,a,55000)
            with torch.no_grad():pc.feature.add_(5.)
            with self.assertRaises(ValueError):validate_feature_gate_model(model,pc,deform,a,55000)
            gate.project(pc);validate_feature_gate_model(model,pc,deform,a,55000)
            marker.write_text(json.dumps(dict(iteration=55000,feature_gate_metadata_sha256='wrong')))
            with self.assertRaises(ValueError):validate_feature_gate_model(model,pc,deform,a,55000)

    def test_provenance_and_config_tampering(self):
        for change in ['gt', 'config', 'source']:
            with tempfile.TemporaryDirectory() as path:
                model,pc,deform=self.fixture(path);a=self.args()
                NativePBRFeatureGate(model,pc,deform,a,35000)
                if change=='gt':
                    p=model/METADATA;r=json.loads(p.read_text());r['gt_normal_inputs']=True;p.write_text(json.dumps(r))
                elif change=='config':a.pbr_feature_gate_lr=.1
                else:(model/'point_cloud/iteration_35000/point_cloud.ply').write_bytes(b'bad')
                with self.assertRaises(ValueError):validate_feature_gate_model(model,pc,deform,a,35000)


if __name__=='__main__': unittest.main()
