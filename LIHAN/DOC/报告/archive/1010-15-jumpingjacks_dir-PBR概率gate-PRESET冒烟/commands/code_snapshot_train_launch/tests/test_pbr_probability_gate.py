"""概率gate的原生前向、饱和梯度、PLY保存与协议隔离。"""
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
    validate_feature_gate_model, native_probability, encode_probability, METADATA)


class ProbabilityGateTest(unittest.TestCase):
    def args(self):
        parser = ArgumentParser(); Stage2PipelineParams(parser); Stage2OptimizationParams(parser)
        args = parser.parse_args([]); args.load_iter = 35000; args.iterations = 55000
        args.loss_preset = 'pbr_gt_directional_plane_native_direct_probability_gate'
        apply_loss_preset(args, [])
        return args

    def fixture(self, path):
        model = Path(path); ply = model/'point_cloud/iteration_35000/point_cloud.ply'
        pth = model/'deform/iteration_35000/deform.pth'
        ply.parent.mkdir(parents=True); pth.parent.mkdir(parents=True)
        vertex = np.zeros(5, dtype=[(k, '<f4') for k in ['x','y','z','fea_0']])
        vertex['fea_0'] = [-20., -1., 0., 1., 20.]
        PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(ply))
        deform = SimpleNamespace(deform=torch.nn.Linear(3, 4))
        torch.save(deform.deform.state_dict(), pth)
        source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ply,pth]}
        (model/'stage1_source.json').write_text(json.dumps(source))
        pc = SimpleNamespace(feature=torch.nn.Parameter(torch.tensor(vertex['fea_0'].copy())[:,None]))
        pc.get_binary_feature = lambda: native_probability(pc.feature)
        return model, pc, deform

    def marker(self, model, gate):
        marker = model/'pbr_directional/iteration_55000/lighting.json'
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps(dict(iteration=55000,feature_gate_metadata_sha256=gate.metadata_sha)))

    def test_preset_and_auto(self):
        args = self.args(); self.assertEqual(args.pbr_feature_gate_parameterization, 'probability')
        self.assertTrue(args.pbr_optimize_feature_gate)
        args.loss_preset = 'auto'
        self.assertEqual(resolve_preset(args,'original_ir'), 'pbr_gt_directional_plane_native_direct_probability_gate')

    def test_initial_exact_source_and_frozen_nn(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path); raw=pc.feature.detach().clone()
            gate=NativePBRFeatureGate(model,pc,deform,self.args(),35000)
            self.assertTrue(torch.equal(raw,pc.feature))
            self.assertTrue(torch.equal(gate.output(pc),pc.get_binary_feature()))
            self.assertEqual(float(gate.loss(pc)),0.)
            self.assertFalse(any(p.requires_grad for p in deform.deform.parameters()))
            self.assertIs(gate.optimizer_parameters(pc)[0],gate.probability)
            gate.project(pc);self.assertTrue(torch.equal(raw,pc.feature))
            validate_feature_gate_model(model,pc,deform,self.args(),35000)

    def test_saturated_identity_gradient_and_native_forward(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path)
            gate=NativePBRFeatureGate(model,pc,deform,self.args(),35000)
            pc.feature.requires_grad_(False)
            hidden=deform.deform(torch.ones(5,3)).sum(-1,keepdim=True)
            output=gate.output(pc);self.assertTrue(torch.equal(output,pc.get_binary_feature()))
            (output*hidden).sum().backward()
            torch.testing.assert_close(gate.probability.grad,hidden,atol=0,rtol=0)
            self.assertTrue((gate.probability.grad[[0,-1]].abs()>0).all())
            self.assertIsNone(pc.feature.grad)
            self.assertTrue(all(p.grad is None for p in deform.deform.parameters()))

    def test_finite_encoding_and_roundtrip(self):
        source=torch.linspace(-20,20,10001)[:,None];source_probability=native_probability(source)
        probability=torch.linspace(0,1,10001)[:,None]
        raw=encode_probability(probability,source,source_probability)
        self.assertTrue(torch.isfinite(raw).all())
        torch.testing.assert_close(native_probability(raw),probability,atol=2e-7,rtol=0)
        torch.testing.assert_close(encode_probability(source_probability,source,source_probability),source,atol=0,rtol=0)

    def test_adam_changes_native_ply_and_saved_validation(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path);args=self.args()
            gate=NativePBRFeatureGate(model,pc,deform,args,35000)
            pc.feature.requires_grad_(False)
            optimizer=torch.optim.Adam(gate.optimizer_parameters(pc),lr=.001)
            desired=torch.full_like(gate.probability,.25)
            (gate.output(pc)-desired).square().sum().backward();optimizer.step();gate.project(pc)
            self.assertTrue(torch.equal(gate.probability.detach(),pc.get_binary_feature()))
            self.assertGreater(float((gate.probability-gate.source_probability).abs().max()),.0009)
            self.marker(model,gate);validate_feature_gate_model(model,pc,deform,args,55000)
            self.assertTrue(torch.isfinite(pc.feature).all())

    def test_probability_projection_bound_and_endpoint(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path);args=self.args();args.pbr_feature_gate_max_probability_delta=.1
            gate=NativePBRFeatureGate(model,pc,deform,args,35000)
            with torch.no_grad():gate.probability.copy_(torch.tensor([[2.],[-1.],[3.],[4.],[-2.]]))
            gate.project(pc)
            self.assertLessEqual(float((gate.probability-gate.source_probability).abs().max()),.100001)
            self.marker(model,gate);validate_feature_gate_model(model,pc,deform,args,55000)
            with torch.no_grad():pc.feature[0]=20.
            with self.assertRaises(ValueError):validate_feature_gate_model(model,pc,deform,args,55000)

    def test_invalid_mode_bound_and_nonfinite_rejected(self):
        for key,value in [('pbr_feature_gate_parameterization','wrong'),
            ('pbr_feature_gate_max_probability_delta',0.),('pbr_feature_gate_max_probability_delta',1.1),
            ('pbr_feature_gate_max_probability_delta',float('nan'))]:
            args=self.args();setattr(args,key,value)
            with self.assertRaises(ValueError):validate_feature_gate(args)
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path);gate=NativePBRFeatureGate(model,pc,deform,self.args(),35000)
            with torch.no_grad():gate.probability[0]=float('nan')
            with self.assertRaises(ValueError):gate.project(pc)

    def test_metadata_protocol_and_config_tampering(self):
        for key,value in [('format_version',1),('method','wrong'),('max_probability_delta',.5),
            ('forward','direct_unsaved_probability'),('gradient','wrong'),('gt_normal_inputs',True),
            ('optimizer_state_saved',True)]:
            with tempfile.TemporaryDirectory() as path:
                model,pc,deform=self.fixture(path);args=self.args();NativePBRFeatureGate(model,pc,deform,args,35000)
                asset=model/METADATA;record=json.loads(asset.read_text());record[key]=value;asset.write_text(json.dumps(record))
                with self.assertRaises(ValueError):validate_feature_gate_model(model,pc,deform,args,35000)

    def test_legacy_metadata_and_optimizer_unchanged(self):
        with tempfile.TemporaryDirectory() as path:
            model,pc,deform=self.fixture(path);args=self.args();args.pbr_feature_gate_parameterization='raw_logit'
            gate=NativePBRFeatureGate(model,pc,deform,args,35000)
            self.assertIs(gate.optimizer_parameters(pc)[0],pc.feature)
            self.assertTrue(torch.equal(gate.output(pc),pc.get_binary_feature()))
            self.assertEqual(json.loads((model/METADATA).read_text())['format_version'],1)
            validate_feature_gate_model(model,pc,deform,args,35000)


if __name__=='__main__':unittest.main()
