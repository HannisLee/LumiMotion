"""来源XYZ锚定、真实来源核验、checkpoint绑定及Preset默认兼容性。"""
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
from scripts.loss_stage2 import apply_loss_preset
from scene.pbr_deformation import (create_deformation_anchor, load_deformation_anchor,
                                   deformation_input, validate_deformation_anchor)


class PBRDeformationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        self.xyz = torch.tensor([[.1, .2, .3], [-.4, .5, .6]])
        self.ply = self.model/'point_cloud/iteration_35000/point_cloud.ply'
        self.ply.parent.mkdir(parents=True)
        values = np.empty(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i, k in enumerate(['x', 'y', 'z']):
            values[k] = self.xyz[:, i].numpy()
        PlyData([PlyElement.describe(values, 'vertex')]).write(self.ply)
        self.deform = self.model/'deform/iteration_35000/deform.pth'
        self.deform.parent.mkdir(parents=True)
        self.deform.write_bytes(b'unit-test-source-not-trained-weights')
        source = {'/source/'+str(p.relative_to(self.model)): self.sha(p)
                  for p in [self.ply, self.deform]}
        (self.model/'stage1_source.json').write_text(json.dumps(source))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def create(self):
        return create_deformation_anchor(self.model, self.xyz, 35000)

    def metadata(self, **updates):
        path = self.model/'pbr_deformation_anchor.json'
        values = json.loads(path.read_text()); values.update(updates)
        path.write_text(json.dumps(values))

    def marker(self, **updates):
        values = dict(iteration=55000,
                      deformation_anchor_sha256=self.sha(self.model/'pbr_deformation_anchor.npz'))
        values.update(updates)
        path = self.model/'pbr_directional/iteration_55000/lighting.json'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(values))

    def test_roundtrip_source_and_updated_checkpoint(self):
        anchor, record = self.create()
        self.assertFalse(anchor.requires_grad)
        self.assertNotEqual(anchor.data_ptr(), self.xyz.data_ptr())
        loaded = load_deformation_anchor(self.model, self.xyz, 35000)
        torch.testing.assert_close(loaded, self.xyz, atol=0, rtol=0)
        self.marker()
        loaded = load_deformation_anchor(self.model, self.xyz+.01, 55000)
        torch.testing.assert_close(loaded, self.xyz, atol=0, rtol=0)
        self.assertEqual(record['points'], 2)
        self.assertEqual(set(np.load(self.model/'pbr_deformation_anchor.npz').files), {'xyz'})

    def test_current_move_only_changes_render_position(self):
        anchor, _ = self.create()
        current = self.xyz.clone().requires_grad_()
        old_input = deformation_input(current, anchor).clone()
        with torch.no_grad(): current.add_(.01)
        query = deformation_input(current, anchor)
        torch.testing.assert_close(query, old_input, atol=0, rtol=0)
        deformation = query.sin()
        render_xyz = current+deformation
        torch.testing.assert_close(render_xyz-(self.xyz+old_input.sin()),
                                   torch.full_like(current, .01), atol=1e-7, rtol=0)
        render_xyz.sum().backward()
        self.assertTrue((current.grad == 1).all())
        self.assertFalse(query.requires_grad)
        self.assertFalse(deformation_input(current).requires_grad)

    def test_shared_storage_shape_dtype_rejected(self):
        for invalid in [self.xyz, self.xyz[:1], self.xyz.double()]:
            with self.assertRaises(ValueError): deformation_input(self.xyz, invalid)

    def test_creation_rejects_wrong_source_and_overwrite(self):
        with self.assertRaises(ValueError):
            create_deformation_anchor(self.model, self.xyz+.01, 35000)
        with self.assertRaises(ValueError):
            create_deformation_anchor(self.model, self.xyz, True)
        self.create()
        with self.assertRaises(FileExistsError): self.create()

    def test_modified_local_source_rejected(self):
        self.create(); self.deform.write_bytes(b'tampered')
        with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 35000)

    def test_asset_tampering_rejected_even_with_new_sha(self):
        self.create(); asset = self.model/'pbr_deformation_anchor.npz'
        asset.write_bytes(b'tampered')
        with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 35000)
        for invalid in [self.xyz.numpy()+.01, self.xyz.numpy().astype('f8'),
                        self.xyz.numpy()[:1], np.full((2, 3), np.nan, dtype='f4')]:
            np.savez(asset, xyz=invalid); self.metadata(asset_sha256=self.sha(asset))
            with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 35000)
        np.savez(asset, xyz=self.xyz.numpy(), gt_normal=np.zeros((2, 3)))
        self.metadata(asset_sha256=self.sha(asset))
        with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 35000)

    def test_checkpoint_marker_required_and_bound(self):
        self.create()
        with self.assertRaises(FileNotFoundError): load_deformation_anchor(self.model, self.xyz, 55000)
        for updates in [dict(iteration=50000), dict(deformation_anchor_sha256='bad')]:
            self.marker(**updates)
            with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 55000)
        with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz[:1], 35000)
        with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz.double(), 35000)
        with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 30000)

    def test_metadata_gt_or_source_changes_rejected(self):
        _, original = self.create()
        for updates in [dict(gt_normal_inputs=True), dict(gt_geometry_inputs=None),
                        dict(gt_material_inputs=True), dict(points=3), dict(method='other'),
                        dict(source_snapshots={}), dict(source_files={})]:
            self.metadata(**original); self.metadata(**updates)
            with self.assertRaises(ValueError): load_deformation_anchor(self.model, self.xyz, 35000)

    def test_anchor_preset_and_auto(self):
        parser = ArgumentParser(); Stage2PipelineParams(parser); Stage2OptimizationParams(parser)
        for argv in [['--loss_preset', 'pbr_gt_directional_anchor'],
                     ['--pbr_light_mode', 'gt_directional', '--pbr_deformation_anchor']]:
            args = parser.parse_args(argv); apply_loss_preset(args, argv)
            self.assertEqual(args.loss_preset, 'pbr_gt_directional_anchor')
            self.assertTrue(args.pbr_deformation_anchor and args.pbr_optimize_geometry)
            self.assertTrue(args.pbr_albedo_interior_init and args.pbr_albedo_project_after_step)
            validate_deformation_anchor(args)
        for name in ['auto', 'irgs_baseline', 'pbr_gt_directional',
                     'pbr_gt_directional_geometry', 'pbr_gt_directional_refine']:
            argv = ['--loss_preset', name]; args = parser.parse_args(argv); apply_loss_preset(args, argv)
            self.assertFalse(args.pbr_deformation_anchor)

    def test_incompatible_mode_rejected(self):
        good = dict(pbr_deformation_anchor=True, render_mode='original_ir',
                    pbr_light_mode='gt_directional', pbr_optimize_geometry=True,
                    geometry_source_deformation_mode='dynamic')
        for updates in [dict(render_mode='photometric_lambertian'), dict(pbr_light_mode='envmap'),
                        dict(pbr_optimize_geometry=False), dict(geometry_source_deformation_mode='static_anchor')]:
            args = SimpleNamespace(**dict(good, **updates))
            with self.assertRaises(ValueError): validate_deformation_anchor(args)
