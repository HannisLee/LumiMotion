"""几何评测真实入口必须共享Anchor/Plane运行时；评分口径不变。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from scripts.eval_geometry_gt import GeometryRun
from utils.geometry_eval_utils import file_sha256


class GeometrySourceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        for rel in ['point_cloud/iteration_55000/point_cloud.ply', 'deform/iteration_55000/deform.pth',
                    'point_cloud/iteration_35000/point_cloud.ply', 'deform/iteration_35000/deform.pth',
                    'pbr_directional/iteration_55000/lighting.json']:
            p = self.model/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fixture')
        source = self.model/'deform/iteration_35000/deform.pth'
        self.record = dict(source_files={str(source): file_sha256(source)},
                           source_snapshots={'deform/iteration_35000/deform.pth': file_sha256(source)})
        for prefix in ['pbr_deformation_anchor', 'pbr_plane_projection']:
            (self.model/(prefix+'.json')).write_text(json.dumps(self.record))
            (self.model/(prefix+'.npz')).write_bytes(b'asset')
        self.context = SimpleNamespace(pc=object(), deform=object(), source_files={},
                                       deformation=MagicMock(return_value={'d_xyz': 'projected'}))

    def config(self, **kwargs):
        config = dict(source_path='/source', render_mode='original_ir', **kwargs)
        (self.model/'stage2_config.json').write_text(json.dumps(config))

    def test_anchor_initialization_uses_shared_runtime_and_hashes_assets(self):
        self.config(pbr_deformation_anchor=True)
        with patch('scripts.stage2_runtime.FrozenScene', return_value=self.context) as loader:
            run = GeometryRun(self.model, 55000)
        self.assertIs(run.pc, self.context.pc); self.assertIs(run.deform, self.context.deform)
        self.assertEqual(loader.call_args[0][0].source_path, '/source')
        for name in ['pbr_deformation_anchor.json', 'pbr_deformation_anchor.npz',
                     'pbr_directional/iteration_55000/lighting.json']:
            self.assertEqual(run.files[str(self.model/name)], file_sha256(self.model/name))

    def test_plane_runtime_records_both_assets_and_source_snapshots(self):
        self.config(pbr_deformation_anchor=True, pbr_plane_projection=True)
        with patch('scripts.stage2_runtime.FrozenScene', return_value=self.context):
            run = GeometryRun(self.model, 55000, source_override='/override')
        self.assertIs(run.context, self.context)
        for name in ['pbr_plane_projection.json', 'pbr_plane_projection.npz',
                     'deform/iteration_35000/deform.pth']:
            self.assertIn(str(self.model/name), run.files)

    def test_runtime_deformation_receives_actual_camera_time(self):
        run = GeometryRun.__new__(GeometryRun); run.context = self.context
        camera = SimpleNamespace(fid=torch.tensor([.7]))
        self.assertEqual(run.deformation(camera), {'d_xyz': 'projected'})
        self.context.deformation.assert_called_once_with(camera)
        camera.fid = torch.tensor([.8]); run.deformation(camera)
        self.assertEqual(float(self.context.deformation.call_args[0][0].fid[0]), float(camera.fid[0]))

    def test_legacy_deformation_preserves_current_xyz_query_and_iteration(self):
        run = GeometryRun.__new__(GeometryRun); run.context = None; run.iteration = 55000
        xyz = torch.tensor([[1., 2., 3.]], requires_grad=True)
        run.pc = SimpleNamespace(get_xyz=xyz, get_binary_feature=lambda: torch.ones(1,1))
        run.deform = SimpleNamespace(step=MagicMock(return_value={'legacy': True}))
        camera = SimpleNamespace(fid=torch.tensor([.4]), camera_center=torch.zeros(3))
        self.assertEqual(run.deformation(camera), {'legacy': True})
        args, kwargs = run.deform.step.call_args
        self.assertTrue(torch.equal(args[0], xyz)); self.assertFalse(args[0].requires_grad)
        self.assertEqual(kwargs['iteration'], 55000); self.assertFalse(kwargs['is_training'])

    def test_non_native_mode_rejects_pbr_runtime_flags(self):
        self.config(pbr_deformation_anchor=True)
        config = json.loads((self.model/'stage2_config.json').read_text()); config['render_mode'] = 'original_sh'
        (self.model/'stage2_config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, '原生PBR'):
            GeometryRun(self.model, 55000)

    def test_loader_rejection_propagates_instead_of_silent_legacy_fallback(self):
        self.config(pbr_deformation_anchor=True, pbr_plane_projection=True)
        with patch('scripts.stage2_runtime.FrozenScene', side_effect=ValueError('invalid asset')):
            with self.assertRaisesRegex(ValueError, 'invalid asset'):
                GeometryRun(self.model, 55000)


if __name__ == '__main__': unittest.main()
