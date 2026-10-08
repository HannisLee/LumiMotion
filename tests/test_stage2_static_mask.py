"""时间稳定图掩码的来源匹配、默认兼容和独立检查点状态。"""
import json
import tempfile
import unittest
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from arguments.stage2 import Stage2OptimizationParams, Stage2PipelineParams
from scene.stage2_geometry import GeometryResidual, load_coherence_source_mask
from scripts.loss_stage2 import apply_loss_preset


class StaticMaskTest(unittest.TestCase):
    def args(self):
        parser = ArgumentParser()
        Stage2OptimizationParams(parser)
        Stage2PipelineParams(parser)
        return parser.parse_args([])

    def test_asset_requires_exact_source_split_and_boolean_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'mask.npz'
            files = {str(root/'point_cloud.ply'): '123', str(root/'deform.pth'): '456'}
            metadata = dict(format_version=1, method='frozen_stage1_training_time_variation',
                gt_normal_inputs=False, source=str(root), source_iteration=35000,
                source_files=files, optimization_keys=['0002', '0003'])
            mask = np.array([True, False, True])

            def save(m=mask, meta=metadata):
                np.savez(path, stable_mask=m, metadata=np.array(json.dumps(meta)))

            def load():
                return load_coherence_source_mask(path, root, 35000, files, ['0002', '0003'], 3)

            save()
            torch.testing.assert_close(load()[0], torch.tensor([True, False, True]))
            for change in [dict(source_iteration=34000), dict(source_files={}),
                           dict(optimization_keys=['0001', '0002']), dict(gt_normal_inputs=True),
                           dict(source=str(root/'other')), dict(method='gt_normal_mask')]:
                save(meta={**metadata, **change})
                with self.assertRaises(ValueError):
                    load()
            for bad in [mask.astype(np.int32), mask[:, None], mask[:2]]:
                save(m=bad)
                with self.assertRaises(ValueError):
                    load()

    def test_preset_requires_mask_only_when_coherence_enabled(self):
        args = self.args()
        xyz = torch.randn(10, 3)
        self.assertEqual(args.geometry_coherence_source_mask, '')
        self.assertNotIn('coherence_source_static', GeometryResidual(xyz, args).state_dict())
        args.render_mode = 'photometric_lambertian'
        args.photometric_light_mode = 'gt_directional'
        args.loss_preset = 'lambertian_geometry_area_static_coherent'
        apply_loss_preset(args, [])
        with self.assertRaises(ValueError):
            GeometryResidual(xyz, args)
        args.lambda_geometry_coherence = 0.
        apply_loss_preset(args, ['--lambda_geometry_coherence', '0'])
        self.assertFalse(GeometryResidual(xyz, args).coherence_enabled)

    def test_filtered_both_endpoints_and_checkpoint_without_asset(self):
        args = self.args()
        args.lambda_geometry_coherence = .2
        args.geometry_coherence_source_mask = '/missing/asset.npz'
        xyz = torch.stack((torch.arange(16).float()*.01, torch.zeros(16), torch.zeros(16)), -1)
        normals = torch.tensor([0., 0., 1.]).expand_as(xyz)
        geo = GeometryResidual(xyz, args)
        stable = torch.ones(16, dtype=torch.bool)
        stable[5:8] = False
        geo.coherence_source_static.copy_(stable)
        geo.initialize_coherence(xyz, normals, torch.full((16, 2), .1), torch.ones(16, 1), static=stable)
        valid = geo.coherence_weights > 0
        self.assertTrue(valid.any())
        self.assertFalse(valid[~stable].any())
        self.assertTrue(stable[geo.coherence_neighbors[valid]].all())
        clone = GeometryResidual(xyz, args)
        clone.load_state_dict(geo.state_dict())
        for name, value in geo.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(geo.coherence_loss(xyz, normals), clone.coherence_loss(xyz, normals))


if __name__ == '__main__':
    unittest.main()
