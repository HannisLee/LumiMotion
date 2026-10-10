"""原生PBR明确选择CUDA来源；CPU测试不构建/调用GPU。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from argparse import ArgumentParser
from unittest.mock import patch
from arguments.stage2 import Stage2PipelineParams,Stage2OptimizationParams
from scene import pbr_rasterizer as backend
from scene.pbr_directional import validate_directional_checkpoint
from scripts.loss_stage2 import apply_loss_preset,resolve_preset


class NativeRasterizerTest(unittest.TestCase):
    def args(self):
        parser=ArgumentParser();Stage2PipelineParams(parser);Stage2OptimizationParams(parser)
        return parser.parse_args([])

    def test_old_pipeline_defaults(self):
        args=self.args();self.assertEqual(args.pbr_rasterizer_backend,'installed')
        self.assertEqual(backend.validate_backend(args),'installed')
        with patch.object(backend,'native_module',side_effect=AssertionError('unexpected build')):
            legacy=backend.rasterizer_classes(args)
        import diff_surfel_rasterization
        self.assertIs(legacy[1],diff_surfel_rasterization.GaussianRasterizer)

    def test_new_presets_and_auto_selection(self):
        for name,plane in [('pbr_gt_directional_native',False),('pbr_gt_directional_plane_native',True)]:
            args=self.args();args.loss_preset=name;apply_loss_preset(args,[])
            self.assertEqual(args.pbr_rasterizer_backend,'lumimotion')
            self.assertEqual(args.pbr_plane_projection,plane)
            self.assertEqual(args.pbr_light_mode,'gt_directional')
            args.loss_preset='auto';self.assertEqual(resolve_preset(args,'original_ir'),name)
        args=self.args();args.loss_preset='pbr_gt_directional_plane';apply_loss_preset(args,[])
        self.assertEqual(args.pbr_rasterizer_backend,'installed')

    def test_explicit_backend_override(self):
        args=self.args();args.loss_preset='pbr_gt_directional_plane_native'
        args.pbr_rasterizer_backend='installed';apply_loss_preset(args,['--pbr_rasterizer_backend=installed'])
        self.assertEqual(args.pbr_rasterizer_backend,'installed')

    def test_invalid_backend_mode_and_covariance(self):
        for args in [SimpleNamespace(pbr_rasterizer_backend='unknown'),
                     SimpleNamespace(pbr_rasterizer_backend='lumimotion',render_mode='photometric_lambertian'),
                     SimpleNamespace(pbr_rasterizer_backend='lumimotion',compute_cov3D_python=True)]:
            with self.assertRaises(ValueError):backend.validate_backend(args)

    def test_native_dispatch_does_not_replace_installed(self):
        fake=SimpleNamespace(GaussianRasterizationSettings=object(),GaussianRasterizer=object())
        with patch.object(backend,'native_module',return_value=fake):
            result=backend.rasterizer_classes(SimpleNamespace(pbr_rasterizer_backend='lumimotion'))
            self.assertIs(result[0],fake.GaussianRasterizationSettings)
            self.assertIs(result[1],fake.GaussianRasterizer)

    def test_source_manifest_uses_repository_cuda(self):
        record=backend.source_manifest();self.assertEqual(record['method'],backend.METHOD)
        self.assertFalse(record['gt_geometry_normal_material_inputs'])
        self.assertEqual(len(record['source_sha256']),64)
        forward='submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization/cuda_rasterizer/forward.cu'
        self.assertIn(forward,record['source_files'])
        self.assertTrue((backend.ROOT/record['glm_include']/'glm/glm.hpp').is_file())

    def test_asset_create_validate_tamper_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(pbr_rasterizer_backend='lumimotion');record=backend.source_manifest()
            fake=SimpleNamespace(provenance=dict(record,extension_sha256='diagnostic'))
            with patch.object(backend,'native_module',return_value=fake):
                backend.prepare_backend(folder,args)
                with self.assertRaises(FileExistsError):backend.prepare_backend(folder,args)
            backend.validate_backend_model(folder,args)
            changed=copy.deepcopy(record);changed['normal_orientation']='view_z_sign'
            (Path(folder)/backend.ASSET).write_text(json.dumps(changed))
            with self.assertRaises(ValueError):backend.validate_backend_model(folder,args)

    def test_checkpoint_requires_native_binding(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'lights.json';source.write_text('{}')
            record=backend.source_manifest();(root/backend.ASSET).write_text(json.dumps(record))
            marker=root/'pbr_directional/iteration_55000/lighting.json';marker.parent.mkdir(parents=True)
            light=dict(format_version=1,iteration=55000,lights_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),intensity_scale=1.)
            config=dict(load_iter=35000,pbr_gt_lights_path=str(source),pbr_light_intensity_scale=1.,pbr_rasterizer_backend='lumimotion')
            marker.write_text(json.dumps(light))
            with self.assertRaises(ValueError):validate_directional_checkpoint(root,55000,config)
            light['rasterizer_source_sha256']=record['source_sha256'];marker.write_text(json.dumps(light))
            self.assertEqual(len(validate_directional_checkpoint(root,55000,config)),3)


if __name__=='__main__':unittest.main()
