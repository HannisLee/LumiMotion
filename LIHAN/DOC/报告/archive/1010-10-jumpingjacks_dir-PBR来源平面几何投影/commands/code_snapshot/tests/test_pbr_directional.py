"""原生方向光PBR：多SUN、checkpoint、delta积分与材质梯度。"""
import hashlib
import json
import math
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import torch
from arguments.stage2 import Stage2PipelineParams, Stage2OptimizationParams
from gaussian_renderer.render_ir import rendering_equation_directional
from scene.pbr_directional import GTDirectionalLights, validate_directional_checkpoint
from scripts.loss_stage2 import apply_loss_preset, build_loss_preset


def light(direction=(0., 0., 1.), intensity=1., color=(1., 1., 1.)):
    return dict(type='directional', source_type='SUN', light_dir_world=direction,
                intensity=intensity, light_rgb=color)


class DirectionalPBRTest(unittest.TestCase):
    def test_metadata_multiple_suns_and_exact_frame_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lights.json'
            path.write_text(json.dumps({'frames': {'0010': {'lights': [light(intensity=2.),
                light((1., 0., 0.), 3., (0., 1., 0.))]}}}))
            table = GTDirectionalLights(path)
            camera = SimpleNamespace(image_name_train_light='images/0010.png')
            dirs, energy = table.for_camera(camera, 2., 'cpu')
            torch.testing.assert_close(dirs, torch.tensor([[0., 0., 1.], [1., 0., 0.]]))
            torch.testing.assert_close(energy, torch.tensor([[4., 4., 4.], [0., 6., 0.]]))
            with self.assertRaises(ValueError):
                table.for_camera(SimpleNamespace(image_name_train_light='0020'), device='cpu')

    def test_invalid_direction_and_point_lights_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lights.json'
            for item in [light((0., 0., 0.)), light((0., 0., 2.)), light(intensity=-1.),
                         dict(type='point', light_pos_world=[0., 0., 1.], intensity=1.)]:
                path.write_text(json.dumps({'0010': item}))
                with self.assertRaises(ValueError):
                    GTDirectionalLights(path)

    def test_delta_diffuse_and_material_gradients(self):
        pipe = SimpleNamespace(wo_indirect=True, wo_indirect_relight=False, wo_specular=False,
                               detach_indirect=False, light_t_min=.1)
        base = torch.tensor([[.2, .4, .6]], requires_grad=True)
        rough = torch.tensor([[.4]], requires_grad=True)
        normal = torch.tensor([[0., 0., 1.]])
        # Unit face-on irradiance pi -> exact diffuse base, no Monte Carlo area factor.
        result = rendering_equation_directional(base, rough, normal, normal, normal, None, pipe,
            (normal, torch.full((1, 3), math.pi)), skip_tracer=True)
        torch.testing.assert_close(result['diffuse'], base)
        self.assertTrue((result['specular'] > 0).all())
        (result['diffuse'] + result['specular']).sum().backward()
        self.assertTrue(torch.isfinite(base.grad).all())
        self.assertGreater(float(rough.grad.abs().sum()), 0.)
        result = rendering_equation_directional(base, rough, normal, normal, normal, None, pipe,
            (-normal, torch.ones((1, 3))), skip_tracer=True)
        torch.testing.assert_close(result['diffuse'] + result['specular'], torch.zeros_like(base))

    def test_bvh_visibility_scales_diffuse(self):
        pipe = SimpleNamespace(wo_indirect=True, wo_indirect_relight=False, wo_specular=True,
                               detach_indirect=False, light_t_min=.1)
        n = torch.tensor([[0., 0., 1.]])
        pc = SimpleNamespace(trace=lambda *a, **k: {'alpha': torch.tensor([[.75]])})
        base = torch.ones((1, 3))*.4
        result = rendering_equation_directional(base, torch.ones((1, 1))*.5, n, n, n, pc, pipe,
            (n, torch.full((1, 3), math.pi)))
        torch.testing.assert_close(result['diffuse'], base*.25)
        torch.testing.assert_close(result['visibility'], torch.tensor([[.25]]))

    def test_native_preset_and_old_envmap_defaults(self):
        parser = ArgumentParser(); Stage2PipelineParams(parser); Stage2OptimizationParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.pbr_light_mode, 'envmap')
        apply_loss_preset(args, [])
        self.assertEqual(args.loss_preset, 'auto')
        args.loss_preset = 'pbr_gt_directional'
        apply_loss_preset(args, [])
        self.assertEqual(args.pbr_light_mode, 'gt_directional')
        self.assertEqual(args.lambda_light, 0.)
        self.assertEqual(args.lambda_light_smooth, 0.)
        build_loss_preset(args, 'original_ir')

    def test_checkpoint_rejects_stage1_and_changed_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'lights.json'; source.write_text('{}')
            marker = root/'pbr_directional/iteration_55000/lighting.json'
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps(dict(format_version=1, iteration=55000,
                lights_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), intensity_scale=1.)))
            config = dict(load_iter=35000, pbr_gt_lights_path=str(source), pbr_light_intensity_scale=1.)
            validate_directional_checkpoint(root, 55000, config)
            source.write_text('{"changed":true}')
            with self.assertRaises(ValueError): validate_directional_checkpoint(root, 55000, config)

    def test_fitted_scale_requires_checkpoint_value_and_keeps_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'lights.json'; source.write_text(json.dumps({'0010':light()}))
            scale=torch.tensor(2., requires_grad=True)
            _, energy=GTDirectionalLights(source).for_camera(SimpleNamespace(image_name_train_light='0010'),scale,'cpu')
            energy.sum().backward();torch.testing.assert_close(scale.grad,torch.tensor(3.))
            marker=root/'pbr_directional/iteration_55000/lighting.json';marker.parent.mkdir(parents=True)
            record=dict(format_version=1, iteration=55000,lights_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),intensity_scale=1.)
            marker.write_text(json.dumps(record));config=dict(load_iter=35000,pbr_gt_lights_path=str(source),pbr_light_intensity_scale=1.,pbr_learn_light_intensity=True)
            with self.assertRaises(ValueError):validate_directional_checkpoint(root,55000,config)
            record.update(effective_intensity_scale=2.,intensity_fitted_from_train_rgb=True)
            marker.write_text(json.dumps(record));validate_directional_checkpoint(root,55000,config)


if __name__ == '__main__':
    unittest.main()
