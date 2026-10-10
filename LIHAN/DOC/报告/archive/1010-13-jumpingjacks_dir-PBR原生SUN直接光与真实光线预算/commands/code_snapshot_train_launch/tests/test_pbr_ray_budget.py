"""SUN预算与Direct Preset的CPU回归；不运行CUDA。"""
import unittest
from argparse import ArgumentParser
from types import SimpleNamespace
import numpy as np
from arguments.stage2 import Stage2PipelineParams, Stage2OptimizationParams
from scene.pbr_ray_budget import photo_pixel_budget, validate_actual_ray_budget
from scripts.loss_stage2 import apply_loss_preset, resolve_preset


class DirectionalRayBudgetTest(unittest.TestCase):
    def args(self):
        parser = ArgumentParser()
        Stage2PipelineParams(parser)
        Stage2OptimizationParams(parser)
        return parser.parse_args([])

    def lights(self, count):
        return np.ones((count, 3)), np.ones((count, 3))

    def direct(self):
        args = self.args()
        args.loss_preset = 'pbr_gt_directional_plane_native_direct'
        apply_loss_preset(args, [])
        return args

    def test_presets_and_auto(self):
        for name, plane in [('pbr_gt_directional_native_direct', False),
                            ('pbr_gt_directional_plane_native_direct', True)]:
            args = self.args()
            args.loss_preset = name
            apply_loss_preset(args, [])
            self.assertEqual(args.pbr_rasterizer_backend, 'lumimotion')
            self.assertEqual(args.pbr_light_mode, 'gt_directional')
            self.assertEqual(args.pbr_plane_projection, plane)
            self.assertTrue(args.wo_indirect and args.wo_indirect_relight)
            self.assertTrue(args.pbr_directional_actual_ray_budget)
            args.loss_preset = 'auto'
            self.assertEqual(resolve_preset(args, 'original_ir'), name)

    def test_default_and_legacy_presets(self):
        for name in ['irgs_baseline', 'pbr_gt_directional_plane_native']:
            args = self.args()
            args.loss_preset = name
            apply_loss_preset(args, [])
            self.assertFalse(args.pbr_directional_actual_ray_budget)
            self.assertEqual(photo_pixel_budget(args, args, self.lights(1)), (1024, 256))
        args = self.args()
        args.wo_indirect = True
        self.assertEqual(photo_pixel_budget(args, args, self.lights(3)), (1024, 256))

    def test_single_and_multiple_suns(self):
        args = self.direct()
        for count in [1, 2, 3, 17]:
            pixels, cost = photo_pixel_budget(args, args, self.lights(count))
            self.assertEqual(cost, count)
            self.assertEqual(pixels, args.trace_num_rays // count)
            self.assertLessEqual(pixels * cost, args.trace_num_rays)

    def test_indirect_cost_and_relight_flag(self):
        args = self.direct()
        args.wo_indirect = False
        self.assertEqual(photo_pixel_budget(args, args, self.lights(1)), (1020, 257))
        self.assertEqual(photo_pixel_budget(args, args, self.lights(2)), (1016, 258))
        self.assertEqual(photo_pixel_budget(args, args, self.lights(2), relight=True), (131072, 2))
        args.wo_indirect_relight = False
        self.assertEqual(photo_pixel_budget(args, args, self.lights(2), relight=True), (1016, 258))

    def test_explicit_hdr_uses_original_environment_budget(self):
        args = self.direct()
        args.diffuse_sample_num = 2048
        self.assertEqual(photo_pixel_budget(args, args, None), (128, 2048))

    def test_explicit_override_is_respected(self):
        args = self.direct()
        args.wo_indirect = False
        apply_loss_preset(args, ['--wo_indirect'])
        self.assertFalse(args.wo_indirect)
        self.assertEqual(photo_pixel_budget(args, args, self.lights(1)), (1020, 257))

    def test_invalid_mode_budget_and_light_shapes(self):
        for key, value in [('render_mode', 'photometric_lambertian'),
                           ('pbr_light_mode', 'envmap'), ('trace_num_rays', 0),
                           ('trace_num_rays', True), ('trace_num_rays', 1.5)]:
            args = self.direct()
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                validate_actual_ray_budget(args)
        args = self.direct()
        for lights in [self.lights(0), (np.ones((1, 2)), np.ones((1, 2))),
                       (np.ones((2, 3)), np.ones((1, 3)))]:
            with self.assertRaises(ValueError):
                photo_pixel_budget(args, args, lights)
        args.trace_num_rays = 1
        with self.assertRaises(ValueError):
            photo_pixel_budget(args, args, self.lights(2))
        args.wo_indirect = False
        args.diffuse_sample_num = 0
        with self.assertRaises(ValueError):
            photo_pixel_budget(args, args, self.lights(1))


if __name__ == '__main__':
    unittest.main()
