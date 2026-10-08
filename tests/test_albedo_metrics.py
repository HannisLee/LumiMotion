"""独立数值样例验证材质误差、前景分母、透明度和训练校准。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from utils.albedo_metrics import (ScaleFit, interior_mask, linear_to_srgb, material_masks,
                                  normalize_accumulated, srgb_to_linear, summarize_errors)


class AlbedoMetricsTest(unittest.TestCase):
    def test_srgb_reference_values_and_roundtrip(self):
        np.testing.assert_allclose(srgb_to_linear([0, 0.5, 1]), [0, 0.21404114048223255, 1], atol=1e-12)
        values = np.array([0, 0.001, 0.03, 0.3, 0.9, 1])
        np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(values)), values, atol=1e-12)

    def test_foreground_denominator_is_independent_of_background(self):
        for size in (2, 20):
            pred = np.zeros((3, size, size))
            target = np.ones_like(pred)
            mask = np.zeros((size, size), dtype=bool)
            mask[0, 0] = True
            pred[:, 0, 0] = [0.9, 0.8, 0.7]
            result = summarize_errors(pred, target, mask, np.ones_like(mask))
            self.assertAlmostEqual(result['mae_linear'], 0.2)
            self.assertAlmostEqual(result['rmse_linear'], np.sqrt(0.14 / 3))
            self.assertEqual(result['pixels'], 1)

    def test_alpha_normalization_does_not_darken_albedo(self):
        color = np.array([0.42, 0.018, 0.028])[:, None, None]
        alpha = np.array([[1., 0.5, 0.1]])
        pred = normalize_accumulated(color * alpha, alpha)
        np.testing.assert_allclose(pred, np.broadcast_to(color, (3, 1, 3)))
        result = summarize_errors(pred, pred.copy(), np.ones((1, 3), bool), alpha)
        self.assertEqual(result['mae_linear'], 0)
        self.assertAlmostEqual(result['coverage_pct'], 100 / 3)

    def test_missing_prediction_is_counted_as_zero_on_gt_mask(self):
        alpha = np.zeros((1, 3))
        pred = normalize_accumulated(np.zeros((3, 1, 3)), alpha)
        target = np.full((3, 1, 3), 0.6)
        core = interior_mask(np.ones((1, 3)), erosion_pixels=0)
        result = summarize_errors(pred, target, core, alpha)
        self.assertEqual(result['pixels'], 3)
        self.assertAlmostEqual(result['mae_linear'], 0.6)
        self.assertEqual(result['coverage_pct'], 0)
        self.assertIsNone(result['high_alpha_mae_linear_diagnostic'])

    def test_gt_core_removes_soft_boundaries(self):
        alpha = np.ones((7, 7))
        alpha[[0, -1]] = 0.6
        mask = interior_mask(alpha, threshold=0.99, erosion_pixels=1)
        self.assertEqual(int(mask.sum()), 15)
        self.assertFalse(mask[0].any())

    def test_palette_regions_do_not_overlap_and_exclude_mixed_colors(self):
        palette = [{'name': 'red', 'srgb8': [255, 0, 0]}, {'name': 'blue', 'srgb8': [0, 0, 255]}]
        target = np.array([[[1, 0, 0.5]], [[0, 0, 0]], [[0, 1, 0.5]]])
        regions = material_masks(target, np.ones((1, 3), bool), palette, erosion_pixels=0)
        np.testing.assert_array_equal(regions['red'], [[True, False, False]])
        np.testing.assert_array_equal(regions['blue'], [[False, True, False]])

    def test_global_train_fit_corrects_scale_on_different_heldout_colors(self):
        train_gt = np.array([[[0.8, 0.2]], [[0.4, 0.3]], [[0.2, 0.1]]])
        fit = ScaleFit()
        fit.update(train_gt / 2, train_gt, np.ones((1, 2), bool))
        result = fit.result()
        self.assertAlmostEqual(result['scalar'], 2)
        np.testing.assert_allclose(result['rgb'], [2, 2, 2])
        heldout_gt = np.array([[[0.42]], [[0.018]], [[0.028]]])
        np.testing.assert_allclose(heldout_gt / 2 * result['scalar'], heldout_gt)

    def test_nan_render_and_empty_scale_fit_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_accumulated(np.full((3, 2, 2), np.nan), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            ScaleFit().result()
        self.assertIsNone(summarize_errors(np.zeros((3, 2, 2)), np.zeros((3, 2, 2)),
                                           np.zeros((2, 2), bool), np.zeros((2, 2))))

    def test_target_resize_does_not_mix_transparent_black_into_white(self):
        from scripts.eval_albedo_gt import load_target
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'albedo_0001.png'
            Image.fromarray(np.array([[[0, 0, 0, 0], [255, 255, 255, 255]]], dtype=np.uint8), 'RGBA').save(path)
            target, alpha, _ = load_target(path, (1, 1))
            np.testing.assert_allclose(target, 1)
            np.testing.assert_allclose(alpha, 0.5)

    def test_missing_input_ply_is_rejected_before_reader_can_create_it(self):
        from argparse import Namespace
        from scripts.eval_albedo_gt import evaluation_config
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in ('transforms_train.json', 'transforms_test.json'):
                (source / name).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'points3d.ply'):
                evaluation_config(Namespace(is_blender=True, source_path=directory), None)
            self.assertFalse((source / 'points3d.ply').exists())


if __name__ == '__main__':
    unittest.main()
