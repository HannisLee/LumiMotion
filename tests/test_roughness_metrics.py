"""验证 roughness 的条件分母、alpha、逐帧浮点真值、异常与原版模型门控。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.roughness_metrics import aggregate, evaluate_region, normalize_roughness, load_target


class RoughnessMetricsTest(unittest.TestCase):
    def test_identity_and_constant_offset(self):
        target = np.full((2, 3), 0.6)
        mask, alpha = np.ones((2, 3), bool), np.ones((2, 3))
        self.assertEqual(evaluate_region(target, target, mask, alpha)['mae'], 0)
        self.assertAlmostEqual(evaluate_region(target + 0.1, target, mask, alpha)['mae'], 0.1)

    def test_background_does_not_dilute_error_or_coverage(self):
        for size in (2, 100):
            pred = np.full((size, size), 0.2)
            target = np.full_like(pred, 0.5)
            alpha = np.zeros_like(pred)
            mask = np.zeros_like(pred, dtype=bool)
            mask[0, :2] = True
            alpha[0, 0] = 1
            result = evaluate_region(pred, target, mask, alpha)
            self.assertAlmostEqual(result['mae'], 0.3)
            self.assertEqual(result['coverage_pct'], 50)
            self.assertEqual(result['valid_pixels'], 1)

    def test_alpha_normalization_and_threshold(self):
        alpha = np.array([[1., 0.9, 0.89, 0.1, 0.]])
        pred = normalize_roughness(alpha * 0.82, alpha)
        np.testing.assert_allclose(pred[0, :4], 0.82)
        self.assertEqual(pred[0, 4], 0)
        result = evaluate_region(pred, np.full_like(pred, 0.82), np.ones_like(pred, bool), alpha)
        self.assertAlmostEqual(result['mae'], 0)
        self.assertEqual(result['coverage_pct'], 40)

    def test_missing_and_empty_regions(self):
        zero = np.zeros((2, 3))
        missing = evaluate_region(zero, zero + 0.82, np.ones_like(zero, bool), zero)
        self.assertIsNone(missing['mae'])
        self.assertEqual(missing['coverage_pct'], 0)
        empty = evaluate_region(zero, zero, np.zeros_like(zero, bool), zero)
        self.assertIsNone(empty['mae'])
        self.assertIsNone(empty['coverage_pct'])

    def test_nan_is_rejected_even_outside_valid_mask(self):
        zero = np.zeros((1, 2))
        for bad in (np.nan, np.inf, -np.inf):
            corrupt = zero.copy()
            corrupt[0, 1] = bad
            with self.assertRaises(ValueError):
                normalize_roughness(corrupt, zero)
            with self.assertRaises(ValueError):
                evaluate_region(corrupt, zero, zero.astype(bool), zero)

    def test_invalid_ranges_and_shapes(self):
        for pred, alpha in [(np.ones((2, 2)) * 2, np.ones((2, 2))),
                            (np.ones((2, 2)), np.ones((2, 2)) * -1),
                            (np.ones((2, 2)), np.ones((1, 2)))]:
            with self.assertRaises(ValueError):
                normalize_roughness(pred, alpha)

    def test_foreground_aggregation_uses_pixel_counts_only(self):
        def frame(width, delta):
            target = np.zeros((1, width))
            stat = evaluate_region(target + delta, target, np.ones_like(target, bool), np.ones_like(target))
            return {'foreground': stat}
        summary = aggregate([frame(1, 0.1), frame(9, 0.3)])
        self.assertEqual(set(summary), {'foreground'})
        self.assertAlmostEqual(summary['foreground']['mae'], 0.28)
        self.assertEqual(summary['foreground']['gt_pixels'], 10)

    def test_npz_target_is_independent_of_scene_frame_count_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roughness_0013.npz'
            r = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.9]])
            alpha = np.ones_like(r)
            np.savez(path, roughness=r, alpha=alpha, time=0.25)
            target, actual_alpha, info = load_target(path, (2, 3))
            np.testing.assert_array_equal(target, r)
            np.testing.assert_array_equal(actual_alpha, alpha)
            self.assertEqual(info['time'], 0.25)
            self.assertEqual(info['original_shape'], [2, 3])

    def test_target_resize_avoids_transparent_background_dilution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gt_0057.npz'
            np.savez(path, roughness=np.array([[0., 0.6]]), alpha=np.array([[0., 1.]]))
            target, alpha, _ = load_target(path, (1, 1))
            np.testing.assert_allclose(target, 0.6)
            np.testing.assert_allclose(alpha, 0.5)

    def test_exr_premultiplied_alpha_is_removed_without_gamma(self):
        import OpenEXR
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roughness_0042.exr'
            alpha = np.array([[1., 0.5]], dtype=np.float32)
            color = alpha * 0.8
            OpenEXR.File({}, {'R': color, 'G': color, 'B': color, 'A': alpha}).write(str(path))
            target, actual_alpha, _ = load_target(path, (1, 2))
            np.testing.assert_allclose(target, 0.8)
            np.testing.assert_array_equal(actual_alpha, alpha)

    def test_display_png_and_nonfinite_gt_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'PNG'):
                load_target(Path(directory) / 'roughness_0001.png', (2, 2))
            path = Path(directory) / 'roughness_0001.npz'
            np.savez(path, roughness=np.array([[np.nan]]), alpha=np.ones((1, 1)))
            with self.assertRaises(ValueError):
                load_target(path, (1, 1))

    def test_lambertian_and_stage1_are_rejected_before_loading(self):
        from scripts.eval_roughness_gt import validate_model
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / 'cfg_args').write_text('Namespace()')
            with self.assertRaisesRegex(ValueError, 'envmap'):
                validate_model(model, 35000)
            (model / 'stage2_photometric').mkdir()
            with self.assertRaisesRegex(ValueError, 'Lambertian'):
                validate_model(model, 55000)

    def test_raw_exr_verifier_reconstructs_resize_and_detects_tampering(self):
        import OpenEXR
        from scripts.verify_roughness_eval import verify_raw_gt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roughness_0010.exr'
            alpha = np.array([[1., 0.5], [0.8, 1.]], dtype=np.float32)
            color = alpha * np.array([[0.4, 0.7], [0.9, 0.2]], dtype=np.float32)
            OpenEXR.File({}, {'R': color, 'G': color, 'B': color, 'A': alpha}).write(str(path))
            target, actual_alpha, _ = load_target(path, (1, 1))
            record = dict(gt_source_path=str(path), height=1, width=1)
            arrays = dict(gt_alpha=actual_alpha, gt_roughness=target.copy())
            provenance = {'input_hashes': {str(path): '由外层验证文件哈希'}}
            verify_raw_gt(record, arrays, provenance)
            arrays['gt_roughness'][0, 0] += 0.1
            with self.assertRaises(AssertionError):
                verify_raw_gt(record, arrays, provenance)

    def test_bad_raw_exr_still_fails_in_loader_and_independent_verifier(self):
        import OpenEXR
        from scripts.verify_roughness_eval import verify_raw_gt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'roughness_0030.exr'
            for opacity, accumulated in [(1., 1.0014025), (0.02, 0.2), (1., np.nan)]:
                alpha = np.array([[opacity]], dtype=np.float32)
                color = np.array([[accumulated]], dtype=np.float32)
                OpenEXR.File({}, {'R': color, 'G': color, 'B': color, 'A': alpha}).write(str(path))
                with self.assertRaises(ValueError):
                    load_target(path, (1, 1))
                with self.assertRaises(AssertionError):
                    verify_raw_gt(dict(gt_source_path=str(path), height=1, width=1),
                        dict(gt_alpha=alpha, gt_roughness=np.zeros((1, 1))),
                        {'input_hashes': {str(path): '由外层验证文件哈希'}})

    def test_raw_npz_verifier_checks_alpha_and_provenance(self):
        from scripts.verify_roughness_eval import verify_raw_gt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gt.npz'
            alpha, target = np.ones((1, 2)), np.array([[0.2, 0.8]])
            np.savez(path, roughness=target, alpha=alpha)
            record = dict(gt_source_path=str(path), height=1, width=2)
            arrays = dict(gt_alpha=alpha.copy(), gt_roughness=target)
            provenance = {'input_hashes': {str(path): '由外层验证文件哈希'}}
            verify_raw_gt(record, arrays, provenance)
            with self.assertRaises(AssertionError):
                verify_raw_gt(record, arrays, {'input_hashes': {}})
            arrays['gt_alpha'][0, 0] = 0.5
            with self.assertRaises(AssertionError):
                verify_raw_gt(record, arrays, provenance)


if __name__ == '__main__':
    unittest.main()
