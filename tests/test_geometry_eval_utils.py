"""用已知方向/平面/漏预测验证评测口径，避免自洽但错误的指标。"""
import unittest

import numpy as np

from utils.geometry_eval_utils import (aggregate_regions, camera_rays, depth_normals,
                                       erode, evaluate_region, normal_errors)


class GeometryEvaluationTest(unittest.TestCase):
    def test_signed_and_axis_angles(self):
        predicted = np.array([[0., 0., 2.], [0., 0., -2.], [1., 0., 0.]])
        target = np.array([[0., 0., 1.]]*3)
        errors, valid = normal_errors(predicted, target)
        np.testing.assert_allclose(errors['angular_deg'], [0, 180, 90])
        np.testing.assert_allclose(errors['axis_deg'], [0, 0, 90])
        np.testing.assert_allclose(errors['component_l1'], [0, 2/3, 2/3])
        self.assertTrue(valid.all())

    def test_pinhole_z_differs_from_ray_distance(self):
        rays = camera_rays(np.eye(4), [1., 1., .5, .5], 1, 2)
        np.testing.assert_allclose(rays[0, 0, 3:], [0, 0, -1])
        np.testing.assert_allclose(rays[0, 1, 3:], [1, 0, -1])
        point = rays[0, 1, :3] + 2*rays[0, 1, 3:]
        self.assertEqual(-point[2], 2)
        self.assertAlmostEqual(np.linalg.norm(point), 2*np.sqrt(2))

    def test_plane_normal_and_invalid_depth_neighborhood(self):
        depth = np.ones((7, 7))*2
        normal, valid = depth_normals(depth, np.eye(4), [5., 5., 3.5, 3.5])
        np.testing.assert_allclose(normal[valid], np.tile([0, 0, 1.], (int(valid.sum()), 1)), atol=1e-7)
        self.assertEqual(int(valid.sum()), 25)
        depth[3, 3] = 0
        normal, valid = depth_normals(depth, np.eye(4), [5., 5., 3.5, 3.5])
        self.assertFalse(valid[2:5, 2:5].any())
        np.testing.assert_allclose(normal[2:5, 2:5], 0)

    def prediction(self, height=1, width=2):
        normal = np.tile([0., 0., 1.], (height, width, 1))
        return dict(normal=normal, alpha=np.ones((height, width)),
                    expected=np.ones((height, width))*3, median=np.ones((height, width))*4)

    def test_missing_predictions_cannot_improve_coverage(self):
        prediction = self.prediction()
        prediction['alpha'][0, 1] = 0
        region = evaluate_region(prediction, prediction['normal'], np.ones((1, 2), bool),
                                 np.ones((1, 2))*2, np.ones((1, 2), bool))
        self.assertEqual(region['normal']['angular_deg']['mean'], 0)
        self.assertEqual(region['normal']['angular_with_missing_deg']['mean'], 90)
        self.assertEqual(region['normal_coverage'], .5)
        self.assertEqual(region['depth']['expected']['absolute']['mean'], 1)
        self.assertEqual(region['depth']['median']['absolute']['mean'], 2)
        self.assertEqual(region['depth']['expected']['coverage'], .5)

    def test_no_valid_prediction_reports_null_error(self):
        prediction = self.prediction()
        prediction['normal'][:] = 0
        prediction['expected'][:] = np.nan
        region = evaluate_region(prediction, np.tile([0., 0., 1.], (1, 2, 1)), np.ones((1, 2), bool),
                                 np.ones((1, 2))*2, np.ones((1, 2), bool))
        self.assertIsNone(region['normal']['angular_deg']['mean'])
        self.assertIsNone(region['depth']['expected']['absolute']['mean'])
        self.assertEqual(region['normal']['angular_with_missing_deg']['mean'], 180)

    def test_frame_and_pixel_weighting_are_explicit(self):
        frames = []
        for width, delta in ((1, 1.), (3, 3.)):
            prediction = self.prediction(width=width)
            prediction['expected'][:] = 2+delta
            region = evaluate_region(prediction, prediction['normal'], np.ones((1, width), bool),
                                     np.ones((1, width))*2, np.ones((1, width), bool))
            frames.append(dict(regions=dict(foreground=region)))
        stats = aggregate_regions(frames)['foreground']['depth']['expected']
        self.assertEqual(stats['mae_frame_mean'], 2)
        self.assertEqual(stats['mae_pixel_mean'], 2.5)

    def test_erode_uses_gt_boundary(self):
        mask = np.ones((5, 5), bool)
        self.assertEqual(int(erode(mask, 1).sum()), 9)

    def test_three_normal_objects_have_independent_values(self):
        prediction = self.prediction(5, 5)
        prediction['depth_normal'] = np.tile([1., 0., 0.], (5, 5, 1))
        prediction['mesh_normal'] = np.tile([0., 0., -1.], (5, 5, 1))
        prediction['mesh_valid'] = np.ones((5, 5), bool)
        region = evaluate_region(prediction, prediction['normal'], np.ones((5, 5), bool),
                                 None, np.ones((5, 5), bool))
        metrics = region['normal_metrics']
        self.assertEqual(metrics['gs_render']['angular_deg']['mean'], 0)
        self.assertEqual(metrics['depth_normal']['angular_deg']['mean'], 90)
        self.assertEqual(metrics['mesh']['angular_deg']['mean'], 180)
        self.assertEqual(metrics['depth_normal']['coverage'], 9/25)
        self.assertEqual(region['common_normal_metrics']['valid_pixels'], 9)
        self.assertEqual(region['normal']['angular_deg'], metrics['gs_render']['angular_deg'])

    def test_absent_mesh_is_unavailable_not_zero_error(self):
        prediction = self.prediction(5, 5)
        region = evaluate_region(prediction, prediction['normal'], np.ones((5, 5), bool),
                                 None, np.ones((5, 5), bool))
        item = region['normal_metrics']['mesh']
        self.assertEqual(item['status'], 'not_available')
        self.assertIsNone(item['angular_deg']['mean'])
        self.assertIsNone(item['coverage'])
        summary = aggregate_regions([dict(regions=dict(foreground=region))])
        self.assertIsNone(summary['foreground']['normal_metrics']['mesh']['mae_frame_mean_deg'])

    def test_mesh_coverage_does_not_depend_on_gaussian_alpha(self):
        prediction = self.prediction(5, 5)
        prediction['alpha'][:] = 0
        prediction['mesh_normal'] = prediction['normal'].copy()
        prediction['mesh_valid'] = np.ones((5, 5), bool)
        prediction['mesh_valid'][0, 0] = False
        region = evaluate_region(prediction, prediction['normal'], np.ones((5, 5), bool),
                                 None, np.ones((5, 5), bool))
        self.assertEqual(region['normal_metrics']['gs_render']['coverage'], 0)
        self.assertEqual(region['normal_metrics']['mesh']['coverage'], 24/25)
        self.assertEqual(region['normal_metrics']['mesh']['angular_deg']['mean'], 0)
        self.assertEqual(region['normal_metrics']['mesh']['angular_with_missing_deg']['mean'], 180/25)

    def test_named_normal_aggregation_uses_equal_frame_weights(self):
        frames = []
        for width, direction in [(1, [0,0,1]), (3, [1,0,0])]:
            prediction = self.prediction(width=width)
            prediction['normal'][:] = direction
            target = np.tile([0.,0.,1.], (1,width,1))
            region = evaluate_region(prediction, target, np.ones((1,width),bool),
                                     None, np.ones((1,width),bool))
            frames.append(dict(regions=dict(foreground=region)))
        item = aggregate_regions(frames)['foreground']['normal_metrics']['gs_render']
        self.assertEqual(item['mae_frame_mean_deg'], 45)
        self.assertEqual(item['mae_pixel_mean_deg'], 67.5)
        self.assertEqual(item['coverage'], 1)


if __name__ == '__main__':
    unittest.main()
