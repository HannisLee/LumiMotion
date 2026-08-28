import unittest

import torch

from utils.normal_eval_utils import (
    blender_camera_normal_to_runtime_view,
    normal_angular_error_degrees,
    resolve_normal_source,
)


class NormalEvalUtilsTest(unittest.TestCase):
    def test_normal_source_auto_preserves_checkpoint_driven_behavior(self):
        self.assertEqual(
            resolve_normal_source("auto", has_independent=True),
            "independent_photometric_normal",
        )
        self.assertEqual(
            resolve_normal_source("auto", has_independent=False),
            "gs_raster_normal",
        )

    def test_normal_source_can_force_gs_when_independent_exists(self):
        self.assertEqual(
            resolve_normal_source("gs", has_independent=True),
            "gs_raster_normal",
        )

    def test_normal_source_rejects_missing_independent_normal(self):
        with self.assertRaisesRegex(
            ValueError, "does not contain independent normals"
        ):
            resolve_normal_source("independent", has_independent=False)

    def test_blender_camera_basis_maps_to_runtime_view_basis(self):
        normal = torch.tensor([[0.0, 1.0, -1.0]])
        converted = blender_camera_normal_to_runtime_view(normal)
        torch.testing.assert_close(converted, torch.tensor([[0.0, -2**-0.5, 2**-0.5]]))

    def test_angular_error_respects_valid_mask(self):
        rendered = torch.tensor([[[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 0.0]]])
        gt = torch.tensor([[[0.0, 1.0]], [[0.0, 0.0]], [[1.0, 0.0]]])
        mask = torch.tensor([[True, False]])
        errors = normal_angular_error_degrees(rendered, gt, mask)
        torch.testing.assert_close(errors, torch.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
