import unittest

import torch

from utils.normal_eval_utils import (
    blender_camera_normal_to_runtime_view,
    normal_angular_error_degrees,
)


class NormalEvalUtilsTest(unittest.TestCase):
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
