import unittest

import numpy as np

from scene.dataset_readers import blender_c2w_to_camera_extrinsics
from utils.graphics_utils import getWorld2View2


class BlenderCameraConventionTest(unittest.TestCase):
    def test_standard_blender_c2w_preserves_camera_center(self):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 3] = (1.0, 2.0, 3.0)
        rotation, translation = blender_c2w_to_camera_extrinsics(c2w)
        runtime_c2w = np.linalg.inv(getWorld2View2(rotation, translation))
        np.testing.assert_allclose(runtime_c2w[:3, 3], c2w[:3, 3], atol=1e-6)

    def test_blender_camera_forward_maps_to_runtime_positive_z(self):
        c2w = np.eye(4, dtype=np.float64)
        rotation, translation = blender_c2w_to_camera_extrinsics(c2w)
        runtime_w2c = getWorld2View2(rotation, translation)
        # Blender looks along local -Z. In this rasterizer the forward ray is +Z.
        point_in_front = np.array([0.0, 0.0, -2.0, 1.0])
        camera_point = runtime_w2c @ point_in_front
        self.assertGreater(camera_point[2], 0.0)


if __name__ == "__main__":
    unittest.main()
