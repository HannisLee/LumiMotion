import unittest

import numpy as np

from scripts.calibrate_directional_light import (
    _directions_world,
    fit_nonnegative_scalar,
    linear_to_srgb,
    srgb_to_linear,
)


class DirectionalLightCalibrationTest(unittest.TestCase):
    def test_srgb_round_trip(self):
        values = np.asarray([0.0, 0.02, 0.18, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(
            linear_to_srgb(srgb_to_linear(values)),
            values,
            atol=1e-7,
        )

    def test_fit_recovers_known_scalar(self):
        prediction = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        target = prediction * 4.25
        self.assertAlmostEqual(fit_nonnegative_scalar(prediction, target), 4.25)

    def test_fit_clamps_negative_scalar(self):
        prediction = np.ones((2, 3))
        target = -np.ones((2, 3))
        self.assertEqual(fit_nonnegative_scalar(prediction, target), 0.0)

    def test_gt_directions_remain_in_world_space(self):
        cameras = {
            "frames": {
                "0001": {
                    "extrinsics": {
                        "position_world": [0.0, 2.0, 0.0],
                        # A non-identity camera rotation must not rotate either
                        # direction returned for world-space EXR normals.
                        "world_to_camera": [
                            [0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                }
            }
        }
        lights = {"0001": {"light_pos_world": [2.0, 0.0, 0.0]}}
        light, view = _directions_world(
            "0001", cameras, lights, np.zeros(3, dtype=np.float64)
        )
        np.testing.assert_allclose(light, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(view, [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
