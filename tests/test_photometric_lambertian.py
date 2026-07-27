import math
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scene.photometric_lambertian import (
    PhotometricLambertianRenderer,
    orient_normal_toward_camera,
)


class PhotometricLambertianTest(unittest.TestCase):
    def test_orient_normal_toward_camera_keeps_front_and_flips_back(self):
        normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        positions = torch.zeros_like(normals)
        oriented, camera_facing = orient_normal_toward_camera(
            normals,
            positions,
            torch.tensor([0.0, 0.0, 2.0]),
        )

        expected = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        torch.testing.assert_close(oriented, expected)
        self.assertEqual(camera_facing[:, 0].tolist(), [True, False])

    def test_orient_normal_toward_camera_outputs_unit_normals(self):
        oriented, _ = orient_normal_toward_camera(
            torch.tensor([[0.0, 0.0, -5.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.0, 0.0, 3.0]),
        )

        torch.testing.assert_close(oriented, torch.tensor([[0.0, 0.0, 1.0]]))
        torch.testing.assert_close(oriented.norm(dim=-1), torch.ones(1))

    def test_known_plane_lambertian_brightness(self):
        cases = [
            ((0.0, 0.0, -1.0), 1.0),
            ((-math.sqrt(3.0) / 2.0, 0.0, -0.5), 0.5),
            ((-1.0, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 1.0), 0.0),
        ]
        for ray_light_to_surface, expected in cases:
            with self.subTest(ray_light_to_surface=ray_light_to_surface):
                renderer = PhotometricLambertianRenderer([0.0], device="cpu")
                with torch.no_grad():
                    renderer.light_model._raw_light_dir_table[0] = torch.tensor(
                        ray_light_to_surface
                    )

                output = renderer(
                    torch.ones((1, 3)),
                    torch.tensor([[0.0, 0.0, 1.0]]),
                    torch.tensor([0.0]),
                )

                self.assertAlmostEqual(output["shading"].item(), expected, places=6)

    def test_orient_normal_validates_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            orient_normal_toward_camera(
                torch.zeros(2, 3),
                torch.zeros(1, 3),
                torch.zeros(3),
            )

    def test_gt_point_light_uses_fixed_world_position(self):
        renderer = PhotometricLambertianRenderer(
            [0.0, 1.0],
            light_mode="gt_point",
            device="cpu",
        )
        payload = {
            "0001": {"light_pos_world": [0.0, 0.0, 2.0]},
            "0002": {"light_pos_world": [0.0, 2.0, 0.0]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            renderer.initialize_gt_point_lights(
                str(path),
                torch.zeros(3),
            )

        output = renderer(
            torch.ones((1, 3)),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.0]),
            position=torch.zeros((1, 3)),
        )
        renderer.training_setup(type("Args", (), {"photometric_light_lr": 1.0})())

        self.assertFalse(renderer.learns_light)
        self.assertIsNone(renderer.optimizer)
        self.assertFalse(renderer.light_model._raw_light_dir_table.requires_grad)
        self.assertEqual(renderer.light_smoothness_loss().item(), 0.0)
        self.assertAlmostEqual(output["shading"].item(), 1.0, places=6)
        torch.testing.assert_close(
            output["surface_to_light_dir"],
            torch.tensor([[0.0, 0.0, 1.0]]),
        )


if __name__ == "__main__":
    unittest.main()
