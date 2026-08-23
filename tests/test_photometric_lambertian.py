import math
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scene.photometric_lambertian import (
    PhotometricLambertianRenderer,
    deform_independent_normal,
    linear_to_srgb,
    orient_normal_toward_camera,
    srgb_to_linear,
)


class PhotometricLambertianTest(unittest.TestCase):
    def test_independent_normal_follows_only_relative_deformation_rotation(self):
        half_angle = math.pi / 4.0
        canonical_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        deformed_rotation = torch.tensor(
            [[math.cos(half_angle), 0.0, math.sin(half_angle), 0.0]]
        )
        transformed = deform_independent_normal(
            torch.tensor([[0.0, 0.0, 1.0]]),
            canonical_rotation,
            deformed_rotation,
        )

        torch.testing.assert_close(
            transformed,
            torch.tensor([[1.0, 0.0, 0.0]]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_independent_normal_is_unchanged_without_deformation(self):
        rotation = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        normal = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        transformed = deform_independent_normal(normal, rotation, rotation)

        torch.testing.assert_close(
            transformed,
            torch.nn.functional.normalize(normal, dim=-1),
            atol=1e-6,
            rtol=1e-6,
        )
        transformed[:, 0].sum().backward()
        self.assertIsNotNone(normal.grad)
        self.assertGreater(normal.grad.abs().sum().item(), 0.0)

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
        torch.testing.assert_close(
            output["color_linear"],
            torch.full((1, 3), 1.0 / (4.0 * math.pi)),
        )
        torch.testing.assert_close(output["color"], linear_to_srgb(output["color_linear"]))
        torch.testing.assert_close(output["light_distance"], torch.tensor([[2.0]]))
        torch.testing.assert_close(output["light_attenuation"], torch.tensor([[0.25]]))

    def test_gt_point_light_preserves_fixed_intensity_and_rgb_color(self):
        renderer = PhotometricLambertianRenderer(
            [0.0],
            light_mode="gt_point",
            gt_light_intensity=2.0,
            gt_light_color="0.25,0.5,1.0",
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lights.json"
            path.write_text(json.dumps({"0001": {"light_pos_world": [0, 0, 1]}}))
            renderer.initialize_gt_point_lights(str(path), torch.zeros(3))
        output = renderer(
            torch.ones((1, 3)),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.0]),
            position=torch.zeros((1, 3)),
        )
        torch.testing.assert_close(
            output["color_linear"], torch.tensor([[0.5, 1.0, 2.0]]) / math.pi
        )

    def test_gt_directional_light_is_fixed_and_ignores_distance(self):
        renderer = PhotometricLambertianRenderer(
            [0.0, 1.0],
            light_mode="gt_directional",
            gt_light_intensity=2.9,
            device="cpu",
        )
        payload = {
            "0001": {"light_pos_world": [0.0, 0.0, 2.0]},
            "0002": {"light_pos_world": [0.0, 0.0, 20.0]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lights.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            renderer.initialize_gt_directional_lights(str(path), torch.zeros(3))

        near = renderer(
            torch.ones((1, 3)),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.0]),
            position=torch.tensor([[100.0, 0.0, -50.0]]),
        )
        far = renderer(
            torch.ones((1, 3)),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([1.0]),
            position=torch.tensor([[-100.0, 0.0, 50.0]]),
        )
        renderer.training_setup(type("Args", (), {})())

        self.assertFalse(renderer.learns_light)
        self.assertIsNone(renderer.optimizer)
        self.assertFalse(renderer.light_model._raw_light_dir_table.requires_grad)
        torch.testing.assert_close(near["surface_to_light_dir"], torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(near["color_linear"], far["color_linear"])
        torch.testing.assert_close(near["light_attenuation"], torch.ones((1, 1)))
        self.assertTrue(torch.isinf(near["light_distance"]).all())
        self.assertFalse(renderer.initialization_metadata["uses_distance_attenuation"])

    def test_from_args_uses_world_normal_calibrated_default_intensity(self):
        renderer = PhotometricLambertianRenderer.from_args(
            [0.0], type("Args", (), {})(), device="cpu"
        )
        torch.testing.assert_close(
            renderer.gt_light_intensity, torch.tensor(5.5)
        )

    def test_directional_intensity_pi_matches_legacy_linear_multiplier(self):
        renderer = PhotometricLambertianRenderer(
            [0.0],
            gt_light_intensity=math.pi,
            device="cpu",
        )
        with torch.no_grad():
            renderer.light_model._raw_light_dir_table[0] = torch.tensor(
                [0.0, 0.0, -1.0]
            )
        albedo_srgb = torch.tensor([[0.25, 0.5, 0.75]])
        output = renderer(
            albedo_srgb,
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.0]),
        )
        torch.testing.assert_close(output["color_linear"], srgb_to_linear(albedo_srgb))

    def test_linear_color_is_not_clamped_before_rasterization(self):
        renderer = PhotometricLambertianRenderer(
            [0.0],
            gt_light_intensity=10.0,
            device="cpu",
        )
        with torch.no_grad():
            renderer.light_model._raw_light_dir_table[0] = torch.tensor(
                [0.0, 0.0, -1.0]
            )
        output = renderer(
            torch.ones((1, 3)),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.0]),
        )
        self.assertGreater(output["color_linear"].max().item(), 1.0)
        self.assertEqual(output["color"].max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
