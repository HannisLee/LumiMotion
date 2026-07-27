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
    def shade_plane(
        self,
        renderer,
        ray_light_to_surface,
        *,
        albedo=None,
        normal=None,
        alpha=1.0,
        background=None,
    ):
        albedo = torch.tensor(
            [0.8, 0.6, 0.4] if albedo is None else albedo,
            dtype=torch.float32,
        )[:, None, None]
        normal = torch.tensor(
            [0.0, 0.0, 1.0] if normal is None else normal,
            dtype=torch.float32,
        )[:, None, None]
        background = torch.tensor(
            [0.1, 0.2, 0.3] if background is None else background,
            dtype=torch.float32,
        )
        alpha_map = torch.full((1, 1, 1), float(alpha))
        rendered_albedo = albedo * alpha_map + background[:, None, None] * (
            1.0 - alpha_map
        )
        accumulated_normal = normal * alpha_map
        with torch.no_grad():
            renderer.light_model._raw_light_dir_table[0] = torch.tensor(
                ray_light_to_surface
            )
        return renderer(
            rendered_albedo,
            accumulated_normal,
            alpha_map,
            torch.zeros((3, 1, 1)),
            torch.tensor([0.0, 0.0, 2.0]),
            background,
            torch.tensor([0.0]),
        )

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
            ((0.0, 0.0, 1.0), 0.0),
        ]
        for ray_light_to_surface, expected in cases:
            with self.subTest(ray_light_to_surface=ray_light_to_surface):
                renderer = PhotometricLambertianRenderer([0.0], device="cpu")
                output = self.shade_plane(renderer, ray_light_to_surface)

                self.assertAlmostEqual(output["shading"].item(), expected, places=6)
                expected_rgb = (
                    torch.tensor([0.8, 0.6, 0.4]) * expected
                )
                torch.testing.assert_close(
                    output["foreground"][:, 0, 0],
                    expected_rgb,
                )

    def test_normal_flip_preserves_two_sided_cloth_shading(self):
        renderer = PhotometricLambertianRenderer([0.0], device="cpu")
        output = self.shade_plane(
            renderer,
            (0.0, 0.0, -1.0),
            normal=(0.0, 0.0, -4.0),
        )

        torch.testing.assert_close(
            output["normal"][:, 0, 0],
            torch.tensor([0.0, 0.0, 1.0]),
        )
        self.assertFalse(output["normal_camera_facing"].item())
        self.assertAlmostEqual(output["shading"].item(), 1.0, places=6)

    def test_low_alpha_uncomposites_albedo_and_recomposites_once(self):
        renderer = PhotometricLambertianRenderer([0.0], device="cpu")
        output = self.shade_plane(
            renderer,
            (0.0, 0.0, -1.0),
            alpha=1e-4,
        )

        torch.testing.assert_close(
            output["albedo"][:, 0, 0],
            torch.tensor([0.8, 0.6, 0.4]),
            atol=3e-4,
            rtol=0.0,
        )
        expected = (
            torch.tensor([0.8, 0.6, 0.4]) * 1e-4
            + torch.tensor([0.1, 0.2, 0.3]) * (1.0 - 1e-4)
        )
        torch.testing.assert_close(
            output["color"][:, 0, 0],
            expected,
            atol=1e-7,
            rtol=0.0,
        )

    def test_background_pixels_have_finite_deferred_gradients(self):
        renderer = PhotometricLambertianRenderer([0.0], device="cpu")
        rendered_albedo = torch.tensor(
            [[[0.4, 0.0]], [[0.3, 0.0]], [[0.2, 0.0]]],
            requires_grad=True,
        )
        accumulated_normal = torch.tensor(
            [[[0.4, 0.0]], [[0.0, 0.0]], [[0.8, 0.0]]],
            requires_grad=True,
        )
        alpha = torch.tensor([[[0.5, 0.0]]], requires_grad=True)
        output = renderer(
            rendered_albedo,
            accumulated_normal,
            alpha,
            torch.zeros((3, 1, 2)),
            torch.tensor([0.0, 0.0, 2.0]),
            torch.zeros(3),
            torch.tensor([0.0]),
        )

        output["color"].sum().backward()

        for tensor in (
            output["color"],
            rendered_albedo.grad,
            accumulated_normal.grad,
            alpha.grad,
            renderer.light_model._raw_light_dir_table.grad,
        ):
            self.assertTrue(torch.isfinite(tensor).all())

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

        albedo = torch.ones((3, 1, 2))
        normal = torch.tensor([0.0, 0.0, 1.0])[:, None, None].expand_as(albedo)
        alpha = torch.ones((1, 1, 2))
        positions = torch.tensor(
            [[[0.0, 2.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
        )
        output = renderer(
            albedo,
            normal,
            alpha,
            positions,
            torch.tensor([0.0, 0.0, 3.0]),
            torch.zeros(3),
            torch.tensor([0.0]),
        )
        renderer.training_setup(type("Args", (), {"photometric_light_lr": 1.0})())

        self.assertFalse(renderer.learns_light)
        self.assertIsNone(renderer.optimizer)
        self.assertFalse(renderer.light_model._raw_light_dir_table.requires_grad)
        self.assertEqual(renderer.light_smoothness_loss().item(), 0.0)
        self.assertAlmostEqual(output["shading"][0, 0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(
            output["shading"][0, 0, 1].item(),
            1.0 / math.sqrt(2.0),
            places=6,
        )
        torch.testing.assert_close(
            output["surface_to_light_dir"][:, 0, 0],
            torch.tensor([0.0, 0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
