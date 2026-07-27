import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scene.photometric_perlight_pbr import (
    PHOTOMETRIC_PBR_VERSION,
    PhotometricPerLightPBRRenderer,
    StructuredDirectionalLightModel,
    _ggx_specular_times_ndotl,
    _rough_diffuse_times_ndotl,
    linear_to_srgb,
    srgb_to_linear,
)


class PhotometricPerLightPBRTest(unittest.TestCase):
    def make_renderer(self, count=4):
        renderer = PhotometricPerLightPBRRenderer(
            [0.0, 0.5, 1.0],
            num_gaussians=count,
            device="cpu",
            light_samples_train=4,
            light_samples_eval=8,
            use_visibility=False,
        )
        with torch.no_grad():
            renderer.light_model.fourier_coefficients.zero_()
            # Stored light direction is light-to-surface. -Z therefore lights
            # a +Z-facing surface.
            renderer.light_model.fourier_coefficients[0, 2] = -1.0
        return renderer

    def test_srgb_linear_roundtrip(self):
        values = torch.tensor([0.0, 0.01, 0.18, 0.5, 1.0])
        torch.testing.assert_close(
            linear_to_srgb(srgb_to_linear(values)),
            values,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_structured_light_initialization_and_bounds(self):
        model = StructuredDirectionalLightModel(
            torch.linspace(0.0, 1.0, 9),
            max_residual_angle_degrees=7.0,
            device="cpu",
        )
        phase = torch.linspace(0.0, 2.0 * torch.pi, 9)
        directions = F.normalize(
            torch.stack(
                (0.3 * torch.cos(phase), 0.2 * torch.sin(phase), -torch.ones(9)),
                dim=-1,
            ),
            dim=-1,
        )
        model.initialize_from_directions(directions)
        with torch.no_grad():
            model.raw_tangent_residual.fill_(100.0)

        output = model.get_all_light_dirs()
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(9))
        self.assertLessEqual(model.tangent_residual_angles().max().item(), 7.0001)

    def test_front_back_diffuse_and_specular_are_finite(self):
        albedo = torch.ones(2, 1, 3)
        normal = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]])
        view = normal.clone()
        light = torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]])
        roughness = torch.full((2, 1, 1), 0.5)
        diffuse = _rough_diffuse_times_ndotl(
            albedo, normal, view, light, roughness
        )
        specular = _ggx_specular_times_ndotl(
            normal,
            view,
            light,
            roughness,
            torch.full_like(albedo, 0.04),
        )

        self.assertTrue(torch.isfinite(diffuse).all())
        self.assertTrue(torch.isfinite(specular).all())
        self.assertGreater(diffuse[0].mean().item(), 0.0)
        self.assertGreater(specular[0].mean().item(), 0.0)
        self.assertEqual(diffuse[1].abs().max().item(), 0.0)
        self.assertEqual(specular[1].abs().max().item(), 0.0)

    def test_forward_constraints_shapes_and_gradients(self):
        renderer = self.make_renderer()
        albedo = torch.full((4, 3), 0.5, requires_grad=True)
        normal = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, -1)
        position = torch.zeros((4, 3))
        roughness = torch.full((4, 1), 0.6, requires_grad=True)
        output = renderer(
            albedo,
            normal,
            torch.tensor([0.5]),
            position,
            torch.tensor([0.0, 0.0, 2.0]),
            roughness,
        )

        self.assertEqual(output["color_linear"].shape, (4, 3))
        self.assertEqual(output["visibility"].shape, (4, 1))
        self.assertEqual(output["surface_to_light_samples"].shape, (4, 3))
        self.assertTrue(torch.isfinite(output["color_linear"]).all())
        torch.testing.assert_close(
            output["residual_multiplier"],
            torch.ones((4, 3)),
        )
        self.assertLessEqual(
            output["normal_residual_angle"].max().item(),
            renderer.normal_residual_angle_degrees,
        )
        self.assertLessEqual(renderer.exposure_values().abs().max().item(), 0.1)
        self.assertAlmostEqual(renderer.exposure_values().mean().item(), 0.0)

        output["color_linear"].mean().backward()
        self.assertTrue(torch.isfinite(albedo.grad).all())
        self.assertTrue(torch.isfinite(roughness.grad).all())
        self.assertTrue(
            torch.isfinite(renderer.raw_global_intensity.grad).all()
        )

    def test_local_knn_visibility_is_bounded_and_differentiable(self):
        renderer = PhotometricPerLightPBRRenderer(
            [0.0],
            num_gaussians=4,
            device="cpu",
            use_visibility=True,
            visibility_backend="local_knn",
            shadow_neighbors=2,
        )
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ]
        )
        scales = torch.full((4, 3), 0.05)
        opacity = torch.full((4, 1), 0.8)
        directions = F.normalize(
            torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]),
            dim=-1,
        ).requires_grad_(True)

        visibility = renderer.compute_local_visibility(
            positions, scales, opacity, directions
        )

        self.assertEqual(visibility.shape, (4, 2, 1))
        self.assertTrue(torch.isfinite(visibility).all())
        self.assertGreaterEqual(visibility.min().item(), 0.0)
        self.assertLessEqual(visibility.max().item(), 1.0)
        visibility.mean().backward()
        self.assertTrue(torch.isfinite(directions.grad).all())

    def test_checkpoint_roundtrip_and_version_guard(self):
        renderer = self.make_renderer()
        args = SimpleNamespace(
            photometric_light_lr=1e-4,
            photometric_pbr_exposure_lr=1e-3,
            photometric_pbr_environment_lr=1e-3,
            photometric_pbr_normal_lr=1e-4,
            photometric_pbr_residual_lr=1e-4,
        )
        renderer.training_setup(args)
        with tempfile.TemporaryDirectory() as directory:
            renderer.save_weights(directory, 12)
            checkpoint_path = (
                Path(directory)
                / "photometric"
                / "iteration_12"
                / "photometric.pth"
            )
            self.assertTrue(checkpoint_path.is_file())
            restored = self.make_renderer()
            restored.load_weights(directory, 12)
            self.assertEqual(
                restored.capture()["photometric_version"],
                PHOTOMETRIC_PBR_VERSION,
            )
            torch.testing.assert_close(
                restored.get_all_light_dirs(),
                renderer.get_all_light_dirs(),
            )

        state = renderer.capture()
        state["photometric_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Expected"):
            self.make_renderer().restore(state)


if __name__ == "__main__":
    unittest.main()
