"""Constrained per-light PBR shading for dynamic Gaussian splats."""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.photometric_lambertian import (
    DIRECTION_CONVENTION,
    DirectionalLightModel,
    _as_timestep_tensor,
)


PHOTOMETRIC_PBR_VERSION = "perlight_pbr_v1"


def srgb_to_linear(value: torch.Tensor) -> torch.Tensor:
    """Convert bounded sRGB values to linear RGB without device-changing constants."""
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(value: torch.Tensor, clip: bool = True) -> torch.Tensor:
    """Convert non-negative linear RGB to sRGB."""
    value = value.clamp_min(0.0)
    result = torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055,
    )
    return result.clamp(0.0, 1.0) if clip else result


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def _orthonormal_tangents(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct stable tangent vectors for normalized [...,3] directions."""
    direction = F.normalize(direction, dim=-1)
    x_axis = torch.zeros_like(direction)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(direction)
    y_axis[..., 1] = 1.0
    reference = torch.where((direction[..., :1].abs() < 0.9), x_axis, y_axis)
    tangent = F.normalize(torch.cross(direction, reference, dim=-1), dim=-1)
    bitangent = F.normalize(torch.cross(direction, tangent, dim=-1), dim=-1)
    return tangent, bitangent


def _bounded_tangent(
    raw: torch.Tensor,
    tangent: torch.Tensor,
    bitangent: torch.Tensor,
    max_angle_radians: float,
) -> torch.Tensor:
    """Map unconstrained 2D values to a tangent displacement with bounded angle."""
    coordinates = torch.tanh(raw)
    magnitude = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
    coordinates = coordinates / magnitude.clamp_min(1.0)
    tangent_scale = math.tan(float(max_angle_radians))
    return tangent_scale * (
        coordinates[..., :1] * tangent + coordinates[..., 1:2] * bitangent
    )


class StructuredDirectionalLightModel(nn.Module):
    """Second-order Fourier trajectory with a bounded per-frame tangent residual."""

    def __init__(
        self,
        timesteps: Any,
        max_residual_angle_degrees: float = 10.0,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        timestep_tensor = _as_timestep_tensor(timesteps, device)
        self.register_buffer("timesteps", timestep_tensor)
        self.register_buffer("fourier_basis", self._make_basis(timestep_tensor))
        self.max_residual_angle_degrees = float(max_residual_angle_degrees)
        self.fourier_coefficients = nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device=device,
            )
        )
        self.raw_tangent_residual = nn.Parameter(
            torch.zeros((timestep_tensor.numel(), 2), dtype=torch.float32, device=device)
        )

    @staticmethod
    def _make_basis(timesteps: torch.Tensor) -> torch.Tensor:
        count = int(timesteps.numel())
        phase = torch.arange(count, dtype=timesteps.dtype, device=timesteps.device)
        phase = phase / float(max(count - 1, 1)) * (2.0 * torch.pi)
        return torch.stack(
            (
                torch.ones_like(phase),
                torch.cos(phase),
                torch.sin(phase),
                torch.cos(2.0 * phase),
                torch.sin(2.0 * phase),
            ),
            dim=-1,
        )

    @property
    def num_timesteps(self) -> int:
        return int(self.timesteps.numel())

    def initialize_from_directions(self, directions: torch.Tensor) -> None:
        directions = F.normalize(
            directions.to(
                device=self.fourier_coefficients.device,
                dtype=self.fourier_coefficients.dtype,
            ),
            dim=-1,
        )
        if directions.shape != (self.num_timesteps, 3):
            raise ValueError(
                "Initial light directions must have shape "
                f"({self.num_timesteps}, 3), got {tuple(directions.shape)}."
            )
        coefficients = torch.linalg.pinv(self.fourier_basis) @ directions
        with torch.no_grad():
            self.fourier_coefficients.copy_(coefficients)
            self.raw_tangent_residual.zero_()

    def get_base_light_dirs(self) -> torch.Tensor:
        raw = self.fourier_basis @ self.fourier_coefficients
        return F.normalize(raw, dim=-1)

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        base = self.get_base_light_dirs()
        tangent, bitangent = _orthonormal_tangents(base)
        displacement = _bounded_tangent(
            self.raw_tangent_residual,
            tangent,
            bitangent,
            math.radians(self.max_residual_angle_degrees),
        )
        return base + displacement

    def get_all_light_dirs(self) -> torch.Tensor:
        return F.normalize(self.get_all_raw_light_dirs(), dim=-1)

    def tangent_residual_angles(self) -> torch.Tensor:
        base = self.get_base_light_dirs()
        final = self.get_all_light_dirs()
        cosine = (base * final).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cosine))

    def timestep_index(self, frame_id: torch.Tensor) -> torch.Tensor:
        values = frame_id.detach().to(self.timesteps.device).float().reshape(-1)
        return torch.abs(values[:, None] - self.timesteps[None]).argmin(dim=1).long()

    def forward(self, frame_id: torch.Tensor) -> torch.Tensor:
        result = self.get_all_light_dirs()[self.timestep_index(frame_id)]
        return result[0] if result.shape[0] == 1 else result

    def first_order_smoothness_loss(self) -> torch.Tensor:
        directions = self.get_all_light_dirs()
        if directions.shape[0] < 2:
            return directions.new_zeros(())
        return (directions[1:] - directions[:-1]).pow(2).mean()

    def second_order_smoothness_loss(self) -> torch.Tensor:
        directions = self.get_all_light_dirs()
        if directions.shape[0] < 3:
            return directions.new_zeros(())
        acceleration = directions[2:] - 2.0 * directions[1:-1] + directions[:-2]
        return acceleration.pow(2).mean()


def _fresnel_schlick(
    f0: torch.Tensor,
    cosine: torch.Tensor,
) -> torch.Tensor:
    cosine = cosine.clamp(1e-4, 1.0)
    return f0 + (1.0 - f0) * (1.0 - cosine).pow(5.0)


def _ggx_specular_times_ndotl(
    normal: torch.Tensor,
    view: torch.Tensor,
    light: torch.Tensor,
    roughness: torch.Tensor,
    f0: torch.Tensor,
) -> torch.Tensor:
    """Evaluate GGX BRDF multiplied by N dot L."""
    half_vector = F.normalize(view + light, dim=-1)
    ndotv = (normal * view).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)
    ndotl = (normal * light).sum(dim=-1, keepdim=True)
    ndoth = (normal * half_vector).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)
    vdoth = (view * half_vector).sum(dim=-1, keepdim=True).clamp(1e-4, 1.0)

    alpha = roughness.clamp(0.08, 1.0).pow(2.0)
    alpha_squared = alpha.pow(2.0)
    denominator = (
        ndoth.pow(2.0) * (alpha_squared - 1.0) + 1.0
    ).pow(2.0)
    distribution = alpha_squared / (math.pi * denominator.clamp_min(1e-6))

    def smith_lambda(cosine: torch.Tensor) -> torch.Tensor:
        cosine_squared = cosine.pow(2.0).clamp_min(1e-6)
        tangent_squared = (1.0 - cosine_squared) / cosine_squared
        return 0.5 * (torch.sqrt(1.0 + alpha_squared * tangent_squared) - 1.0)

    geometry = 1.0 / (1.0 + smith_lambda(ndotv) + smith_lambda(ndotl.clamp(1e-4, 1.0)))
    fresnel = _fresnel_schlick(f0, vdoth)
    shaded = fresnel * distribution * geometry / (4.0 * ndotv)
    return torch.where(ndotl > 0.0, shaded, torch.zeros_like(shaded))


def _rough_diffuse_times_ndotl(
    albedo: torch.Tensor,
    normal: torch.Tensor,
    view: torch.Tensor,
    light: torch.Tensor,
    roughness: torch.Tensor,
) -> torch.Tensor:
    """Energy-compensated Frostbite/Disney rough diffuse times N dot L."""
    ndotl = (normal * light).sum(dim=-1, keepdim=True)
    ndotv = (normal * view).sum(dim=-1, keepdim=True)
    half_vector = F.normalize(view + light, dim=-1)
    ldoth = (light * half_vector).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    energy_bias = 0.5 * roughness
    energy_factor = 1.0 - (0.51 / 1.51) * roughness
    f90 = energy_bias + 2.0 * ldoth.pow(2.0) * roughness
    light_scatter = 1.0 + (f90 - 1.0) * (1.0 - ndotl.clamp(0.0, 1.0)).pow(5.0)
    view_scatter = 1.0 + (f90 - 1.0) * (1.0 - ndotv.clamp(0.0, 1.0)).pow(5.0)
    diffuse = (
        albedo / math.pi
        * light_scatter
        * view_scatter
        * energy_factor
        * ndotl.clamp_min(0.0)
    )
    return torch.where(
        (ndotl > 0.0) & (ndotv > 0.0),
        diffuse,
        torch.zeros_like(diffuse),
    )


def _environment_basis(normal: torch.Tensor) -> torch.Tensor:
    """Real second-order SH-like polynomial basis, with a constant DC term."""
    x, y, z = normal.unbind(dim=-1)
    return torch.stack(
        (
            torch.ones_like(x),
            x,
            y,
            z,
            x * y,
            y * z,
            3.0 * z * z - 1.0,
            x * z,
            x * x - y * y,
        ),
        dim=-1,
    )


class ShadingResidualMLP(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_dim: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class PhotometricPerLightPBRRenderer(nn.Module):
    """Compute constrained, linear-space PBR colors before Gaussian splatting."""

    def __init__(
        self,
        timesteps: Any,
        num_gaussians: int,
        normal_axis: str = "+z",
        device: str | torch.device = "cuda",
        light_samples_train: int = 4,
        light_samples_eval: int = 8,
        light_residual_angle_degrees: float = 10.0,
        normal_residual_angle_degrees: float = 10.0,
        residual_log_scale: float = 0.2,
        environment_init: float = 0.05,
        angular_radius_init_degrees: float = 2.0,
        angular_radius_max_degrees: float = 12.0,
        use_visibility: bool = True,
        visibility_backend: str = "local_knn",
        shadow_neighbors: int = 16,
        shadow_strength: float = 0.5,
        shadow_distance_factor: float = 6.0,
        use_residual: bool = True,
    ):
        super().__init__()
        if int(num_gaussians) <= 0:
            raise ValueError("num_gaussians must be positive.")
        if int(light_samples_train) <= 0 or int(light_samples_eval) <= 0:
            raise ValueError("PBR light sample counts must be positive.")
        self.normal_axis = str(normal_axis)
        self.light_mode = "learned_directional"
        self.num_gaussians = int(num_gaussians)
        self.light_samples_train = int(light_samples_train)
        self.light_samples_eval = int(light_samples_eval)
        self.normal_residual_angle_degrees = float(normal_residual_angle_degrees)
        self.residual_log_scale = float(residual_log_scale)
        self.angular_radius_max_degrees = float(angular_radius_max_degrees)
        self.use_visibility = bool(use_visibility)
        self.visibility_backend = str(visibility_backend).strip().lower()
        if self.visibility_backend not in {"local_knn", "bvh"}:
            raise ValueError(
                "PBR visibility backend must be 'local_knn' or 'bvh'."
            )
        self.shadow_neighbors = max(int(shadow_neighbors), 1)
        self.shadow_strength = float(shadow_strength)
        self.shadow_distance_factor = float(shadow_distance_factor)
        if self.shadow_strength < 0.0 or self.shadow_distance_factor <= 0.0:
            raise ValueError("PBR shadow strength/distance factor must be positive.")
        self.use_residual = bool(use_residual)
        self.light_model = StructuredDirectionalLightModel(
            timesteps,
            max_residual_angle_degrees=light_residual_angle_degrees,
            device=device,
        )
        self.raw_global_intensity = nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=device)
        )
        self.raw_exposure = nn.Parameter(
            torch.zeros(
                self.light_model.num_timesteps,
                dtype=torch.float32,
                device=device,
            )
        )
        self.raw_light_color = nn.Parameter(
            torch.zeros(3, dtype=torch.float32, device=device)
        )
        angular_fraction = (
            float(angular_radius_init_degrees) / self.angular_radius_max_degrees
        )
        angular_fraction = min(max(angular_fraction, 1e-4), 1.0 - 1e-4)
        self.raw_angular_radius = nn.Parameter(
            torch.tensor(
                math.log(angular_fraction / (1.0 - angular_fraction)),
                dtype=torch.float32,
                device=device,
            )
        )
        self.environment_dc = nn.Parameter(
            torch.full(
                (3,),
                _inverse_softplus(environment_init),
                dtype=torch.float32,
                device=device,
            )
        )
        self.environment_rest = nn.Parameter(
            torch.zeros((8, 3), dtype=torch.float32, device=device)
        )
        self.normal_residual_raw = nn.Parameter(
            torch.zeros((self.num_gaussians, 2), dtype=torch.float32, device=device)
        )
        self.residual_mlp = ShadingResidualMLP().to(device)
        self.optimizer = None
        self.initialization_metadata: dict[str, Any] = {}
        self._bvh_ready = False
        self._shadow_neighbor_indices: torch.Tensor | None = None

    @classmethod
    def from_args(
        cls,
        timesteps: Any,
        num_gaussians: int,
        args: Any,
        device: str | torch.device = "cuda",
    ):
        return cls(
            timesteps,
            num_gaussians=num_gaussians,
            normal_axis=getattr(args, "photometric_normal_axis", "+z"),
            device=device,
            light_samples_train=getattr(args, "photometric_pbr_light_samples_train", 4),
            light_samples_eval=getattr(args, "photometric_pbr_light_samples_eval", 8),
            light_residual_angle_degrees=getattr(
                args, "photometric_pbr_light_residual_angle_deg", 10.0
            ),
            normal_residual_angle_degrees=getattr(
                args, "photometric_pbr_normal_residual_angle_deg", 10.0
            ),
            residual_log_scale=getattr(args, "photometric_pbr_residual_log_scale", 0.2),
            environment_init=getattr(args, "photometric_pbr_environment_init", 0.05),
            angular_radius_init_degrees=getattr(
                args, "photometric_pbr_angular_radius_init_deg", 2.0
            ),
            angular_radius_max_degrees=getattr(
                args, "photometric_pbr_angular_radius_max_deg", 12.0
            ),
            use_visibility=bool(getattr(args, "photometric_pbr_visibility", 1)),
            visibility_backend=getattr(
                args, "photometric_pbr_visibility_backend", "local_knn"
            ),
            shadow_neighbors=getattr(
                args, "photometric_pbr_shadow_neighbors", 16
            ),
            shadow_strength=getattr(
                args, "photometric_pbr_shadow_strength", 0.5
            ),
            shadow_distance_factor=getattr(
                args, "photometric_pbr_shadow_distance_factor", 6.0
            ),
            use_residual=bool(getattr(args, "photometric_pbr_residual", 1)),
        )

    @property
    def learns_light(self) -> bool:
        return True

    @property
    def requires_bvh(self) -> bool:
        return self.use_visibility and self.visibility_backend == "bvh"

    @property
    def requires_local_visibility(self) -> bool:
        return self.use_visibility and self.visibility_backend == "local_knn"

    @property
    def bvh_ready(self) -> bool:
        return self._bvh_ready

    def mark_bvh_ready(self) -> None:
        self._bvh_ready = True

    def initialize_shadow_neighbors(self, position: torch.Tensor) -> None:
        """Build a fixed local topology; deformation updates neighbor positions."""
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise RuntimeError(
                "local_knn PBR visibility requires scipy.spatial.cKDTree."
            ) from exc
        points = position.detach().cpu().float().numpy()
        neighbor_count = min(self.shadow_neighbors + 1, points.shape[0])
        _, indices = cKDTree(points).query(
            points,
            k=neighbor_count,
            workers=1,
        )
        indices = np.asarray(indices)
        if indices.ndim == 1:
            indices = indices[:, None]
        # cKDTree returns the query point itself first.
        indices = indices[:, 1:]
        if indices.shape[1] == 0:
            indices = np.zeros((points.shape[0], 1), dtype=np.int64)
        self._shadow_neighbor_indices = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=position.device,
        )

    def compute_local_visibility(
        self,
        position: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        surface_to_light_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate soft cast shadows from a fixed local Gaussian KNN graph."""
        if self._shadow_neighbor_indices is None:
            self.initialize_shadow_neighbors(position)
        neighbor_indices = self._shadow_neighbor_indices
        neighbor_delta = position[neighbor_indices] - position[:, None, :]
        neighbor_radius = (
            2.0
            * scales[neighbor_indices].amax(dim=-1)
        ).clamp_min(1e-5)
        neighbor_opacity = opacity[neighbor_indices, 0].clamp(0.0, 1.0)

        along = torch.einsum(
            "nkd,sd->nks",
            neighbor_delta,
            surface_to_light_samples,
        )
        squared_distance = neighbor_delta.pow(2.0).sum(dim=-1, keepdim=True)
        perpendicular_squared = (
            squared_distance - along.pow(2.0)
        ).clamp_min(0.0)
        sigma = neighbor_radius[..., None]
        forward = torch.sigmoid(
            (along - 0.25 * sigma) / (0.25 * sigma + 1e-6)
        )
        transverse = torch.exp(
            -0.5 * perpendicular_squared / sigma.pow(2.0).clamp_min(1e-8)
        )
        distance_falloff = torch.exp(
            -along.clamp_min(0.0)
            / (self.shadow_distance_factor * sigma + 1e-6)
        )
        optical_depth = (
            neighbor_opacity[..., None]
            * forward
            * transverse
            * distance_falloff
        ).sum(dim=1)
        visibility = torch.exp(-self.shadow_strength * optical_depth)
        return visibility.clamp(0.0, 1.0).unsqueeze(-1)

    def training_setup(self, args: Any) -> None:
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.light_model.parameters()),
                    "lr": 0.0,
                    "name": "photometric_light",
                },
                {
                    "params": [
                        self.raw_global_intensity,
                        self.raw_exposure,
                        self.raw_light_color,
                        self.raw_angular_radius,
                    ],
                    "lr": 0.0,
                    "name": "photometric_pbr_exposure",
                },
                {
                    "params": [self.environment_dc, self.environment_rest],
                    "lr": 0.0,
                    "name": "photometric_pbr_environment",
                },
                {
                    "params": [self.normal_residual_raw],
                    "lr": 0.0,
                    "name": "photometric_pbr_normal",
                },
                {
                    "params": list(self.residual_mlp.parameters()),
                    "lr": 0.0,
                    "name": "photometric_pbr_residual",
                },
            ],
            lr=0.0,
            eps=1e-15,
        )
        self.set_learning_rates(args, active=False, light_lr=0.0)

    def set_learning_rates(self, args: Any, active: bool, light_lr: float) -> None:
        if self.optimizer is None:
            return
        configured = {
            "photometric_light": float(light_lr),
            "photometric_pbr_exposure": float(
                getattr(args, "photometric_pbr_exposure_lr", 1e-3)
            ),
            "photometric_pbr_environment": float(
                getattr(args, "photometric_pbr_environment_lr", 1e-3)
            ),
            "photometric_pbr_normal": float(
                getattr(args, "photometric_pbr_normal_lr", 1e-4)
            ),
            "photometric_pbr_residual": float(
                getattr(args, "photometric_pbr_residual_lr", 1e-4)
            ),
        }
        for group in self.optimizer.param_groups:
            group["lr"] = configured[group["name"]] if active else 0.0
            if group["name"] == "photometric_pbr_residual" and not self.use_residual:
                group["lr"] = 0.0

    def set_light_lr(self, learning_rate: float) -> None:
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] == "photometric_light":
                group["lr"] = float(learning_rate)

    def _initialize_structured_from_legacy(self, legacy: DirectionalLightModel) -> None:
        self.light_model.initialize_from_directions(legacy.get_all_light_dirs())

    def initialize_camera_back_ellipse(
        self,
        camera_right: torch.Tensor,
        camera_up: torch.Tensor,
        camera_forward: torch.Tensor,
        horizontal_radius: float,
        vertical_radius: float,
        back_offset: float,
        phase: float,
        direction_sign: int,
        span: float,
    ) -> None:
        legacy = DirectionalLightModel(
            self.light_model.timesteps,
            device=self.light_model.timesteps.device,
        )
        legacy.initialize_camera_back_ellipse(
            camera_right,
            camera_up,
            camera_forward,
            horizontal_radius,
            vertical_radius,
            back_offset,
            phase,
            direction_sign,
            span,
        )
        self._initialize_structured_from_legacy(legacy)
        self.initialization_metadata = {
            "type": "camera_back_ellipse_fourier",
            "initialization_version": "pbr_v1",
            "direction_convention": DIRECTION_CONVENTION,
            "fourier_order": 2,
        }

    def initialize_camera_pose_xz_ellipse(
        self,
        camera_center: torch.Tensor,
        object_center: torch.Tensor,
        axis_ratio: float,
    ) -> None:
        legacy = DirectionalLightModel(
            self.light_model.timesteps,
            device=self.light_model.timesteps.device,
        )
        trajectory = legacy.initialize_camera_pose_xz_ellipse(
            camera_center,
            object_center,
            axis_ratio,
        )
        self._initialize_structured_from_legacy(legacy)
        self.initialization_metadata = {
            "type": "camera_pose_xz_ellipse_fourier",
            "initialization_version": "pbr_v1",
            "direction_convention": DIRECTION_CONVENTION,
            "fourier_order": 2,
            "camera_center": trajectory["camera_center"].cpu().tolist(),
            "object_center": trajectory["object_center"].cpu().tolist(),
            "major_radius": trajectory["major_radius"],
            "minor_radius": trajectory["minor_radius"],
            "axis_ratio": float(axis_ratio),
        }

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_raw_light_dirs()

    def get_all_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_light_dirs()

    def light_smoothness_loss(self) -> torch.Tensor:
        return self.light_model.first_order_smoothness_loss()

    def light_second_order_smoothness_loss(self) -> torch.Tensor:
        return self.light_model.second_order_smoothness_loss()

    def angular_radius_radians(self) -> torch.Tensor:
        degrees = (
            torch.sigmoid(self.raw_angular_radius)
            * self.angular_radius_max_degrees
        )
        return torch.deg2rad(degrees)

    def exposure_values(self) -> torch.Tensor:
        centered = self.raw_exposure - self.raw_exposure.mean()
        return 0.1 * torch.tanh(centered)

    def light_color(self) -> torch.Tensor:
        color = F.softplus(self.raw_light_color)
        return color / color.mean().clamp_min(1e-6)

    def light_intensity(self, timestep_idx: torch.Tensor | int) -> torch.Tensor:
        return torch.exp(
            self.raw_global_intensity + self.exposure_values()[timestep_idx]
        )

    def sample_surface_to_light_dirs(
        self,
        frame_id: torch.Tensor,
        sample_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_idx = self.light_model.timestep_index(frame_id)[0]
        ray_dir = self.light_model(frame_id)
        if ray_dir.ndim == 2:
            ray_dir = ray_dir[0]
        center = F.normalize(-ray_dir, dim=-1)
        count = int(
            sample_count
            if sample_count is not None
            else (self.light_samples_train if self.training else self.light_samples_eval)
        )
        tangent, bitangent = _orthonormal_tangents(center[None])
        tangent, bitangent = tangent[0], bitangent[0]
        index = torch.arange(count, device=center.device, dtype=center.dtype)
        radius = torch.sqrt((index + 0.5) / float(count))
        angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
        disk = (
            torch.cos(angle)[:, None] * tangent[None]
            + torch.sin(angle)[:, None] * bitangent[None]
        )
        offset = torch.tan(self.angular_radius_radians()) * radius[:, None] * disk
        samples = F.normalize(center[None] + offset, dim=-1)
        return samples, timestep_idx

    def perturb_normals(
        self,
        normal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normal.shape != (self.num_gaussians, 3):
            raise ValueError(
                "PBR normal count must match the renderer checkpoint: "
                f"{tuple(normal.shape)} vs ({self.num_gaussians}, 3)."
            )
        base = F.normalize(normal, dim=-1)
        tangent, bitangent = _orthonormal_tangents(base)
        displacement = _bounded_tangent(
            self.normal_residual_raw,
            tangent,
            bitangent,
            math.radians(self.normal_residual_angle_degrees),
        )
        perturbed = F.normalize(base + displacement, dim=-1)
        cosine = (base * perturbed).sum(dim=-1).clamp(-1.0, 1.0)
        angle_degrees = torch.rad2deg(torch.acos(cosine)).unsqueeze(-1)
        return perturbed, angle_degrees

    def environment_irradiance(self, normal: torch.Tensor) -> torch.Tensor:
        basis = _environment_basis(normal)
        raw = self.environment_dc + basis[..., 1:] @ self.environment_rest
        return F.softplus(raw)

    def forward(
        self,
        albedo_srgb: torch.Tensor,
        normal: torch.Tensor,
        frame_id: torch.Tensor,
        position: torch.Tensor,
        camera_center: torch.Tensor,
        roughness: torch.Tensor,
        surface_to_light_samples: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if position.shape != normal.shape:
            raise ValueError("PBR position and normal must have matching [N,3] shapes.")
        if roughness.shape != (normal.shape[0], 1):
            raise ValueError("PBR roughness must have shape [N,1].")
        normal, normal_residual_angle = self.perturb_normals(normal)
        if surface_to_light_samples is None:
            surface_to_light_samples, timestep_idx = self.sample_surface_to_light_dirs(
                frame_id
            )
        else:
            timestep_idx = self.light_model.timestep_index(frame_id)[0]
        sample_count = surface_to_light_samples.shape[0]
        light = surface_to_light_samples[None].expand(normal.shape[0], -1, -1)
        normal_samples = normal[:, None, :].expand_as(light)
        view = F.normalize(
            camera_center.to(position)[None] - position,
            dim=-1,
        )
        view_samples = view[:, None, :].expand_as(light)
        roughness = roughness.clamp(0.08, 1.0)
        roughness_samples = roughness[:, None, :].expand(-1, sample_count, -1)
        albedo_linear = srgb_to_linear(albedo_srgb)
        albedo_samples = albedo_linear[:, None, :].expand(-1, sample_count, -1)
        if visibility is None:
            visibility = torch.ones(
                (normal.shape[0], sample_count, 1),
                dtype=normal.dtype,
                device=normal.device,
            )
        if visibility.shape != (normal.shape[0], sample_count, 1):
            raise ValueError(
                "PBR visibility must have shape "
                f"({normal.shape[0]}, {sample_count}, 1), got {tuple(visibility.shape)}."
            )
        visibility = visibility.clamp(0.0, 1.0)

        diffuse_samples = _rough_diffuse_times_ndotl(
            albedo_samples,
            normal_samples,
            view_samples,
            light,
            roughness_samples,
        )
        f0 = torch.full_like(albedo_samples, 0.04)
        specular_samples = _ggx_specular_times_ndotl(
            normal_samples,
            view_samples,
            light,
            roughness_samples,
            f0,
        )
        light_radiance = (
            self.light_intensity(timestep_idx)
            * self.light_color()[None, None, :]
        )
        direct_diffuse = (
            diffuse_samples * visibility * light_radiance
        ).mean(dim=1)
        direct_specular = (
            specular_samples * visibility * light_radiance
        ).mean(dim=1)
        environment = albedo_linear * self.environment_irradiance(normal)
        physical_linear = direct_diffuse + direct_specular + environment

        center_light = F.normalize(surface_to_light_samples.mean(dim=0), dim=-1)
        half_vector = F.normalize(view + center_light[None], dim=-1)
        ndotl = (normal * center_light[None]).sum(dim=-1, keepdim=True)
        ndotv = (normal * view).sum(dim=-1, keepdim=True)
        ndoth = (normal * half_vector).sum(dim=-1, keepdim=True)
        visibility_mean = visibility.mean(dim=1)
        residual_inputs = torch.cat(
            (
                albedo_linear,
                normal,
                view,
                center_light[None].expand_as(normal),
                ndotl,
                ndotv,
                ndoth,
                roughness,
                visibility_mean,
            ),
            dim=-1,
        )
        residual_raw = self.residual_mlp(residual_inputs)
        residual_log = (
            self.residual_log_scale * torch.tanh(residual_raw)
            if self.use_residual
            else torch.zeros_like(residual_raw)
        )
        residual_multiplier = torch.exp(residual_log)
        color_linear = physical_linear * residual_multiplier
        return {
            "color_linear": color_linear,
            "color": linear_to_srgb(color_linear),
            "albedo": albedo_srgb,
            "albedo_linear": albedo_linear,
            "normal": normal,
            "normal_residual_angle": normal_residual_angle,
            "light_dir": -center_light,
            "ray_dir": -center_light,
            "surface_to_light_dir": center_light,
            "surface_to_light_samples": surface_to_light_samples,
            "ndotl": ndotl,
            "shading": ndotl.clamp_min(0.0),
            "direct_diffuse_linear": direct_diffuse,
            "direct_specular_linear": direct_specular,
            "environment_linear": environment,
            "visibility": visibility_mean,
            "roughness": roughness,
            "residual_raw": residual_raw,
            "residual_multiplier": residual_multiplier,
            "timestep_idx": timestep_idx,
        }

    def regularization_losses(self, roughness: torch.Tensor) -> dict[str, torch.Tensor]:
        residual_parameters = torch.cat(
            [parameter.reshape(-1) for parameter in self.residual_mlp.parameters()]
        )
        exposure = self.exposure_values()
        environment = F.softplus(self.environment_dc)
        return {
            "light_smooth1": self.light_smoothness_loss(),
            "light_smooth2": self.light_second_order_smoothness_loss(),
            "exposure": exposure.pow(2.0).mean(),
            "normal_residual": self.normal_residual_raw.pow(2.0).mean(),
            "roughness_prior": (roughness - 0.75).pow(2.0).mean(),
            "environment_energy": environment.pow(2.0).mean(),
            "residual": residual_parameters.pow(2.0).mean(),
        }

    def capture(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "timesteps": self.light_model.timesteps.detach().cpu(),
            "photometric_version": PHOTOMETRIC_PBR_VERSION,
            "config": {
                "normal_axis": self.normal_axis,
                "light_mode": self.light_mode,
                "direction_convention": DIRECTION_CONVENTION,
                "num_gaussians": self.num_gaussians,
                "light_samples_train": self.light_samples_train,
                "light_samples_eval": self.light_samples_eval,
                "light_residual_angle_degrees": (
                    self.light_model.max_residual_angle_degrees
                ),
                "normal_residual_angle_degrees": self.normal_residual_angle_degrees,
                "residual_log_scale": self.residual_log_scale,
                "angular_radius_max_degrees": self.angular_radius_max_degrees,
                "use_visibility": self.use_visibility,
                "visibility_backend": self.visibility_backend,
                "shadow_neighbors": self.shadow_neighbors,
                "shadow_strength": self.shadow_strength,
                "shadow_distance_factor": self.shadow_distance_factor,
                "use_residual": self.use_residual,
            },
            "initialization": self.initialization_metadata,
        }

    def restore(self, state: dict[str, Any]) -> None:
        version = state.get("photometric_version")
        if version != PHOTOMETRIC_PBR_VERSION:
            raise ValueError(
                f"Expected {PHOTOMETRIC_PBR_VERSION!r}, got {version!r}."
            )
        config = state.get("config", {})
        checkpoint_count = int(config.get("num_gaussians", -1))
        if checkpoint_count != self.num_gaussians:
            raise ValueError(
                "PBR checkpoint Gaussian count mismatch: "
                f"{checkpoint_count} vs {self.num_gaussians}."
            )
        self.normal_axis = config.get("normal_axis", self.normal_axis)
        self.light_samples_train = int(
            config.get("light_samples_train", self.light_samples_train)
        )
        self.light_samples_eval = int(
            config.get("light_samples_eval", self.light_samples_eval)
        )
        self.light_model.max_residual_angle_degrees = float(
            config.get(
                "light_residual_angle_degrees",
                self.light_model.max_residual_angle_degrees,
            )
        )
        self.normal_residual_angle_degrees = float(
            config.get(
                "normal_residual_angle_degrees",
                self.normal_residual_angle_degrees,
            )
        )
        self.residual_log_scale = float(
            config.get("residual_log_scale", self.residual_log_scale)
        )
        self.angular_radius_max_degrees = float(
            config.get(
                "angular_radius_max_degrees",
                self.angular_radius_max_degrees,
            )
        )
        self.use_visibility = bool(config.get("use_visibility", self.use_visibility))
        self.visibility_backend = str(
            config.get("visibility_backend", self.visibility_backend)
        ).strip().lower()
        if self.visibility_backend not in {"local_knn", "bvh"}:
            raise ValueError(
                f"Unsupported checkpoint visibility backend: "
                f"{self.visibility_backend!r}."
            )
        self.shadow_neighbors = int(
            config.get("shadow_neighbors", self.shadow_neighbors)
        )
        self.shadow_strength = float(
            config.get("shadow_strength", self.shadow_strength)
        )
        self.shadow_distance_factor = float(
            config.get(
                "shadow_distance_factor",
                self.shadow_distance_factor,
            )
        )
        self.use_residual = bool(config.get("use_residual", self.use_residual))
        self.initialization_metadata = dict(state.get("initialization", {}))
        self.load_state_dict(state["state_dict"], strict=True)
        self._bvh_ready = False
        self._shadow_neighbor_indices = None

    def light_trajectory_dict(self) -> dict[str, Any]:
        directions = self.get_all_light_dirs().detach().cpu().float()
        times = self.light_model.timesteps.detach().cpu().float()
        exposure = self.exposure_values().detach().cpu().float()
        return {
            "photometric_version": PHOTOMETRIC_PBR_VERSION,
            "light_mode": self.light_mode,
            "direction_convention": DIRECTION_CONVENTION,
            "initialization": self.initialization_metadata,
            "global_intensity": float(torch.exp(self.raw_global_intensity).detach().cpu()),
            "light_color": self.light_color().detach().cpu().tolist(),
            "angular_radius_degrees": float(
                torch.rad2deg(self.angular_radius_radians()).detach().cpu()
            ),
            "frames": [
                {
                    "index": index,
                    "fid": float(times[index]),
                    "direction": directions[index].tolist(),
                    "ray_direction_light_to_surface": directions[index].tolist(),
                    "exposure_log_delta": float(exposure[index]),
                }
                for index in range(directions.shape[0])
            ],
        }

    def save_weights(self, model_path: str, iteration: int) -> None:
        output_dir = os.path.join(model_path, "photometric", f"iteration_{iteration}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.capture(), os.path.join(output_dir, "photometric.pth"))
        directions = self.get_all_light_dirs().detach().cpu().float().numpy()
        np.save(os.path.join(output_dir, "light_dirs.npy"), directions)
        with open(os.path.join(output_dir, "light_dirs.json"), "w", encoding="utf-8") as handle:
            json.dump(self.light_trajectory_dict(), handle, indent=2)

    def load_weights(self, model_path: str, iteration: int) -> None:
        checkpoint = os.path.join(
            model_path,
            "photometric",
            f"iteration_{iteration}",
            "photometric.pth",
        )
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Missing photometric checkpoint: {checkpoint}")
        self.restore(
            torch.load(checkpoint, map_location=self.light_model.timesteps.device)
        )
