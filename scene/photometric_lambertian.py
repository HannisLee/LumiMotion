"""Per-frame directional-light Lambertian rendering for Stage 1."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import build_rotation


PHOTOMETRIC_VERSION = "stage1_perlight_v3_camera_facing_gt_point"
DIRECTION_CONVENTION = "light_to_surface"
LIGHT_MODES = {"learned_directional", "gt_point"}


def _as_timestep_tensor(timesteps: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(timesteps, dict):
        ordered = [fid for fid, _ in sorted(timesteps.items(), key=lambda item: item[1])]
        values = [float(fid.detach().reshape(-1)[0].item()) for fid in ordered]
        out = torch.tensor(values, dtype=torch.float32, device=device)
    else:
        out = torch.as_tensor(timesteps, dtype=torch.float32, device=device).flatten()
    if out.numel() == 0:
        out = torch.zeros(1, dtype=torch.float32, device=device)
    return out


def parse_axis(axis: str | torch.Tensor, device: str | torch.device) -> torch.Tensor:
    if torch.is_tensor(axis):
        out = axis.to(device=device, dtype=torch.float32).flatten()
    else:
        named = {
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "+z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        text = str(axis).strip()
        if text in named:
            out = torch.tensor(named[text], dtype=torch.float32, device=device)
        else:
            values = [float(part) for part in text.replace(",", " ").split()]
            if len(values) != 3:
                raise ValueError(f"Normal axis must contain three values, got {axis!r}.")
            out = torch.tensor(values, dtype=torch.float32, device=device)
    if out.numel() != 3:
        raise ValueError(f"Normal axis must have three elements, got {tuple(out.shape)}.")
    return F.normalize(out, dim=0)


def get_gaussian_normal(rotation_t: torch.Tensor, normal_axis: str = "+z") -> torch.Tensor:
    """Return the selected local Gaussian axis in world space."""
    if rotation_t.ndim >= 3 and tuple(rotation_t.shape[-2:]) == (3, 3):
        rotation_matrix = rotation_t
    elif rotation_t.shape[-1] == 4:
        rotation_matrix = build_rotation(rotation_t.reshape(-1, 4)).reshape(
            *rotation_t.shape[:-1], 3, 3
        )
    else:
        raise ValueError(
            "rotation_t must contain quaternions [...,4] or matrices [...,3,3], "
            f"got {tuple(rotation_t.shape)}."
        )
    axis = parse_axis(normal_axis, rotation_matrix.device)
    return F.normalize(torch.matmul(rotation_matrix, axis.view(3, 1)).squeeze(-1), dim=-1)


def orient_normal_toward_camera(
    normal: torch.Tensor,
    position: torch.Tensor,
    camera_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the 2D Gaussian normal sign using the rasterizer's view-facing rule."""
    if normal.shape != position.shape or normal.shape[-1] != 3:
        raise ValueError(
            "normal and position must have matching [...,3] shapes, got "
            f"{tuple(normal.shape)} and {tuple(position.shape)}."
        )
    center = camera_center.to(device=normal.device, dtype=normal.dtype)
    if center.numel() != 3:
        raise ValueError(f"camera_center must contain three values, got {tuple(center.shape)}.")
    center = center.reshape(*([1] * (normal.ndim - 1)), 3)
    normal = F.normalize(normal, dim=-1)
    camera_facing = (normal * (center - position)).sum(dim=-1, keepdim=True) >= 0
    oriented = normal * torch.where(camera_facing, 1.0, -1.0)
    return oriented, camera_facing


class DirectionalLightModel(nn.Module):
    """A freely learnable light-to-surface unit ray for every scene timestep."""

    def __init__(self, timesteps: Any, device: str | torch.device = "cuda"):
        super().__init__()
        timestep_tensor = _as_timestep_tensor(timesteps, device)
        self.register_buffer("timesteps", timestep_tensor)
        init = torch.tensor((0.0, 0.0, 1.0), device=device).repeat(timestep_tensor.numel(), 1)
        self._raw_light_dir_table = nn.Parameter(init)

    @property
    def num_timesteps(self) -> int:
        return int(self.timesteps.numel())

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
        device = self.timesteps.device
        dtype = self._raw_light_dir_table.dtype
        right = F.normalize(camera_right.to(device=device, dtype=dtype), dim=0)
        up = F.normalize(camera_up.to(device=device, dtype=dtype), dim=0)
        toward_object = F.normalize(camera_forward.to(device=device, dtype=dtype), dim=0)
        denominator = max(self.num_timesteps - 1, 1)
        time = torch.arange(self.num_timesteps, device=device, dtype=dtype) / float(denominator)
        sign = 1.0 if int(direction_sign) >= 0 else -1.0
        theta = float(phase) + sign * float(span) * time
        # The light position is behind the camera relative to the object:
        #   object_to_light = a*cos(theta)*right + b*sin(theta)*up - back*forward.
        # Store the physical propagation direction from that light to the object.
        raw = (
            -float(horizontal_radius) * torch.cos(theta)[:, None] * right[None]
            - float(vertical_radius) * torch.sin(theta)[:, None] * up[None]
            + float(back_offset) * toward_object[None]
        )
        with torch.no_grad():
            self._raw_light_dir_table.copy_(F.normalize(raw, dim=-1))

    def initialize_camera_pose_xz_ellipse(
        self,
        camera_center: torch.Tensor,
        object_center: torch.Tensor,
        axis_ratio: float,
    ) -> dict[str, torch.Tensor | float]:
        """Initialize rays from a closed world-XZ light-position ellipse."""
        device = self.timesteps.device
        dtype = self._raw_light_dir_table.dtype
        camera = camera_center.to(device=device, dtype=dtype).flatten()
        center = object_center.to(device=device, dtype=dtype).flatten()
        if camera.numel() != 3 or center.numel() != 3:
            raise ValueError("Camera and object centers must each contain three values.")
        ratio = float(axis_ratio)
        if ratio <= 0.0:
            raise ValueError("V2 ellipse axis ratio must be positive.")

        radial_xz = camera - center
        radial_xz = radial_xz.clone()
        radial_xz[1] = 0.0
        major_radius = torch.linalg.vector_norm(radial_xz)
        eps = torch.finfo(dtype).eps * 16.0
        if major_radius.item() <= eps:
            raise ValueError(
                "V2 ellipse requires a non-zero camera-to-object distance in world XZ."
            )
        major_axis = radial_xz / major_radius
        minor_axis = torch.stack(
            (-major_axis[2], major_axis.new_zeros(()), major_axis[0])
        )
        if minor_axis[2].item() < 0.0:
            minor_axis = -minor_axis
        if minor_axis[2].abs().item() <= eps:
            raise ValueError(
                "V2 ellipse cannot initially rise in world Z when the camera-to-object "
                "XZ vector is parallel to world Z."
            )
        minor_radius = major_radius * ratio

        denominator = max(self.num_timesteps - 1, 1)
        time = torch.arange(self.num_timesteps, device=device, dtype=dtype) / float(denominator)
        theta = (2.0 * torch.pi) * time
        plane_center = center.clone()
        plane_center[1] = camera[1]
        positions = (
            plane_center[None]
            + major_radius * torch.cos(theta)[:, None] * major_axis[None]
            + minor_radius * torch.sin(theta)[:, None] * minor_axis[None]
        )
        raw = center[None] - positions
        raw_norm = torch.linalg.vector_norm(raw, dim=-1)
        if raw_norm.min().item() <= eps:
            raise ValueError("V2 ellipse places a virtual light at the object center.")
        directions = F.normalize(raw, dim=-1)
        with torch.no_grad():
            self._raw_light_dir_table.copy_(directions)
        return {
            "camera_center": camera.detach().clone(),
            "object_center": center.detach().clone(),
            "plane_center": plane_center.detach().clone(),
            "major_axis": major_axis.detach().clone(),
            "minor_axis": minor_axis.detach().clone(),
            "major_radius": float(major_radius.item()),
            "minor_radius": float(minor_radius.item()),
            "positions": positions.detach().clone(),
        }

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        return self._raw_light_dir_table

    def get_all_light_dirs(self) -> torch.Tensor:
        return F.normalize(self._raw_light_dir_table, dim=-1)

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


class PhotometricLambertianRenderer(nn.Module):
    """Compute per-Gaussian diffuse colors before rasterization."""

    def __init__(
        self,
        timesteps: Any,
        normal_axis: str = "+z",
        light_mode: str = "learned_directional",
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.normal_axis = normal_axis
        self.light_mode = str(light_mode).strip().lower()
        if self.light_mode not in LIGHT_MODES:
            raise ValueError(
                f"light_mode must be one of {sorted(LIGHT_MODES)}, got {light_mode!r}."
            )
        self.light_model = DirectionalLightModel(timesteps, device=device)
        self.register_buffer(
            "gt_light_positions",
            torch.empty((0, 3), dtype=torch.float32, device=device),
        )
        self.optimizer = None
        self.initialization_metadata: dict[str, Any] = {}

    @classmethod
    def from_args(cls, timesteps: Any, args: Any, device: str | torch.device = "cuda"):
        return cls(
            timesteps,
            normal_axis=getattr(args, "photometric_normal_axis", "+z"),
            light_mode=getattr(args, "photometric_light_mode", "learned_directional"),
            device=device,
        )

    @property
    def learns_light(self) -> bool:
        return self.light_mode == "learned_directional"

    def training_setup(self, args: Any) -> None:
        if not self.learns_light:
            self.optimizer = None
            return
        self.optimizer = torch.optim.Adam(
            [{
                "params": self.light_model.parameters(),
                "lr": 0.0,
                "name": "photometric_light",
            }],
            lr=0.0,
            eps=1e-15,
        )

    def set_light_lr(self, learning_rate: float) -> None:
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = float(learning_rate) if self.learns_light else 0.0

    def initialize_gt_point_lights(
        self,
        lights_path: str,
        reference_center: torch.Tensor,
    ) -> None:
        if not lights_path:
            raise ValueError("gt_point light mode requires photometric_gt_lights_path.")
        with open(lights_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        ordered = [payload[key] for key in sorted(payload, key=lambda value: int(value))]
        positions = torch.tensor(
            [entry["light_pos_world"] for entry in ordered],
            dtype=self.light_model.timesteps.dtype,
            device=self.light_model.timesteps.device,
        )
        if positions.shape != (self.light_model.num_timesteps, 3):
            raise ValueError(
                "GT light count must match scene timesteps: "
                f"got {tuple(positions.shape)}, expected "
                f"({self.light_model.num_timesteps}, 3)."
            )
        center = reference_center.to(
            device=positions.device,
            dtype=positions.dtype,
        ).flatten()
        if center.numel() != 3:
            raise ValueError("reference_center must contain three values.")
        reference_rays = F.normalize(center[None] - positions, dim=-1)
        self.gt_light_positions = positions
        with torch.no_grad():
            self.light_model._raw_light_dir_table.copy_(reference_rays)
        self.light_model._raw_light_dir_table.requires_grad_(False)
        self.initialization_metadata = {
            "type": "gt_point_lights",
            "light_mode": self.light_mode,
            "direction_convention": DIRECTION_CONVENTION,
            "lights_path": os.path.abspath(lights_path),
            "reference_center": center.detach().cpu().tolist(),
            "source_light_type": "area_center_as_point",
            "uses_distance_attenuation": False,
        }

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
        self.light_model.initialize_camera_back_ellipse(
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
        self.initialization_metadata = {
            "type": "camera_back_ellipse",
            "initialization_version": "v1",
            "direction_convention": DIRECTION_CONVENTION,
            "horizontal_radius": float(horizontal_radius),
            "vertical_radius": float(vertical_radius),
            "back_offset": float(back_offset),
            "phase": float(phase),
            "direction_sign": 1 if int(direction_sign) >= 0 else -1,
            "span": float(span),
        }

    def initialize_camera_pose_xz_ellipse(
        self,
        camera_center: torch.Tensor,
        object_center: torch.Tensor,
        axis_ratio: float,
    ) -> None:
        trajectory = self.light_model.initialize_camera_pose_xz_ellipse(
            camera_center,
            object_center,
            axis_ratio,
        )
        self.initialization_metadata = {
            "type": "camera_pose_xz_ellipse",
            "initialization_version": "v2",
            "direction_convention": DIRECTION_CONVENTION,
            "camera_center": trajectory["camera_center"].cpu().tolist(),
            "object_center": trajectory["object_center"].cpu().tolist(),
            "plane_center": trajectory["plane_center"].cpu().tolist(),
            "major_axis": trajectory["major_axis"].cpu().tolist(),
            "minor_axis": trajectory["minor_axis"].cpu().tolist(),
            "major_radius": trajectory["major_radius"],
            "minor_radius": trajectory["minor_radius"],
            "axis_ratio": float(axis_ratio),
            "span": float(2.0 * np.pi),
        }

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_raw_light_dirs()

    def get_all_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_light_dirs()

    def light_smoothness_loss(self) -> torch.Tensor:
        if not self.learns_light:
            return self.light_model.timesteps.new_zeros(())
        return self.light_model.first_order_smoothness_loss()

    def forward(
        self,
        albedo: torch.Tensor,
        normal: torch.Tensor,
        frame_id: torch.Tensor,
        position: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        timestep_idx = self.light_model.timestep_index(frame_id)[0]
        normal = F.normalize(normal, dim=-1)
        if self.light_mode == "gt_point":
            if position is None or position.shape != normal.shape:
                raise ValueError(
                    "gt_point light mode requires position with the same shape as normal."
                )
            if self.gt_light_positions.shape != (self.light_model.num_timesteps, 3):
                raise RuntimeError("GT point lights have not been initialized.")
            light_position = self.gt_light_positions[timestep_idx]
            surface_to_light_dir = F.normalize(light_position[None] - position, dim=-1)
            ray_dir = -surface_to_light_dir
        else:
            ray_dir = self.light_model(frame_id)
            if ray_dir.ndim == 2:
                ray_dir = ray_dir[0]
            ray_dir = F.normalize(ray_dir, dim=-1)
            surface_to_light_dir = -ray_dir
        ndotl = (normal * surface_to_light_dir).sum(dim=-1, keepdim=True)
        shading = ndotl.clamp_min(0.0)
        return {
            "color": (albedo * shading).clamp(0.0, 1.0),
            "albedo": albedo,
            "normal": normal,
            # Keep light_dir as an alias for callers, but its v2 convention is
            # explicitly the physical light-to-surface propagation direction.
            "light_dir": ray_dir,
            "ray_dir": ray_dir,
            "surface_to_light_dir": surface_to_light_dir,
            "ndotl": ndotl,
            "shading": shading,
            "timestep_idx": timestep_idx,
        }

    def capture(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "timesteps": self.light_model.timesteps.detach().cpu(),
            "photometric_version": PHOTOMETRIC_VERSION,
            "config": {
                "normal_axis": self.normal_axis,
                "light_param": (
                    "fixed_gt_point_position"
                    if self.light_mode == "gt_point"
                    else "per_frame"
                ),
                "light_mode": self.light_mode,
                "direction_convention": DIRECTION_CONVENTION,
            },
            "initialization": self.initialization_metadata,
        }

    def restore(self, state: dict[str, Any]) -> None:
        config = state.get("config", {})
        self.normal_axis = config.get("normal_axis", self.normal_axis)
        self.light_mode = config.get("light_mode", "learned_directional")
        if self.light_mode not in LIGHT_MODES:
            raise ValueError(f"Unsupported checkpoint light mode: {self.light_mode!r}.")
        state_dict = dict(state.get("state_dict", state))
        if "raw_light_dir" in state_dict and "light_model._raw_light_dir_table" not in state_dict:
            state_dict["light_model._raw_light_dir_table"] = state_dict.pop("raw_light_dir")
        state_dict.pop("raw_light_rgb", None)
        direction_convention = config.get("direction_convention")
        if direction_convention != DIRECTION_CONVENTION:
            raw_key = "light_model._raw_light_dir_table"
            if raw_key in state_dict:
                # v1 stored surface-to-light vectors. Negating them preserves
                # the rendered Lambertian shading under the v2 ray convention.
                state_dict[raw_key] = -state_dict[raw_key]
        gt_positions = state_dict.get("gt_light_positions")
        if gt_positions is not None:
            self.gt_light_positions = gt_positions.to(
                device=self.light_model.timesteps.device,
                dtype=self.light_model.timesteps.dtype,
            )
        self.load_state_dict(state_dict, strict=False)
        self.light_model._raw_light_dir_table.requires_grad_(self.learns_light)
        self.initialization_metadata = dict(state.get("initialization", {}))
        if "initialization_version" not in self.initialization_metadata:
            init_type = self.initialization_metadata.get("type")
            self.initialization_metadata["initialization_version"] = (
                "v2" if init_type == "camera_pose_xz_ellipse" else "v1"
            )
        self.initialization_metadata["direction_convention"] = DIRECTION_CONVENTION

    def light_trajectory_dict(self) -> dict[str, Any]:
        raw = self.get_all_raw_light_dirs().detach().cpu().float()
        directions = self.get_all_light_dirs().detach().cpu().float()
        times = self.light_model.timesteps.detach().cpu().float()
        return {
            "photometric_version": PHOTOMETRIC_VERSION,
            "light_mode": self.light_mode,
            "direction_convention": DIRECTION_CONVENTION,
            "initialization": self.initialization_metadata,
            "frames": [
                {
                    "index": index,
                    "fid": float(times[index]),
                    "raw": raw[index].tolist(),
                    "direction": directions[index].tolist(),
                    "ray_direction_light_to_surface": directions[index].tolist(),
                    **(
                        {
                            "light_position_world": self.gt_light_positions[index]
                            .detach()
                            .cpu()
                            .tolist()
                        }
                        if self.light_mode == "gt_point"
                        else {}
                    ),
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
        checkpoint = os.path.join(model_path, "photometric", f"iteration_{iteration}", "photometric.pth")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Missing photometric checkpoint: {checkpoint}")
        self.restore(torch.load(checkpoint, map_location=self.light_model.timesteps.device))
