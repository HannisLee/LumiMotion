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


class DirectionalLightModel(nn.Module):
    """A freely learnable unit direction for every scene timestep."""

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
        back = -F.normalize(camera_forward.to(device=device, dtype=dtype), dim=0)
        denominator = max(self.num_timesteps - 1, 1)
        time = torch.arange(self.num_timesteps, device=device, dtype=dtype) / float(denominator)
        sign = 1.0 if int(direction_sign) >= 0 else -1.0
        theta = float(phase) + sign * float(span) * time
        raw = (
            float(horizontal_radius) * torch.cos(theta)[:, None] * right[None]
            + float(vertical_radius) * torch.sin(theta)[:, None] * up[None]
            + float(back_offset) * back[None]
        )
        with torch.no_grad():
            self._raw_light_dir_table.copy_(F.normalize(raw, dim=-1))

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

    def __init__(self, timesteps: Any, normal_axis: str = "+z", device: str | torch.device = "cuda"):
        super().__init__()
        self.normal_axis = normal_axis
        self.light_model = DirectionalLightModel(timesteps, device=device)
        self.optimizer = None
        self.initialization_metadata: dict[str, Any] = {}

    @classmethod
    def from_args(cls, timesteps: Any, args: Any, device: str | torch.device = "cuda"):
        return cls(
            timesteps,
            normal_axis=getattr(args, "photometric_normal_axis", "+z"),
            device=device,
        )

    def training_setup(self, args: Any) -> None:
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
                group["lr"] = float(learning_rate)

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
            "horizontal_radius": float(horizontal_radius),
            "vertical_radius": float(vertical_radius),
            "back_offset": float(back_offset),
            "phase": float(phase),
            "direction_sign": 1 if int(direction_sign) >= 0 else -1,
            "span": float(span),
        }

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_raw_light_dirs()

    def get_all_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_light_dirs()

    def light_smoothness_loss(self) -> torch.Tensor:
        return self.light_model.first_order_smoothness_loss()

    def forward(self, albedo: torch.Tensor, normal: torch.Tensor, frame_id: torch.Tensor) -> dict[str, torch.Tensor]:
        light_dir = self.light_model(frame_id)
        if light_dir.ndim == 2:
            light_dir = light_dir[0]
        normal = F.normalize(normal, dim=-1)
        light_dir = F.normalize(light_dir, dim=-1)
        ndotl = (normal * light_dir[None]).sum(dim=-1, keepdim=True)
        shading = ndotl.clamp_min(0.0)
        return {
            "color": (albedo * shading).clamp(0.0, 1.0),
            "albedo": albedo,
            "normal": normal,
            "light_dir": light_dir,
            "ndotl": ndotl,
            "shading": shading,
            "timestep_idx": self.light_model.timestep_index(frame_id)[0],
        }

    def capture(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "timesteps": self.light_model.timesteps.detach().cpu(),
            "photometric_version": "stage1_perlight_v1_directional_camera_back_ellipse",
            "config": {"normal_axis": self.normal_axis, "light_param": "per_frame"},
            "initialization": self.initialization_metadata,
        }

    def restore(self, state: dict[str, Any]) -> None:
        config = state.get("config", {})
        self.normal_axis = config.get("normal_axis", self.normal_axis)
        state_dict = dict(state.get("state_dict", state))
        if "raw_light_dir" in state_dict and "light_model._raw_light_dir_table" not in state_dict:
            state_dict["light_model._raw_light_dir_table"] = state_dict.pop("raw_light_dir")
        state_dict.pop("raw_light_rgb", None)
        self.load_state_dict(state_dict, strict=False)
        self.initialization_metadata = dict(state.get("initialization", {}))

    def light_trajectory_dict(self) -> dict[str, Any]:
        raw = self.get_all_raw_light_dirs().detach().cpu().float()
        directions = self.get_all_light_dirs().detach().cpu().float()
        times = self.light_model.timesteps.detach().cpu().float()
        return {
            "photometric_version": "stage1_perlight_v1_directional_camera_back_ellipse",
            "initialization": self.initialization_metadata,
            "frames": [
                {
                    "index": index,
                    "fid": float(times[index]),
                    "raw": raw[index].tolist(),
                    "direction": directions[index].tolist(),
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
