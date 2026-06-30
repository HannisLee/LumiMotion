"""Stage 1 V3 directional-light Lambertian photometric renderer."""

from __future__ import annotations

import json
import math
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
        timestep_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    else:
        timestep_tensor = torch.as_tensor(timesteps, dtype=torch.float32, device=device).flatten()

    if timestep_tensor.numel() == 0:
        timestep_tensor = torch.zeros(1, dtype=torch.float32, device=device)
    return timestep_tensor


def parse_axis(axis: str | torch.Tensor, device: torch.device | str) -> torch.Tensor:
    if torch.is_tensor(axis):
        out = axis.to(device=device, dtype=torch.float32).flatten()
    elif isinstance(axis, str):
        text = axis.strip()
        named = {
            "+x": (1.0, 0.0, 0.0),
            "-x": (-1.0, 0.0, 0.0),
            "+y": (0.0, 1.0, 0.0),
            "-y": (0.0, -1.0, 0.0),
            "+z": (0.0, 0.0, 1.0),
            "-z": (0.0, 0.0, -1.0),
        }
        if text in named:
            out = torch.tensor(named[text], dtype=torch.float32, device=device)
        else:
            values = [float(part) for part in text.replace(",", " ").split()]
            if len(values) != 3:
                raise ValueError(f"Axis must be '+z', '-z', or three floats, got {axis!r}.")
            out = torch.tensor(values, dtype=torch.float32, device=device)
    else:
        out = torch.tensor(axis, dtype=torch.float32, device=device).flatten()

    if out.numel() != 3:
        raise ValueError(f"Axis must have 3 elements, got shape {tuple(out.shape)}.")
    return F.normalize(out, dim=0)


def circle_upper_hemisphere_init(
    times: torch.Tensor,
    total_timesteps: int,
    r_xy: float = 0.8,
    z: float | None = 0.6,
    phase: float = 0.0,
    direction_sign: int = 1,
) -> torch.Tensor:
    """Return smooth circular upper-hemisphere light directions for initialization."""
    total = max(int(total_timesteps), 1)
    times = times.to(dtype=torch.float32)
    r_xy = float(r_xy)
    if z is None:
        z_value = math.sqrt(max(1.0 - r_xy * r_xy, 0.0))
    else:
        z_value = float(z)
    sign = 1.0 if int(direction_sign) >= 0 else -1.0
    theta = sign * 2.0 * math.pi * times / float(total) + float(phase)
    dirs = torch.stack(
        (
            r_xy * torch.cos(theta),
            r_xy * torch.sin(theta),
            torch.full_like(theta, z_value),
        ),
        dim=-1,
    )
    return F.normalize(dirs, dim=-1)


def get_gaussian_normal(rotation_t: torch.Tensor, normal_axis: str = "+z") -> torch.Tensor:
    """Compute 2DGS surface normals from dynamic Gaussian rotations.

    LumiMotion stores Gaussian rotations as world-space quaternions and passes the
    same rotations to the 2DGS rasterizer. Directional lights in this module are
    therefore also modeled in world space.
    """
    if rotation_t.ndim >= 3 and tuple(rotation_t.shape[-2:]) == (3, 3):
        rot = rotation_t
    elif rotation_t.shape[-1] == 4:
        rot = build_rotation(rotation_t.reshape(-1, 4)).reshape(*rotation_t.shape[:-1], 3, 3)
    else:
        raise ValueError(
            "rotation_t must be a quaternion tensor [..., 4] or rotation matrix [..., 3, 3], "
            f"got {tuple(rotation_t.shape)}."
        )

    axis = parse_axis(normal_axis, rot.device)
    normal = torch.matmul(rot, axis.view(3, 1)).squeeze(-1)
    return F.normalize(normal, dim=-1)


class DirectionalLightModel(nn.Module):
    """Learnable per-frame directional light table."""

    def __init__(
        self,
        timesteps,
        light_param: str = "per_frame",
        init_r_xy: float = 0.8,
        init_z: float = 0.6,
        init_phase: float = 0.0,
        init_direction_sign: int = 1,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.light_param = str(light_param)
        if self.light_param != "per_frame":
            raise ValueError("Stage1 V3 photometric light_param is fixed to 'per_frame'.")

        timestep_tensor = _as_timestep_tensor(timesteps, device)
        self.register_buffer("timesteps", timestep_tensor)
        self.num_timesteps = int(timestep_tensor.numel())
        self.init_r_xy = float(init_r_xy)
        self.init_z = float(init_z)
        self.init_phase = float(init_phase)
        self.init_direction_sign = 1 if int(init_direction_sign) >= 0 else -1

        frame_times = torch.arange(self.num_timesteps, dtype=torch.float32, device=timestep_tensor.device)
        init_dirs = circle_upper_hemisphere_init(
            frame_times,
            self.num_timesteps,
            self.init_r_xy,
            self.init_z,
            self.init_phase,
            self.init_direction_sign,
        )
        self._raw_light_dir_table = nn.Parameter(init_dirs)

    def config_dict(self) -> dict[str, Any]:
        return {
            "light_param": self.light_param,
            "init_r_xy": self.init_r_xy,
            "init_z": self.init_z,
            "init_phase": self.init_phase,
            "init_direction_sign": self.init_direction_sign,
        }

    def reset_circle_init(
        self,
        phase: float | None = None,
        direction_sign: int | None = None,
        r_xy: float | None = None,
        z: float | None = None,
    ) -> None:
        phase = self.init_phase if phase is None else float(phase)
        direction_sign = self.init_direction_sign if direction_sign is None else int(direction_sign)
        r_xy = self.init_r_xy if r_xy is None else float(r_xy)
        z = self.init_z if z is None else float(z)
        with torch.no_grad():
            times = torch.arange(self.num_timesteps, dtype=torch.float32, device=self.timesteps.device)
            self._raw_light_dir_table.copy_(
                circle_upper_hemisphere_init(times, self.num_timesteps, r_xy, z, phase, direction_sign)
            )
        self.init_phase = phase
        self.init_direction_sign = 1 if int(direction_sign) >= 0 else -1

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        return self._raw_light_dir_table

    def get_all_light_dirs(self) -> torch.Tensor:
        return F.normalize(self.get_all_raw_light_dirs(), dim=-1)

    def timestep_index(self, fid: torch.Tensor) -> torch.Tensor:
        fid_values = fid.detach().to(self.timesteps.device).float().reshape(-1)
        distances = torch.abs(fid_values[:, None] - self.timesteps[None, :])
        return torch.argmin(distances, dim=1).long()

    def forward(self, frame_id: torch.Tensor) -> torch.Tensor:
        all_dirs = self.get_all_light_dirs()
        idx = self.timestep_index(frame_id)
        out = all_dirs[idx]
        return out[0] if out.shape[0] == 1 else out

    def smoothness_loss(self, order: int = 1) -> torch.Tensor:
        light_dirs = self.get_all_light_dirs()
        if order == 1:
            if light_dirs.shape[0] < 2:
                return light_dirs.new_zeros(())
            diff = light_dirs[1:] - light_dirs[:-1]
        elif order == 2:
            if light_dirs.shape[0] < 3:
                return light_dirs.new_zeros(())
            diff = light_dirs[2:] - 2.0 * light_dirs[1:-1] + light_dirs[:-2]
        else:
            raise ValueError(f"Unsupported light smoothness order {order}; expected 1 or 2.")
        return diff.pow(2).mean()

    def hemisphere_loss(self, hemi_axis: str, hemi_margin: float = 0.0) -> torch.Tensor:
        light_dirs = self.get_all_light_dirs()
        axis = parse_axis(hemi_axis, light_dirs.device)
        dot = (light_dirs * axis[None, :]).sum(dim=-1)
        return F.relu(float(hemi_margin) - dot).pow(2).mean()


class PhotometricLambertianRenderer(nn.Module):
    def __init__(
        self,
        timesteps,
        light_param: str = "per_frame",
        init_r_xy: float = 0.8,
        init_z: float = 0.6,
        init_phase: float = 0.0,
        init_direction_sign: int = 1,
        normal_axis: str = "+z",
        hemi_axis: str = "0,0,1",
        hemi_margin: float = 0.0,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.normal_axis = normal_axis
        self.hemi_axis = hemi_axis
        self.hemi_margin = float(hemi_margin)
        self.device_name = str(device)
        self.light_model = DirectionalLightModel(
            timesteps,
            light_param=light_param,
            init_r_xy=init_r_xy,
            init_z=init_z,
            init_phase=init_phase,
            init_direction_sign=init_direction_sign,
            device=device,
        )
        self.optimizer = None
        self.multistart_metadata: dict[str, Any] = {}

    @classmethod
    def from_args(cls, timesteps, training_args, device: str | torch.device = "cuda"):
        return cls(
            timesteps,
            light_param=getattr(training_args, "photometric_light_param", "per_frame"),
            init_r_xy=getattr(training_args, "photometric_init_r_xy", 0.8),
            init_z=getattr(training_args, "photometric_init_z", 0.6),
            init_phase=getattr(training_args, "photometric_init_phase", 0.0),
            init_direction_sign=getattr(training_args, "photometric_init_direction_sign", 1),
            normal_axis=getattr(training_args, "photometric_normal_axis", "+z"),
            hemi_axis=getattr(training_args, "photometric_hemi_axis", "0,0,1"),
            hemi_margin=getattr(training_args, "photometric_hemi_margin", 0.0),
            device=device,
        )

    def config_dict(self) -> dict[str, Any]:
        out = self.light_model.config_dict()
        out.update(
            {
                "normal_axis": self.normal_axis,
                "hemi_axis": self.hemi_axis,
                "hemi_margin": self.hemi_margin,
            }
        )
        return out

    def rebuild_light_model_from_config(self, config: dict[str, Any]) -> None:
        self.light_model = DirectionalLightModel(
            self.light_model.timesteps,
            light_param=config.get("light_param", self.light_model.light_param),
            init_r_xy=config.get("init_r_xy", self.light_model.init_r_xy),
            init_z=config.get("init_z", self.light_model.init_z),
            init_phase=config.get("init_phase", self.light_model.init_phase),
            init_direction_sign=config.get("init_direction_sign", self.light_model.init_direction_sign),
            device=self.light_model.timesteps.device,
        )

    def training_setup(self, training_args):
        lr = getattr(training_args, "photometric_light_lr", 1e-3)
        params = [{"params": self.light_model.parameters(), "lr": lr, "name": "photometric_light"}]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def set_light_lr(self, lr: float) -> None:
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def get_all_raw_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_raw_light_dirs()

    def get_all_light_dirs(self) -> torch.Tensor:
        return self.light_model.get_all_light_dirs()

    def light_smoothness_loss(self, order: int = 1) -> torch.Tensor:
        return self.light_model.smoothness_loss(order=order)

    def hemisphere_loss(self, hemi_axis: str | None = None, hemi_margin: float | None = None) -> torch.Tensor:
        return self.light_model.hemisphere_loss(
            self.hemi_axis if hemi_axis is None else hemi_axis,
            self.hemi_margin if hemi_margin is None else hemi_margin,
        )

    def forward(self, albedo: torch.Tensor, normal: torch.Tensor, fid: torch.Tensor) -> dict[str, torch.Tensor]:
        light_dir_t = self.light_model(fid)
        if light_dir_t.ndim == 2:
            light_dir_t = light_dir_t[0]
        normal_t = F.normalize(normal, dim=-1)
        light_dir_t = F.normalize(light_dir_t, dim=-1)
        ndotl = torch.sum(normal_t * light_dir_t[None, :], dim=-1, keepdim=True)
        shading = ndotl.clamp_min(0.0)
        color = (albedo * shading).clamp(0.0, 1.0)
        return {
            "color": color,
            "albedo": albedo,
            "normal": normal_t,
            "light_dir": light_dir_t,
            "ndotl": ndotl,
            "shading": shading,
            "timestep_idx": self.light_model.timestep_index(fid)[0],
        }

    def capture(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "timesteps": self.light_model.timesteps.detach().cpu(),
            "photometric_version": "stage1_v3_directional_per_frame_light",
            "config": self.config_dict(),
            "multistart": self.multistart_metadata,
        }

    def restore(self, model_args: dict[str, Any]) -> None:
        config = dict(model_args.get("config", {}))
        if config:
            self.normal_axis = config.get("normal_axis", self.normal_axis)
            self.hemi_axis = config.get("hemi_axis", self.hemi_axis)
            self.hemi_margin = float(config.get("hemi_margin", self.hemi_margin))
            if config.get("light_param", self.light_model.light_param) != self.light_model.light_param:
                self.rebuild_light_model_from_config(config)

        state_dict = dict(model_args.get("state_dict", model_args))
        # Legacy V1 checkpoints used a flat raw_light_dir tensor.
        if "raw_light_dir" in state_dict and "light_model._raw_light_dir_table" not in state_dict:
            if self.light_model.light_param != "per_frame":
                self.rebuild_light_model_from_config({"light_param": "per_frame"})
            state_dict["light_model._raw_light_dir_table"] = state_dict.pop("raw_light_dir")
        state_dict.pop("raw_light_rgb", None)
        self.load_state_dict(state_dict, strict=False)
        self.multistart_metadata = dict(model_args.get("multistart", {}))

    def light_trajectory_dict(self) -> dict[str, Any]:
        raw = self.get_all_raw_light_dirs().detach().cpu().float()
        dirs = self.get_all_light_dirs().detach().cpu().float()
        timesteps = self.light_model.timesteps.detach().cpu().float()
        return {
            "photometric_version": "stage1_v3_directional_per_frame_light",
            "config": self.config_dict(),
            "frames": [
                {
                    "index": int(i),
                    "fid": float(timesteps[i].item()),
                    "raw": [float(v) for v in raw[i].tolist()],
                    "direction": [float(v) for v in dirs[i].tolist()],
                }
                for i in range(dirs.shape[0])
            ],
        }

    def save_light_trajectory(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        dirs = self.get_all_light_dirs().detach().cpu().float().numpy()
        np.save(os.path.join(output_dir, "light_dirs.npy"), dirs)
        with open(os.path.join(output_dir, "light_dirs.json"), "w", encoding="utf-8") as handle:
            json.dump(self.light_trajectory_dict(), handle, indent=2)

    def save_weights(self, model_path: str, iteration: int) -> None:
        out_weights_path = os.path.join(model_path, "photometric", f"iteration_{iteration}")
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.capture(), os.path.join(out_weights_path, "photometric.pth"))
        self.save_light_trajectory(out_weights_path)

    def load_weights(self, model_path: str, iteration: int) -> None:
        weights_path = os.path.join(model_path, "photometric", f"iteration_{iteration}", "photometric.pth")
        model_args = torch.load(weights_path, map_location=self.light_model.timesteps.device)
        self.restore(model_args)
