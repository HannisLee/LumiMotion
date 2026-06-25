import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricLambertianRenderer(nn.Module):
    def __init__(self, timesteps, init_light_dir=None, device="cuda"):
        super().__init__()
        if isinstance(timesteps, dict):
            ordered = [fid for fid, _ in sorted(timesteps.items(), key=lambda item: item[1])]
            timestep_tensor = torch.tensor([float(fid.item()) for fid in ordered], dtype=torch.float32, device=device)
        else:
            timestep_tensor = torch.as_tensor(timesteps, dtype=torch.float32, device=device).flatten()

        if timestep_tensor.numel() == 0:
            timestep_tensor = torch.zeros(1, dtype=torch.float32, device=device)
        self.register_buffer("timesteps", timestep_tensor)

        num_timesteps = int(timestep_tensor.numel())
        if init_light_dir is None:
            init_light_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

        init_light_dir = F.normalize(init_light_dir.to(device).float(), dim=0)
        self.raw_light_dir = nn.Parameter(init_light_dir[None].repeat(num_timesteps, 1))
        self.register_buffer("fixed_light_rgb", torch.ones(3, dtype=torch.float32, device=device), persistent=False)
        self.optimizer = None

    def training_setup(self, training_args):
        params = [{"params": self.parameters(), "lr": training_args.photometric_light_lr, "name": "photometric_light"}]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    @property
    def light_dir(self):
        return F.normalize(self.raw_light_dir, dim=-1)

    @property
    def light_rgb(self):
        return self.fixed_light_rgb[None].expand(self.timesteps.numel(), -1)

    def timestep_index(self, fid):
        fid_value = fid.detach().to(self.timesteps.device).float().reshape(-1)[0]
        return torch.argmin(torch.abs(self.timesteps - fid_value)).long()

    def forward(self, albedo, normal, fid):
        timestep_idx = self.timestep_index(fid)
        light_dir_t = self.light_dir[timestep_idx]
        light_rgb_t = self.light_rgb[timestep_idx]
        normal_t = F.normalize(normal, dim=-1)
        ndotl = torch.sum(normal_t * light_dir_t[None, :], dim=-1, keepdim=True).clamp_min(0.0)
        color = albedo * ndotl
        return {
            "color": color,
            "normal": normal_t,
            "light_dir": light_dir_t,
            "light_rgb": light_rgb_t,
            "ndotl": ndotl,
            "timestep_idx": timestep_idx,
        }

    def light_smoothness_loss(self):
        if self.timesteps.numel() < 2:
            return torch.zeros((), dtype=self.raw_light_dir.dtype, device=self.raw_light_dir.device)
        d_dir = self.light_dir[1:] - self.light_dir[:-1]
        return d_dir.pow(2).mean()

    def capture(self):
        return {
            "state_dict": self.state_dict(),
            "timesteps": self.timesteps.detach().cpu(),
            "photometric_version": "directional_uniform_light_v0",
        }

    def restore(self, model_args):
        state_dict = dict(model_args["state_dict"])
        # Older experimental checkpoints optimized per-frame light RGB. Stage 1 v0 fixes
        # intensity to one, so those weights are intentionally ignored on load.
        state_dict.pop("raw_light_rgb", None)
        self.load_state_dict(state_dict, strict=False)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "photometric", "iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.capture(), os.path.join(out_weights_path, "photometric.pth"))

    def load_weights(self, model_path, iteration):
        weights_path = os.path.join(model_path, "photometric", "iteration_{}".format(iteration), "photometric.pth")
        model_args = torch.load(weights_path, map_location="cpu")
        self.restore(model_args)
