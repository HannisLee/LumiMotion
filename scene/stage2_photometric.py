"""固定几何下的 Lambertian 材质、方向光和 HDR 数值计算。"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(0.0031308).pow(1 / 2.4) - 0.055)


def unit_vector(x):
    if not torch.isfinite(x).all() or (x.norm(dim=-1) < 1e-8).any():
        raise ValueError("光照方向必须有限且非零。")
    return F.normalize(x, dim=-1)


def frame_key(name):
    match = re.search(r"(\d+)$", Path(name).stem)
    if match is None:
        raise ValueError("无法从文件名解析原始帧号：" + name)
    return str(int(match.group(1))).zfill(4)


def read_gt_directions(path, center, keys, device):
    data = json.loads(Path(path).read_text())
    output = []
    for key in keys:
        record = data.get(key)
        if record is None:
            raise ValueError("GT 光照缺少帧 " + key)
        if "light_dir_world" in record:
            direction = torch.tensor(record["light_dir_world"], device=device, dtype=torch.float32)
            # 本项目导出约定：light_dir_world 为表面指向光源。
        elif "light_pos_world" in record:
            direction = torch.tensor(record["light_pos_world"], device=device, dtype=torch.float32) - center
        else:
            raise ValueError("GT 光照必须提供 light_dir_world 或 light_pos_world。")
        if direction.shape != (3,):
            raise ValueError("GT 方向/位置应包含三个分量。")
        output.append(unit_vector(direction))
    return torch.stack(output)


class DirectionalLightTable(nn.Module):
    """仅训练时刻具有参数；其他时刻做单位方向插值。"""
    def __init__(self, times, initial_dirs, learned=True):
        super().__init__()
        times = torch.as_tensor(times, device=initial_dirs.device, dtype=torch.float32)
        if times.ndim != 1 or len(times) == 0 or not torch.isfinite(times).all():
            raise ValueError("光照时间表必须为非空有限一维数组。")
        if len(times) > 1 and not (times[1:] > times[:-1]).all():
            raise ValueError("光照时间表必须严格递增。")
        if initial_dirs.shape != (len(times), 3):
            raise ValueError("时间表与方向表形状不匹配。")
        self.register_buffer("times", times)
        self.raw_directions = nn.Parameter(unit_vector(initial_dirs), requires_grad=learned)

    def forward(self, time):
        time = torch.as_tensor(time, device=self.times.device).float().reshape(())
        if not torch.isfinite(time):
            raise ValueError("查询光照时刻必须有限。")
        dirs = F.normalize(self.raw_directions, dim=-1)
        if len(self.times) == 1:
            return dirs[0]
        right = torch.searchsorted(self.times, time).clamp(1, len(self.times) - 1)
        left = right - 1
        weight = ((time - self.times[left]) / (self.times[right] - self.times[left])).clamp(0, 1)
        interpolated = (1 - weight) * dirs[left] + weight * dirs[right]
        # 相反方向的中点无定义，确定性选择较近端点。
        if interpolated.norm() < 1e-6:
            return dirs[left] if weight <= 0.5 else dirs[right]
        return F.normalize(interpolated, dim=0)

    def smoothness(self):
        if len(self.times) < 2 or not self.raw_directions.requires_grad:
            return self.raw_directions.new_zeros(())
        dirs = F.normalize(self.raw_directions, dim=-1)
        return (dirs[1:] - dirs[:-1]).square().mean()


class LambertianMaterial(nn.Module):
    def __init__(self, albedo_srgb):
        super().__init__()
        initial = albedo_srgb.detach().clamp(1e-4, 1 - 1e-4)
        self.register_buffer("initial_albedo", initial.clone())
        self.raw_albedo = nn.Parameter(torch.logit(initial))

    @property
    def albedo(self):
        return self.raw_albedo.sigmoid()

    def prior(self):
        return (self.albedo - self.initial_albedo).square().mean()


def directional_irradiance(normals, direction, intensity):
    direction = unit_vector(direction)
    return (normals * direction).sum(-1, keepdim=True).clamp_min(0) * intensity


def diffuse_color(albedo_linear, normals, direction, intensity):
    return albedo_linear * directional_irradiance(normals, direction, intensity) / math.pi


def world_direction_to_hdr_uv(directions, yaw_degrees=0.0):
    """与原始 HDR 脚本的轴变换及 EnvLight 经纬映射一致。"""
    directions = unit_vector(directions)
    longitude = torch.atan2(-directions[..., 1], directions[..., 0]) - math.radians(yaw_degrees)
    u = (longitude / (2 * math.pi) + 0.5).remainder(1)
    v = torch.acos(directions[..., 2].clamp(-1, 1)) / math.pi
    return torch.stack((u, v), -1)


class HDRIrradiance:
    """世界 Z 为上、经度 atan2(-y,x)，确定性均匀球面积分。"""
    def __init__(self, image, samples=2048, yaw_degrees=0.0, exposure=1.0):
        if samples < 16 or exposure < 0 or not math.isfinite(exposure) or not math.isfinite(yaw_degrees):
            raise ValueError("HDR 样本至少16，曝光非负且参数必须有限。")
        if image.ndim != 3 or image.shape[-1] != 3 or not torch.isfinite(image).all() or (image < 0).any():
            raise ValueError("HDR 必须为非负有限 H×W×3 线性 RGB。")
        i = torch.arange(samples, device=image.device, dtype=torch.float32)
        z = 1 - 2 * (i + 0.5) / samples
        phi = i * (math.pi * (3 - math.sqrt(5)))
        radius = (1 - z.square()).sqrt()
        self.directions = torch.stack((radius * phi.cos(), radius * phi.sin(), z), -1)
        uv = world_direction_to_hdr_uv(self.directions, yaw_degrees)
        u, v = uv.unbind(-1)
        # 水平周期边界，避免经度接缝。
        padded = torch.cat((image[:, -1:], image, image[:, :1]), 1)
        h, w = image.shape[:2]
        grid_x = 2 * (u * w + 1) / (w + 2) - 1
        grid = torch.stack((grid_x, 2 * v - 1), -1).reshape(1, -1, 1, 2)
        rgb = F.grid_sample(padded.permute(2, 0, 1)[None], grid, align_corners=False, padding_mode="border")
        self.radiance = rgb[0, :, :, 0].T * exposure * (4 * math.pi / samples)

    @classmethod
    def from_file(cls, path, device="cuda", **kwargs):
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("无法读取 HDR：" + str(path))
        return cls(torch.tensor(image[..., ::-1].copy(), device=device), **kwargs)

    def __call__(self, normals, chunk_size=4096):
        flat = normals.reshape(-1, 3)
        result = []
        for part in flat.split(chunk_size):
            result.append((part @ self.directions.T).clamp_min(0) @ self.radiance)
        return torch.cat(result).reshape_as(normals)
