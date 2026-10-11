"""静态DL的物理着色：仅方向、训练RGB估计光强、预测几何可见度。"""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from gaussian_renderer import render as render_native
from scene.stage2_photometric import frame_key, srgb_to_linear, linear_to_srgb
from utils.general_utils import build_rotation, build_scaling_rotation


class DirectionOnlyLights(torch.nn.Module):
    """所有灯的全局正光强由训练RGB估计，不接受输入能量或颜色。"""
    def __init__(self, path, initial_intensity=math.pi):
        super().__init__()
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        records = data.get('frames', data)
        self.directions = {}
        light_count = None
        forbidden = {'intensity', 'light_rgb', 'energy', 'power', 'color', 'light_pos_world'}
        for key, record in records.items():
            if not str(key).isdigit():
                continue
            lights = record.get('lights', [record])
            vectors = []
            for light in lights:
                if forbidden.intersection(light):
                    raise ValueError('方向文件必须只保留方向，禁止光强、颜色及位置字段。')
                vector = torch.as_tensor(light['light_dir_world'], dtype=torch.float32)
                if vector.shape != (3,) or not torch.isfinite(vector).all() or abs(float(vector.norm())-1) > 1e-4:
                    raise ValueError('方向必须为有限的世界单位向量。')
                vectors.append(vector)
            if not vectors or (light_count is not None and len(vectors) != light_count):
                raise ValueError('逐帧灯数必须相同且非零。')
            light_count = len(vectors)
            self.directions[str(int(key)).zfill(4)] = torch.stack(vectors)
        if light_count is None:
            raise ValueError('方向文件没有有效帧。')
        self.log_intensity = torch.nn.Parameter(torch.full((light_count,), math.log(initial_intensity), device='cuda'))

    def for_camera(self, camera):
        key = frame_key(camera.image_name_train_light)
        return self.directions[key].to(self.log_intensity.device), self.log_intensity.exp()


class StaticVisibility:
    """用当前预测Gaussian追踪可见度；不允许阴影残差反传旋转或几何。"""
    def __init__(self, bias=.03, update_interval=50):
        self.bias = bias
        self.update_interval = update_interval
        self.last_build = None
        self.last_count = None

    def invalidate(self):
        """增密/裁剪/opacity reset后即使点数相同也必须重建。"""
        self.last_build = None

    @torch.no_grad()
    def __call__(self, pc, directions, iteration):
        xyz, opacity, scales, rotation = pc.get_xyz, pc.get_opacity, pc.get_scaling, pc.get_rotation
        if (self.last_build is None or len(xyz) != self.last_count or
                iteration-self.last_build >= self.update_interval):
            # opacity reset后低于alpha_min的点不得产生sqrt(log) NaN。
            scale3 = torch.cat((scales, torch.full_like(scales[:, :1], 1e-6)), -1)
            transform = build_scaling_rotation(scale3, rotation)
            radius = (2*(opacity/pc.alpha_min).log().clamp_min(0)).sqrt()[:, None]
            vertices = radius*(pc.unit_icosahedron_vertices[None]@transform.transpose(-1, -2))+xyz[:, None]
            faces = pc.unit_icosahedron_faces[None]+torch.arange(len(xyz), device=xyz.device)[:, None, None]*12
            ids = torch.arange(len(xyz), device=xyz.device)[:, None].expand(-1, faces.shape[1])
            pc.gaussian_tracer.build_bvh(vertices.reshape(-1, 3), faces.reshape(-1, 3), ids.reshape(-1))
            self.last_build, self.last_count = iteration, len(xyz)
        incident = directions[None].expand(len(xyz), -1, -1).contiguous()
        traced = pc.trace(xyz[:, None]+incident*self.bias, incident,
                          features=torch.zeros(len(xyz), 4, device=xyz.device), xyz=xyz,
                          scales=scales, rotation=rotation, opacity=opacity, use_zeros=True)
        return (1-traced['alpha']).clamp(0, 1)


def render_photometric(camera, pc, pipe, background, lights, visibility=None, iteration=0):
    """逐surfel在线性域着色再光栅合成；几何map仍为原生光栅输出。"""
    directions, intensity = lights.for_camera(camera)
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    view = camera.camera_center-pc.get_xyz
    facing = torch.where((normal*view).sum(-1, keepdim=True) >= 0, 1., -1.).detach()
    normal = normal*facing
    cosine = (normal@directions.T).clamp_min(0)
    vis = torch.ones_like(cosine) if visibility is None else visibility(pc, directions, iteration)
    irradiance = (cosine*vis*intensity[None]).sum(-1, keepdim=True)
    albedo = srgb_to_linear(pc.get_albedo)
    colors = albedo*irradiance/math.pi
    package = render_native(camera, pc, pipe, srgb_to_linear(background), 0., 0., 0., override_color=colors)
    package['render_linear'] = package['render']
    package['render'] = linear_to_srgb(package['render'].clamp_min(0))
    package['mean_visibility'] = vis.mean().detach()
    package['visibility_below_point_one'] = (vis < .1).float().mean().detach()
    package['visibility_above_point_nine'] = (vis > .9).float().mean().detach()
    package['light_intensity'] = intensity.detach()
    return package


def robust_photometric_loss(image, target, alpha, shadow_weight=.25, highlight_weight=.5, threshold=.08):
    """减弱大幅暗残差和高光；固定分母，所有像素仍保留非零权重。"""
    residual = image-target
    signed = residual.detach().mean(0, keepdim=True)
    weight = torch.ones_like(signed)
    weight = torch.where(signed > threshold, shadow_weight, weight)
    weight = torch.where(signed < -threshold, highlight_weight, weight)
    # 背景正常约束且轮廓单独由Alpha监督，不让残差权重改变几何评测掩膜。
    weight = torch.where(alpha.detach() > .5, weight, torch.ones_like(weight))
    loss = (torch.sqrt(residual.square()+1e-6)*weight).mean()
    return loss, {'shadow_downweight_fraction': (signed > threshold).float().mean().detach(),
                  'highlight_downweight_fraction': (signed < -threshold).float().mean().detach()}
