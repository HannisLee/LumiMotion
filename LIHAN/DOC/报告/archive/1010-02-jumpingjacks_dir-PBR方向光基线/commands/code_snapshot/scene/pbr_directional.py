"""原生PBR的固定方向光元数据；直接读取SUN，不用灯位近似方向。"""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import torch

from scene.stage2_photometric import frame_key


CONVENTION = 'unit vector from surface toward light, for max(n dot l, 0)'


class GTDirectionalLights:
    def __init__(self, path):
        self.path = Path(path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        if data.get('coordinate_space', 'world') != 'world':
            raise ValueError('PBR方向光必须为世界坐标。')
        frames = data.get('frames', data)
        self.frames = {}
        for key, frame in frames.items():
            if not str(key).isdigit():
                continue
            lights = frame.get('lights', [frame])
            if not lights:
                raise ValueError('PBR方向光帧没有灯光：' + str(key))
            directions, irradiances = [], []
            for light in lights:
                if ('light_dir_world' not in light or
                        light.get('type', 'directional') != 'directional' or
                        light.get('source_type', 'SUN') != 'SUN'):
                    raise ValueError('PBR方向光只接受SUN/light_dir_world，不接受灯位。')
                if light.get('direction_convention', CONVENTION) != CONVENTION:
                    raise ValueError('PBR方向光须为表面指向光源的向量。')
                if light.get('sun_angular_size_radians', 0) != 0:
                    raise ValueError('当前PBR方向光仅支持角大小0的SUN。')
                direction = torch.tensor(light['light_dir_world'], dtype=torch.float32)
                rgb = torch.tensor(light.get('light_rgb', [1., 1., 1.]), dtype=torch.float32)
                intensity = light.get('intensity')
                if (direction.shape != (3,) or not torch.isfinite(direction).all() or
                        abs(float(direction.norm()) - 1) > 1e-4):
                    raise ValueError('GT方向必须为有限单位三维向量。')
                if (rgb.shape != (3,) or not torch.isfinite(rgb).all() or (rgb < 0).any() or
                        not isinstance(intensity, (int, float)) or not math.isfinite(intensity) or intensity < 0):
                    raise ValueError('GT光颜色/强度必须有限且非负，并显式提供intensity。')
                directions.append(direction / direction.norm())
                irradiances.append(rgb * intensity)
            normalized_key = str(int(key)).zfill(4)
            if normalized_key in self.frames:
                raise ValueError('重复GT方向光帧号：' + normalized_key)
            self.frames[normalized_key] = (torch.stack(directions), torch.stack(irradiances))
        if not self.frames:
            raise ValueError('GT方向光元数据没有有效帧。')

    def for_camera(self, camera, scale=1., device='cuda'):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('PBR方向光强度倍率必须有限且为正。')
        key = frame_key(camera.image_name_train_light)
        if key not in self.frames:
            raise ValueError('GT方向光缺少相机帧：' + key)
        directions, irradiances = self.frames[key]
        return directions.to(device), irradiances.to(device) * scale


@lru_cache(maxsize=8)
def load_gt_directional_lights(path):
    return GTDirectionalLights(path)


def directional_lights_for_camera(pipe, camera):
    path = getattr(pipe, 'pbr_gt_lights_path', '')
    if not path:
        raise ValueError('gt_directional PBR需要 --pbr_gt_lights_path。')
    table = load_gt_directional_lights(str(Path(path).resolve()))
    return table.for_camera(camera, getattr(pipe, 'pbr_light_intensity_scale', 1.))


def validate_directional_checkpoint(model, iteration, config):
    """拒绝将复制的Stage1默认roughness当作训练后的PBR材质。"""
    path = Path(model) / 'pbr_directional' / f'iteration_{iteration}' / 'lighting.json'
    record = json.loads(path.read_text())
    metadata = Path(config['pbr_gt_lights_path'])
    digest = hashlib.sha256(metadata.read_bytes()).hexdigest()
    if (record.get('format_version') != 1 or record.get('iteration') != iteration or
            iteration <= config.get('load_iter', -1) or
            record.get('lights_sha256') != digest or
            record.get('intensity_scale') != config.get('pbr_light_intensity_scale', 1.)):
        raise ValueError('PBR方向光checkpoint标记/iteration/光照快照不匹配。')
    return [path, metadata]
