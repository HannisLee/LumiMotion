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


class PointLightTable(DirectionalLightTable):
    """已知灯位的近场近似；不学习灯位，强度定义为参考距离的辐照度。"""
    def __init__(self, times, positions, center, reference_distance=8., ambient=0.):
        super().__init__(times, positions-center, learned=False)
        self.register_buffer('positions', positions)
        self.reference_distance=reference_distance
        self.ambient=ambient
        if reference_distance<=0 or ambient<0:raise ValueError('近场光参考距离须正，环境项须非负。')

    def position(self,time):
        time=torch.as_tensor(time,device=self.times.device).float().reshape(())
        if len(self.times)==1:return self.positions[0]
        right=torch.searchsorted(self.times,time).clamp(1,len(self.times)-1);left=right-1
        weight=((time-self.times[left])/(self.times[right]-self.times[left])).clamp(0,1)
        return (1-weight)*self.positions[left]+weight*self.positions[right]

    def render_kwargs(self,time):
        return dict(light_position=self.position(time),reference_distance=self.reference_distance,
                    ambient_irradiance=self.ambient)


class AreaLightTable(PointLightTable):
    """已知矩形面积光四点积分；Blender5默认P/(πA)，旧实验可显式加载legacy约定。"""
    def __init__(self, times, positions, center, records, reference_distance=8., ambient=0., power_convention='blender5'):
        super().__init__(times,positions,center,reference_distance,ambient)
        if power_convention not in {'blender5','legacy'}:raise ValueError('未知面积光功率约定。')
        self.power_convention=power_convention
        samples=[];normals=[];powers=[]
        offsets=positions.new_tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
        for pos,record in zip(positions,records):
            if record.get('area_shape') not in {'SQUARE','RECTANGLE'}:
                raise ValueError('面积光当前只支持矩形或正方形。')
            normal=unit_vector(positions.new_tensor(record['emission_normal_world']))
            u=unit_vector(positions.new_tensor(record['area_axis_u_world']))
            v=unit_vector(positions.new_tensor(record['area_axis_v_world']))
            width=record.get('world_area_size',record['area_size'])
            height=record.get('world_area_size_y',record.get('area_size_y',width) if record['area_shape']=='RECTANGLE' else width)
            if min(width,height,record['intensity'])<=0:raise ValueError('面积光尺寸及功率必须为正。')
            if abs(float(u@v))>1e-4 or abs(float(normal@u))>1e-4 or abs(float(normal@v))>1e-4:
                raise ValueError('面积光坐标轴必须正交。')
            samples.append(pos+(offsets[:,:1]*u*width+offsets[:,1:]*v*height)*.25)
            power=record['intensity']*2**record.get('exposure',0.)*record.get('diffuse_factor',1.)
            if not record.get('normalize',True):power*=width*height
            if not math.isfinite(power) or power<=0:raise ValueError('面积光有效功率必须正且有限。')
            if any(abs(v-1)>1e-5 for v in record.get('light_rgb',[1.,1.,1.])):raise ValueError('当前面积光分支要求白色光源。')
            normals.append(normal);powers.append(power)
        if max(powers)-min(powers)>1e-5:raise ValueError('当前面积光分支要求各帧共享固定功率。')
        self.register_buffer('area_samples',torch.stack(samples))
        self.register_buffer('emission_normals',torch.stack(normals))
        if any('influence_radius' in r for r in records):
            radii=positions.new_tensor([r['influence_radius'] for r in records])
            if not torch.isfinite(radii).all() or (radii<=0).any():raise ValueError('面积光影响距离必须正且有限。')
            self.register_buffer('influence_radii',radii)
        factor=math.pi if power_convention=='blender5' else 4.
        self.reference_intensity=powers[0]/(factor*reference_distance**2)

    def interpolate(self,values,time):
        time=torch.as_tensor(time,device=self.times.device).float().reshape(())
        if len(self.times)==1:return values[0]
        right=torch.searchsorted(self.times,time).clamp(1,len(self.times)-1);left=right-1
        weight=((time-self.times[left])/(self.times[right]-self.times[left])).clamp(0,1)
        return values[left]*(1-weight)+values[right]*weight

    def sample_positions(self,time):
        return self.interpolate(self.area_samples,time)

    def render_kwargs(self,time):
        result=dict(super().render_kwargs(time),light_sample_positions=self.sample_positions(time),
                    light_emission_normal=unit_vector(self.interpolate(self.emission_normals,time)))
        if hasattr(self,'influence_radii'):result['light_influence_radius']=self.interpolate(self.influence_radii,time)
        return result


def area_irradiance(points,normals,samples,emission_normal,intensity,reference_distance,influence_radius=None):
    """矩形四子区域中点积分；包含受光面与发光面的余弦以及距离平方衰减。"""
    delta=samples-points[...,None,:]
    distance2=delta.square().sum(-1).clamp_min(1e-6)
    direction=F.normalize(delta,dim=-1)
    receiver=(normals[...,None,:]*direction).sum(-1).clamp_min(0)
    emitter=(-(direction*emission_normal).sum(-1)).clamp_min(0)
    result=(receiver*emitter/distance2).mean(-1,keepdim=True)*intensity*reference_distance**2
    if influence_radius is not None:
        relative_distance2=(points-samples.mean(0)).square().sum(-1,keepdim=True)/influence_radius**2
        result=result*(1-relative_distance2.square()).clamp(0,1).square()
    return result


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


class PaletteLambertianMaterial(nn.Module):
    """由Stage1颜色自动分组的共享材质；不读取GT标签或GT albedo。"""
    def __init__(self, albedo_srgb, size=4, state=None):
        super().__init__()
        if size<1 or size>len(albedo_srgb):raise ValueError('材质颜色数量必须在1到Gaussian数之间。')
        if state is not None:
            labels=torch.zeros(len(albedo_srgb),device=albedo_srgb.device,dtype=torch.long)
            initial=albedo_srgb.new_full((size,3),.5)
        else:
            with torch.no_grad():
                initial_colors=albedo_srgb.detach().clamp(1e-4,1-1e-4)
                log_color=srgb_to_linear(initial_colors).clamp_min(1e-3).log()
                features=log_color-log_color.mean(-1,keepdim=True)
                # 固定、局部随机源的kmeans++，不消耗训练RNG，不把亮暗当独立材质。
                generator=torch.Generator(device=features.device).manual_seed(0)
                centers=[features[torch.randint(len(features),(1,),device=features.device,generator=generator)[0]]]
                distances=(features-centers[0]).square().sum(-1)
                for _ in range(1,size):
                    index=torch.multinomial(distances+1e-12,1,generator=generator)[0]
                    centers.append(features[index]);distances=torch.minimum(distances,(features-centers[-1]).square().sum(-1))
                centers=torch.stack(centers)
                for _ in range(40):
                    labels=torch.cdist(features,centers).argmin(-1)
                    updated=centers.clone()
                    for k in range(size):
                        selected=labels==k
                        if selected.any():updated[k]=features[selected].mean(0)
                    if torch.max((updated-centers).abs())<1e-5:break
                    centers=updated
                initial=torch.stack([initial_colors[labels==k].median(0).values if (labels==k).any()
                                     else initial_colors.median(0).values for k in range(size)])
        self.register_buffer('labels',labels)
        self.register_buffer('initial_palette',initial)
        self.raw_albedo=nn.Parameter(torch.logit(initial.clamp(1e-4,1-1e-4)))

    @property
    def albedo(self):
        return self.raw_albedo.sigmoid()[self.labels]

    def prior(self):
        return (self.raw_albedo.sigmoid()-self.initial_palette).square().mean()


def make_material(albedo_srgb,args,state=None):
    mode=getattr(args,'photometric_material_mode','per_gaussian')
    initialization=getattr(args,'photometric_palette_init','')
    if initialization and mode!='palette':raise ValueError('共享材质初始化文件仅用于palette模式。')
    if initialization and state is None:
        import numpy as np
        import hashlib
        with np.load(initialization,allow_pickle=False) as data:
            source=Path(str(data['source_model'].item())).resolve()
            if source!=Path(args.stage1_model_path).resolve() or int(data['source_iteration'])!=args.load_iter:
                raise ValueError('共享材质初始化的Stage1来源或迭代不匹配。')
            hashes=json.loads(str(data['source_files_json'].item()))
            expected={str(source/'point_cloud'/('iteration_'+str(args.load_iter))/'point_cloud.ply'),
                      str(source/'deform'/('iteration_'+str(args.load_iter))/'deform.pth')}
            if set(hashes)!=expected:raise ValueError('共享材质初始化缺少完整来源哈希。')
            for path,digest in hashes.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise ValueError('共享材质初始化来源哈希不匹配。')
            labels=data['labels'];palette=data['initial_palette']
            if labels.shape!=(len(albedo_srgb),) or labels.dtype.kind not in 'iu':
                raise ValueError('共享材质初始化标签数量或类型错误。')
            if palette.shape!=(args.photometric_palette_size,3) or not np.isfinite(palette).all() or (palette<0).any() or (palette>1).any():
                raise ValueError('共享材质初始化颜色必须为K×3、有限且在[0,1]。')
            if (labels<0).any() or (labels>=args.photometric_palette_size).any():raise ValueError('共享材质初始化标签越界。')
            initial=torch.as_tensor(palette,device=albedo_srgb.device,dtype=albedo_srgb.dtype).clamp(1e-4,1-1e-4)
            state={'labels':torch.as_tensor(labels,device=albedo_srgb.device,dtype=torch.long),
                   'initial_palette':initial,'raw_albedo':torch.logit(initial)}
    if mode=='per_gaussian':material=LambertianMaterial(albedo_srgb)
    elif mode=='palette':material=PaletteLambertianMaterial(albedo_srgb,args.photometric_palette_size,state)
    else:raise ValueError('未知Lambertian材质参数化：'+mode)
    if state is not None:material.load_state_dict(state)
    return material


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
