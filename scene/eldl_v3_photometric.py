"""V3 受限混合照明：独立共享EL加带可见度DL；固定TRAIN估计参数。"""
import hashlib
import json
import math
from pathlib import Path

import torch


def diffuse_components(albedo_linear,normal,directions,direct_intensity,ambient_rgb,visibility):
    """全局EL不乘方向光可见度，不提供逐点环境残差自由度。"""
    cosine = (normal@directions.T).clamp_min(0)
    direct = albedo_linear*(cosine*visibility*direct_intensity[None]).sum(-1,keepdim=True)/math.pi
    environment = albedo_linear*ambient_rgb.reshape(1,3)/math.pi
    return environment,direct,environment+direct


def censored_symmetric_loss(image,target,alpha,saturated_mask=None):
    """仅精确满码通道作单边约束；不用DL黑暗标签或残差分类阈值。"""
    saturated = ((target.detach()>=1.) & (alpha.detach()>.5) if saturated_mask is None
                 else saturated_mask.to(device=target.device,dtype=torch.bool))
    residual = image-target
    residual = torch.where(saturated,residual.clamp_max(0),residual)
    loss = torch.sqrt(residual.square()+1e-6).mean()
    return loss,dict(saturated_channel_fraction=saturated.float().mean().detach(),
                     shadow_downweight_fraction=loss.detach()*0,
                     highlight_downweight_fraction=loss.detach()*0)


class FixedELDLLights(torch.nn.Module):
    def __init__(self,path,device='cuda'):
        super().__init__()
        raw=Path(path).read_bytes();data=json.loads(raw)
        if data['gt_read'] or data['source_split']!='TRAIN' or data['environment_mode']!='shared_rgb_constant':
            raise ValueError('必须是TRAIN独立共享EL估计资产')
        self.sha256=hashlib.sha256(raw).hexdigest()
        self.directions={}
        for key,record in data['frames'].items():
            value=torch.as_tensor(record['light_dir_world'],dtype=torch.float32,device=device).reshape(1,3)
            if not torch.isfinite(value).all() or abs(float(value.norm())-1)>1e-4:
                raise ValueError('世界灯向必须是单位有限向量')
            self.directions[key]=value
        direct=torch.as_tensor(data['direct_intensity'],dtype=torch.float32,device=device).reshape(1)
        ambient=torch.as_tensor(data['ambient_rgb'],dtype=torch.float32,device=device).reshape(3)
        if not torch.isfinite(direct).all() or not torch.isfinite(ambient).all() or (direct<=0).any() or (ambient<0).any():
            raise ValueError('EL非负，DL严格正且有限')
        self.log_intensity=torch.nn.Parameter(direct.log(),requires_grad=False)
        self.register_buffer('ambient_rgb',ambient)
        self.asset=data

    def for_camera(self,camera):
        from scene.stage2_photometric import frame_key
        return self.directions[frame_key(camera.image_name_train_light)],self.log_intensity.exp()


def render_photometric(camera,pc,pipe,background,lights,visibility=None,iteration=0):
    from gaussian_renderer import render as render_native
    from scene.stage2_photometric import srgb_to_linear,linear_to_srgb
    from utils.general_utils import build_rotation
    directions,intensity=lights.for_camera(camera)
    normal=build_rotation(pc.get_rotation)[:,:,2]
    view=camera.camera_center-pc.get_xyz
    facing=torch.where((normal*view).sum(-1,keepdim=True)>=0,1.,-1.).detach()
    normal=normal*facing
    vis=torch.ones((len(normal),len(directions)),device=normal.device) if visibility is None else visibility(pc,directions,iteration)
    env,direct,colors=diffuse_components(srgb_to_linear(pc.get_albedo),normal,directions,intensity,lights.ambient_rgb,vis)
    package=render_native(camera,pc,pipe,srgb_to_linear(background),0.,0.,0.,override_color=colors)
    package['render_linear']=package['render']
    package['render']=linear_to_srgb(package['render'].clamp_min(0))
    package['mean_visibility']=vis.mean().detach()
    package['visibility_below_point_one']=(vis<.1).float().mean().detach()
    package['visibility_above_point_nine']=(vis>.9).float().mean().detach()
    package['light_intensity']=intensity.detach()
    package['ambient_rgb']=lights.ambient_rgb.detach()
    package['environment_surfel_mean']=env.detach().mean()
    package['direct_surfel_mean']=direct.detach().mean()
    return package
