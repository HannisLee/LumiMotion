"""固定源方向权重：只改变可见度到遮挡XYZ的余切，物理前向不变。"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from gaussian_renderer import render as render_native
from scene.static_dl_shadow_split import rebuild_bvh, visibility_xyz_only, render_split_photometric
from scene.stage2_photometric import frame_key, srgb_to_linear, linear_to_srgb
from scene.static_dl_censored import loss_for_mode
from utils.general_utils import build_rotation


class FixedDirectionWeights:
    def __init__(self, manifest_path, init_model, init_iteration, point_count):
        self.path = Path(manifest_path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.manifest = json.loads(self.path.read_text())
        self.cache = {}
        expected = {str(i).zfill(4) for i in range(1,151) if i%10}
        if (self.manifest['status'] != 'PASS' or set(self.manifest['frames']) != expected
                or self.manifest['point_count'] != point_count
                or Path(self.manifest['model']).resolve() != Path(init_model).resolve()
                or self.manifest['iteration'] != init_iteration):
            raise AssertionError('固定方向资产与共同初始化/135帧不匹配。')
        for path,sha in self.manifest['source_checkpoint_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise AssertionError('方向资产绑定的源检查点变化。')
        for key,record in self.manifest['frames'].items():
            path = self.path.parent/record['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise AssertionError('固定权重SHA失败：'+key)
            value = np.load(path,allow_pickle=False)
            if value.shape != (point_count,) or not np.isfinite(value).all() or value.min()<.5 or value.max()>2:
                raise AssertionError('固定权重形状/范围错误。')
            self.cache[key] = torch.from_numpy(value.copy())

    def for_camera(self,camera,mode,device='cuda'):
        value = self.cache[frame_key(camera.image_name_train_light)].to(device)
        if mode == 'uniform':
            value = torch.ones_like(value)
        elif mode != 'direction':
            raise ValueError('未知方向权重模式。')
        return value.detach()


def render_direction_weighted(camera,pc,pipe,background,directions,intensity,bias,weights,
                               receiver_positions=None,rebuild=True):
    if pipe.compute_cov3D_python:
        raise ValueError('仅支持既有原生scale/rotation路径。')
    if weights.shape != (len(pc.get_xyz),) or weights.requires_grad:
        raise ValueError('固定权重必须逐点且停止梯度。')
    if rebuild:
        rebuild_bvh(pc)
    visibility = visibility_xyz_only(pc,directions,bias,receiver_positions)
    normal = build_rotation(pc.get_rotation)[:,:,2]
    view = camera.camera_center-pc.get_xyz
    facing = torch.where((normal*view).sum(-1,keepdim=True)>=0,1.,-1.).detach()
    cosine = ((normal*facing)@directions.detach().T).clamp_min(0)
    albedo = srgb_to_linear(pc.get_albedo)
    receiver = albedo*(cosine*visibility.detach()*intensity[None]).sum(-1,keepdim=True)/math.pi
    shadow = albedo.detach()*(cosine.detach()*visibility*intensity.detach()[None]).sum(-1,keepdim=True)/math.pi
    colors = receiver+weights.detach()[:,None]*(shadow-shadow.detach())
    package = render_native(camera,pc,pipe,srgb_to_linear(background),0.,0.,0.,override_color=colors,
                            detach_xyz=True,detach_rot=True,detach_scale=True,detach_opacity=True)
    package['render_linear'] = package['render']
    package['render'] = linear_to_srgb(package['render'].clamp_min(0))
    package['override_colors'] = colors
    package['light_visibility'] = visibility.detach()
    package['visibility_graph_for_audit'] = visibility
    package['mean_visibility'] = visibility.detach().mean()
    package['unused_screen_proxy'] = package.pop('viewspace_points')
    return package


def audit_fixed_weight_routes(camera,pc,pipe,background,lights,settings,assets):
    """只验证新增w链条；已验证的censored/深度/trace数学不重复建框架。"""
    directions,_ = lights.for_camera(camera)
    directions = directions.detach().clone().requires_grad_(True)
    receiver = pc.get_xyz.detach().clone().requires_grad_(True)
    parameters = {'xyz':pc._xyz,'rotation':pc._rotation,'material':pc._albedo_dc,'light':lights.log_intensity,
                  'scale':pc._scaling,'opacity':pc._opacity,'direction':directions,'receiver_origin':receiver}
    original = {k:v.requires_grad for k,v in parameters.items()}
    for v in parameters.values():
        v.requires_grad_(True)
    target,alpha = camera.original_image_train_light.cuda(),camera.gt_alpha_mask.cuda()
    if pipe.white_background:
        target = alpha*target+(1-alpha)*background[:,None,None]
    arrays,forwards,losses = {},{},{ }
    weight = assets.for_camera(camera,'direction')
    oracle = None
    try:
        for mode in ('uniform','direction','legacy_split'):
            if mode == 'legacy_split':
                package = render_split_photometric(camera,pc,pipe,background,directions,lights.log_intensity.exp(),
                              settings.shadow_bias,route='split',receiver_positions=receiver)
            else:
                package = render_direction_weighted(camera,pc,pipe,background,directions,lights.log_intensity.exp(),
                              settings.shadow_bias,assets.for_camera(camera,mode),receiver_positions=receiver)
            forwards[mode] = {k:package[k].detach().cpu() for k in
                ('render','render_linear','rend_alpha','rend_normal','surf_depth','surf_normal','rend_dist','light_visibility')}
            loss = loss_for_mode(settings.loss_model)(package['render'],target,alpha,settings.shadow_weight,
                           settings.highlight_weight,settings.residual_threshold)[0]
            losses[mode] = float(loss.detach())
            if mode == 'uniform':
                visibility = package['visibility_graph_for_audit']
                gv = torch.autograd.grad(loss,visibility,retain_graph=True)[0].detach()
                oracle = torch.autograd.grad(visibility,pc._xyz,grad_outputs=gv*weight[:,None],retain_graph=True)[0].detach().cpu()
            grad = torch.autograd.grad(loss,tuple(parameters.values()),allow_unused=True)
            arrays[mode] = {k:None if g is None else g.detach().cpu() for k,g in zip(parameters,grad)}
        def error(a,b):
            if a is None and b is None:
                return [0.,0.]
            a = torch.zeros_like(b) if a is None else a
            b = torch.zeros_like(a) if b is None else b
            return [float((a-b).abs().max()),float((a-b).norm()/b.norm().clamp_min(1e-20))]
        common = {k:error(arrays['direction'][k],arrays['uniform'][k]) for k in ('rotation','material','light')}
        legacy = {k:error(arrays['uniform'][k],arrays['legacy_split'][k]) for k in parameters}
        xyz_error = error(arrays['direction']['xyz'],oracle)
        maps = {mode:{k:float((forwards[mode][k]-forwards['uniform'][k]).abs().max())
                      for k in forwards[mode]} for mode in ('direction','legacy_split')}
        summaries = {mode:{k:{'norm':0. if g is None else float(g.norm()),
                     'finite':g is None or bool(torch.isfinite(g).all()),'is_none':g is None}
                     for k,g in group.items()} for mode,group in arrays.items()}
        checks = {'same_physical_forward':all(v==0 for m in maps.values() for v in m.values()),
                  'all_finite':all(v['finite'] for m in summaries.values() for v in m.values()),
                  'forbidden_routes_zero':all(summaries[m][k]['norm']==0 for m in summaries
                      for k in ('scale','opacity','direction','receiver_origin')),
                  'receiver_gradients_unchanged':all(a<=2e-6 and r<=1e-5 for a,r in common.values()),
                  'uniform_matches_existing_split':all(a<=2e-6 and r<=1e-4 for a,r in legacy.values()),
                  'weighted_xyz_matches_explicit_vjp':xyz_error[1]<=1e-4,
                  'allowed_gradients_nonzero':all(summaries[m][k]['norm']>0 for m in summaries
                      for k in ('xyz','rotation','material','light')),
                  'fixed_weights_nontrivial':bool((weight-1).abs().max()>1e-6),
                  'asset_range':bool((weight>=.5).all() and (weight<=2).all())}
        return {'status':'PASS' if all(checks.values()) else 'FAILED','checks':checks,
                'camera':camera.image_name_train_light,'parameter_updates':0,'losses':losses,
                'forward_max_abs':maps,'common_gradient_error_abs_relative':common,
                'legacy_gradient_error_abs_relative':legacy,'weighted_xyz_vjp_error':xyz_error,
                'gradients':summaries,'asset_manifest_sha256':assets.sha256,
                'weight_min_max_mean':[float(weight.min()),float(weight.max()),float(weight.mean())]}
    finally:
        for name,value in parameters.items():
            value.requires_grad_(original[name])
