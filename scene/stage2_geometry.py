"""Stage2 共享世界空间几何残差；原 Stage1 张量和形变输入始终只读。"""
import math
import json
import torch
from torch import nn
from torch.nn import functional as F


def apply_filmic_lut(rgb, lut):
    """Blender Filmic Medium High Contrast，65³ log域LUT的可微三线性插值。"""
    lo,hi=-12.473931188,4.026068812
    coords=((rgb.clamp_min(2**lo).log2()-lo)/(hi-lo)*2-1).clamp(-1,1)
    original=rgb.shape
    grid=coords.reshape(1,1,1,-1,3)
    result=F.grid_sample(lut,grid,mode='bilinear',padding_mode='border',align_corners=True)
    return result[0,:,0,0].T.reshape(original)


def quaternion_product(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def load_coherence_source_mask(path, source, iteration, source_files, optimization_keys, count):
    """仅接受与当前冻结来源、训练划分对应的布尔掩码，无pickle或其他外部路径读取。"""
    import numpy as np
    from pathlib import Path
    with np.load(path, allow_pickle=False) as asset:
        mask = asset['stable_mask'].copy()
        metadata = json.loads(asset['metadata'].item())
    if mask.dtype != np.bool_ or mask.shape != (count,):
        raise ValueError('时间稳定掩码的dtype或源点数量不匹配。')
    if (metadata.get('format_version') != 1 or
            metadata.get('method') != 'frozen_stage1_training_time_variation' or
            metadata.get('gt_normal_inputs') is not False or
            metadata.get('source_iteration') != iteration or
            metadata.get('source_files') != source_files or
            metadata.get('optimization_keys') != optimization_keys or
            Path(metadata.get('source', '')).resolve() != Path(source).resolve()):
        raise ValueError('时间稳定掩码的来源、SHA、训练划分或协议不匹配。')
    return torch.from_numpy(mask), metadata


class GeometryResidual(nn.Module):
    def __init__(self, xyz, args):
        super().__init__()
        extent = torch.quantile(xyz.detach(), .99, dim=0)-torch.quantile(xyz.detach(), .01, dim=0)
        self.register_buffer('scene_extent', extent.norm().clamp_min(1e-6))
        self.register_buffer('parent_index', None)
        self.register_buffer('child_offsets', None)
        residual_xyz = xyz
        self.source_count = len(xyz)
        self.subdivision_side = 2
        if getattr(args, 'geometry_subdivide', False):
            self.subdivision_side = getattr(args, 'geometry_subdivision_side', 2)
            if self.subdivision_side not in (2, 4):
                raise ValueError('表面细分每轴仅支持2或4片。')
            self.parent_index = torch.arange(len(xyz), device=xyz.device).repeat_interleave(self.subdivision_side**2)
            if self.subdivision_side == 2:
                # 保持原4子片的构造顺序和浮点常量。
                offsets = xyz.new_tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])*math.sqrt(.75)
            else:
                axis = torch.arange(self.subdivision_side, device=xyz.device, dtype=xyz.dtype)*2-(self.subdivision_side-1)
                u, v = torch.meshgrid(axis, axis, indexing='ij')
                offsets = torch.stack((u.flatten(), v.flatten()), -1)*(math.sqrt(3)/self.subdivision_side)
            self.child_offsets = offsets.repeat(len(xyz),1)
            residual_xyz = xyz[self.parent_index]
        self.translation = nn.Parameter(torch.zeros_like(residual_xyz))
        self.axis_angle = nn.Parameter(torch.zeros_like(residual_xyz))
        self.normal_gradient_mode = getattr(args, 'geometry_normal_gradient_mode', 'joint')
        if self.normal_gradient_mode not in {'joint', 'depth', 'position'}:
            raise ValueError('法线-深度一致性梯度模式仅支持joint或depth。')
        unsupported_weight = getattr(args, 'geometry_normal_unsupported_weight', 1.)
        if not math.isfinite(unsupported_weight) or not 0 <= unsupported_weight <= 1:
            raise ValueError('无有效光度支持区域的法线一致性权重必须有限且在[0,1]内。')
        distortion_weight = getattr(args, 'lambda_geometry_distortion', 0.)
        if not math.isfinite(distortion_weight) or distortion_weight < 0:
            raise ValueError('深度畸变权重必须有限且非负。')
        self.distortion_enabled = distortion_weight > 0
        rgb_weight = getattr(args, 'lambda_geometry_rgb_normal_prior', 0.)
        if not math.isfinite(rgb_weight) or rgb_weight < 0:
            raise ValueError('训练RGB法线先验权重必须有限且非负。')
        self.rgb_normal_enabled = rgb_weight > 0
        rgb_warmup = getattr(args, 'geometry_rgb_prior_warmup', 0)
        if (isinstance(rgb_warmup, bool) or not isinstance(rgb_warmup, int) or rgb_warmup < 0 or
                (rgb_warmup > 0 and not self.rgb_normal_enabled)):
            raise ValueError('RGB先验旋转预热步数必须为非负整数，开启时需要训练RGB先验。')
        self.register_buffer('rgb_normal_targets', None)
        self.register_buffer('rgb_normal_valid', None)
        if self.rgb_normal_enabled:
            if not getattr(args, 'geometry_rgb_normal_prior_path', ''):
                raise ValueError('RGB法线先验需要明确的来源资产。')
            self.rgb_normal_targets = torch.zeros_like(residual_xyz)
            self.rgb_normal_valid = torch.zeros(len(residual_xyz), device=xyz.device, dtype=torch.bool)
        coherence_weight = getattr(args, 'lambda_geometry_coherence', 0.)
        if not math.isfinite(coherence_weight) or coherence_weight < 0:
            raise ValueError('邻域一致性权重必须有限且非负。')
        self.coherence_enabled = coherence_weight > 0
        self.coherence_position_gradient_scale = getattr(args, 'geometry_coherence_position_gradient_scale', 1.)
        if (not math.isfinite(self.coherence_position_gradient_scale) or
                not 0 < self.coherence_position_gradient_scale <= 1):
            raise ValueError('邻域位置梯度比例必须有限并在(0,1]内。')
        mask_path = getattr(args, 'geometry_coherence_source_mask', '')
        if (self.coherence_enabled and getattr(args, 'geometry_refined_coherence', False) and
                (not getattr(args, 'geometry_initial_stage2_model', '') or not mask_path)):
            raise ValueError('微调表面邻域需要真实Stage2初始化和训练来源时间稳定掩码。')
        if (self.coherence_enabled and
                args.loss_preset == 'lambertian_geometry_area_static_coherent' and not mask_path):
            raise ValueError('静态邻域preset需要geometry_coherence_source_mask。')
        self.register_buffer('coherence_source_static', None)
        if self.coherence_enabled and mask_path:
            self.coherence_source_static = torch.zeros(self.source_count, device=xyz.device, dtype=torch.bool)
        for name in ('coherence_neighbors', 'coherence_orientation', 'coherence_lengths',
                     'coherence_weights', 'coherence_initialized'):
            self.register_buffer(name, None)
        if self.coherence_enabled:
            self.coherence_k = getattr(args, 'geometry_coherence_neighbors', 6)
            self.coherence_tangent_limit = getattr(args, 'geometry_coherence_tangent_limit', .35)
            self.coherence_angle = getattr(args, 'geometry_coherence_max_angle', 45.)
            self.coherence_radius = getattr(args, 'geometry_coherence_radius', 3.)
            if (self.coherence_k < 1 or not 0 < self.coherence_tangent_limit < 1 or
                    not 0 < self.coherence_angle < 90 or not math.isfinite(self.coherence_radius) or
                    self.coherence_radius <= 0):
                raise ValueError('邻域一致性图的参数不合法。')
            shape = (len(residual_xyz), self.coherence_k)
            self.coherence_neighbors = torch.zeros(shape, device=xyz.device, dtype=torch.long)
            self.coherence_orientation = xyz.new_ones(shape)
            self.coherence_lengths = xyz.new_ones(shape)
            self.coherence_weights = xyz.new_zeros(shape)
            self.coherence_initialized = torch.tensor(False, device=xyz.device)
        self.log_scale = None
        if getattr(args, 'geometry_optimize_scale', False):
            ratio = args.geometry_max_scale_ratio
            if not math.isfinite(ratio) or ratio <= 1:
                raise ValueError('尺寸信赖域倍率必须有限且大于1。')
            self.max_log_scale = math.log(ratio)
            self.log_scale = nn.Parameter(torch.zeros_like(residual_xyz[:, :2]))
        self.log_opacity = None
        if getattr(args, 'geometry_optimize_opacity', False):
            ratio = args.geometry_max_opacity_ratio
            if not math.isfinite(ratio) or ratio <= 1:
                raise ValueError('不透明度衰减信赖域倍率必须有限且大于1。')
            self.max_log_opacity = math.log(ratio)
            self.log_opacity = nn.Parameter(torch.zeros_like(residual_xyz[:, :1]))
        self.max_displacement = args.geometry_max_displacement
        self.max_angle = math.radians(args.geometry_max_angle)
        self.tonemap=getattr(args,'geometry_tonemap','srgb')
        self.shading=getattr(args,'geometry_shading','screen')
        if self.shading not in {'screen','surfel'}:raise ValueError('未知几何管线着色方式。')
        if self.tonemap not in {'srgb','filmic'}:raise ValueError('未知几何管线显示变换。')
        if self.tonemap=='filmic':
            import numpy as np
            from pathlib import Path
            path=Path(__file__).resolve().parents[1]/'assets/stage2/filmic_medium_high.npz'
            with np.load(path) as data:lut=torch.from_numpy(data['lut'].copy()).to(xyz)
            self.register_buffer('display_lut',lut.permute(3,0,1,2).unsqueeze(0))
        if self.max_displacement <= 0 or self.max_angle <= 0:
            raise ValueError('几何信赖域必须为正。')

    def apply(self, means, rotations):
        if self.parent_index is not None and len(means) != len(self.translation):
            raise ValueError('细分几何必须通过prepare同时展开位置、尺寸和不透明度。')
        theta = self.axis_angle.norm(dim=-1, keepdim=True)
        delta = torch.cat((torch.cos(theta/2),
                           self.axis_angle * (.5*torch.sinc(theta/(2*math.pi)))), -1)
        # 两个输入均为单位四元数；避免再次归一化给零残差引入舍入变化。
        return means+self.translation, quaternion_product(delta, rotations)

    def expand_values(self, values):
        return values if self.parent_index is None else values[self.parent_index]

    def prepare(self, means, rotations, scales, opacity):
        """共享渲染入口；先用冻结父表面建立子点，再应用独立残差。"""
        if self.parent_index is not None:
            if len(means)*self.subdivision_side**2 != len(self.translation):
                raise ValueError('细分来源点数与checkpoint不匹配。')
            means, rotations, scales, opacity = [self.expand_values(x) for x in (means,rotations,scales,opacity)]
            local = torch.cat((self.child_offsets*scales, torch.zeros_like(scales[:,:1])),dim=-1)
            unit = F.normalize(rotations,dim=-1)
            # 四元数旋转局部偏移，等价于R(q) @ local，避免保存逐点3x3矩阵。
            qv = unit[:,1:]
            world = local+2*torch.cross(qv,torch.cross(qv,local,dim=-1)+unit[:,:1]*local,dim=-1)
            means = means+world
            scales = scales*(1/self.subdivision_side)
        means, rotations = self.apply(means,rotations)
        return means,rotations,self.apply_scales(scales),self.apply_opacity(opacity)

    def anchors(self):
        return ((self.translation/(self.scene_extent*self.max_displacement)).square().sum(-1).mean(),
                (self.axis_angle/self.max_angle).square().sum(-1).mean())

    @torch.no_grad()
    def initialize_coherence(self, means, normals, scales, opacity, labels=None, static=None):
        """只由冻结来源的实际初始表面片建立图；保存后不随训练或相机重建。"""
        if not self.coherence_enabled or bool(self.coherence_initialized):
            return
        import numpy as np
        from scipy.spatial import cKDTree
        if len(means) != len(self.translation):
            raise ValueError('一致性图必须使用实际渲染表面片。')
        x = means.detach().cpu().double().numpy()
        n = F.normalize(normals, dim=-1).detach().cpu().double().numpy()
        radius = scales.detach().cpu().double().numpy().max(-1)*self.coherence_radius
        usable = opacity.detach().cpu().numpy().reshape(-1) > .2
        if static is not None:usable &= static.detach().cpu().numpy().astype(bool)
        groups = labels.detach().cpu().numpy() if labels is not None else None
        count = min(len(x), max(32, self.coherence_k*4)+1)
        distances, indices = cKDTree(x).query(x, k=count, workers=2)
        if count < 2:raise ValueError('一致性图至少需要两个表面片。')
        # 相同距离按点序确定，独立初始化和检查点重载不消耗训练RNG。
        order = np.lexsort((indices, distances), axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        distances = np.take_along_axis(distances, order, axis=1)
        delta = x[indices]-x[:, None]
        dot = (n[:, None]*n[indices]).sum(-1)
        sign = np.where(dot >= 0, 1., -1.)
        average_normal = (n[:, None]+n[indices]*sign[..., None])*.5
        tangent_error = np.abs((delta*average_normal).sum(-1))/np.maximum(distances, 1e-12)
        eligible = (distances > float(self.scene_extent)*1e-6) & usable[:, None] & usable[indices]
        eligible &= np.abs(dot) > math.cos(math.radians(self.coherence_angle))
        eligible &= tangent_error <= self.coherence_tangent_limit
        eligible &= distances <= np.minimum(radius[:, None], radius[indices])
        if groups is not None:eligible &= groups[:, None] == groups[indices]
        ranks = np.argsort(np.where(eligible, np.arange(count)[None], count), axis=1, kind='stable')
        take = ranks[:, :min(self.coherence_k, count)]
        chosen = np.take_along_axis(indices, take, axis=1)
        valid = np.take_along_axis(eligible, take, axis=1)
        length = np.take_along_axis(distances, take, axis=1)
        orientation = np.take_along_axis(sign, take, axis=1)
        if not valid.any():raise ValueError('来源没有可用的同表面邻域。')
        width = chosen.shape[1]
        self.coherence_neighbors[:, :width].copy_(torch.as_tensor(chosen, device=means.device))
        self.coherence_orientation[:, :width].copy_(torch.as_tensor(orientation, device=means.device, dtype=means.dtype))
        self.coherence_lengths[:, :width].copy_(torch.as_tensor(
            np.maximum(length, float(self.scene_extent)*1e-5), device=means.device, dtype=means.dtype))
        self.coherence_weights[:, :width].copy_(torch.as_tensor(valid, device=means.device, dtype=means.dtype))
        self.coherence_initialized.fill_(True)

    def coherence_loss(self, means, normals):
        if not self.coherence_enabled or not bool(self.coherence_initialized):
            raise ValueError('一致性图尚未初始化。')
        if self.coherence_position_gradient_scale != 1.:
            # 显式优化预条件：前向位置不变，只缩放此项对位置的梯度。
            frozen = means.detach()
            means = frozen+self.coherence_position_gradient_scale*(means-frozen)
        neighbor = self.coherence_neighbors
        average_normal = (normals[:, None]+normals[neighbor]*self.coherence_orientation[..., None])*.5
        residual = ((means[neighbor]-means[:, None])*average_normal).sum(-1)/self.coherence_lengths
        # 双端法线的平均切平面约束；球面相邻点严格为零，不强迫曲面平坦。
        return (residual.square()*self.coherence_weights).sum()/self.coherence_weights.sum().clamp_min(1)

    def apply_scales(self, scales):
        # 关闭时直接返回原张量，旧checkpoint不增加状态或浮点运算。
        return scales if self.log_scale is None else scales*self.log_scale.exp()

    def apply_opacity(self, opacity):
        # 只衰减冻结来源的alpha；零残差精确保留来源，关闭时不增加运算或状态。
        return opacity if self.log_opacity is None else opacity*self.log_opacity.exp()

    @torch.no_grad()
    def project(self, optimizer=None, rotation_only=False):
        parameters = ([(self.axis_angle, self.max_angle)] if rotation_only else
                      [(self.translation, self.scene_extent*self.max_displacement),
                       (self.axis_angle, self.max_angle)])
        for p, radius in parameters:
            length = p.norm(dim=-1, keepdim=True)
            clipped = length > radius
            p.mul_((radius/length.clamp_min(1e-12)).clamp_max(1))
            # 清除越界点的动量，避免持续向约束外推。
            if optimizer is not None and clipped.any():
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = optimizer.state.get(p, {}).get(key)
                    if value is not None:value.masked_fill_(clipped.expand_as(value), 0)
        if self.log_scale is not None and not rotation_only:
            clipped = self.log_scale.abs() > self.max_log_scale
            self.log_scale.clamp_(-self.max_log_scale, self.max_log_scale)
            if optimizer is not None and clipped.any():
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = optimizer.state.get(self.log_scale, {}).get(key)
                    if value is not None:value.masked_fill_(clipped, 0)
        if self.log_opacity is not None and not rotation_only:
            clipped = (self.log_opacity < -self.max_log_opacity) | (self.log_opacity > 0)
            self.log_opacity.clamp_(-self.max_log_opacity, 0)
            if optimizer is not None and clipped.any():
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = optimizer.state.get(self.log_opacity, {}).get(key)
                    if value is not None:value.masked_fill_(clipped, 0)

    @torch.no_grad()
    def diagnostics(self):
        displacement = self.translation.norm(dim=-1)/self.scene_extent
        angle = self.axis_angle.norm(dim=-1)*180/math.pi
        result = {'xyz_p95_fraction':float(torch.quantile(displacement,.95)),
                'xyz_max_fraction':float(displacement.max()),
                'rotation_p95_degrees':float(torch.quantile(angle,.95)),
                'rotation_max_degrees':float(angle.max()),
                'xyz_bound_fraction':float((displacement>=self.max_displacement*.999).float().mean()),
                'rotation_bound_fraction':float((angle>=math.degrees(self.max_angle)*.999).float().mean())}
        if self.log_scale is not None:
            result.update(scale_ratio_min=float(self.log_scale.min().exp()),
                          scale_ratio_max=float(self.log_scale.max().exp()),
                          scale_bound_fraction=float((self.log_scale.abs()>=self.max_log_scale*.999).float().mean()))
        if self.parent_index is not None:
            result.update(rendered_gaussians=len(self.translation),source_gaussians=self.source_count)
        if self.log_opacity is not None:
            result.update(opacity_ratio_min=float(self.log_opacity.min().exp()),
                          opacity_ratio_max=float(self.log_opacity.max().exp()),
                          opacity_lower_bound_fraction=float((self.log_opacity<=-self.max_log_opacity*.999).float().mean()),
                          opacity_source_bound_fraction=float((self.log_opacity>=-1e-8).float().mean()))
        if self.rgb_normal_enabled:
            result['rgb_normal_prior_fraction'] = float(self.rgb_normal_valid.float().mean())
        if self.coherence_enabled:
            result.update(coherence_edge_fraction=float(self.coherence_weights.mean()),
                          coherence_initialized=bool(self.coherence_initialized))
            if self.coherence_source_static is not None:
                result['coherence_source_stable_fraction'] = float(self.coherence_source_static.float().mean())
        return result


def geometry_schedule(step, start, args):
    if step < start:return 0.
    ramp = min(1., (step-start+1)/max(1,args.geometry_ramp))
    progress = min(1., max(0., (step-start)/max(1,args.geometry_schedule_steps-start)))
    return ramp * args.geometry_final_lr_ratio**progress


def geometry_training_phase(step, args):
    """预热独立占用预算，联合调度从预热结束重新起算；旧配置默认为零。"""
    warmup = getattr(args, 'geometry_rgb_prior_warmup', 0)
    return step < warmup, step if step < warmup else step-warmup


def geometry_group_lr(name, step, args):
    prior_only, local = geometry_training_phase(step, args)
    if prior_only:
        return (args.geometry_rotation_lr*min(1., (local+1)/max(1, args.geometry_ramp))
                if name == 'geometry_rotation' else 0.)
    if name == 'geometry_xyz':
        return args.geometry_xyz_lr*geometry_schedule(local, args.geometry_xyz_start, args)
    if name == 'geometry_rotation':
        return args.geometry_rotation_lr*geometry_schedule(local, args.geometry_warmup, args)
    if name == 'geometry_scale':
        return args.geometry_scale_lr*geometry_schedule(local, args.geometry_scale_start, args)
    if name == 'geometry_opacity':
        return args.geometry_opacity_lr*geometry_schedule(local, args.geometry_opacity_start, args)
    if name == 'albedo':
        return args.photometric_albedo_lr*(1. if local < args.geometry_warmup else .3)
    raise ValueError('几何阶段未知优化器参数组：'+name)


def rgb_normal_prior_loss(normals, geometry):
    """来源已核验的训练RGB目标；默认联合分支保持原公式及浮点顺序。"""
    cosine = (normals*geometry.rgb_normal_targets).sum(-1).clamp(-1, 1)
    mask = geometry.rgb_normal_valid
    return ((1-cosine)*mask).sum()/mask.sum().clamp_min(1)


def normal_depth_penalty(cosine, margin_degrees=0.):
    error = 1-cosine
    # 默认分支保持原浮点公式；允许角度差时只惩罚超过容差的部分。
    if margin_degrees == 0:return error
    if not math.isfinite(margin_degrees) or not 0 <= margin_degrees < 90:
        raise ValueError('法线与深度法线的容差须在[0,90)度。')
    return (error-(1-math.cos(math.radians(margin_degrees)))).clamp_min(0)


def supported_normal_loss(penalty, valid, unsupported_weight=1., support=None):
    """只降低无有效光度支持的预测一致性；原有效像素分母保持不变。"""
    if not math.isfinite(unsupported_weight) or not 0 <= unsupported_weight <= 1:
        raise ValueError('无有效光度支持区域的法线一致性权重必须有限且在[0,1]内。')
    count = valid.sum().clamp_min(1)
    if unsupported_weight == 1.:
        return (penalty*valid).sum()/count
    if (support is None or support.shape != valid.shape or
            not torch.isfinite(support).all() or
            not ((support == 0) | (support == 1)).all()):
        raise ValueError('预测一致性区域减权需要同形状的有限二值光度支持掩码。')
    weights = unsupported_weight+(1-unsupported_weight)*support.detach()
    return (penalty*valid*weights).sum()/count


def geometry_losses(camera, package, reference, alpha, geometry, args, normal_support=None):
    from utils.point_utils import depth_to_normal
    normal_depth, _ = depth_to_normal(camera, package['surf_depth'])
    normal_depth = normal_depth.permute(2,0,1)
    # 只在完整邻域都有表面、且参考深度连续的内部计算差分法线。
    valid = ((alpha>.99)&(reference['alpha']>.99)&(package['rend_alpha'].detach()>.99)).float()
    valid = 1-F.max_pool2d((1-valid)[None],5,1,2)[0]
    z0 = reference['depth']
    spread = F.max_pool2d(z0[None],5,1,2)[0]+F.max_pool2d(-z0[None],5,1,2)[0]
    valid = valid * (spread < geometry.scene_extent*.02).float()
    valid[:, :2] = 0; valid[:, -2:] = 0; valid[:, :, :2] = 0; valid[:, :, -2:] = 0
    count = valid.sum().clamp_min(1)
    normal_target = package['normal']
    if getattr(geometry, 'normal_gradient_mode', 'joint') in {'depth', 'position'}:
        # 目标来自本次模型预测；暂时固定GS方向，让预测深度几何跟随光度法线。
        # 光度项仍对XYZ和旋转反传；该项仍经raster深度对二者反传。
        normal_target = normal_target.detach()
    normal = supported_normal_loss(normal_depth_penalty((normal_target*normal_depth).sum(0,keepdim=True),
              getattr(args,'geometry_normal_margin',0.)), valid,
              getattr(args,'geometry_normal_unsupported_weight',1.), normal_support)
    silhouette = (package['rend_alpha']-alpha).abs().mean()
    depth = (F.smooth_l1_loss(package['surf_depth']/(geometry.scene_extent*.002),
              z0/(geometry.scene_extent*.002),reduction='none')*valid).sum()/count
    position, rotation = geometry.anchors()
    terms = {'geometry_normal':normal,'geometry_alpha':silhouette,'geometry_depth':depth,
             'geometry_anchor':position,'geometry_rotation_anchor':rotation}
    if geometry.log_scale is not None:
        terms['geometry_scale_anchor'] = (geometry.log_scale/geometry.max_log_scale).square().mean()
    if geometry.log_opacity is not None:
        terms['geometry_opacity_anchor'] = (geometry.log_opacity/geometry.max_log_opacity).square().mean()
    if geometry.coherence_enabled:
        terms['geometry_coherence'] = geometry.coherence_loss(package['geometry_means'], package['geometry_normals'])
    if geometry.rgb_normal_enabled:
        terms['geometry_rgb_normal_prior'] = rgb_normal_prior_loss(package['geometry_normals'], geometry)
    if geometry.distortion_enabled:
        # 预测ray-splat深度的成对平方差；背景为零，不使用GT法线/深度。
        # 独立稳定核返回原始非负项，不能clamp原核的数值抵消结果。
        terms['geometry_distortion'] = package['geometry_distortion'].mean()
    total = sum(getattr(args,'lambda_'+key)*value for key,value in terms.items())
    return total, terms
