"""原生PBR可选几何：只使用来源预测点和训练图像，不加载GT几何。"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def quaternion_normal(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)


def fit_source_plane(xyz, static, opacity, rotations, band=.08, seed=1010):
    """固定种子RANSAC/SVD，只拟合来源的静态高opacity预测位置。"""
    points = xyz.detach().cpu().double().numpy()
    static_mask = static.detach().cpu().numpy().astype(bool).reshape(-1)
    alpha = opacity.detach().cpu().numpy().reshape(-1)
    if (points.ndim != 2 or points.shape[1] != 3 or len(static_mask) != len(points) or
            len(alpha) != len(points) or not np.isfinite(points).all() or
            not np.isfinite(alpha).all() or not torch.isfinite(rotations).all()):
        raise ValueError('来源几何形状或有限性无效。')
    eligible = static_mask & (alpha > .5)
    if eligible.sum() < 32:
        raise ValueError('来源预测的静态高opacity点不足，不能拟合几何平面。')
    rng = np.random.RandomState(seed)
    ids = np.flatnonzero(eligible)
    cloud = points[rng.choice(ids, min(len(ids), 20000), replace=False)]
    best, score = None, -1
    threshold = band * .375
    for _ in range(500):
        tri = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        offset = -normal @ tri[0]
        mask = np.abs(cloud @ normal + offset) < threshold
        if int(mask.sum()) > score:
            best, score = (normal, offset, mask), int(mask.sum())
    if best is None or score / len(cloud) < .25:
        raise ValueError('来源点无稳定主平面；禁止硬编码世界法线。')
    normal, offset, mask = best
    for _ in range(3):
        selected = cloud[mask]
        # 分量中位数通常不在倾斜平面上，直接以其中心做SVD会给精确平面带来偏斜。
        center = selected.mean(axis=0)
        _, _, vh = np.linalg.svd(selected-center, full_matrices=False)
        normal = vh[-1]
        offset = -np.median(selected @ normal)
        mask = np.abs(cloud @ normal + offset) < threshold
    normal /= np.linalg.norm(normal)
    if normal[np.argmax(np.abs(normal))] < 0:
        normal, offset = -normal, -offset
    normals = quaternion_normal(rotations.detach()).cpu().double().numpy()
    # 高opacity仅用于可靠拟合；低opacity的真实表面片也要能跟随平面，不能只修少量高置信点。
    support = static_mask & (alpha > .05) & (np.abs(points @ normal + offset) <= band)
    support &= np.abs(normals @ normal) >= .5
    if support.sum() < 32:
        raise ValueError('来源主平面缺少方向一致的Gaussian支持。')
    record = dict(method='source_prediction_static_RANSAC_SVD', seed=seed,
                  source_points=len(points), eligible_points=int(eligible.sum()),
                  ransac_inlier_fraction=float(mask.mean()), plane_normal=normal.tolist(),
                  plane_offset=float(offset), band=band, support_points=int(support.sum()),
                  gt_normal_inputs=False, gt_geometry_inputs=False, gt_material_inputs=False)
    return (xyz.new_tensor(normal), xyz.new_tensor(offset),
            torch.as_tensor(support, device=xyz.device), record)


@torch.no_grad()
def project_geometry(xyz, rotations, source_xyz, source_rotations, distance, angle_degrees):
    """限制真实原生PLY参数；四元数采用最短弧，保持来源raw norm以稳定MLP偏置。"""
    delta = xyz-source_xyz
    norm = delta.norm(dim=-1, keepdim=True)
    xyz.copy_(source_xyz + delta * (distance/norm.clamp_min(1e-12)).clamp_max(1))
    q0 = F.normalize(source_rotations, dim=-1)
    q = F.normalize(rotations, dim=-1)
    dot = (q0*q).sum(-1, keepdim=True)
    q = torch.where(dot < 0, -q, q)
    dot = (q0*q).sum(-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    maximum = math.radians(angle_degrees)*.5
    t = (maximum/theta.clamp_min(1e-7)).clamp_max(1)
    denominator = theta.sin().clamp_min(1e-7)
    limited = ((1-t)*theta).sin()/denominator*q0 + (t*theta).sin()/denominator*q
    limited = torch.where(theta < 1e-6, q, limited)
    rotations.copy_(F.normalize(limited, dim=-1)*source_rotations.norm(dim=-1, keepdim=True))


class NativePBRGeometry:
    def __init__(self, pc, opt):
        self.opt = opt
        self.source_xyz = pc._xyz.detach().clone()
        self.source_rotations = pc._rotation.detach().clone()
        self.plane = self.offset = self.support = None
        self.record = dict(method='native_PLY_geometry', gt_normal_inputs=False,
                           gt_geometry_inputs=False, gt_material_inputs=False)
        if opt.lambda_pbr_plane_normal > 0 or opt.lambda_pbr_plane_position > 0:
            self.plane, self.offset, self.support, self.record = fit_source_plane(
                self.source_xyz, pc.get_binary_feature().reshape(-1) < .5,
                pc.get_opacity, self.source_rotations, opt.pbr_plane_band)

    def losses(self, pc, package, camera):
        opt = self.opt
        normal = F.normalize(package['rend_normal'], dim=0)
        depth_normal = F.normalize(package['surf_normal'], dim=0)
        alpha = package['rend_alpha']
        valid = (alpha.detach() > .5) & (package['rend_normal'].detach().norm(dim=0, keepdim=True) > 1e-6)
        valid &= package['surf_normal'].detach().norm(dim=0, keepdim=True) > 1e-6
        valid = (F.avg_pool2d(valid.float()[None], 3, stride=1, padding=1)[0] >= 1).detach()
        zero = pc._rotation.sum()*0
        consistency = ((1-(normal*depth_normal).sum(0, keepdim=True).clamp(-1, 1))*valid).sum()/valid.sum().clamp_min(1)
        delta = pc._xyz-self.source_xyz
        xyz_anchor = (delta/opt.pbr_geometry_max_displacement).square().mean()
        q = F.normalize(pc._rotation, dim=-1)
        q0 = F.normalize(self.source_rotations, dim=-1)
        rotation_anchor = (1-(q*q0).sum(-1).square().clamp_max(1)).mean()
        plane_normal = plane_position = zero
        if self.plane is not None:
            n = quaternion_normal(pc._rotation[self.support])
            plane_normal = (1-(n*self.plane).sum(-1).square().clamp_max(1)).mean()
            plane_position = ((pc._xyz[self.support] @ self.plane + self.offset)/opt.pbr_plane_band).square().mean()
        silhouette = (alpha-camera.gt_alpha_mask.to(alpha)).abs().mean()
        terms = dict(pbr_normal_depth=consistency, pbr_plane_normal=plane_normal,
                     pbr_plane_position=plane_position, pbr_xyz_anchor=xyz_anchor,
                     pbr_rotation_anchor=rotation_anchor, pbr_silhouette=silhouette)
        loss = (opt.lambda_pbr_normal_depth*consistency + opt.lambda_pbr_plane_normal*plane_normal +
                opt.lambda_pbr_plane_position*plane_position + opt.lambda_pbr_xyz_anchor*xyz_anchor +
                opt.lambda_pbr_rotation_anchor*rotation_anchor + opt.lambda_pbr_silhouette*silhouette)
        return loss, terms

    def project(self, pc):
        project_geometry(pc._xyz, pc._rotation, self.source_xyz, self.source_rotations,
                         self.opt.pbr_geometry_max_displacement, self.opt.pbr_geometry_max_angle)


def validate_pbr_geometry(args):
    if not getattr(args, 'pbr_optimize_geometry', False):
        return
    if args.render_mode != 'original_ir' or args.pbr_light_mode != 'gt_directional':
        raise ValueError('原生PBR几何微调仅适用于GT方向光原生PBR。')
    positive = ['pbr_geometry_xyz_lr', 'pbr_geometry_rotation_lr', 'pbr_geometry_max_displacement', 'pbr_plane_band']
    weights = ['lambda_pbr_normal_depth', 'lambda_pbr_plane_normal', 'lambda_pbr_plane_position',
               'lambda_pbr_xyz_anchor', 'lambda_pbr_rotation_anchor', 'lambda_pbr_silhouette']
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k) <= 0 for k in positive):
        raise ValueError('PBR几何学习率、信赖域、平面band必须有限且为正。')
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k) < 0 for k in weights):
        raise ValueError('PBR几何损失权重必须有限且非负。')
    if (args.pbr_geometry_warmup < 0 or args.pbr_geometry_warmup >= args.iterations-args.load_iter or
            not math.isfinite(args.pbr_geometry_max_angle) or not 0 < args.pbr_geometry_max_angle < 180):
        raise ValueError('PBR几何warmup/最大旋转角度不合法。')
