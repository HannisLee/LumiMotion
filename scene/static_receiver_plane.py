"""冻结训练接收面支持上的弱法线朝向先验，不约束平面高度。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class FixedReceiverPlane:
    def __init__(self, source):
        source = Path(source).resolve()
        manifest = json.loads((source/'no_gt_manifest.json').read_text())
        self.metadata = manifest['receiver_plane_prior']
        path = (source/self.metadata['mask_path']).resolve()
        if not path.is_relative_to(source):
            raise ValueError('固定接收面掩膜必须属于本次无GT训练源。')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.metadata['mask_sha256'] or digest != manifest['files'][str(path.relative_to(source))]:
            raise RuntimeError('接收面掩膜SHA与训练源绑定不一致。')
        if self.metadata['gt_read'] or self.metadata['support_status'] != 'PASS_FIXED_REFERENCE_VISIBLE_PLANE':
            raise RuntimeError('接收面支持尚未通过固定参考可见性准入。')
        with np.load(path, allow_pickle=False) as data:
            expected = {f'frame_{i:04d}' for i in range(1,151) if i%10}
            if set(data.files) != expected:
                raise ValueError('接收面支持仅允许原135训练时刻。')
            self.masks = {}
            for name in data.files:
                value = data[name]
                if value.dtype != np.bool_ or value.ndim != 2 or not value.any():
                    raise ValueError('每个训练支持须为固定非空二维布尔掩膜。')
                self.masks[name] = torch.from_numpy(value.copy())
        normal = torch.tensor(self.metadata['normal_world'], dtype=torch.float32, device='cuda')
        if not torch.isfinite(normal).all() or abs(float(normal.norm())-1)>1e-4:
            raise ValueError('预测接收面世界法向非有限或未单位化。')
        self.normal = F.normalize(normal, dim=0).detach()

    def __call__(self, camera, geometry):
        name = Path(camera.image_name_train_light).stem
        mask = self.masks[name].to(device=geometry['rend_normal'].device)
        if mask.shape != geometry['rend_normal'].shape[1:]:
            raise ValueError('固定掩膜分辨率与真实训练渲染不同。')
        gs = F.normalize(geometry['rend_normal'], dim=0)
        depth = F.normalize(geometry['surf_normal'], dim=0)
        dot_gs = (gs*self.normal[:,None,None]).sum(0).clamp(-1,1)
        dot_depth = (depth*self.normal[:,None,None]).sum(0).clamp(-1,1)
        # 分母和支持固定；零法向保留常数惩罚，不通过当前有效性或opacity重选。
        loss = ((.5*(1-dot_gs.square())+.5*(1-dot_depth.square()))*mask).sum()/mask.sum()
        details = {'fixed_support_pixels':int(mask.sum()),
                   'missing_gs_normals':int(((geometry['rend_normal'].norm(dim=0)<1e-6)&mask).sum()),
                   'missing_depth_normals':int(((geometry['surf_normal'].norm(dim=0)<1e-6)&mask).sum())}
        return loss, details


def audit_plane_gradient(loss, normal_loss, pc, lights, screens, destination, iteration):
    """只读核验弱性和断路；恢复非叶screen缓存，禁止污染随后增密。"""
    params = [pc._xyz,pc._rotation,pc._scaling,pc._opacity,pc._albedo_dc,lights.log_intensity]
    saved = [None if x.grad is None else x.grad.detach().clone() for x in screens]
    versions = [x._version for x in params]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    try:
        gradients = torch.autograd.grad(loss,params+screens,retain_graph=True,allow_unused=True)
        reference = torch.autograd.grad(normal_loss,params+screens,retain_graph=True,allow_unused=True)
    finally:
        for x,value in zip(screens,saved):
            x.grad = value
    def norm(value):
        return 0. if value is None else float(value.norm())
    norms = {name:{'plane':norm(a),'normal':norm(b)} for name,a,b in
             zip(['xyz','rotation','scaling','opacity','albedo_dc','log_intensity','photo_screen','geometry_screen'],gradients,reference)}
    geometry_norm = sum(norm(g)**2 for g in gradients[:3])**.5
    reference_norm = sum(norm(g)**2 for g in reference[:3])**.5
    ratio = geometry_norm/max(reference_norm,1e-12)
    forbidden_absent = all(g is None or bool(torch.count_nonzero(g)==0) for g in gradients[3:6])
    unchanged = (versions==[x._version for x in params] and torch.equal(cpu_rng,torch.get_rng_state())
                 and torch.equal(cuda_rng,torch.cuda.get_rng_state()))
    finite = all(g is None or bool(torch.isfinite(g).all()) for g in gradients+reference)
    record = {'iteration':iteration,'component_norms':norms,'geometry_relative_l2_to_normal':ratio,
              'opacity_material_intensity_absent':forbidden_absent,'screen_grad_cache_restored':True,
              'parameter_versions_and_rng_unchanged':unchanged,'all_finite':finite,
              'passed':finite and unchanged and forbidden_absent and 0<ratio<=1.,
              'scope':'只核验固定弱项的梯度路径及规模，不证明最终几何收益；不使用高度损失。'}
    Path(destination).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    if not record['passed']:
        raise RuntimeError('接收面弱先验梯度准入失败；不现场调权，保留失败现场。')
