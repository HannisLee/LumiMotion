"""透明度重置窗口的只读拓扑谱系；训练操作全部委托冻结GaussianModel。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scene.gaussian_model import GaussianModel


class AuditedGaussianModel(GaussianModel):
    def start_topology_audit(self, directory, iteration):
        self.audit_directory = Path(directory)
        self.audit_directory.mkdir(parents=True, exist_ok=False)
        self.audit_ids = torch.arange(len(self.get_xyz), device=self.get_xyz.device, dtype=torch.int64)
        self.audit_next_id = len(self.get_xyz)
        self.audit_iteration = iteration
        self.audit_phase = 'idle'
        self.audit_serial = 0
        self.audit_parent_ids = None
        self.audit_context = None
        self.audit_state('initial')

    def audit_state(self, label, **extra):
        arrays = dict(uid=self.audit_ids, xyz=self.get_xyz, opacity=self.get_opacity,
                      scaling=self.get_scaling, rotation=self._rotation)
        return self._audit_write(label, arrays, **extra)

    def _audit_write(self, kind, arrays, **extra):
        # CPU序列化无随机采样，不改模型、优化器、已有梯度或增密累计。
        name = f'{self.audit_iteration:06d}_{self.audit_serial:04d}_{kind}.npz'
        self.audit_serial += 1
        np.savez_compressed(self.audit_directory / name,
                            **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
        record = dict(iteration=self.audit_iteration, kind=kind, arrays=name, **extra)
        with (self.audit_directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None):
        # 精确复算原选择，仅取父UID；传给原方法的参数不变。
        selected = selected_pts_mask
        if selected is None:
            selected = ((torch.norm(grads, dim=-1) >= grad_threshold)
                        & (self.get_scaling.max(dim=1).values <= self.percent_dense * scene_extent))
        self.audit_phase = 'clone'
        self.audit_parent_ids = self.audit_ids[selected]
        try:
            return super().densify_and_clone(grads, grad_threshold, scene_extent, selected_pts_mask)
        finally:
            self.audit_phase, self.audit_parent_ids = 'filter_prune', None

    def densify_and_split(self, grads=None, grad_threshold=None, scene_extent=None, N=2,
                          selected_pts_mask=None, without_prune=False):
        selected = selected_pts_mask
        if selected is None:
            padded = torch.zeros(len(self.get_xyz), device=self.get_xyz.device)
            padded[:grads.shape[0]] = grads.squeeze()
            selected = ((padded >= grad_threshold)
                        & (self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent))
        self.audit_phase = 'split'
        self.audit_parent_ids = self.audit_ids[selected].repeat(N)
        try:
            return super().densify_and_split(grads, grad_threshold, scene_extent, N,
                                             selected_pts_mask, without_prune)
        finally:
            self.audit_phase, self.audit_parent_ids = 'filter_prune', None

    def densification_postfix(self, *args, **kwargs):
        before = len(self.get_xyz)
        super().densification_postfix(*args, **kwargs)
        count = len(self.get_xyz) - before
        newborn = torch.arange(self.audit_next_id, self.audit_next_id + count,
                               device=self.get_xyz.device, dtype=torch.int64)
        if self.audit_parent_ids is None or len(self.audit_parent_ids) != count:
            raise AssertionError('新生点与原clone/split父点排列不符。')
        self.audit_ids = torch.cat((self.audit_ids, newborn))
        self.audit_next_id += count
        self._audit_write(self.audit_phase + '_birth',
                          dict(uid=newborn, parent_uid=self.audit_parent_ids,
                               xyz=self.get_xyz[before:], opacity=self.get_opacity[before:]),
                          count=count)

    def prune_points(self, mask):
        if self.audit_context is None:
            raise AssertionError('未记录上下文的拓扑删除。')
        arrays = dict(uid=self.audit_ids[mask], xyz=self.get_xyz[mask],
                      opacity=self.get_opacity[mask], scaling=self.get_scaling[mask])
        count = int(mask.sum())
        before = len(self.get_xyz)
        if self.audit_phase == 'split':
            kind, reasons = 'split_parent_replacement', {}
        else:
            minimum, extent, screen = self.audit_context
            low = (self.get_opacity < minimum).squeeze(-1)
            big_screen = (self.max_radii2D > screen) if screen else torch.zeros_like(low)
            big_world = (self.get_scaling.max(dim=1).values > .1 * extent) if screen else torch.zeros_like(low)
            if not torch.equal(mask, low | big_screen | big_world):
                raise AssertionError('日志复算裁剪原因与原生实际删除mask不符。')
            arrays.update(low_opacity=low[mask], big_screen=big_screen[mask], big_world=big_world[mask])
            kind = 'filter_prune'
            reasons = dict(low_opacity=int(low.sum()), big_screen=int(big_screen.sum()),
                           big_world=int(big_world.sum()), union=count,
                           overlap=int((low.int() + big_screen.int() + big_world.int() > 1).sum()))
        self._audit_write(kind, arrays, count=count, before=before, after=before-count, reasons=reasons)
        keep_ids = self.audit_ids[~mask]
        super().prune_points(mask)
        self.audit_ids = keep_ids
        if len(self.audit_ids) != len(self.get_xyz):
            raise AssertionError('UID与原模型点数失配。')

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        before = len(self.get_xyz)
        self.audit_context = (min_opacity, extent, max_screen_size)
        self.audit_phase = 'filter_prune'
        try:
            super().densify_and_prune(max_grad, min_opacity, extent, max_screen_size)
            self.audit_state('after_topology', before=before, after=len(self.get_xyz))
        finally:
            self.audit_context, self.audit_phase = None, 'idle'

    def reset_opacity(self):
        self.audit_state('pre_reset')
        super().reset_opacity()
        self.audit_state('post_reset')


def preserved_gradient_state(pc, lights):
    """零更新路由审计前后核对已有参数梯度与增密累计；不读取GT。"""
    values = [(g['name'], g['params'][0].grad) for g in pc.optimizer.param_groups]
    values += [('light', lights.log_intensity.grad), ('xyz_gradient_accum', pc.xyz_gradient_accum),
               ('denom', pc.denom), ('max_radii2D', pc.max_radii2D)]
    return {name: None if value is None else hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for name, value in values}
