"""足部恢复配对的独立拓扑审计；不使用任何GT空间区域。"""
from pathlib import Path
import json

import torch

from scene.gaussian_model import GaussianModel


def opacity_summary(pc):
    """全点统计仅落盘，不参与损失或点选择。"""
    alpha = pc.get_opacity.detach().flatten()
    return dict(points=len(alpha), minimum=float(alpha.min()), maximum=float(alpha.max()),
                mean=float(alpha.mean()), below_001=int((alpha < .01).sum()),
                quantiles=torch.quantile(alpha, torch.tensor([.01, .1, .5, .9, .99],
                                     device=alpha.device)).cpu().tolist())


def opacity_adam_summary(pc):
    state = pc.optimizer.state.get(pc._opacity, {})
    return {key: (dict(shape=list(value.shape), norm=float(value.norm()),
                       nonzero=int(torch.count_nonzero(value))) if value.ndim else float(value))
            if torch.is_tensor(value) else value for key, value in state.items()}


class FootRecoveryGaussianModel(GaussianModel):
    """原生增密/裁剪算法的逐步同义实现，额外保存删除原因及原始点属性。"""

    @torch.no_grad()
    def densify_and_prune_audited(self, max_grad, min_opacity, extent, max_screen_size,
                                 output, step, iteration):
        before = len(self.get_xyz)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        clone = (torch.norm(grads, dim=-1) >= max_grad) & (
            self.get_scaling.max(dim=1).values <= self.percent_dense * extent)
        clone_count = int(clone.sum())
        self.densify_and_clone(grads, max_grad, extent)
        after_clone = len(self.get_xyz)
        padded = torch.zeros(after_clone, device=self.get_xyz.device)
        padded[:grads.shape[0]] = grads.squeeze()
        split = (padded >= max_grad) & (
            self.get_scaling.max(dim=1).values > self.percent_dense * extent)
        split_count = int(split.sum())
        split_parent_xyz = self._xyz[split].detach().cpu()
        self.densify_and_split(grads, max_grad, extent)
        after_split = len(self.get_xyz)
        low = (self.get_opacity < min_opacity).squeeze(-1)
        screen = torch.zeros_like(low)
        world = torch.zeros_like(low)
        if max_screen_size:
            screen = self.max_radii2D > max_screen_size
            world = self.get_scaling.max(dim=1).values > .1 * extent
        union = low | screen | world
        indices = torch.nonzero(union, as_tuple=False).flatten()
        payload = dict(candidate_index=indices.cpu(), xyz=self._xyz[union].detach().cpu(),
                       opacity=self.get_opacity[union].detach().cpu(),
                       scale=self.get_scaling[union].detach().cpu(),
                       low_opacity=low[union].cpu(), screen_size=screen[union].cpu(),
                       world_size=world[union].cpu(), split_parent_xyz=split_parent_xyz)
        path = Path(output) / 'topology_deleted' / f'iteration_{iteration}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        record = dict(step=step, iteration=iteration, before=before,
                      cloned_parents=clone_count, after_clone=after_clone,
                      split_parents_removed=split_count, split_children=2 * split_count,
                      after_split=after_split, low_opacity=int(low.sum()),
                      screen_size=int(screen.sum()), world_size=int(world.sum()),
                      low_and_screen=int((low & screen).sum()),
                      low_and_world=int((low & world).sum()),
                      screen_and_world=int((screen & world).sum()),
                      all_three=int((low & screen & world).sum()),
                      prune_union=int(union.sum()), deleted_points_path=str(path),
                      min_opacity=min_opacity, max_screen_size=max_screen_size,
                      screen_statistic='保持原算法：clone/split会将max_radii2D重置为零',
                      no_gt_region_used=True)
        self.prune_points(union)
        record.update(after=len(self.get_xyz), opacity_after=opacity_summary(self))
        if after_clone != before + clone_count or after_split != after_clone + split_count:
            raise AssertionError('原增密点数恒等式不成立。')
        if record['after'] != after_split - record['prune_union']:
            raise AssertionError('裁剪并集与点数不一致。')
        with (Path(output) / 'topology_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        torch.cuda.empty_cache()
        return record


@torch.no_grad()
def audit_topology_equivalence(ply_path, pipe, extent, output):
    """真实PLY的两个隔离副本；构造算子覆盖值，绝不写回训练模型。"""
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    models = []
    checks = {}
    try:
        for cls in (GaussianModel, FootRecoveryGaussianModel):
            model = cls(pipe.sh_degree, no_binary_separation=True, fea_dim=pipe.hyper_dim)
            model.load_ply(str(ply_path))
            del model._albedo_dc_stage1
            model.training_setup(pipe)
            model.max_radii2D = torch.zeros(len(model.get_xyz), device='cuda')
            model.xyz_gradient_accum.zero_()
            model.denom.fill_(1.)
            # 明确合成的算子测试：小点clone、大点split、低opacity删除、世界尺度删除。
            model._scaling[0].fill_(torch.tensor(model.percent_dense * extent * .5).log())
            model._scaling[1].fill_(torch.tensor(model.percent_dense * extent * 2.).log())
            model._scaling[2].fill_(torch.tensor(extent * .2).log())
            model._opacity[:3].fill_(0.)
            model._opacity[3].fill_(-12.)
            model.xyz_gradient_accum[:2].fill_(pipe.densify_grad_threshold * 2.)
            model.max_radii2D[4] = 100.
            # 新建Adam不会自然已有moment；此处只为检验结构编辑对已有状态的同义映射。
            for group in model.optimizer.param_groups:
                for parameter in group['params']:
                    model.optimizer.state[parameter] = dict(step=torch.tensor(7.),
                        exp_avg=torch.full_like(parameter, .001),
                        exp_avg_sq=torch.full_like(parameter, .002))
            models.append(model)
        native, audited = models
        before_count = len(native.get_xyz)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        native.densify_and_prune(pipe.densify_grad_threshold, pipe.min_opacity, extent, 20)
        native_cpu_rng, native_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        test_output = Path(output) / 'topology_operator_audit'
        test_output.mkdir(parents=True, exist_ok=False)
        record = audited.densify_and_prune_audited(pipe.densify_grad_threshold, pipe.min_opacity,
                                                  extent, 20, test_output, 0, 0)
        for name in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc',
                     '_albedo_rest', '_roughness', 'feature', 'xyz_gradient_accum',
                     'denom', 'max_radii2D'):
            a, b = getattr(native, name), getattr(audited, name)
            checks['tensor_' + name] = dict(passed=bool(torch.equal(a, b)), shape=list(a.shape))
        first, second = native.optimizer.state_dict(), audited.optimizer.state_dict()
        checks['adam_groups'] = dict(passed=first['param_groups'] == second['param_groups'])
        checks['adam_state_keys'] = dict(passed=first['state'].keys() == second['state'].keys())
        for index, state in first['state'].items():
            for key, value in state.items():
                other = second['state'][index][key]
                checks[f'adam_{index}_{key}'] = dict(passed=bool(torch.equal(value, other))
                                                     if torch.is_tensor(value) else value == other)
        checks['rng_same_after_operator'] = dict(passed=bool(
            torch.equal(native_cpu_rng, torch.get_rng_state())
            and torch.equal(native_cuda_rng, torch.cuda.get_rng_state())))
        checks['all_constructed_branches_covered'] = dict(passed=bool(
            record['cloned_parents'] >= 1 and record['split_parents_removed'] >= 1
            and record['low_opacity'] >= 1 and record['world_size'] >= 1
            and record['screen_size'] == 0),
            screen_reason='原算法clone/split重置max_radii2D，故合成大屏幕值同样清零')
        checks['point_counts_same'] = dict(passed=len(native.get_xyz) == len(audited.get_xyz),
                                          before=before_count, after=len(native.get_xyz))
    finally:
        models.clear()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        torch.cuda.empty_cache()
    return dict(status='PASS' if all(x['passed'] for x in checks.values()) else 'FAILED',
                checks=checks, operator_event=record, optimizer_updates=0,
                no_gt_region_used=True, training_model_modified=False,
                scope='真实PLY副本、合成增密统计和Adam状态的算子同义性；不是额外训练或几何修复效果',
                restored_training_rng=True)
