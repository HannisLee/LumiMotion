"""独立检查Stage3继续的父状态、累计数、冻结范围及Expected先验。"""
import argparse
from copy import copy
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
import torch
import torch.nn.functional as F

from scene.three_stage_pipeline import sha, write


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    marker = read(a.model/'three_stage_training.json')
    config = marker['config']
    assert marker['stage'] == 3 and marker['stage3_continuation'] and marker['parent_stage'] == 3
    assert config['iterations'] == (20 if marker['smoke'] else 4050)
    parent, pi = Path(marker['parent']), marker['parent_iteration']
    assert pi == 4050 and marker['parent_cumulative_actual_updates'] == 51199
    ps = load(parent/f'three_stage/iteration_{pi}/state.pt')
    pd = load(parent/f'deform/iteration_{pi}/deform.pth')
    pp = PlyData.read(parent/f'point_cloud/iteration_{pi}/point_cloud.ply')['vertex'].data
    assert len(pd) == 32
    appearance_fields = {'albedo_dc_0', 'albedo_dc_1', 'albedo_dc_2', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'roughness'}
    geo_fields = {k for k in pp.dtype.names if k in {'x', 'y', 'z', 'opacity'} or k.startswith(('scale_', 'rot_'))}
    assert len(geo_fields) == 10
    groups_expected = ['albedo', 'roughness', 'environment', 'residual_amplitude', 'xyz', 'rotation', 'scale', 'opacity']
    geo_lr = dict(xyz=1e-7, rotation=5e-6, scale=1e-6, opacity=1e-4)
    frozen = config['geometry_continuation'] == 'frozen'
    rows = []
    for step in a.steps:
        directory = a.model/f'three_stage/iteration_{step}'
        receipt = read(directory/'receipt.json')
        paths = [(directory/'state.pt', receipt['checkpoint_sha256']),
                 (a.model/f'point_cloud/iteration_{step}/point_cloud.ply', receipt['ply_sha256']),
                 (a.model/f'deform/iteration_{step}/deform.pth', receipt['deform_sha256'])]
        assert all(sha(path) == digest for path, digest in paths)
        state, deform = load(paths[0][0]), load(paths[2][0])
        ply = PlyData.read(paths[1][0])['vertex'].data
        assert state['config'] == config and state['completed_updates'] == step
        assert state['cumulative_actual_updates'] == receipt['cumulative_actual_updates'] == 51199+step
        assert equal(state['lighting'], ps['lighting']) and equal(deform, pd)
        assert state['stats'] == ps['stats'] and state['binding'] == ps['binding'] and state['env_initial'] == ps['env_initial']
        assert ply.dtype == pp.dtype and ply.shape == pp.shape
        allowed = appearance_fields | (set() if frozen else geo_fields)
        unchanged = [k for k in pp.dtype.names if k not in allowed]
        assert all(np.array_equal(ply[k], pp[k]) for k in unchanged)
        groups = state['optimizer']['param_groups']
        assert [g['name'] for g in groups] == groups_expected
        assert all(g['lr'] == pg['lr'] for g, pg in zip(groups[:4], ps['optimizer']['param_groups']))
        assert all(g['lr'] == (0. if frozen else geo_lr[g['name']]) for g in groups[4:])
        for index, group in enumerate(groups):
            assert len(group['params']) == 1
            key = group['params'][0]
            if step or index < 4:
                opt = state['optimizer']['state'][key]
                expected_step = step+(4050 if index < 4 else 0)
                assert float(opt['step']) == expected_step
                assert all(torch.isfinite(v).all() for v in opt.values() if torch.is_tensor(v))
            else:
                assert key not in state['optimizer']['state']
        if step == 0:
            assert all(np.array_equal(ply[k], pp[k]) for k in pp.dtype.names)
            assert equal(state['env'], ps['env']) and equal(state['residual_raw'], ps['residual_raw'])
            for pg, group in zip(ps['optimizer']['param_groups'], groups[:4]):
                assert group == pg
                assert equal(state['optimizer']['state'][group['params'][0]], ps['optimizer']['state'][pg['params'][0]])
            assert all(equal(state[k], ps[k]) for k in ['torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'])
        delta = {k: float(np.max(np.abs(ply[k]-pp[k]))) for k in geo_fields}
        if frozen:
            assert max(delta.values()) == 0
        elif step:
            assert all(max(delta[k] for k in keys) > 0 for keys in [
                ['x', 'y', 'z'], ['rot_0', 'rot_1', 'rot_2', 'rot_3'], ['scale_0', 'scale_1'], ['opacity']])
        rows.append(dict(iteration=step, cumulative_actual_updates=51199+step,
            checkpoint_sha256=receipt['checkpoint_sha256'], all_32_mlp_equal=True, lighting_equal=True,
            unchanged_ply_fields=unchanged, geometry_max_drift=delta,
            optimizer_names=groups_expected, parent_adam_and_rng_equal=(True if step == 0 else None)))
    log = [json.loads(line) for line in (a.model.parent/'metrics.jsonl').read_text().splitlines()]
    assert log
    for row in log:
        expected = row['photo_alpha']+.5*row['normal']+.01*row['env_anchor']+.01*row['amplitude_anchor']
        assert abs(row['loss']-expected) < 2e-6
        assert row['normal_depth_ratio'] == 0 and row['pbr_depth_ratio'] == 1 and row['normal_weight'] == .5
        assert row['normal_opacity_protected'] == (row['step'] <= config['refine_protect_end'])
        ng = row['geometry_branch_gradients']['opacity']
        assert ng['photo_alpha_gradient_norm'] > 0
        assert ng['normal_gradient_norm'] == 0 if row['normal_opacity_protected'] else ng['normal_gradient_norm'] > 0
    write(a.output/'state_receipt.json', dict(status='PASS_CONTINUATION_STATE_SCOPE_ADAM_RNG_AND_COUNTS',
        rows=rows, geometry_frozen=frozen, source_sha256=sha(__file__), optimizer_updates_in_audit=0,
        gt_read=False, test_read=False))
    # 独立几何光栅重新计算同一先验，既不读取GT也不改变PBR深度语义。
    from scene.three_stage_native_pbr import load_three_stage_model
    from scripts.train_three_stage import load_cameras, install_guard
    from scripts.train_three_stage_refine import expected_geometry_loss
    model = load_three_stage_model(a.model, a.steps[-1])
    install_guard(model.binding['source'], a.output, a.model, model.binding)
    for key in ('_xyz', '_rotation', '_scaling', '_opacity'):
        getattr(model.pc, key).requires_grad_(True)
    camera = load_cameras(model.binding['source'])[0]
    camera.load2device()
    package = model.render(camera, a.steps[-1], training=True)
    from gaussian_renderer import render
    pipe = copy(model.pipe)
    pipe.depth_ratio = 0.
    pc = model.effective(camera)
    geometry = render(camera, pc, pipe, pc.get_xyz.new_zeros(3), 0., 0., 0., override_color=torch.zeros_like(pc.get_xyz))
    inside = -F.max_pool2d(-(camera.gt_alpha_mask.cuda() > .99).float()[None], 5, stride=1, padding=2)[0]
    mask = inside*(package['rend_alpha'].detach() > .5)
    cos = (F.normalize(geometry['rend_normal'], dim=0)*F.normalize(geometry['surf_normal'], dim=0)).sum(0, keepdim=True)
    independent = ((1-cos)*mask).sum()/mask.sum().clamp_min(1)
    actual = expected_geometry_loss(model, camera, package, False)
    assert float((independent-actual).abs()) == 0 and model.pipe.depth_ratio == 1.
    ng = torch.autograd.grad(.5*actual, model.pc._opacity, retain_graph=True)[0]
    protected = expected_geometry_loss(model, camera, package, True)
    pg = torch.autograd.grad(.5*protected, model.pc._opacity, retain_graph=True, allow_unused=True)[0]
    assert ng.isfinite().all() and ng.norm() > 0 and (pg is None or pg.abs().max() == 0)
    from scene.three_stage_lighting import photometric_objective
    photo, _ = photometric_objective(package, camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda(),
        camera.observed_saturation_mask.cuda(), model.stats, support=package['loss_support'])
    photo = photo+.1*F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6, 1-1e-6), camera.gt_alpha_mask.cuda())
    photo_opacity = torch.autograd.grad(photo, model.pc._opacity)[0]
    assert photo_opacity.isfinite().all() and photo_opacity.norm() > 0
    write(a.output/'normal_receipt.json', dict(status='PASS_INDEPENDENT_EXPECTED_PRIOR_AND_BRANCH_PROTECTION',
        independent_loss=float(independent), actual_loss=float(actual), weight=.5, prior_depth_ratio=0.,
        pbr_depth_ratio=model.pipe.depth_ratio, unprotected_opacity_gradient_norm=float(ng.norm()),
        protected_opacity_gradient_norm=0. if pg is None else float(pg.norm()),
        photo_alpha_opacity_gradient_norm=float(photo_opacity.norm()), optimizer_updates=0, gt_read=False, test_read=False))
    print('PASS_STAGE3_CONTINUATION_STATE_AND_EXPECTED_PRIOR', flush=True)


if __name__ == '__main__':
    main()
