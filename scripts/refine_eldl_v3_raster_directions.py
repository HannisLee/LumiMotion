"""冻结I00几何与材质，在TRAIN全图及真实BVH下做一次有界方向细化检验。"""
import argparse
import math
from pathlib import Path
import shlex
import socket
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import calibrate_eldl_v3_raster as f


def normalized(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def penalties(directions, original, frames):
    if (np.asarray(directions).shape != np.asarray(original).shape or
            np.asarray(directions).shape != (len(frames), 3) or len(frames) < 3 or
            not np.isfinite(directions).all() or not np.isfinite(original).all() or
            (np.linalg.norm(directions, axis=-1) <= 0).any() or
            (np.linalg.norm(original, axis=-1) <= 0).any() or
            not np.isfinite(frames).all() or (np.diff(frames) <= 0).any()):
        raise ValueError('方向必须有限非零，时间必须严格递增且至少三点。')
    unit, origin = normalized(directions), normalized(original)
    dt = np.diff(frames).astype(np.float64)
    velocity = np.diff(unit, axis=0) / dt[:, None]
    weight = (dt[1:] + dt[:-1]) / 2
    acceleration = np.diff(velocity, axis=0) / weight[:, None]
    roughness = float(np.sum(np.sum(acceleration**2, axis=1)*weight)/weight.sum())
    angle = np.arccos(np.clip(np.sum(unit*origin, axis=1), -1, 1))
    drift = float(np.mean((angle/math.radians(5))**2))
    return roughness, drift, np.degrees(angle)


def candidates(direction, degrees):
    unit = normalized(direction)
    axis = np.eye(3)[int(np.argmin(np.abs(unit)))]
    first = normalized(np.cross(axis, unit))
    second = np.cross(unit, first)
    angle = math.radians(degrees)
    return [(math.cos(angle)*unit + math.sin(angle)*basis).astype(np.float32).astype(np.float64)
            for basis in (first, -first, second, -second)]


def self_check():
    frame = np.array([1, 2, 4, 26, 27])
    old = normalized(np.array([[.1, .2, 1], [.2, .1, 1], [.15, .25, 1], [.3, .1, 1], [.2, .2, 1.]]))
    a, b, angles = penalties(old, old, frame)
    assert a > 0 and b < 1e-12 and angles.max() < 2e-6
    for candidate in candidates(old[0], 2):
        angle = np.degrees(np.arccos(np.clip(normalized(candidate) @ old[0], -1, 1)))
        assert abs(angle-2) < 2e-5
    flat = np.repeat([[0., 0., 1.]], 5, axis=0)
    assert penalties(flat, flat, frame)[0] == 0
    # 手算非均匀时刻，避免测试只重复实现：速度(1,0,-1)、(-.5,0,.5)，
    # 平均间隔1.5，因此加速度(-1,0,1)，模平方为2。
    corner = np.array([[0.,0.,1.],[1.,0.,0.],[0.,0.,1.]])
    assert abs(penalties(corner, corner, np.array([0,1,3]))[0]-2.) < 1e-12
    try:
        penalties(corner, corner, np.array([0,1,1]))
    except ValueError:
        pass
    else:
        raise AssertionError('重复时间必须拒绝。')
    known_times = torch.tensor(v.FIT, dtype=torch.float64)
    known = torch.tensor(np.repeat([[0.,0.,1.]],117,axis=0), dtype=torch.float64)
    known[v.FIT.index(81)] = torch.tensor([.2,0.,math.sqrt(.96)],dtype=torch.float64)
    interpolated = v.interpolate_frozen(torch.tensor([61.,79.]), known_times, known)
    left, right = known[v.FIT.index(59)], known[v.FIT.index(81)]
    for row, frame_id in zip(interpolated, (61,79)):
        expected = torch.nn.functional.normalize(left+(right-left)*((frame_id-59)/22),dim=0)
        assert torch.equal(row,expected)
    print('PASS：世界球面候选、真实非均匀dt手算、非法时间拒绝、零偏离及59→81边界nlerp；无GPU前向。')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'fit', 'model', 'source', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--iteration', type=int, default=16000)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--cpu-check', action='store_true', help='只核验输入/来源和划分，不加载相机或运行GPU前向')
    a = parser.parse_args()
    if a.self_check:
        self_check()
        return
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    assert all(getattr(a, name) is not None for name in ('cache','fit','model','source','checkpoint','output'))
    assert socket.gethostname() == 'ushas' and Path(sys.executable).resolve() == v.PYTHON.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    assert a.checkpoint == a.cache.parent.parent/'raster_calibration_v2_anneal/init_3/checkpoint_epoch_30.pt'
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.parent.parent, a.model, a.source, a.output])
    meta = v.read(a.checkpoint.parent/'configuration.json')
    assert not meta['gt_read'] and meta['geometry_frozen'] and meta['directions_frozen'] and not meta['constant']
    assert meta['external_test_read'] is False and meta['geometry_fit_frames'] == v.FIT and meta['held_frames'] == v.HELD
    selection_path = a.checkpoint.parent.parent/'endpoint_selection.json'
    selection = v.read(selection_path)
    assert (selection['status'] == 'FIT_BUDGET_PASS' and selection['selected'] == 'init_3' and
            selection['selected_by'] == 'terminal_fit_mse_only' and selection['held_used'] is False and
            selection['epoch'] == 30 and selection['cumulative_epochs'] == 90)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 30 and saved['cumulative_epoch'] == 90 and saved['constant'] is False
    checkpoint_sha = v.sha(a.checkpoint)
    fit_record = v.read(a.checkpoint.parent/'fit_epoch_30.json')
    old_fit = fit_record['frame_mean']
    assert len(fit_record['per_frame']) == len(v.FIT) and np.isfinite(fit_record['per_frame']).all()
    assert abs(float(np.mean(fit_record['per_frame']))-old_fit) < 1e-12 and old_fit > 0
    fixed_mask_path = a.checkpoint.parent/'fixed_foreground.pt'
    assert v.sha(fixed_mask_path) == meta['foreground_sha256']
    fixed_masks = torch.load(fixed_mask_path, map_location='cpu', weights_only=False)
    assert fixed_masks['frames'] == v.TRAIN and fixed_masks['foreground'].dtype == torch.bool
    assert fixed_masks['foreground'].shape == (135,1,400,400)
    assert torch.isfinite(saved['rho']).all() and torch.isfinite(saved['environment']).all()
    assert ((saved['rho'] >= meta['rho_linear_bounds'][0]) & (saved['rho'] <= meta['rho_linear_bounds'][1])).all()
    assert saved['environment'].shape == (3,) and (saved['environment'] > 0).all()
    binding_files = [a.checkpoint, selection_path, a.checkpoint.parent/'configuration.json',
        a.checkpoint.parent/'fit_epoch_30.json',fixed_mask_path,Path(__file__).resolve(),
        Path(f.__file__).resolve(),ROOT/'scene/eldl_v3_photometric.py',
        ROOT/'scene/stage2_photometric.py',ROOT/'gaussian_renderer/__init__.py',
        ROOT/'utils/general_utils.py',ROOT/'scene/cameras.py',ROOT/'utils/rgba16.py']
    bindings = {str(path):v.sha(path) for path in binding_files}
    if a.cpu_check:
        data,state,unused_summary,unused_result,hashes = v.load_inputs(a)
        assert torch.equal(saved['directions'],state['directions'])
        shared = set(hashes).intersection(meta['input_sha256'])
        assert shared and all(hashes[path] == meta['input_sha256'][path] for path in shared)
        v.check_hashes({**hashes,**bindings})
        v.write(a.output/'cpu_review_receipt.json',dict(status='PASS_CPU_INPUTS_ONLY_GPU_NOT_RUN',
            input_sha256={**hashes,**bindings},fit_frames=v.FIT,development_frames=v.HELD,
            loader_policy='生产公共loader将解码135 TRAIN含18开发图；本CPU检查未加载相机，评分/停止只117fit',
            held_scored=False,held_used_for_selection=False,gt_read=False,external_test_read=False,
            optimization_run=False,production_parity='PENDING_GPU_ZERO_UPDATE',
            command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])))
        print('PASS：CPU输入/来源/固定mask/117fit18开发划分核验；未运行GPU或优化。')
        return
    pc, cameras, pipe, data, state, hashes = f.load(a)
    shared = set(hashes).intersection(meta['input_sha256'])
    assert shared and all(hashes[path] == meta['input_sha256'][path] for path in shared)
    hashes.update(bindings)
    assert torch.equal(saved['directions'], state['directions'])
    assert torch.equal(torch.stack([(camera.gt_alpha_mask>.5).cpu() for camera in cameras]),fixed_masks['foreground'])
    from scene.static_dl_photometric import StaticVisibility
    from scene.eldl_v3_photometric import render_photometric
    from scene.stage2_photometric import linear_to_srgb
    from utils.general_utils import build_rotation
    rho, environment = saved['rho'].cuda(), saved['environment'].cuda()
    frozen_rho, frozen_environment = rho.clone(), environment.clone()
    assert rho.shape == pc.get_xyz.shape
    trace = StaticVisibility(bias=float(data['shadow_bias']), update_interval=1000000)
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    fit_indices = [i for i, frame in enumerate(v.TRAIN) if frame in v.FIT]
    frames = np.asarray(v.FIT)
    original = saved['directions'][fit_indices].double().numpy()
    directions = original.copy()
    plane_normal = normalized(data['normal'].cpu().double().numpy())
    geometry = {key:getattr(pc, key).clone() for key in ('_xyz','_rotation','_scaling','_opacity')}
    class MaterialProxy:
        @property
        def get_albedo(self):
            return linear_to_srgb(rho)
        def __getattr__(self, key):
            return getattr(pc, key)
    parity = []
    calls = dict(candidate=0, complete_fit=0)
    def score(index, light, check=False):
        camera = cameras[fit_indices[index]]
        light = torch.as_tensor(light, device='cuda', dtype=torch.float32).reshape(1, 3)
        visibility = trace(pc, light, 0)
        facing = torch.where((normal*(camera.camera_center-pc.get_xyz)).sum(-1, keepdim=True)>=0, 1., -1.)
        q = ((normal*facing) @ light.T).clamp_min(0)*visibility
        image = f.render(camera, pc, pipe, rho*(environment/math.pi+q))['render']
        if check:
            lights = SimpleNamespace(ambient_rgb=environment, for_camera=lambda unused:(light, light.new_tensor([math.pi])))
            production = render_photometric(camera, MaterialProxy(), pipe, torch.zeros(3,device='cuda'), lights, lambda *unused:visibility, 0)
            error = float((image-production['render_linear']).abs().max())
            env = f.render(camera,pc,pipe,rho*environment/math.pi)['render']
            direct = f.render(camera,pc,pipe,rho*q)['render']
            additive = float((image-env-direct).abs().max())
            assert error < 2e-6 and additive < 2e-5
            parity.append(dict(frame=int(frames[index]), production_max_error=error, additivity_max_error=additive))
        value = float(f.objective(image, camera))
        assert math.isfinite(value)
        return value
    errors = np.array([score(i, light, int(frames[i]) in (1,81,149)) for i,light in enumerate(directions)])
    calls['complete_fit'] += 117
    m0 = float(errors.mean())
    assert abs(m0-old_fit) < 1e-7, (m0, old_fit)
    per_frame_restart_error = float(np.max(np.abs(errors-np.asarray(fit_record['per_frame']))))
    assert per_frame_restart_error < 1e-7, per_frame_restart_error
    a0, _, _ = penalties(original, original, frames)
    assert a0 > 1e-12
    hashes[str(Path(__file__).resolve())] = v.sha(__file__)
    config = dict(gt_read=False, held_scored=False, held_used_for_selection=False,
        train_containers_loaded=135, development_containers_loaded=18,
        loader_scope='公共loader解码135 TRAIN；只有117fit进入score/候选/停止，18开发RGB不参与选择',
        external_test_read=False, geometry_frozen=True,
        material_frozen=True, environment_frozen=True, source_checkpoint=str(a.checkpoint),
        source_checkpoint_sha256=checkpoint_sha, input_sha256=hashes,
        fit_frames=v.FIT, development_frames=v.HELD, steps_deg=[2,2,1,1,.5,.5], trust_radius_deg=5,
        tangent_basis='argmin(abs(unit_L))世界轴与L叉乘归一为u，v=L叉乘u；顺序+u,-u,+v,-v',
        traversal='六轮依次正/逆交替；严格J下降超过1e-12才接受，平局保留原值',
        hemisphere='dot(L,原TRAIN预测平面normal)>=0.05',
        objective='M/M0+0.05*A/A0+0.01*B；A按真实frame间隔加速度平方加权；B偏离旧L角平方/5度平方',
        max_candidate_forwards=2808, max_complete_fit_forwards=819,
        minimum_terminal_fit_improvement=.01, old_fit=m0, old_roughness=a0,
        shadow_bias=float(data['shadow_bias']), production_parity=parity,
        production_parity_semantics='已审计MaterialProxy linear-rho→sRGB路径；本脚本不声称真实pc._albedo_dc PLY重载',
        zero_update_max_per_frame_mse_difference=per_frame_restart_error,
        command=shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),
        internal_validation_passed=False, direction_identifiability='unresolved')
    v.write(a.output/'configuration.json', config)
    def objective(errors, value):
        roughness, drift, _ = penalties(value, original, frames)
        return float(errors.mean()/m0+.05*roughness/a0+.01*drift)
    history = []
    for round_index, step in enumerate(config['steps_deg'], 1):
        order = range(117) if round_index % 2 else range(116, -1, -1)
        accepted = 0
        for index in order:
            best, best_error = directions[index].copy(), errors[index]
            best_j = objective(errors, directions)
            proposals = candidates(directions[index], step)
            for candidate_index, proposed in enumerate(proposals):
                deviation = math.degrees(math.acos(np.clip(normalized(proposed)@normalized(original[index]), -1, 1)))
                if deviation > 5+1e-6 or normalized(proposed)@plane_normal < .05:
                    continue
                candidate_error = score(index, proposed)
                calls['candidate'] += 1
                trial_directions, trial_errors = directions.copy(), errors.copy()
                trial_directions[index], trial_errors[index] = proposed, candidate_error
                candidate_j = objective(trial_errors, trial_directions)
                if candidate_j < best_j-1e-12:
                    best, best_error, best_j = proposed, candidate_error, candidate_j
            if not np.array_equal(best, directions[index]):
                accepted += 1
            directions[index], errors[index] = best, best_error
        fresh = np.array([score(i, light) for i,light in enumerate(directions)])
        calls['complete_fit'] += 117
        assert np.max(np.abs(fresh-errors)) < 1e-7
        errors = fresh
        roughness, drift, angles = penalties(directions, original, frames)
        row = dict(round=round_index, step_deg=step, accepted=accepted, fit_mse=float(errors.mean()),
                   objective=objective(errors,directions), roughness=roughness, drift=drift,
                   direction_angle_deg=dict(mean=float(angles.mean()),max=float(angles.max()),p90=float(np.quantile(angles,.9))),
                   per_frame=[dict(frame=int(frame),mse=float(error),direction=light.tolist()) for frame,error,light in zip(frames,errors,directions)])
        history.append(row)
        v.write(a.output/f'round_{round_index}.json', row)
        print({k:value for k,value in row.items() if k!='per_frame'},flush=True)
    assert calls['candidate'] <= 2808 and calls['complete_fit'] <= 819
    assert all(torch.equal(value,getattr(pc,key)) for key,value in geometry.items())
    assert torch.equal(rho,frozen_rho) and torch.equal(environment,frozen_environment)
    fit_light = torch.as_tensor(directions,dtype=torch.float32)
    all_light = v.interpolate_frozen(torch.tensor(v.TRAIN,dtype=torch.float64), torch.tensor(v.FIT,dtype=torch.float64), fit_light.double()).float()
    all_light[fit_indices] = fit_light
    torch.save(dict(directions=all_light, fit_directions=fit_light, frames=v.TRAIN, fit_frames=v.FIT,
                    gt_read=False, held_rgb_used=False, source_checkpoint_sha256=checkpoint_sha), a.output/'direction_state.pt')
    v.check_hashes(hashes)
    improvement = 1-float(errors.mean())/m0
    v.write(a.output/'summary.json', dict(status='PASS_FIT_SCREEN_ONLY' if improvement>=.01 else 'FAILED_INSUFFICIENT_FIT_IMPROVEMENT',
        gt_read=False, held_scored=False, held_used_for_selection=False,
        train_containers_loaded=135, development_containers_loaded=18,
        internal_validation_passed=False, direction_identifiability='unresolved',
        geometry_bitexact_unchanged=True, material_environment_unchanged=True, old_fit=m0,
        final_fit=float(errors.mean()), relative_fit_improvement=improvement, calls=calls,
        eligible_for_equal_budget_calibration=improvement>=.01, training_asset_exported=False,
        configuration_sha256=v.sha(a.output/'configuration.json'), direction_state_sha256=v.sha(a.output/'direction_state.pt'), history=history))


if __name__ == '__main__':
    main()
