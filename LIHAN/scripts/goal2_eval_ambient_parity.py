#!/usr/bin/env python3
"""G2真实ambient checkpoint fresh/fresh前向、T13零底项同义与几何重载。"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--source_path', required=True)
    parser.add_argument('--load_iter', type=int, required=True)
    parser.add_argument('--frame', type=int, default=1)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from argparse import Namespace
    from scripts.eval_geometry_gt import GeometryRun
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility, render_photometric as original_render
    from scene.static_dl_ambient import render_photometric
    from goal2_eval_render import ambient_sources, physical_renderer
    from goal_eval_protocol import FRAMES, CONTROL, read, write, digest
    from utils.geometry_eval_utils import unit_normal
    import diff_surfel_rasterization as rasterizer

    backend = getattr(rasterizer, 'provenance', None)
    if not backend or backend.get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise AssertionError('必须bootstrap绑定原生CUDA光栅。')
    if args.frame in FRAMES or args.frame not in range(1,151):
        raise AssertionError('fresh仅使用训练相机。')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise AssertionError('输出必须位于output。')
    output.mkdir(parents=True, exist_ok=False)
    marker, sources = ambient_sources(args.model_path)
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    run.files.update(sources)
    static = read(run.model / 'static_dl_training.json')
    config, trained = static['args'], static['rasterizer_provenance']
    for key in ('method', 'source_sha256', 'extension_sha256', 'source_files'):
        if trained[key] != backend[key]:
            raise AssertionError('实际train/eval native不同：' + key)
    if digest(backend['extension_path']) != trained['extension_sha256']:
        raise AssertionError('实际.so哈希不同。')
    metadata = read(run.source / 'transforms_train.json')
    record = next(r for r in metadata['frames'] if r['source_frame'] == args.frame)
    camera, _, intrinsics, _ = run.camera(metadata, record, 2)
    lighting_path = run.model / f'static_dl_lighting/iteration_{args.load_iter}/lighting.json'
    lighting = read(lighting_path)
    fraction = float(lighting['ambient_fraction'])
    run.files[str(lighting_path)] = digest(lighting_path)
    run.files[str(Path(__file__).resolve())] = digest(Path(__file__))
    pipe, bg = Namespace(**config), torch.zeros(3, device='cuda')
    lights = DirectionOnlyLights(config['static_dl_directions'])
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    vis = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    wrapper, provenance = physical_renderer(run, args.load_iter, CONTROL / 'eval_directions_only_single.json')
    with torch.no_grad():
        reference = render_photometric(camera, run.pc, pipe, bg, lights, vis, args.load_iter, fraction, component_images=True)
        actual = wrapper(camera)
        geometry = run.render(camera, np.asarray(record['transform_matrix']), intrinsics)
        deformation = run.deformation(camera)
    arrays, checks = {}, {}
    fields = ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth', 'light_intensity',
              'mean_visibility', 'light_visibility', 'ambient_fraction', 'ambient_linear', 'direct_linear')
    for key in fields:
        a, b = reference[key].detach().cpu().numpy(), actual[key].detach().cpu().numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('真实ambient前向非有限：' + key)
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        arrays['training_'+key], arrays['evaluation_'+key] = a, b
        checks[key] = dict(passed=True, max_abs=float(np.max(np.abs(a-b))))
    normal, valid = unit_normal(actual['rend_normal'].permute(1,2,0).cpu().numpy())
    np.testing.assert_allclose(normal[valid], geometry['normal'][valid], atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(actual['rend_alpha'].squeeze(0).cpu().numpy(), geometry['alpha'], atol=2e-5, rtol=1e-5)
    visible = geometry['alpha'] >= .5
    np.testing.assert_allclose(actual['surf_depth'].squeeze(0).cpu().numpy()[visible], geometry['median'][visible], atol=2e-5, rtol=1e-5)
    for key in ('d_xyz', 'd_rotation', 'd_scaling'):
        if torch.count_nonzero(deformation[key]):
            raise AssertionError('时变几何头非零。')
    state = torch.load(run.model / f'deform/iteration_{args.load_iter}/deform.pth', map_location='cpu', weights_only=True)
    heads = {k: int(torch.count_nonzero(v)) for k, v in state.items() if any(h in k for h in ('gaussian_warp.', 'gaussian_rotation.', 'gaussian_scaling.'))}
    if len(heads) != 6 or any(heads.values()):
        raise AssertionError('保存零头不符。')
    # 真正从T13保存PLY/lighting另建GeometryRun，不用续训后几何冒充初始同义。
    init_model = Path(marker['args']['init_model'])
    init_iteration = marker['args']['init_iteration']
    for path, sha in marker['initial_sources'].items():
        if digest(path) != sha:
            raise AssertionError('T13初始化资产变化。')
        run.files[path] = sha
    baseline = GeometryRun(str(init_model), init_iteration, args.source_path)
    baseline_lighting = read(init_model / f'static_dl_lighting/iteration_{init_iteration}/lighting.json')
    baseline_lights = DirectionOnlyLights(config['static_dl_directions'])
    with torch.no_grad():
        baseline_lights.log_intensity.copy_(torch.as_tensor(baseline_lighting['intensity_from_train_rgb'], device='cuda').log())
        zero = render_photometric(camera, baseline.pc, pipe, bg, baseline_lights,
            StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval']), init_iteration, 0., component_images=True)
        old = original_render(camera, baseline.pc, pipe, bg, baseline_lights,
            StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval']), init_iteration)
    baseline_checks = {}
    for key in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth', 'light_intensity', 'mean_visibility'):
        a, b = zero[key].cpu().numpy(), old[key].cpu().numpy()
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
        baseline_checks[key] = dict(passed=True, max_abs=float(np.max(np.abs(a-b))))
    nonzero_covered = fraction > 0 and float(reference['ambient_linear'].max()) > 0
    status = 'PASS' if marker['ambient_mode'] == 'zero' or nonzero_covered else 'FAILED_NONZERO_AMBIENT_UNCOVERED'
    arrays.update(geometry)
    arrays['real_xyz'] = run.pc.get_xyz.detach().cpu().numpy()
    arrays['real_quaternion'] = run.pc.get_rotation.detach().cpu().numpy()
    np.savez_compressed(output / 'render_parity_arrays.npz', **arrays)
    mapped = sorted({line.split()[-1] for line in Path(f'/proc/{os.getpid()}/maps').read_text().splitlines()
        if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line) and line.split()[-1].endswith('.so')})
    if not any('surfel_tracer/' in path for path in mapped):
        raise AssertionError('未实际运行tracer。')
    runtime = {path: digest(path) for path in mapped}
    for path, sha in run.files.items():
        if digest(path) != sha:
            raise AssertionError('fresh来源变化：' + path)
    report = dict(status=status, forward_status='PASS',
        nonzero_branch_coverage='COVERED' if nonzero_covered else ('NOT_REQUIRED_ZERO_CONTROL' if marker['ambient_mode'] == 'zero' else 'UNCOVERED'),
        status_scope='此status仅fresh接入门槛；非零未覆盖不是训练/数学/Goal指标失败。保存a0原样保留。',
        model=str(run.model), source=str(run.source), iteration=args.load_iter,
        frame=args.frame, actual_entrypoint=marker['actual_entrypoint'], ambient_mode=marker['ambient_mode'],
        ambient_fraction=fraction, ambient_cap=marker['ambient_cap'], nonzero_ambient_covered=nonzero_covered,
        checks=checks, baseline_T13_zero_ambient_checks=baseline_checks, baseline_model=str(init_model),
        point_count=len(run.pc.get_xyz), initial_point_count=marker['initial_point_count'], saved_zero_heads=heads,
        geometry_maps='PASS: real XYZ/q/alpha/median depth同原GeometryRun；不施加Stage2点集/scale/opacity冻结条件。',
        parameter_scope='Stage1几何与增密允许更新，唯一冻结三时变几何头；fresh不验证训练反向或损失日程。',
        physical_provenance=provenance, training_rasterizer=trained, effective_rasterizer=backend,
        actual_runtime_modules=runtime, runtime_pid=os.getpid(), sources=run.files,
        component_sum_max_abs=float(reference['linear_component_sum_max_abs']), goal_numeric='未评测')
    write(output / 'parity.json', report)
    (output / 'README.md').write_text('# G2真实ambient单相机重载检查\n\n'
        + f'模型：`{run.model}`；source：`{run.source}`；iteration={args.load_iter}；frame={args.frame}。\n\n'
        + '实际新renderer训练式调用与独立评测wrapper两侧fresh BVH，11项前向/分量核验；另从T13初始保存资产验证a=0与旧Lambertian同义。\n\n'
        + f'ambient_mode={marker["ambient_mode"]}，保存a={fraction}，非零覆盖={nonzero_covered}。\n\n'
        + f'接入结论：**{status}**；Goal指标及全150四类可视化尚未执行。完整日志由调用者保留，原始数组与实际native/tracer SHA见本目录。\n')
    print('G2真实ambient fresh：' + status + '；' + str(output), flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
