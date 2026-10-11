#!/usr/bin/env python3
"""真实Stage2保存后，同训练相机的新分流前向与原物理评测入口一致性。"""
import argparse
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def verify_sources(model, marker, digest):
    from goal_eval_render import training_render_sources
    sources = training_render_sources(model)
    if (model / 'censored_training.json').exists():
        if marker['actual_entrypoint'] != 'scripts/train_static_dl_censored.py':
            raise AssertionError('censored实际训练入口未明确记录。')
        if marker['loss_model'] != marker['args']['loss_model']:
            raise AssertionError('censored损失模式记录冲突。')
        expected = ('scripts/train_static_dl_censored.py', 'scene/static_dl_censored.py',
                    'scene/static_dl_censored_audit.py', 'scene/static_dl_shadow_split.py')
    else:
        expected = ('scripts/train_static_dl_shadow_split.py', 'scene/static_dl_shadow_split.py',
                    'scene/static_dl_shadow_split_audit.py')
    saved = marker['new_sources']
    for relative in expected:
        matching = [(Path(path), sha) for path, sha in saved.items()
                    if Path(path).as_posix().endswith('/' + relative)]
        if len(matching) != 1:
            raise AssertionError('新训练入口来源不唯一：' + relative)
        path, sha = matching[0]
        if digest(path) != sha or digest(ROOT / relative) != sha:
            raise AssertionError('分流前向／审计／训练源码与保存快照不同：' + relative)
        sources[str(path)] = sha
        sources[str(ROOT / relative)] = sha
    return sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--load_iter', type=int, required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--frame', type=int, default=1)
    p.add_argument('--output_dir', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from argparse import Namespace
    from scripts.eval_geometry_gt import GeometryRun
    from scene.static_dl_photometric import DirectionOnlyLights
    from scene.static_dl_shadow_split import render_split_photometric
    from goal_eval_render import physical_renderer
    from goal_eval_protocol import CONTROL, FRAMES, read, write, digest
    from utils.geometry_eval_utils import unit_normal
    import diff_surfel_rasterization as rasterizer

    backend = getattr(rasterizer, 'provenance', None)
    if not backend or backend.get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise AssertionError('新Stage2重载必须经bootstrap绑定原生CUDA光栅。')
    if args.frame in FRAMES or not 1 <= args.frame <= 150:
        raise AssertionError('一致性检查只使用原训练相机。')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise AssertionError('评测输出必须位于output。')
    model = Path(args.model_path)
    censored = (model / 'censored_training.json').is_file()
    marker_path = model / ('censored_training.json' if censored else 'shadow_split_training.json')
    marker = read(marker_path)
    route = 'control' if censored else marker['args']['route']
    if route not in ('control', 'split'):
        raise AssertionError('仅检查实际两组训练路线。')
    sources = verify_sources(Path(args.model_path), marker, digest)
    output.mkdir(parents=True, exist_ok=False)
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    run.files.update(sources)
    run.files[str(marker_path.resolve())] = digest(marker_path)
    metadata = read(run.source / 'transforms_train.json')
    record = next(r for r in metadata['frames'] if r['source_frame'] == args.frame)
    camera, _, intrinsics, _ = run.camera(metadata, record, 2)
    static = read(run.model / 'static_dl_training.json')
    config = static['args']
    trained = static['rasterizer_provenance']
    for key in ('method', 'source_sha256', 'extension_sha256', 'source_files'):
        if trained[key] != backend[key]:
            raise AssertionError('新训练／评测原生光栅来源不同：' + key)
    if digest(backend['extension_path']) != trained['extension_sha256']:
        raise AssertionError('实际加载.so哈希不同。')
    light_path = run.model / 'static_dl_lighting' / f'iteration_{args.load_iter}' / 'lighting.json'
    lighting = read(light_path)
    if lighting['shadow_gradient_route'] != route:
        raise AssertionError('保存光照的梯度路线与训练记录不同。')
    if args.load_iter != lighting['initial_iteration'] + lighting['stage2_step']:
        raise AssertionError('新Stage2全局迭代与保存步数不同。')
    if config['static_dl_mode'] != 'lambertian_visibility' or any(config[k] != 0 for k in
            ('static_dl_physical_start', 'static_dl_visibility_start')) or config['static_dl_bvh_interval'] != 1:
        raise AssertionError('新训练前向必须从首步启用fresh BVH。')
    lights = DirectionOnlyLights(config['static_dl_directions'])
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    directions, intensity = lights.for_camera(camera)
    wrapper, physical = physical_renderer(run, args.load_iter, CONTROL / 'eval_directions_only_single.json')
    if wrapper is None or not physical['visibility_enabled']:
        raise AssertionError('评测入口未启用真实物理可见度。')
    with torch.no_grad():
        reference = render_split_photometric(camera, run.pc, Namespace(**config),
            torch.zeros(3, device='cuda'), directions, intensity, config['static_dl_shadow_bias'],
            route=route, rebuild=True)
        # 新函数的光强为显式输入，没有light_intensity输出字段。
        reference['light_intensity'] = intensity.detach()
        actual = wrapper(camera)  # 独立新建StaticVisibility的首次调用，重新构建BVH。
        eval_directions, _ = wrapper.eval_lights.for_camera(camera)
        # 旧渲染package只返回mean；读取同一fresh评测对象的完整逐surfel可见度。
        actual['light_visibility'] = wrapper.eval_visibility(run.pc, eval_directions, args.load_iter)
        geometry = run.render(camera, np.asarray(record['transform_matrix']), intrinsics)
        deformation = run.deformation(camera)
    arrays, checks = {}, {}
    for key in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth',
                'light_intensity', 'mean_visibility', 'light_visibility'):
        a, b = reference[key].cpu().numpy(), actual[key].cpu().numpy()
        arrays['training_' + key], arrays['evaluation_' + key] = a, b
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('新Stage2出现非有限输出：' + key)
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        checks[key] = dict(max_abs=float(np.max(np.abs(a-b))), passed=True)
    normal, valid = unit_normal(actual['rend_normal'].permute(1, 2, 0).cpu().numpy())
    np.testing.assert_allclose(normal[valid], geometry['normal'][valid], atol=2e-5, rtol=1e-5)
    alpha = actual['rend_alpha'].squeeze(0).cpu().numpy()
    np.testing.assert_allclose(alpha, geometry['alpha'], atol=2e-5, rtol=1e-5)
    mask = geometry['alpha'] >= .5
    mixture = geometry['expected']*(1-config['depth_ratio'])+geometry['median']*config['depth_ratio']
    np.testing.assert_allclose(actual['surf_depth'].squeeze(0).cpu().numpy()[mask],
                               mixture[mask], atol=2e-5, rtol=1e-5)
    for key in ('d_xyz', 'd_rotation', 'd_scaling'):
        if torch.count_nonzero(deformation[key]):
            raise AssertionError('新Stage2时变几何输出不为零：' + key)
    state = torch.load(run.model / 'deform' / f'iteration_{args.load_iter}' / 'deform.pth',
                       map_location='cpu', weights_only=True)
    heads = {key: int(torch.count_nonzero(value)) for key, value in state.items()
             if any(head in key for head in ('gaussian_warp.', 'gaussian_rotation.', 'gaussian_scaling.'))}
    if len(heads) != 6 or any(heads.values()):
        raise AssertionError('保存的六个几何头张量不为零。')
    arrays.update(geometry)
    arrays['real_xyz'], arrays['real_quaternion'] = (run.pc.get_xyz.detach().cpu().numpy(),
                                                  run.pc.get_rotation.detach().cpu().numpy())
    np.savez_compressed(output / 'render_parity_arrays.npz', **arrays)
    mapped = sorted({line.split()[-1] for line in Path(f'/proc/{os.getpid()}/maps').read_text().splitlines()
                     if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line)
                     and line.split()[-1].endswith('.so')})
    if not any('surfel_tracer/' in path for path in mapped):
        raise AssertionError('无法记录实际运行的tracer扩展。')
    runtime_modules = {path: digest(path) for path in mapped}
    report = dict(status='PASS', scope='真实新Stage2 checkpoint，同训练相机，新分流前向vs原物理评测；两侧fresh BVH。',
        model=str(run.model), source=str(run.source), iteration=args.load_iter, route=route,
        actual_entrypoint=marker.get('actual_entrypoint', 'scripts/train_static_dl_shadow_split.py'),
        loss_model=marker.get('loss_model', '原two_sided鲁棒残差'),
        loss_scope='此处仅验证前向重载与来源，不代替新损失梯度审计。',
        frame=args.frame, checks=checks, saved_zero_heads=heads,
        geometry_maps='PASS：真实XYZ/q渲染的原GeometryRun world normal、alpha、mixed depth一致；三时变几何输出严格0。',
        training_rasterizer=trained, effective_rasterizer=backend, physical_provenance=physical,
        runtime_pid=os.getpid(), actual_runtime_modules=runtime_modules,
        sources=run.files, goal_mae='未评测；接入检查不代表Goal达标。')
    write(output / 'parity.json', report)
    command = shlex.join([sys.executable, str(ROOT / 'LIHAN/scripts/bootstrap_1011.py'), str(ROOT),
        'LIHAN.scripts.goal_eval_shadow_split_parity'] + sys.argv[1:])
    (output / 'README.md').write_text('# 新Stage2真实重载一致性检查\n\n'
        + f'模型：`{run.model}`；source：`{run.source}`；iteration={args.load_iter}；路线={route}。\n\n'
        + '完整命令（CUDA_VISIBLE_DEVICES由监督执行日志记录）：\n\n```bash\n' + command + '\n```\n\n'
        + f'输出：`{output}`；原始数组：`render_parity_arrays.npz`；定量：`parity.json`。\n\n'
        + 'RGB、linear RGB、Alpha、world normal、depth、intensity、mean visibility及完整逐surfel visibility八项通过1e-5绝对／相对容差；'
        + '原GeometryRun三类几何map与冻结头检查通过。源码／实际native.so与训练记录一致。\n\n'
        + '四类可视化未在本单相机接入检查执行；完整15帧MAE及150帧视频应另行验收。\n\n'
        + '接入结论：**PASS**；Goal三法线指标尚未评测，不称为Goal通过。\n')
    print('新Stage2真实重载一致性PASS：' + str(output), flush=True)


if __name__ == '__main__':
    main()
