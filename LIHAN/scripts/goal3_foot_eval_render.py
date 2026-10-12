#!/usr/bin/env python3
"""足部恢复独立来源适配；实际RGB仍冻结ambient renderer，a固定0。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import FRAMES, read, digest

ENTRY = 'scripts/train_static_dl_foot_recovery.py'
NEW_SOURCES = (ENTRY, 'scene/static_dl_foot_recovery.py', 'scene/static_dl_normal_opacity.py', 'scene/static_dl_ambient.py',
    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py', 'scene/dataset_readers.py', 'utils/camera_utils.py')


def actual_sources(model):
    model = Path(model)
    marker_path = model / 'foot_recovery_training.json'
    marker = read(marker_path)
    route = marker['normal_opacity_route']
    if marker['actual_entrypoint'] != ENTRY or route != 'detach_opacity':
        raise AssertionError('足部恢复实际入口/route错误。')
    if marker['args']['normal_opacity_route'] != route or marker['inherited_config']['normal_opacity_route'] != route:
        raise AssertionError('顶层/CLI/inherited_config route冲突。')
    if marker['normal_opacity_detached'] != (route == 'detach_opacity'):
        raise AssertionError('route与detached声明不一致。')
    if marker['ambient_mode'] != 'zero' or marker['ambient_initial'] != 0 or marker['ambient_trainable_scalars'] != 0:
        raise AssertionError('本轮仅a0、不训练ambient。')
    if marker['photo_depth_ratio'] != 1 or marker['normal_depth_statistic'] != 'expected' or marker['loss_model'] != 'censored':
        raise AssertionError('固定photo1/expected/censored。')
    if marker['args']['init_iteration'] != 14000 or len(marker['initial_sources']) != 5 or len(marker['new_sources']) != 11:
        raise AssertionError('固定G2-T04@14000、5起点资产与11实际源码。')
    policy = marker['reset_policy']
    if policy not in ('standard', 'skip_15000') or any(mapping['reset_policy'] != policy for mapping in (marker['args'], marker['inherited_config'])):
        raise AssertionError('真实reset策略的顶层/CLI/继承配置不一致。')
    shifted = marker['clock_shift_diagnostic']
    if shifted != (marker['clock_start'] != 14000):
        raise AssertionError('clock诊断标识与实际起始计数冲突。')
    if shifted and (not marker['smoke'] or marker['clock_start'] != 14990 or marker['update_budget'] != 120):
        raise AssertionError('前移clock仅允许14990起点的120步smoke。')
    if marker['train_images'] != 135 or marker['test_images_loaded'] or marker['forbidden_gt_supervision']:
        raise AssertionError('必须仅135训练RGB/Alpha，无额外GT监督。')
    for group in ('initial_sources', 'training_inputs'):
        for path, expected in marker[group].items():
            if digest(path) != expected:
                raise AssertionError('实际训练输入或起点来源变化：' + path)
    sources, snapshots = {}, set()
    available = [path for path in (model.parent / 'commands/train.json', model.parent / 'commands.json') if path.is_file()]
    if not available:
        raise AssertionError('存在marker却没有snapshot启动证据。')
    for path in available:
        record = read(path)
        argv = record.get('argv') or record.get('command')
        if not isinstance(argv, list) or len(argv) < 4 or Path(argv[1]).resolve() != ROOT / 'LIHAN/scripts/bootstrap_1011.py' or argv[3] != 'scripts.train_static_dl_foot_recovery':
            raise AssertionError('必须真实bootstrap+foot_recovery模块。')
        snapshot = Path(argv[2]).resolve()
        if record.get('snapshot') and Path(record['snapshot']).resolve() != snapshot:
            raise AssertionError('snapshot元数据冲突。')
        manifest_path = snapshot / 'snapshot_manifest.json'
        if not manifest_path.is_file():
            raise AssertionError('真实bootstrap target没有冻结snapshot manifest；主仓运行不能冒充snapshot。')
        manifest = read(manifest_path)
        if manifest['status'] != 'READY' or Path(manifest['destination']).resolve() != snapshot:
            raise AssertionError('snapshot尚未READY或目录不符。')
        snapshots.add(snapshot)
        sources.update({str(path.resolve()): digest(path), str(manifest_path): digest(manifest_path), str(Path(argv[1]).resolve()): digest(argv[1])})
        for relative in NEW_SOURCES + ('scene/cameras.py',):
            saved, actual = snapshot / relative, ROOT / relative
            expected = manifest['file_sha256'][relative]
            if digest(saved) != expected or digest(actual) != expected:
                raise AssertionError('实际加载文件与snapshot不同：' + relative)
            sources.update({str(saved): expected, str(actual): expected})
    if len(snapshots) != 1:
        raise AssertionError('启动证据snapshot不唯一。')
    for relative in NEW_SOURCES:
        matches = [(Path(path), sha) for path, sha in marker['new_sources'].items() if Path(path).as_posix().endswith('/'+relative)]
        if len(matches) != 1 or not matches[0][0].resolve().is_relative_to(next(iter(snapshots))):
            raise AssertionError('实际训练新source必须唯一且来自实际snapshot：' + relative)
        saved, sha = matches[0]
        if digest(saved) != sha or digest(ROOT / relative) != sha:
            raise AssertionError('实际new source SHA不同：' + relative)
    sources[str(marker_path.resolve())] = digest(marker_path)
    sources.update(marker['initial_sources'])
    sources.update(marker['training_inputs'])
    return marker, sources


def physical_renderer(run, iteration, eval_directions):
    import torch
    from argparse import Namespace
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    marker, sources = actual_sources(run.model)
    route = marker['normal_opacity_route']
    static_path = run.model / 'static_dl_training.json'
    static = read(static_path)
    config = static['args']
    if static['actual_entrypoint'] != ENTRY or static['normal_opacity_route'] != route or config['normal_opacity_route'] != route:
        raise AssertionError('static marker与真实normal route冲突。')
    if config['depth_ratio'] != 1 or config['static_dl_mode'] != 'lambertian_visibility' or config['white_background']:
        raise AssertionError('固定黑背景、真实物理、photo median1。')
    if iteration <= config['static_dl_physical_start'] or iteration < config['static_dl_visibility_start']:
        raise AssertionError('真实物理/V阶段尚未启用。')
    path = run.model / f'static_dl_lighting/iteration_{iteration}/lighting.json'
    lighting = read(path)
    for key in ('actual_entrypoint', 'ambient_mode', 'ambient_cap', 'loss_model', 'normal_depth_statistic', 'normal_opacity_route', 'reset_policy', 'clock_start', 'clock_shift_diagnostic', 'smoke'):
        if lighting[key] != marker[key]:
            raise AssertionError('lighting与训练marker不一致：' + key)
    if lighting['ambient_fraction'] != 0 or lighting['iteration'] != iteration or lighting['clock_start'] + lighting['continuation_step'] != iteration or lighting['optimizer_updates'] != lighting['continuation_step']:
        raise AssertionError('必须实际a0/相同保存迭代与更新计数。')
    if lighting['initial_iteration'] != 14000 or not 0 < lighting['optimizer_updates'] <= marker['update_budget']:
        raise AssertionError('保存检查点必须来自14000参数起点与真实更新预算。')
    if static.get('reset_policy') != marker['reset_policy'] or config['reset_policy'] != marker['reset_policy']:
        raise AssertionError('static marker和实际reset策略不同。')
    train_path = Path(config['static_dl_directions'])
    if digest(train_path) != lighting['directions_sha256']:
        raise AssertionError('仅训练方向SHA变化。')
    train, evaluation = DirectionOnlyLights(train_path), DirectionOnlyLights(eval_directions)
    keys = {f'{n:04d}' for n in range(1,151) if n not in FRAMES}
    if set(train.directions) != keys or set(evaluation.directions) != {f'{n:04d}' for n in range(1,151)}:
        raise AssertionError('训练135/评测150纯方向严格隔离。')
    if any(not torch.equal(train.directions[key], evaluation.directions[key]) for key in keys):
        raise AssertionError('135训练方向和评测表数值不同。')
    intensity = torch.as_tensor(lighting['intensity_from_train_rgb'], dtype=torch.float32, device='cuda')
    if intensity.shape != evaluation.log_intensity.shape or not torch.isfinite(intensity).all() or (intensity <= 0).any():
        raise AssertionError('保存RGB学习光强无效。')
    with torch.no_grad():
        evaluation.log_intensity.copy_(intensity.log())
    evaluation.requires_grad_(False)
    pipe, background = Namespace(**config), torch.zeros(3, device='cuda')
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])

    def render(camera):
        return render_photometric(camera, run.pc, pipe, background, evaluation, visibility, iteration,
                                  ambient_fraction=0., component_images=True)

    render.eval_lights, render.eval_visibility = evaluation, visibility
    sources.update({str(p.resolve()): digest(p) for p in (static_path, path, train_path, Path(eval_directions), Path(__file__))})
    run.files.update(sources)
    return render, dict(actual_entrypoint=ENTRY, normal_opacity_route=route, reset_policy=marker['reset_policy'], clock_shift_diagnostic=marker['clock_shift_diagnostic'], mode='lambertian_visibility_with_ambient',
                        ambient_fraction=0., ambient_mode='zero', photo_depth_ratio=1,
                        visibility_enabled=True, intensity_from_train_rgb=intensity.tolist(), directions_only=True,
                        test_directions_used_only_for_evaluation=True, sources=sources,
                        scope='实际photo前向不因normal-loss反向route变化；反向证据独立训练audit，不用fresh前向冒充。')
