#!/usr/bin/env python3
"""reset-audit独立来源适配；实际RGB仍冻结ambient renderer，a固定0。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import FRAMES, read, digest

ENTRY = 'scripts/train_static_dl_reset_audit.py'
NEW_SOURCES = (ENTRY, 'scene/static_dl_reset_audit.py', 'scripts/train_static_dl_normal_opacity.py', 'scene/static_dl_normal_opacity.py', 'scene/static_dl_ambient.py',
    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py', 'scene/dataset_readers.py', 'utils/camera_utils.py')


def actual_sources(model):
    model = Path(model)
    marker_path = model / 'reset_audit_training.json'
    marker = read(marker_path)
    route = marker['post_reset_normal_route']
    reset = 14010 if marker['smoke'] else 15000
    budget = 110 if marker['smoke'] else 2000
    saves = {10, 11, 100, 110} if marker['smoke'] else {900, 1000, 1100, 1200, 2000}
    if (marker['pre_reset_normal_route'] != 'full'
        or marker['inherited_config']['pre_reset_normal_route'] != 'full'
        or marker['normal_opacity_detached_before_reset'] is not False
        or marker['scheduled_reset_iteration'] != reset
        or marker['artificial_smoke_reset'] != marker['smoke']
        or marker['route_activation'] != 'first_reset_iteration+1'
        or marker['source_boundary_reset_replayed'] is not False
        or marker['update_budget'] != budget or marker['args']['steps'] != budget
        or marker['args']['smoke'] != marker['smoke']
        or set(marker['args']['save_steps']) != saves
        or marker['inherited_config']['iterations'] != 14000 + budget
        or marker.get('event_only', False)):
        raise AssertionError('必须本轮主模型、full预重置及明确次步切换，不冒充事件快照。')
    if marker['actual_entrypoint'] != ENTRY or route not in ('full', 'detach_opacity'):
        raise AssertionError('normal opacity实际入口/route错误。')
    if marker['args']['post_reset_normal_route'] != route or marker['inherited_config']['post_reset_normal_route'] != route:
        raise AssertionError('顶层/CLI/inherited_config route冲突。')
    if marker['normal_opacity_detached_after_reset'] != (route == 'detach_opacity'):
        raise AssertionError('route与detached声明不一致。')
    if marker['ambient_mode'] != 'zero' or marker['ambient_initial'] != 0 or marker['ambient_trainable_scalars'] != 0:
        raise AssertionError('本轮仅a0、不训练ambient。')
    if marker['photo_depth_ratio'] != 1 or marker['normal_depth_statistic'] != 'expected' or marker['loss_model'] != 'censored':
        raise AssertionError('固定photo1/expected/censored。')
    if marker['args']['init_iteration'] != 14000 or len(marker['initial_sources']) != 5 or len(marker['new_sources']) != 12:
        raise AssertionError('固定G2-T03 full@14000、5起点资产与12实际源码。')
    sources, snapshots = {}, set()
    available = [path for path in (model.parent / 'commands/train.json', model.parent / 'commands.json') if path.is_file()]
    if not available:
        raise AssertionError('存在marker却没有snapshot启动证据。')
    for path in available:
        record = read(path)
        argv = record.get('argv') or record.get('command')
        if not isinstance(argv, list) or len(argv) < 4 or Path(argv[1]).resolve() != ROOT / 'LIHAN/scripts/bootstrap_1011.py' or argv[3] != 'scripts.train_static_dl_reset_audit':
            raise AssertionError('必须真实bootstrap+reset_audit模块。')
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
    return marker, sources


def physical_renderer(run, iteration, eval_directions):
    import torch
    from argparse import Namespace
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    marker, sources = actual_sources(run.model)
    route = marker['post_reset_normal_route']
    static_path = run.model / 'static_dl_training.json'
    static = read(static_path)
    config = static['args']
    if static['actual_entrypoint'] != ENTRY or static['post_reset_normal_route'] != route or config['post_reset_normal_route'] != route:
        raise AssertionError('static marker与真实normal route冲突。')
    if static['pre_reset_normal_route'] != 'full' or config['pre_reset_normal_route'] != 'full' or 'normal_opacity_route' in config or 'normal_opacity_route' in static:
        raise AssertionError('static配置须区分pre/post，不能继承一个旧route冒充全程。')
    if config['depth_ratio'] != 1 or config['static_dl_mode'] != 'lambertian_visibility' or config['white_background']:
        raise AssertionError('固定黑背景、真实物理、photo median1。')
    if iteration <= config['static_dl_physical_start'] or iteration < config['static_dl_visibility_start']:
        raise AssertionError('真实物理/V阶段尚未启用。')
    path = run.model / f'static_dl_lighting/iteration_{iteration}/lighting.json'
    lighting = read(path)
    reset = marker['scheduled_reset_iteration']
    last_route = route if iteration > reset else 'full'
    phase = 'post_reset' if iteration == reset else 'after_update'
    first_reset = reset if iteration >= reset else None
    next_route = route if iteration >= reset else 'full'
    if (lighting['normal_opacity_route'] != last_route
        or lighting['next_update_normal_opacity_route'] != next_route
        or lighting['first_reset_iteration'] != first_reset
        or lighting['checkpoint_phase'] != phase
        or lighting['smoke'] != marker['smoke']):
        raise AssertionError('保存步的实际更新路线／reset时点／checkpoint阶段不符。')
    for key in ('actual_entrypoint', 'ambient_mode', 'ambient_cap', 'loss_model', 'normal_depth_statistic', 'post_reset_normal_route'):
        if lighting[key] != marker[key]:
            raise AssertionError('lighting与训练marker不一致：' + key)
    if lighting['ambient_fraction'] != 0 or lighting['iteration'] != iteration or lighting['initial_iteration'] + lighting['continuation_step'] != iteration or lighting['optimizer_updates'] != lighting['continuation_step']:
        raise AssertionError('必须实际a0/相同保存迭代与更新计数。')
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
    return render, dict(actual_entrypoint=ENTRY, post_reset_normal_route=route, normal_opacity_route=lighting['normal_opacity_route'], next_update_normal_opacity_route=lighting['next_update_normal_opacity_route'], checkpoint_phase=lighting['checkpoint_phase'], first_reset_iteration=lighting['first_reset_iteration'], mode='lambertian_visibility_with_ambient',
                        ambient_fraction=0., ambient_mode='zero', photo_depth_ratio=1,
                        visibility_enabled=True, intensity_from_train_rgb=intensity.tolist(), directions_only=True,
                        test_directions_used_only_for_evaluation=True, sources=sources,
                        scope='实际photo前向不因normal-loss反向route变化；反向证据独立训练audit，不用fresh前向冒充。')
