#!/usr/bin/env python3
"""G2显式重载真实ambient前向；旧Lambertian适配器保持不变。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import FRAMES, read, digest

NEW_SOURCES = ('scripts/train_static_dl_ambient.py', 'scene/static_dl_ambient.py',
               'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
               'scene/gaussian_model.py', 'gaussian_renderer/__init__.py', 'scene/dataset_readers.py', 'utils/camera_utils.py')


def training_snapshot_sources(model):
    """只读兼容标准launcher argv，禁止改历史commands或其启动SHA。"""
    candidates = [Path(model).parent / 'commands/train.json', Path(model).parent / 'commands.json']
    available = [(path, read(path)) for path in candidates if path.is_file()]
    if not available:
        raise AssertionError('ambient marker存在却缺少启动元数据。')
    snapshots, sources = set(), {}
    for path, record in available:
        argv = record.get('argv') or record.get('command')
        if not isinstance(argv, list) or len(argv) < 4:
            raise AssertionError('ambient实际bootstrap argv缺失。')
        bootstrap = Path(argv[1]).resolve()
        if bootstrap != ROOT / 'LIHAN/scripts/bootstrap_1011.py' or argv[3] != 'scripts.train_static_dl_ambient':
            raise AssertionError('ambient必须标准bootstrap+实际模块。')
        snapshot = Path(argv[2]).resolve()
        if record.get('snapshot') and Path(record['snapshot']).resolve() != snapshot:
            raise AssertionError('snapshot字段和argv[2]冲突。')
        manifest_path = snapshot / 'snapshot_manifest.json'
        manifest = read(manifest_path)
        if manifest['status'] != 'READY' or Path(manifest['destination']).resolve() != snapshot:
            raise AssertionError('snapshot manifest无效或来源目录冲突。')
        snapshots.add(snapshot)
        sources[str(path.resolve())] = digest(path)
        sources[str(bootstrap)] = digest(bootstrap)
        sources[str(manifest_path)] = digest(manifest_path)
        for relative in NEW_SOURCES + ('scene/cameras.py',):
            saved, current = snapshot / relative, ROOT / relative
            if digest(saved) != manifest['file_sha256'][relative] or digest(current) != digest(saved):
                raise AssertionError('当前实际加载源码与95文件snapshot记录不同：' + relative)
            sources[str(saved)] = digest(saved)
            sources[str(current)] = digest(current)
    if len(snapshots) != 1:
        raise AssertionError('多个启动元数据snapshot不一致。')
    return sources


def ambient_sources(model):
    marker_path = Path(model) / 'ambient_training.json'
    marker = read(marker_path)
    if marker['actual_entrypoint'] != 'scripts/train_static_dl_ambient.py':
        raise AssertionError('G2 renderer仅允许真实ambient入口。')
    if marker['ambient_mode'] not in ('zero', 'learned') or marker['ambient_cap'] != .1 or marker['photo_depth_ratio'] != 1:
        raise AssertionError('G2底项模式/cap/photo ratio不同。')
    if marker['normal_depth_statistic'] != 'expected' or marker['loss_model'] != 'censored' or not marker['visibility_detached']:
        raise AssertionError('G2固定expected/censored/detached visibility。')
    sources = training_snapshot_sources(model)
    # snapshot记录的9项均核对主仓实际加载文件，禁止只核对旧photo核心。
    if len(marker['new_sources']) != len(NEW_SOURCES):
        raise AssertionError('实际source名单数量不同，需显式审查适配。')
    for relative in NEW_SOURCES:
        matching = [(Path(path), sha) for path, sha in marker['new_sources'].items() if Path(path).as_posix().endswith('/'+relative)]
        if len(matching) != 1:
            raise AssertionError('实际ambient来源不唯一：' + relative)
        saved, expected = matching[0]
        if digest(saved) != expected or digest(ROOT / relative) != expected:
            raise AssertionError('实际ambient源与保存快照不同：' + relative)
        sources.update({str(saved): expected, str(ROOT / relative): expected})
    sources[str(marker_path.resolve())] = digest(marker_path)
    return marker, sources


def physical_renderer(run, iteration, eval_directions):
    import torch
    from argparse import Namespace
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    marker, sources = ambient_sources(run.model)
    static_path = run.model / 'static_dl_training.json'
    static = read(static_path)
    config = static['args']
    if static['actual_entrypoint'] != marker['actual_entrypoint'] or config['depth_ratio'] != 1 or config['static_dl_mode'] != 'lambertian_visibility':
        raise AssertionError('真实static元数据与ambient前向不同。')
    if config['white_background'] or iteration <= config['static_dl_physical_start'] or iteration < config['static_dl_visibility_start']:
        raise AssertionError('G2真实模型必须黑背景且在物理/V阶段。')
    light_path = run.model / f'static_dl_lighting/iteration_{iteration}/lighting.json'
    lighting = read(light_path)
    for key in ('actual_entrypoint', 'ambient_mode', 'ambient_cap', 'loss_model', 'normal_depth_statistic'):
        if lighting[key] != marker[key]:
            raise AssertionError('lighting与实际ambient marker冲突：' + key)
    if lighting['iteration'] != iteration or lighting['initial_iteration'] + lighting['continuation_step'] != iteration:
        raise AssertionError('保存lighting迭代不符。')
    fraction = float(lighting['ambient_fraction'])
    if not 0 <= fraction <= marker['ambient_cap'] or marker['ambient_mode'] == 'zero' and fraction != 0:
        raise AssertionError('实际底项超限或zero不为0。')
    train_path = Path(config['static_dl_directions'])
    if digest(train_path) != lighting['directions_sha256']:
        raise AssertionError('仅训练135方向来源变化。')
    train, evaluation = DirectionOnlyLights(train_path), DirectionOnlyLights(eval_directions)
    train_keys = {f'{n:04d}' for n in range(1, 151) if n not in FRAMES}
    if set(train.directions) != train_keys or set(evaluation.directions) != {f'{n:04d}' for n in range(1, 151)}:
        raise AssertionError('训练135／独立评测150方向混用。')
    for key in train_keys:
        if not torch.equal(train.directions[key], evaluation.directions[key]):
            raise AssertionError('训练／评测方向不同：' + key)
    intensity = torch.as_tensor(lighting['intensity_from_train_rgb'], dtype=torch.float32, device='cuda')
    if intensity.shape != evaluation.log_intensity.shape or not torch.isfinite(intensity).all() or (intensity <= 0).any():
        raise AssertionError('保存RGB学习光强无效。')
    with torch.no_grad():
        evaluation.log_intensity.copy_(intensity.log())
    evaluation.requires_grad_(False)
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    pipe, background = Namespace(**config), torch.zeros(3, device='cuda')

    def render(camera):
        return render_photometric(camera, run.pc, pipe, background, evaluation, visibility, iteration,
                                  ambient_fraction=fraction, component_images=True)

    render.eval_lights, render.eval_visibility = evaluation, visibility
    sources.update({str(path.resolve()): digest(path) for path in (static_path, light_path, train_path, Path(eval_directions), Path(__file__))})
    run.files.update(sources)
    return render, dict(mode='lambertian_visibility_with_ambient', actual_entrypoint=marker['actual_entrypoint'],
                        ambient_fraction=fraction, ambient_mode=marker['ambient_mode'], ambient_cap=marker['ambient_cap'],
                        visibility_enabled=True, visibility_fresh_first_call=True, photo_depth_ratio=1,
                        intensity_from_train_rgb=intensity.tolist(), directions_only=True,
                        test_directions_used_only_for_evaluation=True, sources=sources)
