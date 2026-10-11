#!/usr/bin/env python3
"""只供评测的静态DL真实RGB适配：恢复训练光强，独立150帧纯方向表。"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from goal_eval_protocol import CONTROL, RAW, FRAMES, read, write, digest


def prepare_directions(output):
    import numpy as np
    output = Path(output)
    if output.exists():
        raise FileExistsError('评测方向表禁止覆盖。')
    raw = read(RAW / 'lights.json')['frames']
    frames = {}
    for n in range(1, 151):
        key = str(n).zfill(4)
        vectors = [np.asarray(light['light_dir_world'], dtype=np.float64) for light in raw[key]['lights']]
        if not vectors or not all(np.allclose(vectors[0], v, atol=1e-6) for v in vectors[1:]):
            raise AssertionError('仅适用于训练侧同向SUN合并规则；方向不相同不能合并。')
        # 严格复现goal_prepare_static_dl；学习到的单灯光强已吸收原总能量。
        frames[key] = {'light_dir_world': vectors[0].tolist()}
    result = {'frames': frames}
    write(output, result)
    return output


def training_render_sources(model):
    """兼容旧冒烟与正式启动元数据；缺失来源明确拒绝。无需CUDA。"""
    model = Path(model)
    candidates = [model.parent / 'commands.json', model.parent / 'commands/train.json']
    available = [(path, read(path)) for path in candidates if path.is_file()]
    if not available:
        raise AssertionError('静态DL训练marker存在，但缺少记录snapshot的启动元数据。')
    snapshots = {str(Path(record['snapshot']).resolve()) for _, record in available if record.get('snapshot')}
    if len(snapshots) != 1 or any(not record.get('snapshot') for _, record in available):
        raise AssertionError('静态DL训练启动元数据缺少snapshot或不同记录相互冲突。')
    snapshot = Path(next(iter(snapshots)))
    sources = {str(path): digest(path) for path, _ in available}
    for relative in ('scene/static_dl_photometric.py', 'gaussian_renderer/__init__.py',
                     'scene/gaussian_model.py', 'scene/cameras.py'):
        saved_path = snapshot / relative
        current_path = ROOT / relative
        if digest(saved_path) != digest(current_path):
            raise AssertionError('评测物理代码与训练快照不同，需显式适配：' + relative)
        sources[str(saved_path)] = digest(saved_path)
        sources[str(current_path)] = digest(current_path)
    return sources


def physical_renderer(run, iteration, eval_directions):
    """返回真实物理renderer或None（SH/尚未进入物理阶段）。"""
    import numpy as np
    import torch
    from argparse import Namespace
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility, render_photometric
    marker = run.model / 'static_dl_training.json'
    if not marker.exists():
        return None, None
    run.files.update(training_render_sources(run.model))
    config = read(marker)['args']
    mode = config.get('static_dl_mode', 'sh')
    if mode == 'sh' or iteration <= config['static_dl_physical_start']:
        return None, None
    lighting_path = run.model / 'static_dl_lighting' / f'iteration_{iteration}' / 'lighting.json'
    lighting = read(lighting_path)
    if lighting['iteration'] != iteration or lighting['mode'] != mode:
        raise AssertionError('lighting与检查点迭代／物理模式不同。')
    train_path = Path(config['static_dl_directions'])
    if digest(train_path) != lighting['directions_sha256']:
        raise AssertionError('训练纯方向表与lighting.json哈希不同。')
    train = DirectionOnlyLights(train_path)
    evaluation = DirectionOnlyLights(eval_directions)
    expected_train = {str(n).zfill(4) for n in range(1, 151) if n not in FRAMES}
    if set(train.directions) != expected_train or set(evaluation.directions) != {str(n).zfill(4) for n in range(1, 151)}:
        raise AssertionError('训练仅135方向；评测独立全150方向，不能混用。')
    for key, value in train.directions.items():
        if not torch.equal(value, evaluation.directions[key]):
            raise AssertionError('评测与训练相同帧方向不同：' + key)
    intensity = torch.as_tensor(lighting['intensity_from_train_rgb'], dtype=torch.float32, device='cuda')
    if intensity.shape != evaluation.log_intensity.shape or not torch.isfinite(intensity).all() or (intensity <= 0).any():
        raise AssertionError('训练RGB光强无效。')
    with torch.no_grad():
        evaluation.log_intensity.copy_(intensity.log())
    evaluation.requires_grad_(False)
    visibility = None
    if mode == 'lambertian_visibility' and iteration >= config['static_dl_visibility_start']:
        visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    sources = {str(p): digest(p) for p in (marker, lighting_path, train_path, Path(eval_directions))}
    sources[str(ROOT / 'scene/static_dl_photometric.py')] = digest(ROOT / 'scene/static_dl_photometric.py')
    run.files.update(sources)
    pipe = Namespace(**config)
    background = torch.zeros(3, device='cuda')

    def render(camera):
        return render_photometric(camera, run.pc, pipe, background, evaluation, visibility, iteration)

    # 只供一致性审计读取同一评测对象；不改变任何渲染输出或梯度。
    render.eval_visibility = visibility
    render.eval_lights = evaluation

    return render, dict(mode=mode, visibility_enabled=visibility is not None,
                        intensity_from_train_rgb=intensity.tolist(), directions_only=True,
                        test_directions_used_only_for_evaluation=True, sources=sources)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-directions', type=Path, default=CONTROL / 'eval_directions_only_single.json')
    args = p.parse_args()
    print('仅评测150帧纯方向：' + str(prepare_directions(args.prepare_directions)))


if __name__ == '__main__':
    main()
