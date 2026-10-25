"""1024静态ELDL评测包装器：为 baseline=static_eldl_mixed（A0/G0/G1）登记adapter，不改动冻结评测入口。

- 不修改 LIHAN/scripts/eldl_1024_eval.py（冻结SHA 5821cadb…）中的任何数学定义、掩膜、聚合或门槛；
  只替换其模块内 model_record/adapter/make_plan 三个查找点，其余动作原样调用。
- 几何：沿用 static_sh 语义（零几何头的原生形变step，d_xyz=d_rotation=d_scaling=0，d_color=None），
  因此GS/Depth/Mesh法线与训练几何一致。
- RGB：scene/static_eldl_lighting.render_mixed，加载checkpoint的 lighting/iteration_k/lighting.json；
  每帧主光方向取原始lights.json对应帧 light_dir_world（评测专用：DEV/TEST的GT方向只用于评测渲染，不回传任何参数）。
- separation等带override颜色的渲染退回原生render。
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import LIHAN.scripts.eldl_1024_eval as E  # noqa: E402

BASELINE = 'static_eldl_mixed'
_orig_record, _orig_adapter, _orig_make_plan = E.model_record, E.adapter, E.make_plan


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _marker(model):
    if E.EXTERNAL_RECORD:
        return None
    path = Path(model).resolve() / 'eldl_v3_training.json'
    return json.loads(path.read_text()) if path.is_file() else None


def model_record(model, iteration, split, smoke=False):
    record = _marker(model)
    if record is None or record.get('baseline') != BASELINE:
        return _orig_record(model, iteration, split, smoke)
    model = Path(model).resolve()
    if record.get('gt_supervision') != 'TRAIN_GT_direction_only':
        raise ValueError('static_eldl_mixed只允许TRAIN GT方向这一项例外。')
    if record['train_frames'] != E.FIT:
        raise ValueError('static_eldl_mixed须为FIT117训练。')
    if not smoke and ('smoke_test' in model.parts or record['args'].get('smoke')):
        raise ValueError('冒烟模型只允许action=smoke。')
    plan = json.loads((model / 'physical_plan.json').read_text())
    directions = json.loads(Path(record['args']['directions']).read_text())
    if (directions['split'] != 'TRAIN117' or [r['frame'] for r in directions['records']] != E.FIT
            or plan['inputs']['directions'] != _sha(record['args']['directions'])):
        raise ValueError('训练方向表不是登记的TRAIN117净化表。')
    files = [model / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply',
             model / 'deform' / f'iteration_{iteration}' / 'deform.pth', model / 'cfg_args',
             model / f'lighting/iteration_{iteration}/lighting.json', model / 'physical_plan.json',
             model / 'physical_finished.json', Path(record['args']['directions'])]
    return record, {str(p): _sha(p) for p in files}


def adapter(model, iteration, source, split, smoke=False):
    record = _marker(model)
    if record is None or record.get('baseline') != BASELINE:
        return _orig_adapter(model, iteration, source, split, smoke)
    import numpy as np
    import torch
    from argparse import Namespace
    from gaussian_renderer import render
    from scene.static_dl_photometric import StaticVisibility
    from scene.static_eldl_lighting import MixedLighting, render_mixed
    from scene.stage2_photometric import frame_key
    real_record, files = model_record(model, iteration, split, smoke)
    code_root = Path(real_record['source_code_root'])
    composite = real_record['args'].get('composite', 'linear')
    checked = ['scene/static_eldl_lighting.py', 'scene/static_dl_photometric.py', 'scene/stage2_photometric.py']
    if composite != 'linear':
        checked.append('scene/static_eldl_composite.py')
    for relative in checked:
        if _sha(code_root / relative) != _sha(ROOT / relative):
            raise ValueError('训练快照与运行物理前向不一致：' + relative)
    if composite == 'linear':
        render_rgb = render_mixed
    elif composite == 'srgb_per_gaussian':
        from scene.static_eldl_composite import render_mixed_srgb as render_rgb
    else:
        raise ValueError('未登记的合成域：' + str(composite))
    # 几何部分复用冻结入口的static_sh adapter（全部校验原样执行）
    geometry_record = dict(real_record, baseline='static_sh', gt_supervision=False, init_weights_loaded=False)
    E.model_record = lambda *a, **k: (geometry_record, files)
    try:
        run = _orig_adapter(model, iteration, source, split, smoke)
    finally:
        E.model_record = model_record
    run.record = real_record
    run.training_marker_snapshot = real_record
    lighting_json = json.loads((Path(model) / f'lighting/iteration_{iteration}/lighting.json').read_text())
    lighting = MixedLighting()
    lighting.load_state(lighting_json['lighting'])
    lights = json.loads((E.RAW / 'lights.json').read_text())
    run.files[str(E.RAW / 'lights.json')] = _sha(E.RAW / 'lights.json')
    table = {}
    for n in range(1, 151):
        rec = lights[f'{n:04d}']
        d = np.asarray(rec['light_dir_world'], dtype=np.float64)
        sun = [x for x in rec['lights'] if x['name'] == 'Orbit_Directional_Sun'][0]
        axis = np.asarray(sun['matrix_world'], dtype=np.float64)[:3, 2]
        d /= np.linalg.norm(d); axis /= np.linalg.norm(axis)
        if not np.allclose(d, axis, atol=1e-6):
            raise ValueError('评测GT方向与SUN轴不一致')
        table[f'{n:04d}'] = torch.tensor(d, dtype=torch.float32, device='cuda')
    train_dirs = {f"{r['frame']:04d}": r['direction'] for r in
                  json.loads(Path(real_record['args']['directions']).read_text())['records']}
    for k, v in train_dirs.items():
        if not np.allclose(table[k].cpu().numpy(), v, atol=1e-6):
            raise ValueError('评测方向与训练净化方向不一致')

    class EvalDirections:
        def for_camera(self, camera):
            return table[frame_key(camera.image_name_train_light)]
    directions = EvalDirections()
    visibility = StaticVisibility(real_record['args']['shadow_bias'], 10 ** 9)
    pipe = Namespace(**run.config)

    def rgb(camera, colors=None):
        if colors is None:
            return render_rgb(camera, run.pc, pipe, torch.zeros(3, device='cuda'), directions, lighting,
                              visibility, iteration)
        return render(camera, run.pc, pipe, torch.zeros(3, device='cuda'), 0., 0., 0., override_color=colors)
    run.rgb = rgb
    run.light_prediction_provenance = dict(method='GT主光方向（lights.json light_dir_world，评测专用）', gt_light_read=True,
        training_used='TRAIN117 GT方向（净化表）', heldout_direction_used_for_training=False,
        lighting_checkpoint=lighting_json['lighting'], lighting_sha256=_sha(Path(model) / f'lighting/iteration_{iteration}/lighting.json'),
        wrapper=str(Path(__file__).resolve()), wrapper_sha256=_sha(Path(__file__)), composite=composite)
    return run


def make_plan(args):
    commands = _orig_make_plan(args)
    record = _marker(args.model)
    if record is None or record.get('baseline') != BASELINE:
        return commands
    wrapper = str(Path(__file__).resolve())
    for item in commands:
        cmd = item['command']
        if 'LIHAN.scripts.eldl_1024_eval' in cmd:
            cmd[cmd.index('LIHAN.scripts.eldl_1024_eval')] = 'LIHAN.scripts.eldl_1024_eval_static_gt'
        import shlex
        item['shell'] = shlex.join(cmd)
    plan_path = args.output / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['steps'] = commands
    plan['adapter_wrapper'] = wrapper
    plan['adapter_wrapper_sha256'] = _sha(wrapper)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    with (args.output / 'README.md').open('a') as f:
        f.write('\nadapter：static_eldl_mixed 经 `' + wrapper + '`（SHA256 ' + _sha(wrapper) +
                '）登记；GPU渲染步骤调用该包装器，CPU指标步骤仍为冻结入口原文。RGB用GT主光方向（评测专用）。\n')
    return commands


E.model_record, E.adapter, E.make_plan = model_record, adapter, make_plan

if __name__ == '__main__':
    E.main()
