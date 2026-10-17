"""未知灯向静态DL的独立离线评测；原15帧计量、来源重定位和真实估计光渲染。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / 'LIHAN/scripts'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HELPERS))
PYTHON = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
CONTROL = ROOT / 'output/1017-01-静态DL未知光总控'
ORIGINAL = ROOT / 'output/archive/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议/protocol.json'
ORIGINAL_SHA = '7cf1e7f6e1ce7f223319899febdfb1d443440add68cddee191fe5bf47e1646be'
T06 = ROOT / 'output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k'
FRAMES = list(range(10, 151, 10))
METHODS = ('gs_render', 'depth_normal', 'mesh')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def locate(path):
    """只映射已经归档的output目录，不用内容近似的替代来源。"""
    path = Path(path)
    if path.exists():
        return path.resolve()
    if path.is_relative_to(ROOT / 'output'):
        archived = ROOT / 'output/archive' / path.relative_to(ROOT / 'output')
        if archived.exists():
            return archived.resolve()
    raise FileNotFoundError(path)


def protocol():
    original = locate(ORIGINAL)
    if digest(original) != ORIGINAL_SHA:
        raise AssertionError('原协议身份变化。')
    value = read(original)
    relocated = copy.deepcopy(value)
    mapping = {}
    for group in ('code_sha256', 'source_sha256'):
        relocated[group] = {}
        for path, expected in value[group].items():
            actual = locate(path)
            if digest(actual) != expected:
                raise AssertionError('原评测来源内容变化：' + str(actual))
            relocated[group][str(actual)] = expected
            if str(actual) != path:
                mapping[path] = str(actual)
    for field in ('source_path', 'gt_root', 'gt_depth_dir'):
        relocated[field] = str(locate(value[field]))
    relocated.update(original_protocol_sha256=ORIGINAL_SHA, original_protocol_path=str(original),
                     relocation_only=True, relocated_paths=mapping)
    if relocated['frames'] != FRAMES or relocated['resolution'] != 2:
        raise AssertionError('原15帧协议发生变化。')
    return relocated


def direction_check(train_path, all_path, source):
    """CPU检验135训练值恒定保留，测试值仅由训练时间线性插值。"""
    import numpy as np
    train, full = read(train_path)['frames'], read(all_path)['frames']
    expected = {f'{n:04d}' for n in range(1, 151) if n not in FRAMES}
    if set(train) != expected or set(full) != {f'{n:04d}' for n in range(1, 151)}:
        raise AssertionError('估计灯向必须严格135训练／150评测。')
    vectors = {}
    for key, record in full.items():
        if set(record) != {'light_dir_world'}:
            raise AssertionError('估计灯向文件只允许方向字段。')
        vector = np.asarray(record['light_dir_world'], dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1) > 1e-4:
            raise AssertionError('估计方向不是有限单位向量。')
        vectors[key] = vector
    for key in expected:
        if train[key] != full[key]:
            raise AssertionError('评测篡改训练估计灯向：' + key)
    times = {}
    for split in ('train', 'test'):
        for row in read(Path(source) / f'transforms_{split}.json')['frames']:
            times[f"{row['source_frame']:04d}"] = float(row['time'])
    ordered = sorted(expected, key=lambda key: (times[key], key))
    known_t = np.asarray([times[k] for k in ordered])
    values = np.stack([vectors[k] for k in ordered])
    interpolation_errors = {}
    for frame in FRAMES:
        key = f'{frame:04d}'
        predicted = np.array([np.interp(times[key], known_t, values[:, axis]) for axis in range(3)])
        predicted /= np.linalg.norm(predicted)
        error = float(np.max(np.abs(predicted-vectors[key])))
        interpolation_errors[key] = error
        if error > 2e-6:
            raise AssertionError('测试方向未遵循训练时间归一化线性插值：' + key)
    return dict(status='PASS', train_frames=135, evaluation_frames=150,
                training_values_unchanged=True, test_interpolation_max_abs=interpolation_errors,
                train_file=str(train_path), train_sha256=digest(train_path),
                evaluation_file=str(all_path), evaluation_sha256=digest(all_path),
                gt_light_read=False)


def model_sources(model, iteration):
    marker = read(model / 'unknown_light_training.json')
    static = read(model / 'static_dl_training.json')
    if marker['actual_entrypoint'] != 'scripts/train_static_unknown_light.py' or marker['gt_supervision'] is not False:
        raise AssertionError('不是实际无GT训练来源。')
    if static['actual_entrypoint'] != marker['actual_entrypoint']:
        raise AssertionError('两训练标记入口不一致。')
    sources = {str(model / name): digest(model / name) for name in
               ('unknown_light_training.json', 'static_dl_training.json')}
    for path, expected in marker['new_sources'].items():
        if digest(path) != expected:
            raise AssertionError('实际训练源码已变化：' + path)
        sources[path] = expected
        for relative in ('scripts/train_static_unknown_light.py','scene/static_unknown_light.py',
                         'scripts/train_static_dl.py','scene/static_dl_censored.py','scene/static_dl_photometric.py'):
            if Path(path).as_posix().endswith('/'+relative):
                if digest(ROOT / relative) != expected:
                    raise AssertionError('评测实际源码与真实训练快照不同：'+relative)
                sources[str(ROOT / relative)] = expected
    lighting_path = model / f'static_dl_lighting/iteration_{iteration}/lighting.json'
    lighting = read(lighting_path)
    if lighting['iteration'] != iteration or lighting['gt_supervision'] is not False:
        raise AssertionError('检查点光照不是该次无GT结果。')
    sources[str(lighting_path)] = digest(lighting_path)
    for relative in (f'point_cloud/iteration_{iteration}/point_cloud.ply', f'deform/iteration_{iteration}/deform.pth'):
        sources[str(model / relative)] = digest(model / relative)
    return marker, static['args'], lighting, sources


def physical_renderer(run, iteration, eval_directions):
    import torch
    from argparse import Namespace
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility, render_photometric
    marker, config, lighting, sources = model_sources(run.model, iteration)
    import diff_surfel_rasterization as backend
    expected_native = read(run.model / 'static_dl_training.json')['rasterizer_provenance']
    if any(backend.provenance[k] != expected_native[k] for k in ('source_sha256', 'extension_sha256')):
        raise AssertionError('实际新训练与评测native来源不同。')
    train_path = Path(config['static_dl_directions']).resolve()
    check = direction_check(train_path, eval_directions, run.source)
    if lighting['directions_sha256'] != digest(train_path):
        raise AssertionError('训练保存灯向SHA与实际135训练文件不符。')
    if config['white_background'] or config['depth_ratio'] != 1:
        raise AssertionError('真实物理配置必须黑背景、photo median深度。')
    if iteration <= config['static_dl_physical_start']:
        raise AssertionError('该检查点尚未进入物理训练阶段。')
    lights = DirectionOnlyLights(eval_directions)
    intensity = torch.tensor(lighting['intensity_from_train_rgb'], device='cuda', dtype=torch.float32)
    if intensity.shape != lights.log_intensity.shape or not torch.isfinite(intensity).all() or (intensity <= 0).any():
        raise AssertionError('保存的训练RGB光强无效。')
    with torch.no_grad():
        lights.log_intensity.copy_(intensity.log())
    lights.requires_grad_(False)
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval']) \
        if iteration >= config['static_dl_visibility_start'] else None
    pipe, background = Namespace(**config), torch.zeros(3, device='cuda')
    sources.update({str(path): digest(path) for path in (train_path, Path(eval_directions), Path(__file__))})
    run.files.update(sources)

    def render(camera):
        return render_photometric(camera, run.pc, pipe, background, lights, visibility, iteration)

    render.eval_lights, render.eval_visibility = lights, visibility
    return render, dict(actual_entrypoint=marker['actual_entrypoint'], gt_light_used=False,
                        visibility_enabled=visibility is not None, direction_validation=check,
                        intensity_from_train_rgb=intensity.tolist(), ambient_fraction=0., sources=sources)


def fresh(args):
    """新进程读取真实训练RGB前向，同已保存估计灯向适配器逐图核对，不使用GT灯向。"""
    import numpy as np
    import torch
    from argparse import Namespace
    import diff_surfel_rasterization as backend
    from scripts.eval_geometry_gt import GeometryRun
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility, render_photometric
    run = GeometryRun(args.model, args.iteration, protocol()['source_path'])
    zero_parameters = {}
    for name in ('gaussian_warp','gaussian_rotation','gaussian_scaling'):
        layer = getattr(run.deform.deform,name)
        for suffix in ('weight','bias'):
            value = getattr(layer,suffix)
            maximum = float(value.abs().max())
            zero_parameters[name+'.'+suffix] = maximum
            if maximum != 0:
                raise AssertionError('重载后的静态六参数不为零：'+name+'.'+suffix)
    _, config, lighting, sources = model_sources(args.model, args.iteration)
    adapted, provenance = physical_renderer(run,args.iteration,args.eval_directions)
    training_lights = DirectionOnlyLights(config['static_dl_directions'])
    with torch.no_grad():
        training_lights.log_intensity.copy_(torch.tensor(lighting['intensity_from_train_rgb'],device='cuda').log())
    training_lights.requires_grad_(False)
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval']) \
        if args.iteration >= config['static_dl_visibility_start'] else None
    meta = read(run.source / 'transforms_train.json')
    records = {row['source_frame']: row for row in meta['frames']}
    checks = []
    keys = ('render','render_linear','rend_alpha','rend_normal','surf_depth','surf_normal',
            'rend_dist','radii','visibility_filter','viewspace_points','light_intensity')
    with torch.no_grad():
        for frame in (1,96,149):
            camera, _, _, _ = run.camera(meta,records[frame],2)
            actual = render_photometric(camera,run.pc,Namespace(**config),torch.zeros(3,device='cuda'),
                                        training_lights,visibility,args.iteration)
            reloaded = adapted(camera)
            values = {}
            for key in keys:
                left,right = actual[key],reloaded[key]
                if left.dtype == torch.bool:
                    delta = float(torch.count_nonzero(left != right))
                    good = torch.equal(left,right)
                else:
                    delta = float((left-right).abs().max())
                    good = bool(torch.isfinite(left).all() and torch.isfinite(right).all() and
                                torch.allclose(left,right,atol=1e-6,rtol=1e-5))
                values[key] = dict(passed=good,max_abs=delta)
                if not good:
                    raise AssertionError(f'真实训练前向与估计光适配器不一致：{frame}/{key}')
            deformation = run.deformation(camera)
            heads = {key: float(deformation[key].abs().max()) if torch.is_tensor(deformation[key]) else abs(float(deformation[key]))
                     for key in ('d_xyz','d_rotation','d_scaling')}
            if any(heads.values()):
                raise AssertionError('静态几何形变头不为零。')
            checks.append(dict(frame=frame,maps=values,geometry_head_max_abs=heads))
    for path,expected in {**run.files,**sources}.items():
        if digest(path) != expected:
            raise AssertionError('fresh期间来源改变：'+path)
    maps = Path('/proc/self/maps').read_text()
    loaded = sorted({line.split()[-1] for line in maps.splitlines()
                     if any(word in line for word in ('lumimotion_pbr_cuda','surfel_tracer')) and '/' in line})
    runtime = {path: digest(path) for path in loaded if Path(path).is_file()}
    training_state = read(args.model.parent / 'state.json')
    expected_modules = training_state['runtime_modules']
    expected_hashes = set(expected_modules.values())
    if not training_state.get('captured_while_alive') or not expected_hashes.issubset(set(runtime.values())):
        raise AssertionError('fresh实际native/tracer与存活期训练记录不一致。')
    write(args.output / 'fresh_parity.json',dict(status='PASS',checks=checks,
        six_geometry_parameters_max_abs=zero_parameters,
        effective_native=backend.provenance,physical_renderer=provenance,sources={**run.files,**sources},
        runtime_modules=runtime,training_runtime_modules=expected_modules,
        scope='重载后实际训练RGB物理函数与独立估计光适配器11输出一致；不是训练期快照输出重放。GT未参与此核验。'))


def visuals(args):
    # 仅替换旧可视化器的训练来源/估计光适配；保留150帧、几何map一致性和完整解码检查。
    import goal_eval_render
    import goal_eval_visuals
    goal_eval_render.physical_renderer = physical_renderer
    sys.argv = ['goal_eval_visuals', '--model_path', str(args.model), '--load_iter', str(args.iteration),
                '--source_path', protocol()['source_path'], '--output_dir', str(args.output / 'four_classes'),
                '--eval_directions', str(args.eval_directions)]
    goal_eval_visuals.main()


def extra_regions(args):
    """原语义分区之外，固定相对平台0.02至1.0米人体高度带，避免冒充腿语义。"""
    import numpy as np
    from PIL import Image, ImageDraw
    spatial = args.output / 'spatial_diagnostic'
    original = read(spatial / 'summary.json')
    aggregate = {name: dict(eligible=0, alpha=0, normal={k: [0, 0.] for k in METHODS},
                            points={k: [0, 0., 0] for k in ('expected', 'median', 'mesh_depth')})
                 for name in ('lower_human_height_0p02_to_1m',)}
    for frame in FRAMES:
        with np.load(spatial / f'arrays/diagnostic_{frame:04d}.npz') as z:
            data = {k: z[k] for k in z.files}
        region = data['eligible'] & np.isin(data['object_id'], range(6)) & \
                 (data['gt_world_height'] >= .02) & (data['gt_world_height'] <= 1.)
        item = aggregate['lower_human_height_0p02_to_1m']
        item['eligible'] += int(region.sum())
        item['alpha'] += int((region & data['alpha_visible']).sum())
        for method in METHODS:
            valid = region & data[method + '_valid']
            item['normal'][method][0] += int(valid.sum())
            item['normal'][method][1] += float(data[method + '_angular_deg'][valid].sum())
        for field in item['points']:
            valid = region & data[field + '_point_valid']
            distances = data[field + '_point_distance_m'][valid]
            item['points'][field][0] += int(valid.sum())
            item['points'][field][1] += float(distances.sum())
            item['points'][field][2] += int((distances <= .02).sum())
    item = aggregate['lower_human_height_0p02_to_1m']
    n = item['eligible']
    result = dict(eligible_pixels=n, alpha_coverage=item['alpha']/n if n else None,
                  normal={k: dict(valid_pixels=v[0], coverage=v[0]/n if n else None,
                                  mae_deg=v[1]/v[0] if v[0] else None,
                                  missing180_mae_deg=(v[1]+180*(n-v[0]))/n if n else None)
                          for k,v in item['normal'].items()},
                  point_geometry={k: dict(valid_pixels=v[0], coverage=v[0]/n if n else None,
                                         same_ray_distance_mean_m=v[1]/v[0] if v[0] else None,
                                         same_ray_within_2cm_fraction=v[2]/n if n else None)
                                  for k,v in item['points'].items()})
    write(args.output / 'human_foot_regions.json', dict(
        scope='GT仅离线评测，不用于训练、种子或灯向挑选。高度带不是解剖腿分割，可能含下垂手部；腿部完整性须多视角人工目检。',
        fixed_height_band_m=[.02, 1.], height_origin='原15帧GT平台object6世界Z中位数',
        semantic_regions={k: original['regions'][k] for k in ('human','object_4','footwear')},
        height_regions={'lower_human_height_0p02_to_1m': result},
        source_sha256={str(spatial / 'summary.json'): digest(spatial / 'summary.json')}))
    chosen = [10, 20, 40, 80, 90, 140, 150]
    thumbnails = []
    for frame in chosen:
        with Image.open(spatial / f'figures/footwear_zoom_{frame:04d}.png') as im:
            thumb = im.convert('RGB'); thumb.thumbnail((780, 450)); thumbnails.append((frame, thumb))
    canvas = Image.new('RGB', (1560, 4*480), '#181818')
    draw = ImageDraw.Draw(canvas)
    for index, (frame, thumb) in enumerate(thumbnails):
        x, y = index % 2 * 780, index // 2 * 480
        canvas.paste(thumb, (x, y+25)); draw.text((x+8,y+6), f'Frame {frame:04d}', fill='white')
    canvas.save(args.output / 'footwear_multiview.png')


def summary(args):
    from goal_eval_audit import audit
    measured = audit(args.output, args.iteration, args.output / 'protocol.json')
    write(args.output / 't06_comparison.json', dict(
        original_protocol_sha256=ORIGINAL_SHA, candidate=measured['values'],
        reference_model=str(T06 / 'model_mlp'), reference_iteration=16000,
        reference=read(T06 / 'evaluation_16000/goal_mae_summary.json')['values'],
        delta_deg={k: measured['values'][k]-read(T06 / 'evaluation_16000/goal_mae_summary.json')['values'][k] for k in METHODS},
        reference_geometry_status='FAILED',
        scope='T06仅独立GT灯向参考，不作为初始化；delta只描述差距，不挑选未知灯向种子。'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--eval_directions', type=Path, default=CONTROL / 'estimated_directions_all.json')
    parser.add_argument('--train_directions', type=Path,
                        help='CPU或尚无模型的计划可显式指定135训练表；实际模型以保存args为准。')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--plan_destination', type=Path,
                        help='仅计划时保存到独立目录，保持实际评测output尚不存在。')
    parser.add_argument('--stage', choices=('audit', 'visuals', 'regions', 'fresh'))
    args = parser.parse_args()
    if subprocess.check_output(['hostname'], text=True).strip() != 'ushas':
        parser.error('必须使用ushas/lumimotion-ushas。')
    p = protocol()
    if args.stage:
        {'audit': summary, 'visuals': visuals, 'regions': extra_regions, 'fresh': fresh}[args.stage](args)
        return
    train_path = args.train_directions or CONTROL / 'data_train/estimated_directions_train.json'
    if args.model and (args.model / 'static_dl_training.json').is_file():
        saved_train = Path(read(args.model / 'static_dl_training.json')['args']['static_dl_directions']).resolve()
        if args.train_directions and saved_train != args.train_directions.resolve():
            parser.error('显式train_directions与实际模型保存配置不符。')
        train_path = saved_train
    directions = direction_check(train_path, args.eval_directions, p['source_path'])
    if args.cpu_check:
        print(json.dumps(dict(status='PASS', original_protocol_sha256=ORIGINAL_SHA,
                              frozen_code_count=len(p['code_sha256']), input_count=len(p['source_sha256']),
                              relocated_path_count=len(p['relocated_paths']), direction_check=directions,
                              gpu_work_started=False), ensure_ascii=False, indent=2))
        return
    if args.model is None or args.iteration is None or args.output is None:
        parser.error('必须显式指定model、iteration与独立output。')
    args.model, args.output = args.model.resolve(), args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'output'):
        parser.error('所有结果必须在output内。')
    if args.execute and (args.gpu is None or args.plan_destination is not None):
        parser.error('GPU工作须先协调并显式指定gpu。')
    destination = args.plan_destination.resolve() if args.plan_destination else args.output
    if not destination.is_relative_to(ROOT / 'output') or args.output.exists():
        parser.error('计划必须在output内，实际评测output不得已存在。')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'logs').mkdir()
    write(destination / 'protocol.json', p)
    write(destination / 'direction_validation.json', directions)
    shared = ['--model_path', str(args.model), '--load_iter', str(args.iteration), '--source_path', p['source_path']]
    boot = [PYTHON, str(HELPERS / 'bootstrap_1011.py'), str(ROOT)]
    frames = ['--frames'] + list(map(str, FRAMES))
    own = [PYTHON, str(Path(__file__).resolve()), '--model', str(args.model), '--iteration', str(args.iteration),
           '--output', str(args.output), '--eval_directions', str(args.eval_directions.resolve())]
    steps = [
        ('fresh_parity', boot + ['scripts.eval_static_unknown_light'] + own[2:] + ['--stage','fresh']),
        ('predicted_meshes', boot + ['scripts.extract_geometry_mesh_dynamic'] + shared + frames +
         ['--split','test','--resolution','2','--voxel_size','.02','--sdf_trunc','.08','--depth_trunc','20',
          '--alpha_threshold','.5','--output_dir',str(args.output / 'predicted_meshes')]),
        ('normal_depth', boot + ['scripts.eval_geometry_gt'] + shared + frames +
         ['--gt_root',p['gt_root'],'--gt_depth_dir',p['gt_depth_dir'],'--split','test','--resolution','2',
          '--alpha_threshold','.5','--interior_radius','2','--output_dir',str(args.output / 'normal_depth'),
          '--pred_mesh_manifest',str(args.output / 'predicted_meshes/manifest.json'),'--save_arrays']),
        ('audit', own + ['--stage','audit']),
        ('spatial_diagnostic', [PYTHON, str(HELPERS / 'goal2_error_spatial_diagnostic.py'), '--evaluation',str(args.output),
                                '--gt_depth_dir',p['gt_depth_dir'],'--output',str(args.output / 'spatial_diagnostic'),'--figures']),
        ('human_foot_regions', own + ['--stage','regions']),
    ]
    if not args.geometry_only:
        steps.append(('four_classes', boot + ['scripts.eval_static_unknown_light'] + own[2:] + ['--stage','visuals']))
    actual_sources = [Path(__file__).resolve()] + [HELPERS / name for name in (
        'goal_eval_visuals.py','goal_eval_render.py','goal_eval_audit.py','goal_eval_protocol.py',
        'goal2_error_spatial_diagnostic.py')] + [Path(path) for path in p['code_sha256']]
    sources = {str(path): digest(path) for path in actual_sources}
    for path,expected in sources.items():
        saved = destination / 'evaluation_sources' / Path(path).relative_to(ROOT)
        saved.parent.mkdir(parents=True,exist_ok=True)
        with saved.open('xb') as stream:
            stream.write(Path(path).read_bytes())
        if digest(saved) != expected:
            raise AssertionError('独立评测源码快照写入不一致。')
    write(destination / 'evaluation_sources/manifest.json',sources)
    write(destination / 'plan.json', dict(model=str(args.model), iteration=args.iteration,
        source=p['source_path'], original_protocol_sha256=ORIGINAL_SHA, adapter_sha256=sources,
        steps=[dict(name=n,command=c,shell=shlex.join(c)) for n,c in steps]))
    lines = ['# 未知灯向静态DL独立评测', '', f'模型：`{args.model}`；iteration：{args.iteration}。',
             f"source：`{p['source_path']}`；原协议SHA：`{ORIGINAL_SHA}`。",
             f'输出：`{args.output}`；日志：`{args.output / "logs"}`。',
             '原15测试帧10、20、…、150；foreground像素加权角MAE；参数、掩膜和6份计量源码不改。',
             '评测GT仅用于离线计量，不进入训练、灯向估计或种子选择；T06为独立历史参考。',
             '四类分离图为原binary feature门控，不能解释为语义或运动分离。', '', '## 完整命令', '']
    for name, command in steps:
        lines += [f'### {name}', '', '```bash', shlex.join(command), '```', '']
    lines += ['## 定量和目检', '', '三MAE、覆盖率、Expected/Median深度：待运行。',
              '代表图片：`footwear_multiview.png`；`four_classes/four_classes_10_80_150.png`。',
              '代表视频：`four_classes/rgb.mp4`、`alpha.mp4`、`normal.mp4`、`separation.mp4`。',
              'RGB：PENDING；Alpha：PENDING；Normal：PENDING；分离：PENDING。',
              '人物、腿足完整性：PENDING；最终结论：PENDING。验收失败保留全部产物。', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    if not args.execute:
        print('计划已写入；未启动GPU：' + str(destination / 'plan.json'))
        return
    marker, config, lighting, checkpoint_sources = model_sources(args.model,args.iteration)
    write(args.output / 'checkpoint_before.json', checkpoint_sources)
    training_env = read(args.model.parent / 'plan.json')['env']
    env = dict(os.environ, **training_env)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               TORCH_CUDA_ARCH_LIST='8.6', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    libs = locate(ROOT / 'output/1011-07-六组训练总控/tools/libs/usr/lib/x86_64-linux-gnu')
    env['LD_LIBRARY_PATH'] = str(libs)+':'+env.get('LD_LIBRARY_PATH','')
    outcomes = []
    for name, command in steps:
        protocol()
        if any(digest(path) != expected for path,expected in sources.items()):
            raise AssertionError('评测期间adapter源码变化。')
        with (args.output / f'logs/{name}.log').open('x') as stream:
            stream.write(shlex.join(command)+'\n'); stream.flush()
            code = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        outcomes.append(dict(step=name,exit_code=code,log=str(args.output / f'logs/{name}.log')))
        # 运行状态为本次新输出，允许逐步追加覆盖本状态文件；科学结果始终open(x)。
        (args.output / 'executed.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2)+'\n')
        print(name+' exit='+str(code),flush=True)
        if code:
            write(args.output / 'FAILED.json',dict(step=name,exit_code=code,retained=True))
            with (args.output / 'README.md').open('a') as stream:
                stream.write(f'\n步骤{name}退出{code}；最终结论：FAILED。失败现场完整保留。\n')
            raise SystemExit(code)
    if any(digest(path) != expected for path,expected in checkpoint_sources.items()):
        raise AssertionError('评测期间检查点或训练来源变化。')
    write(args.output / 'checkpoint_after.json',dict(status='PASS',files=checkpoint_sources))
    write(args.output / 'PIPELINE_COMPLETED.json',dict(pipeline='PASS',manual_review='PENDING',
        full_15_measurement=True,geometry_only=args.geometry_only,smoke_schedule=marker['smoke_schedule'],
        final_status='PENDING',note='流水线完成不等于人体腿足或目标通过。'))
    measured = read(args.output / 'goal_mae_summary.json')
    with (args.output / 'README.md').open('a') as stream:
        stream.write('\n## 实测结果\n\n')
        for key,value in measured['values'].items():
            stream.write(f'- {key}：{value:.12f}；覆盖率：{measured["details"][key]["coverage"]:.12f}。\n')
        stream.write('\n实际四类及多视角腿足人工目检待补；最终结论：PENDING。\n')


if __name__ == '__main__':
    main()
