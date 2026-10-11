#!/usr/bin/env python3
"""1011 六组训练、GT 导出、全量 MAE 与中文报告的持久化执行入口。"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PY = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
CONTROL = ROOT / 'output/1011-07-六组训练总控'
BOOT = ROOT / 'LIHAN/scripts/bootstrap_1011.py'
NATIVE = CONTROL / 'native'
CONFIG = CONTROL / 'batch.json'
REPORT = ROOT / 'LIHAN/DOC/报告/1011-07-四个V2数据集六组训练与全部MAE总报告.md'
DATASETS = [
    ('LH-static-jumpingjacks_DL_V2', 'static'),
    ('LH-static-jumpingjacks_EL_V2', 'static'),
    ('LH-dynamic-jumpingjacks_DL_V2', 'danamic'),
    ('LH-dynamic-jumpingjacks_EL_V2', 'danamic'),
]


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def update(exp, **fields):
    path = Path(exp['output']) / 'state.json'
    state = read_json(path) if path.exists() else {}
    state.update(fields, updated_at=now())
    write_json(path, state)


def env(gpu=None):
    result = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  MKL_NUM_THREADS='4', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                  OPENCV_IO_ENABLE_OPENEXR='1', PYTHONUNBUFFERED='1')
    result['PATH'] = str(Path(PY).parent) + ':' + result.get('PATH', '')
    result['LD_LIBRARY_PATH'] = str(CONTROL / 'tools/libs/usr/lib/x86_64-linux-gnu') + ':' + result.get('LD_LIBRARY_PATH', '')
    if gpu is not None:
        result['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return result


def cmd(repo, module, *args):
    return [PY, str(BOOT), str(repo), module] + [str(a) for a in args]


def common(exp, source=None):
    return ['--source_path', source or exp['source'], '--model_path', exp['model_cli'],
            '--is_blender', '--eval', '--load2gpu_on_the_fly', '--resolution', '2',
            '--train_light_folder', 'images', '--test_light_folder', 'images']


def native_commands(exp):
    base = common(exp)
    return [
        ('stage1', cmd(NATIVE, 'scripts.train_stage1', *base,
            '--gt_alpha_mask_as_scene_mask', '--iterations', 35000,
            '--test_iterations', 35000, '--save_iterations', 10000, 20000, 30000, 35000,
            '--densify_until_iter', 20000, '--lambda_separation', .001,
            '--d_xyz_loss_weight', .001, '--binarization_warm_up', 1000,
            '--depth_ratio', 1., '--d_color_reg_loss_weight', .01)),
        ('stage2', cmd(NATIVE, 'scripts.train_stage2', *base,
            '--gt_alpha_mask_as_scene_mask', '--load_iter', 35000,
            '--iterations', 55000, '--test_iterations', 55000,
            '--save_iterations', 40000, 50000, 55000,
            '--diffuse_sample_num', 512, '--depth_ratio', 0)),
        ('stage1_insights', cmd(NATIVE, 'scripts.render_stage1_insights', *base,
            '--load_iter', 35000, '--depth_ratio', 0)),
        ('materials', cmd(NATIVE, 'scripts.render_materials', *base,
            '--load_iter', 55000, '--depth_ratio', 0)),
        ('relight_hdr', cmd(NATIVE, 'scripts.render_relight_with_hdr', *base,
            '--load_iter', 55000, '--depth_ratio', 0, '--diffuse_sample_num', 2048,
            '--hdr', ROOT / 'example_envmaps/golden_bay_4k_32x16_rot330.hdr')),
        ('eval_nvs', cmd(NATIVE, 'scripts.eval_nvs_dynamic',
            *common(exp, exp['eval_native']), '--load_iter', 55000,
            '--depth_ratio', 0, '--diffuse_sample_num', 2048)),
    ]


def variant_commands(exp, base_exp):
    shared = ['--source_path', exp['eval_source'], '--stage1_model_path', base_exp['stage1_model'],
        '--model_path', exp['model_cli'], '--load_iter', 35000, '--iterations', 55000,
        '--save_iterations', 40000, 50000, 55000, '--test_iterations', 55000,
        '--depth_ratio', 0, '--resolution', 2, '--is_blender', '--eval',
        '--load2gpu_on_the_fly', '--gt_alpha_mask_as_scene_mask',
        '--train_light_folder', 'images', '--test_light_folder', 'images',
        '--deform-type', 'mlp', '--sh_degree', 3, '--hyper_dim', 1]
    out = Path(exp['output'])
    if exp['kind'] == 'pbr':
        options = ['--render_mode', 'original_ir', '--loss_preset', 'pbr_gt_directional_native',
            '--pbr_gt_lights_path', out / 'gt_directions_only.json',
            '--pbr_light_intensity_scale', '3.141592653589793']
    else:
        options = ['--render_mode', 'photometric_lambertian', '--loss_preset', 'lambertian_gt',
            '--photometric_light_mode', 'gt_directional',
            '--photometric_gt_lights_path', out / 'gt_directions.json',
            '--photometric_light_intensity', -1, '--photometric_albedo_lr', .001,
            '--lambda_dssim', .2, '--lambda_photometric_albedo_prior', .001,
            '--lambda_photometric_light_smooth', 0, '--photometric_log_interval', 100]
    return [('stage2', cmd(ROOT, 'scripts.train_stage2', *shared, *options)),
        ('render', cmd(ROOT, 'scripts.render_stage2', '--model_path', exp['model'],
            '--load_iter', 55000, '--task', 'all', '--hdr_samples', 2048,
            '--hdr_filepath', ROOT / 'example_envmaps/golden_bay_4k_32x16_rot330.hdr'))]


def eval_commands(exp, iteration):
    out = Path(exp['output']) / f'evaluation_{iteration}'
    gt = Path(exp['gt'])
    shared = ['--model_path', exp['model'], '--source_path', exp['eval_source']]
    return [
        ('albedo', cmd(ROOT, 'scripts.eval_albedo_gt', *shared, '--iterations', iteration,
            '--gt_albedo_dir', Path(exp['raw']) / 'albedo', '--split', 'test',
            '--output_path', out / 'albedo', '--save_float')),
        ('predicted_meshes', cmd(ROOT, 'scripts.extract_geometry_mesh_dynamic', *shared,
            '--load_iter', iteration, '--split', 'test', '--resolution', 2,
            '--voxel_size', .02, '--sdf_trunc', .08, '--alpha_threshold', .5,
            '--output_dir', out / 'predicted_meshes')),
        ('normal_depth', cmd(ROOT, 'scripts.eval_geometry_gt', *shared,
            '--load_iter', iteration, '--gt_root', exp['raw'], '--gt_depth_dir', gt / 'gt_depth',
            '--split', 'test', '--resolution', 2, '--output_dir', out / 'normal_depth',
            '--pred_mesh_manifest', out / 'predicted_meshes/manifest.json', '--save_arrays')),
    ] + ([] if exp['kind'] == 'lambertian' or iteration == 35000 else [
        ('roughness', cmd(ROOT, 'scripts.eval_roughness_gt', *shared, '--iteration', iteration,
            '--gt_roughness_dir', gt / 'roughness', '--split', 'test',
            '--output_path', out / 'roughness')),
        ('verify_roughness', [PY, '-m', 'scripts.verify_roughness_eval', str(out / 'roughness'),
                              '--output', str(out / 'roughness/verification.json')])
    ]) + [('verify_all_mae', [PY, str(ROOT / 'LIHAN/scripts/audit_1011_mae.py'),
                             str(out), str(iteration)] +
          (['--require-roughness'] if iteration == 55000 and exp['kind'] != 'lambertian' else []))]


def shell_line(command):
    import shlex
    return shlex.join([str(x) for x in command])


def prepare():
    if CONFIG.exists():
        raise FileExistsError('批次已存在，禁止覆盖。')
    CONTROL.mkdir(parents=True, exist_ok=True)
    NATIVE.mkdir(exist_ok=False)
    baseline = ROOT.parent / 'LumiMotion'
    tracked = subprocess.check_output(['git', '-C', str(baseline), 'ls-files', '-z']).split(b'\0')
    for raw in tracked:
        if not raw:
            continue
        name = raw.decode()
        source = baseline / name
        if name.split('/')[0] in {'DOC', 'output', 'data'} or not source.is_file():
            continue
        destination = NATIVE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(CONTROL / 'native_provenance.json', {
        '来源仓库': str(baseline),
        '提交': subprocess.check_output(['git', '-C', str(baseline), 'rev-parse', 'HEAD']).decode().strip(),
        '文件SHA256': {str(p.relative_to(NATIVE)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in NATIVE.rglob('*') if p.is_file()},
        '说明': '复制原生仓库实际工作区的已跟踪文件；训练算法保持原生版本。'})
    experiments = []
    for index, (dataset, category) in enumerate(DATASETS, 1):
        title = f'1011-{index:02d}-{dataset}-原生LumiMotion两阶段'
        out = ROOT / 'output' / title
        out.mkdir(exist_ok=False)
        (out / 'logs').mkdir()
        (out / 'commands').mkdir()
        raw = ROOT / 'data/LH-data' / category / dataset
        source = out / 'data_view'
        conversion = [PY, str(ROOT / 'LIHAN/scripts/prepare_jump_lh_dataset.py'), str(raw), str(source),
                      '--test-stride', '10', '--geometry-kind', 'static' if category == 'static' else 'dynamic',
                      '--original-loader-translation']
        with (out / 'logs/conversion.log').open('w') as log:
            subprocess.run(conversion, stdout=log, stderr=subprocess.STDOUT, check=True, env=env(), cwd=ROOT)
        for view_name, compensate, composite in [('eval_view', False, False), ('eval_native', True, True)]:
            view = out / view_name
            view.mkdir()
            if not composite:
                (view / 'images').symlink_to(source / 'images')
            else:
                from PIL import Image
                import numpy as np
                (view / 'images').mkdir()
                for path in (source / 'images').glob('*.png'):
                    rgba = np.asarray(Image.open(path)).copy()
                    rgba[..., :3] = (rgba[..., :3].astype(float) * (rgba[..., 3:4] / 255.)).astype('uint8')
                    Image.fromarray(rgba).save(view / 'images' / path.name)
            for split in ('train', 'test'):
                transforms = read_json(source / f'transforms_{split}.json')
                for frame in transforms['frames']:
                    if not compensate:
                        frame['transform_matrix'] = frame['source_camera_to_world']
                write_json(view / f'transforms_{split}.json', transforms)
        exp = dict(id=index, dataset=dataset, kind='native', raw=str(raw.resolve()), output=str(out),
                   source=str(source), eval_source=str(out / 'eval_view'), eval_native=str(out / 'eval_native'),
                   gt=str(CONTROL / f'GT-{dataset}'), model_cli=str(out / 'model'), model=str(out / 'model_mlp'),
                   stage1_model=str(out / 'stage1_frozen'),
                   document=str(ROOT / 'LIHAN/DOC/训练' / (title + '训练计划.md')))
        experiments.append(exp)
        write_json(out / 'conversion_command.json', conversion)
        update(exp, status='PREPARED', stage='等待启动')
    for index, kind, label in [(5, 'pbr', 'PBR'), (6, 'lambertian', 'Lambertian')]:
        base = experiments[0]
        title = f'1011-{index:02d}-{base["dataset"]}-{label}与GT方向光源'
        out = ROOT / 'output' / title
        out.mkdir(exist_ok=False)
        (out / 'logs').mkdir()
        (out / 'commands').mkdir()
        exp = dict(base, id=index, kind=kind, output=str(out), model_cli=str(out / 'model'),
                   model=str(out / 'model_mlp'), dependency=1,
                   document=str(ROOT / 'LIHAN/DOC/训练' / (title + '训练计划.md')))
        lights = read_json(Path(base['raw']) / 'lights.json')['frames']
        table = {'coordinate_space': 'world', 'frames': {}}
        direction_table = {}
        import numpy as np
        for key, frame in lights.items():
            records = []
            for lamp in frame['lights']:
                direction = lamp['light_dir_world']
                records.append(dict(type='directional', source_type='SUN', light_dir_world=direction,
                                    intensity=1., light_rgb=[1., 1., 1.], sun_angular_size_radians=0,
                                    direction_convention='unit vector from surface toward light, for max(n dot l, 0)'))
            if not all(np.allclose(records[0]['light_dir_world'], r['light_dir_world'], atol=1e-6)
                       for r in records):
                raise ValueError('Lambertian 单方向模型不能合并不同方向的 SUN。')
            table['frames'][key] = {'lights': records}
            direction_table[key] = {'light_dir_world': records[0]['light_dir_world']}
        write_json(out / 'gt_directions_only.json', table)
        write_json(out / 'gt_directions.json', direction_table)
        experiments.append(exp)
        update(exp, status='WAITING_DEPENDENCY', stage='等待01组Stage1 35000完成')
    batch = dict(created_at=now(), server='ushas', environment='lumimotion-ushas',
                 max_tasks_per_gpu=2, experiments=experiments)
    write_json(CONFIG, batch)
    for exp in experiments:
        training = native_commands(exp) if exp['kind'] == 'native' else variant_commands(exp, experiments[0])
        planned = training + [(f'eval55000_{name}', command) for name, command in eval_commands(exp, 55000)]
        write_json(Path(exp['output']) / 'planned_commands.json', planned)
        for step, command in planned:
            (Path(exp['output']) / 'commands' / (step + '.sh')).write_text(
                '#!/bin/bash\nset -euo pipefail\ncd ' + str(ROOT) + '\n' + shell_line(command) + '\n')
        text = f'# {Path(exp["document"]).stem}\n\n服务器：ushas；环境：lumimotion-ushas；批次编号：1011（用户指定）。\n\n'
        text += f'原始数据：`{exp["raw"]}`。训练source：`{exp["source"] if exp["kind"] == "native" else exp["eval_source"]}`。MAE source：`{exp["eval_source"]}`。\n\n'
        text += f'模型：`{exp["model"]}`；输出及日志：`{exp["output"]}`。150帧，135训练／15测试，测试10、20至150；resolution=2（400×400），黑背景。\n\n'
        text += ('从头随机十万点初始化，原生Stage1 35000步→原生IR Stage2累计55000步；两阶段算法均来自独立原生代码快照。\n\n' if exp['kind'] == 'native' else
                 '只读01组35000步Stage1来源，Stage2新增20000步至55000；独立输出，不覆盖原生模型。\n\n')
        text += '输入Alpha沿用既有转换器的albedo Alpha；不以GT albedo RGB、normal、depth、roughness或骨骼作为训练监督。原生组不读取GT光照；05/06使用GT方向，05两盏单位白光×π、06只用训练RGB标定非负光强；不把标定值解释为Blender原能量。\n\n'
        text += '调度按GPU实际空余显存和进程数，每卡最多2个计算任务；05/06的依赖等待不占GPU。原生源相机平移符号仅在输入中预补偿，MAE使用原始世界相机，不改变训练算法。\n\n'
        text += '全15测试帧：Albedo线性RGB MAE、Roughness参数MAE、GS Render／Depth／Mesh Normal角度MAE（度）、Expected／Median Z Depth MAE（世界单位，并记录单位比例）；同时保存覆盖率、共同像素法线分数、逐帧CSV、NPZ、代表图、日志与独立复算。Stage1记录几何MAE，SH无物理Albedo/Roughness。Lambertian无Roughness参数，记N/A。\n\n'
        text += 'RGB／Alpha／Normal／分离四类目检与最终PASS/FAILED必须在实际验收后回填，当前为PENDING；失败产物全部保留。\n\n'
        text += '\n\n'.join('## ' + step + '\n\n```bash\n' + shell_line(command) + '\n```\n\n日志：`logs/' + step + '.log`。' for step, command in planned)
        Path(exp['document']).parent.mkdir(parents=True, exist_ok=True)
        Path(exp['document']).write_text(text + '\n')
        (Path(exp['output']) / 'README.md').write_text(text + '\n')
    refresh()


def execute(exp, step, command, gpu=None):
    out = Path(exp['output'])
    log_path = out / 'logs' / (step + '.log')
    with log_path.open('x') as log:
        log.write(f'开始：{now()}\n命令：{shell_line(command)}\nGPU：{gpu}\n')
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=env(gpu), stdout=log, stderr=subprocess.STDOUT)
        update(exp, stage=step, child_pid=child.pid, gpu=gpu, status='RUNNING')
        refresh()
        code = child.wait()
        log.write(f'\n结束：{now()}；退出码：{code}\n')
    with (out / 'commands/executed.jsonl').open('a') as file:
        file.write(json.dumps(dict(step=step, command=command, gpu=gpu, exit_code=code,
                                   log=str(log_path), end=now()), ensure_ascii=False) + '\n')
    if code:
        raise RuntimeError(f'{step}退出码{code}；日志{log_path}')


def gpu_acquire(exp, update_waiting=True):
    import collections
    lock_file = CONTROL / 'gpu.lock'
    while True:
        with lock_file.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                             '--format=csv,noheader,nounits']).decode().strip().splitlines()
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader,nounits']).decode().strip().splitlines()
            counts = collections.Counter(r.split(',')[0].strip() for r in apps if ',' in r)
            allocations_path = CONTROL / 'gpu_allocations.json'
            allocations = read_json(allocations_path) if allocations_path.exists() else {}
            active = {}
            for pid, record in allocations.items():
                try:
                    os.kill(int(pid), 0)
                    active[pid] = record
                except ProcessLookupError:
                    pass
            options = []
            for row in rows:
                index, uuid, free = [c.strip() for c in row.split(',')]
                reservations = sum(int(r['gpu']) == int(index) for r in active.values())
                # 调度间隔内尚未初始化CUDA的进程也计为预留槽，防止超发。
                initializing = sum(int(r['gpu']) == int(index) and time.time() - r['time'] < 90
                                   for r in active.values())
                occupied = max(counts[uuid] + initializing, reservations)
                if occupied < 2 and int(free) >= 12000:
                    options.append((int(free), int(index)))
            if options:
                gpu = max(options)[1]
                active[str(os.getpid())] = dict(gpu=gpu, experiment=exp['id'], time=time.time())
                write_json(allocations_path, active)
                return gpu
            write_json(allocations_path, active)
        if update_waiting:
            update(exp, status='WAITING_GPU', stage='等待每卡最多2任务及12GB空闲显存')
        time.sleep(20)


def gpu_release():
    with (CONTROL / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = CONTROL / 'gpu_allocations.json'
        data = read_json(path) if path.exists() else {}
        data.pop(str(os.getpid()), None)
        write_json(path, data)


def gpu_step(exp, step, command):
    gpu = gpu_acquire(exp)
    try:
        execute(exp, step, command, gpu)
    finally:
        gpu_release()


def wait_ready(exp, path, dependency_exp=None):
    while not Path(path).exists():
        if (Path(path).parent / 'failed.json').exists():
            raise RuntimeError('GT准备失败：' + str(Path(path).parent / 'failed.json'))
        if dependency_exp:
            state = read_json(Path(dependency_exp['output']) / 'state.json')
            if state.get('status') == 'FAILED':
                raise RuntimeError('依赖训练失败，保留本组等待记录。')
        time.sleep(20)


def run(index):
    batch = read_json(CONFIG)
    exp = batch['experiments'][index - 1]
    out = Path(exp['output'])
    update(exp, runner_pid=os.getpid(), launched_at=now())
    failures = []
    try:
        if exp['kind'] == 'native':
            commands = native_commands(exp)
            gpu_step(exp, *commands[0])
            for name in ('eval_view', 'eval_native'):
                shutil.copy2(Path(exp['source']) / 'points3d.ply', out / name / 'points3d.ply')
            frozen = Path(exp['stage1_model'])
            frozen.mkdir(exist_ok=False)
            for relative in ['cfg_args', 'point_cloud/iteration_35000/point_cloud.ply',
                             'deform/iteration_35000/deform.pth']:
                target = frozen / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(exp['model']) / relative, target)
            write_json(out / 'stage1_ready.json', dict(iteration=35000, model=str(frozen), time=now()))
            gpu_step(exp, *commands[1])
            for step, command in commands[2:]:
                try:
                    gpu_step(exp, step, command)
                except Exception as error:
                    failures.append(str(error))
        else:
            base_exp = batch['experiments'][0]
            wait_ready(exp, Path(base_exp['output']) / 'stage1_ready.json', base_exp)
            for step, command in variant_commands(exp, base_exp):
                if step == 'stage2':
                    gpu_step(exp, step, command)
                else:
                    try:
                        gpu_step(exp, step, command)
                    except Exception as error:
                        failures.append(str(error))
        try:
            execute(exp, 'artifact_audit', [PY, str(ROOT / 'LIHAN/scripts/review_1011_artifacts.py'), str(out)], '')
        except Exception as error:
            failures.append(str(error))
        update(exp, status='WAITING_GT', stage='训练与渲染完成，等待GT评测准备')
        gt_path = Path(exp['gt'])
        while not (gt_path / 'ready.json').exists() and not (gt_path / 'failed.json').exists():
            time.sleep(20)
        # 每项独立落档，单项失败不跳过其他 MAE。
        if (gt_path / 'failed.json').exists():
            failures.append('GT部分准备失败：' + str(gt_path / 'failed.json'))
        for iteration in ([35000, 55000] if exp['kind'] == 'native' else [55000]):
            for name, command in eval_commands(exp, iteration):
                # SH阶段不把SH颜色解释为物理材质。
                if iteration == 35000 and name == 'albedo':
                    continue
                try:
                    gpu_step(exp, f'eval{iteration}_{name}', command)
                except Exception as error:
                    failures.append(str(error))
                    refresh()
        write_json(out / 'evaluation_failures.json', failures)
        update(exp, status='FAILED' if failures else 'AWAITING_VISUAL_REVIEW',
               stage='评测结束，等待真实目检' if not failures else '评测失败记录已保留',
               failures=failures, completed_at=now(), child_pid=None, gpu=None)
    except Exception as error:
        update(exp, status='FAILED', stage='失败', error=str(error), traceback=traceback.format_exc())
        (out / 'FAILED').write_text(f'{now()}\n{error}\n')
        raise
    finally:
        gpu_release()
        refresh()


def prepare_gt(index):
    exp = read_json(CONFIG)['experiments'][index - 1]
    gt = Path(exp['gt'])
    gt.mkdir(exist_ok=False)
    raw = Path(exp['raw'])
    blender = CONTROL / 'tools/blender-5.2.2-linux-x64/blender'
    wrapper = ROOT / 'LIHAN/scripts/blender_1011.py'
    transforms = Path(exp['eval_source']) / 'transforms_test.json'
    prefix = [str(blender), '--factory-startup', '--disable-autoexec', '-t', '4', '-b', str(raw / 'scene_snapshot.blend'),
              '--python', str(wrapper), '--', str(raw / 'export_request.json')]
    steps = [
        ('gt_meshes', prefix + [str(ROOT / 'scripts/export_blender_geometry.py'),
            '--camera_json', str(raw / 'camera.json'), '--frames'] + [str(n) for n in range(10, 151, 10)] +
            ['--output_dir', str(gt / 'gt_meshes'), '--export_ply']),
        ('gt_depth', [PY, '-m', 'scripts.prepare_geometry_gt', '--mesh_manifest', str(gt / 'gt_meshes/manifest.json'),
            '--gt_root', str(raw), '--transforms', str(transforms), '--output_dir', str(gt / 'gt_depth'),
            '--resolution', '2', '--skip_baseline_audit']),
        ('gt_roughness', prefix + [str(ROOT / 'scripts/export_roughness_gt.py'),
            '--output_dir', str(gt / 'roughness'), '--transforms', str(transforms),
            '--resolution', '2', '--alpha_composite', '--engine', 'cycles']),
    ]
    write_json(gt / 'planned_commands.json', steps)
    failures = {}
    for name, command in steps:
        with (gt / f'{name}.log').open('w') as log:
            log.write('命令：' + shell_line(command) + '\n')
            log.flush()
            code = subprocess.run(command, cwd=ROOT, env=env(''), stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            failures[name] = dict(exit_code=code, log=str(gt / f'{name}.log'))
        write_json(gt / 'state.json', dict(step=name, failures=failures, time=now()))
    write_json(gt / ('failed.json' if failures else 'ready.json'), dict(failures=failures, time=now()))
    refresh()


def refresh():
    if (CONTROL / 'final_backfill.json').exists():
        return  # 已验收的实际结论由backfill_1011_report.py维护，避免旧监控回退为PENDING。
    if not CONFIG.exists():
        return
    with (CONTROL / 'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batch = read_json(CONFIG)
        lines = ['# 四个V2数据集六组训练与全部MAE总报告', '',
            f'更新时间：{now()}。批次编号1011为用户指定，实际启动时间另记；服务器ushas，环境lumimotion-ushas。', '',
            '01—04为原生LumiMotion Stage1 35k→Stage2累计55k；05/06继承01的35k检查点，分别训练PBR与Lambertian GT方向光20k。每GPU最多2个实际计算任务，已有其他用户任务纳入槽位计数。', '',
            '| 编号 | 数据集／方法 | 状态 | 当前步骤 | GPU |',
            '|---|---|---|---|---|']
        all_rows = []
        for exp in batch['experiments']:
            state = read_json(Path(exp['output']) / 'state.json')
            lines.append(f'| {exp["id"]:02d} | {exp["dataset"]}／{exp["kind"]} | {state.get("status")} | {state.get("stage")} | {state.get("gpu", "—")} |')
            all_rows.append(dict(experiment=exp['id'], dataset=exp['dataset'], method=exp['kind'], **state))
        lines += ['', '评测均为完整15测试帧，训练和测试135/15严格分离。Albedo为线性RGB，法线为角度（度），深度为相机Z，粗糙度为原始r参数。保留覆盖率和共同有效像素，禁止将缺失预测当作零误差。Lambertian无Roughness，Stage1无物理Albedo/Roughness，均标N/A。', '',
                  '| 编号／迭代 | Albedo MAE | Roughness MAE | GS Normal ° | Depth Normal ° | Mesh Normal ° | Expected Z | Median Z |',
                  '|---|---|---|---|---|---|---|---|']
        metric_rows = []
        for exp in batch['experiments']:
            for iteration in ([35000, 55000] if exp['kind'] == 'native' else [55000]):
                file = Path(exp['output']) / f'evaluation_{iteration}/mae_summary.json'
                result = read_json(file) if file.exists() else {}
                metric_rows.append(dict(experiment=exp['id'], dataset=exp['dataset'], method=exp['kind'],
                    iteration=iteration, status=result.get('status', 'PENDING'),
                    **{key: result.get(key) for key in ['albedo', 'roughness', 'gs_render', 'depth_normal', 'mesh', 'expected', 'median']}))
                values = []
                for key in ['albedo', 'roughness', 'gs_render', 'depth_normal', 'mesh', 'expected', 'median']:
                    value = result.get(key)
                    if key == 'roughness' and (exp['kind'] == 'lambertian' or iteration == 35000) or key == 'albedo' and iteration == 35000:
                        value = 'N/A'
                    values.append(f'{value:.8f}' if isinstance(value, (int, float)) else str(value or 'PENDING'))
                lines.append(f'| {exp["id"]:02d}／{iteration} | ' + ' | '.join(values) + ' |')
        lines += ['', '输入Alpha沿用albedo-alpha转换约定；不使用GT材质RGB、法线、深度、roughness、网格监督训练。已知GT方向只用于05/06，原生组学习共享环境光。EL是原双POINT场景，不能称为HDRI复刻。', '',
                  'GT从源blend在内存中恢复原导出时隐藏副本的可见性，源文件不写回；深度只用于评测，预测Mesh只融合训练相机预测深度。GT导出、训练、渲染、MAE、独立复算均记录完整命令、日志、退出码。', '',
                  '四类可视化目检：RGB、Alpha、Normal、动态／静态分离均PENDING。最终结论：PENDING，待实际训练／评测完成及真实目检后给出PASS/FAILED；失败checkpoint、图像、统计与日志全部保留。', '', '各组文档与输出：', '']
        for exp in batch['experiments']:
            lines.append(f'- {exp["id"]:02d}：[训练文档]({exp["document"]})；输出`{exp["output"]}`；实时状态`state.json`；命令`commands/`；日志`logs/`；评测`evaluation_35000/`、`evaluation_55000/`。')
            artifacts = Path(exp['output']) / 'inspection/artifacts.json'
            if artifacts.exists():
                media = read_json(artifacts)
                lines.append(f'  产物核验：{media.get("产物核验")}；完整视频解码{len(media.get("视频", []))}个；首中尾代表图：`{artifacts.parent}`。人工目检状态：{media.get("人工目检")}。')
        audit_path = CONTROL / 'input_audit.json'
        if audit_path.exists():
            input_audit = read_json(audit_path)
            lines += ['', f'实际输入核验：四组共600帧PASS；原生reader最大相机矩阵误差{max(r["maximum_reader_camera_error"] for r in input_audit):.4g}。证据`{audit_path}`。']
        lines += ['', '共享GT准备状态：', '']
        for exp in batch['experiments'][:4]:
            gt = Path(exp['gt'])
            lines.append(f'- {exp["dataset"]}：' + ('PASS' if (gt / 'ready.json').exists() else 'FAILED' if (gt / 'failed.json').exists() else 'PENDING') + f'；完整GT命令、源哈希、逐帧验证和日志：`{gt}`。')
        for exp in batch['experiments']:
            executed = Path(exp['output']) / 'commands/executed.jsonl'
            if executed.exists():
                status = '\n\n## 实际执行状态\n\n```json\n' + json.dumps(read_json(Path(exp['output']) / 'state.json'), ensure_ascii=False, indent=2) + '\n```\n'
                original = Path(exp['document']).read_text().split('\n\n## 实际执行状态')[0]
                Path(exp['document']).write_text(original + status)
                (Path(exp['output']) / 'README.md').write_text(original + status)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text('\n'.join(lines) + '\n')
        (CONTROL / 'README.md').write_text('\n'.join(lines) + '\n')
        write_json(CONTROL / 'batch_status.json', all_rows)
        write_json(CONTROL / 'mae_summary.json', metric_rows)
        with (CONTROL / 'mae_summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)


def launch():
    batch = read_json(CONFIG)
    if (CONTROL / 'launched.json').exists():
        raise FileExistsError('批次已经提交，禁止重复启动。')
    processes = []
    for index in range(1, 5):
        logfile = (CONTROL / 'logs' / f'gt_{index:02d}.log').open('x')
        child = subprocess.Popen([PY, str(Path(__file__)), 'gt', str(index)], cwd=ROOT,
                                 env=env(''), stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(dict(kind='gt', id=index, pid=child.pid))
    for exp in batch['experiments']:
        logfile = (Path(exp['output']) / 'logs/runner.log').open('x')
        child = subprocess.Popen([PY, str(Path(__file__)), 'run', str(exp['id'])], cwd=ROOT,
                                 env=env(), stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(dict(kind='training', id=exp['id'], pid=child.pid))
    logfile = (CONTROL / 'logs/monitor.log').open('x')
    child = subprocess.Popen([PY, str(Path(__file__)), 'monitor'], cwd=ROOT,
                             env=env(), stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(dict(kind='monitor', pid=child.pid))
    write_json(CONTROL / 'launched.json', dict(time=now(), processes=processes))
    refresh()
    print(json.dumps(processes, ensure_ascii=False))


def monitor():
    while True:
        refresh()
        states = [read_json(Path(e['output']) / 'state.json')['status'] for e in read_json(CONFIG)['experiments']]
        if all(s in {'FAILED', 'AWAITING_VISUAL_REVIEW', 'PASS'} for s in states):
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'run', 'gt', 'refresh', 'monitor'])
    parser.add_argument('index', type=int, nargs='?')
    args = parser.parse_args()
    if args.action == 'run':
        run(args.index)
    elif args.action == 'gt':
        prepare_gt(args.index)
    else:
        globals()[args.action]()
