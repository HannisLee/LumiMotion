#!/usr/bin/env python3
"""最佳T06增密配对独立来源适配；实际RGB仍冻结ambient renderer，a固定0。"""
from pathlib import Path
import sys
import argparse
import random
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_eval_protocol import FRAMES, PYTHON, read, digest

ENTRY = 'scripts/train_static_dl_foot_densify.py'
NEW_SOURCES = (ENTRY, 'scene/static_dl_foot_densify.py', 'scene/static_dl_foot_recovery.py', 'scene/static_dl_normal_opacity.py', 'scene/static_dl_ambient.py',
    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py', 'scene/dataset_readers.py', 'utils/camera_utils.py')


def actual_sources(model):
    model = Path(model).resolve()
    if not model.is_relative_to(ROOT / 'output') or model.name != 'model_mlp':
        raise AssertionError('实际模型必须仓库output内model_mlp。')
    marker_path = model / 'foot_densify_training.json'
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
    if marker['args']['init_iteration'] != 16000 or len(marker['initial_sources']) != 6 or len(marker['new_sources']) != 12:
        raise AssertionError('固定G2-T06@16000、6起点资产与12实际源码。')
    if marker['args']['seed'] != 1012 or marker['optimizer_state_restored'] or marker['initial_adam_state_empty'] is not True or marker['initial_density_statistics_zero'] is not True:
        raise AssertionError('必须seed1012、新Adam零动量、增密统计清零；不得冒称状态恢复。')
    density = marker['densification_route']
    if density not in ('original', 'non_normal_floor') or any(mapping['densification_route'] != density for mapping in (marker['args'],marker['inherited_config'])):
        raise AssertionError('增密支路顶层/CLI/继承配置冲突。')
    policy = marker['reset_policy']
    if policy != 'skip_all' or any(mapping['reset_policy'] != policy for mapping in (marker['args'], marker['inherited_config'])):
        raise AssertionError('真实reset策略的顶层/CLI/继承配置不一致。')
    shifted = marker['clock_shift_diagnostic']
    if shifted or marker['clock_start'] != 16000:
        raise AssertionError('clock诊断标识与实际起始计数冲突。')
    budget = 110 if marker['smoke'] else 2000
    if marker['update_budget'] != budget or marker['args']['steps'] != budget or marker['args']['smoke'] != marker['smoke']:
        raise AssertionError('正式2k或真实110步smoke预算错误。')
    if set(marker['args']['save_steps']) != ({100,110} if marker['smoke'] else {500,1000,1500,2000}):
        raise AssertionError('保存迭代不符预注册预算。')
    if marker['args']['audit_only'] or marker['frozen_training_inputs_match'] is not True or marker['frozen_native_provenance_match'] is not True:
        raise AssertionError('须真实训练入口及原输入/native冻结门禁，不能使用audit_only模型。')
    if marker['train_images'] != 135 or marker['test_images_loaded'] or marker['forbidden_gt_supervision']:
        raise AssertionError('必须仅135训练RGB/Alpha，无额外GT监督。')
    for group in ('initial_sources', 'training_inputs'):
        for path, expected in marker[group].items():
            if digest(path) != expected:
                raise AssertionError('实际训练输入或起点来源变化：' + path)
    from goal3_densify_target import BASELINE, BASELINE_SHA256
    if digest(BASELINE) != BASELINE_SHA256:
        raise AssertionError('新轮冻结JSON发生变化。')
    baseline = read(BASELINE)
    init = Path(marker['args']['init_model']).resolve()
    if str(init) != baseline['model']:
        raise AssertionError('不是已冻结单一G2-T06模型。')
    initial_files = ('static_dl_training.json','reset_audit_training.json','reset_audit_finished.json',
                     'point_cloud/iteration_16000/point_cloud.ply','deform/iteration_16000/deform.pth',
                     'static_dl_lighting/iteration_16000/lighting.json')
    if {Path(p).resolve() for p in marker['initial_sources']} != {init / p for p in initial_files}:
        raise AssertionError('六初始化来源集合不符。')
    for path, expected in baseline['checkpoint_sha256'].items():
        if marker['initial_sources'].get(path) != expected or digest(path) != expected:
            raise AssertionError('冻结T06三检查点SHA错误。')
    initial_marker_sha = {'static_dl_training.json':'8e5d3c826a55b20a1510f3821e741463fea45624eda20fe092b3198f7c32c4c7',
                          'reset_audit_training.json':'5180de352826dc791f3581e35238f99dbb7538fdeaacc2eed6480a4e201996d0',
                          'reset_audit_finished.json':'8008a1185004668e4db77ad86f97d0726676cc737f70429977778a822bb81504'}
    for name,expected in initial_marker_sha.items():
        if marker['initial_sources'].get(str(init/name)) != expected:
            raise AssertionError('冻结T06起点marker SHA错误：'+name)
    initial = read(init/'reset_audit_training.json')
    if len(marker['training_inputs']) != 137 or marker['training_inputs'] != initial['training_inputs']:
        raise AssertionError('必须完整匹配G2-T06冻结的137训练输入集合与SHA。')
    # 只接受entry明确更新的运行/方法元数据，loss和其余物理配置逐项保持。
    inherited = read(init/'static_dl_training.json')['args']
    allowed = {'model_path','source_path','iterations','data_device','load2gpu_on_the_fly','save_iterations',
               'test_iterations','ambient_mode','ambient_cap','ambient_lr','normal_opacity_route','reset_policy','densification_route'}
    config = marker['inherited_config']
    if set(config)-set(inherited)-allowed or set(inherited)-set(config) or any(config[k] != v for k,v in inherited.items() if k not in allowed):
        raise AssertionError('存在未经允许的loss/物理/学习率/增密配置改变。')
    if Path(config['model_path']).resolve() != model or Path(config['source_path']).resolve() != Path(inherited['source_path']).resolve():
        raise AssertionError('实际模型或原训练source不同。')
    if config['iterations'] != 16000+budget or config['save_iterations'] != [16000+n for n in sorted(set(marker['args']['save_steps']))] or config['test_iterations']:
        raise AssertionError('真实保存迭代/预算或测试加载不符。')
    if ('smoke_test' in model.parts) != bool(marker['smoke']):
        raise AssertionError('smoke与正式模型输出范围不符。')
    if marker['initial_point_count'] != 67081 or marker['source_boundary_reset_replayed'] or marker['xyz_lr_global_schedule_steps'] != 30000:
        raise AssertionError('实际T06点数/边界reset/原30k LR声明不符。')
    if marker['inherited_direct_intensity'] != read(init/'static_dl_lighting/iteration_16000/lighting.json')['intensity_from_train_rgb']:
        raise AssertionError('起点RGB光强未真实继承。')
    sources, snapshots = {}, set()
    available = [path for path in (model.parent / 'commands/train.json', model.parent / 'commands.json') if path.is_file()]
    if not available:
        raise AssertionError('存在marker却没有snapshot启动证据。')
    for path in available:
        record = read(path)
        argv = record.get('argv') or record.get('command')
        if not isinstance(argv, list) or len(argv) < 4 or Path(argv[0]).absolute() != Path(PYTHON) or Path(argv[1]).resolve() != ROOT / 'LIHAN/scripts/bootstrap_1011.py' or argv[3] != 'scripts.train_static_dl_foot_densify':
            raise AssertionError('必须ushas Python、真实bootstrap及foot_densify模块。')
        parsed = parse_training_argv(argv)
        for k in ('init_model','model_path','source_path'):
            if Path(parsed[k]).resolve() != Path(marker['args'][k]).resolve():
                raise AssertionError('真实CLI路径与marker不同：'+k)
        for k in ('init_iteration','densification_route','steps','seed','smoke','audit_only'):
            if parsed[k] != marker['args'][k]:
                raise AssertionError('真实CLI与marker不同：'+k)
        if set(parsed['save_steps']) != set(marker['args']['save_steps']) or Path(parsed['model_path']).resolve() != model:
            raise AssertionError('CLI保存预算/模型路径不一致。')
        snapshot = Path(argv[2]).resolve()
        if not snapshot.is_relative_to(ROOT/'output') or snapshot == ROOT/'output':
            raise AssertionError('必须output内独立训练快照。')
        if record.get('snapshot') and Path(record['snapshot']).resolve() != snapshot:
            raise AssertionError('snapshot元数据冲突。')
        manifest_path = snapshot / 'snapshot_manifest.json'
        if not manifest_path.is_file():
            raise AssertionError('真实bootstrap target没有冻结snapshot manifest；主仓运行不能冒充snapshot。')
        manifest = read(manifest_path)
        if manifest['status'] != 'READY' or Path(manifest['destination']).resolve() != snapshot:
            raise AssertionError('snapshot尚未READY或目录不符。')
        if record.get('snapshot_manifest_sha256') != digest(manifest_path) or record.get('entry_sha256') != manifest['file_sha256'].get(ENTRY):
            raise AssertionError('真实命令未绑定manifest/entry SHA。')
        for relative, expected in manifest['file_sha256'].items():
            saved = (snapshot/relative).resolve()
            if not saved.is_relative_to(snapshot) or digest(saved) != expected:
                raise AssertionError('完整快照文件SHA不一致：'+relative)
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
    sources[str(BASELINE)] = BASELINE_SHA256
    sources.update(marker['initial_sources'])
    sources.update(marker['training_inputs'])
    return marker, sources


def parse_training_argv(argv):
    """CPU独立解析真实CLI，不能用marker自我证明入口或配置。"""
    parser = argparse.ArgumentParser(add_help=False)
    for k in ('init_model','model_path','source_path','densification_route'):
        parser.add_argument('--'+k,required=True)
    for k in ('init_iteration','steps'):
        parser.add_argument('--'+k,type=int,required=True)
    parser.add_argument('--seed',type=int,default=1012)
    parser.add_argument('--save_steps',type=int,nargs='+',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--audit_only',action='store_true')
    try:
        return vars(parser.parse_args(argv[4:]))
    except SystemExit as error:
        raise AssertionError('真实训练CLI无效或存在未允许参数。') from error


def initialization_evidence(model, marker):
    """CPU核对真实初始化原始数值和独立seed/LR，PASS只描述已有训练证据。"""
    model = Path(model).resolve()
    path = model/'initialization_audit.json'
    audit = read(path)
    keys = {'parameters_unchanged','gaussian_adam_empty','light_adam_empty','statistics_zero'}
    if audit['status'] != 'PASS' or set(audit['checks']) != keys or not all(v is True for v in audit['checks'].values()):
        raise AssertionError('真实初始化四检查未通过。')
    before,after = audit['loaded_parameter_hashes'],audit['initialized_parameter_hashes']
    required = {'_xyz','_rotation','_scaling','_opacity','_albedo_dc','light_log_intensity'}
    if before != after or not required.issubset(before):
        raise AssertionError('加载后参数与新优化器初始化后参数SHA不同。')
    for name,item in before.items():
        if len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']) or item['dtype'] != 'torch.float32':
            raise AssertionError('实际参数哈希格式/精度无效：'+name)
        if name != 'light_log_intensity' and (not item['shape'] or item['shape'][0] != marker['initial_point_count']):
            raise AssertionError('实际加载参数形状与T06点数不符：'+name)
    if audit['point_count'] != marker['initial_point_count'] or audit['optimizer_updates'] != 0 or audit['restored_optimizer'] is not False or audit['gaussian_optimizer_state_count'] != 0 or audit['light_optimizer_state_count'] != 0:
        raise AssertionError('实际初始化非零更新或非新Adam。')
    if set(audit['statistic_nonzero_counts']) != {'xyz_gradient_accum','denom','max_radii2D'} or any(v != 0 for v in audit['statistic_nonzero_counts'].values()):
        raise AssertionError('实际增密统计未全部清零。')
    config = marker['inherited_config']
    t = min(max(16000/config['position_lr_max_steps'],0),1)
    expected_lr = math.exp((1-t)*math.log(config['position_lr_init']*5)+t*math.log(config['position_lr_final']*5))
    if not math.isclose(audit['xyz_lr'],expected_lr,rel_tol=1e-12,abs_tol=1e-15) or audit['xyz_lr'] != marker['initial_xyz_lr'] or marker['initial_optimizer_groups']['xyz'] != audit['xyz_lr']:
        raise AssertionError('实际16k起点XYZ LR与原30k日程不同。')
    metadata_path = Path(config['source_path'])/'transforms_train.json'
    records = sorted(read(metadata_path)['frames'],key=lambda r:int(Path(r['file_path']).stem.split('_')[-1]))
    if len(records) != 135 or {r['source_frame'] for r in records} != set(range(1,151))-set(FRAMES):
        raise AssertionError('首相机推导必须原135训练输入。')
    index = random.Random(marker['args']['seed']).randrange(len(records))
    name = Path(records[index]['file_path']).stem
    if audit['expected_first_camera_index'] != index or audit['expected_first_camera'] != name or marker['expected_first_camera_index'] != index or marker['expected_first_camera'] != name:
        raise AssertionError('实际初始化首相机与独立seed推导不符。')
    return dict(status='PASS_RECORDED_INITIALIZATION_EVIDENCE',path=str(path),sha256=digest(path),
                checks=audit['checks'],point_count=audit['point_count'],expected_xyz_lr=expected_lr,
                first_camera_index=index,first_camera=name,optimizer_state_restored=False)


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
    if config['depth_ratio'] != 1 or config['static_dl_mode'] != 'lambertian_visibility' or config['white_background'] or config['static_dl_bvh_interval'] != 1:
        raise AssertionError('固定黑背景、真实物理、photo median1。')
    if iteration <= config['static_dl_physical_start'] or iteration < config['static_dl_visibility_start']:
        raise AssertionError('真实物理/V阶段尚未启用。')
    path = run.model / f'static_dl_lighting/iteration_{iteration}/lighting.json'
    lighting = read(path)
    for key in ('actual_entrypoint', 'ambient_mode', 'ambient_cap', 'loss_model', 'normal_depth_statistic', 'normal_opacity_route', 'reset_policy', 'densification_route', 'clock_start', 'clock_shift_diagnostic', 'smoke'):
        if lighting[key] != marker[key]:
            raise AssertionError('lighting与训练marker不一致：' + key)
    if lighting['ambient_fraction'] != 0 or lighting['iteration'] != iteration or lighting['clock_start'] + lighting['continuation_step'] != iteration or lighting['optimizer_updates'] != lighting['continuation_step']:
        raise AssertionError('必须实际a0/相同保存迭代与更新计数。')
    if lighting['initial_iteration'] != 16000 or not 0 < lighting['optimizer_updates'] <= marker['update_budget']:
        raise AssertionError('保存检查点必须来自16000参数起点与真实更新预算。')
    if static.get('reset_policy') != marker['reset_policy'] or config['reset_policy'] != marker['reset_policy']:
        raise AssertionError('static marker和实际reset策略不同。')
    if static.get('densification_route') != marker['densification_route'] or config['densification_route'] != marker['densification_route']:
        raise AssertionError('static marker和实际增密支路不同。')
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
    return render, dict(actual_entrypoint=ENTRY, normal_opacity_route=route, reset_policy=marker['reset_policy'], densification_route=marker['densification_route'], clock_shift_diagnostic=marker['clock_shift_diagnostic'], mode='lambertian_visibility_with_ambient',
                        ambient_fraction=0., ambient_mode='zero', photo_depth_ratio=1,
                        visibility_enabled=True, intensity_from_train_rgb=intensity.tolist(), directions_only=True,
                        test_directions_used_only_for_evaluation=True, sources=sources,
                        scope='实际photo前向不因normal-loss反向route变化；反向证据独立训练audit，不用fresh前向冒充。')
