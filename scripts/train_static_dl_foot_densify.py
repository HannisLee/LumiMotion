"""从当前最佳G2-T06@16000延续足部恢复：只比较增密统计支路。"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scene.static_dl_foot_recovery import FootRecoveryGaussianModel as GaussianModel
from scene.static_dl_foot_recovery import opacity_summary, opacity_adam_summary, audit_topology_equivalence
from scene.static_dl_foot_densify import accumulate_screen_statistics
from scene.dataset_readers import readCamerasFromTransforms, getNerfppNorm
from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
from scene.static_dl_ambient import render_photometric, audit_initial, component_diagnostics
from scene.static_dl_normal_opacity import geometry_losses, audit_normal_opacity
from scene.static_dl_censored import censored_photometric_loss, photometric_diagnostics
from scene.static_dl_shadow_split import require_native_backend
from scene.stage2_photometric import frame_key
from utils.camera_utils import cameraList_from_camInfos
from utils.sh_utils import RGB2SH


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def parameter_hashes(pc, lights):
    values = {name: value for name, value in vars(pc).items() if isinstance(value, torch.nn.Parameter)}
    values['light_log_intensity'] = lights.log_intensity
    return {name: dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            for name,value in sorted(values.items())}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--init_model', type=Path, required=True)
    parser.add_argument('--init_iteration', type=int, required=True)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--source_path', type=Path, required=True)
    parser.add_argument('--densification_route', choices=['original', 'non_normal_floor'], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=1012)
    parser.add_argument('--audit_only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='独立smoke_test，从真实16000起点运行110步至16110。')
    args = parser.parse_args()
    # 本轮不再训练间接底项；以下为显式记录的共同固定配置，不暴露额外CLI变量。
    args.ambient_mode, args.ambient_cap, args.ambient_lr = 'zero', .1, .001
    args.normal_opacity_route = 'detach_opacity'
    args.reset_policy = 'skip_all'
    clock_start = args.init_iteration
    output, init, source = args.model_path.resolve(), args.init_model.resolve(), args.source_path.resolve()
    if 'output' not in output.parts or output.name != 'model_mlp' or output.exists():
        parser.error('只允许output内尚不存在的独立model_mlp目录。')
    if args.smoke:
        if 'smoke_test' not in output.parts or args.steps != 110:
            parser.error('smoke只允许独立smoke_test、110步。')
    elif args.steps != 2000:
        parser.error('本轮正式配对固定新增2000步。')
    if args.init_iteration != 16000:
        parser.error('本轮只从冻结单一基线G2-T06@16000初始化。')
    if any(not 0 < x <= args.steps for x in args.save_steps):
        parser.error('save_steps必须是预算内的正相对步数。')
    if args.seed != 1012:
        parser.error('配对固定seed1012。')
    if set(args.save_steps) != ({100,110} if args.smoke else {500,1000,1500,2000}):
        parser.error('必须保存预注册的真实迭代检查点。')
    provenance = require_native_backend()
    marker_path = init / 'static_dl_training.json'
    continuation_path = init / 'reset_audit_training.json'
    finished_path = init / 'reset_audit_finished.json'
    marker = json.loads(marker_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    finished = json.loads(finished_path.read_text())
    for key in ('method', 'source_sha256', 'extension_sha256'):
        if provenance.get(key) != marker['rasterizer_provenance'][key]:
            raise AssertionError('实际native与冻结G2-T06来源不同：' + key)
    config = marker['args'].copy()
    if (marker['actual_entrypoint'] != 'scripts/train_static_dl_reset_audit.py'
            or continuation['ambient_mode'] != 'zero' or continuation['smoke']
            or continuation['args']['init_iteration'] != 14000
            or continuation['post_reset_normal_route'] != 'detach_opacity' or continuation['args']['steps'] != 2000
            or finished['iteration'] != 16000 or finished['optimizer_updates'] != 2000
            or finished['first_reset_iteration'] != 15000 or not finished['post_reset_route_audited']
            or config['static_dl_mode'] != 'lambertian_visibility'
            or config['loss_model'] != 'censored' or config['normal_depth_statistic'] != 'expected'
            or config['depth_ratio'] != 1 or config['white_background']
            or config['compute_cov3D_python'] or config['densify_grad_threshold'] != .0002
            or config['densification_interval'] != 100 or config['densify_until_iter'] != 20000
            or config['static_dl_bvh_interval'] != 1):
        parser.error('只接受已冻结G2-T06自然reset后detach、真实2k终点及原物理/增密配置。')
    if init != Path(marker['args']['model_path']).resolve():
        parser.error('实际G2-T06模型路径与来源marker不同。')
    if Path(config['source_path']).resolve() != source:
        parser.error('必须沿用G2-T06原训练数据视图。')
    ply_path = init / f'point_cloud/iteration_{args.init_iteration}/point_cloud.ply'
    deform_path = init / f'deform/iteration_{args.init_iteration}/deform.pth'
    lighting_path = init / f'static_dl_lighting/iteration_{args.init_iteration}/lighting.json'
    lighting = json.loads(lighting_path.read_text())
    if (lighting['ambient_fraction'] != 0. or lighting['ambient_mode'] != 'zero'
            or lighting['optimizer_updates'] != 2000 or lighting['iteration'] != 16000
            or lighting['normal_opacity_route'] != 'detach_opacity'):
        parser.error('G2-T06源lighting必须保存a0与2000次真实更新。')
    expected_initial_sha256 = {'static_dl_training.json': '8e5d3c826a55b20a1510f3821e741463fea45624eda20fe092b3198f7c32c4c7', 'reset_audit_training.json': '5180de352826dc791f3581e35238f99dbb7538fdeaacc2eed6480a4e201996d0', 'reset_audit_finished.json': '8008a1185004668e4db77ad86f97d0726676cc737f70429977778a822bb81504', 'point_cloud/iteration_16000/point_cloud.ply': '346cc983263fb5d6243b5b276e73a636e865de9be38662d9db6636c2bdd4fccc', 'deform/iteration_16000/deform.pth': '343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0', 'static_dl_lighting/iteration_16000/lighting.json': '85490b1dcce17306f4bf228a945e6950c89efbfa1fe071a1671523073ada3586'}
    if any(digest(init / rel) != expected for rel,expected in expected_initial_sha256.items()):
        raise AssertionError('G2-T06冻结输入资产SHA不符。')
    expected_shared_sources = {'scene/static_dl_normal_opacity.py': '0d1c9209fea508395b0fd50e75630b289208c59cd45579fb2f42687ba8cb2cc3', 'scene/static_dl_ambient.py': '1c899faba045954d73dff1965960bf12411d5dbfa279b529fe552948304e3a0a', 'scene/static_dl_photometric.py': '11eeb3f6f6edcdf5eb139f39f33299cf13e77080516be70f928b1f4af5c84e05', 'scene/static_dl_censored.py': '0b101abf66a304ffd977fc333e276d20dfd513a0850fc84dbe3bc553d7898d8b', 'scene/static_dl_shadow_split.py': '5b20265f8d48894cd7f570fd709badd2b19c0acf2402b4d2913e211a69277640', 'scene/gaussian_model.py': 'ecc51a879b9de64c15d9ff3f4c56d642bb271351a2bca6065b2d091db0b5fd92', 'gaussian_renderer/__init__.py': 'c9c720974818de8492e51a0082a9d3c860ffa108dde9db8f8d2c01356ad92bf4', 'scene/dataset_readers.py': 'f80470bd1b4ca933c23656cc35938b53f2e4c309e797ed3dc3182cdd7f737a60', 'utils/camera_utils.py': '8300e9c4b03de754cdfc49f2fecabccc695c6aec5bbda58e1bb6d41429ecccbd', 'scene/static_dl_foot_recovery.py': '95429d7aeeae04d5ddaf9d6bd51e1c5e41577d22cbd36e359951f7d2f712d2f3'}
    root = Path(__file__).resolve().parents[1]
    if any(digest(root / rel) != expected for rel,expected in expected_shared_sources.items()):
        raise AssertionError('共享科学来源发生变化。')
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for suffix in ('weight', 'bias'):
            if torch.count_nonzero(state[name + '.' + suffix]):
                raise AssertionError('G2-T06源不是严格静态零几何头。')
    # 三套RNG均固定；相机使用独立Python RNG，增密点数分叉不会改变相机顺序。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    camera_rng = random.Random(args.seed)
    config.update(model_path=str(output), source_path=str(source),
                  iterations=clock_start + args.steps, data_device='cpu',
                  load2gpu_on_the_fly=True, save_iterations=[clock_start + x for x in sorted(set(args.save_steps + [args.steps]))],
                  test_iterations=[], ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                  ambient_lr=args.ambient_lr, normal_opacity_route=args.normal_opacity_route, reset_policy=args.reset_policy, densification_route=args.densification_route)
    pipe = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(ply_path))
    # load_ply为Stage2额外复制此属性，但原Stage1没有此优化器组；保留它会破坏原增密字典契约。
    del pc._albedo_dc_stage1
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('方向文件与G2-T06保存lighting SHA不符。')
    expected = {str(x).zfill(4) for x in range(1, 151) if x % 10}
    if set(lights.directions) != expected:
        raise AssertionError('方向必须恰为原135训练帧。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    metadata_path = source / 'transforms_train.json'
    metadata = json.loads(metadata_path.read_text())
    if len(metadata['frames']) != 135 or {str(x['source_frame']).zfill(4) for x in metadata['frames']} != expected:
        raise AssertionError('相机metadata必须恰为原135训练帧。')
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', config['train_light_folder'], False, no_bg=True)
    cameras = cameraList_from_camInfos(infos, 1., pipe)
    if len(cameras) != 135 or {frame_key(c.image_name_train_light) for c in cameras} != expected:
        raise AssertionError('实际加载相机必须恰为原135训练帧。')
    training_inputs = {str(p): digest(p) for p in [metadata_path, Path(config['static_dl_directions'])]
                       + [Path(i.image_path_train_light) for i in infos]}
    if len(training_inputs) != 137 or training_inputs != continuation['training_inputs']:
        raise AssertionError('原137训练输入必须逐项匹配G2-T06冻结SHA。')
    pc.camera_extent = getNerfppNorm(infos)['radius']
    loaded_parameter_hashes = parameter_hashes(pc, lights)
    # 原G2-T06实参继承原30k XYZ LR，零Adam及零增密累计；不声称恢复不可用的状态。
    pc.training_setup(pipe)
    pc.update_learning_rate(clock_start)
    pc.max_radii2D = torch.zeros(len(pc.get_xyz), device='cuda')
    initial_count = len(pc.get_xyz)
    ambient = torch.zeros((), device='cuda')
    light_groups = [dict(params=[lights.log_intensity], lr=config['static_dl_light_lr'], name='direct_intensity')]
    light_optimizer = torch.optim.Adam(light_groups)
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    background = torch.zeros(3, device='cuda')
    if len(pc.optimizer.state) or len(light_optimizer.state) or torch.count_nonzero(pc.xyz_gradient_accum) or torch.count_nonzero(pc.denom):
        raise AssertionError('共同起点必须新Adam零动量和零增密统计。')
    preview = random.Random(args.seed)
    expected_first_camera_index = list(range(len(cameras))).pop(preview.randrange(len(cameras)))
    initialized_parameter_hashes = parameter_hashes(pc, lights)
    initialization = dict(
        loaded_parameter_hashes=loaded_parameter_hashes,
        initialized_parameter_hashes=initialized_parameter_hashes,
        gaussian_optimizer_state_count=len(pc.optimizer.state),
        light_optimizer_state_count=len(light_optimizer.state),
        statistic_nonzero_counts={name: int(torch.count_nonzero(getattr(pc, name)))
            for name in ('xyz_gradient_accum', 'denom', 'max_radii2D')},
        point_count=initial_count, xyz_lr=float(pc.optimizer.param_groups[0]['lr']),
        expected_first_camera_index=expected_first_camera_index,
        expected_first_camera=cameras[expected_first_camera_index].image_name_train_light,
        optimizer_updates=0, restored_optimizer=False,
        scope='冻结PLY/lighting参数重载后至新优化器初始化前后哈希；不是恢复历史Adam。')
    initialization['checks'] = dict(
        parameters_unchanged=loaded_parameter_hashes == initialized_parameter_hashes,
        gaussian_adam_empty=initialization['gaussian_optimizer_state_count'] == 0,
        light_adam_empty=initialization['light_optimizer_state_count'] == 0,
        statistics_zero=all(v == 0 for v in initialization['statistic_nonzero_counts'].values()))
    initialization['status'] = 'PASS' if all(initialization['checks'].values()) else 'FAILED'
    if initialization['status'] != 'PASS':
        raise AssertionError('真实初始化参数/优化器/统计核验失败。')
    output.mkdir(parents=True)
    write_json(output / 'initialization_audit.json', initialization)
    (output / 'cfg_args').write_text(str(Namespace(**config)))
    root = Path(__file__).resolve().parents[1]
    source_files = ('scripts/train_static_dl_foot_densify.py', 'scene/static_dl_foot_densify.py', 'scene/static_dl_foot_recovery.py',
                    'scene/static_dl_normal_opacity.py',
                    'scene/static_dl_ambient.py',
                    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
                    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                    'scene/dataset_readers.py', 'utils/camera_utils.py')
    initial_files = (marker_path, continuation_path, finished_path, ply_path, deform_path, lighting_path)
    audit = dict(actual_entrypoint='scripts/train_static_dl_foot_densify.py',
                 reset_policy=args.reset_policy, clock_start=clock_start, densification_route=args.densification_route,
                 clock_shift_diagnostic=clock_start != args.init_iteration,
                 normal_opacity_route=args.normal_opacity_route,
                 args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 inherited_config=config,
                 initial_sources={str(p): digest(p) for p in initial_files},
                 new_sources={str(root / p): digest(root / p) for p in source_files},
                 training_inputs=training_inputs,
                 frozen_training_inputs_match=True, frozen_native_provenance_match=True,
                 expected_first_camera_index=expected_first_camera_index, expected_first_camera=cameras[expected_first_camera_index].image_name_train_light,
                 initial_adam_state_empty=True, initial_density_statistics_zero=True,
                 initial_point_count=initial_count, train_images=135, test_images_loaded=False,
                 forbidden_gt_supervision=False, optimizer_state_restored=False,
                 optimizer_moments='新建Adam，所有动量为零', densification_statistics='xyz_gradient_accum/denom/max_radii2D均重置零',
                 saved_optimizer_scope='保存Gaussian和光强Adam、增密累计、RNG；BVH缓存未序列化，不声明完整运行时无缝恢复',
                 inherited_direct_intensity=lighting['intensity_from_train_rgb'],
                 initial_xyz_lr=pc.optimizer.param_groups[0]['lr'], xyz_lr_global_schedule_steps=30000,
                 initial_optimizer_groups={g['name']: g['lr'] for g in pc.optimizer.param_groups},
                 camera_extent=float(pc.camera_extent), inherited_spatial_lr_scale=5,
                 ambient_initial=0., ambient_cap=args.ambient_cap, ambient_mode=args.ambient_mode,
                 ambient_formula='rho_linear * a * sum(E) / pi', ambient_spatial_parameters=0,
                 ambient_temporal_parameters=0, ambient_trainable_scalars=0,
                 rgb_to_real_xyz=True, scale_opacity_trainable=True, visibility_detached=True,
                 shared_deformation_forward_called=False, loss_model='censored', normal_depth_statistic='expected',
                 photo_depth_ratio=1., normal_weight=config['static_dl_normal_weight'],
                 distortion_weight=config['lambda_dist'], alpha_weight=config['lambda_alpha_loss'],
                 densification=True, opacity_resets_in_run='以reset_events.jsonl和finished真实计数为准', source_boundary_reset_replayed=False,
                 normal_opacity_detached=args.normal_opacity_route == 'detach_opacity',
                 density_order='先反传和统计；先优化器更新再原增密；保存前参数已更新；终点不执行无后续优化的增密/reset',
                 camera_sampling='独立固定Python RNG、不放回遍历全部135相机',
                 update_budget=args.steps, smoke=args.smoke)
    write_json(output / 'foot_densify_training.json', audit)
    manifest = dict(marker)
    manifest.update(args=config, method='G2-T06同起点新Adam延续与增密支路下限单因素对照',
                    actual_entrypoint=audit['actual_entrypoint'], rasterizer_provenance=provenance,
                    time_dependent_color_optimized=False, test_images=0,
                    ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap, normal_opacity_route=args.normal_opacity_route, reset_policy=args.reset_policy, densification_route=args.densification_route)
    write_json(output / 'static_dl_training.json', manifest)
    shutil.copyfile(ply_path, output / 'input.ply')
    topology_audit = audit_topology_equivalence(ply_path, pipe, pc.camera_extent, output)
    write_json(output / 'initial_topology_operator_audit.json', topology_audit)
    if topology_audit['status'] != 'PASS':
        raise RuntimeError('隔离副本的原拓扑与审计拓扑不同义；保留现场。')
    cameras[0].load2device()
    initial_audit = audit_initial(cameras[0], pc, pipe, background, lights, visibility, args.init_iteration)
    normal_audit = audit_normal_opacity(cameras[0], pc, pipe, background, lights, visibility, args.init_iteration)
    cameras[0].load2device('cpu')
    write_json(output / 'initial_normal_opacity_audit.json', normal_audit)
    write_json(output / 'initial_checkpoint_gradient_audit.json', initial_audit)
    if initial_audit['status'] != 'PASS' or normal_audit['status'] != 'PASS':
        raise RuntimeError('初始真实photo或normal-opacity路线审计失败；保留现场，不继续训练。')
    if args.audit_only:
        print('零更新初始审计PASS。', flush=True)
        return
    stack = []
    reset_count = 0
    skipped_reset_count = 0
    save_steps = set(args.save_steps + [args.steps])
    settings = Namespace(shadow_weight=config['static_dl_shadow_weight'], highlight_weight=config['static_dl_highlight_weight'],
                         residual_threshold=config['static_dl_residual_threshold'])
    for step in range(1, args.steps + 1):
        iteration = clock_start + step
        if not stack:
            stack = list(range(len(cameras)))
        camera = cameras[stack.pop(camera_rng.randrange(len(stack)))]
        if step == 1 and camera.image_name_train_light != audit['expected_first_camera']:
            raise AssertionError('真实首相机与独立seed推导不同。')
        camera.load2device()
        should_log = step == 1 or step % 100 == 0 or step in save_steps
        package = render_photometric(camera, pc, pipe, background, lights, visibility, iteration,
                                     ambient, component_images=should_log)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        photo, diagnostic = censored_photometric_loss(package['render'], target, alpha,
            settings.shadow_weight, settings.highlight_weight, settings.residual_threshold)
        geometry, geometry_package = geometry_losses(camera, pc, pipe, background, alpha, package, args.normal_opacity_route)
        loss = photo + config['lambda_alpha_loss'] * geometry['alpha'] + config['static_dl_normal_weight'] * geometry['normal'] + config['lambda_dist'] * geometry['distortion']
        if not torch.isfinite(loss):
            raise FloatingPointError('续训loss非有限。')
        details = {}
        if should_log:
            details.update(photometric_diagnostics(package['render'], package['render_linear'], target, alpha, pc, settings))
            details.update(component_diagnostics(package, target, alpha, args.ambient_cap))
        loss.backward()
        gradient_record = {}
        for name, parameter in [('xyz', pc._xyz), ('rotation', pc._rotation), ('scaling', pc._scaling),
                                ('opacity', pc._opacity), ('albedo', pc._albedo_dc), ('light', lights.log_intensity)]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('续训允许路径梯度缺失/非有限：' + name)
            if step == 1:
                gradient_record[name] = dict(norm=float(parameter.grad.norm()), max_abs=float(parameter.grad.abs().max()))
        if step == 1:
            gradient_record['ambient'] = None if ambient.grad is None else float(ambient.grad)
            write_json(output / 'first_training_gradient_audit.json', dict(step=step, iteration=iteration,
                camera=camera.image_name_train_light, gradients=gradient_record,
                points=len(pc.get_xyz), trace_rays=len(pc.get_xyz) * len(lights.log_intensity),
                normal_opacity_route=args.normal_opacity_route, rgb_to_real_geometry=True, visibility_detached=True))
        with torch.no_grad():
            selected = package['visibility_filter']
            pc.max_radii2D[selected] = torch.maximum(pc.max_radii2D[selected], package['radii'][selected])
            density_record = accumulate_screen_statistics(pc, package, geometry_package, args.densification_route, audit=step == 1)
            if step == 1:
                write_json(output / 'initial_densification_route_audit.json', density_record['audit'])
            if should_log:
                density_record.update(step=step, iteration=iteration, camera=camera.image_name_train_light)
                with (output / 'densification_scores.jsonl').open('a') as stream:
                    stream.write(json.dumps(density_record, ensure_ascii=False) + '\n')
            used_xyz_lr = pc.optimizer.param_groups[0]['lr']
            pc.optimizer.step()
            light_optimizer.step()
            pc.optimizer.zero_grad(set_to_none=True)
            light_optimizer.zero_grad(set_to_none=True)
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lights.log_intensity.clamp_(-5, 5)
            ambient.clamp_(0., args.ambient_cap)
            if float(ambient) != 0.:
                raise AssertionError('纯直射控制底项发生更新。')
            pc.update_learning_rate(iteration)
            # 终点保存已更新模型，不做后面已无优化恢复的reset/拓扑改动。
            if step < args.steps and iteration < config['densify_until_iter']:
                if iteration > config['densify_from_iter'] and iteration % config['densification_interval'] == 0:
                    pc.densify_and_prune_audited(config['densify_grad_threshold'], config['min_opacity'],
                        pc.camera_extent, 20 if iteration > config['opacity_reset_interval'] else None,
                        output, step, iteration)
                    visibility.invalidate()
                if iteration % config['opacity_reset_interval'] == 0:
                    reset_before = dict(opacity=opacity_summary(pc), adam=opacity_adam_summary(pc))
                    do_reset = False  # 本轮跳过重置；18000终点无后续更新，本来也不执行结构/reset。
                    if do_reset:
                        pc.reset_opacity()
                        reset_count += 1
                        visibility.invalidate()
                    else:
                        skipped_reset_count += 1
                    event = dict(step=step, iteration=iteration, reset_policy=args.reset_policy,
                        performed=do_reset, before=reset_before,
                        after=dict(opacity=opacity_summary(pc), adam=opacity_adam_summary(pc)),
                        order='optimizer更新→增密及裁剪→本reset事件→保存')
                    with (output / 'reset_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            if should_log:
                record = dict(step=step, iteration=iteration, camera=camera.image_name_train_light,
                    normal_opacity_route=args.normal_opacity_route,
                    loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geometry.items()},
                    **{k: float(v) for k, v in diagnostic.items()}, **details,
                    mean_visibility=float(package['mean_visibility']), points=len(pc.get_xyz),
                    ambient_pre_update=float(package['ambient_fraction']), ambient_post_update=float(ambient),
                    light_post_update=lights.log_intensity.exp().cpu().tolist(), xyz_lr_used=used_xyz_lr)
                with (output / 'foot_densify_metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step in save_steps:
                pc.save_ply(str(output / f'point_cloud/iteration_{iteration}/point_cloud.ply'))
                optimizer_path = output / f'optimizer/iteration_{iteration}.pth'
                optimizer_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(gaussian=pc.optimizer.state_dict(), light=light_optimizer.state_dict(),
                    xyz_gradient_accum=pc.xyz_gradient_accum, denom=pc.denom, max_radii2D=pc.max_radii2D,
                    python_rng=random.getstate(), camera_rng=camera_rng.getstate(),
                    numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), remaining_camera_indices=list(stack),
                    iteration=iteration, optimizer_updates=step, reset_policy=args.reset_policy, densification_route=args.densification_route,
                    clock_shift_diagnostic=clock_start != args.init_iteration), optimizer_path)
                destination = output / f'deform/iteration_{iteration}/deform.pth'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(deform_path, destination)
                write_json(output / f'static_dl_lighting/iteration_{iteration}/lighting.json', dict(
                    iteration=iteration, mode='lambertian_visibility', directions_sha256=lights.sha256,
                    intensity_from_train_rgb=lights.log_intensity.exp().cpu().tolist(),
                    ambient_fraction=float(ambient), ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                    ambient_formula=audit['ambient_formula'], actual_entrypoint=audit['actual_entrypoint'],
                    normal_opacity_route=args.normal_opacity_route, reset_policy=args.reset_policy, densification_route=args.densification_route,
                    clock_start=clock_start, clock_shift_diagnostic=clock_start != args.init_iteration,
                    visibility_detached=True, temporal_geometry=False, temporal_color=False,
                    loss_model='censored', normal_depth_statistic='expected', initial_iteration=args.init_iteration,
                    continuation_step=step, optimizer_updates=step, smoke=args.smoke))
        camera.load2device('cpu')
    final_source_sha = {str(p): digest(p) for p in initial_files}
    if final_source_sha != audit['initial_sources']:
        raise AssertionError('G2-T06起点资产SHA在训练过程中变化。')
    write_json(output / 'foot_densify_finished.json', dict(status='TRAINING_FINISHED', steps=args.steps,
        iteration=clock_start + args.steps, optimizer_updates=args.steps, point_count=len(pc.get_xyz),
        ambient_fraction=float(ambient), normal_opacity_route=args.normal_opacity_route,
        initial_sources_unchanged=True, reset_policy=args.reset_policy, densification_route=args.densification_route,
        opacity_resets_performed=reset_count, opacity_resets_skipped=skipped_reset_count,
        clock_start=clock_start, clock_shift_diagnostic=clock_start != args.init_iteration, goal_mae='尚未评测'))


if __name__ == '__main__':
    main()
