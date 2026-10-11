"""从G2-T03@14000跨自然reset续训；仅重置后改变normal到opacity路线。"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scene.static_dl_reset_audit import AuditedGaussianModel as GaussianModel, preserved_gradient_state
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


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--init_model', type=Path, required=True)
    parser.add_argument('--init_iteration', type=int, required=True)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--source_path', type=Path, required=True)
    parser.add_argument('--post_reset_normal_route', choices=['full', 'detach_opacity'], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=1012)
    parser.add_argument('--audit_only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='独立110步smoke；14010人工reset，只验证状态机，不作为自然15000结果。')
    args = parser.parse_args()
    # 本轮不再训练间接底项；以下为显式记录的共同固定配置，不暴露额外CLI变量。
    args.ambient_mode, args.ambient_cap, args.ambient_lr = 'zero', .1, .001
    output, init, source = args.model_path.resolve(), args.init_model.resolve(), args.source_path.resolve()
    if 'output' not in output.parts or output.name != 'model_mlp' or output.exists():
        parser.error('只允许output内尚不存在的独立model_mlp目录。')
    if args.smoke:
        if 'smoke_test' not in output.parts or args.steps != 110:
            parser.error('smoke只允许独立smoke_test、110步。')
    elif args.steps != 2000:
        parser.error('本轮正式配对固定新增2000步。')
    required_saves = {10, 11, 100, 110} if args.smoke else {900, 1000, 1100, 1200, 2000}
    if set(args.save_steps) != required_saves:
        parser.error('必须保存约定的reset前后与裁剪窗口检查点。')
    if args.seed != 1012:
        parser.error('配对固定seed1012。')
    if args.init_iteration != 14000:
        parser.error('本轮只从新最佳G2-T03@14000初始化。')
    if any(not 0 < x <= args.steps for x in args.save_steps):
        parser.error('save_steps必须是预算内的正相对步数。')
    provenance = require_native_backend()
    marker_path = init / 'static_dl_training.json'
    continuation_path = init / 'normal_opacity_training.json'
    marker = json.loads(marker_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    expected_markers = {'static_dl_training.json': 'bb5cc0077414a95677399b9d31ff188b47c9376b6dd40e1692fac599a4876bab', 'normal_opacity_training.json': 'cbcd62e92357521b3c2eb36be8e1251c34c591d9ef1022fb4fcebcc74adc80ea'}
    if {p.name: digest(p) for p in (marker_path, continuation_path)} != expected_markers:
        raise AssertionError('共同源训练marker SHA不符。')
    config = marker['args'].copy()
    if (marker['actual_entrypoint'] != 'scripts/train_static_dl_normal_opacity.py'
            or continuation['ambient_mode'] != 'zero' or continuation['smoke']
            or continuation['args']['init_iteration'] != 12000 or continuation['args']['steps'] != 2000
            or continuation['normal_opacity_route'] != 'full'
            or config['normal_opacity_route'] != 'full'
            or marker['normal_opacity_route'] != 'full'
            or config['densify_until_iter'] != 20000
            or config['opacity_reset_interval'] != 3000
            or config['static_dl_mode'] != 'lambertian_visibility'
            or config['loss_model'] != 'censored' or config['normal_depth_statistic'] != 'expected'
            or config['depth_ratio'] != 1 or config['white_background']
            or config['compute_cov3D_python']):
        parser.error('只接受G2-T03真实2k终点、a0/censored/expected/黑背景/原生scale-rotation起点。')
    if Path(config['source_path']).resolve() != source:
        parser.error('必须沿用G2-T03原训练数据视图。')
    ply_path = init / f'point_cloud/iteration_{args.init_iteration}/point_cloud.ply'
    deform_path = init / f'deform/iteration_{args.init_iteration}/deform.pth'
    lighting_path = init / f'static_dl_lighting/iteration_{args.init_iteration}/lighting.json'
    lighting = json.loads(lighting_path.read_text())
    if lighting['ambient_fraction'] != 0. or lighting['ambient_mode'] != 'zero' or lighting['optimizer_updates'] != 2000 or lighting['normal_opacity_route'] != 'full':
        parser.error('G2-T03源lighting必须保存a0与2000次真实更新。')
    expected_init = Path('output/1012-06-LH-static-jumpingjacks_DL_V2-G2-T03-normal透明度梯度控制2k/model_mlp').resolve()
    if init != expected_init:
        parser.error('共同源必须是已验收的G2-T03 full@14000。')
    expected_assets = {'point_cloud.ply': '9b43deb28405aa9e07c41026d102e550c1713b8a90b8fee48a3ead8b0cd250e2', 'deform.pth': '343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0', 'lighting.json': 'd190cc031d3ba94cfbeeae4e3c9f4c92e8c9e5dfd7922ddac8c3ef313b947e51'}
    if {p.name: digest(p) for p in (ply_path, deform_path, lighting_path)} != expected_assets:
        raise AssertionError('共同源14000资产SHA不符。')
    root = Path(__file__).resolve().parents[1]
    expected_sources = {'scripts/train_static_dl_normal_opacity.py': '9f0e8d0e2f5a21b670a059aa62ac18b08ceff83c3b4e90aecc86a15663b9e752', 'scene/static_dl_normal_opacity.py': '0d1c9209fea508395b0fd50e75630b289208c59cd45579fb2f42687ba8cb2cc3', 'scene/static_dl_ambient.py': '1c899faba045954d73dff1965960bf12411d5dbfa279b529fe552948304e3a0a', 'scene/static_dl_photometric.py': '11eeb3f6f6edcdf5eb139f39f33299cf13e77080516be70f928b1f4af5c84e05', 'scene/static_dl_censored.py': '0b101abf66a304ffd977fc333e276d20dfd513a0850fc84dbe3bc553d7898d8b', 'scene/static_dl_shadow_split.py': '5b20265f8d48894cd7f570fd709badd2b19c0acf2402b4d2913e211a69277640', 'scene/gaussian_model.py': 'ecc51a879b9de64c15d9ff3f4c56d642bb271351a2bca6065b2d091db0b5fd92', 'gaussian_renderer/__init__.py': 'c9c720974818de8492e51a0082a9d3c860ffa108dde9db8f8d2c01356ad92bf4', 'scene/dataset_readers.py': 'f80470bd1b4ca933c23656cc35938b53f2e4c309e797ed3dc3182cdd7f737a60', 'utils/camera_utils.py': '8300e9c4b03de754cdfc49f2fecabccc695c6aec5bbda58e1bb6d41429ecccbd'}
    # 路径可来自旧snapshot；以项目相对路径严格匹配十项内容SHA。
    source_hashes = {next((r for r in expected_sources if Path(k).as_posix().endswith('/' + r)), k): v
                     for k,v in continuation['new_sources'].items()}
    if source_hashes != expected_sources or any(digest(root / r) != h for r,h in expected_sources.items()):
        raise AssertionError('共同源十项科学实现或当前冻结引用已变化。')
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for suffix in ('weight', 'bias'):
            if torch.count_nonzero(state[name + '.' + suffix]):
                raise AssertionError('G2-T03源不是严格静态零几何头。')
    # 三套RNG均固定；相机使用独立Python RNG，增密点数分叉不会改变相机顺序。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    camera_rng = random.Random(args.seed)
    reset_iteration = 14010 if args.smoke else 15000
    config.pop('normal_opacity_route', None)
    config.update(pre_reset_normal_route='full', model_path=str(output), source_path=str(source),
                  iterations=args.init_iteration + args.steps, data_device='cpu',
                  load2gpu_on_the_fly=True, save_iterations=[args.init_iteration + x for x in sorted(set(args.save_steps + [args.steps]))],
                  test_iterations=[], ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                  ambient_lr=args.ambient_lr, post_reset_normal_route=args.post_reset_normal_route)
    pipe = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(ply_path))
    # load_ply为Stage2额外复制此属性，但原Stage1没有此优化器组；保留它会破坏原增密字典契约。
    del pc._albedo_dc_stage1
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('方向文件与G2-T03保存lighting SHA不符。')
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
    pc.camera_extent = getNerfppNorm(infos)['radius']
    # 原G2-T03实参继承原30k XYZ LR，零Adam及零增密累计；不声称恢复不可用的状态。
    pc.training_setup(pipe)
    pc.update_learning_rate(args.init_iteration)
    pc.max_radii2D = torch.zeros(len(pc.get_xyz), device='cuda')
    initial_count = len(pc.get_xyz)
    ambient = torch.zeros((), device='cuda')
    light_groups = [dict(params=[lights.log_intensity], lr=config['static_dl_light_lr'], name='direct_intensity')]
    light_optimizer = torch.optim.Adam(light_groups)
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    background = torch.zeros(3, device='cuda')
    output.mkdir(parents=True)
    (output / 'cfg_args').write_text(str(Namespace(**config)))
    root = Path(__file__).resolve().parents[1]
    source_files = ('scripts/train_static_dl_reset_audit.py', 'scene/static_dl_reset_audit.py',
                    'scripts/train_static_dl_normal_opacity.py', 'scene/static_dl_normal_opacity.py',
                    'scene/static_dl_ambient.py',
                    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
                    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                    'scene/dataset_readers.py', 'utils/camera_utils.py')
    initial_files = (marker_path, continuation_path, ply_path, deform_path, lighting_path)
    audit = dict(actual_entrypoint='scripts/train_static_dl_reset_audit.py',
                 pre_reset_normal_route='full', post_reset_normal_route=args.post_reset_normal_route,
                 args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 inherited_config=config,
                 initial_sources={str(p): digest(p) for p in initial_files},
                 new_sources={str(root / p): digest(root / p) for p in source_files},
                 training_inputs={str(p): digest(p) for p in [metadata_path, Path(config['static_dl_directions'])]
                                  + [Path(i.image_path_train_light) for i in infos]},
                 initial_point_count=initial_count, train_images=135, test_images_loaded=False,
                 forbidden_gt_supervision=False, optimizer_state_restored=False,
                 optimizer_moments='新建Adam，所有动量为零', densification_statistics='xyz_gradient_accum/denom/max_radii2D均重置零',
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
                 densification=True, opacity_resets_in_run=1, scheduled_reset_iteration=reset_iteration,
                 artificial_smoke_reset=args.smoke, route_activation='first_reset_iteration+1', source_boundary_reset_replayed=False,
                 normal_opacity_detached_before_reset=False,
                 normal_opacity_detached_after_reset=args.post_reset_normal_route == 'detach_opacity',
                 density_order='先反传和统计；先优化器更新再原增密；保存前参数已更新；终点不执行无后续优化的增密/reset',
                 camera_sampling='独立固定Python RNG、不放回遍历全部135相机',
                 update_budget=args.steps, smoke=args.smoke)
    write_json(output / 'reset_audit_training.json', audit)
    manifest = dict(marker)
    manifest.pop('normal_opacity_route', None)
    manifest.update(args=config, method='G2-T03参数续训与reset后normal-opacity单因素梯度对照',
                    actual_entrypoint=audit['actual_entrypoint'], rasterizer_provenance=provenance,
                    time_dependent_color_optimized=False, test_images=0,
                    ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap, pre_reset_normal_route='full',
                    post_reset_normal_route=args.post_reset_normal_route)
    write_json(output / 'static_dl_training.json', manifest)
    shutil.copyfile(ply_path, output / 'input.ply')
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
    pc.start_topology_audit(output / 'topology_audit', args.init_iteration)
    first_reset_iteration = None
    post_reset_audited = False
    stack = []
    save_steps = set(args.save_steps + [args.steps])
    settings = Namespace(shadow_weight=config['static_dl_shadow_weight'], highlight_weight=config['static_dl_highlight_weight'],
                         residual_threshold=config['static_dl_residual_threshold'])
    def save_checkpoint(destination_root, iteration, step, phase):
        if destination_root != output:
            destination_root.mkdir(parents=True, exist_ok=False)
            event_config = dict(config, model_path=str(destination_root))
            (destination_root / 'cfg_args').write_text(str(Namespace(**event_config)))
            write_json(destination_root / 'static_dl_training.json', dict(manifest, args=event_config, checkpoint_phase=phase))
            write_json(destination_root / 'reset_audit_training.json', dict(audit, event_only=True, checkpoint_phase=phase, parent_model=str(output)))
        pc.save_ply(str(destination_root / f'point_cloud/iteration_{iteration}/point_cloud.ply'))
        destination = destination_root / f'deform/iteration_{iteration}/deform.pth'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deform_path, destination)
        write_json(destination_root / f'static_dl_lighting/iteration_{iteration}/lighting.json', dict(
            iteration=iteration, mode='lambertian_visibility', directions_sha256=lights.sha256,
            intensity_from_train_rgb=lights.log_intensity.exp().cpu().tolist(),
            ambient_fraction=float(ambient), ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
            ambient_formula=audit['ambient_formula'], actual_entrypoint=audit['actual_entrypoint'],
            post_reset_normal_route=args.post_reset_normal_route,
            normal_opacity_route=effective_route, checkpoint_phase=phase,
            next_update_normal_opacity_route=(args.post_reset_normal_route if first_reset_iteration is not None else 'full'),
            first_reset_iteration=first_reset_iteration,
            visibility_detached=True, temporal_geometry=False, temporal_color=False,
            loss_model='censored', normal_depth_statistic='expected', initial_iteration=args.init_iteration,
            continuation_step=step, optimizer_updates=step, smoke=args.smoke))

    for step in range(1, args.steps + 1):
        iteration = args.init_iteration + step
        pc.audit_iteration = iteration
        effective_route = (args.post_reset_normal_route if first_reset_iteration is not None else 'full')
        if not stack:
            stack = list(range(len(cameras)))
        camera = cameras[stack.pop(camera_rng.randrange(len(stack)))]
        camera.load2device()
        if first_reset_iteration is not None and not post_reset_audited:
            preserved = preserved_gradient_state(pc, lights)
            reset_audit = audit_normal_opacity(camera, pc, pipe, background, lights, visibility, iteration)
            same_state = preserved == preserved_gradient_state(pc, lights)
            reset_audit['checks']['existing_gradients_and_density_statistics_unchanged'] = dict(passed=same_state)
            reset_audit.update(iteration=iteration, first_reset_iteration=first_reset_iteration, effective_route=effective_route)
            if not same_state:
                reset_audit['status'] = 'FAILED'
            write_json(output / 'post_reset_normal_opacity_audit.json', reset_audit)
            if reset_audit['status'] != 'PASS':
                raise RuntimeError('reset后真实路线审计失败；不继续更新。')
            post_reset_audited = True
        should_log = (step == 1 or step % 100 == 0 or step in save_steps
                      or iteration in (reset_iteration, reset_iteration + 1))
        package = render_photometric(camera, pc, pipe, background, lights, visibility, iteration,
                                     ambient, component_images=should_log)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        photo, diagnostic = censored_photometric_loss(package['render'], target, alpha,
            settings.shadow_weight, settings.highlight_weight, settings.residual_threshold)
        geometry, geometry_package = geometry_losses(camera, pc, pipe, background, alpha, package, effective_route)
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
                post_reset_normal_route=args.post_reset_normal_route, rgb_to_real_geometry=True, visibility_detached=True))
        with torch.no_grad():
            # 两副本屏幕余切向量相加，然后只计一次norm/denom；不重复真实XYZ梯度。
            screen = geometry_package['viewspace_points'].grad
            if screen is None or not torch.isfinite(screen).all():
                raise FloatingPointError('expected副本screen余切缺失/非有限。')
            package['viewspace_points'].grad.add_(screen)
            selected = package['visibility_filter']
            pc.max_radii2D[selected] = torch.maximum(pc.max_radii2D[selected], package['radii'][selected])
            pc.add_densification_stats(package['viewspace_points'], selected)
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
                    before_count = len(pc.get_xyz)
                    pc.densify_and_prune(config['densify_grad_threshold'], config['min_opacity'],
                                         pc.camera_extent, 20 if iteration > config['opacity_reset_interval'] else None)
                    visibility.invalidate()
                    with (output / 'topology_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(step=step, iteration=iteration, before=before_count,
                                                     after=len(pc.get_xyz)), ensure_ascii=False) + '\n')
                if iteration == reset_iteration:
                    save_checkpoint(output / 'reset_event/pre_reset/model_mlp', iteration, step, 'pre_reset')
                    pc.reset_opacity()
                    first_reset_iteration = iteration
                    save_checkpoint(output / 'reset_event/post_reset/model_mlp', iteration, step, 'post_reset')
                    visibility.invalidate()
            if should_log:
                record = dict(step=step, iteration=iteration, camera=camera.image_name_train_light,
                    post_reset_normal_route=args.post_reset_normal_route,
                    effective_normal_opacity_route=effective_route, first_reset_iteration=first_reset_iteration,
                    loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geometry.items()},
                    **{k: float(v) for k, v in diagnostic.items()}, **details,
                    mean_visibility=float(package['mean_visibility']), points=len(pc.get_xyz),
                    ambient_pre_update=float(package['ambient_fraction']), ambient_post_update=float(ambient),
                    light_post_update=lights.log_intensity.exp().cpu().tolist(), xyz_lr_used=used_xyz_lr)
                with (output / 'reset_audit_metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step in save_steps:
                save_checkpoint(output, iteration, step, 'post_reset' if iteration == reset_iteration else 'after_update')
        camera.load2device('cpu')
    if first_reset_iteration != reset_iteration or not post_reset_audited:
        raise AssertionError('未覆盖约定reset和reset后路线验证。')
    pc.audit_state('final')
    final_source_sha = {str(p): digest(p) for p in initial_files}
    if final_source_sha != audit['initial_sources']:
        raise AssertionError('G2-T03起点资产SHA在训练过程中变化。')
    write_json(output / 'reset_audit_finished.json', dict(status='TRAINING_FINISHED', steps=args.steps,
        iteration=args.init_iteration + args.steps, optimizer_updates=args.steps, point_count=len(pc.get_xyz),
        ambient_fraction=float(ambient), post_reset_normal_route=args.post_reset_normal_route,
        first_reset_iteration=first_reset_iteration, post_reset_route_audited=post_reset_audited,
        initial_sources_unchanged=True, goal_mae='尚未评测'))


if __name__ == '__main__':
    main()
