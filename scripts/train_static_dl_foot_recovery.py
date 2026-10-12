"""从G2-T04@14000恢复足部：只比较15000步opacity reset策略。"""
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
    parser.add_argument('--reset_policy', choices=['standard', 'skip_15000'], required=True)
    parser.add_argument('--smoke_clock_start', type=int, default=None)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=1012)
    parser.add_argument('--audit_only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='独立smoke_test，最多150步；可显式前移全局计数验证reset。')
    args = parser.parse_args()
    # 本轮不再训练间接底项；以下为显式记录的共同固定配置，不暴露额外CLI变量。
    args.ambient_mode, args.ambient_cap, args.ambient_lr = 'zero', .1, .001
    args.normal_opacity_route = 'detach_opacity'
    if args.smoke_clock_start is not None and (not args.smoke or args.smoke_clock_start != 14990):
        parser.error('计数前移只允许smoke明确指定14990。')
    clock_start = args.init_iteration if args.smoke_clock_start is None else args.smoke_clock_start
    output, init, source = args.model_path.resolve(), args.init_model.resolve(), args.source_path.resolve()
    if 'output' not in output.parts or output.name != 'model_mlp' or output.exists():
        parser.error('只允许output内尚不存在的独立model_mlp目录。')
    if args.smoke:
        if 'smoke_test' not in output.parts or not 0 < args.steps <= 150:
            parser.error('smoke只允许独立smoke_test、1至150步。')
    elif args.steps != 2000:
        parser.error('本轮正式配对固定新增2000步。')
    if args.init_iteration != 14000:
        parser.error('本轮只从冻结单一基线G2-T04@14000初始化。')
    if any(not 0 < x <= args.steps for x in args.save_steps):
        parser.error('save_steps必须是预算内的正相对步数。')
    if not args.smoke and not {999, 1000, 1100, 1200, 1500, 2000}.issubset(args.save_steps):
        parser.error('正式训练必须保存14999/15000/15100/15200/15500/16000。')
    if args.smoke_clock_start is not None and (args.steps != 120 or not {9, 10, 110, 120}.issubset(args.save_steps)):
        parser.error('跨reset冒烟固定120步并保存14999/15000/15100/15110。')
    provenance = require_native_backend()
    marker_path = init / 'static_dl_training.json'
    continuation_path = init / 'normal_opacity_training.json'
    marker = json.loads(marker_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    config = marker['args'].copy()
    if (marker['actual_entrypoint'] != 'scripts/train_static_dl_normal_opacity.py'
            or continuation['ambient_mode'] != 'zero' or continuation['smoke']
            or continuation['args']['init_iteration'] != 12000
            or continuation['normal_opacity_route'] != 'detach_opacity' or continuation['args']['steps'] != 2000
            or config['static_dl_mode'] != 'lambertian_visibility'
            or config['loss_model'] != 'censored' or config['normal_depth_statistic'] != 'expected'
            or config['depth_ratio'] != 1 or config['white_background']
            or config['compute_cov3D_python']):
        parser.error('只接受G2-T04真实2k终点、a0/censored/expected/黑背景/原生scale-rotation起点。')
    if Path(config['source_path']).resolve() != source:
        parser.error('必须沿用G2-T04原训练数据视图。')
    ply_path = init / f'point_cloud/iteration_{args.init_iteration}/point_cloud.ply'
    deform_path = init / f'deform/iteration_{args.init_iteration}/deform.pth'
    lighting_path = init / f'static_dl_lighting/iteration_{args.init_iteration}/lighting.json'
    lighting = json.loads(lighting_path.read_text())
    if (lighting['ambient_fraction'] != 0. or lighting['ambient_mode'] != 'zero'
            or lighting['optimizer_updates'] != 2000 or lighting['iteration'] != 14000
            or lighting['normal_opacity_route'] != 'detach_opacity'):
        parser.error('G2-T04源lighting必须保存a0与2000次真实更新。')
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for suffix in ('weight', 'bias'):
            if torch.count_nonzero(state[name + '.' + suffix]):
                raise AssertionError('G2-T04源不是严格静态零几何头。')
    # 三套RNG均固定；相机使用独立Python RNG，增密点数分叉不会改变相机顺序。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    camera_rng = random.Random(args.seed)
    config.update(model_path=str(output), source_path=str(source),
                  iterations=clock_start + args.steps, data_device='cpu',
                  load2gpu_on_the_fly=True, save_iterations=[clock_start + x for x in sorted(set(args.save_steps + [args.steps]))],
                  test_iterations=[], ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                  ambient_lr=args.ambient_lr, normal_opacity_route=args.normal_opacity_route, reset_policy=args.reset_policy)
    pipe = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(ply_path))
    # load_ply为Stage2额外复制此属性，但原Stage1没有此优化器组；保留它会破坏原增密字典契约。
    del pc._albedo_dc_stage1
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('方向文件与G2-T04保存lighting SHA不符。')
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
    # 原G2-T04实参继承原30k XYZ LR，零Adam及零增密累计；不声称恢复不可用的状态。
    pc.training_setup(pipe)
    pc.update_learning_rate(clock_start)
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
    source_files = ('scripts/train_static_dl_foot_recovery.py', 'scene/static_dl_foot_recovery.py',
                    'scene/static_dl_normal_opacity.py',
                    'scene/static_dl_ambient.py',
                    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
                    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                    'scene/dataset_readers.py', 'utils/camera_utils.py')
    initial_files = (marker_path, continuation_path, ply_path, deform_path, lighting_path)
    audit = dict(actual_entrypoint='scripts/train_static_dl_foot_recovery.py',
                 reset_policy=args.reset_policy, clock_start=clock_start,
                 clock_shift_diagnostic=clock_start != args.init_iteration,
                 normal_opacity_route=args.normal_opacity_route,
                 args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 inherited_config=config,
                 initial_sources={str(p): digest(p) for p in initial_files},
                 new_sources={str(root / p): digest(root / p) for p in source_files},
                 training_inputs={str(p): digest(p) for p in [metadata_path, Path(config['static_dl_directions'])]
                                  + [Path(i.image_path_train_light) for i in infos]},
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
    write_json(output / 'foot_recovery_training.json', audit)
    manifest = dict(marker)
    manifest.update(args=config, method='G2-T04同起点足部恢复与15000reset单因素对照',
                    actual_entrypoint=audit['actual_entrypoint'], rasterizer_provenance=provenance,
                    time_dependent_color_optimized=False, test_images=0,
                    ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap, normal_opacity_route=args.normal_opacity_route, reset_policy=args.reset_policy)
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
                    pc.densify_and_prune_audited(config['densify_grad_threshold'], config['min_opacity'],
                        pc.camera_extent, 20 if iteration > config['opacity_reset_interval'] else None,
                        output, step, iteration)
                    visibility.invalidate()
                if iteration % config['opacity_reset_interval'] == 0:
                    reset_before = dict(opacity=opacity_summary(pc), adam=opacity_adam_summary(pc))
                    do_reset = not (args.reset_policy == 'skip_15000' and iteration == 15000)
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
                with (output / 'foot_recovery_metrics.jsonl').open('a') as stream:
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
                    iteration=iteration, optimizer_updates=step, reset_policy=args.reset_policy,
                    clock_shift_diagnostic=clock_start != args.init_iteration), optimizer_path)
                destination = output / f'deform/iteration_{iteration}/deform.pth'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(deform_path, destination)
                write_json(output / f'static_dl_lighting/iteration_{iteration}/lighting.json', dict(
                    iteration=iteration, mode='lambertian_visibility', directions_sha256=lights.sha256,
                    intensity_from_train_rgb=lights.log_intensity.exp().cpu().tolist(),
                    ambient_fraction=float(ambient), ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                    ambient_formula=audit['ambient_formula'], actual_entrypoint=audit['actual_entrypoint'],
                    normal_opacity_route=args.normal_opacity_route, reset_policy=args.reset_policy,
                    clock_start=clock_start, clock_shift_diagnostic=clock_start != args.init_iteration,
                    visibility_detached=True, temporal_geometry=False, temporal_color=False,
                    loss_model='censored', normal_depth_statistic='expected', initial_iteration=args.init_iteration,
                    continuation_step=step, optimizer_updates=step, smoke=args.smoke))
        camera.load2device('cpu')
    final_source_sha = {str(p): digest(p) for p in initial_files}
    if final_source_sha != audit['initial_sources']:
        raise AssertionError('G2-T04起点资产SHA在训练过程中变化。')
    write_json(output / 'foot_recovery_finished.json', dict(status='TRAINING_FINISHED', steps=args.steps,
        iteration=clock_start + args.steps, optimizer_updates=args.steps, point_count=len(pc.get_xyz),
        ambient_fraction=float(ambient), normal_opacity_route=args.normal_opacity_route,
        initial_sources_unchanged=True, reset_policy=args.reset_policy,
        opacity_resets_performed=reset_count, opacity_resets_skipped=skipped_reset_count,
        clock_start=clock_start, clock_shift_diagnostic=clock_start != args.init_iteration, goal_mae='尚未评测'))


if __name__ == '__main__':
    main()
