"""从G2-T01@12000续训；只改变normal一致性对opacity的梯度路线。"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scene.gaussian_model import GaussianModel
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
    parser.add_argument('--normal_opacity_route', choices=['full', 'detach_opacity'], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=1012)
    parser.add_argument('--audit_only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='独立smoke_test内20/最多150步，沿用真实全局步与参数。')
    args = parser.parse_args()
    # 本轮不再训练间接底项；以下为显式记录的共同固定配置，不暴露额外CLI变量。
    args.ambient_mode, args.ambient_cap, args.ambient_lr = 'zero', .1, .001
    output, init, source = args.model_path.resolve(), args.init_model.resolve(), args.source_path.resolve()
    if 'output' not in output.parts or output.name != 'model_mlp' or output.exists():
        parser.error('只允许output内尚不存在的独立model_mlp目录。')
    if args.smoke:
        if 'smoke_test' not in output.parts or not 0 < args.steps <= 150:
            parser.error('smoke只允许独立smoke_test、1至150步。')
    elif args.steps != 2000:
        parser.error('本轮正式配对固定新增2000步。')
    if args.init_iteration != 12000:
        parser.error('本轮只从新最佳G2-T01@12000初始化。')
    if any(not 0 < x <= args.steps for x in args.save_steps):
        parser.error('save_steps必须是预算内的正相对步数。')
    provenance = require_native_backend()
    marker_path = init / 'static_dl_training.json'
    continuation_path = init / 'ambient_training.json'
    marker = json.loads(marker_path.read_text())
    continuation = json.loads(continuation_path.read_text())
    config = marker['args'].copy()
    if (marker['actual_entrypoint'] != 'scripts/train_static_dl_ambient.py'
            or continuation['ambient_mode'] != 'zero' or continuation['smoke']
            or continuation['args']['init_iteration'] != 10000 or continuation['args']['steps'] != 2000
            or config['static_dl_mode'] != 'lambertian_visibility'
            or config['loss_model'] != 'censored' or config['normal_depth_statistic'] != 'expected'
            or config['depth_ratio'] != 1 or config['white_background']
            or config['compute_cov3D_python']):
        parser.error('只接受G2-T01真实2k终点、a0/censored/expected/黑背景/原生scale-rotation起点。')
    if Path(config['source_path']).resolve() != source:
        parser.error('必须沿用G2-T01原训练数据视图。')
    ply_path = init / f'point_cloud/iteration_{args.init_iteration}/point_cloud.ply'
    deform_path = init / f'deform/iteration_{args.init_iteration}/deform.pth'
    lighting_path = init / f'static_dl_lighting/iteration_{args.init_iteration}/lighting.json'
    lighting = json.loads(lighting_path.read_text())
    if lighting['ambient_fraction'] != 0. or lighting['ambient_mode'] != 'zero' or lighting['optimizer_updates'] != 2000:
        parser.error('G2-T01源lighting必须保存a0与2000次真实更新。')
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for suffix in ('weight', 'bias'):
            if torch.count_nonzero(state[name + '.' + suffix]):
                raise AssertionError('G2-T01源不是严格静态零几何头。')
    # 三套RNG均固定；相机使用独立Python RNG，增密点数分叉不会改变相机顺序。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    camera_rng = random.Random(args.seed)
    config.update(model_path=str(output), source_path=str(source),
                  iterations=args.init_iteration + args.steps, data_device='cpu',
                  load2gpu_on_the_fly=True, save_iterations=[args.init_iteration + x for x in sorted(set(args.save_steps + [args.steps]))],
                  test_iterations=[], ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                  ambient_lr=args.ambient_lr, normal_opacity_route=args.normal_opacity_route)
    pipe = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(ply_path))
    # load_ply为Stage2额外复制此属性，但原Stage1没有此优化器组；保留它会破坏原增密字典契约。
    del pc._albedo_dc_stage1
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('方向文件与G2-T01保存lighting SHA不符。')
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
    # 原G2-T01实参继承原30k XYZ LR，零Adam及零增密累计；不声称恢复不可用的状态。
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
    source_files = ('scripts/train_static_dl_normal_opacity.py', 'scene/static_dl_normal_opacity.py',
                    'scene/static_dl_ambient.py',
                    'scene/static_dl_photometric.py', 'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
                    'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                    'scene/dataset_readers.py', 'utils/camera_utils.py')
    initial_files = (marker_path, continuation_path, ply_path, deform_path, lighting_path)
    audit = dict(actual_entrypoint='scripts/train_static_dl_normal_opacity.py',
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
                 densification=True, opacity_resets_in_run=0, source_boundary_reset_replayed=False,
                 normal_opacity_detached=args.normal_opacity_route == 'detach_opacity',
                 density_order='先反传和统计；先优化器更新再原增密；保存前参数已更新；终点不执行无后续优化的增密/reset',
                 camera_sampling='独立固定Python RNG、不放回遍历全部135相机',
                 update_budget=args.steps, smoke=args.smoke)
    write_json(output / 'normal_opacity_training.json', audit)
    manifest = dict(marker)
    manifest.update(args=config, method='G2-T01参数续训与normal-opacity单因素梯度对照',
                    actual_entrypoint=audit['actual_entrypoint'], rasterizer_provenance=provenance,
                    time_dependent_color_optimized=False, test_images=0,
                    ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap, normal_opacity_route=args.normal_opacity_route)
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
    stack = []
    save_steps = set(args.save_steps + [args.steps])
    settings = Namespace(shadow_weight=config['static_dl_shadow_weight'], highlight_weight=config['static_dl_highlight_weight'],
                         residual_threshold=config['static_dl_residual_threshold'])
    for step in range(1, args.steps + 1):
        iteration = args.init_iteration + step
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
                    before_count = len(pc.get_xyz)
                    pc.densify_and_prune(config['densify_grad_threshold'], config['min_opacity'],
                                         pc.camera_extent, 20 if iteration > config['opacity_reset_interval'] else None)
                    visibility.invalidate()
                    with (output / 'topology_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(step=step, iteration=iteration, before=before_count,
                                                     after=len(pc.get_xyz)), ensure_ascii=False) + '\n')
                if iteration % config['opacity_reset_interval'] == 0:
                    pc.reset_opacity()
                    visibility.invalidate()
            if should_log:
                record = dict(step=step, iteration=iteration, camera=camera.image_name_train_light,
                    normal_opacity_route=args.normal_opacity_route,
                    loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geometry.items()},
                    **{k: float(v) for k, v in diagnostic.items()}, **details,
                    mean_visibility=float(package['mean_visibility']), points=len(pc.get_xyz),
                    ambient_pre_update=float(package['ambient_fraction']), ambient_post_update=float(ambient),
                    light_post_update=lights.log_intensity.exp().cpu().tolist(), xyz_lr_used=used_xyz_lr)
                with (output / 'normal_opacity_metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if step in save_steps:
                pc.save_ply(str(output / f'point_cloud/iteration_{iteration}/point_cloud.ply'))
                destination = output / f'deform/iteration_{iteration}/deform.pth'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(deform_path, destination)
                write_json(output / f'static_dl_lighting/iteration_{iteration}/lighting.json', dict(
                    iteration=iteration, mode='lambertian_visibility', directions_sha256=lights.sha256,
                    intensity_from_train_rgb=lights.log_intensity.exp().cpu().tolist(),
                    ambient_fraction=float(ambient), ambient_mode=args.ambient_mode, ambient_cap=args.ambient_cap,
                    ambient_formula=audit['ambient_formula'], actual_entrypoint=audit['actual_entrypoint'],
                    normal_opacity_route=args.normal_opacity_route,
                    visibility_detached=True, temporal_geometry=False, temporal_color=False,
                    loss_model='censored', normal_depth_statistic='expected', initial_iteration=args.init_iteration,
                    continuation_step=step, optimizer_updates=step, smoke=args.smoke))
        camera.load2device('cpu')
    final_source_sha = {str(p): digest(p) for p in initial_files}
    if final_source_sha != audit['initial_sources']:
        raise AssertionError('G2-T01起点资产SHA在训练过程中变化。')
    write_json(output / 'normal_opacity_finished.json', dict(status='TRAINING_FINISHED', steps=args.steps,
        iteration=args.init_iteration + args.steps, optimizer_updates=args.steps, point_count=len(pc.get_xyz),
        ambient_fraction=float(ambient), normal_opacity_route=args.normal_opacity_route,
        initial_sources_unchanged=True, goal_mae='尚未评测'))


if __name__ == '__main__':
    main()
