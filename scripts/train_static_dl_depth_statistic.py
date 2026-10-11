"""独立法线一致性深度统计量对照；只改变训练geometry的median或expected深度。"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import torch

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms, getNerfppNorm
from scene.static_dl_photometric import DirectionOnlyLights
from scene.static_dl_censored import loss_for_mode, photometric_diagnostics
from scene.static_dl_shadow_split import render_split_photometric, require_native_backend
from scene.static_dl_depth_statistic import geometry_regularization
from utils.camera_utils import cameraList_from_camInfos
from utils.sh_utils import RGB2SH


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--init_model', type=Path, required=True)
    parser.add_argument('--init_iteration', type=int, required=True)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--source_path', type=Path, required=True)
    parser.add_argument('--route', choices=['control'], required=True)
    parser.add_argument('--loss_model', choices=['censored'], required=True)
    parser.add_argument('--normal_depth_statistic', choices=['median', 'expected'], required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--save_steps', type=int, nargs='+', required=True)
    for name in ('xyz_lr', 'rotation_lr', 'albedo_lr', 'light_lr', 'normal_weight',
                 'dist_weight', 'alpha_weight', 'shadow_bias', 'shadow_weight',
                 'highlight_weight', 'residual_threshold'):
        parser.add_argument('--'+name, type=float, required=True)
    parser.add_argument('--seed', type=int, default=1011)
    parser.add_argument('--audit_only', action='store_true', help='仅在第一张原训练相机验证梯度，不优化参数。')
    args = parser.parse_args()
    if args.steps <= 0 or args.init_iteration <= 0:
        parser.error('迭代数必须为正。')
    for name in ('xyz_lr', 'rotation_lr', 'albedo_lr', 'light_lr', 'normal_weight', 'dist_weight',
                 'alpha_weight', 'shadow_bias', 'shadow_weight', 'highlight_weight', 'residual_threshold'):
        if getattr(args, name) < 0:
            parser.error(name+' 不得为负。')
    output = args.model_path.resolve()
    if 'output' not in output.parts or not output.name.endswith('_mlp'):
        parser.error('必须显式给出 output 内独立 model_mlp 目录。')
    if output.exists():
        parser.error('输出模型目录已存在，拒绝覆盖。')
    provenance = require_native_backend()
    init = args.init_model.resolve()
    marker_path = init/'static_dl_training.json'
    marker = json.loads(marker_path.read_text())
    config = marker['args'].copy()
    if config['static_dl_mode'] not in {'lambertian', 'lambertian_visibility'}:
        parser.error('本最小对照仅接已有物理 checkpoint，不加入材质重初始化这一额外因素。')
    if Path(config['source_path']).resolve() != args.source_path.resolve():
        parser.error('Stage2 必须沿用共同 checkpoint 的原训练数据源。')
    if config.get('depth_ratio') != 1.:
        parser.error('本单因素对照固定原RGB前向depth_ratio=1；统计量只作用于独立几何loss。')
    if config.get('compute_cov3D_python', False):
        parser.error('只支持原生 scale/rotation 光栅路径。')
    lighting_path = init/f'static_dl_lighting/iteration_{args.init_iteration}/lighting.json'
    lighting = json.loads(lighting_path.read_text())
    ply_path = init/f'point_cloud/iteration_{args.init_iteration}/point_cloud.ply'
    deform_path = init/f'deform/iteration_{args.init_iteration}/deform.pth'
    # 读取原零头验证，后续完全不调用共享形变网络，也不更新其参数。
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for suffix in ('weight', 'bias'):
            value = state[name+'.'+suffix]
            if torch.count_nonzero(value).item() != 0:
                raise ValueError('起点不是严格静态零几何头：'+name+'.'+suffix)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config.update(model_path=str(output), source_path=str(args.source_path.resolve()),
                  static_dl_mode='lambertian_visibility', static_dl_physical_start=0,
                  static_dl_visibility_start=0, static_dl_bvh_interval=1,
                  static_dl_shadow_bias=args.shadow_bias, iterations=args.init_iteration+args.steps,
                  load2gpu_on_the_fly=True, data_device='cpu')
    dataset = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config.get('hyper_dim', 1))
    pc.load_ply(str(ply_path))
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if lights.sha256 != lighting['directions_sha256']:
        raise ValueError('方向表与初始化光强记录 SHA 不一致。')
    expected = {str(n).zfill(4) for n in range(1, 151) if n % 10}
    if set(lights.directions) != expected:
        raise ValueError('方向必须严格来自原 135 个训练帧。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    infos = readCamerasFromTransforms(config['source_path'], 'transforms_train.json',
                                     config['train_light_folder'], config['white_background'], no_bg=True)
    cameras = cameraList_from_camInfos(infos, 1., dataset)
    if len(cameras) != 135:
        raise ValueError('必须保留完整 135 训练相机。')
    pc.camera_extent = getNerfppNorm(infos)['radius']
    for camera in cameras:
        lights.for_camera(camera)
    # 不沿用任何旧优化器动量；两组从同一参数与零动量开始。
    allowed = {'_xyz', '_rotation', '_albedo_dc'}
    for name, value in vars(pc).items():
        if isinstance(value, torch.Tensor) and value.is_leaf:
            value.requires_grad_(name in allowed)
    frozen = {k:getattr(pc, k).detach().clone() for k in ('_opacity', '_scaling')}
    count = len(pc.get_xyz)
    output.mkdir(parents=True)
    (output/'cfg_args').write_text(str(Namespace(**config)))
    manifest = dict(marker)
    manifest.update(args=config, method='独立Stage2法线一致性深度统计量单因素对照',
                    loss_model=args.loss_model, actual_entrypoint='scripts/train_static_dl_depth_statistic.py',
                    normal_depth_statistic=args.normal_depth_statistic,
                    rasterizer_provenance=provenance, time_dependent_color_optimized=False)
    write_json(output/'static_dl_training.json', manifest)
    audit = {'args':{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
             'initial_sources':{str(p):digest(p) for p in (marker_path, ply_path, deform_path, lighting_path)},
             'new_sources':{str(p):digest(p) for p in (Path(__file__), Path(__file__).resolve().parents[1]/'scene/static_dl_shadow_split.py',
                Path(__file__).resolve().parents[1]/'scene/static_dl_depth_statistic_audit.py',
                Path(__file__).resolve().parents[1]/'scene/static_dl_depth_statistic.py',
                Path(__file__).resolve().parents[1]/'scene/static_dl_censored.py')},
             'point_count':count, 'densification':False, 'opacity_scale_frozen':True,
             'shared_deformation_forward_called':False, 'test_images_loaded':False,
             'rgb_loss_count_per_step':1, 'xyz_lr_absolute_world_units':True,
             'actual_entrypoint':'scripts/train_static_dl_depth_statistic.py', 'loss_model':args.loss_model,
             'normal_depth_statistic':args.normal_depth_statistic, 'photo_depth_ratio':1.,
             'saturation_rule':'target == 1.0 and original_alpha > 0.5, channelwise',
             'nominal_srgb_lower_bound':1., 'prediction_upper_clamp':False,
             'robust_weights_from_raw_residual':True, 'loss_denominator':'full_image_channel_count'}
    write_json(output/'depth_statistic_training.json', audit)
    write_json(output/'censored_training.json', audit)
    write_json(output/'shadow_split_training.json', audit)  # 兼容固定几何审计，真实入口仍由独立marker声明。
    shutil.copyfile(ply_path, output/'input.ply')
    background = torch.tensor([1.,1.,1.] if config['white_background'] else [0.,0.,0.], device='cuda')
    from scene.static_dl_depth_statistic_audit import audit_checkpoint_routes
    cameras[0].load2device()
    checkpoint_audit = audit_checkpoint_routes(cameras[0], pc, dataset, background, lights, args)
    cameras[0].load2device('cpu')
    write_json(output/'initial_checkpoint_gradient_audit.json', checkpoint_audit)
    if checkpoint_audit['status'] != 'PASS':
        raise RuntimeError('真实checkpoint梯度审计未通过；禁止继续训练。')
    if args.audit_only:
        print('真实checkpoint梯度审计 PASS；audit_only 未建立优化器或更新模型。', flush=True)
        return
    optimizer = torch.optim.Adam([
        {'params':[pc._xyz], 'lr':args.xyz_lr, 'name':'xyz'},
        {'params':[pc._rotation], 'lr':args.rotation_lr, 'name':'rotation'},
        {'params':[pc._albedo_dc], 'lr':args.albedo_lr, 'name':'albedo'},
        {'params':[lights.log_intensity], 'lr':args.light_lr, 'name':'light'}], lr=0., eps=1e-15)
    stack = []
    save_steps = set(args.save_steps+[args.steps])
    for step in range(1, args.steps+1):
        if not stack:
            stack = list(range(len(cameras)))
        index = stack.pop(random.randrange(len(stack)))
        camera = cameras[index]
        camera.load2device()
        directions, intensity = lights.for_camera(camera)
        package = render_split_photometric(camera, pc, dataset, background, directions, intensity,
                                            args.shadow_bias, route=args.route)
        target = camera.original_image_train_light.cuda()
        alpha = camera.gt_alpha_mask.cuda()
        if config['white_background']:
            target = alpha*target+(1-alpha)*background[:, None, None]
        photo, diagnostic = loss_for_mode(args.loss_model)(package['render'], target, alpha,
            args.shadow_weight, args.highlight_weight, args.residual_threshold)
        geometry, _ = geometry_regularization(camera, pc, dataset, background, alpha, args.normal_depth_statistic)
        loss = photo+args.alpha_weight*geometry['alpha']+args.normal_weight*geometry['normal']+args.dist_weight*geometry['distortion']
        if not torch.isfinite(loss):
            raise FloatingPointError('Stage2 loss 非有限。')
        should_log = step == 1 or step % 100 == 0 or step in save_steps
        extra_diagnostic = photometric_diagnostics(package['render'], package['render_linear'],
            target, alpha, pc, args) if should_log else {}
        loss.backward()
        for name, parameter in [('xyz',pc._xyz), ('rotation',pc._rotation),
                                 ('material',pc._albedo_dc), ('light',lights.log_intensity)]:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('允许路径梯度缺失或非有限：'+name)
        if len(pc.get_xyz) != count or any(not torch.equal(getattr(pc,k),v) for k,v in frozen.items()):
            raise RuntimeError('冻结的点数、opacity 或 scale 发生变化。')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lights.log_intensity.clamp_(-5, 5)
        # 不调用 add_densification_stats、densify/prune、reset_opacity 或任何共享网络。
        if should_log:
            record = dict(step=step, iteration=args.init_iteration+step, route=args.route, loss_model=args.loss_model, normal_depth_statistic=args.normal_depth_statistic,
                          camera=camera.image_name_train_light, loss=float(loss.detach()), photo=float(photo.detach()),
                          **{k:float(v.detach()) for k,v in geometry.items()},
                          mean_visibility=float(package['mean_visibility']), points=count,
                          light=lights.log_intensity.exp().detach().cpu().tolist(),
                          **{k:float(v) for k,v in diagnostic.items()}, **extra_diagnostic)
            with (output/'depth_statistic_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step in save_steps:
            iteration = args.init_iteration+step
            pc.save_ply(str(output/f'point_cloud/iteration_{iteration}/point_cloud.ply'))
            destination = output/f'deform/iteration_{iteration}/deform.pth'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(deform_path, destination)
            write_json(output/f'static_dl_lighting/iteration_{iteration}/lighting.json',
                {'iteration':iteration, 'mode':'lambertian_visibility', 'directions_sha256':lights.sha256,
                 'intensity_from_train_rgb':lights.log_intensity.exp().detach().cpu().tolist(),
                 'visibility_detached':args.route=='control', 'temporal_geometry':False, 'temporal_color':False,
                 'shadow_gradient_route':args.route, 'loss_model':args.loss_model,
                 'normal_depth_statistic':args.normal_depth_statistic, 'stage2_step':step, 'initial_iteration':args.init_iteration})
        camera.load2device('cpu')
    write_json(output/'depth_statistic_finished.json', {'status':'TRAINING_FINISHED', 'steps':args.steps,
        'iteration':args.init_iteration+args.steps, 'goal_mae':'尚未评测', 'point_count':count})


if __name__ == '__main__':
    main()
