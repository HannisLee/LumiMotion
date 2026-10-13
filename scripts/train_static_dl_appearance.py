"""固定G2-T06几何的独立外观阶段；仅原135训练RGB、Alpha、相机及GT方向。"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms, getNerfppNorm
from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
from scene.static_dl_photometric import render_photometric as render_original
from scene.static_dl_ambient import render_photometric, geometry_losses
from scene.static_dl_censored import censored_photometric_loss, photometric_diagnostics
from scene.static_dl_shadow_split import require_native_backend
from scene.stage2_photometric import frame_key
from utils.camera_utils import cameraList_from_camInfos
from utils.sh_utils import RGB2SH


ENTRY = 'scripts/train_static_dl_appearance.py'
BASE_ITERATION = 16000
BASE_FILES = {
    'static_dl_training.json': '8e5d3c826a55b20a1510f3821e741463fea45624eda20fe092b3198f7c32c4c7',
    'reset_audit_training.json': '5180de352826dc791f3581e35238f99dbb7538fdeaacc2eed6480a4e201996d0',
    'reset_audit_finished.json': '8008a1185004668e4db77ad86f97d0726676cc737f70429977778a822bb81504',
    'point_cloud/iteration_16000/point_cloud.ply': '346cc983263fb5d6243b5b276e73a636e865de9be38662d9db6636c2bdd4fccc',
    'deform/iteration_16000/deform.pth': '343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0',
    'static_dl_lighting/iteration_16000/lighting.json': '85490b1dcce17306f4bf228a945e6950c89efbfa1fe071a1671523073ada3586',
}
SOURCE_FILES = (ENTRY, 'scene/static_dl_ambient.py', 'scene/static_dl_photometric.py',
                'scene/static_dl_censored.py', 'scene/static_dl_shadow_split.py',
                'scene/gaussian_model.py', 'gaussian_renderer/__init__.py',
                'scene/dataset_readers.py', 'utils/camera_utils.py',
                'scene/stage2_photometric.py', 'utils/general_utils.py', 'utils/sh_utils.py')
PARITY_MAPS = ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth',
               'surf_normal', 'rend_dist', 'radii', 'visibility_filter',
               'mean_visibility', 'light_intensity')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def frozen_hashes(pc):
    """覆盖全部叶张量，DC反照率是唯一例外；含feature和Stage1颜色副本。"""
    return {name: tensor_digest(value) for name, value in vars(pc).items()
            if isinstance(value, torch.Tensor) and value.is_leaf and name != '_albedo_dc'}


def assert_frozen(pc, frozen, count):
    if len(pc.get_xyz) != count:
        raise AssertionError('冻结点集的点数发生变化。')
    for name, value in frozen.items():
        current = getattr(pc, name)
        if current.requires_grad or current.grad is not None or not torch.equal(current, value):
            raise AssertionError('冻结张量变化或收到梯度：' + name)


def validate_source(init, source):
    """绝对marker锚定已验收T06，避免bootstrap的快照cwd改变来源含义。"""
    actual = {r: digest(init / r) for r in BASE_FILES}
    if actual != BASE_FILES:
        raise AssertionError('必须使用已验收G2-T06@16000原六资产，SHA不符。')
    marker = json.loads((init / 'static_dl_training.json').read_text())
    audit = json.loads((init / 'reset_audit_training.json').read_text())
    finished = json.loads((init / 'reset_audit_finished.json').read_text())
    lighting = json.loads((init / 'static_dl_lighting/iteration_16000/lighting.json').read_text())
    config = marker['args'].copy()
    if not Path(config['model_path']).is_absolute() or Path(config['model_path']).resolve() != init:
        raise AssertionError('起点必须为已验证marker中的绝对T06模型路径。')
    if Path(config['source_path']).resolve() != source:
        raise AssertionError('必须沿用T06原训练数据视图。')
    if (marker['actual_entrypoint'] != 'scripts/train_static_dl_reset_audit.py'
            or finished['iteration'] != BASE_ITERATION or finished['optimizer_updates'] != 2000
            or audit['smoke'] or lighting['smoke'] or lighting['ambient_fraction'] != 0.
            or config['static_dl_mode'] != 'lambertian_visibility'
            or config['loss_model'] != 'censored' or config['normal_depth_statistic'] != 'expected'
            or config['depth_ratio'] != 1 or config['white_background']
            or config['compute_cov3D_python']
            or config['post_reset_normal_route'] != 'detach_opacity'):
        raise AssertionError('来源必须为T06真实纯直射、censored、黑背景、原生路径终点。')
    return marker, audit, config, lighting, actual


def validate_native_provenance(provenance, marker):
    """T06训练的CUDA扩展与源码身份必须匹配，不能仅验证backend名称。"""
    expected = marker['rasterizer_provenance']
    keys = ('method', 'source_sha256', 'extension_sha256')
    if any(not expected.get(k) or provenance.get(k) != expected[k] for k in keys):
        raise AssertionError('实际原生光栅与T06的method/source/extension SHA不一致。')
    return dict(status='PASS', expected={k: expected[k] for k in keys},
                actual={k: provenance[k] for k in keys})


def compare_maps(first, second):
    checks = {}
    for key in PARITY_MAPS:
        a, b = first[key].detach(), second[key].detach()
        same_shape = a.shape == b.shape and a.dtype == b.dtype
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        error = float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if same_shape and a.numel() else 0.
        equal = same_shape and finite and torch.equal(a, b)
        checks[key] = dict(passed=bool(equal), max_abs=error, shape=list(a.shape), dtype=str(a.dtype))
    return dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'FAILED', checks=checks)


def save_forward(output, label, camera, pc, pipe, background, lights, visibility, iteration):
    """仅保存第一张原TRAIN相机的真实前向，供独立fresh逐map比较。"""
    before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    camera.load2device()
    with torch.no_grad():
        package = render_photometric(camera, pc, pipe, background, lights, visibility, iteration, 0.)
    path = output / 'appearance_forward' / (label + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: package[key].detach().cpu().numpy() for key in PARITY_MAPS})
    camera.load2device('cpu')
    if not torch.equal(before_cpu, torch.get_rng_state()) or any(
            not torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all())):
        raise AssertionError('记录真实前向不应改变Torch RNG。')
    write_json(path.with_suffix('.json'), dict(actual_entrypoint=ENTRY, iteration=iteration,
        camera=camera.image_name_train_light, camera_index=0, dataset_split='train',
        parameter_updates=0, maps=list(PARITY_MAPS), npz_sha256=digest(path),
        frozen_sha256=frozen_hashes(pc)))
    return path


def rng_state(camera_rng):
    n = np.random.get_state()
    return dict(python=random.getstate(), camera=camera_rng.getstate(),
                numpy=dict(kind=n[0], keys=n[1].tolist(), position=n[2], has_gauss=n[3], cached=n[4]),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state, camera_rng):
    random.setstate(state['python'])
    camera_rng.setstate(state['camera'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--init_model', type=Path, required=True)
    parser.add_argument('--init_iteration', type=int, default=BASE_ITERATION)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--source_path', type=Path, required=True)
    parser.add_argument('--loss_model', choices=['censored', 'mse'], required=True)
    parser.add_argument('--steps', type=int, required=True, help='本次实际新增optimizer更新次数。')
    parser.add_argument('--save_steps', type=int, nargs='+', required=True, help='相对于本次运行的保存步数。')
    parser.add_argument('--resume_state', type=Path, help='独立外观阶段保存的完整状态；禁止跨损失路线恢复。')
    parser.add_argument('--seed', type=int, default=1012)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit_only', action='store_true')
    args = parser.parse_args()
    output, init, source = args.model_path.resolve(), args.init_model.resolve(), args.source_path.resolve()
    if 'output' not in output.parts or output.name != 'model_mlp' or output.exists():
        parser.error('只允许output内尚不存在的独立model_mlp目录，拒绝覆盖。')
    if args.init_iteration != BASE_ITERATION or args.seed != 1012:
        parser.error('固定T06@16000来源与配对seed1012。')
    if args.steps <= 0 or any(not 0 < x <= args.steps for x in args.save_steps):
        parser.error('steps与save_steps必须为预算内正数。')
    if args.smoke and ('smoke_test' not in output.parts or args.steps > 110):
        parser.error('smoke必须在output/smoke_test内，最多110次更新。')
    if not args.smoke and 'smoke_test' in output.parts:
        parser.error('正式训练不能写入smoke_test。')
    provenance = require_native_backend()
    marker, base_audit, config, lighting, base_sources = validate_source(init, source)
    native_audit = validate_native_provenance(provenance, marker)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {r: digest(root / r) for r in SOURCE_FILES}
    # 所有被复用的历史科学依赖必须与T06登记内容一致；只新增独立入口。
    for name, expected in base_audit['new_sources'].items():
        relative = next((r for r in SOURCE_FILES if name.endswith('/' + r)), None)
        if relative is not None and source_hashes[relative] != expected:
            raise AssertionError('既有科学依赖已变：' + relative)
    deform_path = init / 'deform/iteration_16000/deform.pth'
    deform = torch.load(deform_path, map_location='cpu', weights_only=True)
    zero_heads = {}
    for head in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for suffix in ('weight', 'bias'):
            name = head + '.' + suffix
            zero_heads[name] = bool(torch.count_nonzero(deform[name]) == 0)
    if not all(zero_heads.values()):
        raise AssertionError('源形变几何头必须严格为零，不调用或训练共享形变网络。')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    camera_rng = random.Random(args.seed)
    state = None if args.resume_state is None else torch.load(args.resume_state.resolve(), map_location='cpu', weights_only=True)
    completed = 0 if state is None else state['appearance_updates']
    if state is not None:
        if (state['format_version'] != 1 or state['actual_entrypoint'] != ENTRY
                or state['loss_model'] != args.loss_model or state['base_sources'] != base_sources
                or state['source_hashes'] != source_hashes or state['seed'] != args.seed
                or state['iteration'] != BASE_ITERATION + completed or state['smoke'] != args.smoke):
            raise AssertionError('恢复状态来源、损失、源码、计数或smoke身份不一致。')
    config.update(model_path=str(output), source_path=str(source), iterations=BASE_ITERATION + completed + args.steps,
                  data_device='cpu', load2gpu_on_the_fly=True, loss_model=args.loss_model,
                  save_iterations=[BASE_ITERATION + completed + s for s in sorted(set(args.save_steps + [args.steps]))],
                  test_iterations=[], appearance_geometry_frozen=True, ambient_mode='zero', ambient_fraction=0.)
    pipe = Namespace(**config)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(init / 'point_cloud/iteration_16000/point_cloud.ply'))
    lights = DirectionOnlyLights(config['static_dl_directions'])
    expected_frames = {str(x).zfill(4) for x in range(1, 151) if x % 10}
    if lights.sha256 != lighting['directions_sha256'] or set(lights.directions) != expected_frames:
        raise AssertionError('方向表必须严格为原135训练方向，与源lighting SHA一致。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    metadata_path = source / 'transforms_train.json'
    metadata = json.loads(metadata_path.read_text())
    if len(metadata['frames']) != 135 or {str(x['source_frame']).zfill(4) for x in metadata['frames']} != expected_frames:
        raise AssertionError('只接受原完整135训练相机，不加载test或GT资产。')
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', config['train_light_folder'], False, no_bg=True)
    cameras = cameraList_from_camInfos(infos, 1., pipe)
    if len(cameras) != 135 or {frame_key(c.image_name_train_light) for c in cameras} != expected_frames:
        raise AssertionError('实际加载相机必须恰为原135训练帧。')
    training_inputs = {str(p): digest(p) for p in [metadata_path, Path(config['static_dl_directions'])]
                       + [Path(i.image_path_train_light) for i in infos]}
    if training_inputs != base_audit['training_inputs']:
        raise AssertionError('原训练RGB、相机和方向SHA发生变化。')
    pc.camera_extent = getNerfppNorm(infos)['radius']
    count = len(pc.get_xyz)
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    background = torch.zeros(3, device='cuda')
    cameras[0].load2device()
    # 冻结前后实际同一前向；同时核对a=0与旧纯直接光实现同义。
    with torch.no_grad():
        original = render_original(cameras[0], pc, pipe, background, lights, visibility, BASE_ITERATION)
        before = render_photometric(cameras[0], pc, pipe, background, lights, visibility, BASE_ITERATION, 0.)
    for name, value in vars(pc).items():
        if isinstance(value, torch.Tensor) and value.is_leaf:
            value.requires_grad_(name == '_albedo_dc')
            value.grad = None
    frozen = {name: value.detach().clone() for name, value in vars(pc).items()
              if isinstance(value, torch.Tensor) and value.is_leaf and name != '_albedo_dc'}
    geometry_hashes = frozen_hashes(pc)
    with torch.no_grad():
        after = render_photometric(cameras[0], pc, pipe, background, lights, visibility, BASE_ITERATION, 0.)
        constant_geometry, _ = geometry_losses(cameras[0], pc, pipe, background, cameras[0].gt_alpha_mask.cuda(), after)
    initial_audit = dict(status='PASS', baseline_to_zero_ambient=compare_maps(original, before),
                         before_to_frozen=compare_maps(before, after), zero_deform_heads=zero_heads,
                         requires_grad={n: bool(v.requires_grad) for n, v in vars(pc).items()
                                        if isinstance(v, torch.Tensor) and v.is_leaf},
                         frozen_sha256=geometry_hashes, parameter_updates=0,
                         constant_geometry_reference={k: float(v) for k, v in constant_geometry.items()})
    if (initial_audit['baseline_to_zero_ambient']['status'] != 'PASS'
            or initial_audit['before_to_frozen']['status'] != 'PASS'
            or {n for n, enabled in initial_audit['requires_grad'].items() if enabled} != {'_albedo_dc'}):
        initial_audit['status'] = 'FAILED'
    cameras[0].load2device('cpu')
    del original, before, after, constant_geometry
    optimizer = torch.optim.Adam([dict(params=[pc._albedo_dc], lr=config['albedo_lr'], name='albedo_dc'),
                                  dict(params=[lights.log_intensity], lr=config['static_dl_light_lr'], name='light')],
                                 lr=0., eps=1e-15)
    allowed_ids = {id(pc._albedo_dc), id(lights.log_intensity)}
    if {id(p) for g in optimizer.param_groups for p in g['params']} != allowed_ids:
        raise AssertionError('optimizer必须严格只含albedo_dc与light。')
    stack = []
    resume_audit = None
    if state is not None:
        if state['frozen_sha256'] != geometry_hashes or state['training_inputs'] != training_inputs or state['point_count'] != count:
            raise AssertionError('恢复状态冻结几何、点数或训练输入不一致。')
        with torch.no_grad():
            pc._albedo_dc.copy_(state['albedo_dc'].to('cuda'))
            lights.log_intensity.copy_(state['log_intensity'].to('cuda'))
        optimizer.load_state_dict(state['optimizer'])
        stack = list(state['camera_stack'])
        if len(set(stack)) != len(stack) or any(not 0 <= x < len(cameras) for x in stack):
            raise AssertionError('恢复相机余栈非法。')
        restore_rng(state['rng'], camera_rng)
        steps = {g['name']: float(optimizer.state[g['params'][0]]['step']) for g in optimizer.param_groups}
        if any(x != completed for x in steps.values()):
            raise AssertionError('Adam实际步数与累计外观更新预算不一致。')
        resume_audit = dict(status='PASS', appearance_updates=completed, optimizer_steps=steps,
                            camera_stack=stack, state_sha256=digest(args.resume_state.resolve()))
    output.mkdir(parents=True)
    (output / 'cfg_args').write_text(str(Namespace(**config)))
    audit = dict(actual_entrypoint=ENTRY, format_version=1,
                 args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 inherited_config=config, initial_sources={str(init / r): h for r, h in base_sources.items()},
                 base_sources=base_sources, new_sources={str(root / r): h for r, h in source_hashes.items()},
                 source_hashes=source_hashes, training_inputs=training_inputs,
                 initial_point_count=count, initial_frozen_sha256=geometry_hashes,
                 train_images=135, test_images_loaded=False, forbidden_gt_supervision=False,
                 shared_deformation_forward_called=False, geometry_heads_zero=True,
                 densification=False, pruning=False, opacity_reset=False, geometry_frozen=True,
                 optimizer_groups={g['name']: g['lr'] for g in optimizer.param_groups},
                 native_provenance_validation=native_audit,
                 optimizer_state_restored=state is not None, appearance_updates_before=completed,
                 update_budget=args.steps, base_iteration=BASE_ITERATION, loss_model=args.loss_model,
                 rgb_loss_denominator='原完整图像通道数', prediction_upper_clamp=False,
                 geometry_losses='冻结后为常量，仅作起点诊断；不加入外观optimizer',
                 camera_sampling='独立固定Python RNG，不放回遍历原135训练相机', smoke=args.smoke)
    manifest = dict(marker)
    manifest.update(args=config, method='固定T06全部几何的独立Lambertian外观训练', actual_entrypoint=ENTRY,
                    canonical_geometry_optimized=False, geometry_frozen=True, time_dependent_color_optimized=False,
                    temporal_geometry_optimized=False, test_images=0, rasterizer_provenance=provenance,
                    appearance_loss_model=args.loss_model)
    write_json(output / 'static_dl_training.json', manifest)
    write_json(output / 'appearance_training.json', audit)
    write_json(output / 'initial_checkpoint_gradient_audit.json', initial_audit)
    if resume_audit is not None:
        write_json(output / 'appearance_resume_audit.json', resume_audit)
    if initial_audit['status'] != 'PASS':
        raise RuntimeError('初始前向或冻结参数审计失败，保留现场。')
    save_forward(output, 'initial', cameras[0], pc, pipe, background, lights, visibility, BASE_ITERATION + completed)
    if args.audit_only:
        print('外观起点零更新前向与冻结审计PASS。', flush=True)
        return
    settings = Namespace(shadow_weight=config['static_dl_shadow_weight'],
                         highlight_weight=config['static_dl_highlight_weight'],
                         residual_threshold=config['static_dl_residual_threshold'])
    save_steps = set(args.save_steps + [args.steps])
    for step in range(1, args.steps + 1):
        total = completed + step
        iteration = BASE_ITERATION + total
        if not stack:
            stack = list(range(len(cameras)))
        camera_index = stack.pop(camera_rng.randrange(len(stack)))
        camera = cameras[camera_index]
        camera.load2device()
        package = render_photometric(camera, pc, pipe, background, lights, visibility, iteration, 0.)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        if args.loss_model == 'censored':
            loss, diagnostic = censored_photometric_loss(package['render'], target, alpha,
                settings.shadow_weight, settings.highlight_weight, settings.residual_threshold)
        else:
            loss, diagnostic = (package['render'] - target).square().mean(), {}
        if not torch.isfinite(loss):
            raise FloatingPointError('外观loss非有限。')
        should_log = step == 1 or step % 100 == 0 or step in save_steps
        details = photometric_diagnostics(package['render'], package['render_linear'], target, alpha, pc, settings) if should_log else {}
        with torch.no_grad():
            mse = (package['render'] - target).square().mean()
            clipped_mse = (package['render'].clamp(0, 1) - target).square().mean()
        loss.backward()
        gradients = {}
        for name, parameter in (('albedo_dc', pc._albedo_dc), ('light', lights.log_intensity)):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('外观允许路径梯度缺失/非有限：' + name)
            gradients[name] = dict(norm=float(parameter.grad.norm()), max_abs=float(parameter.grad.abs().max()))
        assert_frozen(pc, frozen, count)
        if step == 1:
            write_json(output / 'first_training_gradient_audit.json', dict(status='PASS', gradients=gradients,
                allowed_optimizer_groups=['albedo_dc', 'light'], frozen_gradients_absent=True,
                camera_index=camera_index, camera=camera.image_name_train_light,
                iteration=iteration, appearance_updates_before=completed, parameter_updates=0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lights.log_intensity.clamp_(-5, 5)
        assert_frozen(pc, frozen, count)
        camera.load2device('cpu')
        if should_log:
            record = dict(step=step, appearance_updates=total, iteration=iteration,
                          camera_index=camera_index, camera=camera.image_name_train_light,
                          loss_model=args.loss_model, photo=float(loss.detach()),
                          raw_mse=float(mse), clipped_mse=float(clipped_mse),
                          raw_psnr=-10 * math.log10(max(float(mse), 1e-30)),
                          clipped_psnr=-10 * math.log10(max(float(clipped_mse), 1e-30)),
                          points=count, geometry_frozen=True, mean_visibility=float(package['mean_visibility']),
                          light_post_update=lights.log_intensity.exp().detach().cpu().tolist(),
                          gradients=gradients, **{k: float(v) for k, v in diagnostic.items()}, **details)
            with (output / 'appearance_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step in save_steps:
            current_hashes = frozen_hashes(pc)
            if current_hashes != geometry_hashes:
                raise AssertionError('保存时冻结几何SHA不一致。')
            ply = output / f'point_cloud/iteration_{iteration}/point_cloud.ply'
            pc.save_ply(str(ply))
            destination = output / f'deform/iteration_{iteration}/deform.pth'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(deform_path, destination)
            light_path = output / f'static_dl_lighting/iteration_{iteration}/lighting.json'
            write_json(light_path, dict(iteration=iteration, mode='lambertian_visibility',
                directions_sha256=lights.sha256, intensity_from_train_rgb=lights.log_intensity.exp().detach().cpu().tolist(),
                ambient_fraction=0., ambient_mode='zero', ambient_cap=.1, actual_entrypoint=ENTRY,
                visibility_detached=True, temporal_geometry=False, temporal_color=False,
                loss_model=args.loss_model, normal_depth_statistic='expected',
                initial_iteration=BASE_ITERATION, appearance_updates=total, optimizer_updates=total,
                run_updates=step, geometry_frozen=True, smoke=args.smoke))
            forward_path = save_forward(output, 'iteration_' + str(iteration), cameras[0], pc,
                                        pipe, background, lights, visibility, iteration)
            state_path = output / f'appearance_state/iteration_{iteration}.pth'
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(format_version=1, actual_entrypoint=ENTRY, loss_model=args.loss_model,
                base_sources=base_sources, source_hashes=source_hashes, training_inputs=training_inputs,
                seed=args.seed, smoke=args.smoke, iteration=iteration, appearance_updates=total,
                albedo_dc=pc._albedo_dc.detach().cpu(), log_intensity=lights.log_intensity.detach().cpu(),
                optimizer=optimizer.state_dict(), rng=rng_state(camera_rng), camera_stack=stack,
                frozen_sha256=geometry_hashes, point_count=count,
                assets={str(p.relative_to(output)): digest(p) for p in (ply, destination, light_path, forward_path)}), state_path)
            write_json(output / f'appearance_checkpoint/iteration_{iteration}.json', dict(
                status='SAVED', iteration=iteration, appearance_updates=total, run_updates=step,
                frozen_sha256=current_hashes, point_count=count, state_sha256=digest(state_path),
                assets={str(p.relative_to(output)): digest(p) for p in (ply, destination, light_path, forward_path)}))
        del package, target, alpha, loss
    if {r: digest(init / r) for r in BASE_FILES} != base_sources:
        raise AssertionError('T06源资产在运行中变化。')
    if {r: digest(root / r) for r in SOURCE_FILES} != source_hashes:
        raise AssertionError('科学源在运行中变化。')
    if {p: digest(p) for p in training_inputs} != training_inputs:
        raise AssertionError('原训练输入在运行中变化。')
    final_hashes = frozen_hashes(pc)
    write_json(output / 'appearance_finished.json', dict(status='TRAINING_FINISHED',
        actual_entrypoint=ENTRY, steps=args.steps, optimizer_updates=args.steps,
        appearance_updates=completed + args.steps, iteration=BASE_ITERATION + completed + args.steps,
        point_count=count, loss_model=args.loss_model, initial_sources_unchanged=True,
        training_inputs_unchanged=True, source_files_unchanged=True,
        initial_frozen_sha256=geometry_hashes, final_frozen_sha256=final_hashes,
        geometry_unchanged=final_hashes == geometry_hashes, goal_mae='尚未评测', smoke=args.smoke))


if __name__ == '__main__':
    main()
