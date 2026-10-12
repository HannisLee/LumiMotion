#!/usr/bin/env python3
"""固定G2-T06原135TRAIN的增密屏幕支路抵消诊断；无优化器、无拓扑、无参数更新。"""
import argparse
from argparse import Namespace
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / 'LIHAN/scripts/bootstrap_1011.py'
THRESHOLD = .0002
SEED = 1012
TRACER_SHA = '5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e'
OUTPUT = ROOT / 'output/1012-14-LH-static-jumpingjacks_DL_V2-F-增密梯度抵消零更新诊断'
INITIAL_SHA = {
    'static_dl_training.json': '8e5d3c826a55b20a1510f3821e741463fea45624eda20fe092b3198f7c32c4c7',
    'reset_audit_training.json': '5180de352826dc791f3581e35238f99dbb7538fdeaacc2eed6480a4e201996d0',
    'reset_audit_finished.json': '8008a1185004668e4db77ad86f97d0726676cc737f70429977778a822bb81504',
    'point_cloud/iteration_16000/point_cloud.ply': '346cc983263fb5d6243b5b276e73a636e865de9be38662d9db6636c2bdd4fccc',
    'deform/iteration_16000/deform.pth': '343c279787bb2cf08d8d5527e80d02bec5dc510a2f01fd00d04431f654383ef0',
    'static_dl_lighting/iteration_16000/lighting.json': '85490b1dcce17306f4bf228a945e6950c89efbfa1fe071a1671523073ada3586',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def camera_order(count, seed):
    """逐式重现新Adam续训的独立random.Random不放回采样。"""
    rng = random.Random(seed)
    stack = list(range(count))
    return [stack.pop(rng.randrange(len(stack))) for _ in range(count)]


def floor_scores(other, normal):
    import torch
    with torch.no_grad():
        total = torch.norm((other + normal)[:, :2], dim=-1)
        non_normal = torch.norm(other[:, :2], dim=-1)
        normal_norm = torch.norm(normal[:, :2], dim=-1)
        floor = torch.maximum(total, non_normal)
    return total, floor, non_normal, normal_norm


def dispatch_bootstrap(args):
    """不改旧快照：临时finder只定位本helper，原bootstrap仍负责原生装载及科学路径。"""
    entry = '_lumimotion_goal3_densification_floor_profile'
    helper = str(Path(__file__).resolve())
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == entry:
                return importlib.util.spec_from_file_location(fullname, helper)
            return None
    sys.meta_path.insert(0, Finder())
    original = [sys.executable, helper] + sys.argv[1:]
    os.environ['LUMI_FLOOR_PROFILE_COMMAND_JSON'] = json.dumps(original)
    sys.argv = [str(BOOTSTRAP), str(args.snapshot), entry, '--bootstrapped'] + sys.argv[1:]
    runpy.run_path(str(BOOTSTRAP), run_name='__main__')


def initialize(args):
    # 已由本脚本分发到真实bootstrap_1011.py，科学函数仅从原T06快照导入。
    import diff_surfel_rasterization as native
    if Path.cwd().resolve() != args.snapshot:
        raise AssertionError('实际bootstrap cwd不是原训练快照。')
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms, getNerfppNorm
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    from scene.static_dl_normal_opacity import geometry_losses
    from scene.static_dl_censored import censored_photometric_loss
    from scene.stage2_photometric import frame_key
    from utils.camera_utils import cameraList_from_camInfos

    model = args.model
    if any(digest(model / path) != sha for path, sha in INITIAL_SHA.items()):
        raise AssertionError('冻结G2-T06六资产SHA不符。')
    training_path, static_path = model / 'reset_audit_training.json', model / 'static_dl_training.json'
    training, static = json.loads(training_path.read_text()), json.loads(static_path.read_text())
    config = static['args'].copy()
    lighting_path = model / 'static_dl_lighting/iteration_16000/lighting.json'
    lighting = json.loads(lighting_path.read_text())
    command_path = model.parent / 'commands/train.json'
    command = json.loads(command_path.read_text())['argv']
    if (Path(command[2]).resolve() != args.snapshot or training['smoke']
            or training['post_reset_normal_route'] != 'detach_opacity'
            or lighting['normal_opacity_route'] != 'detach_opacity' or lighting['iteration'] != 16000
            or lighting['ambient_fraction'] != 0 or config['depth_ratio'] != 1
            or config['white_background'] or training['actual_entrypoint'] != 'scripts/train_static_dl_reset_audit.py'):
        raise AssertionError('只接受T06真实16000、训练retry1快照和原a0/detach/median-photo设置。')
    for key in ('method', 'source_sha256', 'extension_sha256'):
        if native.provenance[key] != static['rasterizer_provenance'][key]:
            raise AssertionError('实际native与T06训练不同：' + key)
    sources = {str(Path(__file__).resolve()): digest(__file__), str(args.snapshot / 'snapshot_manifest.json'): digest(args.snapshot / 'snapshot_manifest.json'), str(BOOTSTRAP): digest(BOOTSTRAP)}
    for path, sha in training['new_sources'].items():
        p = Path(path).resolve()
        if not p.is_relative_to(args.snapshot) or digest(p) != sha:
            raise AssertionError('T06科学快照文件内容或路径变化。')
        sources[str(p)] = sha
    for func in (render_photometric, geometry_losses, censored_photometric_loss, readCamerasFromTransforms, cameraList_from_camInfos):
        p = Path(sys.modules[func.__module__].__file__).resolve()
        if not p.is_relative_to(args.snapshot) or str(p) not in sources:
            raise AssertionError('实际科学函数未从固定T06快照载入。')
    if config['densify_grad_threshold'] != THRESHOLD or config['densification_interval'] != 100 or config['static_dl_bvh_interval'] != 1:
        raise AssertionError('增密阈值/间隔/BVH配置不是预注册值。')
    if config['static_dl_normal_weight'] != .1 or config['lambda_alpha_loss'] != .1 or config['lambda_dist'] != 1000:
        raise AssertionError('实际三项几何权重变化。')
    source = Path(config['source_path'])
    metadata_path = source / 'transforms_train.json'
    metadata = json.loads(metadata_path.read_text())
    expected_frames = {str(i).zfill(4) for i in range(1, 151) if i % 10}
    if len(metadata['frames']) != 135 or {str(f['source_frame']).zfill(4) for f in metadata['frames']} != expected_frames:
        raise AssertionError('必须只加载原135 TRAIN帧。')
    config.update(data_device='cpu', load2gpu_on_the_fly=True)
    pipe = Namespace(**config)
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', config['train_light_folder'], False, no_bg=True)
    cameras = cameraList_from_camInfos(infos, 1., pipe)
    if len(cameras) != 135 or {frame_key(c.image_name_train_light) for c in cameras} != expected_frames:
        raise AssertionError('实际加载相机不是原135 TRAIN。')
    extent = float(getNerfppNorm(infos)['radius'])
    if abs(extent - training['camera_extent']) > 1e-7:
        raise AssertionError('仅训练相机重算extent与保存配置不一致。')
    ply_path, deform_path = model / 'point_cloud/iteration_16000/point_cloud.ply', model / 'deform/iteration_16000/deform.pth'
    state = torch.load(deform_path, map_location='cpu', weights_only=True)
    if any(torch.count_nonzero(value) for key, value in state.items() if any(h in key for h in ('gaussian_warp.', 'gaussian_rotation.', 'gaussian_scaling.'))):
        raise AssertionError('静态几何头不是零。')
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(ply_path))
    del pc._albedo_dc_stage1
    # 保持真实训练可导参数；不调用training_setup、不构建任何优化器。
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if set(lights.directions) != expected_frames or lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('方向只允许原135训练帧。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    files = [training_path, static_path, model / 'reset_audit_finished.json', lighting_path, ply_path, deform_path, command_path, metadata_path,
             Path(config['static_dl_directions'])] + [Path(info.image_path_train_light) for info in infos]
    inputs = {str(p): digest(p) for p in files}
    # 与真实训练记录的原输入逐一匹配；只读135图、原相机和纯方向。
    for path, sha in training['training_inputs'].items():
        if inputs.get(path) != sha:
            raise AssertionError('训练输入与T06来源不符：' + path)
    return pc, lights, cameras, pipe, StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval']), extent, inputs, sources, native.provenance


def tensor_digest(values):
    sha = hashlib.sha256()
    for name, value in sorted(values.items()):
        sha.update(name.encode())
        sha.update(str(tuple(value.shape)).encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def rng_snapshot():
    import numpy as np
    import torch
    return dict(python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                torch=digest_tensor_bytes(torch.get_rng_state()),
                cuda=digest_tensor_bytes(torch.cuda.get_rng_state()))


def digest_tensor_bytes(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def runtime_evidence(provenance):
    import subprocess
    import torch
    maps = Path('/proc/self/maps').read_text()
    modules = sorted({line.split()[-1] for line in maps.splitlines()
                      if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line)
                      and line.split()[-1].endswith('.so')})
    hashes = {p: digest(p) for p in modules}
    if not any('surfel_tracer/' in p and h == TRACER_SHA for p, h in hashes.items()):
        raise AssertionError('实际tracer与已审计G2-T06来源不同或未加载。')
    if not any('lumimotion_pbr_cuda' in p and h == provenance['extension_sha256'] for p, h in hashes.items()):
        raise AssertionError('实际native映射与冻结训练来源不同。')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = [line.strip() for line in result.stdout.splitlines()
            if line.split(',')[0].strip() == str(os.getpid())]
    if not rows:
        raise AssertionError('活跃进程未在GPU查询中找到。')
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if selected.startswith('GPU-') and any(row.split(',')[1].strip() != selected for row in rows):
        raise AssertionError('实际GPU UUID与显式选择不同。')
    return dict(pid=os.getpid(), proc_cmdline=Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode(),
                runtime_modules=hashes, native_provenance=provenance, gpu_rows=rows,
                torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                cuda_device_name=torch.cuda.get_device_name(0), python=sys.executable,
                environment={k: os.environ.get(k) for k in ('CONDA_PREFIX', 'CUDA_VISIBLE_DEVICES',
                    'CUDA_DEVICE_ORDER', 'CUDA_HOME', 'CXX', 'CC', 'OMP_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCH_CUDA_ARCH_LIST')})


def summarize_window(output, label, accum, support, raised_support, max_scale, opacity, extent, percent_dense, frames):
    import numpy as np
    import torch
    arrays = {k + '_sum': v.detach().cpu().numpy().copy() for k, v in accum.items()}
    counts = support.detach().cpu().numpy().copy()
    raised = raised_support.detach().cpu().numpy().copy()
    arrays.update(visible_support=counts, floor_raised_support=raised,
                  max_scale=max_scale, opacity=opacity)
    # 在Torch float32中按原累计/分母计算，避免换成float64改变临界选择。
    scores = {}
    for key, total in accum.items():
        score = total / support
        score = torch.where(support > 0, score, torch.zeros_like(score))
        scores[key] = score.cpu().numpy()
        arrays[key + '_score'] = scores[key]
    if not np.all(scores['floor'] >= scores['original']):
        raise AssertionError('候选累计分数不能小于原分数。')
    original = scores['original'] >= THRESHOLD
    candidate = scores['floor'] >= THRESHOLD
    extra = candidate & ~original
    clone_type = max_scale <= percent_dense * extent
    can_survive_existing_filters = (opacity >= .01) & (max_scale <= .1 * extent)
    arrays.update(original_selected=original, floor_selected=candidate, additional_selected=extra,
                  clone_type=clone_type, current_filter_survival_proxy=can_survive_existing_filters)
    target = output / f'{label}_points.npz'
    np.savez_compressed(target, **arrays)

    def counts_for(mask):
        clone, split = int((mask & clone_type).sum()), int((mask & ~clone_type).sum())
        return dict(selected=int(mask.sum()), cloned_parents=clone, split_parents=split,
                    projected_net_births_before_filter=clone + split,
                    projected_peak_during_split=len(mask) + clone + 2 * split,
                    current_filter_survival_proxy=int((mask & can_survive_existing_filters).sum()))

    return dict(frames=frames, point_count=len(counts), visible_points=int((counts > 0).sum()),
                threshold=THRESHOLD, original=counts_for(original), floor=counts_for(candidate),
                additional=counts_for(extra), additional_fraction_of_all_points=float(extra.mean()),
                additional_support=dict(minimum=int(counts[extra].min()) if extra.any() else None,
                    median=float(np.median(counts[extra])) if extra.any() else None,
                    maximum=int(counts[extra].max()) if extra.any() else None,
                    at_least_two_views=int((extra & (counts >= 2)).sum()),
                    at_least_five_views=int((extra & (counts >= 5)).sum())),
                changed_score_points=int((scores['floor'] > scores['original']).sum()),
                formula='mean_visible(max(norm2_xy(go+gn),norm2_xy(go))); 原分母每可见点每视角仅加1',
                arrays=str(target), arrays_sha256=digest(target),
                limit='固定模型零更新选择预测；没有执行clone/split/prune，投影点数不等于实际训练峰值或鞋恢复。')


def cpu_check():
    import ast
    import torch
    other = torch.tensor([[3., 0., 9.], [0., 0., 8.], [1., 2., 7.], [0., 0., 0.]])
    normal = torch.tensor([[-2., 0., 6.], [2., 0., 5.], [1., 2., 4.], [0., 0., 0.]])
    a, b = other.clone(), normal.clone()
    original, floor, go, gn = floor_scores(other, normal)
    order = camera_order(135, SEED)
    calls = {node.func.attr for node in ast.walk(ast.parse(Path(__file__).read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    checks = dict(cancellation_keeps_other=float(original[0]) == 1. and float(floor[0]) == 3.,
                  normal_only_preserved=float(original[1]) == 2. and float(floor[1]) == 2.,
                  aligned_preserved=bool(original[2] == floor[2]), zero_preserved=float(floor[3]) == 0.,
                  floor_never_smaller=bool((floor >= original).all()),
                  input_tensors_unchanged=bool(torch.equal(a, other) and torch.equal(b, normal)),
                  xy_only=bool(torch.equal(go, other[:, :2].norm(dim=-1))),
                  full_permutation=len(order) == 135 and set(order) == set(range(135)),
                  first100_unique=len(set(order[:100])) == 100,
                  seed_repeatable=order == camera_order(135, SEED),
                  no_training_mutator_calls=not bool(calls & {'step', 'training_setup', 'densify_and_prune',
                      'densify_and_clone', 'densify_and_split', 'prune_points', 'reset_opacity'}))
    print(json.dumps(dict(status='PASS' if all(checks.values()) else 'FAILED', checks=checks,
                          first100_indices=order[:100], scope='CPU公式/采样/静态调用预检，未导入科学renderer，未启动GPU'), ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError('CPU预检失败。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--iteration', type=int, default=16000)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--bootstrapped', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    if args.cpu_check:
        cpu_check()
        return
    if any(x is None for x in (args.model, args.snapshot, args.output)):
        parser.error('真实诊断需要model/snapshot/output。')
    args.model, args.snapshot, args.output = args.model.resolve(), args.snapshot.resolve(), args.output.resolve()
    canonical = ROOT / 'output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp'
    if args.model != canonical or args.iteration != 16000 or args.seed != SEED:
        parser.error('仅接受预注册G2-T06@16000及seed1012。')
    if args.output != OUTPUT:
        parser.error('仅接受预注册独立输出14；失败重试需另行登记新目录。')
    if any((args.output / name).exists() for name in ('plan.json', 'per_frame', 'per_frame.jsonl', 'summary.json')):
        parser.error('已有诊断资产，禁止覆盖或复跑。')
    if not args.bootstrapped:
        dispatch_bootstrap(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    pc, lights, cameras, pipe, visibility, extent, inputs, sources, provenance = initialize(args)
    import numpy as np
    import torch
    from scene.static_dl_ambient import render_photometric
    from scene.static_dl_normal_opacity import geometry_losses
    from scene.static_dl_censored import censored_photometric_loss
    params = {name: getattr(pc, name) for name in ('_xyz', '_rotation', '_scaling', '_opacity',
              '_albedo_dc', '_albedo_rest', '_roughness', 'feature')}
    params['light_log_intensity'] = lights.log_intensity
    required = ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc', 'light_log_intensity')
    if pc.optimizer is not None or any(p.grad is not None for p in params.values()):
        raise AssertionError('初始化不应有优化器或参数梯度。')
    parameter_before, rng_before = tensor_digest(params), rng_snapshot()
    order = camera_order(135, SEED)
    original_command = json.loads(os.environ['LUMI_FLOOR_PROFILE_COMMAND_JSON'])
    bootstrap_command = [sys.executable, str(BOOTSTRAP), str(args.snapshot),
                         '_lumimotion_goal3_densification_floor_profile'] + sys.argv[1:]
    plan = dict(model=str(args.model), iteration=16000, source=pipe.source_path, snapshot=str(args.snapshot),
                inputs_before=inputs, code_sources=sources, scientific_source_count=len(sources) - 3,
                command=original_command, command_shell=shlex.join(original_command), bootstrap_command=bootstrap_command,
                seed=SEED, all135_camera_indices=order, first100_camera_indices=order[:100],
                all135_camera_names=[cameras[i].image_name_train_light for i in order],
                threshold=THRESHOLD, densification_interval=100, loss_weights=dict(photo=1., alpha=pipe.lambda_alpha_loss,
                    normal=pipe.static_dl_normal_weight, distortion=pipe.lambda_dist),
                formula='d_original=norm2_xy(go+gn); d_floor=max(d_original,norm2_xy(go)); go已含RGB+alpha+dist权重，gn已含normal权重，不再乘权重',
                optimizer_created=False, parameter_updates=0, densification_called=False, reset_called=False,
                train_only=True, test_images_loaded=False, gt_geometry_material_shadow_intensity_loaded=False,
                windows='首100为实际seed新起点至首个增密窗口；全135为独立全训练集描述，非跨拓扑连续训练预测',
                decision='首100无新增选择则不支持T05；两窗口均无新增则否决。任何正新增先报告点数/可见支持/预计峰值，资源复核前不启动T05。',
                parameter_digest_before=parameter_before, rng_before=rng_before, native_provenance=provenance)
    write(args.output / 'plan.json', plan)
    (args.output / 'per_frame').mkdir(exist_ok=False)
    count = len(pc.get_xyz)
    accum = {key: torch.zeros(count, device='cuda') for key in ('original', 'floor')}
    support = torch.zeros(count, device='cuda')
    raised_support = torch.zeros(count, dtype=torch.int32, device='cuda')
    max_scale = pc.get_scaling.detach().max(dim=1).values.cpu().numpy()
    opacity = pc.get_opacity.detach().flatten().cpu().numpy()
    background = torch.zeros(3, device='cuda')
    rows, windows = [], {}
    for index, camera_index in enumerate(order, 1):
        camera = cameras[camera_index]
        camera.load2device()
        # 固定参数但推进BVH时钟，按interval=1每帧重建；不推进优化或损失日程。
        photo = render_photometric(camera, pc, pipe, background, lights, visibility, 16000 + index, 0.)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        rgb = censored_photometric_loss(photo['render'], target, alpha, pipe.static_dl_shadow_weight,
                                       pipe.static_dl_highlight_weight, pipe.static_dl_residual_threshold)[0]
        losses, geometry = geometry_losses(camera, pc, pipe, background, alpha, photo, 'detach_opacity')
        loss = rgb + pipe.lambda_alpha_loss * losses['alpha'] + pipe.static_dl_normal_weight * losses['normal'] + pipe.lambda_dist * losses['distortion']
        if not torch.isfinite(loss):
            raise FloatingPointError('冻结训练损失非有限。')
        loss.backward()
        for name in required:
            if params[name].grad is None or not torch.isfinite(params[name].grad).all():
                raise FloatingPointError('实际训练参数梯度缺失/非有限：' + name)
        go = photo['viewspace_points'].grad
        gn = geometry['viewspace_points'].grad
        if go is None or gn is None or not torch.isfinite(go).all() or not torch.isfinite(gn).all():
            raise FloatingPointError('实际加权屏幕支路余切缺失/非有限。')
        grads_before = {k: p.grad.detach().clone() for k, p in params.items() if p.grad is not None}
        forward = {'rgb': photo['render'], 'alpha': photo['rend_alpha'], 'expected': geometry['surf_depth'],
                   'gs_normal': geometry['rend_normal'], 'depth_normal': geometry['surf_normal']}
        forward_before = tensor_digest(forward)
        go_before, gn_before = go.detach().clone(), gn.detach().clone()
        original, floor, go_norm, gn_norm = floor_scores(go, gn)
        reference = torch.norm(go.detach().clone().add_(gn.detach())[:, :2], dim=-1)
        selected = photo['visibility_filter']
        with torch.no_grad():
            accum['original'][selected] += original[selected]
            accum['floor'][selected] += floor[selected]
            support[selected] += 1
            raised_support[selected] += (floor[selected] > original[selected]).int()
        checks = dict(original_score_exact=bool(torch.equal(original, reference)),
                      floor_never_smaller=bool((floor >= original).all()),
                      forward_unchanged=forward_before == tensor_digest(forward),
                      all_parameter_gradients_unchanged=all(torch.equal(params[k].grad, v) for k, v in grads_before.items()),
                      both_screen_gradients_unchanged=bool(torch.equal(go, go_before) and torch.equal(gn, gn_before)),
                      visibility_shared=bool(torch.equal(selected, geometry['visibility_filter'])),
                      finite_scores=bool(torch.isfinite(original).all() and torch.isfinite(floor).all()))
        if not all(checks.values()):
            raise AssertionError('分数诊断影响原前向/真实梯度或统计公式不一致。')
        frame_path = args.output / 'per_frame' / f'{index:03d}_{camera.image_name_train_light}.npz'
        np.savez_compressed(frame_path, original=original.cpu().numpy(), floor=floor.cpu().numpy(),
                            non_normal_norm=go_norm.cpu().numpy(), normal_norm=gn_norm.cpu().numpy(),
                            visible=selected.cpu().numpy())
        row = dict(order=index, camera_index=camera_index, frame=camera.image_name_train_light,
                   in_first100=index <= 100, loss=float(loss), photo_loss=float(rgb),
                   checks=checks, visible_points=int(selected.sum()),
                   raised_visible_points=int(((floor > original) & selected).sum()),
                   parameter_grad_norms={k: float(v.norm()) for k, v in grads_before.items()},
                   arrays=str(frame_path), arrays_sha256=digest(frame_path))
        rows.append(row)
        with (args.output / 'per_frame.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        for parameter in params.values():
            parameter.grad = None
        camera.load2device('cpu')
        if index == 1:
            write(args.output / 'runtime_active.json', runtime_evidence(provenance))
        if index == 100:
            windows['first100'] = summarize_window(args.output, 'first100', accum, support, raised_support,
                                                   max_scale, opacity, extent, pipe.percent_dense, index)
        if index % 15 == 0 or index == 100:
            print(json.dumps(dict(completed=index, total=135, frame=row['frame'], raised_visible_points=row['raised_visible_points']), ensure_ascii=False), flush=True)
        del photo, geometry, losses, loss, rgb, go, gn, grads_before, forward, go_before, gn_before
    windows['all135'] = summarize_window(args.output, 'all135', accum, support, raised_support,
                                         max_scale, opacity, extent, pipe.percent_dense, 135)
    runtime = runtime_evidence(provenance)
    rng_after = rng_snapshot()
    invariants = dict(parameters_unchanged=parameter_before == tensor_digest(params),
                      inputs_unchanged=inputs == {p: digest(p) for p in inputs},
                      code_unchanged=sources == {p: digest(p) for p in sources},
                      all_rng_unchanged=rng_before == rng_after, no_optimizer=pc.optimizer is None,
                      parameter_gradients_cleared=all(p.grad is None for p in params.values()),
                      all_135_processed=len(rows) == 135, first100_correct=[r['camera_index'] for r in rows[:100]] == order[:100],
                      all_frame_checks=all(all(r['checks'].values()) for r in rows))
    first_extra, all_extra = windows['first100']['additional']['selected'], windows['all135']['additional']['selected']
    decision = ('REJECT_T05_NO_ADDITIONAL_SELECTION' if first_extra == 0 and all_extra == 0 else
                'NOT_SUPPORTED_FOR_FIRST100_WINDOW' if first_extra == 0 else 'REQUIRES_RESOURCE_REVIEW')
    summary = dict(status='PASS' if all(invariants.values()) else 'FAILED', invariants=invariants,
                   model=str(args.model), iteration=16000, frames=135, seed=SEED, windows=windows,
                   candidate_decision=decision, runtime=runtime, rng_after=rng_after,
                   optimizer_created=False, parameter_updates=0, test_images_loaded=False,
                   conclusion='零更新机制诊断，不产生新MAE；新选择不等于鞋尖恢复，不能以此宣布T05有效或联合目标PASS。')
    write(args.output / 'summary.json', summary)
    previous = (args.output / 'README.md').read_text() if (args.output / 'README.md').exists() else ''
    (args.output / 'README.md').write_text(previous + '\n\n# 增密支路抵消零更新诊断实际执行\n\n'
        + f'固定模型 `{args.model}`，iteration16000，source `{pipe.source_path}`；仅原135 TRAIN。完整命令：\n\n```bash\n'
        + shlex.join(original_command) + '\n```\n\n'
        + '原bootstrap及科学来源见plan.json，活跃运行库及GPU见runtime_active.json；日志logs/profile.log。逐帧分数per_frame/，逐帧核验per_frame.jsonl，首100及全135逐点累计first100_points.npz／all135_points.npz，汇总summary.json。\n\n'
        + '无optimizer、拓扑、reset和参数更新；未读取GT或test，无新科学MAE。代表图片/视频、四类目检均不适用：本项没有图像效果验收。\n\n'
        + f'执行与无更新核验 **{summary["status"]}**；候选决策 `{decision}`。新增选择须再经资源审核与真实训练，不代表足部修复。\n')
    if summary['status'] != 'PASS':
        raise AssertionError('来源/无更新/随机状态不变性核验失败。')
    print('增密支路抵消135 TRAIN零更新诊断PASS：' + str(args.output), flush=True)


if __name__ == '__main__':
    main()

