#!/usr/bin/env python3
"""固定T06@16000、原135训练帧：预测深度混合与normal/photo梯度零更新诊断。"""
import argparse
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
BANDS = ('below_half_tau', 'half_to_one_tau', 'above_tau')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def confidence_maps(expected, median, tau):
    import torch.nn.functional as F
    delta = F.max_pool2d((expected.detach() - median.detach()).abs()[None], 3, stride=1, padding=1)[0]
    ratio = delta / tau
    weight = (.1 + .9 / (1 + ratio.square())).detach()
    return delta, ratio, weight, (ratio < .5, (ratio >= .5) & (ratio <= 1), ratio > 1)


def gradient_stats(normal, photo):
    """仅同一参数组内求内积；逐点与整帧向量的符号分别报告。"""
    import torch
    a, b = normal.detach().double().reshape(len(normal), -1), photo.detach().double().reshape(len(photo), -1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('真实参数梯度非有限。')
    an, bn = a.norm(), b.norm()
    dots = (a * b).sum(-1)
    joint = (a.norm(dim=-1) > 0) & (b.norm(dim=-1) > 0)
    count = int(joint.sum())
    return dict(norm=float(an), photo_norm=float(bn), dot=float(dots.sum()),
                cosine=float(dots.sum() / (an * bn)) if an > 0 and bn > 0 else None,
                norm_to_photo=float(an / bn) if bn > 0 else None,
                joint_active_points=count, negative_point_count=int(((dots < 0) & joint).sum()),
                negative_point_fraction=float((dots[joint] < 0).double().mean()) if count else None)


def distributions(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return dict(count=0, mean=None, quantiles=None)
    if not np.isfinite(x).all():
        raise FloatingPointError('诊断分布非有限。')
    return dict(count=int(x.size), mean=float(x.mean()),
                quantiles={str(q): float(np.quantile(x, q)) for q in (0., .1, .5, .9, .95, .99, 1.)})


def initialize(args):
    # 与冻结bootstrap同一原生装载方式：先原仓构建/复用同SHA .so，再只从固定训练快照导入科学函数。
    env = Path('/home/han.li/miniconda3/envs/lumimotion-ushas')
    os.environ['PATH'] = str(env / 'bin') + ':' + os.environ.get('PATH', '')
    os.environ['CPATH'] = str(env / 'targets/x86_64-linux/include') + ':' + os.environ.get('CPATH', '')
    for key, value in dict(CUDA_HOME=str(env), CXX=str(env / 'bin/x86_64-conda-linux-gnu-g++'),
                           CC=str(env / 'bin/x86_64-conda-linux-gnu-gcc'), TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4').items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(ROOT))
    from scene.pbr_rasterizer import native_module
    native = native_module()
    sys.modules['diff_surfel_rasterization'] = native
    for name in list(sys.modules):
        if name.split('.')[0] in {'scene', 'utils', 'arguments', 'scripts', 'gaussian_renderer'}:
            del sys.modules[name]
    sys.path = [str(args.snapshot)] + [p for p in sys.path if p != str(ROOT)]
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
    sources = {str(Path(__file__).resolve()): digest(__file__), str(args.snapshot / 'snapshot_manifest.json'): digest(args.snapshot / 'snapshot_manifest.json')}
    for path, sha in training['new_sources'].items():
        p = Path(path).resolve()
        if not p.is_relative_to(args.snapshot) or digest(p) != sha:
            raise AssertionError('T06科学快照文件内容或路径变化。')
        sources[str(p)] = sha
    for func in (render_photometric, geometry_losses, censored_photometric_loss, readCamerasFromTransforms, cameraList_from_camInfos):
        p = Path(sys.modules[func.__module__].__file__).resolve()
        if not p.is_relative_to(args.snapshot) or str(p) not in sources:
            raise AssertionError('实际科学函数未从固定T06快照载入。')
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
    for value in vars(pc).values():
        if isinstance(value, torch.Tensor) and value.is_leaf:
            value.requires_grad_(False)
    pc._xyz.requires_grad_(True)
    pc._rotation.requires_grad_(True)
    lights = DirectionOnlyLights(config['static_dl_directions'])
    if set(lights.directions) != expected_frames or lights.sha256 != lighting['directions_sha256']:
        raise AssertionError('方向只允许原135训练帧。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    lights.log_intensity.requires_grad_(False)
    files = [training_path, static_path, lighting_path, ply_path, deform_path, command_path, metadata_path,
             Path(config['static_dl_directions'])] + [Path(info.image_path_train_light) for info in infos]
    inputs = {str(p): digest(p) for p in files}
    # 与真实训练记录的原输入逐一匹配；只读135图、原相机和纯方向。
    for path, sha in training['training_inputs'].items():
        if inputs.get(path) != sha:
            raise AssertionError('训练输入与T06来源不符：' + path)
    return pc, lights, cameras, pipe, StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval']), extent, inputs, sources, native.provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=16000)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.model, args.snapshot, args.output = args.model.resolve(), args.snapshot.resolve(), args.output.resolve()
    canonical = ROOT / 'output/1012-09-LH-static-jumpingjacks_DL_V2-G2-T06-reset后normal阻断透明度2k/model_mlp'
    if args.model != canonical or args.iteration != 16000:
        parser.error('仅诊断预注册T06@16000。')
    if not args.output.is_relative_to(ROOT / 'output'):
        parser.error('输出必须位于项目output。')
    if any((args.output / name).exists() for name in ('plan.json', 'per_frame.jsonl', 'summary.json', 'pooled_distribution.npz')):
        parser.error('已有诊断资产，失败重试不可覆盖；仅允许预注册README/启动日志。')
    args.output.mkdir(parents=True, exist_ok=True)
    pc, lights, cameras, pipe, visibility, extent, inputs, sources, provenance = initialize(args)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scene.static_dl_ambient import render_photometric, parameter_digest
    from scene.static_dl_normal_opacity import geometry_losses
    from scene.static_dl_censored import censored_photometric_loss
    tau = .002 * extent
    params = (pc._xyz, pc._rotation)
    names = ('xyz', 'quaternion_raw')
    parameter_before = parameter_digest(pc, lights)
    rng_before = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
    write(args.output / 'plan.json', dict(model=str(args.model), iteration=16000, snapshot=str(args.snapshot),
        source=pipe.source_path, frames=135, tau=tau, camera_extent=extent, extent_source='原135训练相机getNerfppNorm与保存值核对',
        formula='delta=max_pool3x3(abs(expected-median)); w=detach(.1+.9/(1+(delta/tau)^2)); L=sum(M*w*r)/sum(M*w)',
        normal_weight=pipe.static_dl_normal_weight, bands=list(BANDS), inputs_before=inputs, code_sources=sources,
        depth_units='两次相同原生光栅的同相机camera-Z；expected=allmap0/alpha，median=allmap5，无归一化/尺度拟合',
        native_provenance=provenance, optimizer_created=False, parameter_updates=0, densification_called=False,
        train_only=True, test_images_loaded=False, gt_geometry_material_shadow_intensity_loaded=False,
        parameter_units=dict(xyz='模型世界坐标参数', quaternion_raw='原始未归一化四元数参数，非弧度'),
        command=shlex.join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])))
    rows, distributions_all = [], {k: [] for k in ('delta', 'ratio', 'weight', 'residual')}
    mean_grads = {key: [torch.zeros_like(p) for p in params] for key in ('photo', 'uniform', 'candidate') + BANDS}
    background = torch.zeros(3, device='cuda')
    for index, camera in enumerate(cameras):
        camera.load2device()
        photo = render_photometric(camera, pc, pipe, background, lights, visibility, 16000, 0.)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        rgb = censored_photometric_loss(photo['render'], target, alpha, pipe.static_dl_shadow_weight,
                                       pipe.static_dl_highlight_weight, pipe.static_dl_residual_threshold)[0]
        old, geometry = geometry_losses(camera, pc, pipe, background, alpha, photo, 'detach_opacity')
        # 完全复用原M；唯一新量来自同模型expected/median深度，均camera-Z单位。
        mask = -F.max_pool2d(-(alpha > .99).float()[None], 5, stride=1, padding=2)[0]
        mask = mask * (photo['rend_alpha'].clamp(1e-6, 1-1e-6).detach() > .5)
        residual = 1 - (F.normalize(geometry['rend_normal'], dim=0) * F.normalize(geometry['surf_normal'], dim=0)).sum(0, keepdim=True)
        uniform = (mask * residual).sum() / mask.sum().clamp_min(1)
        if not torch.equal(uniform, old['normal']):
            raise AssertionError('重构原M/normal损失与冻结训练实现不相等。')
        if not torch.equal(geometry['rend_alpha'], photo['rend_alpha']):
            raise AssertionError('expected和median副本并非同一几何/alpha。')
        delta, ratio, weight, band_masks = confidence_maps(geometry['surf_depth'], photo['surf_depth'], tau)
        candidate = (mask * weight * residual).sum() / (mask * weight).sum().clamp_min(1)
        losses = dict(uniform=pipe.static_dl_normal_weight * uniform,
                      candidate=pipe.static_dl_normal_weight * candidate)
        for name, bmask in zip(BANDS, band_masks):
            # 分档使用原总M分母，三档相加等于原normal，避免小档平均夸大梯度。
            losses[name] = pipe.static_dl_normal_weight * (mask * bmask * residual).sum() / mask.sum().clamp_min(1)
        photo_grad = torch.autograd.grad(rgb, params, retain_graph=True)
        for i, g in enumerate(photo_grad):
            mean_grads['photo'][i].add_(g.detach() / 135)
        row = dict(frame=camera.image_name_train_light, photo_loss=float(rgb.detach()),
                   uniform_normal=float(uniform.detach()), candidate_normal=float(candidate.detach()),
                   support=int(mask.sum()), bands={}, gradients={})
        grads_saved = {}
        for key, loss in losses.items():
            grad = torch.autograd.grad(loss, params, retain_graph=True)
            row['gradients'][key] = {n: gradient_stats(g, p) for n, g, p in zip(names, grad, photo_grad)}
            grads_saved[key] = tuple(g.detach() for g in grad)
            for i, g in enumerate(grad):
                mean_grads[key][i].add_(g.detach() / 135)
        row['candidate_vs_uniform'] = {n: dict(
            norm_ratio=float(c.norm() / u.norm()) if u.norm() > 0 else None,
            cosine=gradient_stats(c, u)['cosine']) for n, c, u in zip(names, grads_saved['candidate'], grads_saved['uniform'])}
        support = mask.bool()
        for k, value in dict(delta=delta, ratio=ratio, weight=weight, residual=residual).items():
            a = value.detach()[support].cpu().numpy()
            distributions_all[k].append(a)
            row[k] = distributions(a)
        for name, bm in zip(BANDS, band_masks):
            selected = support & bm
            count = int(selected.sum())
            row['bands'][name] = dict(pixels=count, support_fraction=count/max(1,row['support']),
                normal_residual_mean=float(residual.detach()[selected].mean()) if count else None,
                normal_residual_sum=float(residual.detach()[selected].sum()),
                delta_tau=distributions(ratio[selected].cpu().numpy()))
        if not all(torch.isfinite(v) for v in losses.values()):
            raise FloatingPointError('normal诊断loss非有限。')
        rows.append(row)
        with (args.output / 'per_frame.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        camera.load2device('cpu')
        if (index + 1) % 15 == 0:
            print(json.dumps(dict(completed=index+1, total=135, frame=row['frame'], mean_weight=row['weight']['mean']),ensure_ascii=False),flush=True)
        del photo, geometry, old, losses, grads_saved, photo_grad, residual, uniform, candidate, rgb
    pooled = {k: np.concatenate(v) for k,v in distributions_all.items()}
    np.savez_compressed(args.output / 'pooled_distribution.npz', **pooled)
    gradient_summary = {}
    for key in ('uniform','candidate') + BANDS:
        gradient_summary[key] = {}
        for i, name in enumerate(names):
            stats = [r['gradients'][key][name] for r in rows]
            available = [r for r in stats if r['cosine'] is not None]
            active = sum(r['joint_active_points'] for r in stats)
            gradient_summary[key][name] = dict(
                mean_gradient_comparison=gradient_stats(mean_grads[key][i], mean_grads['photo'][i]),
                mean_per_view_norm=float(np.mean([r['norm'] for r in stats])),
                mean_per_view_photo_norm=float(np.mean([r['photo_norm'] for r in stats])),
                mean_per_view_dot=float(np.mean([r['dot'] for r in stats])),
                mean_per_view_cosine=float(np.mean([r['cosine'] for r in available])) if available else None,
                negative_view_count=sum(r['dot'] < 0 for r in available), defined_view_count=len(available),
                negative_view_fraction=sum(r['dot'] < 0 for r in available)/len(available) if available else None,
                negative_point_view_fraction=sum(r['negative_point_count'] for r in stats)/active if active else None)
    modules = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
               if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line) and line.split()[-1].endswith('.so')}
    runtime = {p: digest(p) for p in sorted(modules)}
    invariants = dict(parameters_unchanged=parameter_before == parameter_digest(pc, lights),
        inputs_unchanged=inputs == {p:digest(p) for p in inputs}, code_unchanged=sources == {p:digest(p) for p in sources},
        cpu_rng_unchanged=bool(torch.equal(rng_before[0],torch.get_rng_state())),
        cuda_rng_unchanged=bool(torch.equal(rng_before[1],torch.cuda.get_rng_state())),
        no_optimizer=pc.optimizer is None, no_parameter_grad_accumulation=all(p.grad is None for p in params),
        all_135_processed=len(rows)==135)
    summary = dict(status='PASS' if all(invariants.values()) else 'FAILED', invariants=invariants,
        model=str(args.model), iteration=16000, frames=135, tau=tau, extent=extent,
        distributions={k:distributions(v) for k,v in pooled.items()}, gradient_summary=gradient_summary,
        bands={key:dict(pixels=sum(r['bands'][key]['pixels'] for r in rows),
            normal_residual_sum=sum(r['bands'][key]['normal_residual_sum'] for r in rows),
            normal_residual_mean=(sum(r['bands'][key]['normal_residual_sum'] for r in rows)/max(1,sum(r['bands'][key]['pixels'] for r in rows)))) for key in BANDS},
        candidate_vs_uniform={n:dict(
            mean_gradient_norm_ratio=float(mean_grads['candidate'][i].norm()/mean_grads['uniform'][i].norm()) if mean_grads['uniform'][i].norm()>0 else None,
            mean_per_view_norm_ratio=float(np.mean([r['candidate_vs_uniform'][n]['norm_ratio'] for r in rows if r['candidate_vs_uniform'][n]['norm_ratio'] is not None])),
            mean_gradient_cosine=gradient_stats(mean_grads['candidate'][i],mean_grads['uniform'][i])['cosine']) for i,n in enumerate(names)},
        parameter_updates=0, optimizer_created=False, test_images_loaded=False,
        runtime_modules=runtime, native_provenance=provenance,
        conclusion='仅训练输入下的相关性与局部梯度诊断；不证明GT法线正确、不证明候选训练有效，须另行科学决策。')
    write(args.output/'summary.json',summary)
    readme = args.output/'README.md'
    previous = readme.read_text() if readme.exists() else ''
    readme.write_text(previous + '\n\n# 预测深度置信度零更新诊断实际执行\n\n'
        +f'固定模型 `{args.model}`，iteration16000，原135 TRAIN。完整命令、source和输入/code SHA见 plan.json；逐帧数据 per_frame.jsonl，汇总 summary.json，分布 pooled_distribution.npz。\n\n'
        +'未执行optimizer、增密或参数更新；XYZ与原始quaternion分别统计。没有GT/test监督，没有生成训练mask资产。无新渲染图片/视频与四类目检，均不适用；不是模型Goal验收。\n\n'
        +f'诊断执行/无更新核验 **{summary["status"]}**；是否值得训练须读取完整结果判断。\n')
    if summary['status'] != 'PASS':
        raise AssertionError('零更新或来源不变性核验失败。')
    print('135 TRAIN零更新置信度诊断PASS：'+str(args.output),flush=True)


if __name__ == '__main__':
    main()
