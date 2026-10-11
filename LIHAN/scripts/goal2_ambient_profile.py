#!/usr/bin/env python3
"""固定T13、原135训练RGB的一维底项零更新剖面及独立非零接口探针。"""
import argparse
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VALUES = (0., .001, .01, .05, .1)
FIELDS = ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth',
          'light_intensity', 'mean_visibility', 'light_visibility', 'ambient_fraction',
          'direct_linear', 'ambient_linear')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def initialize(args):
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_shadow_split import require_native_backend
    from scene.stage2_photometric import frame_key
    from utils.camera_utils import cameraList_from_camInfos

    provenance = require_native_backend()
    model = args.model.resolve()
    static_path = model / 'static_dl_training.json'
    early_path = model / 'early_geometry_training.json'
    lighting_path = model / 'static_dl_lighting/iteration_10000/lighting.json'
    ply_path = model / 'point_cloud/iteration_10000/point_cloud.ply'
    deform_path = model / 'deform/iteration_10000/deform.pth'
    static = json.loads(static_path.read_text())
    early = json.loads(early_path.read_text())
    config = static['args'].copy()
    if static['actual_entrypoint'] != 'scripts/train_static_dl_early_geometry.py' or early['normal_start'] != 2000:
        raise AssertionError('只允许预注册T13早启normal起点。')
    if config['white_background'] or config['depth_ratio'] != 1 or config['static_dl_mode'] != 'lambertian_visibility':
        raise AssertionError('原T13物理/背景/深度前向不符。')
    for key in ('source_sha256', 'extension_sha256'):
        if provenance[key] != static['rasterizer_provenance'][key]:
            raise AssertionError('native与T13真实训练不同：' + key)
    actual_sources = {}
    for relative in ('scene/static_dl_ambient.py', 'scripts/train_static_dl_ambient.py'):
        file = ROOT / relative
        if digest(file) != digest(args.snapshot / relative):
            raise AssertionError('稳定核心与本轮95文件快照不符：' + relative)
        actual_sources[str(file)] = digest(file)
    actual_sources[str(Path(__file__).resolve())] = digest(__file__)
    actual_sources[str(args.snapshot / 'snapshot_manifest.json')] = digest(args.snapshot / 'snapshot_manifest.json')
    config.update(data_device='cpu', load2gpu_on_the_fly=True)
    source = Path(config['source_path']).resolve()
    metadata_path = source / 'transforms_train.json'
    metadata = json.loads(metadata_path.read_text())
    expected = {str(x).zfill(4) for x in range(1, 151) if x % 10}
    if len(metadata['frames']) != 135 or {str(f['source_frame']).zfill(4) for f in metadata['frames']} != expected:
        raise AssertionError('必须恰好原135训练帧。')
    lights = DirectionOnlyLights(config['static_dl_directions'])
    original_lighting = json.loads(lighting_path.read_text())
    if set(lights.directions) != expected or lights.sha256 != original_lighting['directions_sha256']:
        raise AssertionError('纯训练方向来源不符。')
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(original_lighting['intensity_from_train_rgb'], device='cuda').log())
    lights.log_intensity.requires_grad_(False)
    pc = GaussianModel(config['sh_degree'], no_binary_separation=True, fea_dim=config['hyper_dim'])
    pc.load_ply(str(ply_path))
    for value in vars(pc).values():
        if isinstance(value, torch.Tensor) and value.is_leaf:
            value.requires_grad_(False)
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', config['train_light_folder'], False, no_bg=True)
    cameras = cameraList_from_camInfos(infos, 1., Namespace(**config))
    if len(cameras) != 135 or {frame_key(c.image_name_train_light) for c in cameras} != expected:
        raise AssertionError('实际相机必须恰好原135训练帧。')
    files = [static_path, early_path, ply_path, deform_path, lighting_path, metadata_path,
             Path(config['static_dl_directions'])] + [Path(i.image_path_train_light) for i in infos]
    before = {str(p): digest(p) for p in files}
    return pc, lights, cameras, Namespace(**config), StaticVisibility(config['static_dl_shadow_bias'], 1), before, actual_sources, provenance


def render_probe(pc, lights, camera, pipe, a):
    import numpy as np
    import torch
    from scene.static_dl_ambient import render_photometric
    from scene.static_dl_photometric import StaticVisibility
    visibility = StaticVisibility(pipe.static_dl_shadow_bias, 1)
    camera.load2device()
    with torch.no_grad():
        package = render_photometric(camera, pc, pipe, torch.zeros(3, device='cuda'), lights,
                                     visibility, 10000, a, component_images=True)
    arrays = {key: package[key].detach().cpu().numpy() for key in FIELDS}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise AssertionError('独立探针出现非有限值。')
    camera.load2device('cpu')
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe_child', action='store_true', help='仅由父剖面进程启动的独立保存重载接口检查。')
    args = parser.parse_args()
    args.model, args.snapshot, args.output = args.model.resolve(), args.snapshot.resolve(), args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'output'):
        parser.error('输出必须位于项目output。')
    if args.probe_child:
        if not (args.output / 'probe_lighting.json').exists() or (args.output / 'probe_reloaded.npz').exists():
            parser.error('独立子进程必须读已有探针JSON，且不可覆盖输出。')
    elif args.output.exists():
        parser.error('输出已存在，失败重试必须使用新目录。')
    else:
        args.output.mkdir(parents=True)
    import numpy as np
    import torch
    from scene.static_dl_ambient import render_photometric, parameter_digest
    from scene.static_dl_censored import censored_photometric_loss

    pc, lights, cameras, pipe, visibility, sources, code, provenance = initialize(args)
    parameter_before = parameter_digest(pc, lights)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    if args.probe_child:
        saved = json.loads((args.output / 'probe_lighting.json').read_text())
        if saved['diagnostic_only'] is not True or saved['ambient_fraction'] != .05 or saved['trained_parameter'] is not False:
            raise AssertionError('探针必须明确为非训练参数的固定0.05。')
        if saved['directions_sha256'] != lights.sha256 or saved['source_model'] != str(args.model):
            raise AssertionError('探针源参数不符。')
        with torch.no_grad():
            lights.log_intensity.copy_(torch.as_tensor(saved['intensity_from_train_rgb'], device='cuda').log())
        if parameter_digest(pc, lights) != parameter_before:
            raise AssertionError('探针读回光强必须与固定T13完全相同。')
        arrays = render_probe(pc, lights, cameras[0], pipe, saved['ambient_fraction'])
        np.savez_compressed(args.output / 'probe_reloaded.npz', **arrays)
        unchanged = parameter_before == parameter_digest(pc, lights) and sources == {p: digest(p) for p in sources}
        write(args.output / 'probe_child_audit.json', dict(status='PASS' if unchanged else 'FAILED',
            diagnostic_only=True, trained_parameter=False, parameter_updates=0,
            parameters_unchanged=unchanged, ambient_fraction=.05, pid=os.getpid(),
            sources=sources, code=code, native_provenance=provenance))
        if not unchanged:
            raise AssertionError('独立探针改变源资产或参数。')
        return

    write(args.output / 'plan.json', dict(model=str(args.model), iteration=10000, snapshot=str(args.snapshot),
        values=list(VALUES), frames=135, source=pipe.source_path, sources_before=sources, code=code,
        native_provenance=provenance, parameter_updates=0, optimizer_created=False,
        train_only=True, test_images_loaded=False, gt_geometry_material_shadow_intensity_loaded=False,
        aggregation='135相机均匀平均；逐帧为原全图通道数分母的censored loss',
        derivative='a=0实际标量autograd；原残差权重detach规则不变',
        scope='固定T13几何/材质/直接光强的一维局部剖面，不证明全局不存在间接照明',
        command=shlex.join(sys.argv),
        full_invocation=shlex.join([v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v])))
    rows = []
    options = (pipe.static_dl_shadow_weight, pipe.static_dl_highlight_weight, pipe.static_dl_residual_threshold)
    for index, camera in enumerate(cameras):
        camera.load2device()
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        row = dict(camera=camera.image_name_train_light, losses={})
        for value in VALUES:
            fraction = torch.tensor(value, device='cuda', requires_grad=value == 0.)
            with torch.set_grad_enabled(value == 0.):
                package = render_photometric(camera, pc, pipe, torch.zeros(3, device='cuda'), lights,
                                             visibility, 10000 + index, fraction)
                loss = censored_photometric_loss(package['render'], target, alpha, *options)[0]
            if not torch.isfinite(loss):
                raise FloatingPointError('剖面loss非有限。')
            row['losses'][str(value)] = float(loss.detach())
            if value == 0.:
                derivative = torch.autograd.grad(loss, fraction)[0]
                if not torch.isfinite(derivative):
                    raise FloatingPointError('a0导数非有限。')
                row['derivative_at_zero'] = float(derivative)
        rows.append(row)
        with (args.output / 'per_frame.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if (index + 1) % 15 == 0:
            print(json.dumps(dict(completed=index + 1, total=135, last=row), ensure_ascii=False), flush=True)
        camera.load2device('cpu')
    means = {str(value): float(np.mean([r['losses'][str(value)] for r in rows])) for value in VALUES}
    baseline = means['0.0']
    # 独立诊断资产；不在模型目录写lighting，不伪造训练checkpoint。
    write(args.output / 'probe_lighting.json', dict(ambient_fraction=.05, ambient_cap=.1,
        intensity_from_train_rgb=lights.log_intensity.exp().detach().cpu().tolist(),
        directions_sha256=lights.sha256, source_model=str(args.model), source_iteration=10000,
        diagnostic_only=True, trained_parameter=False, parameter_updates=0,
        purpose='检验非零底项保存为JSON后在独立新进程实际读回新renderer，不是训练收益或非零traineda证据'))
    expected = render_probe(pc, lights, cameras[0], pipe, .05)
    np.savez_compressed(args.output / 'probe_expected.npz', **expected)
    child = [sys.executable, str(ROOT / 'LIHAN/scripts/bootstrap_1011.py'), str(ROOT),
             'LIHAN.scripts.goal2_ambient_profile', '--model', str(args.model), '--snapshot', str(args.snapshot),
             '--output', str(args.output), '--probe_child']
    write(args.output / 'probe_child_command.json', dict(argv=child, shell=shlex.join(child)))
    with (args.output / 'probe_child.log').open('w') as stream:
        result = subprocess.run(child, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('独立非零探针子进程失败，日志保留。')
    child_audit = json.loads((args.output / 'probe_child_audit.json').read_text())
    actual = np.load(args.output / 'probe_reloaded.npz')
    probe_checks = {}
    for key in FIELDS:
        a, b = expected[key], actual[key]
        probe_checks[key] = dict(passed=bool(np.allclose(a, b, atol=1e-5, rtol=1e-5)), max_abs=float(np.max(np.abs(a - b))))
    probe_pass = all(x['passed'] for x in probe_checks.values()) and child_audit['status'] == 'PASS'
    after = {p: digest(p) for p in sources}
    invariants = dict(input_files_unchanged=sources == after,
        parameters_unchanged=parameter_before == parameter_digest(pc, lights),
        cpu_rng_unchanged=bool(torch.equal(cpu_rng, torch.get_rng_state())),
        cuda_rng_unchanged=bool(torch.equal(cuda_rng, torch.cuda.get_rng_state())),
        optimizer_absent=pc.optimizer is None,
        no_trainable_scene_parameters=not any(v.requires_grad for v in vars(pc).values() if isinstance(v, torch.Tensor)),
        all_135_recorded=len(rows) == 135)
    derivative_mean = float(np.mean([r['derivative_at_zero'] for r in rows]))
    report = dict(status='PASS' if all(invariants.values()) and probe_pass else 'FAILED',
        model=str(args.model), iteration=10000, frames=135, values=list(VALUES), mean_photo_loss=means,
        delta_from_zero={k: value - baseline for k, value in means.items()},
        relative_delta_from_zero={k: (value - baseline) / baseline for k, value in means.items()},
        mean_derivative_at_zero=derivative_mean,
        negative_derivative_frames=sum(r['derivative_at_zero'] < 0 for r in rows),
        minimum_grid_value=float(min(means, key=means.get)),
        fixed_origin_grid_supports_positive_ambient=min(means, key=means.get) != '0.0',
        invariants=invariants, parameter_updates=0, parameters_sha256=parameter_before,
        sources_before=sources, sources_after=after, code=code,
        nonzero_interface_probe=dict(status='PASS' if probe_pass else 'FAILED', ambient_fraction=.05,
            diagnostic_only=True, trained_parameter=False, separate_process=True, checks=probe_checks,
            actual_mean_ambient_linear=float(actual['ambient_linear'].mean())),
        interpretation='只检验固定T13起点的一维RGB目标；不能排除联合优化后底项作用，也不能证明场景没有间接光。',
        weighting_note='原robust分组权重逐a重算但反传detach；跨门槛loss曲线可能非光滑，a0导数对应真实训练反传。')
    write(args.output / 'summary.json', report)
    invocation = json.loads((args.output / 'plan.json').read_text())['full_invocation']
    table = '\n'.join(f'| {value} | {means[str(value)]:.12g} | {means[str(value)] - baseline:+.12g} |' for value in VALUES)
    (args.output / 'README.md').write_text(f'''# 原135训练帧固定T13的间接底项零更新剖面

source：{pipe.source_path}；model：{args.model}；iteration：10000。固定所有场景参数，只读取原135训练RGB、原Alpha、相机和方向，未加载测试或GT几何/材质/阴影/光强。

完整实际命令：

```bash
{invocation}
```

| a | 原censored平均RGB loss | 相对a=0差值 |
| --- | ---: | ---: |
{table}

a=0平均导数：{derivative_mean:.12g}；负导数帧数：{report['negative_derivative_frames']}/135。细节在per_frame.jsonl、summary.json；来源前后SHA和零参数更新审计在plan/summary。该剖面只检验固定起点一维方向，不证明不存在间接光或联合优化无效。

非零a=.05接口探针：{report['nonzero_interface_probe']['status']}。独立probe_lighting.json明确为诊断值，原训练checkpoint没有修改；实际新进程命令与日志为probe_child_command.json、probe_child.log，线性直接/间接渲染数组为probe_expected.npz及probe_reloaded.npz。它验证非零保存重载接口，不能冒充实际训练学得非零a。

本次没有GS/Depth/Mesh MAE，没有四类可视化或代表视频；对应目检不适用，不作为Goal几何验收。运行的渲染属于上述仅训练相机诊断，输出均在本目录，未修改原完整15帧评测。

最终结论：**{report['status']}（仅剖面数据完整性、零更新与接口检查）**。科学方案支持与否看summary的loss/导数，不能用审计PASS替代收益。
''')
    print(json.dumps({k: report[k] for k in ('status', 'mean_photo_loss', 'mean_derivative_at_zero', 'negative_derivative_frames', 'minimum_grid_value')}, ensure_ascii=False), flush=True)
    if report['status'] != 'PASS':
        raise AssertionError('零更新或独立保存重载审计未通过。')


if __name__ == '__main__':
    main()
