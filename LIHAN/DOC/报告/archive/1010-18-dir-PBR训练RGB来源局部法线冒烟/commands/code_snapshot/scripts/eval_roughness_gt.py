"""原版 PBR 的通用前景 roughness MAE；逐帧浮点真值，联报覆盖率。"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from utils.geometry_eval_utils import file_sha256
from utils.roughness_metrics import aggregate, evaluate_region, normalize_roughness, load_target


ROOT = Path(__file__).resolve().parents[1]
def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_model(model, iteration):
    """在 CUDA 加载前拒绝 Lambertian、缺失 roughness 及没有 PBR Stage2 证据的权重。"""
    from plyfile import PlyData
    from scripts.eval_geometry_gt import read_config
    config, config_path = read_config(model)
    if (model / 'stage2_photometric').exists() or config.get('render_mode', 'original_ir') not in ('original_ir', 'ir'):
        raise ValueError('仅接受原版 PBR Stage2，拒绝 Lambertian/SH 模型。')
    if config.get('geometry_source_deformation_mode', 'dynamic') != 'dynamic':
        raise ValueError('评测必须使用每帧真实 time 的原版形变。')
    ply = model / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
    if config.get('pbr_light_mode', 'envmap') == 'gt_directional':
        from scene.pbr_directional import validate_directional_checkpoint
        light_files = validate_directional_checkpoint(model, iteration, config)
    else:
        env = model / 'envmap' / f'iteration_{iteration}'
        if not env.is_dir() or not any(env.iterdir()):
            raise ValueError('缺少同 iteration 的 PBR envmap，不能将 Stage1 当作已训练材质。')
        light_files = list(env.rglob('*'))
    data = PlyData.read(ply)
    if 'roughness' not in data['vertex'].data.dtype.names:
        raise ValueError('checkpoint 没有 roughness，禁止使用加载器的初始化默认值。')
    if not np.isfinite(data['vertex']['roughness']).all():
        raise ValueError('checkpoint roughness 包含 NaN/Inf。')
    return config, config_path, light_files


class RoughnessRun:
    def __init__(self, model, iteration, source_path):
        import torch
        from arguments import PipelineParams
        from scripts.eval_albedo_gt import evaluation_config
        from scripts.stage2_runtime import FrozenScene
        values, config_path, env_files = validate_model(model, iteration)
        parser = argparse.ArgumentParser()
        PipelineParams(parser)
        self.config = parser.parse_args([])
        vars(self.config).update(values)
        self.config = evaluation_config(self.config, source_path)
        self.context = FrozenScene(self.config, model, iteration)
        self.files = {str(config_path.resolve()): file_sha256(config_path)}
        self.files.update(self.context.source_files)
        self.files.update({str(p.resolve()): file_sha256(p) for p in env_files if p.is_file()})
        if self.context.pbr_feature_gate_provenance is not None:
            asset = self.context.source/'pbr_feature_gate.json'
            self.files[str(asset)] = file_sha256(asset)
        if self.context.pbr_rasterizer_provenance is not None:
            from scene.pbr_rasterizer import ROOT
            self.files.update({str(ROOT/relative):digest for relative,digest in
                               self.context.pbr_rasterizer_provenance['source_files'].items()})
        self.torch = torch

    def render(self, camera, verify=False):
        from gaussian_renderer.render_roughness import render_roughness
        ctx, torch = self.context, self.torch
        ctx.load_camera(camera)
        checks = []
        try:
            with torch.no_grad():
                deformation = ctx.deformation(camera)
                pkg = render_roughness(camera, ctx.pc, deformation, self.config.compute_cov3D_python,
                                       getattr(self.config,'pbr_rasterizer_backend','installed'))
                if verify:
                    from gaussian_renderer.render_ir import render_ir
                    # 同时核验黑、白背景；原版输出=(S+bg*(1-A))*A。
                    for white in (False, True):
                        bg = torch.ones_like(ctx.background) if white else torch.zeros_like(ctx.background)
                        legacy = render_ir(camera, ctx.pc, self.config, bg,
                            **{k: deformation[k] for k in ('d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color')},
                            material_only=True)
                        expected = (pkg['accumulated'] + bg[:, None, None] * (1 - pkg['alpha'])) * pkg['alpha']
                        torch.testing.assert_close(legacy['rend_alpha'], pkg['alpha'], atol=2e-6, rtol=1e-5)
                        torch.testing.assert_close(legacy['roughness'], expected, atol=2e-6, rtol=1e-5)
                        checks.append({'background': 'white' if white else 'black', 'passed': True,
                                       'max_abs_difference': float((legacy['roughness'] - expected).abs().max())})
                result = {k: v.cpu().numpy()[0] for k, v in pkg.items()}
        finally:
            ctx.unload_camera(camera)
        return result, checks


def preview(path, key, arrays, mae):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    core, score, valid = arrays['gt_core'], arrays['gt_mask'], arrays['valid_mask']
    ys, xs = np.where(core)
    crop = (slice(max(0, ys.min() - 8), ys.max() + 9), slice(max(0, xs.min() - 8), xs.max() + 9))
    coverage = np.zeros(core.shape, dtype=np.uint8)
    coverage[core] = 1
    coverage[score] = 2
    coverage[valid] = 3
    maps = [np.where(score, arrays['gt_roughness'], np.nan),
            np.where(valid, arrays['roughness'], np.nan),
            np.where(valid, np.abs(arrays['roughness'] - arrays['gt_roughness']), np.nan), coverage]
    titles = ['GT roughness', 'Predicted roughness (valid)', 'Absolute error', 'Coverage: gray=excluded, red=miss, green=valid']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for i, (ax, value, title) in enumerate(zip(axes, maps, titles)):
        cmap = 'magma' if i == 2 else 'gray'
        if i == 3:
            cmap = ListedColormap(['white', 'gray', 'red', 'green'])
        artist = ax.imshow(value[crop], cmap=cmap, vmin=0, vmax=3 if i == 3 else 1, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        if i < 3:
            fig.colorbar(artist, ax=ax, fraction=0.04)
    fig.suptitle(f'Frame {key}; conditional MAE={mae}')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate(args, destination):
    import torch
    from scene.stage2_photometric import frame_key
    from utils.albedo_metrics import interior_mask
    torch.set_num_threads(4)
    started = time.monotonic()
    run = RoughnessRun(args.model_path, args.iteration, args.source_path)
    ctx = run.context
    selected = {'test': ctx.test_cameras, 'train': ctx.train_cameras,
                'all': ctx.train_cameras + ctx.test_cameras}[args.split]
    cameras = sorted(selected, key=lambda c: frame_key(c.image_name_train_light))
    if args.frames:
        wanted = {f'{frame:04d}' for frame in args.frames}
        cameras = [c for c in cameras if frame_key(c.image_name_train_light) in wanted]
    keys = [frame_key(c.image_name_train_light) for c in cameras]
    if not keys or len(set(keys)) != len(keys) or (args.frames and set(keys) != wanted):
        raise ValueError('所选评测帧为空、重复，或请求的帧不在指定 split 中。')
    gt_files = {}
    for path in args.gt_roughness_dir.iterdir():
        if path.suffix.lower() not in ('.npz', '.exr'):
            continue
        key = frame_key(path.name)
        if key in gt_files:
            raise ValueError('GT 存在重复帧号：' + key)
        gt_files[key] = path.resolve()
    if set(keys) - set(gt_files):
        raise ValueError('GT 缺少评测帧：' + str(set(keys) - set(gt_files)))
    source = Path(run.config.source_path)
    inputs = [source / 'transforms_train.json', source / 'transforms_test.json',
              source / 'points3d.ply'] + [gt_files[k] for k in keys]
    gt_manifest_path = args.gt_roughness_dir / 'manifest.json'
    gt_manifest = None
    if gt_manifest_path.exists():
        gt_manifest = json.loads(gt_manifest_path.read_text())
        if gt_manifest.get('parameter') != 'roughness_r':
            raise ValueError('GT manifest 必须声明 roughness_r 原始参数。')
        inputs.append(gt_manifest_path)
        for path_key, sha_key in [('source_blend', 'source_blend_sha256'), ('transforms', 'transforms_sha256')]:
            path = Path(gt_manifest[path_key])
            if file_sha256(path) != gt_manifest[sha_key]:
                raise ValueError('GT 来源文件哈希不匹配：' + str(path))
            inputs.append(path)
    inputs.extend(Path(camera.image_path_train_light) for camera in cameras)
    hashes = {str(p.resolve()): file_sha256(p) for p in inputs}
    hashes.update(run.files)
    code_files = [ROOT / name for name in (
        'scripts/eval_roughness_gt.py', 'utils/roughness_metrics.py', 'gaussian_renderer/render_roughness.py',
        'gaussian_renderer/render_ir.py', 'utils/albedo_metrics.py', 'scripts/eval_albedo_gt.py',
        'scripts/stage2_runtime.py', 'scene/gaussian_model.py', 'scene/dataset_readers.py',
        'scene/deform_model.py', 'scene/stage2_source_deformation.py', 'utils/camera_utils.py')]
    provenance = {'command_argv': sys.argv, 'cwd': str(Path.cwd()),
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': {str(p.relative_to(ROOT)): file_sha256(p) for p in code_files},
                  'input_hashes': hashes, 'effective_config': vars(run.config), 'gt_manifest': gt_manifest}
    write_json(destination / 'provenance.json', provenance)
    (destination / 'float').mkdir()
    (destination / 'previews').mkdir()
    records, renderer_checks = [], []
    representatives = {0, len(cameras) // 2, len(cameras) - 1}
    for index, camera in enumerate(cameras):
        key = keys[index]
        time_value = float(camera.fid.item())
        size = (camera.image_height, camera.image_width)
        target, gt_alpha, info = load_target(gt_files[key], size)
        core = interior_mask(gt_alpha, threshold=args.gt_alpha_threshold, erosion_pixels=args.erosion_pixels)
        if not core.any() or camera.gt_alpha_mask is None:
            raise ValueError('GT 前景为空或相机缺少真实 alpha：' + key)
        camera_alpha = camera.gt_alpha_mask[0].cpu().numpy()
        if (camera_alpha[core] < 0.5).any():
            raise ValueError('GT roughness 与相机前景不匹配：' + key)
        gt_camera_info = info
        if gt_manifest is not None:
            entry = gt_manifest['frames'][key]
            if entry['file'] != gt_files[key].name or entry['sha256'] != file_sha256(gt_files[key]):
                raise ValueError('GT 逐帧文件/哈希不匹配：' + key)
            gt_camera_info = entry
        if 'time' in gt_camera_info and not np.isclose(time_value, gt_camera_info['time'], atol=1e-6, rtol=0):
            raise ValueError('GT 与模型 time 不一致：' + key)
        if 'camera_to_world' in gt_camera_info:
            # rasterizer相机为+Z前向；转换回Blender的-Y、-Z局部轴。
            c2w = np.linalg.inv(camera.world_view_transform.cpu().numpy().T) @ np.diag([1., -1., -1., 1.])
            if not np.allclose(c2w, gt_camera_info['camera_to_world'], atol=1e-5, rtol=0):
                raise ValueError('GT 与模型相机矩阵不一致：' + key)
        pkg, checks = run.render(camera, verify=index in representatives)
        if checks:
            renderer_checks.append({'frame': key, 'checks': checks})
        prediction = normalize_roughness(pkg['accumulated'], pkg['alpha'])
        np.testing.assert_allclose(prediction, pkg['roughness'], atol=1e-6, rtol=1e-6)
        valid = core & (pkg['alpha'].astype(np.float64) >= args.pred_alpha_threshold)
        record = {'frame': key, 'time': time_value, 'width': size[1], 'height': size[0],
                  'foreground': evaluate_region(prediction, target, core, pkg['alpha'], args.pred_alpha_threshold),
                  'gt_info': info, 'world_view_transform': camera.world_view_transform.cpu().tolist()}
        records.append(record)
        np.savez_compressed(destination / 'float' / f'{key}.npz',
                            accumulated=pkg['accumulated'], alpha=pkg['alpha'], roughness=prediction,
                            gt_roughness=target, gt_alpha=gt_alpha, gt_core=core, gt_mask=core, valid_mask=valid)
        print('frame', key, record['foreground'], flush=True)
    summaries = aggregate(records)
    candidates = [r for r in records if r['foreground']['mae'] is not None]
    worst = max(candidates, key=lambda r: r['foreground']['mae'])['frame'] if candidates else None
    representative_keys = sorted(set([keys[i] for i in representatives] + ([worst] if worst else [])))
    for key in representative_keys:
        rec = next(r for r in records if r['frame'] == key)
        with np.load(destination / 'float' / f'{key}.npz') as arrays:
            preview(destination / 'previews' / f'{key}.png', key, arrays, rec['foreground']['mae'])
    ctx.verify_frozen()
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise AssertionError('评测期间输入文件变化：' + path)
    for path, digest in provenance['source_hashes'].items():
        if file_sha256(ROOT / path) != digest:
            raise AssertionError('评测期间依赖代码变化：' + path)
    report = {'schema_version': 2, 'mode': 'original_ir',
              'pbr_light_mode': run.config.pbr_light_mode if hasattr(run.config, 'pbr_light_mode') else 'envmap',
              'source_path': str(source),
              'model_path': str(args.model_path), 'iteration': args.iteration,
              'protocol': {'split': args.split, 'frames': keys, 'region': 'GT foreground interior; no material/object subsets',
                  'parameter': 'r (not r_squared)', 'gt_alpha_threshold': args.gt_alpha_threshold,
                  'gt_erosion_pixels': args.erosion_pixels, 'pred_alpha_threshold': args.pred_alpha_threshold,
                  'aggregation': 'pooled valid foreground pixels',
                  'missing_prediction': 'excluded from conditional MAE; included in coverage denominator'},
              'summaries': summaries, 'frames': records, 'renderer_checks': renderer_checks,
              'preview_frames': representative_keys, 'worst_mae_frame': worst,
              'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started}
    write_json(destination / 'metrics.json', report)
    with (destination / 'frames.csv').open('w', newline='') as stream:
        fields = ['frame', 'time', 'width', 'height', 'mae', 'coverage_pct', 'gt_pixels', 'valid_pixels']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in fields[:4]}
            row.update({k: record['foreground'][k] for k in fields[4:]})
            writer.writerow(row)
    print('summary', json.dumps(summaries, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=55000)
    parser.add_argument('--source_path')
    parser.add_argument('--gt_roughness_dir', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'train', 'all'], default='test')
    parser.add_argument('--frames', type=int, nargs='+')
    parser.add_argument('--gt_alpha_threshold', type=float, default=0.99)
    parser.add_argument('--pred_alpha_threshold', type=float, default=0.9)
    parser.add_argument('--erosion_pixels', type=int, default=2)
    args = parser.parse_args(argv)
    if (args.iteration < 1 or args.erosion_pixels < 0 or not 0 < args.gt_alpha_threshold <= 1
            or not 0 < args.pred_alpha_threshold <= 1 or (args.frames and (min(args.frames) < 0 or len(set(args.frames)) != len(args.frames)))):
        parser.error('iteration、帧号、收缩像素或alpha阈值非法。')
    for name in ('model_path', 'gt_roughness_dir', 'output_path'):
        setattr(args, name, getattr(args, name).resolve())
    args.output_path.mkdir(parents=True, exist_ok=False)
    try:
        evaluate(args, args.output_path)
    except Exception as error:
        (args.output_path / 'FAILED').write_text(str(error) + '\n')
        raise
    (args.output_path / 'COMPLETED').write_text('所选帧评测与输入不变检查通过；不表示材质恢复质量合格。\n')


if __name__ == '__main__':
    main()
