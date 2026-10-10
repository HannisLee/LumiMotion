"""读取 PLY/PTH 材质，在对应相机与时刻直接评测 GT albedo。"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import time
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_renderer.render_albedo import render_albedo
from scene.stage2_photometric import LambertianMaterial, frame_key, make_material
from scripts.stage2_runtime import FrozenScene, checkpoint_path, file_sha256, model_dir
from utils.albedo_metrics import (ScaleFit, interior_mask, linear_to_srgb,
                                  normalize_accumulated, srgb_to_linear, summarize_errors)


def evaluation_config(config, source_path):
    config.eval = True
    config.load2gpu_on_the_fly = True
    if source_path:
        config.source_path = source_path
    if config.is_blender:
        for name in ('transforms_train.json', 'transforms_test.json', 'points3d.ply'):
            if not (Path(config.source_path) / name).is_file():
                # Blender reader 会在缺少 PLY 时自动生成；只读评测应提前拒绝。
                raise ValueError('评测 source 缺少必要文件：' + name)
    return config


class AlbedoRun:
    def __init__(self, model, iteration, source_path=None):
        checkpoint = checkpoint_path(model, iteration)
        self.material = None
        self.checkpoint_files = []
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cuda')
            if state.get('format_version') not in {1, 2} or state['iteration'] != iteration:
                raise ValueError('Stage2 checkpoint 格式或 iteration 不匹配。')
            config = evaluation_config(Namespace(**state['config']), source_path)
            self.context = FrozenScene(config, config.stage1_model_path, state['source_iteration'])
            if (self.context.source_files != state['source_files'] or
                    self.context.fingerprint != state['frozen_fingerprint']):
                raise ValueError('Stage2 checkpoint 的来源几何不匹配。')
            self.material = make_material(self.context.pc.get_albedo.detach(), config, state['material'])
            self.material.requires_grad_(False)
            if state.get('geometry') is not None:
                from scene.stage2_geometry import GeometryResidual
                self.context.pc.stage2_geometry = GeometryResidual(self.context.pc.get_xyz, config)
                self.context.pc.stage2_geometry.load_state_dict(state['geometry'])
                self.context.pc.stage2_geometry.requires_grad_(False)
            self.mode = 'photometric_lambertian'
            self.checkpoint_files.append(checkpoint)
        else:
            # 原版训练只保存 ModelParams 的 cfg_args；安全解析并补齐渲染默认项。
            from scripts.eval_geometry_gt import read_config
            from arguments import PipelineParams
            config_values, config_path = read_config(model)
            defaults_parser = argparse.ArgumentParser()
            PipelineParams(defaults_parser)
            config = defaults_parser.parse_args([])
            vars(config).update(config_values)
            config = evaluation_config(config, source_path)
            self.context = FrozenScene(config, model, iteration)
            self.mode = 'original_ir'
            self.checkpoint_files.append(config_path)
        self.config = config
        self.checkpoint_hashes = {str(p.resolve()): file_sha256(p) for p in self.checkpoint_files}
        self.checkpoint_hashes.update(self.context.source_files)

    @torch.no_grad()
    def render(self, camera):
        self.context.load_camera(camera)
        deformation = self.context.deformation(camera)
        albedo = self.context.pc.get_albedo if self.material is None else self.material.albedo
        pkg = render_albedo(camera, self.context.pc, albedo, deformation,
                            self.mode == 'original_ir' and self.config.compute_cov3D_python)
        self.context.unload_camera(camera)
        return {key: value.cpu().numpy() for key, value in pkg.items()}

    @torch.no_grad()
    def verify_renderer(self, camera):
        """真实权重上核验与原渲染器的材质光栅化一致。"""
        self.context.load_camera(camera)
        deformation = self.context.deformation(camera)
        albedo = self.context.pc.get_albedo if self.material is None else self.material.albedo
        pkg = render_albedo(camera, self.context.pc, albedo, deformation,
                            self.mode == 'original_ir' and self.config.compute_cov3D_python)
        if self.mode == 'original_ir':
            from gaussian_renderer.render_ir import render_ir
            legacy = render_ir(camera, self.context.pc, self.config, self.context.background,
                               **{key: deformation[key] for key in ('d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color')},
                               material_only=True)
            alpha = pkg['alpha']
            bg = self.context.background[:, None, None]
            expected = (pkg['accumulated_linear'] + bg * (1 - alpha)) * alpha + bg * (1 - alpha)
            actual = legacy['base_color_linear']
        else:
            from gaussian_renderer.render_stage2_lambertian import render_stage2_lambertian
            legacy = render_stage2_lambertian(camera, self.context.pc, self.material, deformation,
                                              self.context.background, direction=albedo.new_tensor([1., 0., 0.]))
            expected = pkg['accumulated_linear'] / pkg['alpha'].clamp_min(1e-6)
            actual = legacy['base_color_linear']
        torch.testing.assert_close(legacy['rend_alpha'], pkg['alpha'], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        self.context.unload_camera(camera)
        return {'frame': frame_key(camera.image_name_train_light), 'passed': True,
                'max_abs_difference_linear': float((actual - expected).abs().max())}

    def verify_unchanged(self):
        self.context.verify_frozen()
        for path, digest in self.checkpoint_hashes.items():
            if file_sha256(path) != digest:
                raise AssertionError('评测改动了权重文件：' + path)


def load_target(path, size):
    image = Image.open(path)
    if image.mode != 'RGBA':
        raise ValueError('GT albedo 必须具有真实 alpha：' + str(path))
    rgba = np.asarray(image, dtype=np.float64) / 255
    rgb = torch.from_numpy(srgb_to_linear(rgba[..., :3]).transpose(2, 0, 1)).float()
    alpha = torch.from_numpy(rgba[..., 3])[None].float()
    # 先在线性空间按 alpha 加权重采样，避免透明背景黑色污染表面颜色。
    resized_alpha = F.interpolate(alpha[None], size=size, mode='bilinear', align_corners=False)[0]
    resized_rgb = F.interpolate((rgb * alpha)[None], size=size, mode='bilinear', align_corners=False)[0]
    target = resized_rgb / resized_alpha.clamp_min(1e-6)
    return target.numpy().astype(np.float64), resized_alpha[0].numpy(), {
        'original_size': list(image.size), 'png_srgb_chunk': image.info.get('srgb'),
        'png_gamma_chunk': image.info.get('gamma'),
    }


class Samples:
    def __init__(self):
        self.data = {}

    def add(self, name, pred, target, mask, alpha):
        if mask.any():
            self.data.setdefault(name, []).append((pred[:, mask], target[:, mask], alpha[mask]))

    def summary(self, threshold):
        result = {}
        for name, values in self.data.items():
            pred = np.concatenate([x[0] for x in values], axis=1)[:, None]
            target = np.concatenate([x[1] for x in values], axis=1)[:, None]
            alpha = np.concatenate([x[2] for x in values])[None]
            result[name] = summarize_errors(pred, target, np.ones(alpha.shape, dtype=bool), alpha, threshold)
        return result


def preview(path, key, pred, target, gt_alpha, alpha, core):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ys, xs = np.where(gt_alpha > 0.5)
    crop = (slice(max(int(ys.min()) - 8, 0), int(ys.max()) + 9),
            slice(max(int(xs.min()) - 8, 0), int(xs.max()) + 9))
    images = [np.moveaxis(linear_to_srgb(target).clip(0, 1) * gt_alpha, 0, -1),
              np.moveaxis(linear_to_srgb(pred).clip(0, 1) * alpha, 0, -1),
              np.where(core, np.abs(pred - target).mean(0), np.nan), alpha]
    titles = ['GT albedo (sRGB display)', 'Predicted albedo (sRGB display)', 'Linear RGB MAE (0-0.25)', 'Predicted alpha']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for index, (ax, image, title) in enumerate(zip(axes, images, titles)):
        if index < 2:
            ax.imshow(image[crop])
        else:
            ax.imshow(image[crop], cmap='magma' if index == 2 else 'gray', vmin=0, vmax=0.25 if index == 2 else 1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle('Frame ' + key)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_iteration(args, model, iteration, destination, gt_files):
    started = time.monotonic()
    run = AlbedoRun(model, iteration, args.source_path)
    ctx = run.context
    train_keys = {frame_key(c.image_name_train_light) for c in ctx.train_cameras}
    test_keys = {frame_key(c.image_name_train_light) for c in ctx.test_cameras}
    if train_keys & test_keys:
        raise ValueError('校准训练帧与测试帧重叠。')
    train = sorted(ctx.train_cameras, key=lambda c: float(c.fid.item()))
    cameras = ctx.test_cameras if args.split == 'test' else (train if args.split == 'train' else train + ctx.test_cameras)
    cameras = sorted(cameras, key=lambda c: float(c.fid.item()))
    if args.max_frames:
        train, cameras = train[:args.max_frames], cameras[:args.max_frames]
    if not cameras:
        raise ValueError('所选评测集合为空。')
    destination.mkdir()
    used_gt = {}

    def target_for(camera):
        key = frame_key(camera.image_name_train_light)
        if key not in gt_files:
            raise ValueError('GT albedo 缺少准确帧号：' + key)
        path = gt_files[key]
        used_gt[str(path)] = file_sha256(path)
        target, alpha, info = load_target(path, (camera.image_height, camera.image_width))
        if info['png_srgb_chunk'] is None and info['png_gamma_chunk'] is None and not args.assume_srgb:
            raise ValueError('GT PNG 无颜色空间标记；核实后可显式 --assume_srgb。')
        if camera.gt_alpha_mask is None:
            raise ValueError('训练相机没有 GT alpha。')
        loader_alpha = camera.gt_alpha_mask[0].cpu().numpy()
        core = interior_mask(alpha, args.gt_alpha_threshold, args.erosion_pixels)
        if not core.any() or (loader_alpha[core] < 0.5).any():
            raise ValueError('GT albedo 前景为空或与训练相机 alpha 不对齐：' + key)
        return target, alpha, core, info

    renderer_check = run.verify_renderer(cameras[0])
    fit_result = None
    calibration_keys = []
    if args.calibrate_scale:
        fit = ScaleFit()
        for index, camera in enumerate(train):
            pkg = run.render(camera)
            pred = normalize_accumulated(pkg['accumulated_linear'], pkg['alpha'])
            target, _, core, _ = target_for(camera)
            fit.update(pred, target, core)
            calibration_keys.append(frame_key(camera.image_name_train_light))
            if (index + 1) % 25 == 0:
                print('calibration', iteration, index + 1, '/', len(train), flush=True)
        fit_result = fit.result()
    variants = {'raw': np.ones(3)}
    if fit_result:
        variants['train_scalar_diagnostic'] = np.full(3, fit_result['scalar'])
        variants['train_rgb_diagnostic'] = np.array(fit_result['rgb'])
    pooled = {name: Samples() for name in variants}
    records = []
    representatives = {0, len(cameras) // 2, len(cameras) - 1}
    for index, camera in enumerate(cameras):
        key = frame_key(camera.image_name_train_light)
        pkg = run.render(camera)
        alpha = pkg['alpha'][0]
        pred = normalize_accumulated(pkg['accumulated_linear'], alpha)
        target, gt_alpha, core, image_info = target_for(camera)
        truth_fg, pred_fg = gt_alpha >= 0.5, alpha >= 0.5
        rec = {'frame': key, 'time': float(camera.fid.item()), 'width': camera.image_width,
               'height': camera.image_height, 'gt_core_pixels': int(core.sum()),
               'gt_core_image_fraction': float(core.mean()), 'png_color_space': image_info,
               'alpha_iou': float((truth_fg & pred_fg).sum() / max((truth_fg | pred_fg).sum(), 1)),
               'variants': {}}
        for name, scale in variants.items():
            scaled = pred * scale[:, None, None]
            metrics = {'foreground': summarize_errors(scaled, target, core, alpha, args.pred_alpha_threshold)}
            pooled[name].add('foreground', scaled, target, core, alpha)
            rec['variants'][name] = metrics
        records.append(rec)
        if args.save_float:
            np.savez_compressed(destination / (key + '.npz'), albedo_linear=pred.astype(np.float32),
                                gt_albedo_linear=target.astype(np.float32), alpha=alpha, gt_alpha=gt_alpha,
                                gt_core=core, world_view=camera.world_view_transform.cpu().numpy(),
                                projection=camera.full_proj_transform.cpu().numpy(), time=float(camera.fid.item()))
        if index in representatives:
            preview(destination / (key + '_comparison.png'), key, pred, target, gt_alpha, alpha, core)
        print('evaluation', iteration, key, 'MAE', rec['variants']['raw']['foreground']['mae_linear'], flush=True)
    summaries = {}
    for name, samples in pooled.items():
        values = samples.summary(args.pred_alpha_threshold)
        summaries[name] = {'foreground': values['foreground'],
                           'frame_mean_foreground_mae_linear': float(np.mean([
                               r['variants'][name]['foreground']['mae_linear'] for r in records]))}
    run.verify_unchanged()
    report = {
        'protocol_version': 2, 'mode': run.mode, 'model_path': str(model), 'iteration': iteration,
        'source_path': run.config.source_path, 'split': args.split, 'frames': records,
        'protocol': {
            'prediction': '线性RGB零背景光栅化累加 / 预测alpha；alpha<=1e-6记零，未删除漏检GT像素',
            'gt_foreground': 'sRGB PNG解码为线性，alpha加权双线性重采样，GT alpha>=阈值后收缩',
            'primary': 'raw.foreground.mae_linear，全前景、不按材质拆分、不缩放、不裁剪',
            'aggregation': '全前景像素汇总；同时保存逐帧均值',
            'calibration': '仅训练帧GT albedo拟合；全模型一个scalar或RGB三系数；诊断指标，输出不裁剪',
            'gt_alpha_threshold': args.gt_alpha_threshold, 'erosion_pixels': args.erosion_pixels,
            'coverage_alpha_threshold': args.pred_alpha_threshold,
            'limited_frame_run': args.max_frames,
        },
        'calibration': fit_result, 'calibration_train_keys': calibration_keys,
        'renderer_check': renderer_check, 'summaries': summaries, 'checkpoint_hashes': run.checkpoint_hashes,
        'gt_file_hashes': used_gt, 'effective_config': vars(run.config),
        'checkpoint_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
    }
    (destination / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (destination / 'frames.csv').open('w') as stream:
        fields = ['frame', 'time', 'alpha_iou', 'mae_linear', 'rmse_linear', 'coverage_pct']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            raw = record['variants']['raw']
            row = {k: record[k] for k in ('frame', 'time', 'alpha_iou')}
            row.update({k: raw['foreground'][k] for k in ('mae_linear', 'rmse_linear', 'coverage_pct')})
            writer.writerow(row)
    print('summary', iteration, {name: {
        'foreground_mae_linear': values['foreground']['mae_linear'],
        'coverage_pct': values['foreground']['coverage_pct'],
    } for name, values in summaries.items()}, flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iterations', type=int, nargs='+', default=[55000])
    parser.add_argument('--gt_albedo_dir', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--source_path')
    parser.add_argument('--split', choices=['test', 'train', 'all'], default='test')
    parser.add_argument('--gt_alpha_threshold', type=float, default=0.99)
    parser.add_argument('--pred_alpha_threshold', type=float, default=0.9)
    parser.add_argument('--erosion_pixels', type=int, default=2)
    parser.add_argument('--calibrate_scale', action='store_true')
    parser.add_argument('--save_float', action='store_true')
    parser.add_argument('--assume_srgb', action='store_true')
    parser.add_argument('--max_frames', type=int, default=0)
    args = parser.parse_args(argv)
    if (args.max_frames < 0 or args.erosion_pixels < 0 or
            not 0 < args.gt_alpha_threshold <= 1 or not 0 < args.pred_alpha_threshold <= 1 or
            min(args.iterations) < 1 or
            len(set(args.iterations)) != len(args.iterations)):
        parser.error('帧数、iteration、收缩或阈值参数非法。')
    model = model_dir(args.model_path, 'mlp')
    args.gt_albedo_dir = args.gt_albedo_dir.resolve()
    gt_files = {}
    for path in sorted(args.gt_albedo_dir.glob('*.png')):
        key = frame_key(path.name)
        if key in gt_files:
            raise ValueError('GT 文件帧号重复：' + key)
        gt_files[key] = path
    args.output_path.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    provenance = {
        'command': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'source_hashes': {str(path.relative_to(root)): file_sha256(path) for path in
                          (Path(__file__).resolve(), root / 'utils/albedo_metrics.py', root / 'gaussian_renderer/render_albedo.py')},
        'dataset_metadata_hashes': {},
    }
    (args.output_path / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    try:
        for iteration in args.iterations:
            report = evaluate_iteration(args, model, iteration, args.output_path / ('iteration_' + str(iteration)), gt_files)
            source = Path(report['source_path'])
            for name in ('transforms_train.json', 'transforms_test.json', 'dataset_manifest.json'):
                path = source / name
                if path.is_file():
                    provenance['dataset_metadata_hashes'][str(path)] = file_sha256(path)
            (args.output_path / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
            del report
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as error:
        (args.output_path / 'FAILED').write_text(str(error) + '\n')
        raise
    (args.output_path / 'COMPLETED').write_text('评测完成；这是评测流程通过，非材质质量阈值验收。\n')


if __name__ == '__main__':
    main()
