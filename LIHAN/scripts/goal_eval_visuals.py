#!/usr/bin/env python3
"""实际渲染全150帧RGB／Alpha／世界法线／原binary feature分离；只作目检。"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--load_iter', type=int, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--eval_directions', type=Path)
    p.add_argument('--smoke_frames', type=int, nargs='+',
                   help='仅接入冒烟允许指定帧；正式验收不要提供，默认完整150帧。')
    args = p.parse_args()
    import imageio.v2 as imageio
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from argparse import Namespace
    from scripts.eval_geometry_gt import GeometryRun
    from gaussian_renderer import render
    from scene.pbr_rasterizer import rasterizer_classes
    from goal_eval_protocol import read, write
    from goal_eval_render import physical_renderer
    from goal_eval_protocol import CONTROL, digest
    import diff_surfel_rasterization as effective_rasterizer
    effective_backend = getattr(effective_rasterizer, 'provenance', None)
    if not effective_backend or effective_backend.get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise AssertionError('Goal评测渲染必须经bootstrap明确绑定原生CUDA光栅器。')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    records = []
    for split in ('train', 'test'):
        metadata = read(run.source / f'transforms_{split}.json')
        records.extend((r['source_frame'], metadata, r) for r in metadata['frames'])
    records.sort(key=lambda r: r[0])
    if [n for n, _, _ in records] != list(range(1, 151)):
        raise AssertionError('四类可视化必须全150帧。')
    if args.smoke_frames:
        if not args.output_dir.resolve().is_relative_to(ROOT / 'output/smoke_test'):
            raise AssertionError('仅接入冒烟的选帧输出必须在output/smoke_test。')
        if not set(args.smoke_frames).issubset(range(1, 151)):
            raise AssertionError('冒烟帧必须在1..150。')
        records = [r for r in records if r[0] in args.smoke_frames]
    expected_frames = len(records)
    categories = ['rgb', 'alpha', 'normal', 'separation']
    writers = {}
    for key in categories:
        (args.output_dir / key).mkdir()
        writers[key] = imageio.get_writer(str(args.output_dir / (key + '.mp4')), fps=24,
                                         macro_block_size=1, codec='libx264')
    stage2 = None
    if (run.model / 'stage2_config.json').exists() or run.mode == 'photometric_lambertian':
        from scripts.render_stage2 import Stage2RenderRun
        stage2 = Stage2RenderRun(run.model, args.load_iter)
    cfg = dict(run.config)
    for key, default in dict(compute_cov3D_python=False, convert_SHs_python=False,
                             debug=False, depth_ratio=0., lambda_separation=0.).items():
        cfg.setdefault(key, default)
    settings_class, rasterizer_class = rasterizer_classes(Namespace(**cfg))
    physical, physical_provenance = physical_renderer(run, args.load_iter,
        args.eval_directions or CONTROL / 'eval_directions_only_single.json')
    marker = run.model / 'static_dl_training.json'
    if marker.exists():
        run.files[str(marker)] = digest(marker)
    source_snapshot = dict(run.files)
    geometry_checks = []
    try:
        with torch.no_grad():
            for n, metadata, record in records:
                camera, _, intrinsics, _ = run.camera(metadata, record, 2)
                c2w = np.asarray(record['transform_matrix'], dtype=np.float64)
                prediction = run.render(camera, c2w, intrinsics)
                deformation = run.deformation(camera)
                if (run.model / 'static_dl_training.json').exists():
                    maximum = {key: float(deformation[key].abs().max()) if torch.is_tensor(deformation[key])
                               else abs(float(deformation[key])) for key in ['d_xyz', 'd_rotation', 'd_scaling']}
                    if any(maximum.values()):
                        raise AssertionError('静态几何头重载后不为零。')
                else:
                    maximum = None
                if physical is not None:
                    package = physical(camera)
                elif stage2 is None:
                    package = render(camera, run.pc, Namespace(**cfg), torch.zeros(3, device='cuda'),
                        **{k: deformation[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color']})
                else:
                    package = stage2.render(camera)
                rgb = package['render'].permute(1, 2, 0).cpu().numpy()
                from utils.geometry_eval_utils import unit_normal
                rendered_normal, valid_normal = unit_normal(package['rend_normal'].permute(1, 2, 0).cpu().numpy())
                normal_difference = float(np.max(np.abs(rendered_normal[valid_normal]-prediction['normal'][valid_normal])))
                alpha_difference = float(np.max(np.abs(package['rend_alpha'].squeeze(0).cpu().numpy()-prediction['alpha'])))
                ratio = cfg['depth_ratio']
                surface_reference = prediction['expected']*(1-ratio)+prediction['median']*ratio
                visible = prediction['alpha'] >= .5
                depth_difference = float(np.max(np.abs(package['surf_depth'].squeeze(0).cpu().numpy()[visible]-surface_reference[visible])))
                if normal_difference > 2e-5 or alpha_difference > 2e-5 or depth_difference > 2e-5:
                    raise AssertionError('真实RGB渲染几何map与原GeometryRun不同。')
                geometry_checks.append(dict(source_frame=n, geometry_head_max_abs=maximum,
                    normal_max_abs=normal_difference, alpha_max_abs=alpha_difference,
                    surface_depth_max_abs_on_visible=depth_difference))
                feature = run.pc.get_binary_feature()
                colors = torch.cat((feature, 1-feature, torch.zeros_like(feature)), dim=1)
                means = run.pc.get_xyz + deformation['d_xyz']
                rotations = run.pc.get_rotation_bias(deformation['d_rotation'])
                scales = run.pc.get_scaling + deformation['d_scaling']
                opacity = run.pc.get_opacity
                if deformation['d_opacity'] is not None:
                    opacity = opacity + deformation['d_opacity']
                if hasattr(run.pc, 'stage2_geometry'):
                    means, rotations, scales, opacity = run.pc.stage2_geometry.prepare(means, rotations, scales, opacity)
                    # 分裂子点颜色继承来源点，与真实geometry映射一致。
                    parent = run.pc.stage2_geometry.parent_index
                    if parent is not None:
                        colors = colors[parent]
                settings = settings_class(image_height=camera.image_height, image_width=camera.image_width,
                    tanfovx=np.tan(camera.FoVx/2), tanfovy=np.tan(camera.FoVy/2),
                    bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
                    viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
                    sh_degree=run.pc.active_sh_degree, campos=camera.camera_center,
                    prefiltered=False, debug=False)
                sep, _, _ = rasterizer_class(settings)(means3D=means, means2D=torch.zeros_like(means),
                    shs=None, colors_precomp=colors, opacities=opacity, scales=scales,
                    rotations=rotations, cov3D_precomp=None)
                alpha = prediction['alpha'][..., None]
                images = dict(rgb=rgb, alpha=np.repeat(alpha, 3, axis=-1),
                    normal=(prediction['normal']*.5+.5)*alpha,
                    separation=sep.permute(1, 2, 0).cpu().numpy())
                for key, pixels in images.items():
                    if not np.isfinite(pixels).all():
                        raise AssertionError(f'{key} frame {n}出现非有限像素。')
                    pixels = (np.clip(pixels, 0, 1)*255).round().astype(np.uint8)
                    Image.fromarray(pixels).save(args.output_dir / key / f'{n:04d}.png')
                    writers[key].append_data(pixels)
                del camera
                print(f'四类可视化 frame {n:04d}；目标帧数{expected_frames}', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    videos = {}
    for key in categories:
        path = args.output_dir / (key + '.mp4')
        cap = cv2.VideoCapture(str(path)); count = 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        pngs = len(list((args.output_dir / key).glob('*.png')))
        videos[key] = dict(path=str(path), decoded_frames=count, png_count=pngs)
        if count != expected_frames or pngs != expected_frames:
            raise AssertionError('四类可视化不完整：' + key)
    width, height = 400, 400
    canvas = Image.new('RGB', (3*width, 4*(height+25)), '#181818')
    draw = ImageDraw.Draw(canvas)
    representatives = (10, 80, 150) if not args.smoke_frames else tuple(r[0] for r in
        (records[0], records[len(records)//2], records[-1]))
    for i, key in enumerate(categories):
        for j, n in enumerate(representatives):
            image = Image.open(args.output_dir / key / f'{n:04d}.png').convert('RGB')
            image.thumbnail((width, height))
            canvas.paste(image, (j*width, i*(height+25)+25))
            draw.text((j*width+8, i*(height+25)+5), f'{key}: frame {n}', fill='white')
    representative_name = 'four_classes_smoke.png' if args.smoke_frames else 'four_classes_10_80_150.png'
    canvas.save(args.output_dir / representative_name)
    for path, expected in source_snapshot.items():
        if digest(path) != expected:
            raise AssertionError('渲染期间来源改变：' + path)
    report = dict(status='PASS', frames=expected_frames, smoke=bool(args.smoke_frames),
        effective_rasterizer=effective_backend,
        geometry_renderer_consistency=geometry_checks, videos=videos, physical_renderer=physical_provenance,
        representative=str(args.output_dir / representative_name), manual_review='PENDING',
        separation_semantics='原binary feature红=高、绿=低；几何头全零冻结时仅表示特征门控，不能声称学到运动分离。',
        command={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, sources=source_snapshot)
    write(args.output_dir / 'artifacts.json', report)


if __name__ == '__main__':
    main()
