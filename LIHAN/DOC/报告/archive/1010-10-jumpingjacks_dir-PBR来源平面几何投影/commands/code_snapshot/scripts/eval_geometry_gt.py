"""对 PLY + deformation PTH 评测 GS Render / Depth / Mesh Normal MAE 和深度。

三类法线共用 GT Normal EXR；Mesh 需模型预测清单。支持 Stage1 / 两类 Stage2，
无需 EnvLight、BRDF 或间接光采样；不会写入模型/数据目录。
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.geometry_eval_utils import (aggregate_regions, depth_normals, erode,
    evaluate_region, file_sha256, frame_number, load_normal_exr, normal_errors, unit_normal)


def read_config(model):
    candidates = [model/'stage2_config.json'] + sorted(model.glob('resume_*_config.json'),
        key=lambda p: int(p.name.split('_')[1]), reverse=True)
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text()), path
    path = model/'cfg_args'
    node = ast.parse(path.read_text().strip(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
        raise ValueError(f'不支持的 cfg_args 格式：{path}')
    return {item.arg: ast.literal_eval(item.value) for item in node.keywords}, path


def model_path(path):
    path = Path(path).resolve()
    if path.is_dir():
        return path
    alternative = Path(str(path)+'_mlp')
    if alternative.is_dir():
        return alternative
    raise FileNotFoundError(path)


class GeometryRun:
    def __init__(self, model, iteration, source_override=None):
        import torch
        from scene import GaussianModel, DeformModel
        self.model = model_path(model)
        config, config_path = read_config(self.model)
        photo = self.model/'stage2_photometric'/f'iteration_{iteration}'/'checkpoint.pth'
        self.files = {str(config_path): file_sha256(config_path)}
        if photo.is_file():
            state = torch.load(photo, map_location='cpu')
            if state.get('format_version') not in {1, 2}:
                raise ValueError('不支持的 photometric checkpoint 格式。')
            if state.get('format_version') == 2 and state.get('geometry') is None:
                raise ValueError('v2 checkpoint 缺少几何残差。')
            config = state['config']
            geometry_model = model_path(config['stage1_model_path'])
            geometry_iteration = state['source_iteration']
            for path, digest in state['source_files'].items():
                if file_sha256(path) != digest:
                    raise ValueError(f'冻结的 Stage1 来源哈希不符：{path}')
            self.files[str(photo)] = file_sha256(photo)
            self.mode = 'photometric_lambertian'
        else:
            geometry_model, geometry_iteration = self.model, iteration
            self.mode = config.get('render_mode', 'original_sh')
        self.geometry_model, self.geometry_iteration = geometry_model, geometry_iteration
        self.iteration = geometry_iteration
        self.source = Path(source_override or config['source_path']).resolve()
        self.config = config
        ply = geometry_model/'point_cloud'/f'iteration_{geometry_iteration}'/'point_cloud.ply'
        pth = geometry_model/'deform'/f'iteration_{geometry_iteration}'/'deform.pth'
        self.files.update({str(p): file_sha256(p) for p in (ply, pth)})
        self.context = None
        if config.get('pbr_deformation_anchor', False) or config.get('pbr_plane_projection', False):
            if self.mode != 'original_ir' or photo.is_file():
                raise ValueError('PBR来源形变与平面模型只能用于原生PBR几何评测。')
            from argparse import Namespace
            from scripts.stage2_runtime import FrozenScene
            runtime_config = dict(config, source_path=str(self.source))
            self.context = FrozenScene(Namespace(**runtime_config), geometry_model, geometry_iteration)
            self.pc, self.deform = self.context.pc, self.context.deform
            self.files.update(self.context.source_files)
            for enabled, prefix in [(config.get('pbr_deformation_anchor', False), 'pbr_deformation_anchor'),
                                    (config.get('pbr_plane_projection', False), 'pbr_plane_projection')]:
                if enabled:
                    metadata = geometry_model/(prefix+'.json')
                    asset = geometry_model/(prefix+'.npz')
                    self.files.update({str(p): file_sha256(p) for p in (metadata, asset)})
                    record = json.loads(metadata.read_text())
                    self.files.update(record['source_files'])
                    for relative, digest in record['source_snapshots'].items():
                        self.files[str(geometry_model/relative)] = digest
            marker = geometry_model/'pbr_directional'/f'iteration_{geometry_iteration}'/'lighting.json'
            if marker.is_file():
                self.files[str(marker)] = file_sha256(marker)
            return
        self.pc = GaussianModel(config.get('sh_degree', 3), config.get('no_binary_separation', False),
                                fea_dim=config.get('hyper_dim', 1))
        self.pc.load_ply(str(ply))
        if photo.is_file() and state.get('geometry') is not None:
            from argparse import Namespace
            from scene.stage2_geometry import GeometryResidual
            self.pc.stage2_geometry = GeometryResidual(self.pc.get_xyz, Namespace(**config))
            self.pc.stage2_geometry.load_state_dict(state['geometry'])
        self.deform = DeformModel(deform_type=config.get('deform_type', 'mlp'),
                                 is_blender=config.get('is_blender', True),
                                 hyper_dim=config.get('hyper_dim', 1), pred_color=config.get('pred_color', True))
        if not self.deform.load_weights(str(geometry_model), geometry_iteration):
            raise FileNotFoundError(pth)
        self.deform.deform.eval().requires_grad_(False)

    def deformation(self, camera):
        """几何评测/逐time Mesh与训练共享来源查询及真实几何模型。"""
        if self.context is not None:
            return self.context.deformation(camera)
        count = len(self.pc.get_xyz)
        return self.deform.step(self.pc.get_xyz.detach(), camera.fid.unsqueeze(0).expand(count, -1),
            iteration=self.iteration, feature=self.pc.get_binary_feature(), camera_center=camera.camera_center,
            is_training=False)

    def camera(self, metadata, record, divisor):
        import torch
        from scene.cameras import Camera
        from scene.dataset_readers import blender_c2w_to_camera_extrinsics
        image_path = self.source/self.config.get('train_light_folder', 'images')/record['file_path']
        if not image_path.suffix:
            image_path = image_path.with_suffix('.png')
        with Image.open(image_path) as image:
            original_width, original_height = image.size
            width, height = round(original_width/divisor), round(original_height/divisor)
            alpha = np.asarray(image.convert('RGBA').getchannel('A').resize((width, height))).astype(np.float32)/255.
        fx = record.get('fl_x', metadata.get('fl_x'))
        fy = record.get('fl_y', metadata.get('fl_y'))
        fovx = 2*math.atan(original_width/(2*fx)) if fx else record.get('camera_angle_x', metadata['camera_angle_x'])
        fovy = 2*math.atan(original_height/(2*fy)) if fy else record.get('camera_angle_y', metadata.get('camera_angle_y'))
        if fovy is None:
            fovy = 2*math.atan(original_height/original_width*math.tan(fovx/2))
        cx, cy = record.get('cx', metadata.get('cx', original_width/2)), record.get('cy', metadata.get('cy', original_height/2))
        if abs(cx-original_width/2) > 1e-5 or abs(cy-original_height/2) > 1e-5:
            raise ValueError('当前 rasterizer 仅支持居中主点。')
        rotation, translation = blender_c2w_to_camera_extrinsics(record['transform_matrix'])
        camera = Camera(colmap_id=0, R=rotation, T=translation, FoVx=fovx, FoVy=fovy,
            image_train_light=torch.zeros((3, height, width)), gt_alpha_mask=torch.from_numpy(alpha[None]),
            image_name_train_light=Path(record['file_path']).stem, image_path_train_light=str(image_path),
            uid=0, data_device='cuda', fid=record['time'])
        return camera, alpha, np.array([width/(2*math.tan(fovx/2)), height/(2*math.tan(fovy/2)), width/2, height/2]), image_path

    def render(self, camera, c2w, intrinsics):
        import torch
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        with torch.no_grad():
            deformation = self.deformation(camera)
            means = self.pc.get_xyz + deformation['d_xyz']
            rotations = self.pc.get_rotation_bias(deformation['d_rotation'])
            scales = self.pc.get_scaling+deformation['d_scaling']
            opacity = self.pc.get_opacity
            if deformation['d_opacity'] is not None:
                opacity = opacity + deformation['d_opacity']
            if hasattr(self.pc, 'stage2_geometry'):
                means,rotations,scales,opacity = self.pc.stage2_geometry.prepare(means,rotations,scales,opacity)
            settings = GaussianRasterizationSettings(image_height=camera.image_height, image_width=camera.image_width,
                tanfovx=math.tan(camera.FoVx/2), tanfovy=math.tan(camera.FoVy/2), bg=torch.zeros(3, device='cuda'),
                scale_modifier=1., viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
                sh_degree=self.pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=False)
            _, _, maps = GaussianRasterizer(settings)(means3D=means, means2D=torch.zeros_like(means), shs=None,
                colors_precomp=torch.zeros_like(means), opacities=opacity,
                scales=scales,
                rotations=rotations, cov3D_precomp=None)
            alpha = maps[1].cpu().numpy()
            normal = (maps[2:5].permute(1, 2, 0) @ camera.world_view_transform[:3, :3].T).cpu().numpy()
            expected = (maps[0]/maps[1].clamp_min(1e-6)).cpu().numpy()
            median = maps[5].cpu().numpy()
        normal, _ = unit_normal(normal)
        depth_normal, _ = depth_normals(expected, c2w, intrinsics)
        return dict(alpha=alpha, normal=normal, expected=expected, median=median, depth_normal=depth_normal)


def save_preview(path, prediction, gt_normal, gt_depth, mask):
    """数值来自 float arrays；PNG 仅用于观察，不参与计算。"""
    from matplotlib import colormaps
    panels = []
    for normal in (gt_normal, prediction['normal'], prediction['depth_normal']):
        panel = ((normal*.5+.5).clip(0, 1)*255).astype(np.uint8)
        panel[~mask] = 0
        panels.append(panel)
    error, _ = normal_errors(prediction['normal'], gt_normal)
    panel = (colormaps['turbo']((error['angular_deg']/90).clip(0, 1))[..., :3]*255).astype(np.uint8)
    panel[~mask] = 0
    panels.append(panel)
    if gt_depth is not None:
        for depth in (gt_depth, prediction['expected'], prediction['median']):
            sample = gt_depth[mask & (gt_depth > 0)]
            lo, hi = np.quantile(sample, (.01, .99))
            panel = (colormaps['viridis'](((depth-lo)/max(hi-lo, 1e-6)).clip(0, 1))[..., :3]*255).astype(np.uint8)
            panel[~mask] = 0
            panels.append(panel)
        panel = (colormaps['magma']((np.abs(prediction['expected']-gt_depth)/.2).clip(0, 1))[..., :3]*255).astype(np.uint8)
        panel[~mask] = 0
        panels.append(panel)
    labels = ['GT world normal EXR', 'GS Render Normal', 'Depth Normal (expected Z)',
              'GS Render Normal error (0-90 deg)', 'GT Z depth', 'Expected Z depth',
              'Median Z depth', 'Expected depth error (0-0.2 world units)']
    if 'mesh_normal' in prediction:
        mesh_mask = mask & prediction['mesh_valid']
        panel = ((prediction['mesh_normal']*.5+.5).clip(0, 1)*255).astype(np.uint8)
        panel[~mesh_mask] = 0
        panels.append(panel)
        labels.append('Mesh Normal (model reconstructed mesh)')
        for field, label in [('depth_normal', 'Depth Normal error'), ('mesh_normal', 'Mesh Normal error')]:
            error, valid = normal_errors(prediction[field], gt_normal)
            panel = (colormaps['turbo']((error['angular_deg']/90).clip(0, 1))[..., :3]*255).astype(np.uint8)
            panel[~(mask & valid)] = 0
            panels.append(panel)
            labels.append(label+' (0-90 deg)')
    height, width = mask.shape
    columns = 4
    canvas = Image.new('RGB', (columns*width, math.ceil(len(panels)/columns)*(height+28)))
    draw = ImageDraw.Draw(canvas)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    for index, panel in enumerate(panels):
        x, y = (index % columns)*width, (index // columns)*(height+28)
        draw.text((x+6, y+4), labels[index], fill='white', font=font)
        canvas.paste(Image.fromarray(panel), (x, y+28))
    canvas.save(path)


def save_metric_tables(output_dir, summary, frames, meters_per_world_unit=None):
    fields = ['source_frame', 'split', 'region', 'metric', 'mae', 'unit', 'coverage',
              'valid_pixels', 'status']
    aggregate_rows, frame_rows = [], []
    for region, stats in summary.items():
        for item in stats['normal_metrics'].values():
            aggregate_rows.append(dict(region=region, metric=item['label'],
                mae=item['mae_frame_mean_deg'], unit='degree', coverage=item['coverage'],
                valid_pixels=item['valid_pixels'], status=item['status']))
        for name, item in stats['depth'].items():
            value = item['mae_frame_mean']
            if value is not None and meters_per_world_unit is not None:
                value *= meters_per_world_unit
            aggregate_rows.append(dict(region=region, metric=f'{name.title()} Depth MAE',
                mae=value, unit='meter' if meters_per_world_unit is not None else 'world_unit', coverage=item['coverage'],
                valid_pixels=item['valid_pixels'], status='evaluated'))
    for frame in frames:
        for region, stats in frame['regions'].items():
            for item in stats['normal_metrics'].values():
                frame_rows.append(dict(source_frame=frame['source_frame'], split=frame['split'],
                    region=region, metric=item['label'], mae=item['angular_deg']['mean'],
                    unit='degree', coverage=item['coverage'], valid_pixels=item['valid_pixels'], status=item['status']))
    for name, rows in [('metrics_summary.csv', aggregate_rows), ('normal_metrics_per_frame.csv', frame_rows)]:
        with (output_dir/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--load_iter', type=int, required=True)
    parser.add_argument('--gt_root', type=Path, required=True)
    parser.add_argument('--gt_depth_dir', type=Path)
    parser.add_argument('--source_path', type=Path)
    parser.add_argument('--split', choices=('train', 'test', 'all'), default='test')
    parser.add_argument('--frames', type=int, nargs='+')
    parser.add_argument('--resolution', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--alpha_threshold', type=float, default=.5)
    parser.add_argument('--interior_radius', type=int, default=2)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--save_arrays', action='store_true')
    parser.add_argument('--pred_mesh_manifest', type=Path,
                        help='模型重建 Mesh 清单；不提供则 Mesh Normal MAE 为 not_available/null。')
    args = parser.parse_args()
    if not 0 < args.alpha_threshold < 1 or args.interior_radius < 0 or args.load_iter < 1:
        parser.error('alpha threshold、腐蚀半径或 iteration 不合法。')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    depth_manifest = None
    if args.gt_depth_dir is not None:
        path = args.gt_depth_dir/'manifest.json'
        depth_manifest = json.loads(path.read_text())
        if depth_manifest.get('depth_convention') != 'positive camera Z; unnormalized pinhole rays':
            raise ValueError('GT 深度定义不符。')
        run.files[str(path.resolve())] = file_sha256(path)
    mesh_manifest = None
    loaded_meshes = {}
    if args.pred_mesh_manifest is not None:
        args.pred_mesh_manifest = args.pred_mesh_manifest.resolve()
        mesh_manifest = json.loads(args.pred_mesh_manifest.read_text())
        if mesh_manifest.get('kind') != 'predicted_mesh' or mesh_manifest.get('gt_geometry_used') is not False:
            raise ValueError('需要仅来自模型预测的 Mesh；GT 网格不能当作模型重建结果。')
        if Path(mesh_manifest['model_path']).resolve() != run.model or mesh_manifest['iteration'] != args.load_iter:
            raise ValueError('预测 Mesh 与评测模型或迭代不对应。')
        if Path(mesh_manifest['source_path']).resolve() != run.source:
            raise ValueError('预测 Mesh 与评测数据坐标来源不对应。')
        if mesh_manifest.get('static_scene') and depth_manifest and not depth_manifest['static_geometry']:
            raise ValueError('动态 GT 不能用跨时刻融合的静态 Mesh 评测。')
        run.files[str(args.pred_mesh_manifest)] = file_sha256(args.pred_mesh_manifest)
        for path, digest in mesh_manifest.get('sources', {}).items():
            if file_sha256(path) != digest:
                raise ValueError(f'提取 Mesh 的模型/相机来源已改变：{path}')
            run.files[path] = digest
    frames = []
    for split in (('train', 'test') if args.split == 'all' else (args.split,)):
        path = run.source/f'transforms_{split}.json'
        run.files[str(path)] = file_sha256(path)
        metadata = json.loads(path.read_text())
        for record in metadata['frames']:
            number = int(record.get('source_frame', frame_number(record['file_path'])))
            if args.frames is None or number in args.frames:
                frames.append((number, split, metadata, record))
    frames.sort(key=lambda f: f[0])
    if not frames or len({f[0] for f in frames}) != len(frames):
        raise ValueError('评测帧为空或 source_frame 重复。')
    if args.frames and set(args.frames) != {f[0] for f in frames}:
        raise ValueError('请求的帧不在指定 split 中。')
    camera_metadata_path = args.gt_root/'camera.json'
    camera_metadata = json.loads(camera_metadata_path.read_text())
    run.files[str(camera_metadata_path.resolve())] = file_sha256(camera_metadata_path)
    rows = []
    for index, (number, split, metadata, record) in enumerate(frames):
        camera, alpha, intrinsics, image_path = run.camera(metadata, record, args.resolution)
        c2w = np.array(record['transform_matrix'], dtype=np.float64)
        gt_camera = camera_metadata['frames'][f'{number:04d}']
        if not np.allclose(c2w, gt_camera['extrinsics']['camera_to_world'], atol=1e-5, rtol=0):
            raise ValueError(f'frame {number}: GT 相机不匹配。')
        exr = args.gt_root/'normal_exr'/f'normal_{number:04d}.exr'
        gt_normal, gt_valid = load_normal_exr(exr, camera.image_height, camera.image_width)
        run.files[str(exr.resolve())] = file_sha256(exr)
        run.files[str(image_path)] = file_sha256(image_path)
        gt_depth, hit_valid = None, np.ones_like(alpha, bool)
        if depth_manifest is not None:
            entry = depth_manifest['frames'][str(number)]
            if entry.get('source_frame', number) != number or (
                    'time' in entry and not np.isclose(entry['time'], record['time'], atol=1e-7, rtol=0)):
                raise ValueError(f'frame {number}: GT 几何帧/时间与预测形变不匹配。')
            path = args.gt_depth_dir/entry['file']
            if file_sha256(path) != entry['sha256']:
                raise ValueError(f'GT 文件哈希不符：{path}')
            with np.load(path, allow_pickle=False) as data:
                gt_depth = data['depth_z']
                hit_valid = data['valid']
                if not np.allclose(data['camera_to_world'], c2w, atol=1e-5, rtol=0) or not np.allclose(data['intrinsics'], intrinsics, atol=1e-5, rtol=0):
                    raise ValueError('GT 深度与当前相机/内参不符。')
                if gt_depth.shape != alpha.shape:
                    raise ValueError('GT 深度分辨率不符；请重新生成，不能直接 resize 深度。')
            run.files[str(path.resolve())] = entry['sha256']
        prediction = run.render(camera, c2w, intrinsics)
        if mesh_manifest is not None:
            from utils.predicted_mesh import PredictedMesh
            if mesh_manifest.get('static_scene'):
                mesh_name, mesh_hash = mesh_manifest['mesh'], mesh_manifest['mesh_sha256']
            else:
                entry = mesh_manifest['frames'][str(number)]
                if not np.isclose(entry['time'], record['time'], atol=1e-7, rtol=0):
                    raise ValueError('逐帧预测 Mesh 与评测帧时间不同。')
                mesh_name, mesh_hash = entry['mesh'], entry['mesh_sha256']
            mesh_path = (args.pred_mesh_manifest.parent/mesh_name).resolve()
            if mesh_path not in loaded_meshes:
                if file_sha256(mesh_path) != mesh_hash:
                    raise ValueError('预测 Mesh 文件哈希不符。')
                loaded_meshes[mesh_path] = PredictedMesh(mesh_path)
                run.files[str(mesh_path)] = mesh_hash
            prediction.update(loaded_meshes[mesh_path].render(c2w, intrinsics, *alpha.shape))
        masks = dict(foreground=(alpha >= .5) & hit_valid,
                     interior=erode((alpha >= .99) & hit_valid, args.interior_radius))
        regions = {name: evaluate_region(prediction, gt_normal, gt_valid, gt_depth, mask, args.alpha_threshold)
                   for name, mask in masks.items()}
        predicted_mask = prediction['alpha'] >= args.alpha_threshold
        foreground = alpha >= .5
        iou = float((predicted_mask & foreground).sum()/max(1, (predicted_mask | foreground).sum()))
        rows.append(dict(source_frame=number, split=split, time=record['time'], alpha_iou=iou, regions=regions))
        if args.save_arrays:
            np.savez_compressed(args.output_dir/f'prediction_{number:04d}.npz',
                                **prediction, gt_normal=gt_normal, gt_depth=gt_depth if gt_depth is not None else np.zeros_like(alpha),
                                gt_foreground=masks['foreground'])
        if index in {0, len(frames)//2, len(frames)-1}:
            save_preview(args.output_dir/f'comparison_{number:04d}.png', prediction, gt_normal, gt_depth, masks['foreground'])
        del camera
        stat = regions['interior']
        named = ', '.join(f'{i["label"]}={i["angular_deg"]["mean"]} deg / coverage={i["coverage"]}'
                          for i in stat['normal_metrics'].values())
        print(f'{number:04d}: {named}', flush=True)
    repository = Path(__file__).resolve().parents[1]
    report = dict(format_version=2, command={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        model_path=str(run.model), iteration=args.load_iter, geometry_model=str(run.geometry_model),
        geometry_iteration=run.geometry_iteration, mode=run.mode, gaussian_count=len(run.pc.get_xyz),
        protocol=dict(normal='unit world-space Blender EXR vs normalized rasterized world normals; signed angular MAE in degrees',
            normal_axis='acos(abs(dot)), supplementary unoriented surface error',
            normal_component='mean absolute XYZ component error on unit normals',
            depth='positive camera Z in original world units; no scale, shift, ICP or per-frame alignment',
            foreground='GT alpha >= 0.5 and GT ray hit when available',
            interior=f'GT alpha >= 0.99, eroded {args.interior_radius} evaluation pixels',
            visibility=f'predicted alpha >= {args.alpha_threshold}; invalid predictions excluded from conditional MAE and counted in coverage',
            depth_derived_normal='centered finite differences of expected Z depth with pixel center +0.5',
            normal_metrics=dict(
                gs_render='GS Render Normal MAE: normalized alpha/transmittance blended surfel normals vs GT Normal EXR',
                depth_normal='Depth Normal MAE: expected Z -> world points -> finite-difference normals vs the same GT Normal EXR',
                mesh='Mesh Normal MAE: predicted triangle mesh -> first ray hit -> interpolated geometry-derived vertex normal; camera-facing; vs the same GT Normal EXR',
                missing_mesh='not_available/null when no predicted mesh; never substitute GT mesh/EXR consistency error',
                common_pixels='common_normal_metrics scores available normal methods on the identical valid-pixel intersection',
                legacy='normal.angular_deg and normal_coverage are aliases of GS Render Normal; depth_derived_angular_deg is Depth Normal'),
            aggregation='frame_mean: equal frame weights; pixel_mean: valid pixel weights; never average frame medians as a global median',
            primary='foreground; interior is a supplementary boundary diagnostic; no object or material subsets'),
        unit_system=depth_manifest['unit_system'] if depth_manifest else None,
        meters_per_world_unit=depth_manifest['meters_per_world_unit'] if depth_manifest else None,
        summary=aggregate_regions(rows), alpha_iou_frame_mean=float(np.mean([r['alpha_iou'] for r in rows])),
        frames=rows, sources=run.files, elapsed_seconds=time.monotonic()-start,
        predicted_mesh=dict(manifest=str(args.pred_mesh_manifest), extraction=mesh_manifest) if mesh_manifest else None,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repository).decode().strip(),
        code_status=subprocess.check_output(['git','status','--short'], cwd=repository).decode())
    geometry = getattr(run.pc, 'stage2_geometry', None)
    if geometry is not None and geometry.parent_index is not None:
        report['source_gaussian_count'] = len(run.pc.get_xyz)
        report['gaussian_count'] = report['rendered_gaussian_count'] = len(geometry.translation)
    report['code_files'] = {str(p.relative_to(repository)):file_sha256(p) for p in (Path(__file__), repository/'utils/geometry_eval_utils.py')}
    if mesh_manifest:
        p = repository/'utils/predicted_mesh.py'
        report['code_files'][str(p.relative_to(repository))] = file_sha256(p)
    # 读入的所有来源（包含模型、相机、EXR、GT）在整个评测中保持不变。
    for path, digest in run.files.items():
        if file_sha256(path) != digest:
            raise AssertionError(f'评测过程中来源改变：{path}')
    (args.output_dir/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    save_metric_tables(args.output_dir, report['summary'], rows,
                       report['meters_per_world_unit'] if report['unit_system'] == 'METRIC' else None)
    (args.output_dir/'EVAL_COMPLETED').write_text('PASS: evaluation pipeline completed; no reconstruction quality threshold asserted\n')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
