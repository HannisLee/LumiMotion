"""仅融合训练视角的模型预测 Z 深度，提取静态场景的 TSDF Mesh。

不读取 GT 网格、GT 法线或 GT 深度；动态场景需逐时刻重建，不能使用本入口。
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.eval_geometry_gt import GeometryRun
from utils.geometry_eval_utils import file_sha256, frame_number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--load_iter', type=int, required=True)
    parser.add_argument('--source_path', type=Path)
    parser.add_argument('--static_scene', action='store_true', required=True,
                        help='显式确认场景静态；不允许跨时刻融合动态物体。')
    parser.add_argument('--resolution', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--voxel_size', type=float, default=.02)
    parser.add_argument('--sdf_trunc', type=float, default=.08)
    parser.add_argument('--depth_trunc', type=float, default=20.)
    parser.add_argument('--alpha_threshold', type=float, default=.5)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    if not (args.voxel_size > 0 and args.sdf_trunc > args.voxel_size and
            args.depth_trunc > 0 and 0 < args.alpha_threshold < 1):
        parser.error('体素、截断距离或 alpha 阈值不合法。')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import open3d as o3d
    start = time.monotonic()
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    transforms = run.source/'transforms_train.json'
    metadata = json.loads(transforms.read_text())
    run.files[str(transforms)] = file_sha256(transforms)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    basis = np.diag([1., -1., -1., 1.])
    frames = []
    for record in metadata['frames']:
        number = int(record.get('source_frame', frame_number(record['file_path'])))
        camera, _, intrinsic, _ = run.camera(metadata, record, args.resolution)
        c2w = np.array(record['transform_matrix'], dtype=np.float64)
        prediction = run.render(camera, c2w, intrinsic)
        depth = prediction['expected'].astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth < args.depth_trunc) & (
                prediction['alpha'] >= args.alpha_threshold)
        depth = np.where(valid, depth, 0).astype(np.float32)
        height, width = depth.shape
        # Open3D 使用整数像素索引；cx/cy 减 0.5 等价于原渲染的像素中心 +0.5。
        fx, fy, cx, cy = intrinsic
        k = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx-.5, cy-.5)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), np.uint8)),
            o3d.geometry.Image(depth), depth_scale=1., depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, k, basis @ np.linalg.inv(c2w))
        frames.append(dict(source_frame=number, time=record['time'], split='train',
                           integrated_pixels=int(valid.sum())))
        print(f'TSDF train {number:04d}: {valid.sum()} predicted pixels', flush=True)
        del camera
    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not mesh.has_triangles():
        raise ValueError('预测深度融合未产生三角网格。')
    mesh_path = args.output_dir/'predicted_mesh.ply'
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=False):
        raise IOError('预测 Mesh 写入失败。')
    for path, digest in run.files.items():
        if file_sha256(path) != digest:
            raise AssertionError(f'来源改变：{path}')
    manifest = dict(format_version=1, kind='predicted_mesh', static_scene=True,
        model_path=str(run.model), iteration=args.load_iter, source_path=str(run.source),
        mesh=mesh_path.name, mesh_sha256=file_sha256(mesh_path),
        method='Open3D 0.18 ScalableTSDFVolume; expected Z; training cameras only',
        gt_geometry_used=False, fit_scale_or_pose=False, sources=run.files, frames=frames,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        pixel_center=.5, world_units='same units as training cameras',
        vertices=len(mesh.vertices), triangles=len(mesh.triangles), elapsed_seconds=time.monotonic()-start)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (args.output_dir/'MESH_COMPLETED').write_text('PASS: prediction-only static mesh extracted\n')
    print(f'Mesh: {len(mesh.vertices)} vertices / {len(mesh.triangles)} triangles', flush=True)


if __name__ == '__main__':
    main()
