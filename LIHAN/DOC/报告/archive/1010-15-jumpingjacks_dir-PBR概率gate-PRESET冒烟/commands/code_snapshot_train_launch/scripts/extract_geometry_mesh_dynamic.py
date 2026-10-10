"""动态逐时刻 TSDF Mesh：每个时刻只融合训练相机下的模型预测深度。

测试 transforms 只提供待评测帧号和时间，不使用测试相机、GT网格或GT深度重建。
每个时刻新建独立TSDF，禁止跨动作融合。体素与截断参数预先固定，不按GT误差调整。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.eval_geometry_gt import GeometryRun
from utils.geometry_eval_utils import file_sha256, frame_number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--load_iter', type=int, required=True)
    parser.add_argument('--source_path', type=Path)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--frames', type=int, nargs='+')
    parser.add_argument('--resolution', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--voxel_size', type=float, default=.02)
    parser.add_argument('--sdf_trunc', type=float, default=.08)
    parser.add_argument('--depth_trunc', type=float, default=20.)
    parser.add_argument('--alpha_threshold', type=float, default=.5)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    if not (args.voxel_size > 0 and args.sdf_trunc > args.voxel_size and
            args.depth_trunc > 0 and 0 < args.alpha_threshold < 1):
        parser.error('体素、截断距离或alpha阈值不合法。')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import open3d as o3d
    import torch
    start = time.monotonic()
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    training_path = run.source / 'transforms_train.json'
    targets_path = run.source / ('transforms_' + args.split + '.json')
    train = json.loads(training_path.read_text())
    targets = json.loads(targets_path.read_text())['frames']
    if args.frames:
        targets = [r for r in targets if int(r.get('source_frame', frame_number(r['file_path']))) in args.frames]
        if {int(r.get('source_frame', frame_number(r['file_path']))) for r in targets} != set(args.frames):
            raise ValueError('请求帧不在目标split中。')
    if not targets or not train['frames']:
        raise ValueError('重建相机或目标时刻为空。')
    run.files.update({str(p): file_sha256(p) for p in (training_path, targets_path)})
    cameras = []
    for record in train['frames']:
        camera, _, intrinsic, _ = run.camera(train, record, args.resolution)
        c2w = np.asarray(record['transform_matrix'], dtype=np.float64)
        fx, fy, cx, cy = intrinsic
        k = o3d.camera.PinholeCameraIntrinsic(camera.image_width, camera.image_height,
                                            fx, fy, cx - .5, cy - .5)
        cameras.append((record, camera, intrinsic, c2w, k))
    basis = np.diag([1., -1., -1., 1.])
    report = dict(format_version=1, kind='predicted_mesh', static_scene=False,
                  model_path=str(run.model), iteration=args.load_iter, source_path=str(run.source),
                  method='Open3D ScalableTSDFVolume; expected Z; train camera poses at fixed target time',
                  gt_geometry_used=False, test_camera_used_for_reconstruction=False,
                  fit_scale_or_pose=False, sources=run.files, frames={}, pixel_center=.5,
                  world_units='same units as training cameras',
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    for record in sorted(targets, key=lambda r: int(r.get('source_frame', frame_number(r['file_path'])))):
        number = int(record.get('source_frame', frame_number(record['file_path'])))
        target_time = float(record['time'])
        volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=args.voxel_size,
            sdf_trunc=args.sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
        integrations = []
        for train_record, camera, intrinsic, c2w, k in cameras:
            camera.fid = torch.tensor([target_time], dtype=torch.float32, device='cuda')
            prediction = run.render(camera, c2w, intrinsic)
            depth = prediction['expected'].astype(np.float32)
            valid = np.isfinite(depth) & (depth > 0) & (depth < args.depth_trunc) & (
                prediction['alpha'] >= args.alpha_threshold)
            depth = np.where(valid, depth, 0).astype(np.float32)
            height, width = depth.shape
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((height, width, 3), np.uint8)), o3d.geometry.Image(depth),
                depth_scale=1., depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
            volume.integrate(rgbd, k, basis @ np.linalg.inv(c2w))
            integrations.append(dict(source_frame=int(train_record.get('source_frame', frame_number(train_record['file_path']))),
                                     split='train', render_time=target_time, integrated_pixels=int(valid.sum())))
        mesh = volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        if not mesh.has_triangles() or not np.isfinite(np.asarray(mesh.vertices)).all():
            raise ValueError('无有效逐时刻预测网格：' + str(number))
        path = args.output_dir / ('predicted_mesh_%04d.ply' % number)
        if not o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=False):
            raise IOError('Mesh写入失败：' + str(path))
        report['frames'][str(number)] = dict(source_frame=number, time=target_time, mesh=path.name,
            mesh_sha256=file_sha256(path), vertices=len(mesh.vertices), triangles=len(mesh.triangles),
            integrations=integrations)
        print('Mesh frame %04d time %.8f: %d vertices / %d triangles; %d train poses' %
              (number, target_time, len(mesh.vertices), len(mesh.triangles), len(integrations)), flush=True)
        del volume, mesh
    for path, digest in run.files.items():
        if file_sha256(path) != digest:
            raise AssertionError('来源改变：' + path)
    report['elapsed_seconds'] = time.monotonic() - start
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'MESH_COMPLETED').write_text('PASS: independent prediction-only meshes at each target time\n')


if __name__ == '__main__':
    main()
