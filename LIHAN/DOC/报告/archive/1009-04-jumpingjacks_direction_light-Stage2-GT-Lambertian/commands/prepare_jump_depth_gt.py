"""从 Blender 导出的三角网格生成像素中心 Z 深度真值（CPU，无光照）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.geometry_eval_utils import (camera_rays, erode, file_sha256, frame_number,
                                       load_normal_exr, normal_errors, summarize, unit_normal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh_manifest', type=Path, required=True)
    parser.add_argument('--gt_root', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip_baseline_audit', action='store_true',
                        help='旧 canonical baseline 不作为逐帧真值；只使用新导出的 evaluated mesh。')
    args = parser.parse_args()
    import open3d as o3d
    from plyfile import PlyData

    manifest = json.loads(args.mesh_manifest.read_text())
    camera_path = args.gt_root / 'camera.json'
    if file_sha256(camera_path) != manifest['camera_json_sha256']:
        raise ValueError('相机元数据与导出的网格不匹配。')
    if file_sha256(manifest['source_blend']) != manifest['source_blend_sha256']:
        raise ValueError('源 blend 已改变，需重新导出。')
    camera_metadata = json.loads(camera_path.read_text())
    resolution = camera_metadata['resolution']
    width = round(resolution['width'] / args.resolution)
    height = round(resolution['height'] / args.resolution)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(format_version=1, depth_convention='positive camera Z; unnormalized pinhole rays',
                  pixel_center=.5, width=width, height=height,
                  unit_system=manifest['unit_system'], meters_per_world_unit=manifest['meters_per_world_unit'],
                  mesh_manifest=str(args.mesh_manifest.resolve()), mesh_manifest_sha256=file_sha256(args.mesh_manifest),
                  camera_json=str(camera_path.resolve()), camera_json_sha256=file_sha256(camera_path),
                  transforms=str(args.transforms.resolve()), transforms_sha256=file_sha256(args.transforms),
                  static_geometry=manifest['static_geometry'], frames={}, mesh_audits={},
                  baseline_audit_skipped=args.skip_baseline_audit)
    loaded_meshes = {}
    baseline = args.gt_root / 'baseline.ply'
    for record in json.loads(args.transforms.read_text())['frames']:
        frame = int(record.get('source_frame', frame_number(record['file_path'])))
        frame_mesh = manifest['frames'][str(frame)]['mesh']
        if frame_mesh not in loaded_meshes:
            path = args.mesh_manifest.parent / frame_mesh
            if file_sha256(path) != manifest['meshes'][frame_mesh]['sha256']:
                raise ValueError(f'网格哈希不符：{path}')
            mesh = dict(np.load(path, allow_pickle=False))
            scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
            scene.add_triangles(o3d.core.Tensor(mesh['vertices'], dtype=o3d.core.Dtype.Float32),
                                o3d.core.Tensor(mesh['triangles'].astype(np.uint32), dtype=o3d.core.Dtype.UInt32))
            loaded_meshes[frame_mesh] = (scene, mesh)
            if baseline.exists() and manifest['static_geometry'] and not args.skip_baseline_audit:
                ply = PlyData.read(baseline)['vertex']
                points = np.stack([ply[k] for k in ('x', 'y', 'z')], axis=-1)[::3]
                distances = scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)), nthreads=args.threads).numpy()
                audit = summarize(distances, np.ones(len(distances), bool))
                audit['max'] = float(distances.max())
                audit['baseline_sha256'] = file_sha256(baseline)
                report['mesh_audits'][frame_mesh] = audit
                if audit['p95'] > 1e-4:
                    raise ValueError(f'导出网格与 baseline 表面不符：P95={audit["p95"]}')
        scene, mesh = loaded_meshes[frame_mesh]
        ply_audit = None
        ply_entry = manifest['frames'][str(frame)]
        if 'ply' in ply_entry:
            ply_path = args.mesh_manifest.parent / ply_entry['ply']
            if file_sha256(ply_path) != ply_entry['ply_sha256']:
                raise ValueError(f'导出 PLY 哈希不符：{ply_path}')
            ply = PlyData.read(ply_path)
            vertices = np.stack([ply['vertex'][k] for k in ('x', 'y', 'z')], axis=-1)
            normals_ply = np.stack([ply['vertex'][k] for k in ('nx', 'ny', 'nz')], axis=-1)
            triangles_ply = np.stack(ply['face']['vertex_indices']).astype(np.uint32)
            if (not np.array_equal(vertices[triangles_ply], mesh['vertices'][mesh['triangles']]) or
                    not np.array_equal(normals_ply[triangles_ply], mesh['corner_normals']) or
                    not np.array_equal(ply['face']['object_id'], mesh['object_ids'])):
                raise ValueError('重新导出的 PLY 与 evaluated mesh 不一致。')
            # 实际 GT 射线求交使用对应帧 PLY 三角面，而不是旧 baseline 点云。
            scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
            scene.add_triangles(o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
                                o3d.core.Tensor(triangles_ply, dtype=o3d.core.Dtype.UInt32))
            ply_audit = dict(file=str(ply_path.resolve()), sha256=ply_entry['ply_sha256'],
                             exact_mesh_match=True, vertices=len(vertices), triangles=len(triangles_ply))
        camera = camera_metadata['frames'][f'{frame:04d}']
        c2w = np.array(camera['extrinsics']['camera_to_world'], dtype=np.float64)
        intrinsic = camera['intrinsics']
        sx, sy = width / resolution['width'], height / resolution['height']
        intrinsics = np.array([intrinsic['fx']*sx, intrinsic['fy']*sy,
                               intrinsic['cx']*sx, intrinsic['cy']*sy])
        if not np.allclose(c2w, record['transform_matrix'], atol=1e-5, rtol=0):
            raise ValueError(f'frame {frame}: transforms 相机与 GT 相机不同。')
        rays = camera_rays(c2w, intrinsics, height, width)
        hits = scene.cast_rays(o3d.core.Tensor(rays), nthreads=args.threads)
        depth_z = hits['t_hit'].numpy()
        valid = np.isfinite(depth_z) & (depth_z > 0)
        triangles = hits['primitive_ids'].numpy()
        triangle_index = np.where(valid, triangles, 0).astype(np.int64)
        uv = hits['primitive_uvs'].numpy()
        weights = np.stack((1-uv.sum(-1), uv[..., 0], uv[..., 1]), axis=-1)
        normal = (mesh['corner_normals'][triangle_index] * weights[..., None]).sum(-2)
        normal, _ = unit_normal(normal)
        normal[~valid] = 0
        objects = mesh['object_ids'][triangle_index].astype(np.int32)
        objects[~valid] = -1
        depth_z = np.where(valid, depth_z, 0)
        depth_ray = depth_z * np.linalg.norm(rays[..., 3:], axis=-1)
        exr = args.gt_root / 'normal_exr' / f'normal_{frame:04d}.exr'
        exr_normal, exr_valid = load_normal_exr(exr, height, width)
        alpha_path = args.gt_root / 'albedo' / f'albedo_{frame:04d}.png'
        if not alpha_path.exists():
            candidates = [p for p in (args.gt_root / 'albedo').glob('*.png') if frame_number(p) == frame]
            if len(candidates) != 1:
                raise FileNotFoundError(f'frame {frame}: 找不到唯一 albedo alpha。')
            alpha_path = candidates[0]
        alpha = np.asarray(Image.open(alpha_path).convert('RGBA').getchannel('A').resize((width, height))) / 255.
        errors, error_valid = normal_errors(normal, exr_normal)
        interior = erode((alpha >= .99) & valid & exr_valid, 2) & error_valid
        alpha_mask = alpha >= .5
        iou = float((alpha_mask & valid).sum()/max(1, (alpha_mask | valid).sum()))
        audit = dict(alpha_hit_iou=iou, valid_pixels=int(valid.sum()),
                     exr_mesh_normal_deg=summarize(errors['angular_deg'], interior),
                     exr_mesh_normal_axis_deg=summarize(errors['axis_deg'], interior),
                     exr_sha256=file_sha256(exr), alpha_sha256=file_sha256(alpha_path))
        if iou < .97:
            raise ValueError(f'frame {frame}: GT 网格与 alpha 不匹配，IoU={iou:.4f}')
        # 此数据EXR明确为包含法线贴图的shading Normal；几何/着色法线差作为诊断，不否决相机对齐的几何Z深度。
        if audit['exr_mesh_normal_axis_deg']['mean'] is None:
            raise ValueError(f'frame {frame}: 没有可核查的GT几何/着色法线交集')
        audit['normal_reference_semantics'] = 'Blender shading Normal may include material normal maps; geometry/EXR error is diagnostic only'
        audit['shading_normal_not_geometry'] = True
        filename = f'gt_{frame:04d}.npz'
        np.savez_compressed(args.output_dir / filename, depth_z=depth_z, depth_ray=depth_ray,
                            normal_world=normal.astype(np.float32), valid=valid, object_id=objects,
                            object_names=mesh['object_names'], intrinsics=intrinsics, camera_to_world=c2w)
        report['frames'][str(frame)] = dict(file=filename, sha256=file_sha256(args.output_dir/filename),
                                          mesh=frame_mesh, audit=audit, ply=ply_audit,
                                          source_frame=frame, time=record['time'])
        print(f'GT {frame:04d}: IoU={iou:.5f}, mesh/EXR normal MAE={audit["exr_mesh_normal_deg"]["mean"]:.4f} deg', flush=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
