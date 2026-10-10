"""用 Blender 后台导出逐帧可见网格；不保存或修改源 blend 文件。

blender --factory-startup --disable-autoexec -b scene.blend \
  --python scripts/export_blender_geometry.py -- --camera_json camera.json \
  --frames 8 16 --output_dir output/.../meshes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_frame(scene):
    # 依赖图按 viewport 求值，因此在内存中使用 render 的 modifier 开关。
    for obj in scene.objects:
        for modifier in obj.modifiers:
            modifier.show_viewport = modifier.show_render
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles, normals, object_ids, names = [], [], [], [], []
    offset = 0
    for obj in sorted(scene.objects, key=lambda o: o.name):
        if obj.type != 'MESH' or obj.hide_render:
            continue
        if obj.users_collection and all(c.hide_render for c in obj.users_collection):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                continue
            matrix = np.array(evaluated.matrix_world, dtype=np.float64)
            local = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
            world = local @ matrix[:3, :3].T + matrix[:3, 3]
            faces = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int32)
            loops = np.array([t.loops[:] for t in mesh.loop_triangles], dtype=np.int32)
            corner = np.array([n.vector[:] for n in mesh.corner_normals], dtype=np.float64)
            corner = corner[loops] @ np.linalg.inv(matrix[:3, :3])
            corner /= np.maximum(np.linalg.norm(corner, axis=-1, keepdims=True), 1e-12)
            vertices.append(world.astype(np.float32))
            triangles.append(faces + offset)
            normals.append(corner.astype(np.float32))
            object_ids.append(np.full(len(faces), len(names), dtype=np.int32))
            names.append(obj.name)
            offset += len(world)
        finally:
            evaluated.to_mesh_clear()
    if not vertices:
        raise ValueError('没有可渲染网格。')
    return dict(vertices=np.concatenate(vertices), triangles=np.concatenate(triangles),
                corner_normals=np.concatenate(normals), object_ids=np.concatenate(object_ids),
                object_names=np.array(names))


def write_mesh_ply(path, data):
    """保留三角面、世界坐标、逐角点法线和物体 ID，避免点云近似深度。"""
    points = data['vertices'][data['triangles']].reshape(-1, 3)
    normals = data['corner_normals'].reshape(-1, 3)
    vertex = np.empty(len(points), dtype=[(k, '<f4') for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')] + [('object_id', '<i4')])
    for i, key in enumerate(('x', 'y', 'z')):
        vertex[key] = points[:, i]
    for i, key in enumerate(('nx', 'ny', 'nz')):
        vertex[key] = normals[:, i]
    vertex['object_id'] = np.repeat(data['object_ids'], 3)
    face = np.empty(len(data['triangles']), dtype=[('count', 'u1'), ('indices', '<i4', (3,)), ('object_id', '<i4')])
    face['count'] = 3
    face['indices'] = np.arange(len(points)).reshape(-1, 3)
    face['object_id'] = data['object_ids']
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Blender evaluated world mesh; split vertices preserve corner normals\n'
              f'element vertex {len(vertex)}\n' + ''.join(f'property float {k}\n' for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')) +
              'property int object_id\n' + f'element face {len(face)}\n'
              'property list uchar int vertex_indices\nproperty int object_id\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(vertex.tobytes())
        stream.write(face.tobytes())


def scene_audit(scene):
    materials = {}
    for material in bpy.data.materials:
        nodes = []
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type in ('BSDF_PRINCIPLED', 'BSDF_DIFFUSE'):
                    socket = node.inputs.get('Base Color') or node.inputs.get('Color')
                    nodes.append(dict(type=node.type, color=list(socket.default_value), linked=socket.is_linked))
        materials[material.name] = nodes
    return dict(materials=materials,
                lights=[dict(name=o.name, type=o.data.type, energy=o.data.energy, color=list(o.data.color))
                        for o in scene.objects if o.type == 'LIGHT'],
                view_transform=scene.view_settings.view_transform,
                exposure=scene.view_settings.exposure, gamma=scene.view_settings.gamma,
                shape_keys={o.name: [dict(name=k.name, value=k.value) for k in o.data.shape_keys.key_blocks]
                            for o in scene.objects if o.type == 'MESH' and o.data.shape_keys is not None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera_json', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--export_ply', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(bpy.data.filepath).resolve()
    initial_hash = sha256(source)
    cameras = json.loads(args.camera_json.read_text())['frames']
    scene = bpy.context.scene
    original_frame = scene.frame_current
    report = dict(source_blend=str(source), source_blend_sha256=initial_hash,
                  blender_version=bpy.app.version_string, original_frame=original_frame,
                  unit_system=scene.unit_settings.system,
                  meters_per_world_unit=scene.unit_settings.scale_length,
                  camera_json_sha256=sha256(args.camera_json), frames={}, meshes={})
    report['scene_audit'] = scene_audit(scene)
    unique = {}
    for frame in sorted(set(args.frames)):
        scene.frame_set(frame)
        expected = np.array(cameras[f'{frame:04d}']['extrinsics']['camera_to_world'])
        error = float(np.max(np.abs(np.array(scene.camera.matrix_world) - expected)))
        if error > 1e-4:
            raise ValueError(f'frame {frame}: blend 相机与导出元数据不符 ({error})。')
        data = export_frame(scene)
        ply_record = {}
        if args.export_ply:
            ply_path = args.output_dir / f'mesh_{frame:04d}.ply'
            write_mesh_ply(ply_path, data)
            ply_record = dict(ply=ply_path.name, ply_sha256=sha256(ply_path))
        digest = hashlib.sha256()
        for key in sorted(data):
            digest.update(key.encode())
            digest.update(data[key].tobytes())
        geometry_hash = digest.hexdigest()
        if geometry_hash not in unique:
            filename = f'mesh_{frame:04d}.npz'
            np.savez_compressed(args.output_dir / filename, **data)
            unique[geometry_hash] = filename
            report['meshes'][filename] = dict(sha256=sha256(args.output_dir / filename),
                vertices=len(data['vertices']), triangles=len(data['triangles']),
                objects=data['object_names'].tolist())
        report['frames'][str(frame)] = dict(mesh=unique[geometry_hash],
            geometry_sha256=geometry_hash, camera_max_error=error,
            scene_frame=scene.frame_current, **ply_record)
        print(f'frame {frame}: {unique[geometry_hash]}, camera error {error:.3g}', flush=True)
    if sha256(source) != initial_hash:
        raise AssertionError('源 blend 文件发生变化。')
    report['static_geometry'] = len(unique) == 1
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
