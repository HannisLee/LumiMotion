#!/usr/bin/env python3
"""隔离生成V3评测几何、分区与正camera-Z；不读取灯光或材质节点。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = ROOT / 'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/eval_only'
SOURCE = ROOT / 'data/LH-data/static/LH-static-jumpingjacks-ELDL-V3'
TOOLS = ROOT / 'output/archive/1011-07-六组训练总控/tools'
HUMAN = ('Character_Mesh_Ch02_Body', 'Character_Mesh_Ch02_Cloth',
         'Character_Mesh_Ch02_Eyelashes', 'Character_Mesh_Ch02_Hair',
         'Character_Mesh_Ch02_Sneakers', 'Character_Mesh_Ch02_Socks')
GROUND = ('Reference_Checkered_Platform', 'Reference_Platform_Base')
LABELS = {'unknown': 0, 'human_other': 1, 'leg': 2, 'foot': 3, 'shoe': 4, 'platform': 5}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def guard(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ALLOWED.resolve()):
        raise ValueError('输出仅允许位于预登记eval_only目录。')
    return path


def bone_class(name):
    name = re.sub(r'[^a-z]', '', name.split(':')[-1].lower())
    if re.fullmatch(r'(left|right|l|r)?(foot|toe|toebase|toes)', name):
        return 'foot'
    if re.fullmatch(r'(left|right|l|r)?(upleg|leg|thigh|calf)', name):
        return 'leg'
    return None


def worker(args):
    """仅由Blender后台调用；不调用旧导出器的scene_audit。"""
    import bpy
    import numpy as np
    out = guard(args.output)
    camera_path = args.source / 'camera.json'
    cameras = json.loads(camera_path.read_text())['frames']
    blend = Path(bpy.data.filepath).resolve()
    original_hash = digest(blend)
    scene = bpy.context.scene
    # 仅对内存中的求值模式做render/viewport一致化，不保存blend。
    for obj in scene.objects:
        for mod in obj.modifiers:
            mod.show_viewport = mod.show_render
    report = dict(format_version=1, evaluation_only=True, source_blend=str(blend),
                  source_blend_sha256=original_hash, camera_json_sha256=digest(camera_path),
                  blender_version=bpy.app.version_string, unit_system=scene.unit_settings.system,
                  meters_per_world_unit=scene.unit_settings.scale_length,
                  color_management=dict(view_transform=scene.view_settings.view_transform,
                      look=scene.view_settings.look, exposure=scene.view_settings.exposure,
                      gamma=scene.view_settings.gamma),
                  material_or_light_values_exported=False, frames={}, objects={}, meshes={})
    unique = {}
    for frame in range(1, 151):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        expected = np.asarray(cameras[f'{frame:04d}']['extrinsics']['camera_to_world'])
        error = float(np.max(np.abs(np.asarray(scene.camera.matrix_world) - expected)))
        if error > 1e-4:
            raise ValueError(f'相机元数据不一致，帧{frame}。')
        dg = bpy.context.evaluated_depsgraph_get()
        verts, faces, normals, ids, semantics, names = [], [], [], [], [], []
        offset = 0
        for obj in sorted(scene.objects, key=lambda o: o.name):
            if obj.type != 'MESH' or obj.hide_render:
                continue
            if obj.users_collection and all(c.hide_render for c in obj.users_collection):
                continue
            evaluated = obj.evaluated_get(dg)
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
            try:
                mesh.calc_loop_triangles()
                if not len(mesh.loop_triangles):
                    continue
                matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
                world = np.asarray([v.co[:] for v in mesh.vertices]) @ matrix[:3, :3].T + matrix[:3, 3]
                tri = np.asarray([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int32)
                loops = np.asarray([t.loops[:] for t in mesh.loop_triangles])
                corner = np.asarray([n.vector[:] for n in mesh.corner_normals])[loops] @ np.linalg.inv(matrix[:3, :3])
                corner /= np.maximum(np.linalg.norm(corner, axis=-1, keepdims=True), 1e-12)
                label = np.full(len(tri), LABELS['unknown'], dtype=np.uint8)
                matched = {g.index: bone_class(g.name) for g in obj.vertex_groups}
                matched = {k: v for k, v in matched.items() if v}
                if obj.name in GROUND:
                    label[:] = LABELS['platform']
                elif obj.name in HUMAN:
                    label[:] = LABELS['human_other']
                    if obj.name.endswith('_Sneakers'):
                        label[:] = LABELS['shoe']
                    elif obj.name.endswith('_Socks'):
                        label[:] = LABELS['foot']
                    elif obj.name.endswith(('_Body', '_Cloth')):
                        weights = np.zeros((len(mesh.vertices), 3))
                        for v in mesh.vertices:
                            for group in v.groups:
                                weights[v.index, 0] += group.weight
                                if group.group in matched:
                                    weights[v.index, 1 if matched[group.group] == 'leg' else 2] += group.weight
                        sums = weights[tri].sum(axis=1)
                        ratio = sums[:, 1:] / np.maximum(sums[:, :1], 1e-12)
                        label[ratio[:, 0] >= .5] = LABELS['leg']
                        label[ratio[:, 1] >= .5] = LABELS['foot']
                if frame == 1:
                    report['objects'][obj.name] = dict(
                        object_id=len(names), triangles=len(tri),
                        matched_bone_groups={g.name: bone_class(g.name) for g in obj.vertex_groups if bone_class(g.name)},
                        semantic_triangle_counts={k: int((label == v).sum()) for k, v in LABELS.items()})
                verts.append(world.astype(np.float32)); faces.append(tri + offset)
                normals.append(corner.astype(np.float32))
                ids.append(np.full(len(tri), len(names), np.int32)); semantics.append(label)
                names.append(obj.name); offset += len(world)
            finally:
                evaluated.to_mesh_clear()
        if not verts:
            raise ValueError('没有可见评测网格。')
        data = dict(vertices=np.concatenate(verts), triangles=np.concatenate(faces),
                    corner_normals=np.concatenate(normals), object_ids=np.concatenate(ids),
                    semantic_ids=np.concatenate(semantics), object_names=np.asarray(names))
        h = hashlib.sha256()
        for key in sorted(data):
            h.update(key.encode()); h.update(data[key].tobytes())
        geometry_hash = h.hexdigest()
        if geometry_hash not in unique:
            filename = f'mesh_{frame:04d}.npz'
            np.savez_compressed(out / 'meshes' / filename, **data)
            unique[geometry_hash] = filename
            report['meshes'][filename] = dict(sha256=digest(out / 'meshes' / filename),
                vertices=len(data['vertices']), triangles=len(data['triangles']))
        report['frames'][str(frame)] = dict(mesh=unique[geometry_hash], geometry_sha256=geometry_hash,
                                           camera_max_error=error, scene_frame=scene.frame_current)
        if frame in (1, 10, 80, 150):
            print(f'只读几何核查 {frame:04d}，唯一几何数{len(unique)}', flush=True)
    report['static_geometry'] = len(unique) == 1
    if digest(blend) != original_hash:
        raise AssertionError('源blend变化。')
    write(out / 'meshes/manifest.json', report)
    if not report['static_geometry'] and not getattr(args, 'allow_dynamic', False):
        raise ValueError('全部150帧evaluated几何并非完全静态；保留现场。')


def prepare(args, out):
    import numpy as np
    import cv2
    import open3d as o3d
    sys.path.insert(0, str(ROOT))
    from utils.geometry_eval_utils import camera_rays, load_normal_exr, unit_normal, normal_errors, erode, summarize
    manifest = json.loads((out / 'meshes/manifest.json').read_text())
    cameras = json.loads((args.source / 'camera.json').read_text())
    width = round(cameras['resolution']['width'] / 2)
    height = round(cameras['resolution']['height'] / 2)
    if not manifest['static_geometry'] and not getattr(args, 'allow_dynamic', False):
        raise ValueError('静态入口禁止读取动态manifest，必须使用独立动态入口。')
    # 只缓存当前实际网格；动态150帧不常驻150个BVH。
    previous_mesh = None
    report = dict(format_version=1, evaluation_only=True,
                  depth_convention='positive camera Z; unnormalized pinhole rays', pixel_center=.5,
                  width=width, height=height, resolution=2, static_geometry=manifest['static_geometry'],
                  unit_system=manifest['unit_system'], meters_per_world_unit=manifest['meters_per_world_unit'],
                  mesh_manifest=str(out / 'meshes/manifest.json'),
                  mesh_manifest_sha256=digest(out / 'meshes/manifest.json'),
                  semantic_labels=LABELS, frames={}, source_sha256={})
    composites = dict(human=list(HUMAN), platform=list(GROUND),
                      leg='semantic_id=2', foot='semantic_id=3', shoe='semantic_id=4',
                      footwear='semantic_id in {3,4}')
    anatomy = {k: sum(o['semantic_triangle_counts'][k] for o in manifest['objects'].values()) for k in ('leg', 'foot', 'shoe')}
    write(out / 'partition_definition.json', dict(labels=LABELS, objects=manifest['objects'], composites=composites,
        anatomy_status={k: 'AVAILABLE' if n else 'NOT_AVAILABLE' for k, n in anatomy.items()},
        rule='按1019-02固定协议；顶点组三角面权重比例>=0.5，足优先；GT仅供评测。'))
    for number in range(1, 151):
        frame_mesh = manifest['frames'][str(number)]['mesh']
        if frame_mesh != previous_mesh:
            meshfile = out / 'meshes' / frame_mesh
            if digest(meshfile) != manifest['meshes'][frame_mesh]['sha256']:
                raise ValueError('实际时刻GT网格SHA不符：'+frame_mesh)
            mesh = dict(np.load(meshfile, allow_pickle=False))
            scene = o3d.t.geometry.RaycastingScene(nthreads=args.threads)
            scene.add_triangles(o3d.core.Tensor(mesh['vertices'], dtype=o3d.core.Dtype.Float32),
                               o3d.core.Tensor(mesh['triangles'].astype(np.uint32)))
            previous_mesh = frame_mesh
        record = cameras['frames'][f'{number:04d}']
        c2w = np.asarray(record['extrinsics']['camera_to_world'])
        i = record['intrinsics']; sx = width / cameras['resolution']['width']; sy = height / cameras['resolution']['height']
        intrinsic = np.asarray([i['fx']*sx, i['fy']*sy, i['cx']*sx, i['cy']*sy])
        rays = camera_rays(c2w, intrinsic, height, width)
        hits = scene.cast_rays(o3d.core.Tensor(rays), nthreads=args.threads)
        z = hits['t_hit'].numpy(); valid = np.isfinite(z) & (z > 0)
        tri = np.where(valid, hits['primitive_ids'].numpy(), 0).astype(np.int64)
        uv = hits['primitive_uvs'].numpy()
        weights = np.stack((1-uv.sum(-1), uv[..., 0], uv[..., 1]), axis=-1)
        normal, _ = unit_normal((mesh['corner_normals'][tri] * weights[..., None]).sum(-2))
        normal[~valid] = 0
        oid = np.where(valid, mesh['object_ids'][tri], -1).astype(np.int32)
        sid = np.where(valid, mesh['semantic_ids'][tri], 0).astype(np.uint8)
        alpha_path = args.source / f'albedo/albedo_{number:04d}.png'
        rgba = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.dtype != np.uint16 or rgba.shape[2] != 4:
            raise ValueError('要求16位RGBA输入。')
        alpha = cv2.resize(rgba[..., 3].astype(np.float32) / 65535., (width, height), interpolation=cv2.INTER_AREA)
        del rgba  # 颜色不统计、不保存、不用于训练或评价。
        exr = args.source / f'normal_exr/normal_{number:04d}.exr'
        target, target_valid = load_normal_exr(exr, height, width)
        errors, normal_valid = normal_errors(normal, target)
        inside = erode((alpha >= .99) & valid & target_valid, 2) & normal_valid
        foreground = alpha >= .5
        iou = float((foreground & valid).sum() / max(1, (foreground | valid).sum()))
        audit = dict(alpha_hit_iou=iou, valid_pixels=int(valid.sum()),
            exr_mesh_normal_deg=summarize(errors['angular_deg'], inside),
            exr_mesh_normal_axis_deg=summarize(errors['axis_deg'], inside))
        filename = f'gt_{number:04d}.npz'
        depth = np.where(valid, z, 0)
        np.savez_compressed(out / 'gt_depth' / filename, depth_z=depth,
            depth_ray=depth*np.linalg.norm(rays[..., 3:], axis=-1), valid=valid,
            normal_world=normal.astype(np.float32), object_id=oid, semantic_id=sid,
            object_names=mesh['object_names'], alpha=alpha, gt_exr_normal=target.astype(np.float32),
            gt_exr_valid=target_valid, intrinsics=intrinsic, camera_to_world=c2w)
        report['frames'][str(number)] = dict(file=filename, sha256=digest(out / 'gt_depth' / filename),
            source_frame=number, mesh=manifest['frames'][str(number)]['mesh'], audit=audit)
        if getattr(args, 'allow_dynamic', False):
            report['frames'][str(number)]['time'] = record.get('time_normalized', (number-1)/149)
            report['frames'][str(number)]['geometry_sha256'] = manifest['frames'][str(number)]['geometry_sha256']
        report['source_sha256'][str(alpha_path.resolve())] = digest(alpha_path)
        report['source_sha256'][str(exr.resolve())] = digest(exr)
        # 先保留数组/审计再拒绝，失败不删除。
        write(out / 'gt_depth/manifest.json', report)
        if iou < .97 or audit['exr_mesh_normal_axis_deg']['mean'] is None or audit['exr_mesh_normal_axis_deg']['mean'] > 3:
            raise ValueError(f'GT一致性失败，帧{number}；详见已保存审计。')
        if number in (10, 80, 150):
            palette = np.asarray([[0,0,0],[160,160,160],[255,150,0],[0,180,255],[255,0,180],[0,180,50]], np.uint8)
            cv2.imwrite(str(out / 'previews' / f'partition_{number:04d}.png'), palette[sid][..., ::-1])
        if number % 10 == 0:
            print(f'GT {number:04d} 一致性核查完成', flush=True)
    for path, sha in report['source_sha256'].items():
        if digest(path) != sha:
            raise ValueError('GT输入在执行中变化。')
    return report


def main(source=None, allowed=None, script=None, blend_relative='_export/source_scene.blend', allow_dynamic=False):
    global SOURCE, ALLOWED
    if source is not None:
        SOURCE = Path(source).resolve()
    if allowed is not None:
        ALLOWED = Path(allowed).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ALLOWED / 'gt_v1')
    parser.add_argument('--blender', type=Path, default=TOOLS / 'blender-5.2.2-linux-x64/blender')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--blender-worker', action='store_true', help=argparse.SUPPRESS)
    parser.set_defaults(allow_dynamic=allow_dynamic)
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None
    args = parser.parse_args(argv)
    if socket.gethostname() != 'ushas':
        parser.error('当前入口仅登记ushas。')
    args.source = args.source.resolve(); args.output = guard(args.output)
    if args.source != SOURCE.resolve():
        parser.error('只允许已登记V3源数据。')
    if args.blender_worker:
        worker(args); return
    if Path(sys.executable).resolve() != Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve():
        parser.error('必须使用lumimotion-ushas的Python。')
    if args.output.exists():
        parser.error('输出禁止覆盖；失败重试须使用eval_only下的新目录。')
    args.output.mkdir(parents=True)
    for sub in ('meshes', 'gt_depth', 'previews', 'logs'):
        (args.output / sub).mkdir()
    blend = args.source / blend_relative
    entry = Path(script or __file__).resolve()
    snapshots = {str(p): digest(p) for p in (blend, args.source / 'camera.json', args.source / 'object_pose.json', Path(__file__), entry)}
    command = [str(args.blender.resolve()), '--factory-startup', '--disable-autoexec', '--threads', str(args.threads), '-b', str(blend),
               '--python', str(entry), '--', '--blender-worker',
               '--source', str(args.source), '--output', str(args.output)]
    write(args.output / 'plan.json', dict(command=command, shell=shlex.join(command), source_sha256=snapshots,
        python=sys.executable, host=socket.gethostname(), evaluated_frames=list(range(1,151)),
        light_or_material_parameters_read=False, model_access=False, output_guard=str(ALLOWED)))
    readme = args.output / 'README.md'
    readme.write_text('# V3隔离评测GT准备\n\n仅准备GT，不运行训练/模型评测。\n\n完整Blender命令：\n\n```bash\n' + shlex.join(command) + '\n```\n\nsource：`' + str(args.source) + '`；model/iteration：不适用。\n渲染输出：不生成模型渲染；GT为gt_depth、分区图为previews，日志为logs。\n定量指标：GT一致性审计在gt_depth/manifest.json；非模型重建指标。\n代表图：previews/partition_0010.png、0080、0150。\n四类模型可视化RGB/Alpha/Normal/分离：未生成，均不适用。\n最终结论：PENDING。\n')
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(TOOLS / 'libs/usr/lib/x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
    try:
        with (args.output / 'logs/blender.log').open('x') as log:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f'Blender退出码{rc}；失败日志保留。')
        report = prepare(args, args.output)
        if any(digest(path) != sha for path, sha in snapshots.items()):
            raise AssertionError('源文件发生变化。')
        write(args.output / 'GT_COMPLETED.json', dict(status='PASS_GT_PREPARATION', frames=len(report['frames']),
            source_unchanged=True, evaluation_only=True, model_evaluation='NOT_RUN'))
        with readme.open('a') as f:
            f.write('\n最终结论：**PASS（仅GT准备与一致性，模型验收未执行）**。\n')
        print('PASS：隔离GT准备完成，未运行模型评测。', flush=True)
    except Exception as exc:
        write(args.output / 'FAILED.json', dict(status='FAILED', error=str(exc), retained=True))
        with readme.open('a') as f:
            f.write('\n最终结论：**FAILED**；数组、网格、统计和日志全部保留。\n')
        raise


if __name__ == '__main__':
    main()
