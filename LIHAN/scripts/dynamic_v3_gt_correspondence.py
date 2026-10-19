"""隔离证明动态GT的持久顶点索引；不评价候选、不读取训练输出。

依次核验150帧每对象顶点数、原始polygon/loop连接与保序modifier栈，
并与现有当时GT网格的world顶点逐字节对照。三角剖分重排不作为顶点拓扑证据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控/eval_only'
TOOLS=ROOT/'output/archive/1011-07-六组训练总控/tools'
RAW=dict(elv2=ROOT/'data/LH-data/danamic/LH-dynamic-jumpingjacks_EL_V2',
    v3=ROOT/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3')
GT=dict(elv2=CONTROL/'old_dynamic_gt_v1/meshes',v3=CONTROL/'dynamic_gt_v1/meshes')
# 这些modifier不重建vertex/polygon索引；不认识的类型明确拒绝推断。
PRESERVE_INDEX={'ARMATURE','SMOOTH','CORRECTIVE_SMOOTH','LAPLACIANSMOOTH','NORMAL_EDIT','WEIGHTED_NORMAL'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def worker(args):
    import bpy
    import numpy as np
    raw=RAW[args.dataset];gt_dir=GT[args.dataset];manifest=json.loads((gt_dir/'manifest.json').read_text())
    blend=Path(bpy.data.filepath).resolve()
    if blend!=(raw/'scene_snapshot.blend').resolve():raise ValueError('源blend不符。')
    hashes={str(p):sha(p) for p in (blend,Path(__file__).resolve(),gt_dir/'manifest.json')}
    scene=bpy.context.scene
    for obj in scene.objects:
        for mod in obj.modifiers:mod.show_viewport=mod.show_render
    reference=None;frames={};errors=[];unsupported=[];reference_names=None
    for n in range(1,151):
        scene.frame_set(n);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
        objects=[];all_vertices=[];offset=0
        for obj in sorted(scene.objects,key=lambda x:x.name):
            if obj.type!='MESH' or obj.hide_render:continue
            if obj.users_collection and all(c.hide_render for c in obj.users_collection):continue
            evaluated=obj.evaluated_get(dg);mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
            try:
                if not len(mesh.polygons):continue
                loop_start=np.asarray([p.loop_start for p in mesh.polygons],dtype=np.int32)
                loop_total=np.asarray([p.loop_total for p in mesh.polygons],dtype=np.int32)
                loop_vertices=np.asarray([loop.vertex_index for loop in mesh.loops],dtype=np.int32)
                h=hashlib.sha256()
                for array in (np.asarray([len(mesh.vertices)],np.int64),loop_start,loop_total,loop_vertices):h.update(array.tobytes())
                matrix=np.asarray(evaluated.matrix_world,dtype=np.float64)
                vertices=(np.asarray([v.co[:] for v in mesh.vertices])@matrix[:3,:3].T+matrix[:3,3]).astype(np.float32)
                modifiers=[dict(name=m.name,type=m.type,show_render=bool(m.show_render)) for m in obj.modifiers]
                bad=[m['type'] for m in modifiers if m['show_render'] and m['type'] not in PRESERVE_INDEX]
                if bad:unsupported.append(dict(frame=n,object=obj.name,modifier_types=bad))
                objects.append(dict(name=obj.name,vertex_count=len(mesh.vertices),polygon_count=len(mesh.polygons),
                    loop_count=len(mesh.loops),polygon_loop_sha256=h.hexdigest(),modifiers=modifiers,
                    vertex_slice=[offset,offset+len(vertices)],world_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest()))
                offset+=len(vertices);all_vertices.append(vertices)
            finally:evaluated.to_mesh_clear()
        if not all_vertices:raise ValueError('没有GT网格对象。')
        topology=[{k:v for k,v in o.items() if k!='world_vertices_sha256'} for o in objects]
        names=[o['name'] for o in objects]
        if reference is None:reference=topology;reference_names=names
        topology_equal=topology==reference
        if not topology_equal:errors.append(dict(frame=n,reason='对象顺序/顶点数/多边形连接/modifier栈发生变化'))
        vertices=np.concatenate(all_vertices)
        gt_file=gt_dir/manifest['frames'][str(n)]['mesh'];expected=manifest['meshes'][gt_file.name]['sha256']
        if sha(gt_file)!=expected:raise ValueError('当时GT文件SHA不符。')
        with np.load(gt_file,allow_pickle=False) as z:
            coords_equal=np.array_equal(vertices,z['vertices']) and names==list(z['object_names'])
        if not coords_equal:errors.append(dict(frame=n,reason='持久索引world顶点或对象顺序不等于当前GT NPZ'))
        frames[str(n)]=dict(objects=objects,vertex_count=len(vertices),topology_equal_reference=topology_equal,
            world_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),gt_vertices_exact_equal=bool(coords_equal),
            gt_mesh=str(gt_file),gt_mesh_sha256=expected)
        if n%10==0:print(f'持久GT顶点对应审计 {args.dataset} {n:04d}/0150',flush=True)
    if any(sha(Path(p))!=h for p,h in hashes.items()):raise ValueError('源文件发生变化。')
    proven=not errors and not unsupported
    write(args.output/'correspondence.json',dict(format_version=1,status='PROVEN_PERSISTENT_VERTEX_INDICES' if proven else 'NOT_AVAILABLE',
        evaluation_only=True,dataset=args.dataset,source_sha256=hashes,frames=frames,reference_object_names=reference_names,
        errors=errors,unsupported_modifiers=unsupported,preserving_modifier_types=sorted(PRESERVE_INDEX),
        definitions=dict(index='按对象名字排序，对象内evaluated mesh vertex index；拼接slice与现有GT NPZ逐帧完全一致',
            topology='vertex数量、polygon loop_start/loop_total及loop.vertex_index完整连接哈希；独立于calc_loop_triangles',
            allowed='全部150帧对象/vertex/polygon/loop/modifier保持，且只有登记的保序modifier，且GT NPZ顶点逐元素相等',
            limitation='只证明本源文件/Blender版本的索引对应；不将GT顶点用于任何训练监督或参数选择'),
        blender_version=bpy.app.version_string,candidate_model_loaded=False,training_or_selection_use=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=('elv2','v3'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    args.output=args.output.resolve()
    if not args.output.is_relative_to(CONTROL):raise ValueError('仅允许本轮eval_only隔离目录。')
    if socket.gethostname()!='ushas':raise ValueError('仅登记ushas。')
    if args.worker:worker(args);return
    if Path(sys.executable).resolve()!=Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve():raise ValueError('必须使用lumimotion-ushas。')
    args.output.mkdir(parents=True,exist_ok=False)
    blend=RAW[args.dataset]/'scene_snapshot.blend';blender=TOOLS/'blender-5.2.2-linux-x64/blender'
    if not blend.is_file() or not blender.is_file():raise FileNotFoundError('已登记blend或Blender不可用。')
    command=[str(blender),'--factory-startup','--disable-autoexec','--threads','4','-b',str(blend),'--python',str(Path(__file__).resolve()),
        '--','--worker','--dataset',args.dataset,'--output',str(args.output)]
    write(args.output/'plan.json',dict(command=command,shell=shlex.join(command),frames=list(range(1,151)),
        source_sha256={str(p):sha(p) for p in (blend,Path(__file__).resolve(),GT[args.dataset]/'manifest.json')},
        candidate_model_loaded=False,training_or_selection_use=False))
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(TOOLS/'libs/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
    try:
        with (args.output/'blender.log').open('x') as log:
            rc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise RuntimeError('Blender退出码：'+str(rc))
        result=json.loads((args.output/'correspondence.json').read_text())
        print(json.dumps(dict(status=result['status'],output=str(args.output),errors=len(result['errors']),unsupported=len(result['unsupported_modifiers'])),ensure_ascii=False),flush=True)
    except Exception as exc:
        write(args.output/'FAILED.json',dict(status='FAILED',error=str(exc),retained=True));raise


if __name__=='__main__':main()
