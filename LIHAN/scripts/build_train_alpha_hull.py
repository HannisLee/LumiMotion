"""仅用135 TRAIN Alpha/相机构造一次锁定的保守视觉外壳位置初始化。"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import math
import socket
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_dilation
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'output/1017-01-静态DL未知光总控'
BASE = CONTROL / 'data_train_T04_temporal08'
FROZEN = CONTROL / 'snapshots/fixed_T08_skip_final_reset_v1'
OUT = CONTROL / '1017-04-TRAIN轮廓外壳初始化'
SOURCE = CONTROL / 'data_train_T13_alpha_hull_v1'
RANDOM_PLY_SHA = 'f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971'
CORNER = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
PRODUCTION_STARTED = False


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value); self.log.write(value)
        return len(value)

    def flush(self):
        self.stream.flush(); self.log.flush()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def original_functions():
    """执行冻结原函数AST，避免加载数据集模块的无关依赖。"""
    ns = {'np': np, 'math': math}
    for relative, names in [('scene/dataset_readers.py', ['blender_c2w_to_camera_extrinsics']),
                            ('utils/graphics_utils.py', ['getWorld2View2', 'focal2fov', 'fov2focal'])]:
        module = ast.parse((FROZEN / relative).read_text())
        nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), ns)
    return ns


def integral_mask(mask):
    return np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def rectangle_test(boxes, done, cameras, lower, h):
    """仅整节点投影矩形无支持时删除；全支持标志对子节点安全。"""
    keep = np.ones(len(boxes), dtype=bool)
    for v, camera in enumerate(cameras):
        word, bit = divmod(v, 64)
        flag = np.uint64(1) << np.uint64(bit)
        ids = np.flatnonzero(keep & ((done[:, word] & flag) == 0))
        if not len(ids):
            continue
        nodes = boxes[ids]
        xyz = lower + (nodes[:, None, :3] + CORNER[None] *
                       (nodes[:, None, 3:] - nodes[:, None, :3])) * h
        pc = xyz @ camera['E'][:3, :3].T + camera['E'][:3, 3]
        positive = (pc[:, :, 2] > 1e-6).all(1)
        # 跨相机平面的节点保守保留并细分，不把投影极值公式用于负分母。
        ids = ids[positive]
        pc = pc[positive]
        if not len(ids):
            continue
        uv = pc[:, :, :2] / pc[:, :, 2:3] * camera['f'] + camera['size'] / 2
        lo = np.ceil(uv.min(1) - 1e-6).astype(np.int64) - 1
        hi = np.floor(uv.max(1) + 1e-6).astype(np.int64)
        width, height = camera['size']
        inside = ((lo >= 0).all(1) & (hi[:, 0] < width) & (hi[:, 1] < height))
        lo_clip = np.maximum(lo, 0)
        hi_clip = np.minimum(hi, camera['size'] - 1)
        valid = (lo_clip <= hi_clip).all(1)
        count = np.zeros(len(ids), dtype=np.int64)
        i = np.flatnonzero(valid)
        x0, y0 = lo_clip[i].T
        x1, y1 = (hi_clip[i] + 1).T
        table = camera['integral']
        count[i] = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
        keep[ids[count == 0]] = False
        area = np.prod(hi - lo + 1, axis=1)
        full = inside & (count == area)
        done[ids[full], word] |= flag
    return keep, done


def carve(cameras, lower, h, shape, log=None, batch=4096):
    occupancy = np.zeros(tuple(shape), dtype=bool)
    boxes = np.array([[0, 0, 0, *shape]], dtype=np.int32)
    done = np.zeros((1, (len(cameras) + 63) // 64), dtype=np.uint64)
    full_words = np.zeros(done.shape[1], dtype=np.uint64)
    for v in range(len(cameras)):
        full_words[v // 64] |= np.uint64(1) << np.uint64(v % 64)
    levels = []
    while len(boxes):
        next_boxes, next_done = [], []
        record = dict(level=len(levels), input_nodes=len(boxes), rejected=0,
                      filled_interior_nodes=0, surviving_leaf_nodes=0, split_nodes=0)
        for start in range(0, len(boxes), batch):
            b = boxes[start:start + batch]
            k, state = rectangle_test(b, done[start:start + batch].copy(), cameras, lower, h)
            record['rejected'] += int((~k).sum())
            b, state = b[k], state[k]
            leaves = ((b[:, 3:] - b[:, :3]) == 1).all(1)
            interior = (state == full_words).all(1) & ~leaves
            for node in b[interior]:
                occupancy[tuple(slice(int(a), int(z)) for a, z in zip(node[:3], node[3:]))] = True
            leaf = b[leaves, :3]
            if len(leaf):
                occupancy[tuple(leaf.T)] = True
            record['filled_interior_nodes'] += int(interior.sum())
            record['surviving_leaf_nodes'] += int(leaves.sum())
            todo = ~(leaves | interior)
            b, state = b[todo], state[todo]
            record['split_nodes'] += len(b)
            if not len(b):
                continue
            middle = (b[:, :3] + b[:, 3:]) // 2
            for side in CORNER:
                lo = np.where(side, middle, b[:, :3])
                hi = np.where(side, b[:, 3:], middle)
                valid = (hi > lo).all(1)
                if valid.any():
                    next_boxes.append(np.concatenate([lo[valid], hi[valid]], axis=1))
                    next_done.append(state[valid])
        levels.append(record)
        if log:
            log.write(json.dumps(record, ensure_ascii=False) + '\n'); log.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)
        boxes = np.concatenate(next_boxes) if next_boxes else np.empty((0, 6), np.int32)
        done = np.concatenate(next_done) if next_done else np.empty((0, len(full_words)), np.uint64)
    return occupancy, levels


def exposed_faces(occupied):
    faces = []
    for axis in range(3):
        for side in (0, 1):
            neighbor = np.zeros_like(occupied)
            dest = [slice(None)] * 3; src = [slice(None)] * 3
            dest[axis] = slice(1, None) if side == 0 else slice(None, -1)
            src[axis] = slice(None, -1) if side == 0 else slice(1, None)
            neighbor[tuple(dest)] = occupied[tuple(src)]
            indices = np.argwhere(occupied & ~neighbor).astype(np.int32)
            faces.append(np.column_stack([indices, np.full(len(indices), axis * 2 + side, np.int32)]))
    return np.concatenate(faces)


def sample_faces(faces, lower, h, count=100000):
    rng = np.random.RandomState(0)
    def generate(n):
        selected = faces[rng.randint(len(faces), size=n)]
        uv = rng.random_sample((n, 2))
        offsets = np.zeros((n, 3), np.float64)
        for axis in range(3):
            rows = selected[:, 3] // 2 == axis
            offsets[rows, axis] = selected[rows, 3] % 2
            offsets[np.ix_(rows, [a for a in range(3) if a != axis])] = uv[rows]
        return (lower + (selected[:, :3] + offsets) * h).astype(np.float32)
    points = generate(count)
    removed = 0
    while True:
        _, first = np.unique(points, axis=0, return_index=True)
        unique = np.zeros(len(points), bool); unique[first] = True
        missing = int((~unique).sum())
        if not missing:
            break
        removed += missing
        points[~unique] = generate(missing)
    assert len(points) == count and np.isfinite(points).all()
    return points, removed


def self_test():
    """小格逐叶独立穷举对照，包含细线/孔洞，核验层次排除没有漏格。"""
    masks = [np.eye(12, dtype=bool), np.ones((12, 12), bool), np.zeros((12, 12), bool)]
    masks[1][4:8, 4:8] = False
    for mask in masks:
        camera = dict(E=np.eye(4), f=np.array([7., 7.]), size=np.array([12, 12]),
                      integral=integral_mask(mask))
        lower = np.array([-1., -1., 2.]); h = .25; shape = np.array([7, 8, 5])
        hierarchical, _ = carve([camera], lower, h, shape)
        dense = np.zeros(tuple(shape), bool)
        for index in np.ndindex(*shape):
            lo = lower + np.array(index) * h
            pc = lo + CORNER * h
            uv = pc[:, :2] / pc[:, 2:3] * camera['f'] + camera['size'] / 2
            a = np.ceil(uv.min(0) - 1e-6).astype(int) - 1
            z = np.floor(uv.max(0) + 1e-6).astype(int)
            a = np.maximum(a, 0); z = np.minimum(z, camera['size'] - 1)
            dense[index] = bool((a <= z).all() and mask[a[1]:z[1]+1, a[0]:z[0]+1].any())
        assert np.array_equal(dense, hierarchical)
    # 单格六面均应暴露；相邻两格恰十面。
    assert len(exposed_faces(np.ones((1, 1, 1), bool))) == 6
    assert len(exposed_faces(np.ones((2, 1, 1), bool))) == 10
    state = np.random.get_state()
    pts, repeated = sample_faces(exposed_faces(np.ones((2, 1, 1), bool)), np.zeros(3), .1)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(state, after))
    return dict(status='PASS',hierarchical_equals_independent_dense_cases=3,
                six_and_ten_faces=True, sampled_unique=len(np.unique(pts,axis=0)),
                duplicate_replacements=repeated, global_numpy_rng_unchanged=True)


def build():
    global PRODUCTION_STARTED
    assert socket.gethostname() == 'ushas'
    decision = json.loads((CONTROL/'T12_scientific_decision.json').read_text())
    assert decision['complete_16000_review'] and not decision['adopted'] and decision['next_control']=='T08'
    assert not OUT.exists() and not SOURCE.exists()
    OUT.mkdir()
    PRODUCTION_STARTED = True
    (OUT/'logs').mkdir()
    log = (OUT/'logs/construction.log').open('w', buffering=1)
    sys.stdout = Tee(sys.stdout, log); sys.stderr = Tee(sys.stderr, log)
    import shutil
    shutil.copyfile(Path(__file__), OUT/'generator.py')
    (OUT/'README.md').write_text('''# T13训练Alpha保守视觉外壳初始化构造

状态：CPU构造运行中，尚无科学PASS。完整计划见LIHAN/DOC/训练/1017-21-T13训练Alpha保守视觉外壳位置初始化计划.md。

实际命令（ushas/lumimotion-ushas）：

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/han.li/miniconda3/envs/lumimotion-ushas/bin/python LIHAN/scripts/build_train_alpha_hull.py
```

source=总控/data_train_T04_temporal08原135 TRAIN RGB/Alpha/相机与旧.80灯，唯一派生位置点云目标为总控/data_train_T13_alpha_hull_v1；model/iteration=N/A（CPU构造，不加载模型或GT）。冻结实际构造源generator.py、相机审计camera_audit.json、LP六极值lp.json、网格grid.json、层次carving.jsonl、occupancy.npy及暴露面exposed_faces.npy完整保留；日志logs/construction.log。

渲染命令、渲染目录、三MAE、图像/视频及RGB/Alpha/normal/separation四类目检=N/A，当前构造不渲染/不加载GT。覆盖/外溢不作为参数选择。定量工程项为LP有限、固定512与h、保留格数/面数、100000唯一有限XYZ、非XYZ原属性及原138输入保持；实际完成后追加result.json与退出结论。科研结论待独立工程和fresh16k原协议完整评测，不能将CPU PASS当科研PASS。
''')
    write_json(OUT/'state.json',dict(status='CPU_RUNNING',started=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write_json(OUT/'self_test.json',self_test())
    manifest = json.loads((BASE/'no_gt_manifest.json').read_text())
    assert not manifest['gt_read'] and not manifest['init_weights_loaded']
    assert len(manifest['files'])==138 and sha(BASE/'points3d.ply')==RANDOM_PLY_SHA
    assert all(sha(BASE/k)==v for k,v in manifest['files'].items())
    snapshot = json.loads((FROZEN/'snapshot_manifest.json').read_text())
    assert len(snapshot)==96 and all(sha(FROZEN/k)==v for k,v in snapshot.items())
    ns=original_functions()
    transforms=json.loads((BASE/'transforms_train.json').read_text())
    frames=sorted(transforms['frames'],key=lambda f:int(Path(f['file_path']).stem.split('_')[-1]))
    assert len(frames)==135 and {int(Path(f['file_path']).stem.split('_')[-1]) for f in frames}=={i for i in range(1,151) if i%10}
    cameras=[]; constraints=[]; camera_audit=[]
    for frame in frames:
        name=Path(frame['file_path']).name
        if not Path(name).suffix:name+='.png'
        path=BASE/'images'/name
        with Image.open(path) as im:
            rgba=np.array(im.convert('RGBA')); width,height=im.size
        assert (width,height)==(800,800)
        original=rgba[:,:,3]>0; mask=binary_dilation(original,structure=np.ones((3,3),bool))
        assert not original[[0,-1],:].any() and not original[:,[0,-1]].any()
        def value(k):return frame.get(k,transforms.get(k))
        fx=value('fl_x'); fy=value('fl_y'); fovx=value('camera_angle_x'); fovy=value('camera_angle_y')
        fovx=ns['focal2fov'](fx,width) if fx is not None and fx>0 else fovx
        assert fovx is not None
        fovy=ns['focal2fov'](fy,height) if fy is not None and fy>0 else (fovy if fovy is not None else ns['focal2fov'](ns['fov2focal'](fovx,width),height))
        focal=np.array([ns['fov2focal'](fovx,width),ns['fov2focal'](fovy,height)])
        rotation,translation=ns['blender_c2w_to_camera_extrinsics'](frame['transform_matrix'])
        E=ns['getWorld2View2'](rotation,translation).astype(np.float64)
        # 原相机坐标中心、向右/下已知点的纯数学投影核验，无物体/GT点。
        probe_camera=np.array([[0,0,2,1],[1,0,2,1],[0,1,2,1]],np.float64)
        world=probe_camera@np.linalg.inv(E).T
        back=world@E.T
        uv=back[:,:2]/back[:,2:3]*focal+np.array([width,height])/2
        expected=np.array([[width/2,height/2],[width/2+focal[0]/2,height/2],[width/2,height/2+focal[1]/2]])
        assert np.max(np.abs(uv-expected))<1e-8
        ys=np.flatnonzero(mask.any(1)); left=mask[ys].argmax(1); right=width-1-mask[ys,::-1].argmax(1)
        centers=np.concatenate([np.column_stack([left,ys]),np.column_stack([right,ys])])
        corners=(centers[:,None,:]+np.array([[0,0],[0,1],[1,0],[1,1]])).reshape(-1,2)
        hull=ConvexHull(corners)
        p=np.stack([focal[0]*E[0]+width/2*E[2],focal[1]*E[1]+height/2*E[2],E[2]])
        planes=hull.equations@p
        planes/=np.linalg.norm(planes[:,:3],axis=1)[:,None]
        constraints.extend(planes)
        constraints.append(np.r_[-E[2,:3],-E[2,3]+1e-6])
        cameras.append(dict(E=E,f=focal,size=np.array([width,height]),integral=integral_mask(mask)))
        camera_audit.append(dict(frame=name,original_support=int(original.sum()),conservative_support=int(mask.sum()),
                                 focal=focal.tolist(),world_to_camera=E.tolist(),hull_planes=len(planes),projection_max_abs=float(np.max(np.abs(uv-expected)))))
    constraints=np.array(constraints); A=constraints[:,:3]; b=-constraints[:,3]
    minima=[];maxima=[]; lp_records=[]
    for axis in range(3):
        for sign in (1,-1):
            objective=np.zeros(3);objective[axis]=sign
            result=linprog(objective,A_ub=A,b_ub=b,bounds=[(None,None)]*3,method='highs',
                           options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            lp_records.append(dict(axis=axis,sign=sign,status=result.status,message=result.message,
                                   point=None if result.x is None else result.x.tolist()))
            write_json(OUT/'lp.json',lp_records)
            assert result.success and np.isfinite(result.x).all(), '轮廓锥LP无界/不可行，禁止bbox回退'
            assert np.max(A@result.x-b)<1e-6
            (minima if sign==1 else maxima).append(result.x[axis])
    lower=np.array(minima)-1e-6; upper=np.array(maxima)+1e-6
    h=float(np.max(upper-lower)/512)
    shape=np.minimum(512,np.ceil((upper-lower)/h).astype(int))
    assert ((lower+shape*h)>=upper-1e-12).all() and (shape>0).all()
    write_json(OUT/'camera_audit.json',camera_audit)
    write_json(OUT/'grid.json',dict(lower=lower.tolist(),upper=upper.tolist(),h=h,shape=shape.tolist(),
                                   grid_upper=(lower+shape*h).tolist(),LP_constraints=len(A),numerical_world_padding=1e-6))
    with (OUT/'carving.jsonl').open('w') as stream:
        occupied,levels=carve(cameras,lower,h,shape,stream)
    assert occupied.any()
    np.save(OUT/'occupancy.npy',occupied)
    faces=exposed_faces(occupied);assert len(faces)>0
    np.save(OUT/'exposed_faces.npy',faces)
    rng_before=np.random.get_state(); points,replaced=sample_faces(faces,lower,h)
    assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(rng_before,np.random.get_state()))
    original_ply=PlyData.read(BASE/'points3d.ply'); original_vertices=original_ply['vertex'].data
    assert len(original_vertices)==100000
    vertices=original_vertices.copy()
    for i,key in enumerate(('x','y','z')):vertices[key]=points[:,i]
    nonxyz=[k for k in vertices.dtype.names if k not in ('x','y','z')]
    assert all(np.array_equal(vertices[k],original_vertices[k]) for k in nonxyz)
    cloud=OUT/'points3d.ply';PlyData([PlyElement.describe(vertices,'vertex')],text=original_ply.text,byte_order=original_ply.byte_order).write(cloud)
    # 输入来源和固定构造参数绑定到五项训练源manifest，不复制外部几何/权重。
    SOURCE.mkdir();(SOURCE/'images').mkdir()
    import os
    for relative in manifest['files']:
        if relative=='points3d.ply':continue
        dest=SOURCE/relative
        os.link(BASE/relative,dest)
    shutil.copyfile(cloud,SOURCE/'points3d.ply')
    files={k:sha(SOURCE/k) for k in manifest['files']}
    assert [k for k in files if files[k]!=manifest['files'][k]]==['points3d.ply']
    evidence=dict(method='train_alpha_conservative_visual_hull_v1',gt_read=False,init_weights_loaded=False,
                  baseline_manifest_sha256=sha(BASE/'no_gt_manifest.json'),baseline_initial_point_cloud_sha256=RANDOM_PLY_SHA,
                  generator_sha256=sha(Path(__file__)),T08_snapshot_manifest_sha256=sha(FROZEN/'snapshot_manifest.json'),
                  camera_audit_sha256=sha(OUT/'camera_audit.json'),grid_sha256=sha(OUT/'grid.json'),
                  lp_sha256=sha(OUT/'lp.json'),occupancy_sha256=sha(OUT/'occupancy.npy'),faces_sha256=sha(OUT/'exposed_faces.npy'),
                  Alpha_rule='original800 Alpha>0 then3x3 dilation1',longest_axis_cells=512,
                  surface='explicit_exposed_voxel_faces',face_order='axis_then_side_then_lexicographic_voxel',
                  sample_rng='local RandomState(0)',sample_count=100000,unique_float32_points=100000,
                  nonXYZ_exact=True,global_numpy_rng_unchanged=True,duplicate_replacements=replaced,
                  occupied_cells=int(occupied.sum()),exposed_faces=len(faces),level_records=levels,
                  point_cloud_sha256=sha(cloud))
    new=dict(manifest,initial_point_cloud_sha256=sha(cloud),files=files,initialization=evidence)
    write_json(SOURCE/'no_gt_manifest.json',new)
    assert all(sha(BASE/k)==v for k,v in manifest['files'].items())
    assert all(sha(SOURCE/k)==v for k,v in files.items())
    assert sha(Path(__file__))==sha(OUT/'generator.py'), '运行期间构造源变化，禁止准入'
    write_json(OUT/'result.json',dict(status='CPU_CONSTRUCTION_PASS_NOT_SCIENTIFIC_PASS',**evidence,
                                   source=str(SOURCE),source_manifest_sha256=sha(SOURCE/'no_gt_manifest.json'),
                                   finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write_json(OUT/'state.json',dict(status='CPU_CONSTRUCTION_FINISHED',exit_code=0,
                                   finished=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with (OUT/'README.md').open('a') as stream:
        stream.write('\n\nCPU构造实际完成：工程PASS，科研尚未评测。完整结果result.json；保留格'+str(int(occupied.sum()))+'、暴露面'+str(len(faces))+'、100000唯一有限位置、非XYZ exact与原输入SHA保持；点云SHA '+sha(cloud)+'，新manifest SHA '+sha(SOURCE/'no_gt_manifest.json')+'。没有任何新渲染/三MAE或四类目检，N/A。\n')
    print(json.dumps({'status':'PASS','source':str(SOURCE),'point_cloud_sha256':sha(cloud)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),ensure_ascii=False))
    else:
        try:
            build()
        except Exception as error:
            if PRODUCTION_STARTED:
                write_json(OUT/'state.json',dict(status='CPU_CONSTRUCTION_FAILED',exit_code=1,
                           error=repr(error),failed=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                with (OUT/'README.md').open('a') as stream:
                    stream.write('\n\nCPU构造FAILED：'+repr(error)+'；全部失败现场保留，无模型训练/科学PASS。\n')
            raise
