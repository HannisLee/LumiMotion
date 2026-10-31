"""严格RGB与相机输入：边界连通背景分割、16位观测及随机初始化。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-only', action='store_true',help='从已准备source_all生成训练隔离副本')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    if a.train_only:
        src=a.source.resolve()
        manifest=json.loads((src/'rgb_only_manifest.json').read_text())
        for i in manifest['train_frames']:
            os.link(src/'images'/f'image_{i:04d}.png',out/'images'/f'image_{i:04d}.png')
        os.link(src/'points3d.ply',out/'points3d.ply')
        (out/'transforms_train.json').write_bytes((src/'transforms_train.json').read_bytes())
        (out/'transforms_test.json').write_text(json.dumps({'frames':[]}))
        manifest['files']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}
        write(out/'rgb_only_manifest.json',manifest)
        print('训练隔离完成：仅117张TRAIN与空测试集',flush=True)
        return
    camera = json.loads((a.source/'camera.json').read_text())
    records, stats, hashes = [], [], {}
    for i in range(1, 151):
        path = a.source/'image'/f'image_{i:04d}.png'
        raw = cv2.imread(str(path), -1)
        assert raw.shape == (800,800,4) and raw.dtype == np.uint16
        # 明确丢弃原始alpha，不访问albedo、normal、light或其他GT文件。
        rgb = raw[...,:3].copy()
        border = np.concatenate([rgb[:8].reshape(-1,3),rgb[-8:].reshape(-1,3),
                                 rgb[:,:8].reshape(-1,3),rgb[:,-8:].reshape(-1,3)])
        bg = np.median(border, axis=0)
        candidate = (np.max(np.abs(rgb.astype(np.int32)-bg),axis=-1)<=2).astype(np.uint8)
        count, labels = cv2.connectedComponents(candidate, connectivity=4)
        touching = np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
        touching = touching[touching != 0]
        foreground = ~np.isin(labels, touching)
        alpha = foreground.astype(np.uint16)*65535
        rgba = np.concatenate([rgb,alpha[...,None]],axis=-1)
        assert cv2.imwrite(str(out/'images'/path.name),rgba)
        f = camera['frames'][f'{i:04d}']; intr = f['intrinsics']
        records.append(dict(file_path=path.name,source_frame=i,time=(i-1)/149,
                            camera_angle_x=intr['fov_x_rad'],camera_angle_y=intr['fov_y_rad'],
                            fl_x=intr['fx'],fl_y=intr['fy'],cx=intr['cx'],cy=intr['cy'],
                            transform_matrix=f['extrinsics']['camera_to_world']))
        hashes[path.name] = sha(path)
        stats.append(dict(frame=i,background_bgr=bg.tolist(),foreground_fraction=float(foreground.mean()),
                          derived_mask_sha256=hashlib.sha256(alpha.tobytes()).hexdigest()))
    train = [i for i in range(1,151) if i%10 and not 61<=i<=79]
    dev = [i for i in range(61,80) if i%10]
    test = [i for i in range(10,151,10)]
    common = dict(w=800,h=800,per_frame_intrinsics=True,rgba16=True)
    for name, ids in [('train',train),('dev',dev),('test',test),('all',list(range(1,151)))]:
        write(out/f'transforms_{name}.json',dict(**common,frames=[r for r in records if r['source_frame'] in ids]))
    rng = np.random.RandomState(0)
    xyz = rng.random((100000,3))*2.6-1.3
    colors = ((rng.random((100000,3))/255*.28209479177387814+.5)*255).astype(np.uint8)
    vertex = np.zeros(100000,dtype=[(k,'f4') for k in ('x','y','z','nx','ny','nz')]+[(k,'u1') for k in ('red','green','blue')])
    for j,k in enumerate(('x','y','z')): vertex[k]=xyz[:,j]
    for j,k in enumerate(('red','green','blue')): vertex[k]=colors[:,j]
    PlyData([PlyElement.describe(vertex,'vertex')]).write(out/'points3d.ply')
    write(out/'rgb_only_manifest.json',dict(dataset=a.source.name,allowed=['原始RGB的三个颜色通道','camera.json','随机数'],
          gt_read=False,original_alpha_used=False,mask_method='边界8像素中位RGB，最大通道距离<=2/65535且4邻域连通到边界视为背景',
          train_frames=train,dev_frames=dev,test_frames=test,statistics=stats,raw_rgb_files_sha256=hashes,
          camera_sha256=sha(a.source/'camera.json'),initialization='RandomState(0)，100k随机立方体',
          files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}))
    print('RGB-only输入准备完成：TRAIN117，DEV18，TEST15；Alpha不参与监督，未读取独立GT属性。',flush=True)


if __name__ == '__main__':
    main()
