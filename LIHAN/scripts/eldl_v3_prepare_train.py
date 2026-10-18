"""仅从原始 TRAIN RGBA、相机和帧时间建立 V3 无 GT 输入与图像统计。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import cv2
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-name', default='train_audit_v2')
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    images = out/'images'
    images.mkdir()
    camera = json.loads((source/'camera.json').read_text())
    ids = sorted(int(k) for k in camera['frames'])
    assert ids == list(range(1,151))
    train = [i for i in ids if i % 10]
    test = [i for i in ids if not i % 10]
    # 保留一个完整相机/时间块，估光与几何初始化均不得训练该块。
    held = [i for i in train if 61 <= i <= 79]
    fit = [i for i in train if i not in held]
    width, height = (int(camera['resolution'][k]) for k in ('width','height'))
    records, stats, input_sha, tiles = [], [], {}, []
    selected = [1,21,41,61,81,101,121,141]
    for i in train:
        key = f'{i:04d}'
        paths = list((source/'image').glob(f'*{key}.png'))
        if len(paths) != 1:
            raise ValueError(f'帧{key}输入不唯一：{paths}')
        path = paths[0]
        raw = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        assert raw.dtype == np.uint16 and raw.shape == (height,width,4)
        rgba = raw[...,[2,1,0,3]].copy()
        # 导出器把原始前景Alpha打包在albedo PNG。这里只取Alpha，丢弃所有材质RGB。
        alpha_path = source/'albedo'/f'albedo_{key}.png'
        alpha = cv2.imread(str(alpha_path),cv2.IMREAD_UNCHANGED)[...,3].copy()
        assert alpha.dtype == np.uint16 and alpha.shape == (height,width)
        rgba[...,3] = alpha
        fg = alpha > 32767
        rgb = rgba[...,:3]/65535.
        lin = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        lum = lin @ np.array([.2126,.7152,.0722])
        srgb_mean = rgb.mean(-1)
        stats.append(dict(frame=i,foreground_pixels=int(fg.sum()),
                          srgb_mean_quantiles=np.quantile(srgb_mean[fg],[0,.01,.05,.1,.5,.9,.99,1]).tolist(),
                          assumed_linear_luminance_quantiles=np.quantile(lum[fg],[0,.01,.05,.1,.5,.9,.99,1]).tolist(),
                          dark_srgb_mean={str(t):int((fg&(srgb_mean<t)).sum()) for t in (.02,.05,.1,.2)},
                          saturated_any_channel=int((fg&(rgba[...,:3]==65535).any(-1)).sum()),
                          saturated_all_channels=int((fg&(rgba[...,:3]==65535).all(-1)).sum()),
                          alpha_nonbinary=int(((alpha>0)&(alpha<65535)).sum()),
                          alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest()))
        input_sha[path.name] = sha(path)
        rgba[rgba[...,3]==0,:3] = 0
        assert cv2.imwrite(str(images/path.name),rgba[...,[2,1,0,3]])
        f = camera['frames'][key]
        c2w = np.asarray(f['extrinsics']['camera_to_world'])
        w2c = np.asarray(f['extrinsics']['world_to_camera'])
        assert np.allclose(c2w@w2c,np.eye(4),atol=1e-5)
        intr = f['intrinsics']
        records.append(dict(file_path=path.name, source_frame=i,
                            time=float(f.get('time_normalized',(i-1)/(len(ids)-1))),
                            time_seconds_from_first_frame=(i-1)/float(camera['fps']),
                            camera_angle_x=intr['fov_x_rad'],camera_angle_y=intr['fov_y_rad'],
                            fl_x=intr['fx'],fl_y=intr['fy'],cx=intr['cx'],cy=intr['cy'],
                            transform_matrix=c2w.tolist()))
        if i in selected:
            preview = np.rint(rgb*(alpha[...,None]/65535.)*255).astype(np.uint8)
            tile = Image.fromarray(preview).resize((300,300))
            ImageDraw.Draw(tile).text((8,8),f'TRAIN {key}',fill='white')
            tiles.append(tile)
    first = records[0]
    common = {k:first[k] for k in ('camera_angle_x','camera_angle_y','fl_x','fl_y','cx','cy')}
    common.update(w=width,h=height,camera_extent=1.,per_frame_intrinsics=True,rgba16=True)
    write(out/'transforms_train.json',dict(**common,frames=records))
    # 原生合成数据随机初始化的相同构造，不读取任何历史点云或模型。
    rng = np.random.RandomState(0)
    xyz = rng.random((100000,3))*2.6-1.3
    colors = ((rng.random((100000,3))/255.*.28209479177387814+.5)*255).astype(np.uint8)
    vertices = np.empty(100000,dtype=[(s,'f4') for s in ('x','y','z','nx','ny','nz')]+[(s,'u1') for s in ('red','green','blue')])
    for j,k in enumerate(('x','y','z')): vertices[k] = xyz[:,j]
    for k in ('nx','ny','nz'): vertices[k] = 0
    for j,k in enumerate(('red','green','blue')): vertices[k] = colors[:,j]
    PlyData([PlyElement.describe(vertices,'vertex')]).write(out/'points3d.ply')
    files = {str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    manifest = dict(dataset=source.name,source=str(source),gt_read=False,init_weights_loaded=False,
                    allowed_inputs=['TRAIN RGB','TRAIN Alpha','camera','frame/time'],
                    alpha_source='原导出albedo PNG的Alpha通道；禁止材质RGB进入任何下游产物',
                    alpha_channel_sha256={str(r['frame']):r['alpha_sha256'] for r in stats},
                    files=files,original_train_image_sha256=input_sha,camera_sha256=sha(source/'camera.json'),
                    train_frames=train,test_frames=test,internal_fit_frames=fit,internal_held_frames=held,
                    initialization=dict(method='原生随机100k立方体从头构造',seed=0),
                    initial_point_cloud_sha256=sha(out/'points3d.ply'),
                    resolution=[width,height],time_rule='(frame-1)/149；秒=(frame-1)/fps',
                    camera_convention=camera.get('blender_camera_convention'),
                    source_code_sha256=sha(__file__))
    write(out/'no_gt_manifest.json',manifest)
    audit = out.parent/args.audit_name
    audit.mkdir(exist_ok=False)
    total = sum(r['foreground_pixels'] for r in stats)
    write(audit/'image_statistics.json',dict(status='PASS',scope='仅135 TRAIN原始图像；阈值仅描述，不生成阴影标签',
         source=str(source),resolution=[width,height],train_count=len(train),test_count=len(test),
         color_assumption='统计同时给出原码值和暂按sRGB逆变换的亮度；最终颜色解释以导出元数据为准',
         foreground_pixels=total,
         dark_srgb_mean_fraction={str(t):sum(r['dark_srgb_mean'][str(t)] for r in stats)/total for t in (.02,.05,.1,.2)},
         saturated_any_channel_fraction=sum(r['saturated_any_channel'] for r in stats)/total,
         per_frame=stats))
    grid = Image.new('RGB',(1200,600))
    for j,tile in enumerate(tiles):grid.paste(tile,((j%4)*300,(j//4)*300))
    grid.save(audit/'train_representatives.jpg')
    print(json.dumps(dict(status='PASS',output=str(out),train=len(train),test=len(test),
                         internal_fit=len(fit),internal_held=len(held)),ensure_ascii=False))


if __name__=='__main__':
    main()
