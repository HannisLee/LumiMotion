#!/usr/bin/env python3
"""Stage2专用输入：只读取原RGB/RGBA和相机，不打开任何材质/几何GT。"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(source, reference, output):
    source, reference, output = [Path(p).resolve() for p in (source, reference, output)]
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出必须为空，禁止覆盖旧数据。')
    camera_path = source / 'camera.json'
    cameras = json.loads(camera_path.read_text())['frames']
    transforms = {split: json.loads((reference / ('transforms_'+split+'.json')).read_text())
                  for split in ('train', 'test')}
    numbers = [[int(f['source_frame']) for f in transforms[s]['frames']] for s in ('train', 'test')]
    if len(numbers[0]) != 135 or numbers[1] != list(range(10, 151, 10)):
        raise ValueError('必须保留原135/15训练测试划分。')
    if len(set(numbers[0]+numbers[1])) != 150 or set(numbers[0]+numbers[1]) != set(range(1,151)):
        raise ValueError('帧重复或缺失。')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir()
    records = {}
    for split, data in transforms.items():
        for f in data['frames']:
            n = int(f['source_frame']); key = '%04d' % n; c = cameras[key]
            if not np.allclose(f['transform_matrix'], c['extrinsics']['camera_to_world'], atol=1e-10, rtol=0):
                raise ValueError('相机与原相机元数据不同：'+key)
            if not np.isclose(f['time'], c['time_normalized'], atol=1e-10, rtol=0):
                raise ValueError('时间与原相机元数据不同：'+key)
            for field, original in [('fl_x','fx'),('fl_y','fy'),('camera_angle_x','fov_x_rad'),('camera_angle_y','fov_y_rad')]:
                if not np.isclose(f[field], c['intrinsics'][original], atol=1e-10, rtol=0):
                    raise ValueError('内参与原相机元数据不同：'+key)
            raw = source / 'image' / ('image_'+key+'.png')
            with Image.open(raw) as image:
                if image.mode != 'RGBA':
                    raise ValueError('原始图像必须有自身Alpha，禁止读取其他GT补齐：'+str(raw))
                pixels = np.array(image, dtype=np.uint8)
            alpha = pixels[...,3]
            if pixels.shape != (int(data['h']),int(data['w']),4) or not alpha.any():
                raise ValueError('原始RGBA形状或Alpha无效：'+key)
            pixels[alpha == 0,:3] = 0
            target = output / 'images' / f['file_path']
            if target.parent != output / 'images':
                raise ValueError('非法图像路径')
            Image.fromarray(pixels, 'RGBA').save(target)
            records[key] = dict(source_image=str(raw), source_image_sha256=sha(raw),
                                image_sha256=sha(target), split=split,
                                alpha_nonzero_pixels=int(np.count_nonzero(alpha)))
        (output / ('transforms_'+split+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    # 仅保持Scene reader的来源点数元数据；Stage2实际从指定35k预测PLY初始化。
    random_ply = reference / 'points3d.ply'
    shutil.copy2(random_ply, output / 'points3d.ply')
    manifest = dict(format='LumiMotion Stage2 original RGB/RGBA only',
                    source_scene=str(source), generated_root=str(output),
                    summary=dict(frames=150,train_frames=135,test_frames=15,
                                 mask_source='original-image-alpha',background='black'),
                    training_inputs=['original RGB/RGBA','camera intrinsics/extrinsics','time'],
                    material_geometry_GT_opened=False,
                    camera_json_sha256=sha(camera_path),
                    inherited_scene_reader_random_ply=dict(path=str(random_ply),sha256=sha(random_ply),
                        used_for_stage2_geometry_initialization=False),
                    frames=records)
    (output / 'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--reference_view',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=convert(args.source,args.reference_view,args.output)
    print(json.dumps(manifest['summary'],ensure_ascii=False))
