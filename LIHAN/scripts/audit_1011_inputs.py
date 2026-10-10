#!/usr/bin/env python3
"""核对原生 reader 的全部相机及输入来源、Alpha、划分与静态骨骼。"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'output/1011-07-六组训练总控'
sys.path.insert(0, str(CONTROL / 'native'))
from scene.dataset_readers import readCamerasFromTransforms

batch = json.loads((CONTROL / 'batch.json').read_text())
reports = []
for exp in batch['experiments'][:4]:
    source = Path(exp['source'])
    raw = Path(exp['raw'])
    maximum_error = 0.
    hashes = {}
    for split in ('train', 'test'):
        data = json.loads((source / f'transforms_{split}.json').read_text())
        cameras = readCamerasFromTransforms(str(source), f'transforms_{split}.json', 'images', False, no_bg=True)
        if len(cameras) != (135 if split == 'train' else 15):
            raise AssertionError('135/15划分不正确。')
        for record, camera in zip(data['frames'], cameras):
            expected = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(record['source_camera_to_world'])
            actual = np.eye(4)
            actual[:3, :3] = camera.R.T
            actual[:3, 3] = camera.T
            error = float(np.abs(expected - actual).max())
            maximum_error = max(error, maximum_error)
            if error > 1e-8 or abs(camera.FovX - record['camera_angle_x']) > 1e-8 or abs(camera.fid - record['time']) > 1e-8:
                raise AssertionError(f'原生reader相机／时间不符：{record["source_frame"]}')
            n = record['source_frame']
            input_path = source / 'images' / record['file_path']
            rgba = np.asarray(Image.open(input_path))
            image_path = raw / 'image' / f'image_{n:04d}.png'
            alpha_path = raw / 'albedo' / f'albedo_{n:04d}.png'
            original = np.asarray(Image.open(image_path).convert('RGBA'))
            alpha = np.asarray(Image.open(alpha_path).convert('RGBA'))[..., 3]
            if not np.array_equal(rgba[..., 3], alpha):
                raise AssertionError('软Alpha转换不符。')
            reference_rgb = original[..., :3].copy()
            reference_rgb[alpha == 0] = 0
            if not np.array_equal(rgba[..., :3], reference_rgb):
                raise AssertionError('输入RGB转换不符。')
            for path in (image_path, alpha_path, input_path, raw / 'normal_exr' / f'normal_{n:04d}.exr'):
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if split == 'test' and [f['source_frame'] for f in data['frames']] != list(range(10, 151, 10)):
            raise AssertionError('测试帧不符。')
    reports.append(dict(dataset=exp['dataset'], status='PASS', frames=150,
                        maximum_reader_camera_error=maximum_error, source_hashes=hashes,
                        conversion_manifest=json.loads((source / 'dataset_manifest.json').read_text())))
    print(exp['dataset'], 'PASS', maximum_error, flush=True)
(CONTROL / 'input_audit.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
