#!/usr/bin/env python3
"""核验检查点有限性、完整解码视频，并生成首中尾代表图供真实目检。"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
import torch

out = Path(sys.argv[1])
model = out / 'model_mlp'
review = out / 'inspection'
review.mkdir(exist_ok=True)
report = {'视频': [], '检查点': [], '代表图': [], '人工目检': 'PENDING'}
errors = []
for path in sorted(model.glob('point_cloud/iteration_*/point_cloud.ply')):
    if path.parent.name not in {'iteration_35000', 'iteration_55000'}:
        continue
    vertices = PlyData.read(path)['vertex']
    for name in vertices.data.dtype.names:
        if not np.isfinite(vertices[name]).all():
            errors.append(str(path) + '非有限字段：' + name)
    report['检查点'].append({'文件': str(path), '高斯数': len(vertices), '有限性': True})


def check_tensors(value, label):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            errors.append(label + '非有限张量')
    elif isinstance(value, dict):
        for key, item in value.items():
            check_tensors(item, label + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_tensors(item, label + '/' + str(index))


for path in sorted(model.rglob('*.pth')):
    if '35000' not in str(path) and '55000' not in str(path):
        continue
    check_tensors(torch.load(path, map_location='cpu', weights_only=False), str(path))
    report['检查点'].append({'文件': str(path), '张量检查': '完成'})
for path in sorted(model.rglob('*.mp4')):
    cap = cv2.VideoCapture(str(path))
    expected = 72 if path.stem == 'rotating_directional' else 150
    frames = []
    count = 0
    while True:
        ok, pixels = cap.read()
        if not ok:
            break
        if count in {0, expected // 2, expected - 1}:
            frames.append(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
        count += 1
    cap.release()
    if count != expected:
        errors.append(f'{path}解码帧数{count}，预期{expected}')
    record = {'文件': str(path), '解码帧数': count, '预期帧数': expected}
    if frames:
        tile = 320
        canvas = Image.new('RGB', (tile * len(frames), tile + 28), '#181818')
        draw = ImageDraw.Draw(canvas)
        for index, frame in enumerate(frames):
            frame.thumbnail((tile, tile))
            canvas.paste(frame, (index * tile, 28))
        draw.text((8, 8), path.stem, fill='white')
        name = '__'.join(path.relative_to(model).parts).replace('.mp4', '.jpg')
        target = review / name
        canvas.save(target)
        record['代表图'] = str(target)
        report['代表图'].append(str(target))
    report['视频'].append(record)
report['图像数'] = len(list(model.rglob('*.png')))
report['错误'] = errors
report['产物核验'] = 'FAILED' if errors else 'PASS'
(review / 'artifacts.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)
