#!/usr/bin/env python3
"""从已完成评价生成全帧与固定TEST局部接触表，不重渲染、不改指标。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    source, out = Path(args.evaluation).resolve(), Path(args.output).resolve()
    root = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(root / 'output') or out.is_relative_to(source):
        raise ValueError('接触表必须写入仓库output下独立于源评价的新目录。')
    if out.exists():
        raise ValueError('禁止覆盖已有目检接触表。')
    if not (source / 'COMPLETED.json').is_file():
        raise ValueError('只能读取完整评价。')
    completed = json.loads((source / 'COMPLETED.json').read_text())
    if (completed.get('status') != 'PASS_EXECUTION_ONLY' or completed.get('frames') != 15
            or completed.get('video_frames') != 150):
        raise ValueError('评价执行完成状态或固定帧数不符。')
    out.mkdir(parents=True)
    hashes, pages = {}, []

    def load(path):
        digest = sha(path)
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError('重复读取时输入改变：' + str(path))
        hashes.setdefault(str(path), digest)
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(str(path))
        return image

    def save(name, canvas, inputs, scope):
        path = out / name
        if not cv2.imwrite(str(path), canvas):
            raise IOError(str(path))
        pages.append(dict(path=str(path), sha256=sha(path), inputs=inputs, scope=scope))

    manifest_path = source / 'four_classes/manifest.json'
    hashes[str(manifest_path)] = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for category in manifest['videos']:
        # 每页50个连续时刻；保留整张面板，原PNG仍为全分辨率。
        for start in (1, 51, 101):
            canvas = np.zeros((10 * 138, 5 * 480, 3), np.uint8)
            inputs = []
            for index, frame in enumerate(range(start, start + 50)):
                path = source / 'four_classes' / category / f'frame_{frame:04d}.png'
                image = load(path)
                scale = min(480 / image.shape[1], 114 / image.shape[0])
                width, height = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                y, x = (index // 5) * 138, (index % 5) * 480
                cv2.putText(canvas, f'{category} {frame:04d}', (x + 5, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                left, top = x + (480 - width) // 2, y + 24 + (114 - height) // 2
                canvas[top:top + height, left:left + width] = image
                inputs.append(str(path))
            save(f'{category}_{start:04d}_{start + 49:04d}.png', canvas, inputs,
                 '全150帧逐帧缩略总览；不能替代细节或视频实时播放。')
    categories = ('rgb', 'normal', 'depth_mesh', 'separation_components')
    for region in ('human', 'leg', 'foot', 'shoe'):
        for row, category in enumerate(categories):
            for first in (10, 60, 110):
                inputs, entries = [], []
                for frame in range(first, first + 50, 10):
                    path = source / 'review_crops' / f'{region}_{frame:04d}.png'
                    if path.is_file():
                        image = load(path)
                        if image.shape[:2] != (1152, 1024):
                            raise ValueError('局部四行面板尺寸改变：' + str(path))
                        entries.append((frame, image[row * 288:(row + 1) * 288], None))
                        inputs.append(str(path))
                    else:
                        record = next(r for r in manifest['frames'] if r['frame'] == frame)
                        if record['crops'][region]['path'] is not None:
                            raise ValueError('manifest登记的局部图缺失：' + str(path))
                        entries.append((frame, None, record['crops'][region]['status']))
                canvas = np.zeros((5 * 312, 1024, 3), np.uint8)
                for index, (frame, image, missing) in enumerate(entries):
                    cv2.putText(canvas, f'{region} {category} frame {frame:04d}', (8, index * 312 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                    if image is not None:
                        canvas[index * 312 + 24:(index + 1) * 312] = image
                    else:
                        cv2.putText(canvas, 'MISSING: ' + str(missing), (8, index * 312 + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 255), 1)
                save(f'{region}_{category}_{first:04d}_{first + 40:04d}.png', canvas, inputs,
                     '固定15TEST逐类局部原尺寸面板；未按候选结果挑帧。')
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise ValueError('生成期间输入改变。')
    receipt = dict(status='PASS_CONTACT_GENERATION_ONLY', evaluation=str(source),
                   argv=sys.argv, source_sha256=hashes, code_sha256=sha(Path(__file__)),
                   pages=pages, manual_review='PENDING', model_selection_modified=False,
                   limitation='接触表无时间播放；Mesh/GT仅15TEST，其余135帧未重建；模型gate不是GT分割。')
    (out / 'manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    (out / 'README.md').write_text('# 只读目检接触表\n\n完整命令见manifest.json的argv，输入和输出逐文件SHA已保存。'
        '\n\n全150帧每类三页；15TEST每个语义区域逐类三页。原图和原视频均保留。'
        '\n\n生成结论PASS，仅文件处理；人工目检PENDING，不代表模型质量通过。\n')
    print(json.dumps(dict(status=receipt['status'], pages=len(pages)), ensure_ascii=False))


if __name__ == '__main__':
    main()
