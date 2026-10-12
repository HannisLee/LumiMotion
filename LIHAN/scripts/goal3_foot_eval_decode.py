#!/usr/bin/env python3
"""足部恢复评测产物独立CPU解码及检查点保留核验。"""
import argparse
import hashlib
from pathlib import Path
import cv2
from PIL import Image
from goal_eval_protocol import ROOT, read, write, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--fresh', type=Path, required=True)
    args = p.parse_args()
    evaluation, fresh = args.evaluation.resolve(), args.fresh.resolve()
    if not evaluation.is_relative_to(ROOT / 'output'):
        p.error('必须使用output评测产物。')
    destination = evaluation / 'artifact_decode_audit.json'
    if destination.exists():
        raise FileExistsError('独立解码记录禁止覆盖。')
    checks = {}
    for category in ('rgb', 'alpha', 'normal', 'separation'):
        folder = evaluation / 'four_classes' / category
        pngs = sorted(folder.glob('*.png'))
        if [f.name for f in pngs] != [f'{n:04d}.png' for n in range(1, 151)]:
            raise AssertionError('PNG帧集合不完整：' + category)
        for path in pngs:
            with Image.open(path) as im:
                im.load()
                if im.size != (400, 400):
                    raise AssertionError('PNG尺寸不符。')
        video = folder.parent / (category + '.mp4')
        cap, count, first, last = cv2.VideoCapture(str(video)), 0, None, None
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if pixels.shape != (400, 400, 3):
                raise AssertionError('视频尺寸不符。')
            current = hashlib.sha256(pixels.tobytes()).hexdigest()
            first = current if count == 0 else first
            last = current
            count += 1
        cap.release()
        if count != 150:
            raise AssertionError('视频完整解码帧数不足：' + category)
        checks[category] = dict(png_count=150, png_all_loaded=True,
            video_decoded_frames=count, size=[400, 400], first_frame_decoded_sha256=first,
            last_frame_decoded_sha256=last, video_sha256=digest(video), video_path=str(video))
    parity = read(fresh / 'parity.json')
    if parity['status'] != 'PASS' or parity['clock_shift_diagnostic']:
        raise AssertionError('正式16000必须真实fresh通过且未前移clock。')
    hashes = read(fresh / 'checkpoint_sha256.json')
    if any(digest(path) != expected for path, expected in hashes.items()):
        raise AssertionError('checkpoint在完整评测后变化。')
    write(evaluation / 'checkpoint_after_audit.json', dict(status='PASS', files=hashes, unchanged=True))
    write(destination, dict(status='PASS', categories=checks,
        fresh=str(fresh / 'parity.json'), fresh_sha256=digest(fresh / 'parity.json'),
        checkpoint_unchanged=True, manual_review_scope='自动完整解码不是人工逐帧目检。'))
    print('四类600PNG/4×150视频及正式checkpoint保留核验PASS。', flush=True)


if __name__ == '__main__':
    main()
