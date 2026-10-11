#!/usr/bin/env python3
"""固定静态 DL Goal 的原协议、完整划分及评测来源；只执行 CPU 审计。"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python'
BASE = ROOT / 'output/1011-01-LH-static-jumpingjacks_DL_V2-原生LumiMotion两阶段'
RAW = Path('/mnt/workspace/users/han.li/lumimotion/data/LH-data/static/LH-static-jumpingjacks_DL_V2')
GT_DEPTH = ROOT / 'output/1011-07-六组训练总控/GT-LH-static-jumpingjacks_DL_V2/gt_depth'
CONTROL = ROOT / 'output/1011-08-LH-static-jumpingjacks_DL_V2-Goal评测协议'
FRAMES = list(range(10, 151, 10))
FIXED_CODE = ['scripts/eval_geometry_gt.py', 'scripts/extract_geometry_mesh_dynamic.py',
              'utils/geometry_eval_utils.py', 'utils/predicted_mesh.py',
              'LIHAN/scripts/audit_1011_mae.py', 'LIHAN/scripts/bootstrap_1011.py']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify(protocol):
    for group in ('code_sha256', 'source_sha256'):
        for path, expected in protocol[group].items():
            if digest(path) != expected:
                raise AssertionError('固定评测来源已改变：' + path)
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=CONTROL / 'protocol.json')
    args = p.parse_args()
    if subprocess.check_output(['hostname']).decode().strip() != 'ushas':
        raise RuntimeError('当前执行器只允许ushas/lumimotion-ushas。')
    if args.output.exists():
        raise FileExistsError('协议禁止覆盖：' + str(args.output))
    import numpy as np
    records = {}
    files = {}
    for split in ('train', 'test'):
        path = BASE / 'eval_view' / f'transforms_{split}.json'
        records[split] = read(path)['frames']
        files[str(path)] = digest(path)
    train_ids = [r['source_frame'] for r in records['train']]
    test_ids = [r['source_frame'] for r in records['test']]
    if len(train_ids) != 135 or len(set(train_ids)) != 135 or test_ids != FRAMES:
        raise AssertionError('必须严格135训练／15固定测试。')
    if set(train_ids) & set(test_ids) or set(train_ids + test_ids) != set(range(1, 151)):
        raise AssertionError('划分必须互斥并完整覆盖1..150。')
    camera = read(RAW / 'camera.json')['frames']
    files[str(RAW / 'camera.json')] = digest(RAW / 'camera.json')
    for r in records['train'] + records['test']:
        n = r['source_frame']
        if not np.allclose(r['transform_matrix'], camera[f'{n:04d}']['extrinsics']['camera_to_world'], atol=1e-5, rtol=0):
            raise AssertionError('真实c2w不符：' + str(n))
        image = BASE / 'eval_view/images' / r['file_path']
        if not image.suffix:
            image = image.with_suffix('.png')
        files[str(image)] = digest(image)
        if n in FRAMES:
            normal = RAW / 'normal_exr' / f'normal_{n:04d}.exr'
            files[str(normal)] = digest(normal)
    manifest = read(GT_DEPTH / 'manifest.json')
    files[str(GT_DEPTH / 'manifest.json')] = digest(GT_DEPTH / 'manifest.json')
    for n in FRAMES:
        entry = manifest['frames'][str(n)]
        path = GT_DEPTH / entry['file']
        if digest(path) != entry['sha256']:
            raise AssertionError('原GT深度哈希不符。')
        files[str(path)] = entry['sha256']
    protocol = dict(version=1, server='ushas', python=PYTHON,
        source_path=str(BASE / 'eval_view'), gt_root=str(RAW), gt_depth_dir=str(GT_DEPTH),
        split='test', frames=FRAMES, train_frames=train_ids, resolution=2,
        alpha_threshold=.5, interior_radius=2,
        tsdf=dict(voxel_size=.02, sdf_trunc=.08, depth_trunc=20., alpha_threshold=.5),
        primary_region='foreground', aggregation='mae_pixel_mean_deg',
        thresholds=dict(gs_render=3.51, depth_normal=7.35, mesh=5.46), strict_less_than=True,
        masks='沿用旧eval_view Alpha；GT alpha>=0.5与GT ray hit；Depth法线沿用原3×3支持腐蚀。',
        missing='沿用原有效交集条件MAE；同时记录coverage、共同像素与缺失180°惩罚MAE。',
        mesh='每测试时刻独立TSDF，仅融合全部135训练相机预测Expected Z；不使用测试相机重建。',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        code_sha256={str(ROOT / f): digest(ROOT / f) for f in FIXED_CODE}, source_sha256=files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, protocol)
    print('协议固定：' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
