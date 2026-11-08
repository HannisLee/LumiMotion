"""T01(1107-03) I2：Stage 2 E4 光向 -> 原生PBR方向光 JSON（FIT/DEV/TEST 三份）。

只读 E4 冻结预测与相机变换，不读任何 GT；仅依赖 numpy/json/hashlib，可在无 torch 环境自检。
断言见规划 1.2 节 A5/A6/A8/A10。缺任何断言即抛错，不静默回退。
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E4_DIR = ROOT / 'output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角'
CAMERA_REFERENCE = ROOT / 'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/source_all/transforms_all.json'
EXPECTED_SELECTED = 'a1e+00_e1e+01'
NORM_TOL_FLOAT64 = 1e-10   # A5：E4 原始预测的单位范数容差
NORM_TOL_FLOAT32 = 1e-4    # A8：pbr_directional.py:45-47 转 float32 后的容差
CAMERA_TOL = 1e-5          # A10


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_frames():
    """FIT117 = {1..150} 去掉 10 的倍数与 61..79。"""
    return [i for i in range(1, 151) if i % 10 and not 61 <= i <= 79]


def dev_frames():
    """DEV18 = 61..79 去掉 70。"""
    return [i for i in range(61, 80) if i % 10]


def test_frames():
    """TEST15 = 10,20,...,150。"""
    return list(range(10, 151, 10))


def split_frames():
    fit = fit_frames()
    return dict(train=fit, dev=fit + dev_frames(), test=fit + test_frames())


def repository_convention(root=ROOT):
    """从 scene/pbr_directional.py 文本读取 CONVENTION，避免导入 torch。"""
    text = (Path(root) / 'scene/pbr_directional.py').read_text()
    match = re.search(r"^CONVENTION\s*=\s*'([^']*)'", text, re.M)
    if not match:
        raise ValueError('scene/pbr_directional.py 中找不到 CONVENTION。')
    return match.group(1)


def load_e4(e4_dir=E4_DIR, key='E4', check_manifest=True, expected_selected=EXPECTED_SELECTED, root=ROOT):
    """A5/A6：读取 E4 预测并逐条断言；返回 (frames, vectors, info)。"""
    e4_dir = Path(e4_dir)
    pred_path = e4_dir / 'fitting/predictions.npz'
    config_path = e4_dir / 'fitting/method_config.json'
    freeze_path = e4_dir / 'freeze_manifest.json'
    info = dict(e4_dir=str(e4_dir), predictions=str(pred_path), key=key)
    info['predictions_sha256'] = sha256_file(pred_path)
    if check_manifest:
        freeze = json.loads(freeze_path.read_text())
        if freeze.get('prediction_sha256') != info['predictions_sha256']:
            raise ValueError('A5：predictions.npz 的 SHA256 与 freeze_manifest.prediction_sha256 不一致。')
        if freeze.get('gt_read') is not False:
            raise ValueError('A5：freeze_manifest 未声明 gt_read=false。')
        if freeze.get('direction_convention') != 'surface_to_light' or freeze.get('coordinate_system') != 'world':
            raise ValueError('A6：freeze_manifest 方向约定/坐标系不是 world surface_to_light。')
        info['freeze_manifest_sha256'] = sha256_file(freeze_path)
        info['freeze_selected'] = freeze.get('selected')
    config = json.loads(config_path.read_text())
    if config.get('direction_convention') != 'surface_to_light' or config.get('coordinate_system') != 'world':
        raise ValueError('A6：method_config 方向约定/坐标系不是 world surface_to_light。')
    if expected_selected is not None and config.get('selected') != expected_selected:
        raise ValueError(f'A6：method_config.selected={config.get("selected")!r}，预期 {expected_selected!r}。')
    if config.get('gt_read') is not False:
        raise ValueError('A6：method_config 未声明 gt_read=false。')
    convention = repository_convention(root)
    if 'toward light' not in convention:
        raise ValueError('A6：仓库 CONVENTION 不是表面指向光源：' + convention)
    info.update(method_config_sha256=sha256_file(config_path), selected=config.get('selected'),
                repository_convention=convention)
    with np.load(pred_path) as data:
        if 'frames' not in data.files or key not in data.files:
            raise ValueError(f'A5：predictions.npz 缺少 frames/{key}。')
        frames = data['frames'].astype(np.int64)
        vectors = data[key]
    if vectors.dtype != np.float64 or vectors.shape != (150, 3):
        raise ValueError(f'A5：{key} 形状/类型应为 (150,3) float64，实际 {vectors.shape} {vectors.dtype}。')
    if frames.shape != (150,) or not np.array_equal(frames, np.arange(1, 151)):
        raise ValueError('A5：frames 必须恰为 1..150 且递增。')
    if not np.isfinite(vectors).all():
        raise ValueError('A5：E4 含非有限值。')
    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.abs(norms - 1).max())
    if worst > NORM_TOL_FLOAT64:
        raise ValueError(f'A5：E4 行范数偏离 1 最大 {worst:.3e} > {NORM_TOL_FLOAT64}。')
    info['max_norm_deviation'] = worst
    return frames, vectors, info


def check_camera_consistency(reference, sources, tol=CAMERA_TOL):
    """A10：各 split 的相机矩阵与 E4 提取所用 transforms_all 逐帧一致。

    reference：transforms_all.json 路径；sources：{名称: transforms 路径}。返回逐文件最大差值。
    """
    ref = json.loads(Path(reference).read_text())['frames']
    ref_matrix = {int(f['source_frame']): np.asarray(f['transform_matrix'], dtype=np.float64) for f in ref}
    if sorted(ref_matrix) != list(range(1, 151)):
        raise ValueError('A10：参考相机表不是 1..150。')
    result = {}
    for name, path in sources.items():
        worst = 0.
        count = 0
        for frame in json.loads(Path(path).read_text())['frames']:
            key = int(frame['source_frame'])
            diff = float(np.abs(np.asarray(frame['transform_matrix'], dtype=np.float64) - ref_matrix[key]).max())
            worst = max(worst, diff)
            count += 1
        if worst > tol:
            raise ValueError(f'A10：{name} 相机与 E4 提取相机最大差 {worst:.3e} > {tol}，停止并诊断。')
        result[name] = dict(frames=count, max_abs_diff=worst)
    return result


def make_payload(frames, vectors, ids, info, split):
    index = {int(f): i for i, f in enumerate(frames)}
    out = {}
    for i in ids:
        if i not in index:
            raise ValueError(f'A9：E4 缺少帧 {i}。')
        v = vectors[index[i]].astype(np.float64)
        v = v / np.linalg.norm(v)
        v32 = v.astype(np.float32)
        if abs(float(np.linalg.norm(v32.astype(np.float64))) - 1) > NORM_TOL_FLOAT32:
            raise ValueError(f'A8：帧 {i} 转 float32 后范数偏离过大。')
        out[f'{i:04d}'] = {'light_dir_world': [float(x) for x in v], 'intensity': 1.0,
                           'type': 'directional', 'source_type': 'SUN'}
    return {'coordinate_space': 'world', 'frames': out, 'gt_directions_used': False, 'gt_read': False,
            'direction_source': f'Stage2 E4 ({info["selected"]})，{info["predictions"]}',
            'predictions_sha256': info['predictions_sha256'], 'split': split,
            'intensity_source': '单位初值；Stage 3 只学习一个正标量光强，不使用 GT 光强'}


def convert(out_dir, e4_dir=E4_DIR, key='E4', check_manifest=True, expected_selected=EXPECTED_SELECTED, root=ROOT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, vectors, info = load_e4(e4_dir, key, check_manifest, expected_selected, root)
    record = dict(info, files={}, splits={})
    for split, ids in split_frames().items():
        payload = make_payload(frames, vectors, ids, info, split)
        path = out_dir / f'e4_{split}_directions.json'
        if path.exists():
            raise FileExistsError('禁止覆盖：' + str(path))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')
        record['files'][path.name] = sha256_file(path)
        record['splits'][split] = len(ids)
    record['gt_read'] = False
    (out_dir / 'e4_directions_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--e4-dir', type=Path, default=E4_DIR)
    p.add_argument('--key', default='E4')
    p.add_argument('--expected-selected', default=EXPECTED_SELECTED)
    p.add_argument('--camera-reference', type=Path, default=CAMERA_REFERENCE)
    p.add_argument('--camera-sources', type=Path, nargs='*', default=[], help='待核对的 transforms_*.json')
    a = p.parse_args()
    record = convert(a.output, a.e4_dir, a.key, True, a.expected_selected)
    if a.camera_sources:
        record['camera_check'] = check_camera_consistency(a.camera_reference, {str(s): s for s in a.camera_sources})
        (a.output / 'e4_directions_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
