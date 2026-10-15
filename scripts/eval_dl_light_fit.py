"""训练后隔离评测灯向；只读拟合结果，GT 不写回任何训练参数。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True)
    p.add_argument('--gt', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('禁止覆盖离线评测。')
    result = json.loads(Path(args.result).read_text())
    gt = json.loads(Path(args.gt).read_text())['frames']
    truth = []
    for name in result['frames']:
        key = str(int(name.rsplit('_', 1)[-1])).zfill(4)
        lights = gt[key]['lights']
        directions = []
        for light in lights:
            if light.get('type') != 'directional' or 'surface toward light' not in light.get('direction_convention', ''):
                raise ValueError('GT灯类型或方向约定不受支持。')
            d = np.asarray(light['light_dir_world'], dtype=float)
            directions.append(d / np.linalg.norm(d))
        if not all(np.allclose(d, directions[0], atol=1e-5) for d in directions):
            raise ValueError('多灯方向不重合，不能将单主光对比任意一盏GT。')
        truth.append(directions[0])
    prediction = np.asarray(result['directions'])
    prediction /= np.linalg.norm(prediction, axis=-1, keepdims=True)
    errors = np.rad2deg(np.arccos(np.clip((prediction*np.asarray(truth)).sum(-1), -1, 1)))
    stats = {}
    for name, indices in [('fit', result['train_indices']), ('held_time', result['held_indices'])]:
        e = errors[indices]
        stats[name] = dict(mean=float(e.mean()), median=float(np.median(e)), p90=float(np.quantile(e, .9)),
                           maximum=float(e.max()), over_30_degrees=int((e>30).sum()), count=len(e))
    record = dict(stats=stats, frames=result['frames'], angle_degrees=errors.tolist(),
                  result_sha256=hashlib.sha256(Path(args.result).read_bytes()).hexdigest(),
                  gt_sha256=hashlib.sha256(Path(args.gt).read_bytes()).hexdigest(),
                  gt_path=str(Path(args.gt).resolve()), protocol='最终固定600步结果独立评测，不回灌训练',
                  energy_evaluation='未执行：当前只评方向，不把Blender能量直接当辐射标定')
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
