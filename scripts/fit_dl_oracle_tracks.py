"""隔离的 GT 光向参考组，未知光入口不调用本模块。"""
import json
import hashlib
from pathlib import Path
import torch
from scripts.fit_dl_surface_tracks import argument_parser, fit


def main():
    parser = argument_parser()
    parser.add_argument('--gt-directions', required=True)
    args = parser.parse_args()
    data = torch.load(args.cache, weights_only=False, map_location='cpu')
    raw = Path(args.gt_directions).read_bytes()
    gt = json.loads(raw)['frames']
    directions = []
    for frame in data['frames']:
        key = str(int(frame['name'].rsplit('_', 1)[-1])).zfill(4)
        lights = gt[key]['lights']
        if any(light.get('type') != 'directional' or 'surface toward light' not in light.get('direction_convention', '') for light in lights):
            raise ValueError('参考组只支持约定一致的远场光。')
        vectors = torch.tensor([light['light_dir_world'] for light in lights])
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        if not torch.allclose(vectors, vectors[:1].expand_as(vectors), atol=1e-5):
            raise ValueError('非同向多灯不能合并。')
        directions.append(vectors[0])
    args.gt_sha256 = hashlib.sha256(raw).hexdigest()
    # fit 只将拟合时间的参考方向固定为参数；保留时间沿用训练节点插值规则。
    fit(args, oracle_directions=torch.stack(directions))


if __name__ == '__main__':
    main()
