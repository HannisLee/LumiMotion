"""显示实际拟合的表面采样点；这不是完整图像重渲染。"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.dl_selfcal import SurfaceLightFit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('禁止覆盖采样点可视化。')
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if hashlib.sha256(Path(args.cache).read_bytes()).hexdigest() != state['cache_sha256']:
        raise ValueError('缓存身份不匹配。')
    weights = state['model']
    model = SurfaceLightFit(weights['times'], weights['raw_directions'], len(data['ids']))
    model.load_state_dict(weights)
    indices = [0, len(data['times'])//2, len(data['times'])-1]
    with torch.no_grad():
        prediction = model(data['times'][indices], data['normals'][indices])
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    for row, index in enumerate(indices):
        name = data['frames'][index]['name']
        width, height = Image.open(Path(args.cache).parent / f'{name}_rgb.png').size
        mask = data['valid'][index] & state['supported']
        uv = data['pixels'][index][mask]
        target = data['rgb'][index][mask]
        pred = prediction[row][mask]
        error = (pred-target).abs().mean(-1)
        for col, value in enumerate([target, pred, error[:, None].expand(-1, 3)]):
            if col < 2:
                value = torch.where(value <= .0031308, 12.92*value, 1.055*value.clamp_min(.0031308).pow(1/2.4)-.055)
            canvas = np.zeros((height, width, 3), dtype=np.float32)
            canvas[uv[:, 1], uv[:, 0]] = value.clamp(0, 1).numpy()
            axes[row, col].imshow(canvas)
            axes[row, col].set_title(name + ' ' + ['observed samples', 'predicted samples', 'absolute linear error'][col])
            axes[row, col].axis('off')
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)


if __name__ == '__main__':
    main()
