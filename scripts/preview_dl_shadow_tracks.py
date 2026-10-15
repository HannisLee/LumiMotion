"""将投影代理在固定观测上的预测可视化；不冒充完整物理渲染。"""
import argparse
import hashlib
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('禁止覆盖可视化。')
    torch.set_num_threads(4)
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert hashlib.sha256(Path(args.cache).read_bytes()).hexdigest() == state['cache_sha256']
    # 预先固定早／中／晚三个保留时间，不按误差挑图。
    held = torch.where(~state['train'])[0]
    indices = held[[0, len(held)//2, len(held)-1]].tolist()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, index in enumerate(indices):
        name = data['frames'][index]['name']
        # 缓存只保存部分整图，尺寸取同一数据集已有缓存图。
        image = Image.open(next(Path(args.cache).parent.glob('*_rgb.png')))
        width, height = image.size
        mask = state['positive'][index] | state['negative'][index]
        uv = data['pixels'][index][mask]
        target = state['positive'][index][mask].float()
        values = [target, state['predicted']['rgb_baseline'][index][mask],
                  state['predicted']['smooth'][index][mask],
                  (state['predicted']['smooth'][index][mask]-target).abs()]
        for col, value in enumerate(values):
            ax = axes[row, col]
            ax.set_facecolor('#aaaaaa')
            ax.scatter(uv[:, 0], uv[:, 1], c=value, cmap='gray_r', vmin=0, vmax=1, s=4)
            ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.set_aspect('equal')
            ax.set_title(name+' '+['observed dark labels', 'RGB light shadow', 'T05 light shadow', 'T05 disagreement'][col])
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Fixed held-time samples; black = shadow / disagreement, gray = unobserved')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
