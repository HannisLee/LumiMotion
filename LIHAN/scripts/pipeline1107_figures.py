"""T01(1107-03) I8：收敛曲线叠加图与 GT|E0|E1|E2|误差 并排图。

  curves   --run E0=<臂目录> --run E1=... --run E2=... --output <png>   读各臂 model_mlp/metrics.jsonl
  sidebyside --eval E0=<test评测目录> --eval E1=... --eval E2=... --output <目录>
             每个 TEST 帧一张 GT | E0 | E1 | E2 | |E1-GT| | |E2-GT| (误差色标统一)，另有总览联系表；
             先断言各组帧集合一致且 GT 数组逐元素相同(同相机同目标的必要条件)；E0 缺失时对应列留空。
  panels   --eval E0=... --eval E1=... --eval E2=... --frames 10 80 150 --kind albedo|normal --output <png>
"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_named(items):
    out = {}
    for item in items or []:
        name, path = item.split('=', 1)
        out[name] = Path(path)
    return out


def read_metrics(run):
    path = Path(run) / 'model_mlp/metrics.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def curves(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = {k: read_metrics(v) for k, v in parse_named(a.run).items()}
    keys = ['total_loss', 'rgb_l1', 'sun_intensity', 'env_mean', 'cuda_max_allocated_mb', 'step_ms']
    extra = sorted({k for rows in runs.values() for r in rows for k in r if k.startswith(('loss_', 'geo_'))})
    keys += [k for k in extra if k != 'loss_rgb_l1']
    fig, axes = plt.subplots((len(keys) + 2) // 3, 3, figsize=(15, 3.2 * ((len(keys) + 2) // 3)))
    for ax, key in zip(axes.flat, keys):
        for name, rows in runs.items():
            pts = [(r['iteration'], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), label=name, linewidth=1)
        ax.set_title(key)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis('off')
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=110)


def load_arrays(run, key):
    with np.load(Path(run) / 'arrays' / f'{key}.npz') as d:
        return d['rgb'].transpose(1, 2, 0), d['target'].transpose(1, 2, 0)


def assert_same_targets(evals):
    frames = {k: sorted(p.stem for p in (v / 'arrays').glob('*.npz')) for k, v in evals.items()}
    base_name = next(iter(frames))
    for k, f in frames.items():
        if f != frames[base_name]:
            raise AssertionError(f'{k} 的帧集合与 {base_name} 不一致。')
    for key in frames[base_name]:
        reference = load_arrays(evals[base_name], key)[1]
        for k, v in evals.items():
            if not np.array_equal(load_arrays(v, key)[1], reference):
                raise AssertionError(f'帧 {key}：{k} 的 GT 数组与 {base_name} 不同(相机/目标不一致)。')
    return frames[base_name]


def sidebyside(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evals = parse_named(a.eval)
    keys = assert_same_targets(evals)
    a.output.mkdir(parents=True, exist_ok=False)
    arms = [n for n in ('E0', 'E1', 'E2') if n in evals]
    errors = {}
    vmax = 0.
    for key in keys:
        for n in arms:
            rgb, gt = load_arrays(evals[n], key)
            errors[key, n] = np.abs(rgb - gt).mean(-1)
            vmax = max(vmax, float(np.percentile(errors[key, n], 99.5)))
    vmax = a.error_max or max(vmax, 1e-3)
    cols = ['GT'] + arms + [f'|{n}-GT|' for n in arms if n != 'E0']
    sheet, axes_sheet = plt.subplots(len(keys), len(cols), figsize=(2.2 * len(cols), 2.2 * len(keys)))
    for i, key in enumerate(keys):
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4))
        for ax_row in (axes, axes_sheet[i]):
            for j, name in enumerate(cols):
                ax = ax_row[j]
                if name == 'GT':
                    ax.imshow(load_arrays(evals[arms[0]], key)[1].clip(0, 1))
                elif name in arms:
                    ax.imshow(load_arrays(evals[name], key)[0].clip(0, 1))
                else:
                    ax.imshow(errors[key, name[1:-4]], cmap='magma', vmin=0, vmax=vmax)
                ax.set_title(f'{name} {key}' if ax_row is axes else name, fontsize=7)
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(a.output / f'frame_{key}.png', dpi=100)
        plt.close(fig)
    sheet.tight_layout()
    sheet.savefig(a.output / 'overview.png', dpi=80)
    (a.output / 'sidebyside.json').write_text(json.dumps(dict(frames=keys, arms=arms, error_vmax=vmax,
                                                              error_definition='三通道平均绝对误差，sRGB [0,1]，统一色标'),
                                                         ensure_ascii=False, indent=2) + '\n')


def panels(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    evals = parse_named(a.eval)
    arms = [n for n in ('E0', 'E1', 'E2') if n in evals]
    fig, axes = plt.subplots(len(a.frames), len(arms), figsize=(3.4 * len(arms), 3.4 * len(a.frames)), squeeze=False)
    for i, frame in enumerate(a.frames):
        for j, n in enumerate(arms):
            axes[i, j].imshow(Image.open(evals[n] / a.kind / f'{frame:04d}.png'))
            axes[i, j].set_title(f'{n} {a.kind} {frame:04d}', fontsize=8)
            axes[i, j].axis('off')
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=100)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('curves')
    s.add_argument('--run', action='append', required=True)
    s.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('sidebyside')
    s.add_argument('--eval', action='append', required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--error-max', type=float)
    s = sub.add_parser('panels')
    s.add_argument('--eval', action='append', required=True)
    s.add_argument('--frames', type=int, nargs='+', default=[10, 80, 150])
    s.add_argument('--kind', choices=['albedo', 'normal', 'alpha', 'separation'], default='albedo')
    s.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    {'curves': curves, 'sidebyside': sidebyside, 'panels': panels}[a.action](a)


if __name__ == '__main__':
    main()
