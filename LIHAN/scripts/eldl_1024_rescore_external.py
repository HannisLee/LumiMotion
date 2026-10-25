#!/usr/bin/env python3
"""用V3固定协议(eldl_1024_eval.metric_arrays/aggregate)重新计分历史EL已保存的预测数组。

只读：EL预测数组(法线/深度/Mesh)与V3 gt_v1；不加载模型、不训练。
同时从EL数组按其原GT(gt_normal/gt_foreground)独立复算3.50676425，核对基准来源。
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LIHAN.scripts import eldl_1024_eval as E


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--gt', type=Path, default=E.GT)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    out = E.output_path(a.output, create=True)
    gtm = E.read(a.gt / 'gt_depth/manifest.json')
    rows, hashes, own = [], {}, dict(sum=0., n=0, miss=0, gt=0)
    for n in E.TEST:
        pp = a.pred_dir / f'prediction_{n:04d}.npz'
        gp = a.gt / 'gt_depth' / gtm['frames'][str(n)]['file']
        assert E.sha(gp) == gtm['frames'][str(n)]['sha256']
        hashes[str(pp)] = E.sha(pp); hashes[str(gp)] = gtm['frames'][str(n)]['sha256']
        pred = dict(np.load(pp, allow_pickle=False)); gt = dict(np.load(gp, allow_pickle=False))
        # EL原GT口径复算：unit_normal后有向角、预测alpha>=0.5
        from utils.geometry_eval_utils import normal_errors, unit_normal
        tgt, tv = unit_normal(pred['gt_normal'])
        err, val = normal_errors(pred['normal'], tgt)
        base = pred['gt_foreground'].astype(bool) & tv
        ok = base & val & (pred['alpha'] >= .5)
        own['sum'] += float(err['angular_deg'][ok].sum()); own['n'] += int(ok.sum()); own['gt'] += int(base.sum())
        for k in ('camera_to_world', 'intrinsics'):
            pred[k] = gt[k]
        pred['source_alpha'] = gt['alpha']
        pred['rgb'] = np.zeros(pred['alpha'].shape + (3,), np.float32)   # RGB不由本脚本评测
        pred['target_rgb'] = np.zeros_like(pred['rgb'])
        regions, other, _, _ = E.metric_arrays(pred, gt)
        rows.append(dict(source_frame=n, split='test', regions=regions, alpha_iou=other['alpha_iou']))
        print('rescore', n, flush=True)
    summary = E.aggregate(rows)
    el_own = dict(gs_conditional_mae_deg=own['sum'] / own['n'], valid_pixels=own['n'], gt_pixels=own['gt'],
                  coverage=own['n'] / own['gt'],
                  missing180_mae_deg=(own['sum'] + 180 * (own['gt'] - own['n'])) / own['gt'])
    E.write(out / 'metrics.json', dict(status='PASS_RESCORE', pred_dir=str(a.pred_dir), gt=str(a.gt),
        source_hashes=hashes, el_own_gt_recompute=el_own, rows=rows, summary=summary,
        note='RGB指标未评测(无预测RGB数组)；Mesh/Depth法线沿用EL已保存数组。'))
    fg = summary['foreground']['normal']
    print(json.dumps(dict(el_own_gt_recompute=el_own, v3_gt={k: {m: v[m] for m in ('mae_pixel_mean_deg', 'mae_frame_mean_deg', 'coverage', 'missing180_pixel_mean_deg', 'missing180_frame_mean_deg', 'eligible_pixels')} for k, v in fg.items()}), indent=1))


if __name__ == '__main__':
    main()
