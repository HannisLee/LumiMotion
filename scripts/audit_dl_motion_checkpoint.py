"""审计位移头checkpoint在全部离散时间的输出边界；零训练、无GT。"""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
import torch
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scripts.probe_dynamic_dl_selfcal import read_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['model', 'cache', 'checkpoint', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    cfg = read_config(Path(args.model) / 'cfg_args')
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    ply = Path(args.model) / 'point_cloud/iteration_35000/point_cloud.ply'
    pc.load_ply(str(ply))
    base = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender,
                       hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    assert base.load_weights(args.model, 35000)
    candidate = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender,
                            hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    candidate.deform.load_state_dict(checkpoint['deform_state_dict'], strict=True)
    base.deform.eval(); candidate.deform.eval()
    assert all(torch.equal(getattr(pc, k).cpu(), v) for k, v in checkpoint['geometry'].items())
    changed = [k for k, v in base.deform.state_dict().items()
               if not torch.equal(v, candidate.deform.state_dict()[k])]
    assert changed and all(k.startswith('gaussian_warp.') for k in changed)
    tracks = torch.load(args.cache, map_location='cpu', weights_only=False)
    xyz = pc.get_xyz; feature = pc.get_binary_feature(); high = feature.flatten() > .5
    assert torch.equal(xyz[tracks['ids'].cuda()].cpu(), tracks['canonical_xyz'])
    rows = []
    for i, time in enumerate(tracks['times']):
        t = time.cuda().reshape(1, 1).expand(len(xyz), 1)
        a = base.step(xyz, t, feature=feature)
        b = candidate.step(xyz, t, feature=feature)
        for key in ['d_rotation', 'd_scaling', 'd_opacity', 'd_color', 'hidden']:
            assert torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) else a[key] == b[key]
        cached = candidate.deform.gaussian_warp(a['hidden']) * feature
        assert torch.equal(cached, b['d_xyz'])
        delta = (b['d_xyz'] - a['d_xyz']).norm(dim=-1)
        assert torch.isfinite(delta).all()
        row = dict(frame=i, time=float(time), split='held' if i % 9 == 4 else 'train',
                   maximum=float(delta.max()), mean=float(delta.mean()),
                   high_gate_max=float(delta[high].max()), low_gate_max=float(delta[~high].max()))
        rows.append(row)
        if i % 15 == 0:
            print(json.dumps(row), flush=True)
    maxima = {split: max(r['maximum'] for r in rows if r['split'] == split) for split in ['train', 'held']}
    paths = [Path(__file__), Path(args.checkpoint), Path(args.cache), ply,
             Path(args.model) / 'cfg_args', Path(args.model) / 'deform/iteration_35000/deform.pth']
    result = dict(status='PASS' if max(maxima.values()) <= .00101 else 'FAILED',
                  scope='仅135个离散时间的输出边界和重载一致性；非连续时间保证或几何改善',
                  gt_read=False, changed_keys=changed, gaussian_geometry_unchanged=True,
                  other_heads_unchanged=True, cached_and_native_equal=True,
                  maxima=maxima, points=len(xyz), rows=rows,
                  inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'inputs']}, ensure_ascii=False))
    assert result['status'] == 'PASS'


if __name__ == '__main__':
    main()
