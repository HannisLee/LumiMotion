"""重建T05原始投影与膨胀边界，分层审计固定T58/T59预测；不拟合参数。"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from fit_dl_shadow_tracks import project_shadow, sample_shadow


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--fit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); torch.set_num_threads(4)
    out = Path(a.output); out.mkdir(exist_ok=False)
    root = Path(a.audit); meta = json.loads((root/'result.json').read_text())
    assert not meta['gt_read']
    for path, sha in meta['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    sources = meta['source_hashes']
    cache = torch.load(next(x for x in sources if x.endswith('tracks.pt')), weights_only=False)
    state = torch.load(next(x for x in sources if 'T05' in x and x.endswith('checkpoint.pt')), weights_only=False)
    prior = json.loads(Path(next(x for x in sources if 'T05' in x and x.endswith('result.json'))).read_text())
    d = torch.load(root/'predictions.pt', weights_only=False)
    fm = json.loads((Path(a.fit)/'result.json').read_text())
    assert not fm['gt_read'] and fm['source_audit_sha256'] == hashlib.sha256((root/'predictions.pt').read_bytes()).hexdigest()
    fitted = torch.load(Path(a.fit)/'checkpoint.pt', weights_only=False)
    xyz, pos = cache['canonical_xyz'], cache['positions']
    train = state['train']; origin, normal, basis = [state[k] for k in ('plane_origin','plane_normal','basis')]
    motion = (pos[train]-pos[train].mean(0)[None]).square().sum(-1).mean(0).sqrt()
    receiver = (cache['valid'][train].sum(0)>=10) & (((xyz-origin)@normal).abs()<prior['scene_extent']*.01) & (motion<prior['scene_extent']*.01)
    uv = (xyz[receiver]-origin)@basis.T
    lower = uv.amin(0); span = (uv.amax(0)-lower).clamp_min(.01)
    lower = lower-.05*span; span = span*1.1
    raw, expanded, inside_all = [], [], []
    for i, direction in enumerate(state['smooth']):
        points = pos[i,state['occluder']]
        height = (points-origin)@normal
        projected = (points-origin)-height[:,None]/(direction@normal).clamp_min(.001)*direction
        index = (((projected@basis.T)-lower)/span*128).floor().long()
        inside = ((index>=0)&(index<128)).all(-1)
        flat = index[:,1].clamp(0,127)*128+index[:,0].clamp(0,127)
        grid = torch.zeros(1,128*128); grid.scatter_add_(1,flat[None],inside.float()[None]); grid.clamp_max_(1)
        dilated = F.max_pool2d(grid.view(1,1,128,128),3,stride=1,padding=1).flatten(1)
        reference = project_shadow(points,direction[None],origin,normal,basis,lower,span)
        assert torch.equal(dilated,reference), i
        r, ins = sample_shadow(grid,pos[i],origin,basis,lower,span)
        e, _ = sample_shadow(dilated,pos[i],origin,basis,lower,span)
        raw.append(r[0].bool()); expanded.append(e[0].bool()); inside_all.append(ins)
    raw, expanded, inside = map(torch.stack,(raw,expanded,inside_all))
    assert torch.equal(expanded,state['predicted']['smooth'].bool())
    assert torch.equal(expanded,d['blocked']) and not (raw & ~expanded).any()
    strata = {'raw_hit':raw, 'dilation_only':expanded & ~raw, 'outside_shadow':~expanded}
    rows = {}
    # 统一使用广支持拟合的三组材质，避免把不同材质训练支持混为同一对照。
    g = d['groups']['all_unsaturated']; predictions = dict(g['predictions'])
    predictions['calibrated'] = fitted['all_unsaturated']['prediction']
    for split, times in [('train',d['train']),('held',d['held'])]:
        for name, region in strata.items():
            for label, extra in [('all',torch.ones_like(raw)),('bright',state['negative']),('dark',state['positive']),('strict',d['groups']['strict_bright']['mask'])]:
                mask = (g['mask'] & region & extra)[times]; n = int(mask.sum())
                record = {'observations':n,'points':int(mask.any(0).sum())}
                for mode, pred in predictions.items():
                    err = (pred-d['rgb'])[times][mask]
                    record[mode] = {'mse':float(err.square().mean()) if n else None,'signed_mean':float(err.mean()) if n else None}
                rows[f'{split}/{name}/{label}'] = record
    result = dict(status='PASS',gt_read=False,parameter_updates=0,exact_proxy_mismatches=0,
                  outside_grid_supported=int((g['mask'] & ~inside).sum()),statistics=rows,
                  scope='观察亮暗标签不是遮挡真值；原始点投影不是连续表面遮挡。固定T59模型，仅诊断，不代表动态几何改善。',
                  source_hashes={str(Path(x).resolve()):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in [__file__,str(root/'predictions.pt'),str(Path(a.fit)/'checkpoint.pt'),'scripts/fit_dl_shadow_tracks.py']})
    torch.save(dict(raw=raw,expanded=expanded,inside=inside,lower=lower,span=span),out/'projection.pt')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
