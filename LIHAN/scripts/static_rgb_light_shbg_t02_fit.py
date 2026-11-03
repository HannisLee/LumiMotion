"""T02 候选拟合；不进行留出评分或候选选择。"""
import json
from pathlib import Path
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t02_templates import load_support, template, save_template
from LIHAN.scripts.static_rgb_light_shbg_t02_optimizer import fit
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters

CANDIDATES = (('C01', 1, 1e-4), ('C02', 1, 1e-3), ('C03', 2, 1e-4), ('C04', 2, 1e-3))
RUNS = ('full', 'A', 'B', 'boot103108', 'boot103109', 'hi', 'lo')


def fit_candidates(observations, baseline, sh_values, extraction_metadata, output, smoke=False):
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    indices, _, anchors = load_support(baseline, observations)
    for anchor in anchors.values(): anchor_parameters(anchor)
    with np.load(observations, allow_pickle=False) as d:
        n, y, w = (d[k].astype(np.float64) for k in ('normals', 'intensity', 'weights'))
        ids, frames = d['gaussian_ids'].copy(), d['frames'].copy()
    if y.shape != (150, len(ids)) or y.shape != w.shape or not np.isfinite(w).all() or (w < 0).any() or ((w > 0) & ~np.isfinite(y)).any() or not np.isfinite(n).all(): raise ValueError('基础观测无效')
    if n.shape not in ((150, len(ids), 3), (len(ids), 3)) or not np.allclose(np.linalg.norm(n, axis=-1), 1, atol=1e-4): raise ValueError('法线无效')
    with np.load(sh_values, allow_pickle=False) as s:
        if not np.array_equal(ids, s['gaussian_ids']) or not np.array_equal(frames, s['frames']): raise ValueError('SH 与观测错位')
        h = {L:s[f'h_L{L}'].copy() for L in (0, 1, 2)}
    diag = json.loads(Path(extraction_metadata).read_text())['diagnostics']
    l0 = template(h[0], n, y, w, indices['full'], anchors['full'])
    if np.max(np.abs(l0['q'])) > 1e-8: raise ArithmeticError('L0 退化自检失败')
    keys = ('full',) if smoke else RUNS; invalid = {}; cache = {}
    for L in ((1,) if smoke else (1, 2)):
        if diag[str(L)]['invalid_clipping']: invalid[L] = 'SH_CLIPPING'; continue
        for key in keys:
            rec = template(h[L], n, y, w, indices[key], anchors[key]); cache[L, key] = rec
            save_template(out/'templates'/f'L{L}_{key}.npz', h[L], indices[key], ids, frames, rec, L)
            if rec['nondegenerate_fraction'] < .2: invalid[L] = 'SH_DEGENERATE'
    specs = CANDIDATES[:1] if smoke else (('C00L', 0, 0.),) + CANDIDATES
    results = {}
    for name, L, lam in specs:
        results[name] = {}
        if L in invalid:
            results[name]['failure'] = invalid[L]; continue
        failures = []
        for key in keys:
            idx = indices[key]; nn = n[:,idx] if n.ndim == 3 else np.broadcast_to(n[idx], (150, len(idx), 3))
            H = np.zeros((150,len(idx))) if L == 0 else cache[L,key]['H']
            run_dir = out/'runs'/name/key
            cert = fit(nn, y[:,idx], w[:,idx], H, anchors[key], lam, run_dir, key, L == 0)
            results[name][key] = cert
            if not cert['converged']: failures.append(key)
            (out/'progress.json').write_text(json.dumps(results, indent=2))
        results[name]['failure'] = failures or None
    (out/'fit_results.json').write_text(json.dumps(results, indent=2))
    return results
