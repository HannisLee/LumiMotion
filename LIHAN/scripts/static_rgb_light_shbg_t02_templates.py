"""T02 逐流程模板；支持点与锚点沿用冻结 E4 白名单读取器。"""
from pathlib import Path
import json
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t01_templates import load_support
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters, design, project


def template(h, n, y, w, idx, anchor, holdout=False):
    nn = n[:, idx] if n.ndim == 3 else np.broadcast_to(n[idx], (150, len(idx), 3))
    yy, ww, hh = y[:, idx], w[:, idx], h[:, idx]
    valid = np.isfinite(hh) & np.isfinite(yy) & np.isfinite(ww) & (ww > 0)
    phi, a = anchor_parameters(anchor)
    result = project(hh, design(nn, phi, a), ww, valid, holdout)
    weighted = np.where(valid, ww, 0.); mass = weighted.sum(0)
    rms = np.sqrt((weighted*result['q']**2).sum(0)/np.maximum(mass, 1e-30))
    eligible = mass > 0
    result.update(valid=valid, nondegenerate_fraction=float((rms[eligible] >= 1e-4).mean()) if eligible.any() else 0., eligible=eligible)
    return result


def save_template(path, h, idx, ids, frames, record, degree):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, L=degree, gaussian_ids=ids[idx], frames=frames, indices=idx, h=h[:, idx], **{k:v for k,v in record.items() if isinstance(v, np.ndarray)})
    path.with_suffix('.json').write_text(json.dumps({'nondegenerate_fraction':record['nondegenerate_fraction'], 'eligible_points':int(record['eligible'].sum())}, indent=2))
