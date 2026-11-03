"""T03 鲁棒背光校准与有限割线证书；仅操作调用者提供的数组。"""
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t02_core import basis, design

R = np.deg2rad(7.501)


def template(h):
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != 150 or not np.isfinite(h).all():
        raise ValueError('SH 亮度无效')
    m = h.min(axis=0); r = h.max(axis=0) - m
    s = np.ones_like(r)
    np.divide(.1, r, out=s, where=r > .1)
    return dict(m=m, r=r, s=s, H=(h-m)*s)


def backlight(normals, anchor, valid):
    n = np.asarray(normals, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if n.ndim != 3 or n.shape[0] != 150 or n.shape[2] != 3 or valid.shape != n.shape[:2] or anchor.shape != (150, 3):
        raise ValueError('背光输入尺寸无效')
    length = np.linalg.norm(n, axis=2)
    if not np.isfinite(n[valid]).all() or np.any(np.abs(length[valid]-1) > 1e-6):
        raise ValueError('有效观测法线无效')
    nn = np.zeros_like(n)
    np.divide(n, length[..., None], out=nn, where=(length > 0)[..., None])
    return nn, valid & (np.einsum('tgc,tc->tg', nn, anchor) <= -np.sin(R)-1e-6)


def calibrate(H, y, w, A, ids, lam):
    H, y, w = (np.asarray(v, dtype=np.float64) for v in (H, y, w))
    A = np.asarray(A, dtype=bool); ids = np.asarray(ids)
    if H.shape != y.shape or H.shape != w.shape or H.shape != A.shape or H.shape != (150, len(ids)):
        raise ValueError('校准维度无效')
    if lam not in (1e-4, 1e-3) or not np.isfinite(H).all() or not np.isfinite(w).all() or (w < 0).any() or np.any(A & (~np.isfinite(y) | (w <= 0))):
        raise ValueError('校准输入无效')
    eligible = (A.sum(axis=0) >= 12) & (A[::2].sum(axis=0) >= 6) & (A[1::2].sum(axis=0) >= 6)
    n_unique = len(np.unique(ids)); e_unique = len(np.unique(ids[eligible]))
    records = {}
    for name, parity in (('all', None), ('odd', 0), ('even', 1)):
        mask = A & eligible[None, :]
        if parity is not None:
            mask = mask.copy(); mask[1-parity::2] = False
        ww = np.where(mask, w, 0.)
        Wg = ww.sum(axis=0); W = float(Wg.sum())
        hs = np.sum(ww*H, axis=0)
        ys = np.sum(ww*np.where(mask, y, 0.), axis=0)
        hm = np.divide(hs, Wg, out=np.zeros_like(hs), where=Wg > 0)
        ym = np.divide(ys, Wg, out=np.zeros_like(ys), where=Wg > 0)
        hc = H-hm; yc = np.where(mask, y-ym, 0.)
        numerator = float(np.sum(ww*hc*yc)); energy = float(np.sum(ww*hc**2))
        raw = numerator/(energy+lam*W) if W > 0 else np.nan
        records[name] = dict(W=W, numerator=numerator, energy=energy, denominator=energy+lam*W,
                             variance=energy/W if W else np.nan, gamma_raw=raw,
                             gamma=float(np.clip(raw, 0, 1)))
    gammas = [records[k]['gamma'] for k in ('all', 'odd', 'even')]
    identified = (e_unique >= max(32, int(np.ceil(.1*n_unique))) and
                  all(np.isfinite(list(v.values())).all() and v['W'] > 0 and v['variance'] >= 1e-8 and .01 <= v['gamma'] <= .99 for v in records.values()) and
                  max(abs(gammas[0]-gammas[1]), abs(gammas[0]-gammas[2]), abs(gammas[1]-gammas[2])) <= .05)
    return dict(eligible=eligible, eligible_unique=e_unique, total_unique=n_unique,
                records=records, identified=bool(identified), failure=None if identified else 'ANCHOR_UNIDENTIFIED')


def certificate(H, w, valid, normals, anchor, phi, a, final, holdout=False):
    H, w, valid = np.asarray(H), np.asarray(w), np.asarray(valid, dtype=bool)
    if H.shape != w.shape or H.shape != valid.shape or H.shape != normals.shape[:2]:
        raise ValueError('证书维度无效')
    D = design(normals, phi, a)
    x0 = np.maximum(np.einsum('tgc,tc->tg', normals, anchor), 0)
    x1 = np.maximum(np.einsum('tgc,tc->tg', normals, final), 0)
    delta = x1-x0
    D = np.concatenate((D, delta[..., None]), axis=2)
    cal = valid.copy()
    if holdout: cal[1::2] = False
    eligible = cal.sum(axis=0) >= 24
    total = np.sum(np.where(cal, w, 0.)); covered = np.sum(np.where(cal & eligible[None, :], w, 0.))
    coverage = float(covered/total) if total > 0 else 0.
    eta = np.zeros(H.shape[1]); corr = np.zeros(H.shape[1]); singular = np.zeros((H.shape[1], 13)); rank = np.zeros(H.shape[1], dtype=int)
    for g in np.flatnonzero(eligible):
        m = cal[:, g]; wg = w[m, g]; sw = np.sqrt(wg)
        q = H[m, g]-np.average(H[m, g], weights=wg)
        d = delta[m, g]-np.average(delta[m, g], weights=wg)
        _, _, u, s, _, r = basis(D[m, g], wg)
        v = sw*q; z = sw*d; nv = np.linalg.norm(v); nz = np.linalg.norm(z)
        eta[g] = np.linalg.norm(u[:, :r].T@v)/nv if nv > 1e-12 else 0.
        corr[g] = abs(v@z)/(nv*nz) if nv > 1e-12 and nz > 1e-12 else 0.
        singular[g, :len(s)] = s; rank[g] = r
    return dict(eligible=eligible, coverage=coverage, eta=eta, secant_correlation=corr,
                singular=singular, rank=rank,
                passed=bool(coverage >= .9 and np.all(eta[eligible] <= .2) and np.all(corr[eligible] <= .2)))
