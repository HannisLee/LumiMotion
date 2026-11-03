"""T02 低频轨迹和受保护模板的数值核心；仅操作调用者提供的数组。"""
import numpy as np
import torch


def fourier():
    theta = 2 * np.pi * np.arange(150, dtype=np.float64) / 150
    return np.stack((np.ones(150), np.cos(theta), np.sin(theta),
                     np.cos(2 * theta), np.sin(2 * theta)), axis=1)


def anchor_parameters(anchor):
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (150, 3) or not np.isfinite(anchor).all() or not np.allclose(np.linalg.norm(anchor, axis=1), 1, atol=1e-10):
        raise ValueError('锚点方向无效')
    phi = np.unwrap(np.arctan2(anchor[:, 1], anchor[:, 0]))
    a = np.arctanh(np.clip((np.arcsin(np.clip(anchor[:, 2], -1, 1)) - np.pi/4)/(np.pi/36), -.999999, .999999))
    e = np.pi/4 + np.pi/36*np.tanh(a)
    reconstructed = np.stack((np.cos(e)*np.cos(phi), np.cos(e)*np.sin(phi), np.sin(e)), axis=1)
    error = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(anchor, reconstructed), axis=1), np.sum(anchor*reconstructed, axis=1)))
    if error.max() > 1e-4:
        raise ValueError('锚点反算误差超过门槛')
    return phi, a


def trajectory(phi0, a0, U, V):
    f = torch.as_tensor(fourier(), dtype=U.dtype, device=U.device)
    phi = phi0 + f @ (np.pi/180 * torch.tanh(U))
    a = a0 + f @ (.1 * torch.tanh(V))
    e = np.pi/4 + np.pi/36 * torch.tanh(a)
    l = torch.stack((torch.cos(e)*torch.cos(phi), torch.cos(e)*torch.sin(phi), torch.sin(e)), dim=1)
    return phi, a, e, l


def design(normals, phi, a):
    n = np.asarray(normals, dtype=np.float64)
    phi, a = np.asarray(phi), np.asarray(a)
    if n.ndim != 3 or n.shape[0] != 150 or n.shape[2] != 3 or phi.shape != (150,) or a.shape != (150,):
        raise ValueError('切空间输入维度错误')
    e = np.pi/4 + np.pi/36*np.tanh(a)
    l = np.stack((np.cos(e)*np.cos(phi), np.cos(e)*np.sin(phi), np.sin(e)), axis=1)
    dp = np.stack((-np.cos(e)*np.sin(phi), np.cos(e)*np.cos(phi), np.zeros_like(e)), axis=1)
    de = np.stack((-np.sin(e)*np.cos(phi), -np.sin(e)*np.sin(phi), np.cos(e)), axis=1)
    dot = np.einsum('tgc,tc->tg', n, l)
    active = dot > 0
    p = active*np.einsum('tgc,tc->tg', n, dp)
    z = active*np.einsum('tgc,tc->tg', n, de)*(np.pi/36*(1-np.tanh(a)**2))[:, None]
    f = fourier()
    return np.concatenate((np.ones((*dot.shape, 1)), np.maximum(dot, 0)[..., None], p[..., None]*f[:, None, :], z[..., None]*f[:, None, :]), axis=2)


def basis(d, w):
    A = np.sqrt(w)[:, None]*d
    scales = np.linalg.norm(A, axis=0)
    inv = np.divide(1., scales, out=np.zeros_like(scales), where=scales > 1e-12)
    u, s, vt = np.linalg.svd(A*inv, full_matrices=False)
    rank = int(np.sum(s > (s[0]*1e-8 if len(s) else 0)))
    return scales, inv, u, s, vt, rank


def project(h, d, weights, valid, holdout=False):
    h, d, weights = (np.asarray(x, dtype=np.float64) for x in (h, d, weights))
    valid = np.asarray(valid, dtype=bool) & np.isfinite(h) & np.isfinite(weights) & (weights > 0)
    if h.ndim != 2 or h.shape[0] != 150 or d.shape != (*h.shape, 12) or weights.shape != h.shape or valid.shape != h.shape or not np.isfinite(d).all():
        raise ValueError('模板维度或设计矩阵错误')
    cal = valid.copy()
    if holdout:
        cal[1::2] = False
    G = h.shape[1]
    q = np.zeros_like(h); beta = np.zeros((G, 12)); scales = np.zeros((G, 12)); singular = np.zeros((G, 12)); rank = np.zeros(G, dtype=np.int64)
    error = np.zeros(G); relative = np.zeros(G)
    for g in range(G):
        m = cal[:, g]
        if m.sum() < 24:
            continue
        sc, inv, u, s, vt, r = basis(d[m, g], weights[m, g])
        scales[g], singular[g, :len(s)], rank[g] = sc, s, r
        if r:
            beta[g] = inv*(vt[:r].T @ ((u[:, :r].T @ (np.sqrt(weights[m, g])*h[m, g]))/s[:r]))
        raw = h[:, g] - d[:, g] @ beta[g]
        if not np.isfinite(raw).all():
            raise ValueError('模板含非有限值')
        maximum = np.max(np.abs(raw))
        q[:, g] = raw*min(1., .1/maximum) if maximum else raw
        if np.sqrt(np.sum(weights[m, g]*q[m, g]**2)/np.sum(weights[m, g])) < 1e-8:
            q[:, g] = 0
        v = np.sqrt(weights[m, g])*q[m, g]
        error[g] = np.max(np.abs(d[m, g].T @ (weights[m, g]*q[m, g])))/(np.sum(weights[m, g])*max(1., np.max(np.abs(d[m, g]))))
        relative[g] = np.linalg.norm(u[:, :r].T @ v)/max(np.linalg.norm(v), 1e-12)
        if error[g] > 1e-8 or relative[g] > 1e-6:
            raise ArithmeticError('投影完整性错误')
    return dict(q=q, H=q-q.min(axis=0), beta=beta, cal=cal, scales=scales, singular=singular, rank=rank, counts=cal.sum(axis=0), error=error, relative=relative)


def final_eta(q, d, weights, cal):
    eta = np.zeros(q.shape[1])
    for g in range(q.shape[1]):
        m = cal[:, g]
        if m.sum() < 24:
            continue
        _, _, u, _, _, r = basis(d[m, g], weights[m, g])
        v = np.sqrt(weights[m, g])*q[m, g]
        den = np.linalg.norm(v)
        eta[g] = np.linalg.norm(u[:, :r].T @ v)/den if den > 1e-12 else 0.
    return eta
