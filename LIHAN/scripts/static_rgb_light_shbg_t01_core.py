"""T01 的纯数值组件；本模块不读取数据，也不是正式实验入口。"""
import numpy as np
import torch
from utils.sh_utils import eval_sh
from LIHAN.scripts.static_rgb_light_fit_r3 import kkt


def validate_alignment(ids, expected_ids, frames):
    if not np.array_equal(ids, expected_ids) or not np.array_equal(frames, np.arange(1, 151)):
        raise ValueError('Gaussian ID 或帧号不一致')


def sh_luminance(coeff, xyz, centers, degree):
    """未调制 SH；输入顺序为点×通道×系数，视向为相机指向点。"""
    if degree not in (0, 1, 2):
        raise ValueError('T01 只允许 0/1/2 阶')
    v = np.asarray(xyz, dtype=np.float64) - np.asarray(centers, dtype=np.float64)[:, None, :]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm == 0) or not np.isfinite(norm).all():
        raise ValueError('相机和点的视向无效')
    raw = eval_sh(degree, np.asarray(coeff, dtype=np.float64)[None], v / norm) + .5
    rgb = np.clip(raw, 0, 1)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return linear @ np.array([.2126, .7152, .0722]), raw


def design(normals, directions):
    n = np.asarray(normals, dtype=np.float64)
    l = np.asarray(directions, dtype=np.float64)
    phi = np.arctan2(l[:, 1], l[:, 0])
    e = np.arcsin(np.clip(l[:, 2], -1, 1))
    dp = np.stack([-np.cos(e)*np.sin(phi), np.cos(e)*np.cos(phi), np.zeros_like(e)], axis=1)
    de = np.stack([-np.sin(e)*np.cos(phi), -np.sin(e)*np.sin(phi), np.cos(e)], axis=1)
    dot = np.einsum('tgc,tc->tg', n, l)
    active = dot > 0
    return np.stack([np.ones_like(dot), np.maximum(dot, 0),
                     active*np.einsum('tgc,tc->tg', n, dp),
                     active*np.einsum('tgc,tc->tg', n, de)], axis=-1)


def project_template(h, d, weights, valid, holdout=False):
    """留出点的投影仅用奇数帧；无效点严格返回零。"""
    h, d, weights = map(lambda x: np.asarray(x, dtype=np.float64), (h, d, weights))
    valid = np.asarray(valid, dtype=bool) & np.isfinite(h) & np.isfinite(weights) & (weights > 0)
    if h.ndim != 2 or d.shape != (*h.shape, 4) or weights.shape != h.shape or valid.shape != h.shape:
        raise ValueError('模板维度不一致')
    cal = valid.copy()
    if holdout:
        cal[1::2] = False
    q = np.zeros_like(h); beta = np.zeros((h.shape[1], 4)); errors = np.zeros(h.shape[1])
    for g in range(h.shape[1]):
        m = cal[:, g]
        if m.sum() < 8 or weights[m, g].sum() <= 0:
            continue
        sw = np.sqrt(weights[m, g]); a = d[m, g] * sw[:, None]
        beta[g] = np.linalg.pinv(a, rcond=1e-8) @ (sw * h[m, g])
        residual = h[:, g] - d[:, g] @ beta[g]
        if not np.isfinite(residual).all():
            raise ValueError('模板含非有限值')
        maximum = np.max(np.abs(residual))
        q[:, g] = residual * min(1., .1 / maximum) if maximum else residual
        errors[g] = np.max(np.abs(d[m, g].T @ (weights[m, g] * q[m, g]))) / (weights[m, g].sum() * max(1., np.max(np.abs(d[m, g]))))
        if errors[g] > 1e-8:
            raise ArithmeticError('投影正交性校验失败')
    return q, q - q.min(axis=0), beta, cal, errors


def photometric(x, y, w, H, gamma):
    """复用 E4 KKT；外层正则保持原实现，不进入内层解。"""
    rho, b = kkt(x, y - gamma * H, w)
    residual = x*rho + b + gamma*H - y
    value = (w*.5*residual.square()).sum()/w.sum() + 1e-5*(rho.square().mean()+b.square().mean())
    return value, rho, b
