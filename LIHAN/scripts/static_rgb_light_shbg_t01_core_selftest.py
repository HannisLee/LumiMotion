"""仅合成数值的 T01 数值组件自检；不访问真实数据或 GT。"""
import numpy as np
import torch
from LIHAN.scripts.static_rgb_light_shbg_t01_core import design, photometric, project_template, sh_luminance, validate_alignment
from LIHAN.scripts.static_rgb_light_fit_r3 import kkt


def main():
    rng = np.random.default_rng(1103)
    frames = np.arange(1, 151); ids = np.arange(5)
    validate_alignment(ids, ids, frames)
    try:
        validate_alignment(ids[::-1], ids, frames)
        raise AssertionError('未拒绝重排 ID')
    except ValueError:
        pass
    phi = np.linspace(0, 2*np.pi, 150); e = np.pi/4
    l = np.stack([np.cos(e)*np.cos(phi), np.cos(e)*np.sin(phi), np.full(150, np.sin(e))], axis=1)
    n = rng.normal(size=(150, 5, 3)); n /= np.linalg.norm(n, axis=2, keepdims=True)
    d = design(n, l); w = rng.uniform(.1, 1, (150, 5)); valid = np.ones_like(w, dtype=bool)
    h = np.broadcast_to(np.linspace(.2, .6, 5), w.shape).copy()
    q, _, _, _, err = project_template(h, d, w, valid)
    assert np.max(abs(q)) < 1e-8 and err.max() < 1e-8
    h += rng.normal(0, .03, h.shape)
    q, H, beta, _, err = project_template(h, d, w, valid, holdout=True)
    h2 = h.copy(); h2[1::2] += 100; w2 = w.copy(); w2[1::2] *= 50
    _, _, beta2, _, _ = project_template(h2, d, w2, valid, holdout=True)
    np.testing.assert_allclose(beta, beta2, atol=0, rtol=0)
    assert err.max() < 1e-8 and H.min() >= 0
    coeff = np.zeros((5, 3, 9)); coeff[:, :, 3] = .1
    xyz = np.broadcast_to(np.array([1., 0., 0.]), (150, 5, 3)); centers = np.zeros((150, 3))
    _, raw = sh_luminance(coeff, xyz, centers, 1)
    np.testing.assert_allclose(raw, .5 - .1*.4886025119029199)
    x = torch.tensor(rng.uniform(0, 1, (150, 5)), dtype=torch.float64)
    wt = torch.ones_like(x); ht = torch.tensor(H); yt = .7*x + .2 + .4*ht
    gamma = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    loss, _, _ = photometric(x, yt, wt, ht, gamma); loss.backward()
    assert gamma.grad is not None and np.isfinite(gamma.grad.item())
    zero = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    v, r, b = photometric(x, yt, wt, ht, zero)
    oldr, oldb = kkt(x, yt, wt); old = (wt*.5*(x*oldr+oldb-yt).square()).sum()/wt.sum()+1e-5*(oldr.square().mean()+oldb.square().mean())
    np.testing.assert_allclose(v.detach(), old.detach(), atol=1e-10, rtol=0)
    assert abs(photometric(x, yt, wt, ht, torch.tensor(.4))[0].item()) < abs(v.item())
    print('PASS: 合成数值核心检查；非完整 T01 自检；未读取真实数据或 GT')


if __name__ == '__main__':
    main()
