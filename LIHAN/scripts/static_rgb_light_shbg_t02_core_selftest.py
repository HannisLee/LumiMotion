"""T02 核心局部合成检查；并非规划规定的完整自检。"""
import numpy as np
import torch
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters, design, fourier, project, trajectory, final_eta


def main():
    rng = np.random.default_rng(1103)
    t = np.arange(150)*2*np.pi/150
    anchor = np.stack((np.cos(t)/np.sqrt(2), np.sin(t)/np.sqrt(2), np.full(150, 1/np.sqrt(2))), axis=1)
    phi, a = anchor_parameters(anchor)
    n = rng.normal(size=(150, 1, 3)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    d = design(n, phi, a)
    x = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    def shade(z):
        _, _, _, l = trajectory(torch.tensor(phi), torch.tensor(a), z[:5], z[5:])
        return torch.relu((torch.tensor(n[:, 0])*l).sum(1))
    jac = torch.autograd.functional.jacobian(shade, x).detach().numpy()
    for j in range(10):
        step = np.zeros(10); step[j] = 1e-5
        fd = (shade(torch.tensor(step)).detach().numpy()-shade(torch.tensor(-step)).detach().numpy())/(2e-5)
        assert np.max(np.abs(fd-jac[:, j])) < 1e-7
    assert np.max(np.abs(jac[:, :5] - d[:, 0, 2:7]*np.pi/180)) < 1e-12
    assert np.max(np.abs(jac[:, 5:] - d[:, 0, 7:]*.1)) < 1e-12
    w = np.ones((150, 1)); valid = np.ones((150, 1), bool)
    h = d[:, :, :] @ rng.normal(size=(1, 12, 1)); h = h[..., 0]
    r = project(h, d, w, valid)
    assert np.max(np.abs(r['q'])) <= 1e-8
    h += rng.normal(size=(150, 1))
    r = project(h, d, w, valid)
    assert r['relative'][0] <= 1e-6 and final_eta(r['q'], d, w, r['cal'])[0] <= 1e-6
    v = valid.copy(); v[23:] = False
    assert not np.any(project(h, d, w, v)['q'])
    v[23] = True
    assert np.any(project(h, d, w, v)['q'])
    r1 = project(h, d, w, valid, holdout=True)
    h2 = h.copy(); h2[1::2] += 4
    r2 = project(h2, d, w, valid, holdout=True)
    assert np.allclose(r1['beta'], r2['beta'], atol=0, rtol=0)
    print('局部合成数值检查通过；不构成完整自检 receipt。')


if __name__ == '__main__':
    main()
