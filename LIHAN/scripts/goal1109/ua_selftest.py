"""1109-01 T01 合成自检 S1–S15（仅合成数据；合成真值不是数据集 GT；不读取 GT、不打开 observations.npz）。"""
import inspect
import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r6_env as r6
from LIHAN.scripts.static_rgb_light_fit_r3 import unit, write
from LIHAN.scripts.static_rgb_light_fit_r5b import elev, compose, penalties, orbit_sign
from LIHAN.scripts.goal1109 import ua_core as C
from LIHAN.scripts.goal1109 import ua_diag as D
from LIHAN.scripts.goal1109 import ua_figs as FG


# ------------------------------------------------------------------ 合成场景
def synth_scene(rng, T=150, G=600, a_true=.3, sigma=.003, rot_fraction=1 / 3, rho_range=(.3, .8)):
    """150 帧变速环绕、带宽内平滑游走仰角；1/3 高斯法线随时间旋转；rho~U；y=rho(a+x)+noise。"""
    speed = 2.4 * (1 + .3 * np.sin(np.linspace(0, 3, T)))
    phi = np.deg2rad(np.cumsum(speed))
    walk = np.cumsum(rng.normal(0, .15, T))
    walk = np.convolve(walk, np.ones(9) / 9, 'same')
    walk = walk - walk.mean()
    walk = 3.9 * walk / abs(walk).max()
    et = np.deg2rad(45 + walk)
    truth = np.column_stack([np.cos(et) * np.cos(phi), np.cos(et) * np.sin(phi), np.sin(et)])
    n0 = unit(rng.normal(size=(G, 3)))
    n = np.broadcast_to(n0[None], (T, G, 3)).copy()
    nrot = int(G * rot_fraction)
    ang = np.deg2rad(0.4) * np.arange(T)                       # 绕 z 轴缓慢旋转
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    n[:, :nrot, 0] = c * n0[None, :nrot, 0] - s * n0[None, :nrot, 1]
    n[:, :nrot, 1] = s * n0[None, :nrot, 0] + c * n0[None, :nrot, 1]
    x = np.maximum(np.einsum('tgc,tc->tg', n, truth), 0)
    rho = rng.uniform(*rho_range, (G, 3))
    y = rho[None] * (a_true + x[..., None]) + rng.normal(0, sigma, (T, G, 3)) if sigma > 0 else rho[None] * (a_true + x[..., None])
    return dict(n=n, y=y, w=np.ones((T, G)), truth=truth, rho=rho, a_true=a_true, phi=phi, e=et)


def perturbed_init(truth, rng, s=.03):
    return np.column_stack([unit(truth[:, :2] + rng.normal(0, s, (len(truth), 2))) * np.cos(C.E_C), np.full(len(truth), np.sin(C.E_C))])


def synth_bundle(rng, G=500):
    """与真实数据包同构的合成数据包（供 run_arm 端到端/注入测试）。"""
    sc = synth_scene(rng, G=G)
    T = 150
    y = np.clip(sc['y'], 1e-3, 1.0)
    rgb = D.oetf(y)
    y = r6.srgb_to_linear(rgb)
    w = rng.uniform(.5, 1.0, (T, G))
    bad = rng.random((T, G)) < .08
    rgb[bad] = np.nan
    y[bad] = np.nan
    w[bad] = 0
    inten = np.where(bad, np.nan, np.nan_to_num(y) @ C.KAPPA)
    fi = np.arange(0, int(.6 * G))
    vi = np.arange(int(.6 * G), int(.8 * G))
    A, Bs = fi[::2], fi[1::2]
    picks = {s: np.random.default_rng(s).choice(fi, len(fi), replace=True) for s in C.BOOT}
    supports = {'full': fi, 'A': A, 'B': Bs, **{f'boot{s}': picks[s] for s in C.BOOT}}
    initial = {k: perturbed_init(sc['truth'], rng, .03 + .01 * i) for i, k in enumerate(supports)}
    ref = perturbed_init(sc['truth'], rng, .05)
    c01 = perturbed_init(sc['truth'], rng, .04)
    hist = {k: perturbed_init(sc['truth'], rng, .08) for k in ['E1', 'E2', 'E3']}
    obs = dict(frames=np.arange(1, T + 1), n=sc['n'], y=y, rgb=rgb, intensity=inten, w=w)
    support = dict(body_fit=fi, body_val=vi, split_A=A, split_B=Bs)
    return dict(obs=obs, support=support, supports=supports, picks=picks, initial=initial, reference=ref, historical=hist, c01=c01, truth=sc['truth'], rho=sc['rho'])


def tensors(sc, device='cpu'):
    return C.make_ctx(sc['n'], sc['y'], sc['w'], device)


# ------------------------------------------------------------------ S1
def s1():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(1)
    T, G = 60, 40
    A = rng.uniform(.1, 1.2, (T, G))
    rho_true = rng.uniform(.2, .8, (G, 3))
    y = rho_true[None] * A[..., None] + rng.normal(0, .01, (T, G, 3))
    w = rng.random((T, G)) + .1
    y[:, 0] = -A[:, :1] * np.array([[.5, .5, .5]])               # rho -> 0
    y[:, 1] = 2.0 * A[:, 1:2] * np.ones((1, 3))                   # rho -> 1 （上界活跃）
    y[:, 2] = .5 * A[:, 2:3] * np.ones((1, 3))                    # 内点
    w[:, 3] = 0                                                   # 零权重
    r = C.rho_box(torch.as_tensor(A), torch.as_tensor(y), torch.as_tensor(w)).numpy()
    worst = 0.0
    for g in range(G):
        for c in range(3):
            sw = np.sqrt(w[:, g])
            if (sw == 0).all():
                ref = 0.0
            else:
                ref = lsq_linear((A[:, g] * sw)[:, None], y[:, g, c] * sw, bounds=([0.0], [1.0]), tol=1e-14).x[0]
            worst = max(worst, abs(r[g, c] - ref))
    assert r[0].max() == 0 and np.allclose(r[1], 1.0) and 0 < r[2].min() and r[3].max() == 0, r[:4]
    assert worst < 2e-5, worst
    return dict(max_abs_diff=worst, covered=['rho=0', 'rho=1', 'interior', 'zero weight'])


# ------------------------------------------------------------------ S2
def s2():
    rng = np.random.default_rng(2)
    T, G = 50, 70
    x = torch.as_tensor(np.maximum(rng.normal(0, .5, (T, G)), 0))
    y = torch.as_tensor(rng.uniform(0, 1, (T, G, 3)))
    w = torch.as_tensor(rng.random((T, G)))
    S = C.suff_stats(x, y, w)
    amb = rng.uniform(0, 3, 50)
    F = C.profile_F(torch.as_tensor(amb), S).numpy()
    worst = 0.0
    for a_, f in zip(amb, F):
        A = a_ + x
        rho = C.rho_box(A, y, w)
        d = float(C.data_loss(A, rho, y, w))
        worst = max(worst, abs(f - d) / abs(d))
    assert worst < 1e-12, worst
    # 近似拟合点（噪声 0.003）的相对差，作为参考信息
    sc = synth_scene(np.random.default_rng(22), T=40, G=80)
    ctx = tensors(sc)
    xx = torch.relu(torch.einsum('tgc,tc->tg', ctx['nt'], torch.as_tensor(sc['truth'][:40])))
    S2_ = C.suff_stats(xx, ctx['yt'], ctx['ww'])
    f2 = float(C.profile_F(.3, S2_)[0])
    A2 = .3 + xx
    d2 = float(C.data_loss(A2, C.rho_box(A2, ctx['yt'], ctx['ww']), ctx['yt'], ctx['ww']))
    return dict(max_rel_diff_random=worst, rel_diff_near_fit_reference=abs(f2 - d2) / d2)


# ------------------------------------------------------------------ S3
def s3():
    res = {}
    for name, a_true, hi in [('interior', .3, 4.0), ('boundary_zero', 0.0, 4.0), ('beyond_initial_gridmax', 6.0, 8.0)]:
        sc = synth_scene(np.random.default_rng(3), T=60, G=120, a_true=a_true, rho_range=(.08, .13) if a_true > 1 else (.3, .8))
        ctx = tensors(sc)
        l = torch.as_tensor(sc['truth'][:60])
        x = torch.relu(torch.einsum('tgc,tc->tg', ctx['nt'], l))
        S = C.suff_stats(x, ctx['yt'], ctx['ww'])
        r = C.search_amb(S)
        dense = torch.linspace(0, hi, 100001, dtype=torch.float64)
        Fd = C.profile_F(dense, S)
        i = int(torch.argmin(Fd))
        fs = float(C.profile_F(r['amb'], S)[0])
        assert fs <= float(Fd[i]) + 1e-15, (name, fs, float(Fd[i]))
        assert abs(r['amb'] - float(dense[i])) <= 1e-4, (name, r['amb'], float(dense[i]))
        res[name] = dict(amb_search=r['amb'], amb_dense=float(dense[i]), F_search=fs, F_dense=float(Fd[i]), expansions=r['expansions'], flags=r['flags'])
    assert res['beyond_initial_gridmax']['expansions'] >= 1
    return res


# ------------------------------------------------------------------ S4
def s4():
    sc = synth_scene(np.random.default_rng(4), T=40, G=80, sigma=.01)
    ctx = tensors(sc)
    rng = np.random.default_rng(44)
    phi0 = np.unwrap(np.arctan2(sc['truth'][:40, 1], sc['truth'][:40, 0])) + rng.normal(0, .02, 40)
    nu0 = rng.normal(0, .3, 40)
    phi = torch.tensor(phi0, requires_grad=True)
    nu = torch.tensor(nu0, requires_grad=True)
    total = C.objective_ua(phi, nu, ctx, 'A0', 1.0)[0]
    gphi, gnu = torch.autograd.grad(total, (phi, nu))

    def f(p, v):
        with torch.no_grad():
            return float(C.objective_ua(p, v, ctx, 'A0', 1.0)[0])
    worst = 0.0
    h = 1e-6
    checked = []
    for k in [0, 7, 13, 21, 33, 39]:
        for which, g_ in (('phi', gphi), ('nu', gnu)):
            pp, pm = phi.detach().clone(), phi.detach().clone()
            vp, vm = nu.detach().clone(), nu.detach().clone()
            if which == 'phi':
                pp[k] += h
                pm[k] -= h
            else:
                vp[k] += h
                vm[k] -= h
            fd = (f(pp, vp) - f(pm, vm)) / (2 * h)
            rel = abs(fd - float(g_[k])) / max(abs(fd), 1e-12)
            checked.append((which, k, float(g_[k]), fd, rel))
            worst = max(worst, rel)
    assert worst < 1e-5, checked
    return dict(max_rel_err=worst, checked=checked)


# ------------------------------------------------------------------ S5
def s5(workdir):
    rng = np.random.default_rng(103108)
    sc = synth_scene(rng, T=150, G=600, a_true=.3, sigma=.003)
    init = perturbed_init(sc['truth'], rng, .03)
    y = sc['y']
    out = workdir / 's5'
    out.mkdir()
    end, info = C.fit_traj_ua(sc['n'], y, sc['w'], init, 0., out, 'synthetic', arm='A0', quiet=True)
    err = float(C.chord_deg(end, sc['truth']).mean())
    assert info['converged'], info['certificate']
    assert err < 1.0, err
    assert abs(info['amb'] - .3) < .01, info['amb']
    end2, info2 = C.fit_traj_ua(sc['n'], y, sc['w'], init, 0., out, 'synthetic_repeat', arm='A0', quiet=True)
    assert np.allclose(end, end2, atol=1e-8), float(abs(end - end2).max())
    # A1 对照：同数据下 amb 恒 0（数据来自 a=0.3，A1 应有更大数据残差，仅记录）
    return dict(mean_err_deg=err, amb=info['amb'], amb_true=.3, converged=info['converged'], certificate_path=info['certificate']['path'], final_loss=info['final_loss'],
                repeat_max_abs_diff=float(abs(end - end2).max()), elev_range_deg=[float(np.rad2deg(sc['e'].min())), float(np.rad2deg(sc['e'].max()))])


# ------------------------------------------------------------------ S6
def s6():
    rng = np.random.default_rng(6)
    T, G = 60, 120
    phi = np.deg2rad(np.arange(T) * 6.0)
    e0 = np.deg2rad(45.)
    l = np.column_stack([np.cos(e0) * np.cos(phi), np.cos(e0) * np.sin(phi), np.full(T, np.sin(e0))])
    nz = rng.uniform(.8, .99, G)
    az = rng.uniform(0, 2 * np.pi, G)
    n0 = np.column_stack([np.sqrt(1 - nz ** 2) * np.cos(az), np.sqrt(1 - nz ** 2) * np.sin(az), nz])
    n = np.broadcast_to(n0[None], (T, G, 3)).copy()
    x = n0 @ l.T
    assert (x > 0).all()
    rho = rng.uniform(.3, .6, (G, 3))
    amb_true = .3
    y = rho[None] * (amb_true + x.T[..., None])
    ctx = C.make_ctx(n, y, np.ones((T, G)), 'cpu')

    def ua_loss(lv):
        xx = torch.relu(torch.einsum('tgc,tc->tg', ctx['nt'], torch.as_tensor(lv)))
        return C.search_amb(C.suff_stats(xx, ctx['yt'], ctx['ww']))['F']
    L0 = ua_loss(l)
    out = {}
    for c in (-.05, .05):
        lp = l + np.array([0, 0, c])
        lp = lp / np.linalg.norm(lp, axis=1, keepdims=True)
        k = np.linalg.norm(l + np.array([0, 0, c]), axis=1)
        dL = ua_loss(lp) - L0
        assert dL > 1e-8, (c, dL)
        # 对照组：加入 beta_z n_z 自由项（E = amb + beta_z*nz）。解析点 amb'=amb/k, beta=-c/k, rho'=k*rho 精确复现，重解 rho 后损失 <= 该点损失
        xp = torch.as_tensor(n0 @ lp.T).T
        kk = float(k.mean())
        # k 随帧变化（水平分量不变、仅 z 平移）：k_t 不是常数 -> 对照点按帧精确复现需要 rho 常数；改用最优化 (amb, beta)
        from scipy.optimize import minimize
        yt, wt = ctx['yt'], ctx['ww']

        def ctrl(p):
            A = p[0] + p[1] * torch.as_tensor(nz)[None, :] + xp
            r_ = C.rho_box(A, yt, wt, upper=None)
            return float(C.data_loss(A, r_, yt, wt))
        best = min((minimize(ctrl, np.array([a0, b0]), method='Nelder-Mead', options=dict(xatol=1e-12, fatol=1e-18, maxiter=4000)) for a0, b0 in [(.3, -c), (.3 / kk, -c / kk)]), key=lambda r_: r_.fun)
        out[str(c)] = dict(dL_UA=dL, L_control_free_beta=best.fun, k_range=[float(k.min()), float(k.max())])
        assert best.fun - L0 < 1e-12 or best.fun < 1e-12 + L0, (c, best.fun)
    return out


# ------------------------------------------------------------------ S7
def s7():
    sc = synth_scene(np.random.default_rng(7), T=60, G=120, rho_range=(.2, .6))
    ctx = tensors(sc)
    l = torch.as_tensor(sc['truth'][:60])
    x = torch.relu(torch.einsum('tgc,tc->tg', ctx['nt'], l))
    S1_ = C.suff_stats(x, ctx['yt'], ctx['ww'])
    S2_ = C.suff_stats(x, 0.5 * ctx['yt'], ctx['ww'])
    a1, a2 = C.search_amb(S1_)['amb'], C.search_amb(S2_)['amb']
    r1 = C.rho_box(a1 + x, ctx['yt'], ctx['ww'])
    r2 = C.rho_box(a2 + x, 0.5 * ctx['yt'], ctx['ww'])
    assert r1.max() < 1 and abs(a1 - a2) <= 1e-10, (a1, a2)
    assert float((r2 - 0.5 * r1).abs().max()) < 1e-10
    return dict(amb=a1, amb_half=a2, rho_ratio_max_dev=float((r2 - 0.5 * r1).abs().max()))


# ------------------------------------------------------------------ S8
def s8():
    sc = synth_scene(np.random.default_rng(8), T=150, G=60, a_true=.3)
    ctx = tensors(sc)
    rng = np.random.default_rng(88)
    phi = torch.tensor(np.linspace(0, 7, 150) + rng.normal(0, .03, 150), requires_grad=True)
    nu = torch.tensor(rng.normal(0, .3, 150), requires_grad=True)
    C.SEARCH_CALLS[0] = 0
    for _ in range(3):
        total, data, pen, e, amb, rho = C.objective_ua(phi, nu, ctx, 'A1', 1.0)
        total.backward()
    assert C.SEARCH_CALLS[0] == 0 and float(amb) == 0.0
    # 与纯 rho*x 模型逐位相等（同一组 helper）
    l, _ = C.lightdir(phi, nu)
    x = C.shading(ctx['nt'], l)
    rho_ref = C.rho_box(x, ctx['yt'], ctx['ww'])
    data_ref = C.data_loss(x, rho_ref, ctx['yt'], ctx['ww'])
    assert torch.equal(rho, rho_ref) and torch.equal(data, data_ref)
    # 独立 numpy 实现（容差 1e-12）
    xn, yn, wn = x.detach().numpy(), ctx['yt'].numpy(), ctx['ww'].numpy()
    rr = np.clip((wn[..., None] * xn[..., None] * yn).sum(0) / ((wn * xn * xn).sum(0)[:, None] + 1e-6), 0, 1)
    d_np = (wn[..., None] * (rr[None] * xn[..., None] - yn) ** 2).sum() * 0.5 / 3 / wn.sum()
    assert abs(float(data.detach()) - d_np) < 1e-12
    C.objective_ua(phi, nu, ctx, 'A0', 1.0)
    assert C.SEARCH_CALLS[0] == 1
    return dict(search_calls_A1=0, A0_search_calls_after_one_eval=1, amb=0.0, numpy_independent_diff=abs(float(data.detach()) - d_np))


# ------------------------------------------------------------------ S9
class _Stop(Exception):
    pass


def s9():
    res = C.regularization_audit_probe()
    # 优化器参数清单：拦截 Adam / LBFGS 构造
    seen = {}
    orig_adam, orig_lbfgs = torch.optim.Adam, torch.optim.LBFGS

    class SpyAdam(orig_adam):
        def __init__(self, params, *a, **k):
            groups = list(params)
            seen.setdefault('adam', [[tuple(p.shape) for p in (g['params'] if isinstance(g, dict) else [g])] for g in groups])
            super().__init__(groups, *a, **k)

    class SpyLBFGS(orig_lbfgs):
        def __init__(self, *a, **k):
            raise _Stop()
    sc = synth_scene(np.random.default_rng(9), T=150, G=40)
    init = perturbed_init(sc['truth'], np.random.default_rng(99))
    torch.optim.Adam, torch.optim.LBFGS = SpyAdam, SpyLBFGS
    try:
        d = Path(tempfile.mkdtemp())
        try:
            C.fit_traj_ua(sc['n'], sc['y'], sc['w'], init, 0., d, 'spy', arm='A0', steps=3, quiet=True, device='cpu')
        except _Stop:
            pass
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        torch.optim.Adam, torch.optim.LBFGS = orig_adam, orig_lbfgs
    assert seen['adam'] == [[(150,)], [(150,)]], seen
    src = inspect.getsource(C.fit_traj_ua)
    assert 'Pphi = torch.eye' in src and 'Pnu = Pphi.clone()' in src and 'precond' not in src
    res['optimizer_param_groups'] = seen['adam']
    return res


# ------------------------------------------------------------------ S10
def s10():
    sc = synth_scene(np.random.default_rng(10), T=150, G=50)
    ctx = tensors(sc)
    rng = np.random.default_rng(100)
    phi = torch.tensor(np.linspace(0, 7, 150) + rng.normal(0, .03, 150))
    nu = torch.tensor(rng.normal(0, .3, 150))
    base = {arm: C.objective_ua(phi, nu, ctx, arm, 1.0) for arm in ('A0', 'A1')}

    def boom(*a, **k):
        raise AssertionError('绕过路径被调用')
    saved = (r3.kkt, r6.kkt, r6.sh9, r6.env_terms, C.r3.kkt)
    r3.kkt = r6.kkt = r6.sh9 = r6.env_terms = boom
    try:
        for arm in ('A0', 'A1'):
            now = C.objective_ua(phi, nu, ctx, arm, 1.0)
            assert all(torch.equal(u, v) for u, v in zip(now[:2], base[arm][:2]) if torch.is_tensor(u)), arm
            assert torch.equal(now[4], base[arm][4]) and torch.equal(now[5], base[arm][5])
    finally:
        r3.kkt, r6.kkt, r6.sh9, r6.env_terms, C.r3.kkt = saved
    forbidden = ('kkt', 'sh9', 'env_terms', 'get_features', 'eval_sh', 'f_dc', 'f_rest', 'get_albedo', 'mixed_log_intensity', 'd_color', 'shadowed_modulation')
    for fn in (C.objective_ua, C.rho_box, C.suff_stats, C.profile_F, C.search_amb, C.shading, C.lightdir, C.data_loss, C.fit_traj_ua):
        src = inspect.getsource(fn)
        hits = [t for t in forbidden if t in src]
        assert not hits, (fn.__name__, hits)
    return dict(patched=['r3.kkt', 'r6.kkt', 'r6.sh9', 'r6.env_terms'], static_scan_functions=9, forbidden=forbidden)


# ------------------------------------------------------------------ S11
def s11():
    rng = np.random.default_rng(11)
    T, G, Cn = 12, 14, 3
    phi = np.deg2rad(np.arange(T) * 25.)
    e = np.deg2rad(45 + 4 * np.sin(np.arange(T) / 2.))
    l = np.column_stack([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), np.sin(e)])
    n = unit(rng.normal(size=(T, G, 3)) * .3 + unit(rng.normal(size=(G, 3)))[None])
    rho = rng.uniform(.2, .7, (G, Cn))
    rho[0] = 0.0
    rho[1] = 1.0
    amb = .3
    ww = rng.uniform(.5, 1, (T, G))
    yt = torch.as_tensor(rng.uniform(0, 1, (T, G, Cn)))
    nt = torch.as_tensor(n)
    F = D.fisher_ua(nt, yt, torch.as_tensor(ww), l, amb, rho, include_a=True)
    # 稠密对照：J^T J（autograd 雅可比），然后用稠密求逆
    Wsum = ww.sum()
    v = torch.as_tensor(np.sqrt(ww / (Cn * Wsum)))
    act = ((rho > 1e-12) & (rho < 1 - 1e-12)).reshape(-1)
    idx = np.flatnonzero(act)
    rho_t = torch.as_tensor(rho)

    def pred(p):
        ph, ee, a_ = p[:T], p[T:2 * T], p[2 * T]
        rr = rho_t.reshape(-1).clone()
        rr = rr.index_put((torch.as_tensor(idx),), p[2 * T + 1:])
        lv = compose(torch.stack([torch.cos(ph), torch.sin(ph)], 1), ee)
        x = torch.relu(torch.einsum('tgc,tc->tg', nt, lv))
        return (v.unsqueeze(-1) * rr.reshape(G, Cn)[None] * (a_ + x).unsqueeze(-1)).reshape(-1)
    p0 = torch.cat([torch.as_tensor(np.arctan2(l[:, 1], l[:, 0])), torch.as_tensor(np.arcsin(l[:, 2])), torch.tensor([amb], dtype=torch.float64), rho_t.reshape(-1)[torch.as_tensor(idx)]])
    J = torch.autograd.functional.jacobian(pred, p0)
    M = J.T @ J
    Minv = torch.linalg.inv(M)
    th = slice(0, 2 * T)
    Atilde = torch.linalg.inv(Minv[th, th])
    # K-A4：固定方向，(amb, 活跃 rho) 的约化 Fisher（θ 行列去掉）
    ar_idx = list(range(2 * T, M.shape[0]))
    Mar = M[ar_idx][:, ar_idx]
    Mar_inv = torch.linalg.inv(Mar)
    s_a = float(1.0 / Mar_inv[0, 0])
    Faa = float(M[2 * T, 2 * T])
    # 仅消去 rho（amb 固定）：去掉 amb 行列后求逆
    keep = [i for i in range(M.shape[0]) if i != 2 * T]
    Mn = M[keep][:, keep]
    A_only = torch.linalg.inv(torch.linalg.inv(Mn)[th, th])
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    d1, d2 = rel(F['Atilde'], Atilde), rel(F['A'], A_only)
    d3 = abs(F['s_a'] - s_a) / s_a
    d4 = abs(F['r_a'] - s_a / Faa) / (s_a / Faa)
    # corr(amb, mean active rho of channel c)
    corr_ref = []
    for c in range(Cn):
        cols = [2 * T + 1 + j for j, ii in enumerate(idx) if ii % Cn == c]
        u = torch.zeros(Mar.shape[0], dtype=torch.float64)
        u[[cc - 2 * T for cc in cols]] = 1.0 / len(cols)
        ua = torch.zeros(Mar.shape[0], dtype=torch.float64)
        ua[0] = 1.0
        cov = float(ua @ Mar_inv @ u)
        corr_ref.append(cov / np.sqrt(float(ua @ Mar_inv @ ua) * float(u @ Mar_inv @ u)))
    d5 = max(abs(a - b) for a, b in zip(F['corr_amb_rhobar'], corr_ref))
    assert max(d1, d2, d3, d4, d5) < 1e-8, (d1, d2, d3, d4, d5)
    return dict(rel_diff_Atilde=d1, rel_diff_A=d2, rel_diff_s_a=d3, rel_diff_r_a=d4, abs_diff_corr=d5, active_rho=F['active_rho'], total_rho=F['total_rho'])


# ------------------------------------------------------------------ S12
def s12(workdir):
    rng = np.random.default_rng(12)
    T, G = 60, 50
    sc = synth_scene(rng, T=T, G=G, a_true=.0, sigma=.01)
    # 构造 T=60 的环绕：使用前 60 帧（需要总绕行 >= pi）
    phi = np.deg2rad(np.arange(T) * 8.0)
    e = np.deg2rad(45.)
    truth = np.column_stack([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), np.full(T, np.sin(e))])
    n = np.broadcast_to(unit(rng.normal(size=(G, 3)))[None], (T, G, 3)).copy()
    x = np.maximum(np.einsum('tgc,tc->tg', n, truth), 0)
    lum = rng.uniform(.3, .8, G)[None] * x + rng.uniform(.03, .12, G)[None] + rng.normal(0, .005, (T, G))
    w = np.ones((T, G))
    init = np.column_stack([unit(truth[:, :2] + rng.normal(0, .03, (T, 2))) * np.cos(C.E_C), np.full(T, np.sin(C.E_C))])
    o1, o2 = workdir / 's12_ref', workdir / 's12_new'
    o1.mkdir()
    o2.mkdir()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        end1, info1 = C.g1108.fit_traj(n, lum, w, init, 0., o1, 'c01')
        end2, info2 = C.fit_traj_ua(n, lum, w, init, 0., o2, 'c01', arm='A0', objective_fn=lambda p, v, c, s, ov: C.objective_c01(p, v, c, s), quiet=True)
    dmax = float(C.chord_deg(end1, end2).max())
    c1, c2 = info1['certificate'], info2['certificate']
    keys = ['path', 'C1_ok', 'C2_ok', 'C3_ok', 'converged', 'extended', 'outer_iters']
    same = {k: (c1.get(k) == c2.get(k)) for k in keys}
    assert dmax <= 1e-10, dmax
    assert all(same.values()), (same, c1, c2)
    return dict(max_direction_diff_deg=dmax, certificate_fields_equal=same, certificate_path=c1.get('path'), loss_ref=info1['final_loss'], loss_new=info2['final_loss'])


# ------------------------------------------------------------------ S13
def s13():
    # PSNR_lin：预测 = 观测 + 0.1，w 均匀 -> mse=0.01 -> 20 dB
    T, G = 3, 4
    y = np.full((T, G, 3), .4)
    w = np.ones((T, G))
    rgb = D.oetf(y)
    q = D.quality(y + .1, y, rgb, y @ C.KAPPA, w)
    assert abs(q['PSNR_lin'] - 20.0) < 1e-12, q['PSNR_lin']
    # PSNR_sRGB：令 sRGB(预测)=sRGB(观测)+0.1
    rgb2 = np.full((T, G, 3), .5)
    pred2 = r6.srgb_to_linear(rgb2 + .1)
    q2 = D.quality(pred2, r6.srgb_to_linear(rgb2), rgb2, r6.srgb_to_linear(rgb2) @ C.KAPPA, w)
    assert abs(q2['PSNR_sRGB'] - 20.0) < 1e-9, q2['PSNR_sRGB']
    # PSNR_Y：预测 luminance 与 intensity 差 0.1
    pred3 = np.full((T, G, 3), .3)
    q3 = D.quality(pred3, pred3, D.oetf(pred3), np.full((T, G), .3 * (C.KAPPA.sum()) + .1), w)
    assert abs(q3['PSNR_Y'] - 20.0) < 1e-9, q3['PSNR_Y']
    # SSIM_pts 手算：p=[0,1], o=[1,0], w=[1,1]；第三点权重 0 不影响
    C1, C2 = .01 ** 2, .03 ** 2
    hand = (2 * .25 * 1 + C1) * (2 * (-.25) + C2) / ((.5 + C1) * (.5 + C2)) if False else ((2 * .5 * .5 + C1) * (2 * (-.25) + C2)) / ((.25 + .25 + C1) * (.25 + .25 + C2))
    p = np.array([[[0.0] * 3, [1.0] * 3, [.5] * 3]])
    o = np.array([[[1.0] * 3, [0.0] * 3, [.2] * 3]])
    s_val, _ = D.ssim_pts(p, o, np.array([[1.0, 1.0, 0.0]]))
    assert abs(s_val - hand) < 1e-12, (s_val, hand)
    s_one, _ = D.ssim_pts(p, p, np.array([[1.0, 1.0, 1.0]]))
    assert abs(s_one - 1.0) < 1e-12
    return dict(psnr_lin=q['PSNR_lin'], psnr_srgb=q2['PSNR_sRGB'], psnr_Y=q3['PSNR_Y'], ssim_hand=hand, ssim_code=s_val)


# ------------------------------------------------------------------ S14（图非空 + NaN 灰色）
def s14_unit():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    M = np.random.default_rng(14).normal(size=(20, 30))
    M[3:8, 4:9] = np.nan
    im = FG.image(ax, M, vmin=-2, vmax=2)
    bad = im.get_cmap()._rgba_bad
    assert np.allclose(bad[:3], matplotlib_color('lightgray')), bad
    fig.canvas.draw()
    chk = FG.check_figure(fig)
    plt.close(fig)
    assert chk and chk[0]['finite_fraction'] > 0 and chk[0]['pixel_std'] > 0
    # 全 NaN 面板应被检查识别为空
    fig, ax = plt.subplots()
    FG.image(ax, np.full((5, 5), np.nan))
    fig.canvas.draw()
    chk2 = FG.check_figure(fig)
    plt.close(fig)
    assert chk2[0]['finite_fraction'] == 0
    return dict(bad_color=list(map(float, bad)), empty_panel_detected=True)


def matplotlib_color(name):
    import matplotlib.colors as mc
    return mc.to_rgb(name)


def check_fig_dir(out, expected):
    """图文件存在且每个面板有限占比 >0、像素标准差 >0（由 run_arm 写入 diagnostics/figure_checks.json）。"""
    chk = json.loads((out / 'diagnostics/figure_checks.json').read_text())
    assert set(expected) <= set(chk), (expected, list(chk))
    for fn in expected:
        assert (out / fn).stat().st_size > 1000, fn
        for p in chk[fn]:
            assert p['finite_fraction'] > 0 and p['pixel_std'] > 0, (fn, p)
    return {fn: len(chk[fn]) for fn in expected}


def s14_end_to_end(workdir, steps):
    from LIHAN.scripts.goal1109 import fit_t01_uniform_ambient as F
    B = synth_bundle(np.random.default_rng(1409), G=500)
    res = {}
    for arm, expected in (('A0', ['01_support_residual.png', '02_trajectories.png', '03_rho_ambient.png', '04_uncertainty_convergence.png', '05_ambient_profile.png', '06_backlit_residual_nz.png']),
                          ('A1', ['01_support_residual.png', '02_trajectories.png', '03_rho_ambient.png', '04_uncertainty_convergence.png'])):
        out = workdir / f'e2e_{arm}'
        t0 = time.time()
        r = F.run_arm(B, out, arm, steps=steps, quiet=True)
        fc = json.loads((out / 'fit_complete.json').read_text())
        res[arm] = dict(status=r['status'], seconds=time.time() - t0, figure_panels=check_fig_dir(out, expected), flags=fc['flags'], diagnostic_errors=fc['diagnostic_errors'],
                        full_amb=r['flows']['full']['amb'], full_converged=r['flows']['full']['converged'],
                        full_mean_err_deg_vs_synth_truth=float(C.chord_deg(np.load(out / 'runs/full/direction.npy'), B['truth']).mean()))
        assert fc['diagnostic_errors'] == [], fc['diagnostic_errors']
        assert r['status'] in ('AWAITING_REVIEW', 'INCOMPLETE_DIAGNOSTICS'), r['status']
        # A1 不得产生仅 A0 的 K 项文件
        if arm == 'A1':
            assert not (out / 'diagnostics/K_A1_profile.json').exists() and not (out / 'diagnostics/K_A7_chroma.json').exists()
            assert float(np.load(out / 'runs/full/parameters.npz')['amb']) == 0.0
    return res


# ------------------------------------------------------------------ S15（有效性与回退）
def s15(workdir):
    from LIHAN.scripts.goal1109 import fit_t01_uniform_ambient as F
    rng = np.random.default_rng(15)
    good = perturbed_init(np.column_stack([np.cos(np.linspace(0, 7, 150)) * np.cos(C.E_C), np.sin(np.linspace(0, 7, 150)) * np.cos(C.E_C), np.full(150, np.sin(C.E_C))]), rng, 0.0)
    rho_ok = rng.uniform(.1, .9, (10, 3))
    assert C.endpoint_valid(good, rho_ok, .2)
    bad = good.copy()
    bad[5, 0] = np.nan
    assert not C.endpoint_valid(bad, rho_ok, .2)
    assert not C.endpoint_valid(good, rho_ok * 0 + 1.1, .2) and not C.endpoint_valid(good, rho_ok, -.1) and not C.endpoint_valid(good * 1.01, rho_ok, .2)
    below = good.copy()
    below[:, 2] = -.1
    assert not C.endpoint_valid(below / np.linalg.norm(below, axis=1, keepdims=True), rho_ok, .2)
    assert C.choose(False, 'A0') == {'selected': None, 'status': 'NO_VALID_PREDICTION', 'reference_only': 'E4'}
    assert C.choose(True, 'A1')['selected'] == 'A1/full'
    assert C.final_status(False, []) == 'NO_VALID_PREDICTION' and C.final_status(True, ['hi']) == 'INCOMPLETE_DIAGNOSTICS' and C.final_status(True, []) == 'AWAITING_REVIEW'
    rot = good.copy()
    rot[:, :2] = -rot[:, :2]
    assert C.g1108.diff(good, rot)['large_disagreement']
    assert C.repeat_status(good, good) == 'DETERMINISTIC' and C.repeat_status(good, rot) == 'NONDETERMINISTIC' and C.repeat_status(good, None) == 'UNAVAILABLE'
    # 注入：三种情形经 run_arm（假拟合；其余走真实诊断/出图路径）
    B = synth_bundle(np.random.default_rng(1510), G=500)
    real_fit = C.fit_traj_ua
    cases = {}
    for case in ['invalid_full', 'invalid_aux', 'valid_unconverged']:
        def fake(n, y, w, init, nu0, out, label, arm='A0', amb_init_mode=None, **kw):
            dest = out / 'runs' / label
            dest.mkdir(parents=True)
            if (case == 'invalid_full' and label == 'full') or (case == 'invalid_aux' and label == 'hi'):
                raise FloatingPointError('非有限loss(合成注入)')
            end = init.copy()
            if case == 'valid_unconverged' and label in ('lo', 'ainit_lo'):
                end[:, :2] *= -1
            ctx = C.make_ctx(n, y, w, 'cpu')
            x = torch.relu(torch.einsum('tgc,tc->tg', ctx['nt'], torch.as_tensor(end)))
            amb = C.search_amb(C.suff_stats(x, ctx['yt'], ctx['ww']))['amb'] if arm == 'A0' else 0.0
            rho = C.rho_box(amb + x, ctx['yt'], ctx['ww']).numpy()
            np.save(dest / 'direction.npy', end)
            np.savez_compressed(dest / 'parameters.npz', phi=np.zeros(150), nu=np.zeros(150), amb=np.float64(amb), rho=rho)
            write(dest / 'history.json', dict(adam=[], adam_losses=[], optimization=[], final_losses=[1., .9]))
            conv = case != 'valid_unconverged'
            return end, dict(penalties=dict(a1=0., a2=0., e1=0., e2=0., rev=0.), final_loss=1., certificate=dict(path='fake', converged=conv), converged=conv, amb=amb)
        C.fit_traj_ua = fake
        try:
            out = workdir / f's15_{case}'
            r = F.run_arm(B, out, 'A0', quiet=True)
        finally:
            C.fit_traj_ua = real_fit
        sel = json.loads((out / 'selection.json').read_text())
        fc = json.loads((out / 'fit_complete.json').read_text())
        pred = np.load(out / 'predictions.npz')
        if case == 'invalid_full':
            assert sel['selected'] is None and fc['status'] == 'NO_VALID_PREDICTION' and 'A0_full' not in pred.files and 'reference_E4' in pred.files and (out / 'runs/full/failure.json').exists()
            assert not (out / 'diagnostics/K_A1_profile.json').exists()
        if case == 'invalid_aux':
            assert sel['selected'] == 'A0/full' and fc['status'] == 'INCOMPLETE_DIAGNOSTICS' and 'hi' in sel['diagnostic_incomplete'] and 'A0_hi' not in pred.files
        if case == 'valid_unconverged':
            gen = json.loads((out / 'diagnostics/general_7_1.json').read_text())
            assert sel['selected'] == 'A0/full' and 'NOT_CONVERGED' in gen['flags'] and any(f.startswith('LARGE_DISAGREEMENT') for f in gen['flags'])
            assert fc['status'] == 'AWAITING_REVIEW' or fc['status'] == 'INCOMPLETE_DIAGNOSTICS'
        cases[case] = dict(status=fc['status'], selected=sel['selected'], diagnostic_errors=fc['diagnostic_errors'])
    return dict(unit_checks='PASS', injection=cases, no_fallback_to_E4=True)


# ------------------------------------------------------------------ 入口
def run_all(out, e2e_steps=C.STEPS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'work'
    work.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(C.SEED)
    plan = [('S1', s1), ('S2', s2), ('S3', s3), ('S4', s4), ('S5', lambda: s5(work)), ('S6', s6), ('S7', s7), ('S8', s8), ('S9', s9), ('S10', s10), ('S11', s11),
            ('S12', lambda: s12(work)), ('S13', s13), ('S14a', s14_unit), ('S14b', lambda: s14_end_to_end(work, e2e_steps)), ('S15', lambda: s15(work))]
    results, ok = {}, True
    for name, fn in plan:
        t0 = time.time()
        try:
            r = fn()
            results[name] = dict(status='PASS', seconds=time.time() - t0, detail=r)
        except Exception as exc:
            ok = False
            results[name] = dict(status='FAIL', seconds=time.time() - t0, error=repr(exc), traceback=traceback.format_exc())
        print(name, results[name]['status'], f"{results[name]['seconds']:.1f}s", flush=True)
        from LIHAN.scripts.goal1109.fit_t01_uniform_ambient import jsonable
        write(out / 'numeric_tests.json', jsonable(dict(status='PASS' if ok and len(results) == len(plan) else ('FAIL' if not ok else 'RUNNING'), gt_read=False, synthetic_only=True,
                                                         plan_sha256=C.PLAN_SHA, tests=results, utc=C.utc())))
    print('UA_SELF_TEST', 'PASS' if ok else 'FAIL', flush=True)
    if not ok:
        raise SystemExit(1)
