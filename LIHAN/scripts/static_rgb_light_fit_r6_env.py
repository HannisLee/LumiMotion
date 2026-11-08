"""T01(1108-01) R6-ENV：环境光建模(L2 SH辐照度,27参数,乘在rho上) + 线性RGB逐通道光度损失的新Stage 2（不读取GT）。

观测模型 y_tgc = rho_gc * (E_c(n_tg) + max(0, n_tg . l_t))，光强规范 s≡1，rho 每次目标评估时闭式(一维NNLS)求解。
E1-E3与支撑集逐字复用 r3/r5b；只 import，不修改。规划：LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r5b as r5b
from LIHAN.scripts.static_rgb_light_fit_r3 import sha, write, unit, sub, compare, ols_score, initial, kkt
from LIHAN.scripts.static_rgb_light_fit_r5b import elev, compose, penalties, orbit_sign, a_from_z, ang_deg, traj_stats

PLAN_DEFAULT = Path('LIHAN/DOC/规划/1108-01-T01环境光建模与RGB损失新Stage2规划.md')
E4_ROOT = Path('output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角')
OBS = r3.OBS
OBS_HASH = r3.OBS_HASH
R3_PRED = r5b.R3_PRED
STEPS = r5b.STEPS; LR_PHI = r5b.LR_PHI; LR_A = r5b.LR_A; DROPS = r5b.DROPS; BOOT = r5b.BOOT
OUTER = r5b.OUTER; OUTER_EXT = r5b.OUTER_EXT; LB_ITER = r5b.LB_ITER; WIN = r5b.WIN
TC_STEPS = r5b.TC_STEPS; TC_LR = r5b.TC_LR; TC_OUTER = r5b.TC_OUTER
LAM_A, LAM_E, LAM_REV = 1., 10., .1          # 预注册固定，不重跑网格（规划第7节）
EPS_LR = 1e-3                                # epsilon 的 Adam 学习率（规划第3.9节；冒烟未收敛时唯一允许改为3e-4）
COEF_REG = 1e-5                              # 系数正则 1e-5*mean(rho^2)；epsilon 无正则（Q1）
DAMP = 1e-6                                  # rho 闭式解分母稳定项（沿用 r3.kkt）
KAPPA = np.array([.2126, .7152, .0722])      # Rec.709 luminance 权重（只用于一致性断言与诊断投影）
SEED = 103108
BASIS_NAME = 'real_SH_L2_v1'
A_HAT = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3] + [np.pi / 4] * 5)  # 各基函数对应的 Funk-Hecke 系数 A_l


# ---------------------------------------------------------------- 基础函数
def sh9(n):
    """real_SH_L2_v1：k=0..8 顺序 (0,0),(1,-1),(1,0),(1,1),(2,-2),(2,-1),(2,0),(2,1),(2,2)；n 为 (...,3) 的 torch 或 numpy。"""
    lib = torch if torch.is_tensor(n) else np
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    cols = [0.282095 * lib.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z,
            0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y)]
    return lib.stack(cols, -1)


def srgb_to_linear(u):
    u = np.asarray(u, dtype=np.float64)
    return np.where(u <= .04045, u / 12.92, ((np.maximum(u, 0) + .055) / 1.055) ** 2.4)


def rho_closed(A, y, w, damp=DAMP):
    """逐 (g,c) 的一维加权NNLS闭式解（规划3.8(a)）。A,y:(T,G,C)，w:(T,G)；autograd 穿过该闭式解。"""
    ww = w[..., None]
    num = (ww * A * y).sum(0)
    den = (ww * A * A).sum(0) + damp
    return (num / den).clamp_min(0)


def env_terms(l, eps, Yb, nt, yt, ww, damp=DAMP, s=None):
    """给定方向 l:(T,3)、epsilon:(3,9) 返回 (data, rho, res, A, x)。data 为 sum w*omega_c*r^2/2 / sum w，omega_c=1/C。"""
    x = torch.relu(torch.einsum('tgc,tc->tg', nt, l))
    xs = x if s is None else x * s[:, None]
    A = Yb @ eps.T + xs[..., None]
    rho = rho_closed(A, yt, ww, damp)
    res = rho[None] * A - yt
    C = yt.shape[-1]
    data = (ww[..., None] * (1. / C) * .5 * res.square()).sum() / ww.sum()
    return data, rho, res, A, x


def solve_eps(Yb, yt, ww, rho, x):
    """规划3.8(b)：每通道 9x9 正规方程，岭项 1e-12*tr/9。返回 (3,9)。"""
    out = []
    for c in range(yt.shape[-1]):
        r = rho[:, c]
        M = torch.einsum('tg,tgk,tgm->km', ww * (r ** 2)[None], Yb, Yb)
        b = torch.einsum('tg,tgk->k', ww * r[None] * (yt[..., c] - r[None] * x), Yb)
        M = M + 1e-12 * torch.trace(M) / M.shape[0] * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        out.append(torch.linalg.solve(M, b))
    return torch.stack(out, 0)


def als_init(Yb, nt, yt, ww, l, max_iter=500, eps0=None, damp=DAMP):
    """规划3.8(d)：eps=0 起，交替 rho(闭式)/eps(线性)；max|Δeps|<1e-10 且 data 相对变化<1e-12 停止。"""
    with torch.no_grad():
        x = torch.relu(torch.einsum('tgc,tc->tg', nt, l))
        eps = torch.zeros(yt.shape[-1], 9, dtype=yt.dtype, device=yt.device) if eps0 is None else eps0.clone()
        prev = None; iters = 0; dmax = float('nan'); drel = float('nan')
        for it in range(1, max_iter + 1):
            A = Yb @ eps.T + x[..., None]
            rho = rho_closed(A, yt, ww, damp)
            new = solve_eps(Yb, yt, ww, rho, x)
            data = float(env_terms(l, new, Yb, nt, yt, ww, damp)[0])
            dmax = float((new - eps).abs().max()); eps = new; iters = it
            drel = float('inf') if prev is None else abs(prev - data) / max(abs(prev), 1e-300)
            prev = data
            if dmax < 1e-10 and drel < 1e-12:
                break
        return eps, dict(iterations=iters, final_data=prev, last_max_delta_eps=dmax, last_rel_data_change=drel,
                         converged=bool(dmax < 1e-10 and drel < 1e-12))


def eps_precond(Yb, ww, rho, omega):
    """Jacobi 预条件 P_eps=diag(h^-1/2)，h_ck=sum w*omega*rho_gc^2*Y_k^2 / W（规划3.9）。"""
    h = torch.einsum('tg,gc,tgk->ck', ww, rho ** 2, Yb ** 2) * omega / ww.sum()
    return h.clamp_min(1e-12).rsqrt()


def total_objective(phi, a, eps, Yb, nt, yt, ww, lam_a, lam_e, s):
    """规划3.6：data + 1e-5*mean(rho^2) + lam_a(a1+a2) + lam_e(e1+e2) + 0.1*rev；惩罚项直接调用 r5b.penalties。"""
    e = elev(a); l = compose(torch.stack([torch.cos(phi), torch.sin(phi)], 1), e)
    data, rho, _, _, _ = env_terms(l, eps, Yb, nt, yt, ww)
    p = penalties(phi, e, s)
    total = data + COEF_REG * rho.square().mean() + lam_a * (p['a1'] + p['a2']) + lam_e * (p['e1'] + p['e2']) + LAM_REV * p['rev']
    return total, data, p, e, rho


# ---------------------------------------------------------------- 优化：复刻 r5b.fit_traj，增加 epsilon
def fit_traj_env(n, y, w, init, a0, lam_a, lam_e, out, label, eps_lr=EPS_LR, steps=STEPS, outer=OUTER, outer_ext=OUTER_EXT,
                 force_tc=False, force_fallback=False):
    """n:(T,G,3)，y:(T,G,3) 线性RGB(无效为NaN)，w:(T,G)。参数：逐帧phi、a(仰角带宽内)、全局eps(3,9)；rho 闭式。"""
    assert n.ndim == 3 and y.ndim == 3
    dest = out / 'runs' / label; dest.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nt = torch.as_tensor(n, dtype=torch.float64, device=device); Yb = sh9(nt)
    valid = np.isfinite(y).all(-1)
    yt = torch.as_tensor(np.nan_to_num(y), dtype=torch.float64, device=device)
    ww = torch.as_tensor(np.where(valid, w, 0), dtype=torch.float64, device=device)
    C = y.shape[-1]; omega = 1. / C
    phi0 = np.unwrap(np.arctan2(init[:, 1], init[:, 0])); s = orbit_sign(init)
    if abs(np.diff(phi0).sum()) < np.pi:
        raise RuntimeError(f'环绕先验不成立：|sum dphi|={abs(np.diff(phi0).sum())}')
    a0v = np.broadcast_to(a0, (len(init),)).copy()
    phi = torch.nn.Parameter(torch.as_tensor(phi0, dtype=torch.float64, device=device).clone())
    a = torch.nn.Parameter(torch.as_tensor(a0v, dtype=torch.float64, device=device).clone())
    # ---- ALS 初始化（在本流程自己的初始轨迹上）
    with torch.no_grad():
        l_init = compose(torch.stack([torch.cos(phi), torch.sin(phi)], 1), elev(a))
    eps_init, als = als_init(Yb, nt, yt, ww, l_init)
    write(dest / 'als_init.json', als)
    epsp = torch.nn.Parameter(eps_init.clone())

    def objective(phi_=None, a_=None, eps_=None):
        phi_ = phi if phi_ is None else phi_; a_ = a if a_ is None else a_; eps_ = epsp if eps_ is None else eps_
        return total_objective(phi_, a_, eps_, Yb, nt, yt, ww, lam_a, lam_e, s)

    def direction(phi_=None, a_=None):
        phi_ = phi if phi_ is None else phi_; a_ = a if a_ is None else a_
        with torch.no_grad():
            return compose(torch.stack([torch.cos(phi_), torch.sin(phi_)], 1), elev(a_)).cpu().numpy()

    T_ = len(init); D1 = torch.diff(torch.eye(T_, dtype=torch.float64, device=device), dim=0); D2 = torch.diff(D1, dim=0) if T_ > 2 else D1

    def precond(lam, wv):
        L2 = lam / (T_ * 1e-4 * wv); M = torch.eye(T_, dtype=torch.float64, device=device) + L2 * D1.T @ D1 + L2 ** 2 * D2.T @ D2
        ev, U = torch.linalg.eigh(M); return U @ torch.diag(ev.rsqrt()) @ U.T
    Pphi = precond(lam_a, .5); Pa = precond(lam_e, 1.)

    def state(): return phi.detach().clone(), a.detach().clone(), epsp.detach().clone()

    def restore(st):
        with torch.no_grad(): phi.copy_(st[0]); a.copy_(st[1]); epsp.copy_(st[2])

    def make_peps():
        with torch.no_grad():
            rho = objective()[4]
        return eps_precond(Yb, ww, rho, omega)

    hist = []; losses = []
    # ---- T-A Adam ----
    opt = torch.optim.Adam([dict(params=[phi], lr=LR_PHI), dict(params=[a], lr=LR_A), dict(params=[epsp], lr=eps_lr)])
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True); loss, data, p, e, _ = objective()
        if not torch.isfinite(loss): raise RuntimeError('非有限loss(Adam)')
        loss.backward(); opt.step(); losses.append(float(loss.detach()))
        if step in DROPS:
            for g in opt.param_groups: g['lr'] *= .3
        if step % 50 == 0 or step == 1:
            entry = dict(stage='TA', step=step, loss=losses[-1], data=float(data.detach()), **{k: float(v.detach()) for k, v in p.items()},
                         e_min_deg=float(torch.rad2deg(e.min()).detach()), e_max_deg=float(torch.rad2deg(e.max()).detach()), eps=epsp.detach().cpu().flatten().tolist())
            hist.append(entry)
            print(json.dumps({k: v for k, v in dict(run=label, **entry).items() if k != 'eps'}), flush=True)
            write(out / 'progress.json', dict(pid=os.getpid(), run=label, **{k: v for k, v in entry.items() if k != 'eps'}, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

    # ---- T-B L-BFGS + 证书 ----
    def run_lbfgs(max_outer, start=0):
        x0p = phi.detach().clone(); x0a = a.detach().clone(); x0e = epsp.detach().clone()
        Pe = make_peps()
        qp = torch.nn.Parameter(torch.zeros_like(x0p)); qa = torch.nn.Parameter(torch.zeros_like(x0a)); qe = torch.nn.Parameter(torch.zeros_like(x0e))
        lb = torch.optim.LBFGS([qp, qa, qe], lr=1, history_size=50, line_search_fn='strong_wolfe', max_iter=LB_ITER, tolerance_grad=1e-12, tolerance_change=1e-15)
        eff = lambda: (x0p + Pphi @ qp, x0a + Pa @ qa, x0e + Pe * qe)

        def closure():
            lb.zero_grad(); l_ = objective(*eff())[0]; l_.backward(); return l_
        trace = []; dirs = []; epss = []

        def sync():
            with torch.no_grad():
                pe, ae, ee = eff(); phi.copy_(pe); a.copy_(ae); epsp.copy_(ee)

        def verdict(done):
            rel = abs(trace[-1 - WIN] - trace[-1]) / abs(trace[-1 - WIN]); mx = float(ang_deg(dirs[-1 - WIN], dirs[-1]).max()); me = float((epss[-1 - WIN] - epss[-1]).abs().max())
            return rel, mx, me
        for k in range(start, max_outer):
            if force_fallback: raise RuntimeError('LBFGS injected failure (测试用)')
            lb.step(closure)
            with torch.no_grad():
                pe, ae, ee = eff(); l_ = objective(pe, ae, ee)[0]
            if not torch.isfinite(l_): raise FloatingPointError('非有限loss(LBFGS)')
            trace.append(float(l_)); dirs.append(direction(pe, ae)); epss.append(ee.detach().clone())
            if len(trace) % 10 == 0: print(json.dumps(dict(run=label, stage='TB', outer=k + 1, loss=trace[-1])), flush=True)
            if len(trace) > WIN:
                rel, mx, me = verdict(False)
                if rel < 1e-9 and mx < .005 and me < 1e-6: sync(); return trace, dirs, True, rel, mx, me
        sync(); rel, mx, me = verdict(True); return trace, dirs, False, rel, mx, me

    def probe():
        base = direction(); st = state(); lb0 = float(objective()[0].detach())
        o = torch.optim.Adam([dict(params=[phi], lr=1e-4), dict(params=[a], lr=1e-3), dict(params=[epsp], lr=1e-5)])
        for _ in range(500):
            o.zero_grad(); l_ = objective()[0]; l_.backward(); o.step()
        d1 = direction(); l1 = float(objective()[0].detach()); disp = float(ang_deg(base, d1).max()); drop = (lb0 - l1) / abs(lb0)
        de = float((epsp.detach() - st[2]).abs().max()); restore(st); return disp, drop, de

    def cert_from(path, rel, mx, me, ext, nout, disp, drop, de, **kw):
        c = dict(path=path, C1_rel=rel, C1_ok=bool(rel < 1e-9), C2_max_deg=mx, C2_max_eps=me, C2_ok=bool(mx < .005 and me < 1e-6), extended=ext, outer_iters=nout,
                 C3_disp_deg=disp, C3_drop=drop, C3_eps_disp=de, C3_ok=bool(disp < .1 and drop < 1e-4 and de < 1e-3), **kw)
        c['converged'] = bool(c['C1_ok'] and c['C2_ok'] and c['C3_ok']); return c
    cert = dict(path='lbfgs'); fallback = False
    try:
        trace, dirs, ok, rel, mx, me = run_lbfgs(outer)
        ext = False
        if not ok:
            ext = True; t2, d2, ok, rel, mx, me = run_lbfgs(outer_ext, len(trace)); trace += t2
        disp, drop, de = probe(); cert = cert_from('lbfgs', rel, mx, me, ext, len(trace), disp, drop, de)
        if (not cert['converged']) or force_tc:
            # T-C：预条件 q 空间 Adam 长程续跑(余弦衰减) + 短 L-BFGS 抛光，证书重新判定（阈值不变）
            pre = dict(cert); x0p = phi.detach().clone(); x0a = a.detach().clone(); x0e = epsp.detach().clone(); Pe = make_peps()
            qp = torch.nn.Parameter(torch.zeros_like(x0p)); qa = torch.nn.Parameter(torch.zeros_like(x0a)); qe = torch.nn.Parameter(torch.zeros_like(x0e))
            oc = torch.optim.Adam([qp, qa, qe], lr=TC_LR); sc = torch.optim.lr_scheduler.CosineAnnealingLR(oc, TC_STEPS, eta_min=0)
            for k in range(TC_STEPS):
                oc.zero_grad(); l_ = objective(x0p + Pphi @ qp, x0a + Pa @ qa, x0e + Pe * qe)[0]
                if not torch.isfinite(l_): raise FloatingPointError('非有限loss(LBFGS-TC)')
                l_.backward(); oc.step(); sc.step()
                if k % 500 == 0: print(json.dumps(dict(run=label, stage='TC', step=k, loss=float(l_.detach()))), flush=True)
            with torch.no_grad(): phi.copy_(x0p + Pphi @ qp); a.copy_(x0a + Pa @ qa); epsp.copy_(x0e + Pe * qe)
            t3, d3, ok3, rel3, mx3, me3 = run_lbfgs(TC_OUTER); trace += t3
            disp, drop, de = probe()
            cert = cert_from('lbfgs+TC', rel3, mx3, me3, ext, len(trace), disp, drop, de, pre_TC=pre, TC_steps=TC_STEPS, TC_lr=TC_LR)
    except (FloatingPointError, RuntimeError) as exc:
        if 'LBFGS' not in str(exc): raise
        fallback = True
        restore((torch.as_tensor(phi0, dtype=torch.float64, device=device), torch.as_tensor(a0v, dtype=torch.float64, device=device), eps_init.clone()))
        o = torch.optim.Adam([dict(params=[phi], lr=LR_PHI), dict(params=[a], lr=LR_A), dict(params=[epsp], lr=eps_lr)]); N = 6000 + steps
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(o, N, eta_min=0); fl = []
        for _ in range(N):
            o.zero_grad(); l_ = objective()[0]; l_.backward(); o.step(); sched.step(); fl.append(float(l_.detach()))
        rel = abs(fl[-501] - fl[-1]) / abs(fl[-501]); disp, drop, de = probe()
        cert = dict(path='adam_fallback', reason=str(exc), C1_rel=rel, C3_disp_deg=disp, C3_drop=drop, C3_eps_disp=de,
                    converged=bool(rel < 1e-7 and disp < .1 and drop < 1e-4 and de < 1e-3))
    with torch.no_grad():
        loss, data, p, e, rho = objective(); end = direction(); pp = {k: float(v) for k, v in p.items()}; eld = np.rad2deg(e.cpu().numpy())
    info = dict(penalties=pp, orbit_sign=s, final_loss=float(loss), final_data=float(data), certificate=cert, converged=cert['converged'], elev_deg=eld.tolist(),
                eps=epsp.detach().cpu().numpy().copy(), rho=rho.cpu().numpy().copy(), als_init=als, eps_init=eps_init.cpu().numpy().copy())
    np.save(dest / 'direction.npy', end); np.save(dest / 'eps.npy', info['eps']); write(dest / 'history.json', hist); write(dest / 'certificate.json', cert)
    write(dest / 'config.json', dict(model='R6-ENV', basis=BASIS_NAME, gauge='s=1', lam_a=lam_a, lam_e=lam_e, lam_rev=LAM_REV, coef_reg=COEF_REG, damp=DAMP, omega=omega, adam_steps=steps,
                                     lr_phi=LR_PHI, lr_a=LR_A, lr_eps=eps_lr, drops=list(DROPS), e_center_deg=45., e_halfwidth_deg=5., device=device, dtype='float64',
                                     support_count=y.shape[1], final_loss=float(loss), certificate=cert, penalties=pp, adam_fallback=fallback))
    print(json.dumps(dict(run=label, stage='DONE', final_loss=float(loss), certificate=cert)), flush=True)
    return end, info


# ================================================================ 诊断 K1–K8（只报告，不改输出，不读GT）
def fibonacci(N):
    i = np.arange(N) + .5; z = 1 - 2 * i / N; th = np.pi * (1 + 5 ** .5) * i
    r = np.sqrt(1 - z * z); return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


def make_ctx(n, y, w):
    """CPU float64 诊断上下文。y:(T,G,3) 线性RGB(无效NaN)。"""
    valid = np.isfinite(y).all(-1)
    nt = torch.as_tensor(n, dtype=torch.float64)
    return dict(nt=nt, Yb=sh9(nt), yt=torch.as_tensor(np.nan_to_num(y), dtype=torch.float64), ww=torch.as_tensor(np.where(valid, w, 0), dtype=torch.float64))


def dir_angles(l):
    l = np.asarray(l); return np.arctan2(l[:, 1], l[:, 0]), np.arcsin(np.clip(l[:, 2], -1, 1))


def theta_partials(nt, l):
    """phi、e 坐标下的 ∂(n.l)/∂θ_t 乘 Heaviside：(T,G) 两个张量与 H。"""
    phi, e = dir_angles(l.numpy() if torch.is_tensor(l) else l)
    phi = torch.as_tensor(phi); e = torch.as_tensor(e)
    dlp = torch.stack([-torch.cos(e) * torch.sin(phi), torch.cos(e) * torch.cos(phi), torch.zeros_like(e)], 1)
    dle = torch.stack([-torch.sin(e) * torch.cos(phi), -torch.sin(e) * torch.sin(phi), torch.cos(e)], 1)
    lt = torch.as_tensor(l) if not torch.is_tensor(l) else l
    H = (torch.einsum('tgc,tc->tg', nt, lt) > 0).to(torch.float64)
    return H * torch.einsum('tgc,tc->tg', nt, dlp), H * torch.einsum('tgc,tc->tg', nt, dle)


def fisher_env(ctx, l, eps):
    """K1：返回 dict(A=ρ消去后θθ块, B=εθ块, C=εε块)，θ=(phi_1..T, e_1..T)(弧度)。规划6.4。"""
    nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']
    T, G, C = yt.shape; lt = torch.as_tensor(l, dtype=torch.float64); eps = torch.as_tensor(eps, dtype=torch.float64)
    _, rho, _, A, x = env_terms(lt, eps, Yb, nt, yt, ww)
    v = ww[..., None] / C / ww.sum()
    pp, pe = theta_partials(nt, lt)
    r2 = rho ** 2
    Ath = torch.zeros(2 * T, 2 * T, dtype=torch.float64)
    ar = torch.arange(T)
    Ath[ar, ar] = torch.einsum('tgc,gc,tg->t', v, r2, pp * pp); Ath[T + ar, T + ar] = torch.einsum('tgc,gc,tg->t', v, r2, pe * pe)
    pe_pp = torch.einsum('tgc,gc,tg->t', v, r2, pp * pe); Ath[ar, T + ar] = pe_pp; Ath[T + ar, ar] = pe_pp
    Fthr = torch.cat([(v * rho[None] * pp[..., None] * A).reshape(T, G * C), (v * rho[None] * pe[..., None] * A).reshape(T, G * C)], 0)
    Fthe = torch.cat([torch.einsum('tgc,gc,tg,tgk->tck', v, r2, pp, Yb).reshape(T, C * 9), torch.einsum('tgc,gc,tg,tgk->tck', v, r2, pe, Yb).reshape(T, C * 9)], 0)
    Fre_ = torch.einsum('tgc,tgk->gck', v * A, Yb) * rho[..., None]
    Fre = torch.zeros(G, C, C, 9, dtype=torch.float64)
    for c in range(C): Fre[:, c, c, :] = Fre_[:, c, :]
    Fre = Fre.reshape(G * C, C * 9)
    Fee_ = torch.einsum('tgc,gc,tgk,tgm->ckm', v, r2, Yb, Yb); Fee = torch.zeros(C * 9, C * 9, dtype=torch.float64)
    for c in range(C): Fee[c * 9:(c + 1) * 9, c * 9:(c + 1) * 9] = Fee_[c]
    Frr = torch.einsum('tgc,tgc->gc', v, A * A).reshape(G * C)
    act = (rho.reshape(G * C) > 1e-12) & (Frr > 1e-300)
    Fuu = torch.cat([torch.cat([Ath, Fthe], 1), torch.cat([Fthe.T, Fee], 1)], 0)
    Fur = torch.cat([Fthr, Fre.T], 0)[:, act]
    S = Fuu - (Fur / Frr[act][None]) @ Fur.T
    return dict(A=S[:2 * T, :2 * T], B=S[2 * T:, :2 * T], C=S[2 * T:, 2 * T:], active_rho=int(act.sum()), total_rho=int(G * C))


def fisher_e4(ctx, ylum, w, l):
    """E4 对照：luminance 数据，干扰参数 (rho_g,b_g)。返回 A_rho(只消去rho，b已知) 与 A_both(消去rho,b)。"""
    nt = ctx['nt']; T, G = ylum.shape
    yt = torch.as_tensor(np.nan_to_num(ylum), dtype=torch.float64); ww = torch.as_tensor(np.where(np.isfinite(ylum), w, 0), dtype=torch.float64)
    lt = torch.as_tensor(l, dtype=torch.float64); x = torch.relu(torch.einsum('tgc,tc->tg', nt, lt))
    rho, b = kkt(x, yt, ww); v = ww / ww.sum(); pp, pe = theta_partials(nt, lt)
    Ath = torch.zeros(2 * T, 2 * T, dtype=torch.float64); ar = torch.arange(T); r2 = rho ** 2
    Ath[ar, ar] = torch.einsum('tg,g,tg->t', v, r2, pp * pp); Ath[T + ar, T + ar] = torch.einsum('tg,g,tg->t', v, r2, pe * pe)
    c_ = torch.einsum('tg,g,tg->t', v, r2, pp * pe); Ath[ar, T + ar] = c_; Ath[T + ar, ar] = c_
    Jr = torch.cat([v * rho[None] * pp * x, v * rho[None] * pe * x], 0)       # (2T,G)  ∂r/∂rho 与 θ 的交叉
    Jb = torch.cat([v * rho[None] * pp, v * rho[None] * pe], 0)
    Frr = (v * x * x).sum(0); Frb = (v * x).sum(0); Fbb = v.sum(0)
    on = rho > 1e-12; onb = on & (b > 1e-12)
    A_rho = Ath - ((Jr[:, on] / Frr[on][None]) @ Jr[:, on].T)
    # rho,b 两者同时为 nuisance 的 g：2x2 块
    V = torch.stack([Jr[:, onb].T, Jb[:, onb].T], 2)                         # (g,2T,2)
    N = torch.stack([torch.stack([Frr[onb], Frb[onb]], 1), torch.stack([Frb[onb], Fbb[onb]], 1)], 1)  # (g,2,2)
    N = N + 1e-14 * torch.eye(2, dtype=torch.float64)[None]
    corr_both = torch.einsum('gik,gkl,gjl->ij', V, torch.linalg.inv(N), V)
    only = on & ~onb
    corr_only = (Jr[:, only] / Frr[only][None]) @ Jr[:, only].T
    A_both = Ath - corr_both - corr_only
    data = float((ww * .5 * (rho[None] * x + b[None] - yt).square()).sum() / ww.sum())
    return dict(A_rho=A_rho, A_both=A_both, data=data, active_rho=int(on.sum()), active_b=int(onb.sum()))


def prior_hessian(phi, e, s):
    T = len(phi)
    def f(xv):
        p = penalties(xv[:T], xv[T:], s); return LAM_A * (p['a1'] + p['a2']) + LAM_E * (p['e1'] + p['e2']) + LAM_REV * p['rev']
    return torch.autograd.functional.hessian(f, torch.cat([torch.as_tensor(phi, dtype=torch.float64), torch.as_tensor(e, dtype=torch.float64)]))


def dct_modes(T):
    out = {}
    t = np.arange(T)
    for k in range(1, 11):
        u = np.cos(np.pi * k * (t + .5) / T); u /= np.linalg.norm(u); out[f'dct{k}'] = u
    return out


def mode_vectors(T):
    z = np.zeros(T); o = np.ones(T) / np.sqrt(T); modes = {'u_e0': np.concatenate([z, o]), 'u_phi0': np.concatenate([o, z])}
    for k, u in dct_modes(T).items():
        modes[f'e_{k}'] = np.concatenate([z, u]); modes[f'phi_{k}'] = np.concatenate([u, z])
    return modes


def mode_info(M, u):
    M = M + 1e-12 * torch.trace(M) / M.shape[0] * torch.eye(M.shape[0], dtype=M.dtype)
    u = torch.as_tensor(u, dtype=torch.float64); return float(1. / (u @ torch.linalg.solve(M, u)))


def fisher_report(ctx, l, eps, rho_unused, phi, e, s, e4_lum, e4_w, e4_l, data_r6, data_e4):
    """K1 汇总：模式保留率、E4 对照、典型相关。"""
    F = fisher_env(ctx, l, eps); P = prior_hessian(phi, e, s); T = len(phi)
    A, B, Cc = F['A'], F['B'], F['C']
    At = A - B.T @ torch.linalg.solve(Cc + 1e-12 * torch.trace(Cc) / Cc.shape[0] * torch.eye(Cc.shape[0], dtype=torch.float64), B)
    tau = 1e-10 * float(torch.trace(A)) / (2 * T); I2 = torch.eye(2 * T, dtype=torch.float64)
    E4 = fisher_e4(ctx, e4_lum, e4_w, e4_l); modes = mode_vectors(T); rep = {}
    for name, u in modes.items():
        a_p, at_p = mode_info(A + P, u), mode_info(At + P, u)
        a_d, at_d = mode_info(A + tau * I2, u), mode_info(At + tau * I2, u)
        e_p, eb_p = mode_info(E4['A_rho'] + P, u), mode_info(E4['A_both'] + P, u)
        e_d, eb_d = mode_info(E4['A_rho'] + tau * I2, u), mode_info(E4['A_both'] + tau * I2, u)
        rep[name] = dict(R_new_with_prior=at_p / a_p, R_new_data_only=at_d / a_d, R_E4_with_prior=eb_p / e_p, R_E4_data_only=eb_d / e_d,
                         info_new_rhoeps_eliminated_over_sigma2=at_p / (2 * data_r6), info_E4_eliminated_over_sigma2=eb_p / (2 * data_e4),
                         info_new_eliminated_prior_only_total=at_p)
    Ai = A + tau * I2
    Lc = torch.linalg.cholesky(Ai); Lcc = torch.linalg.cholesky(Cc + 1e-12 * torch.trace(Cc) / Cc.shape[0] * torch.eye(Cc.shape[0], dtype=torch.float64))
    Z = torch.linalg.solve_triangular(Lcc, torch.linalg.solve_triangular(Lc, B.T, upper=False).T, upper=False)  # 白化后的 θ-ε 耦合
    sv = torch.linalg.svdvals(Z)[:5].tolist()
    return dict(modes=rep, canonical_corr_top5_theta_eps=sv, flag_F1=bool(rep['u_e0']['R_new_with_prior'] < .1), F1_value=rep['u_e0']['R_new_with_prior'],
                active_rho=F['active_rho'], total_rho=F['total_rho'], e4_active_rho=E4['active_rho'], e4_active_b=E4['active_b'],
                sigma2_r6=2 * data_r6, sigma2_e4=2 * data_e4, note='同噪声模型假设下的信息量比；保留率R=消去epsilon(或b)后相对只消去rho的信息比', gt_read=False)


def energy_share(ctx, l, eps, e4_lum, e4_w, e4_l):
    """K2：环境项能量占比（线性、时变方差），分通道、受光/背光。"""
    nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']; C = yt.shape[-1]
    lt = torch.as_tensor(l, dtype=torch.float64); _, rho, _, A, x = env_terms(lt, torch.as_tensor(eps, dtype=torch.float64), Yb, nt, yt, ww)
    Et = Yb @ torch.as_tensor(eps, dtype=torch.float64).T; ren = rho[None] * Et; yh = rho[None] * A; v = ww[..., None].expand_as(yh)
    lit = (x > 0)[..., None].expand_as(yh)
    def tmean(z): return (v * z).sum(0, keepdim=True) / v.sum(0, keepdim=True).clamp_min(1e-300)
    dev_e = ren - tmean(ren); dev_y = yh - tmean(yh); res = {}
    for name, m in [('all', torch.ones_like(lit)), ('lit', lit), ('back', ~lit)]:
        vm = v * m
        res[name] = dict(eta_lin=[float((vm[..., c] * ren[..., c]).sum() / (vm[..., c] * yh[..., c]).sum().clamp_min(1e-300)) for c in range(C)],
                         eta_var=[float((vm[..., c] * dev_e[..., c].square()).sum() / (vm[..., c] * dev_y[..., c].square()).sum().clamp_min(1e-300)) for c in range(C)])
    nt_ = ctx['nt']; yl = torch.as_tensor(np.nan_to_num(e4_lum), dtype=torch.float64); wl = torch.as_tensor(np.where(np.isfinite(e4_lum), e4_w, 0), dtype=torch.float64)
    xl = torch.relu(torch.einsum('tgc,tc->tg', nt_, torch.as_tensor(e4_l, dtype=torch.float64))); rl, bl = kkt(xl, yl, wl)
    e4 = dict(eta_lin=float((wl * bl[None]).sum() / (wl * (xl * rl[None] + bl[None])).sum()), eta_var=0.)
    return dict(r6=res, e4=e4, flag_F2=bool(max(res['all']['eta_var']) > .25), F2_max_eta_var=max(res['all']['eta_var']), gt_read=False)


def sphere_irradiance(eps, N=10000):
    pts = fibonacci(N); E = sh9(pts) @ np.asarray(eps).T; return pts, E


def irradiance_check(ctx, eps):
    """K3：辐照度物理性（只报告，不施加约束）。"""
    Yb, ww = ctx['Yb'], ctx['ww']; Et = (Yb @ torch.as_tensor(eps, dtype=torch.float64).T)
    neg = (Et < 0).any(-1).to(torch.float64); frac = float((ww * neg).sum() / ww.sum())
    pts, E = sphere_irradiance(eps); area = (E < 0).mean(0)
    return dict(obs_negative_weighted_fraction=frac, sphere_negative_area_fraction=area.tolist(), sphere_min=E.min(0).tolist(), sphere_max=E.max(0).tolist(),
                flag_F3=bool(frac > .01), gt_read=False), pts, E


def dir_from_ea(phi, e):
    return np.stack([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), np.sin(e)], 1)


def profile_zmode(ctx, l, eps, e4_lum, e4_w, e4_l, deltas=(-4, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 4), als_iter=300):
    """K4：沿整体仰角平移 Δ(度，φ不变、不受带宽限制)的 data 轮廓；(a)ALS重解rho,eps (b)eps固定只重解rho (c)E4 kkt。"""
    nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']; eps = torch.as_tensor(eps, dtype=torch.float64)
    phi, e = dir_angles(l); out = {'delta_deg': list(deltas), 'a_R6_als': [], 'b_R6_eps_fixed': [], 'c_E4_kkt': []}
    yl = torch.as_tensor(np.nan_to_num(e4_lum), dtype=torch.float64); wl = torch.as_tensor(np.where(np.isfinite(e4_lum), e4_w, 0), dtype=torch.float64)
    ph4, e4e = dir_angles(e4_l)
    for d in deltas:
        dr = np.deg2rad(d); lt = torch.as_tensor(dir_from_ea(phi, e + dr))
        ea, _ = als_init(Yb, nt, yt, ww, lt, max_iter=als_iter, eps0=eps); out['a_R6_als'].append(float(env_terms(lt, ea, Yb, nt, yt, ww)[0]))
        out['b_R6_eps_fixed'].append(float(env_terms(lt, eps, Yb, nt, yt, ww)[0]))
        l4 = torch.as_tensor(dir_from_ea(ph4, e4e + dr)); x4 = torch.relu(torch.einsum('tgc,tc->tg', nt, l4)); r4, b4 = kkt(x4, yl, wl)
        out['c_E4_kkt'].append(float((wl * .5 * (r4[None] * x4 + b4[None] - yl).square()).sum() / wl.sum()))
    curv = {}
    for k in ['a_R6_als', 'b_R6_eps_fixed', 'c_E4_kkt']:
        arr = np.array(out[k]); i0 = list(deltas).index(0); dl = arr - arr[i0]
        coef = np.polyfit(np.array(deltas, dtype=float), dl, 2); curv[k] = dict(delta_data=dl.tolist(), curvature_per_deg2=float(2 * coef[0]), linear_per_deg=float(coef[1]), min_at_delta_deg=float(-coef[1] / (2 * coef[0])) if coef[0] > 0 else None)
    out['curvature'] = curv; out['curvature_ratio_a_over_b'] = curv['a_R6_als']['curvature_per_deg2'] / curv['b_R6_eps_fixed']['curvature_per_deg2'] if curv['b_R6_eps_fixed']['curvature_per_deg2'] else None
    out['gt_read'] = False; return out


def wquantile(v, w, qs):
    o = np.argsort(v); v = v[o]; cw = np.cumsum(w[o]); cw /= cw[-1]; return [float(v[min(np.searchsorted(cw, q), len(v) - 1)]) for q in qs]


def channel_residuals(ctx, l, eps, e4_lum, e4_w, e4_l):
    """K5：逐通道残差统计。"""
    nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']; C = yt.shape[-1]; T = yt.shape[0]
    lt = torch.as_tensor(l, dtype=torch.float64); _, rho, res, A, x = env_terms(lt, torch.as_tensor(eps, dtype=torch.float64), Yb, nt, yt, ww)
    r = res.numpy(); w = ww.numpy(); lit = (x.numpy() > 0); nz = nt[..., 2].numpy(); out = {'per_channel': {}, 'frame_rms': []}
    bins = [(-1.01, -.5), (-.5, 0), (0, .5), (.5, .85), (.85, 1.01)]
    groups = {'all': np.ones_like(lit), 'lit': lit, 'back': ~lit}
    for i, (lo, hi) in enumerate(bins): groups[f'nz_bin{i}[{lo:g},{hi:g})'] = (nz >= lo) & (nz < hi)
    rms_all = []
    for c in range(C):
        d = {}
        for name, m in groups.items():
            wm = w * m; sw = wm.sum()
            if sw <= 0: d[name] = None; continue
            rc = r[..., c]; q = wquantile(rc[wm > 0], wm[wm > 0], [.05, .5, .95])
            d[name] = dict(mean=float((wm * rc).sum() / sw), rms=float(np.sqrt((wm * rc ** 2).sum() / sw)), mae=float((wm * abs(rc)).sum() / sw), p5=q[0], p50=q[1], p95=q[2], weight=float(sw))
        out['per_channel']['RGB'[c]] = d; rms_all.append(d['all']['rms'])
    out['frame_rms'] = [[float(np.sqrt((w[t] * r[t, :, c] ** 2).sum() / max(w[t].sum(), 1e-300))) for t in range(T)] for c in range(C)]
    sw = w.sum(); flat = r.reshape(-1, C); wf = np.repeat(w.reshape(-1), 1); mean = (wf[:, None] * flat).sum(0) / sw
    cov = ((flat - mean).T * wf) @ (flat - mean) / sw; sd = np.sqrt(np.diag(cov)); out['channel_corr'] = (cov / np.outer(sd, sd)).tolist()
    lum_r = (r * KAPPA[None, None, :C]).sum(-1) if C == 3 else r[..., 0]
    out['r6_luminance_projected_rms'] = float(np.sqrt((w * lum_r ** 2).sum() / sw))
    yl = torch.as_tensor(np.nan_to_num(e4_lum), dtype=torch.float64); wl = torch.as_tensor(np.where(np.isfinite(e4_lum), e4_w, 0), dtype=torch.float64)
    xl = torch.relu(torch.einsum('tgc,tc->tg', nt, torch.as_tensor(e4_l, dtype=torch.float64))); rl, bl = kkt(xl, yl, wl)
    re = (rl[None] * xl + bl[None] - yl).numpy(); out['e4_luminance_rms_at_frozen_direction'] = float(np.sqrt((wl.numpy() * re ** 2).sum() / wl.numpy().sum()))
    out['rms_ratio_max_over_min'] = float(max(rms_all) / min(rms_all)); out['flag_F4'] = bool(out['rms_ratio_max_over_min'] > 2); out['gt_read'] = False; return out


def elev_deg_(l): return np.rad2deg(np.arcsin(np.clip(l[:, 2], -1, 1)))


def repeatability(ends, infos, e4_row):
    """K6：A/B、hi/lo、bootstrap 诊断（不作门槛）。"""
    sp = compare(ends['A'], ends['B']); hl = compare(ends['hi'], ends['lo'])
    row = dict(split_mean_deg=sp['mean_deg'], split_median_deg=sp['median_deg'], split_p90_deg=sp['p90_deg'], split_el_mae_deg=float(abs(elev_deg_(ends['A']) - elev_deg_(ends['B'])).mean()),
               hilo_mean_deg=hl['mean_deg'], hilo_elev_signed_diff_deg=float((elev_deg_(ends['hi']) - elev_deg_(ends['lo'])).mean()),
               boot_each_mean_deg=[compare(ends[f'boot{s}'], ends['full'])['mean_deg'] for s in BOOT])
    row['boot_mean_deg'] = float(np.mean(row['boot_each_mean_deg']))
    row['J_components'] = dict(split_mean=row['split_mean_deg'], half_split_p90=.5 * row['split_p90_deg'], split_el_mae=row['split_el_mae_deg'], boot_mean=row['boot_mean_deg'], hilo_mean=row['hilo_mean_deg'])
    row['J'] = float(sum(row['J_components'].values()))
    return dict(r6=row, e4_candidate_a1e00_e1e01=e4_row, certificates={k: v['certificate'] for k, v in infos.items()}, converged={k: bool(v['converged']) for k, v in infos.items()},
                note='只解释为对支撑/初值/抽样的敏感性，不解释为精度；不作门槛', gt_read=False)


def heldout(n_val, y_lin_val, y_lum_val, w_val, eps, dirs):
    """K7：(a) E4口径 luminance 留出MAE (b) 新模型口径：固定eps，奇数帧闭式rho，偶数帧逐通道截断MAE。"""
    res_a = {k: ols_score(n_val, y_lum_val, w_val, d) for k, d in dirs.items()}
    eps_t = torch.as_tensor(eps, dtype=torch.float64); out_b = {}
    ctx = make_ctx(n_val, y_lin_val, w_val); nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']; T = yt.shape[0]
    cal = torch.zeros(T, 1, dtype=torch.float64); cal[::2] = 1.; tst = 1 - cal
    for k, d in dirs.items():
        x = torch.relu(torch.einsum('tgc,tc->tg', nt, torch.as_tensor(d, dtype=torch.float64))); A = Yb @ eps_t.T + x[..., None]
        rho = rho_closed(A, yt, ww * cal)
        res = (rho[None] * A - yt).numpy(); wt = (ww * tst).numpy(); ok = (ww * cal).sum(0).numpy() > 0
        wt = wt * ok[None]; clipped = np.minimum(abs(res), .2)
        lum = np.minimum(abs((res * KAPPA[None, None, :]).sum(-1)), .2) if res.shape[-1] == 3 else None
        out_b[k] = dict(channel_mae=[float((wt * clipped[..., c]).sum() / wt.sum()) for c in range(res.shape[-1])], luminance_projected_mae=float((wt * lum).sum() / wt.sum()) if lum is not None else None)
    return dict(E4_style_luminance_ols_mae=res_a, new_style_fixed_eps_mae=out_b, gt_read=False)


def stage1_chroma(ids_fit, rho, ply_path, ids_all):
    """K8（可选）：Stage 1 DC 色（线性化）与 rho_gc 的 RGB 夹角中位数。"""
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))['vertex']; N = len(v.data)
    assert ids_all.min() >= 0 and ids_all.max() < N and len(np.unique(ids_all)) == len(ids_all), 'gaussian_ids 与 PLY 行索引不一致'
    dc = np.stack([np.asarray(v[f'f_dc_{i}']) for i in range(3)], 1)[ids_all[ids_fit]]
    col = np.clip(.5 + .28209479177387814 * dc, .03, .97); lin = srgb_to_linear(col)
    cs = (lin * rho).sum(1) / np.maximum(np.linalg.norm(lin, axis=1) * np.linalg.norm(rho, axis=1), 1e-12)
    ang = np.rad2deg(np.arccos(np.clip(cs, -1, 1))); ok = np.linalg.norm(rho, axis=1) > 1e-8
    return dict(median_angle_deg=float(np.median(ang[ok])), p90_angle_deg=float(np.percentile(ang[ok], 90)), count=int(ok.sum()), ply_vertices=int(N), gt_read=False)


# ================================================================ 合成自检 S1–S10（合成真值不是数据集GT）
EPS_TRUE = np.array([[.30 / .282095, .02 / .488603, .05 / .488603, .02 / .488603, .01, .01, .02, .01, .01],
                     [.28 / .282095, -.03 / .488603, .08 / .488603, .01 / .488603, .01, -.01, .02, .01, -.01],
                     [.35 / .282095, .01 / .488603, .12 / .488603, -.02 / .488603, -.01, .01, -.02, .01, .02]])


def synth_scene(rng, T=150, G=600, sigma=.003):
    """150帧变速环绕，仰角在带宽内平滑游走；1/3高斯带随时间旋转的法线；rho∈U(.3,.8)^3；正的L2环境(β_z≠0,三通道不同色)。"""
    nn = unit(rng.normal(size=(G, 3)))
    ang = .4 * np.sin(np.linspace(0, 2 * np.pi * 1.5, T)); rot = nn[:G // 3]
    nt = np.broadcast_to(nn, (T, G, 3)).copy()
    for t in range(T):   # 绕x轴转动 ang[t]
        c, s_ = np.cos(ang[t]), np.sin(ang[t]); R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]); nt[t, :G // 3] = rot @ R.T
    speed = 2.4 * (1 + .3 * np.sin(np.linspace(0, 3, T))); phi = np.deg2rad(np.cumsum(speed))
    walk = np.cumsum(rng.normal(0, .15, T)); walk = np.convolve(walk, np.ones(9) / 9, 'same'); walk -= walk.mean(); walk = 3.9 * walk / abs(walk).max()
    truth = dir_from_ea(phi, np.deg2rad(45 + walk)); rho = rng.uniform(.3, .8, (G, 3))
    E = sh9(nt) @ EPS_TRUE.T; x = np.maximum(np.einsum('tgc,tc->tg', nt, truth), 0)
    y = rho[None] * (E + x[..., None]) + rng.normal(0, sigma, (T, G, 3)); w = np.ones((T, G))
    init = np.column_stack([unit(truth[:, :2] + rng.normal(0, .03, (T, 2))) * np.cos(r5b.E_C), np.full(T, np.sin(r5b.E_C))])
    return nt, y, w, truth, init, rho


def lit_scene(rng, T=40, G=30, e_deg=45., theta_max_deg=28.):
    """所有法线始终处于受光锥内(θ<45°)、仰角为常数：用于命题2/3的退化族验证。"""
    th = np.deg2rad(rng.uniform(0, theta_max_deg, G)); al = rng.uniform(0, 2 * np.pi, G)
    nn = np.stack([np.sin(th) * np.cos(al), np.sin(th) * np.sin(al), np.cos(th)], 1)
    phi = np.cumsum(rng.uniform(.05, .3, T)); l = dir_from_ea(phi, np.full(T, np.deg2rad(e_deg)))
    rho = rng.uniform(.3, .8, (G, 3)); eps = rng.normal(0, .25, (3, 9)); eps[:, 0] += 1.
    nt = np.broadcast_to(nn, (T, G, 3)).copy(); return nt, l, rho, eps


def selftest(out):
    from scipy.optimize import nnls
    rng = np.random.default_rng(SEED); res = {}
    # S1 SH 基正交性
    P = fibonacci(100000); Y = sh9(P); gram = 4 * np.pi / len(P) * Y.T @ Y; res['S1_gram_max_dev'] = float(abs(gram - np.eye(9)).max()); assert res['S1_gram_max_dev'] < 1e-3, res
    # S2 辐照度公式：随机 L2 辐亮度 -> 暴力球面积分 vs 解析 A_l
    Q = fibonacci(1000000); YQ = sh9(Q); rel = []
    for _ in range(3):
        Lc = rng.normal(0, 1, 9); radiance = YQ @ Lc
        for nrm in unit(rng.normal(size=(4, 3))):
            brute = (radiance * np.maximum(Q @ nrm, 0)).sum() * 4 * np.pi / len(Q); ana = (A_HAT * Lc * sh9(nrm)).sum(); rel.append(abs(brute - ana) / (abs(Lc).max() * np.pi))
    res['S2_irradiance_rel_err_max'] = float(max(rel)); assert max(rel) < 1e-3, res
    # S3 rho 一维 NNLS 对照
    x = rng.random((60, 40)); y = rng.random((60, 40)); w = rng.random((60, 40))
    y[:, 0] = -.3 - .2 * x[:, 0]; y[:, 1] = 2 * x[:, 1] - .5; y[:, 2] = .2 + .5 * x[:, 2]; w[:, 4] = 0
    rh = rho_closed(torch.tensor(x)[..., None], torch.tensor(y)[..., None], torch.tensor(w))[:, 0].numpy(); err = 0.
    for g in range(40):
        got = nnls((x[:, g] * np.sqrt(w[:, g]))[:, None], y[:, g] * np.sqrt(w[:, g]))[0][0]; err = max(err, abs(got - rh[g]))
    res['S3_rho_vs_nnls_max_diff'] = float(err); assert err < 2e-5, err; assert rh[0] == 0 and rh[4] == 0, rh[:5]
    # S4 命题3：常数仰角 + 受光锥 + L≥1 环境 -> 竖直平移退化族 L_data 不变（damp=0 以排除1e-6稳定项）
    nt, l, rho0, eps = lit_scene(rng); T, G = nt.shape[:2]; ntt = torch.as_tensor(nt); Yb = sh9(ntt)
    yy = torch.as_tensor(np.maximum(rho0[None] * (sh9(nt) @ EPS_TRUE.T + np.einsum('tgc,tc->tg', nt, l)[..., None]) + rng.normal(0, .02, (T, G, 3)), 1e-3)); ww = torch.ones(T, G, dtype=torch.float64)
    base = float(env_terms(torch.as_tensor(l), torch.as_tensor(eps), Yb, ntt, yy, ww, damp=0.)[0]); diffs = []
    sin_e = np.sin(np.deg2rad(45.))
    for c in (.05, -.05, .1, -.1):
        k = np.sqrt(1 + 2 * c * sin_e + c * c); l2 = (l + np.array([0, 0, c])) / k; e2 = eps / k; e2[:, 2] = (eps[:, 2] * .488603 - c) / (k * .488603)
        diffs.append(abs(float(env_terms(torch.as_tensor(l2), torch.as_tensor(e2), Yb, ntt, yy, ww, damp=0.)[0]) - base))
    k_ = np.sqrt(1 + .2 * sin_e + .01); ctrl = abs(float(env_terms(torch.as_tensor((l + np.array([0, 0, .1])) / k_), torch.as_tensor(eps / k_), Yb, ntt, yy, ww, damp=0.)[0]) - base)   # 阴性对照：只平移l、不修正beta_z
    res['S4_prop3_data_diff_max'] = float(max(diffs)); res['S4_negative_control_diff'] = float(ctrl); assert max(diffs) < 1e-12 and ctrl > 1e-8, (diffs, ctrl)
    # S5 命题2：逐帧 s_t + L≥1 -> 任意 c 的三维精确退化
    s_t = rng.uniform(.8, 1.2, T); diffs = []; base = float(env_terms(torch.as_tensor(l), torch.as_tensor(eps), Yb, ntt, yy, ww, damp=0., s=torch.as_tensor(s_t))[0])
    for cvec in (np.array([.05, -.03, .07]), np.array([-.1, .08, .02]), np.array([.0, .0, -.1])):
        v = s_t[:, None] * l + cvec[None]; s2 = np.linalg.norm(v, axis=1); l2 = v / s2[:, None]; e2 = eps.copy()
        e2[:, 1] -= cvec[1] / .488603; e2[:, 2] -= cvec[2] / .488603; e2[:, 3] -= cvec[0] / .488603     # β=(x,y,z) 对应 k=3,1,2
        diffs.append(abs(float(env_terms(torch.as_tensor(l2), torch.as_tensor(e2), Yb, ntt, yy, ww, damp=0., s=torch.as_tensor(s2))[0]) - base))
    ctrl5 = abs(float(env_terms(torch.as_tensor(l2), torch.as_tensor(eps), Yb, ntt, yy, ww, damp=0., s=torch.as_tensor(s2))[0]) - base)       # 阴性对照：不修正beta
    res['S5_prop2_data_diff_max'] = float(max(diffs)); res['S5_negative_control_diff'] = float(ctrl5); assert max(diffs) < 1e-12 and ctrl5 > 1e-8, (diffs, ctrl5)
    # S6 恢复测试(含 T-C 路径与 Adam 回退路径、确定性重复)
    nt6, y6, w6, truth, init, rho_true = synth_scene(rng)
    end, info = fit_traj_env(nt6, y6, w6, init, 0., 1e-2, 1e-2, out, 'synthetic'); err6 = compare(end, truth)
    res['S6_synthetic'] = err6; res['S6_certificate'] = info['certificate']; res['S6_eps_max_abs_err'] = float(abs(info['eps'] - EPS_TRUE).max())
    assert err6['mean_deg'] < 1., err6; assert info['converged'], info['certificate']
    ez = np.rad2deg(np.arcsin(end[:, 2])); assert ez.min() > 40 and ez.max() < 50
    end2, _ = fit_traj_env(nt6, y6, w6, init, 0., 1e-2, 1e-2, out, 'synthetic_repeat'); res['S6_repeat_max_abs'] = float(abs(end - end2).max()); assert np.allclose(end, end2, atol=1e-8)
    tcend, itc = fit_traj_env(nt6, y6, w6, init, np.arctanh(.9), 1e-2, 1e-2, out, 'synthetic_tc', force_tc=True)
    res['S6_tc_certificate'] = itc['certificate']; res['S6_tc_vs_main_deg'] = compare(tcend, end); assert itc['converged'] and itc['certificate']['path'] == 'lbfgs+TC', itc['certificate']; assert compare(tcend, end)['mean_deg'] < .1
    fbend, ifb = fit_traj_env(nt6, y6, w6, init, 0., 1e-2, 1e-2, out, 'synthetic_fallback', force_fallback=True)
    res['S6_fallback_certificate'] = ifb['certificate']; res['S6_fallback_vs_main_deg'] = compare(fbend, end); assert ifb['certificate']['path'] == 'adam_fallback' and ifb['converged'], ifb['certificate']; assert compare(fbend, end)['mean_deg'] < .1
    # S7 梯度检查(内点、远离ReLU边界、rho>0)
    nt7, l7, rho7, eps7 = lit_scene(rng, T=5, G=8, theta_max_deg=15.); n7 = torch.as_tensor(nt7); Y7 = sh9(n7); ww7 = torch.ones(5, 8, dtype=torch.float64)
    y7 = torch.as_tensor(np.maximum(rho7[None] * (sh9(nt7) @ EPS_TRUE.T + np.einsum('tgc,tc->tg', nt7, l7)[..., None]) + rng.normal(0, .02, (5, 8, 3)), 1e-2))
    ph0 = torch.tensor(np.arctan2(l7[:, 1], l7[:, 0]) + .01, requires_grad=True); a7 = torch.tensor(rng.normal(0, .2, 5), requires_grad=True); e7 = torch.tensor(EPS_TRUE + .01, requires_grad=True)
    f7 = lambda p_, a_, e_: total_objective(p_, a_, e_, Y7, n7, y7, ww7, 1e-2, 1e-2, 1.)[0]
    assert torch.autograd.gradcheck(f7, (ph0, a7, e7), eps=1e-6, atol=1e-6, rtol=1e-5); res['S7_gradcheck'] = 'PASS'
    # S8 惩罚项与 r5b.penalties 的数值恒等
    pt = torch.tensor(rng.normal(size=20).cumsum() * .1); a8 = torch.tensor(rng.normal(size=20)); nn8 = torch.as_tensor(unit(rng.normal(size=(20, 3, 3))))
    tot8 = total_objective(pt, a8, torch.zeros(3, 9, dtype=torch.float64), sh9(nn8), nn8, torch.rand(20, 3, 3, dtype=torch.float64), torch.ones(20, 3, dtype=torch.float64), 1., 10., 1.)
    et = elev(a8); p5 = tot8[2]
    d_ = pt[1:] - pt[:-1]; de_ = et[1:] - et[:-1]
    ref = dict(a1=(d_ - d_.mean()).square().mean(), a2=(d_[1:] - d_[:-1]).square().mean(), e1=de_.square().mean(), e2=(de_[1:] - de_[:-1]).square().mean(), rev=torch.relu(-1. * d_).square().mean())
    res['S8_penalty_max_diff'] = float(max(abs(p5[k] - ref[k]) for k in ref)); assert res['S8_penalty_max_diff'] < 1e-15 and penalties is r5b.penalties
    # S9 K1 Schur 补与稠密求逆对照
    nt9, l9, rho9, eps9 = lit_scene(rng, T=6, G=10, theta_max_deg=70.); nt9 = unit(rng.normal(size=(6, 10, 3))); y9 = np.maximum(rho9[None] * (sh9(nt9) @ EPS_TRUE.T + np.maximum(np.einsum('tgc,tc->tg', nt9, l9), 0)[..., None]) + rng.normal(0, .02, (6, 10, 3)), 1e-3)
    ctx9 = make_ctx(nt9, y9, np.ones((6, 10))); F9 = fisher_env(ctx9, l9, eps9); ref9 = dense_fisher(ctx9, l9, eps9)
    res['S9_schur_rel_diff'] = max(float((F9[k] - ref9[k]).abs().max() / ref9[k].abs().max()) for k in 'ABC'); assert res['S9_schur_rel_diff'] < 1e-8, res['S9_schur_rel_diff']
    # S10 eps≡0 + luminance 退化路径与 b≡0 的 E4 目标一致
    nt10 = torch.as_tensor(nt9); yl = torch.as_tensor(rng.uniform(.05, .9, (6, 10, 1))); w10 = torch.as_tensor(rng.uniform(.5, 1., (6, 10)))
    l10 = torch.as_tensor(l9); d10, r10, _, _, x10 = env_terms(l10, torch.zeros(1, 9, dtype=torch.float64), sh9(nt10), nt10, yl, w10)
    rb = ((w10 * x10 * yl[..., 0]).sum(0) / ((w10 * x10.square()).sum(0) + DAMP)).clamp_min(0); ref10 = (w10 * .5 * (x10 * rb[None] - yl[..., 0]).square()).sum() / w10.sum()
    res['S10_diff'] = float(abs(d10 - ref10)); assert res['S10_diff'] < 1e-12
    # 诊断函数在合成数据上的试运行（保证K1–K7代码路径可执行，数值有限）
    res['diagnostics_dry_run'] = diag_dry_run(nt6, y6, w6, end, info, rng)
    res.update(status='PASS', gt_read=False, note='合成真值不是数据集GT'); write(out / 'numeric_tests.json', res); print('R6_ENV_SELF_TEST_PASS', json.dumps({k: v for k, v in res.items() if k.startswith('S') and not isinstance(v, (dict, list))}), flush=True)


def dense_fisher(ctx, l, eps):
    """S9 参照：用 autograd 全雅可比构造稠密 Fisher，再稠密求逆消去 rho。"""
    nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']; T, G, C = yt.shape
    lt = torch.as_tensor(l); eps = torch.as_tensor(eps, dtype=torch.float64); phi, e = dir_angles(l)
    _, rho, _, _, _ = env_terms(lt, eps, Yb, nt, yt, ww); v = (ww[..., None] / C / ww.sum()).expand(T, G, C).reshape(-1).sqrt()

    def r(th, rh, ep):
        ph, ee = th[:T], th[T:]; ll = compose(torch.stack([torch.cos(ph), torch.sin(ph)], 1), ee)
        x = torch.relu(torch.einsum('tgc,tc->tg', nt, ll)); A = Yb @ ep.T + x[..., None]
        return (v * (rh.reshape(G, C)[None] * A - yt).reshape(-1))
    th0 = torch.cat([torch.as_tensor(phi), torch.as_tensor(e)]); Jt, Jr, Je = torch.autograd.functional.jacobian(r, (th0, rho.reshape(-1).clone(), eps))
    J = torch.cat([Jt, Jr, Je.reshape(len(v), -1)], 1); Fm = J.T @ J; nu = 2 * T; nr = G * C; act = torch.nonzero(rho.reshape(-1) > 1e-12).squeeze(1)
    idx_u = torch.cat([torch.arange(nu), nu + nr + torch.arange(C * 9)]); idx_r = nu + act
    Fuu = Fm[idx_u][:, idx_u]; Fur = Fm[idx_u][:, idx_r]; Frr = Fm[idx_r][:, idx_r]
    S = Fuu - Fur @ torch.linalg.inv(Frr) @ Fur.T
    return dict(A=S[:nu, :nu], B=S[nu:, :nu], C=S[nu:, nu:])


def diag_dry_run(n, y, w, end, info, rng):
    ctx = make_ctx(n, y, w); lum = (y * KAPPA).sum(-1); l = end; eps = info['eps']; T = len(l); phi, e = dir_angles(l); s = orbit_sign(l)
    fk = fisher_report(ctx, l, eps, None, phi, e, s, lum, w, l, info['final_data'], info['final_data'])
    en = energy_share(ctx, l, eps, lum, w, l); ir, pts, E = irradiance_check(ctx, eps)
    pr = profile_zmode(ctx, l, eps, lum, w, l, deltas=(-2, 0, 2), als_iter=40); ch = channel_residuals(ctx, l, eps, lum, w, l)
    ho = heldout(n[:, :100], y[:, :100], lum[:, :100], w[:, :100], eps, {'R6': l})
    allv = [fk['F1_value'], en['r6']['all']['eta_lin'][0], ir['obs_negative_weighted_fraction'], pr['a_R6_als'][0], ch['r6_luminance_projected_rms'], ho['new_style_fixed_eps_mae']['R6']['channel_mae'][0]]
    assert np.isfinite(allv).all(), allv
    return dict(F1_value=fk['F1_value'], eta_var=en['r6']['all']['eta_var'], curvature_ratio=pr['curvature_ratio_a_over_b'], ok=True)


# ================================================================ 图
def figures(out, ctx, results, ends, infos, fi, d, e4_dir, eps, l):
    nt, Yb, yt, ww = ctx['nt'], ctx['Yb'], ctx['yt'], ctx['ww']; T = yt.shape[0]
    _, rho, res, A, x = env_terms(torch.as_tensor(l), torch.as_tensor(eps), Yb, nt, yt, ww); res = res.numpy(); wn = ww.numpy()
    uv = d['projection_uv'][:, fi] if 'projection_uv' in d.files else None
    fig, axs = plt.subplots(3, 5, figsize=(22, 11))
    for j, t in enumerate([0, 37, 74, 111, 149]):
        ok = wn[t] > 0; pix = (uv[t] + 1) * 200 - .5 if uv is not None else d['xyz'][fi, :2]
        for c in range(3):
            sc = axs[c, j].scatter(pix[ok, 0], pix[ok, 1], s=3, c=res[t, ok, c], cmap='coolwarm', vmin=-.1, vmax=.1); axs[c, j].set_title(f'Frame {t + 1} ch {"RGB"[c]} residual'); axs[c, j].set_aspect('equal'); axs[c, j].invert_yaxis()
        fig.colorbar(sc, ax=axs[:, j], shrink=.6)
    fig.savefig(out / '01_residual_rgb.png', dpi=130); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8)); fig.suptitle('E3 / E4(frozen) / R6 and A/B splits')
    for k, ll in [('E3', results['E3']), ('E4', e4_dir), ('R6', ends['full'])]:
        axs[0, 0].plot(np.rad2deg(np.unwrap(np.arctan2(ll[:, 1], ll[:, 0]))), label=k); axs[0, 1].plot(elev_deg_(ll), label=k)
    for k in ['A', 'B']:
        axs[1, 0].plot(np.rad2deg(np.unwrap(np.arctan2(ends[k][:, 1], ends[k][:, 0]))), label=f'R6 {k}'); axs[1, 1].plot(elev_deg_(ends[k]), label=f'R6 {k}')
    for ax in axs[:, 1]: ax.axhline(40, c='k', ls=':'); ax.axhline(50, c='k', ls=':')
    axs[0, 1].set_ylim(35, 75)
    for ax in axs.flat: ax.legend(); ax.set_xlabel('Frame index')
    axs[0, 0].set_ylabel('Unwrapped azimuth (deg)'); axs[0, 1].set_ylabel('Elevation (deg)'); fig.tight_layout(); fig.savefig(out / '02_direction.png', dpi=140); plt.close(fig)
    ident = json.loads((out / 'diagnostics/fisher.json').read_text()); prof = json.loads((out / 'diagnostics/profile.json').read_text())
    fig, axs = plt.subplots(1, 2, figsize=(15, 5)); names = list(ident['modes']); xs = np.arange(len(names))
    axs[0].bar(xs - .2, [ident['modes'][k]['R_new_with_prior'] for k in names], .4, label='R6 (rho,eps eliminated)'); axs[0].bar(xs + .2, [ident['modes'][k]['R_E4_with_prior'] for k in names], .4, label='E4 (rho,b eliminated)')
    axs[0].set_xticks(xs); axs[0].set_xticklabels(names, rotation=70, fontsize=7); axs[0].axhline(.1, c='k', ls='--'); axs[0].legend(); axs[0].set_ylabel('information retention R(u)')
    for k, lab in [('a_R6_als', 'R6 ALS'), ('b_R6_eps_fixed', 'R6 eps fixed'), ('c_E4_kkt', 'E4 kkt')]: axs[1].plot(prof['delta_deg'], prof['curvature'][k]['delta_data'], 'o-', label=lab)
    axs[1].set_xlabel('elevation shift (deg)'); axs[1].set_ylabel('Δ L_data'); axs[1].legend(); axs[1].set_yscale('symlog', linthresh=1e-8)
    fig.tight_layout(); fig.savefig(out / '03_identifiability.png', dpi=140); plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for k in ['full', 'hi', 'lo'] + [f'boot{s}' for s in BOOT]:
        axs[0, 0].plot(elev_deg_(ends[k]), label=k); axs[0, 1].plot(np.rad2deg(np.unwrap(np.arctan2(ends[k][:, 1], ends[k][:, 0]))), label=k)
    for k in ['full', 'hi', 'lo']:
        h = json.loads((out / 'runs' / f'R6_{k}' / 'history.json').read_text()); axs[1, 0].plot([q['step'] for q in h], [q['loss'] for q in h], label=k)
    h = json.loads((out / 'runs' / 'R6_full' / 'history.json').read_text()); E = np.array([q['eps'] for q in h]); axs[1, 1].plot([q['step'] for q in h], E, lw=.8); axs[1, 1].set_title('eps trajectory (full, T-A)')
    axs[0, 0].axhline(40, c='k', ls=':'); axs[0, 0].axhline(50, c='k', ls=':'); axs[1, 0].set_yscale('log'); axs[1, 0].set_ylabel('loss')
    for ax in axs.flat[:3]: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / '04_hilo_boot_convergence.png', dpi=140); plt.close(fig)
    sph = np.load(out / 'diagnostics/irradiance_sphere.npz'); pts, Es = sph['points'], sph['E']
    lon = np.arctan2(pts[:, 1], pts[:, 0]); lat = np.arcsin(pts[:, 2]); fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    for c in range(3):
        sc = axs[c].scatter(np.rad2deg(lon), np.rad2deg(lat), c=Es[:, c], s=6, cmap='viridis'); fig.colorbar(sc, ax=axs[c])
        neg = Es[:, c] < 0; axs[c].scatter(np.rad2deg(lon[neg]), np.rad2deg(lat[neg]), s=10, c='none', edgecolors='r', linewidths=.4); axs[c].set_title(f'E_{"RGB"[c]}(n) (red ring: E<0)'); axs[c].set_xlabel('azimuth (deg)'); axs[c].set_ylabel('elevation (deg)')
    fig.tight_layout(); fig.savefig(out / '05_irradiance_map.png', dpi=140); plt.close(fig)


# ================================================================ 主流程
def support_and_obs(a):
    d = np.load(a.observations); n = d['normals'].astype(np.float64); y = d['intensity'].astype(np.float64); w = d['weights'].astype(np.float64)
    rgb = d['rgb'].astype(np.float64); xyz = d['xyz']; ids = d['gaussian_ids']; frames = d['frames']
    assert np.array_equal(frames, np.arange(1, 151)) and y.shape == w.shape == (150, len(ids)) and rgb.shape == (150, len(ids), 3)
    # I1：RGB 线性化与 luminance 一致性
    assert np.array_equal(np.isfinite(y), np.isfinite(rgb).all(-1)), 'intensity 与 rgb 有限性掩码不一致'
    lin = srgb_to_linear(rgb); m = np.isfinite(y); lum_err = float(abs((lin * KAPPA).sum(-1)[m] - y[m]).max()); assert lum_err <= 1e-5, lum_err
    nerr = float(abs(np.linalg.norm(n[m], axis=-1) - 1).max()); assert nerr <= 1e-4, nerr
    lin[~m] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning); q20 = np.nanquantile(y, .2, axis=0); q95 = np.nanquantile(y, .95, axis=0)
    eligible = (np.isfinite(y).sum(0) >= 8) & (q95 - q20 > .015)
    mn = n.mean(0); body = (xyz[:, 2] >= .05) & (abs(mn[:, 2]) <= .85)
    cells = np.floor(xyz * 20).astype(np.int64); hval = cells[:, 0] * 73856093 ^ cells[:, 1] * 19349663 ^ cells[:, 2] * 83492791
    held = hval % 5 == 0; fi = np.flatnonzero(eligible & body & ~held); vi = np.flatnonzero(eligible & body & held); allvi = np.flatnonzero(eligible & held)
    A = fi[(hval[fi] // 5) % 2 == 0]; B = fi[(hval[fi] // 5) % 2 != 0]
    assert (len(fi), len(vi), len(A), len(B)) == (2683, 648, 1450, 1233), (len(fi), len(vi), len(A), len(B))
    relative = np.clip((y - q20) / (q95 - q20 + .001), 0, 1.5); relative[:, ~eligible] = np.nan
    return d, n, y, w, lin, xyz, ids, frames, relative, fi, vi, allvi, A, B, hval, body, dict(luminance_max_abs_err=lum_err, normal_norm_max_err=nerr)


def check_inputs(a, e4):
    """I1/I2：输入与依赖代码哈希。"""
    assert sha(a.observations) == OBS_HASH
    man = json.loads((e4 / 'freeze_manifest.json').read_text()); rec = {}
    for p in (r3.__file__, r5b.__file__):
        rel = str(Path(p).resolve().relative_to(Path.cwd().resolve())); assert man['files'][rel] == sha(p), f'依赖代码与冻结不一致：{rel}'; rec[rel] = sha(p)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); p.add_argument('--observations', type=Path, default=OBS)
    p.add_argument('--e4-root', type=Path, default=E4_ROOT); p.add_argument('--plan', type=Path, default=PLAN_DEFAULT); p.add_argument('--self-test', action='store_true')
    p.add_argument('--smoke', action='store_true'); p.add_argument('--e4-regress-only', action='store_true', help='仅做支撑集/E1-E3/E4回归检查，不拟合R6'); p.add_argument('--k8', action='store_true')
    p.add_argument('--eps-lr', type=float, default=EPS_LR); a = p.parse_args()
    out = a.output; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED)
    if a.self_test: selftest(out); return
    e4 = a.e4_root; deps = check_inputs(a, e4)
    write(out / 'launch.json', dict(pid=os.getpid(), argv=sys.argv, plan_sha256=sha(a.plan), code_sha256=sha(__file__), dependency_sha256=deps, input_sha256=sha(a.observations), eps_lr=a.eps_lr, gt_read=False))
    (d, n, y, w, lin, xyz, ids, frames, relative, fi, vi, allvi, A, B, hval, body, qc) = support_and_obs(a)
    # I4：支撑集与冻结值逐元素相等
    np.savez_compressed(out / 'support.npz', body_fit=fi, body_val=vi, all_val=allvi, split_A=A, split_B=B, gaussian_ids=ids, hval=hval, body_mask=body)
    with np.load(e4 / 'fitting/support.npz') as fz:
        for k, v in dict(body_fit=fi, body_val=vi, all_val=allvi, split_A=A, split_B=B, hval=hval, body_mask=body, gaussian_ids=ids).items(): assert np.array_equal(fz[k], v), f'支撑集与冻结不一致：{k}'
    write(out / 'support_manifest.json', dict(support_sha256=sha(out / 'support.npz'), body_fit=2683, body_val=648, split_A=1450, split_B=1233, equals_frozen=True, linearization_qc=qc))
    scoredata = (sub(n, vi), y[:, vi], w[:, vi]); results = {'E0': np.tile([0., 0., 1.], (150, 1))}; voteinfo = {}
    for k in ['E1', 'E2']: results[k], voteinfo[k] = initial(sub(n, fi), y[:, fi], w[:, fi], relative[:, fi], k, scoredata)
    results['E3'], _, _ = r3.fit(sub(n, fi), y[:, fi], w[:, fi], results['E2'], False, .03, 0., out, 'E3', False)
    with np.load(R3_PRED) as old: repro = dict(E1=float(abs(results['E1'] - old['E1']).max()), E2=float(abs(results['E2'] - old['E2']).max()), E3=float(abs(results['E3'] - old['E3']).max()))
    write(out / 'r3_reproduction.json', dict(max_abs_diff=repro, tolerance=1e-6, source=str(R3_PRED), gt_read=False))
    if max(repro.values()) > 1e-6: raise RuntimeError(f'第三轮E1–E3未复现：{repro}')
    # I7：E4 回归（调用 r5b.fit_traj，λ=(1,10)）；超差则停止并上报（D12）
    e4dir_frozen = np.load(e4 / 'fitting/runs/a1e+00_e1e+01_full/direction.npy'); e4pred = np.load(e4 / 'fitting/predictions.npz')['E4']
    assert np.array_equal(e4dir_frozen, e4pred) or float(ang_deg(e4dir_frozen, e4pred).max()) < 1e-9
    t0 = time.time(); reg, reginfo = r5b.fit_traj(sub(n, fi), y[:, fi], w[:, fi], results['E3'], a_from_z(results['E3'][:, 2].mean()), 1., 10., out, 'E4regress_full')
    dd = float(ang_deg(reg, e4dir_frozen).max())
    write(out / 'e4_regression.json', dict(max_angle_diff_deg=dd, tolerance_deg=1e-6, ok=bool(dd <= 1e-6), seconds=time.time() - t0, certificate=reginfo['certificate'], device='cuda' if torch.cuda.is_available() else 'cpu', gt_read=False))
    print('E4_REGRESSION_MAX_DEG', dd, flush=True)
    if dd > 1e-6: raise RuntimeError(f'E4回归超差：{dd}°>1e-6°（规划D12：停止并上报）')
    if a.e4_regress_only: print('R6_E4_REGRESS_ONLY_OK', flush=True); return
    for sd in BOOT:
        pick = np.random.default_rng(sd).choice(fi, len(fi), replace=True); assert np.array_equal(pick, np.load(e4 / f'fitting/bootstrap_{sd}_sample.npy')); np.save(out / f'bootstrap_{sd}_sample.npy', pick)
    full_init = results['E3']; a_full = a_from_z(full_init[:, 2].mean())
    if a.smoke:
        t0 = time.time(); _, inf = fit_traj_env(sub(n, fi), sub(lin, fi), w[:, fi], full_init, a_full, LAM_A, LAM_E, out, 'R6_full', eps_lr=a.eps_lr)
        write(out / 'smoke_result.json', dict(certificate=inf['certificate'], converged=inf['converged'], final_loss=inf['final_loss'], als_init=inf['als_init'], eps_lr=a.eps_lr, seconds=time.time() - t0, gt_read=False)); print('R6_SMOKE', inf['converged'], flush=True); return
    supports = {'full': (fi, full_init)}
    for key, idx in [('A', A), ('B', B)]:
        init, voteinfo[key] = initial(sub(n, idx), y[:, idx], w[:, idx], relative[:, idx], 'E2'); supports[key] = (idx, r3.fit(sub(n, idx), y[:, idx], w[:, idx], init, False, .03, 0., out, f'stage1_{key}', False)[0])
    for sd in BOOT:
        pick = np.load(out / f'bootstrap_{sd}_sample.npy'); init, voteinfo[f'boot{sd}'] = initial(sub(n, pick), y[:, pick], w[:, pick], relative[:, pick], 'E2')
        supports[f'boot{sd}'] = (pick, r3.fit(sub(n, pick), y[:, pick], w[:, pick], init, False, .03, 0., out, f'stage1_boot{sd}', False)[0])
    ends = {}; infos = {}
    for key, (idx, s1) in supports.items(): ends[key], infos[key] = fit_traj_env(sub(n, idx), sub(lin, idx), w[:, idx], s1, a_from_z(s1[:, 2].mean()), LAM_A, LAM_E, out, f'R6_{key}', eps_lr=a.eps_lr)
    for key, a0 in [('hi', np.arctanh(.9)), ('lo', -np.arctanh(.9))]: ends[key], infos[key] = fit_traj_env(sub(n, fi), sub(lin, fi), w[:, fi], full_init, a0, LAM_A, LAM_E, out, f'R6_{key}', eps_lr=a.eps_lr)
    results['R6'] = ends['full'].copy(); fit_info = infos['full']
    for k, v in results.items(): assert v.shape == (150, 3) and np.isfinite(v).all() and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-10), k
    np.savez_compressed(out / 'predictions.npz', frames=frames, **results)
    np.savez_compressed(out / 'env.npz', eps=fit_info['eps'], basis=BASIS_NAME, gauge='s=1', rho=fit_info['rho'], body_fit=fi)
    # ---- 诊断 K1–K7(K8)
    dg = out / 'diagnostics'; dg.mkdir(); ctx = make_ctx(sub(n, fi), sub(lin, fi), w[:, fi]); lum_fit = y[:, fi]; w_fit = w[:, fi]; e4_dir = e4pred; l = ends['full']; eps = fit_info['eps']
    phi, e = dir_angles(l); s_or = orbit_sign(l); e4_data = None
    xe = torch.relu(torch.einsum('tgc,tc->tg', ctx['nt'], torch.as_tensor(e4_dir))); yl_ = torch.as_tensor(np.nan_to_num(lum_fit)); wl_ = torch.as_tensor(np.where(np.isfinite(lum_fit), w_fit, 0)); r4, b4 = kkt(xe, yl_, wl_)
    e4_data = float((wl_ * .5 * (r4[None] * xe + b4[None] - yl_).square()).sum() / wl_.sum())
    write(dg / 'fisher.json', fisher_report(ctx, l, eps, None, phi, e, s_or, lum_fit, w_fit, e4_dir, fit_info['final_data'], e4_data))
    write(dg / 'profile.json', profile_zmode(ctx, l, eps, lum_fit, w_fit, e4_dir)); write(dg / 'energy.json', energy_share(ctx, l, eps, lum_fit, w_fit, e4_dir))
    irr, pts, Es = irradiance_check(ctx, eps); write(dg / 'irradiance.json', irr); np.savez_compressed(dg / 'irradiance_sphere.npz', points=pts, E=Es)
    write(dg / 'residual_channels.json', channel_residuals(ctx, l, eps, lum_fit, w_fit, e4_dir))
    cand = json.loads((e4 / 'fitting/candidate_table.json').read_text())['candidates']; e4row = [r for r in cand if r['name'] == 'a1e+00_e1e+01'][0]
    write(dg / 'repeatability.json', repeatability(ends, infos, e4row))
    write(dg / 'heldout.json', heldout(sub(n, vi), sub(lin, vi), y[:, vi], w[:, vi], eps, {'R6': l, 'E3': results['E3'], 'E4': e4_dir}))
    if a.k8:
        ply = Path('output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp/point_cloud/iteration_35000/point_cloud.ply')
        assert sha(ply) == 'adf94a9370f2251532d8885ef3f3e5a188ad2cdb6daa41539cb8d227d39222ff'; write(dg / 'stage1_chroma.json', stage1_chroma(fi, fit_info['rho'], ply, ids))
    flags = dict(F1=json.loads((dg / 'fisher.json').read_text())['flag_F1'], F2=json.loads((dg / 'energy.json').read_text())['flag_F2'], F3=irr['flag_F3'], F4=json.loads((dg / 'residual_channels.json').read_text())['flag_F4'])
    config = dict(final_method='R6', model='R6-ENV', basis=BASIS_NAME, gauge='s=1', eps_params=27, omega='1/3 per channel', lam=dict(a=LAM_A, e=LAM_E, rev=LAM_REV), coef_reg=COEF_REG, eps_lr=a.eps_lr,
                  elevation_center_deg=45., elevation_halfwidth_deg=5., plan_sha256=sha(a.plan), code_sha256=sha(__file__), input_sha256=sha(a.observations), dependency_sha256=deps, flags=flags,
                  full_certificate=fit_info['certificate'], full_converged=fit_info['converged'], als_init_full=fit_info['als_init'], e4_regression_max_deg=dd, r3_reproduction=repro, linearization_qc=qc, votes=voteinfo,
                  trajectory_stats=traj_stats(l, s_or), gt_read=False, coordinate_system='world', direction_convention='surface_to_light',
                  information_limitations='稳定性/平滑/留出残差均为无GT经验量，不证明GT精度；E4的G0-G7与J选档在本轮仅作诊断')
    write(out / 'method_config.json', config); figures(out, ctx, results, ends, infos, fi, d, e4_dir, eps, l)
    write(out / 'fit_complete.json', dict(status='R6_FIT_COMPLETE_AWAITING_REVIEW', full_converged=fit_info['converged'], flags=flags, pid=os.getpid(), gt_read=False))
    print('R6_FIT_COMPLETE', fit_info['converged'], flags, flush=True)


if __name__ == '__main__':
    main()
