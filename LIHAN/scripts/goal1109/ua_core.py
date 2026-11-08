"""1109-01 T01 全局环境光单标量 Stage 2 核心数值模块（不读取 GT）。

观测模型：y_hat_tgc = rho_gc * (amb + x_tg)，x_tg = max(0, n_tg . l_t)，amb >= 0 全局单标量，rho in [0,1]，方向光强固定为 1，V = 1。
估计：rho 逐 (g,c) 闭式一维 WLS 并截断到 [0,1]；amb 为剖面目标 F(amb) 的一维确定性全局搜索（包络定理：amb 取 detach）；
梯度变量只有逐帧 phi、nu。规划：LIHAN/DOC/规划/1109-01-T01全局环境光单标量Stage2规划.md

注意：LIHAN 是软链接，本模块不用 Path(__file__).resolve().parents 推导仓库根，统一使用相对 cwd 的路径。
变量命名：仰角内部参数 nu，环境光标量 amb；本模块禁止使用名为 a 的变量，以免与 r5b 的 a 混淆。
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r5b as r5b
from LIHAN.scripts.static_rgb_light_fit_r3 import sha, write, unit, sub
from LIHAN.scripts.static_rgb_light_fit_r5b import elev, compose, penalties, orbit_sign, a_from_z, ang_deg, elev_deg
from LIHAN.scripts.static_rgb_light_fit_r6_env import srgb_to_linear
from LIHAN.scripts import static_rgb_light_fit_goal1108_02_t01 as g1108

# ------------------------------------------------------------------ 常量与路径（全部相对 cwd）
PLAN = Path('LIHAN/DOC/规划/1109-01-T01全局环境光单标量Stage2规划.md')
PLAN_SHA = 'eb12c0e4ffe2d4ab4e5d681e5c26588b188d6cf770136d5e738936428b66f405'
GOAL_DOC = Path('LIHAN/DOC/GOAL/1109-01-Stage2全局环境光建模.md')
SCRIPT_DIR = Path('LIHAN/scripts/goal1109')
BASE = g1108.BASE                                      # 1031-12 fitting（只读）
E4_RUN = BASE / 'runs/a1e+00_e1e+01_full'
C01_RUN_DIR = Path('output/1108-05-LH-static-ELDL-V3-T01去系数与光向平滑')
R6_RUN_DIR = Path('output/1108-03-LH-static-ELDL-V3-T01环境光RGB新Stage2')
MODEL = Path('output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp')
SOURCE = Path('data/LH-data/static/LH-static-jumpingjacks-ELDL-V3')
OBS_KEYS = ('frames', 'gaussian_ids', 'xyz', 'normals', 'intensity', 'weights', 'rgb')   # 加载白名单

E_C, E_MAX, LAM_REV = r5b.E_C, r5b.E_MAX, r5b.LAM_REV
STEPS, LR_PHI, LR_NU, DROPS, BOOT = g1108.STEPS, g1108.LR_PHI, g1108.LR_A, g1108.DROPS, g1108.BOOT
OUTER, OUTER_EXT, LB_ITER, WIN = g1108.OUTER, g1108.OUTER_EXT, g1108.LB_ITER, g1108.WIN
TC_STEPS, TC_LR, TC_OUTER = g1108.TC_STEPS, g1108.TC_LR, g1108.TC_OUTER
DAMP = 1e-6                                            # rho 分母稳定项（沿用 r3.kkt）
OMEGA = 1.0 / 3.0                                      # 通道等权
KAPPA = np.array([.2126, .7152, .0722])
SEED = 103108
GRID_N = 400                                           # 网格 401 点
A_MAX0, EXPAND_MAX, GOLD_TOL = 4.0, 3, 1e-10
AINIT_STEPS = 300
SEARCH_CALLS = [0]                                     # search_amb 调用计数（S8 用）

ARM_FLOWS = {
    'A0': ['full', 'A', 'B', 'boot103108', 'boot103109', 'hi', 'lo', 'ainit_lo', 'ainit_hi', 'repeat_full'],
    'A1': ['full', 'A', 'B', 'boot103108', 'boot103109', 'hi', 'lo', 'repeat_full'],
}


def check_root():
    """入口前置断言：cwd 必须是仓库根（下面两个文件同时存在）。"""
    if not (Path('scene/gaussian_model.py').is_file() and PLAN.is_file()):
        raise RuntimeError(f'cwd 不是仓库根：{Path.cwd()}')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ------------------------------------------------------------------ 数据加载
def load_obs_rgb(path=None):
    """白名单键读取 observations.npz；y_lin = 逆 sRGB(rgb)；断言 kappa.y_lin == intensity 且有效掩码一致。"""
    path = Path(path) if path is not None else r3.OBS
    with np.load(path, allow_pickle=False) as d:
        raw = {k: d[k] for k in OBS_KEYS}
    n = raw['normals'].astype(np.float64)
    rgb = raw['rgb'].astype(np.float64)
    inten = raw['intensity'].astype(np.float64)
    w = raw['weights'].astype(np.float64)
    with np.errstate(invalid='ignore'):
        y_lin = srgb_to_linear(rgb)
    finite_rgb = np.isfinite(rgb).all(-1)
    if not np.array_equal(finite_rgb, np.isfinite(inten)):
        raise AssertionError('intensity 与 rgb 的有效掩码不一致')
    err = float(np.abs(y_lin[finite_rgb] @ KAPPA - inten[finite_rgb]).max())
    if not err <= 1e-5:
        raise AssertionError(f'kappa.y_lin 与 intensity 不一致：{err}')
    if ((w > 0) & ~finite_rgb).any():
        raise AssertionError('存在 w>0 但观测非有限')
    return dict(frames=raw['frames'], gaussian_ids=raw['gaussian_ids'], xyz=raw['xyz'], n=n, rgb=rgb, y=y_lin, intensity=inten, w=w,
                kappa_consistency_max_abs=err)


def build_support(n, inten, w, xyz, ids, frames, strict=True):
    """与 goal1108_02_t01.main 的支撑集代码逐字等价；strict 时断言与 1031-12 support.npz / bootstrap 文件逐元素相等。"""
    import warnings
    assert np.array_equal(frames, np.arange(1, 151)) and inten.shape == w.shape == (150, len(ids))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        q20 = np.nanquantile(inten, .2, axis=0)
        q95 = np.nanquantile(inten, .95, axis=0)
    eligible = (np.isfinite(inten).sum(0) >= 8) & (q95 - q20 > .015)
    mn = n.mean(0) if n.ndim == 3 else n
    body = (xyz[:, 2] >= .05) & (abs(mn[:, 2]) <= .85)
    cells = np.floor(xyz * 20).astype(np.int64)
    hval = cells[:, 0] * 73856093 ^ cells[:, 1] * 19349663 ^ cells[:, 2] * 83492791
    held = hval % 5 == 0
    fi = np.flatnonzero(eligible & body & ~held)
    vi = np.flatnonzero(eligible & body & held)
    allvi = np.flatnonzero(eligible & held)
    A = fi[(hval[fi] // 5) % 2 == 0]
    B = fi[(hval[fi] // 5) % 2 != 0]
    support = dict(body_fit=fi, body_val=vi, all_val=allvi, split_A=A, split_B=B, gaussian_ids=ids, hval=hval, body_mask=body)
    picks = {}
    for seed in BOOT:
        picks[seed] = np.random.default_rng(seed).choice(fi, len(fi), replace=True)
    if strict:
        assert (len(fi), len(vi), len(A), len(B)) == (2683, 648, 1450, 1233), (len(fi), len(vi), len(A), len(B))
        with np.load(BASE / 'support.npz', allow_pickle=False) as old:
            for k, v in support.items():
                assert np.array_equal(v, old[k]), k
        for seed in BOOT:
            assert np.array_equal(picks[seed], np.load(BASE / f'bootstrap_{seed}_sample.npy')), seed
    supports = {'full': fi, 'A': A, 'B': B, **{f'boot{s}': picks[s] for s in BOOT}}
    return support, supports, picks


def load_initial(supports, strict=True):
    """历史 E3 / stage1 初值（1108-02 D3 口径，白名单哈希在 verify_inputs 中校验）。"""
    mapping = {'full': 'E3', 'A': 'stage1_A', 'B': 'stage1_B', 'boot103108': 'stage1_boot103108', 'boot103109': 'stage1_boot103109'}
    initial = {}
    for key, dirname in mapping.items():
        l = np.load(BASE / 'runs' / dirname / 'direction.npy')
        conf = json.loads((BASE / 'runs' / dirname / 'config.json').read_text())
        assert l.shape == (150, 3) and np.isfinite(l).all() and np.max(abs(np.linalg.norm(l, axis=1) - 1)) <= 1e-10
        if strict:
            assert conf['support_count'] == len(supports[key]) and conf['stage2'] is False and conf['geo_smooth'] == 0 and conf['balanced'] is False \
                and conf['steps'] == 250 and conf['delta'] == .03 and conf['dtype'] == 'float64'
        initial[key] = l
    with np.load(BASE / 'predictions.npz', allow_pickle=False) as old:
        reference = old['E4'].copy()
        historical = {k: old[k].copy() for k in ['E1', 'E2', 'E3']}
    return initial, reference, historical


def make_ctx(n, y, w, device=None):
    """n:(T,G,3)；y:(T,G,3) 线性 RGB（无效为 NaN）；w:(T,G)。"""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    valid = np.isfinite(y).all(-1) if y.ndim == 3 else np.isfinite(y)
    ww = torch.as_tensor(np.where(valid, w, 0), dtype=torch.float64, device=device)
    return dict(nt=torch.as_tensor(n, dtype=torch.float64, device=device), yt=torch.as_tensor(np.nan_to_num(y), dtype=torch.float64, device=device),
                ww=ww, W=ww.sum(), device=device)


# ------------------------------------------------------------------ 观测模型
def lightdir(phi, nu):
    e = elev(nu)
    return compose(torch.stack([torch.cos(phi), torch.sin(phi)], 1), e), e


def shading(nt, l):
    """x = max(0, n.l)；nt:(T,G,3) 逐帧法线。"""
    return torch.relu(torch.einsum('tgc,tc->tg', nt, l) if nt.ndim == 3 else l @ nt.T)


def rho_box(A, y, w, upper=1.0):
    """rho_gc = clip( sum_t w A y_c / (sum_t w A^2 + 1e-6), 0, upper )。A:(T,G)，y:(T,G,C)，w:(T,G)；返回 (G,C)，autograd 穿过闭式解。"""
    wa = w * A
    num = (wa.unsqueeze(-1) * y).sum(0)
    den = (wa * A).sum(0).unsqueeze(-1) + DAMP
    r = num / den
    return r.clamp(0.0, upper) if upper is not None else r.clamp_min(0.0)


def data_loss(A, rho, y, w):
    """L_data = (1/W) sum_tg w sum_c omega_c (1/2) r^2，omega_c=1/3。"""
    res = rho.unsqueeze(0) * A.unsqueeze(-1) - y
    return (w.unsqueeze(-1) * res.square()).sum() * (0.5 * OMEGA) / w.sum()


def suff_stats(x, y, w):
    """充分统计量（no_grad 使用）：每高斯 S1,Sx,Sxx；每 (g,c) Sy,Sxy,Syy。"""
    wy = w.unsqueeze(-1) * y
    return dict(S1=w.sum(0), Sx=(w * x).sum(0), Sxx=(w * x * x).sum(0), Sy=wy.sum(0), Sxy=(wy * x.unsqueeze(-1)).sum(0), Syy=(wy * y).sum(0))


def profile_F(amb, S, upper=1.0, chunk=256):
    """剖面目标 F(amb)=min_rho L_data；amb 可为标量或向量；返回形状 (K,) 的张量。归一化 1/(C*W)，C 为 S 的通道数。"""
    dev = S['S1'].device
    ambv = torch.as_tensor(amb, dtype=torch.float64, device=dev).reshape(-1)
    C = S['Sy'].shape[-1]
    W = S['S1'].sum()
    out = []
    for i in range(0, len(ambv), chunk):
        ak = ambv[i:i + chunk]
        D = S['Sxx'][None] + 2 * ak[:, None] * S['Sx'][None] + (ak * ak)[:, None] * S['S1'][None]             # (k,G)
        N = S['Sxy'][None] + ak[:, None, None] * S['Sy'][None]                                                  # (k,G,C)
        r = N / (D.unsqueeze(-1) + DAMP)
        r = r.clamp(0.0, upper) if upper is not None else r.clamp_min(0.0)
        out.append((0.5 * (S['Syy'][None] - 2 * r * N + r * r * D.unsqueeze(-1))).sum((1, 2)) / (C * W))
    return torch.cat(out)


def _F1(amb, S, upper):
    return float(profile_F(amb, S, upper)[0])


def search_amb(S, upper=1.0):
    """规划 3.4：确定性一维全局搜索（401 点网格 → 必要时扩张 → 黄金分割）。返回 dict(amb, i_star, a_max, expansions, flags, F)。"""
    SEARCH_CALLS[0] += 1
    dev = S['S1'].device
    a_max, expansions, flags = A_MAX0, 0, []
    while True:
        h = a_max / GRID_N
        grid = torch.arange(GRID_N + 1, dtype=torch.float64, device=dev) * h
        Fv = profile_F(grid, S, upper)
        i = int(torch.argmin(Fv))                       # 并列取最小下标 = 最小 amb
        if i < GRID_N:
            break
        if expansions < EXPAND_MAX:
            a_max *= 2
            expansions += 1
            continue
        flags.append('A_AT_SEARCH_LIMIT')
        return dict(amb=float(grid[GRID_N]), i_star=i, a_max=a_max, expansions=expansions, flags=flags, F=float(Fv[GRID_N]))
    lo = float(grid[i - 1]) if i > 0 else 0.0
    lo = max(0.0, lo)
    hi = float(grid[i + 1])
    gr = (5 ** .5 - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = _F1(x1, S, upper), _F1(x2, S, upper)
    while hi - lo >= GOLD_TOL:
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = _F1(x1, S, upper)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = _F1(x2, S, upper)
    at = 0.5 * (lo + hi)
    fa, fi_ = _F1(at, S, upper), float(Fv[i])
    ai = float(grid[i])
    if fa < fi_:
        best, fb = at, fa
    elif fa > fi_:
        best, fb = ai, fi_
    else:
        best, fb = min(at, ai), fa
    return dict(amb=float(best), i_star=i, a_max=a_max, expansions=expansions, flags=flags, F=fb)


def objective_ua(phi, nu, ctx, arm, s, amb_override=None):
    """UA 目标：返回 (total, data, pen, e, amb, rho)。A1：amb 恒为 0 且不调用搜索；A0：amb=search_amb(...)（detach），或 amb_override。"""
    l, e = lightdir(phi, nu)
    x = shading(ctx['nt'], l)
    dev = x.device
    if arm == 'A1':
        amb = torch.zeros((), dtype=torch.float64, device=dev)
    elif amb_override is not None:
        amb = torch.as_tensor(float(amb_override), dtype=torch.float64, device=dev)
    elif arm == 'A0':
        with torch.no_grad():
            amb = torch.as_tensor(search_amb(suff_stats(x.detach(), ctx['yt'], ctx['ww']))['amb'], dtype=torch.float64, device=dev)
    else:
        raise ValueError(arm)
    A = amb + x
    rho = rho_box(A, ctx['yt'], ctx['ww'])
    data = data_loss(A, rho, ctx['yt'], ctx['ww'])
    pen = penalties(phi, e, s)
    total = data + LAM_REV * pen['rev']
    return total, data, pen, e, amb, rho


def objective_c01(phi, nu, ctx, s):
    """1108-02 C01 目标（luminance + kkt + 0.1 rev）的复刻，用于配置切换验证与 S12 对照。ctx['yt'] 为 (T,G)。返回与 objective_ua 同形的元组。"""
    l, e = lightdir(phi, nu)
    x = shading(ctx['nt'], l)
    rho, b = r3.kkt(x, ctx['yt'], ctx['ww'])
    data = (ctx['ww'] * .5 * (x * rho + b - ctx['yt']).square()).sum() / ctx['ww'].sum()
    pen = penalties(phi, e, s)
    return data + LAM_REV * pen['rev'], data, pen, e, torch.zeros((), dtype=torch.float64, device=x.device), rho


def objective_e4(phi, nu, ctx, s, lam_a=1.0, lam_e=10.0):
    """E4(r5b) 目标复刻（含 1e-5 系数项与平滑项），仅用于回归验证。"""
    l, e = lightdir(phi, nu)
    x = shading(ctx['nt'], l)
    rho, b = r3.kkt(x, ctx['yt'], ctx['ww'])
    res = x * rho + b - ctx['yt']
    data = (ctx['ww'] * .5 * res.square()).sum() / ctx['ww'].sum() + 1e-5 * (rho.square().mean() + b.square().mean())
    p = penalties(phi, e, s)
    return data + lam_a * (p['a1'] + p['a2']) + lam_e * (p['e1'] + p['e2']) + LAM_REV * p['rev'], data, p, e


# ------------------------------------------------------------------ 优化（复刻 goal1108_02_t01.fit_traj）
def fit_traj_ua(n, y, w, init, nu0, out, label, arm='A0', objective_fn=None, amb_init_mode=None, steps=STEPS, outer=OUTER, outer_ext=OUTER_EXT,
                force_tc=False, device=None, quiet=False):
    """与 goal1108_02_t01.fit_traj 结构完全相同（Adam→L-BFGS→C1–C3→T-C→Adam fallback），只替换目标函数。

    objective_fn(phi, nu, ctx, s, amb_override) -> (total,data,pen,e,amb,rho)；默认使用 objective_ua(arm)。
    amb_init_mode in {None,'lo','hi'}：仅 A0 的 ainit 流程，T-A 第 1–300 步固定 amb（0 或 2*amb0；amb0=0 时取 1.0），之后恢复剖面估计。
    y:(T,G,3)（或 (T,G) 仅供 C01 对照）；w:(T,G)；n:(T,G,3)。
    """
    dest = out / 'runs' / label
    dest.mkdir(parents=True, exist_ok=False)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = make_ctx(n, y, w, device)
    T_ = len(init)
    phi0 = np.unwrap(np.arctan2(init[:, 1], init[:, 0]))
    s = orbit_sign(init)
    if abs(np.diff(phi0).sum()) < np.pi:
        raise RuntimeError(f'环绕先验不成立：|sum dphi|={abs(np.diff(phi0).sum())}')
    phi = torch.nn.Parameter(torch.as_tensor(phi0, dtype=torch.float64, device=device).clone())
    nu = torch.nn.Parameter(torch.as_tensor(np.broadcast_to(nu0, (T_,)).copy(), dtype=torch.float64, device=device))
    if objective_fn is None:
        objective_fn = lambda p_, n_, c_, s_, ov: objective_ua(p_, n_, c_, arm, s_, ov)

    def objective(phi_=None, nu_=None, override=None):
        phi_ = phi if phi_ is None else phi_
        nu_ = nu if nu_ is None else nu_
        return objective_fn(phi_, nu_, ctx, s, override)

    def direction(phi_=None, nu_=None):
        phi_ = phi if phi_ is None else phi_
        nu_ = nu if nu_ is None else nu_
        with torch.no_grad():
            return compose(torch.stack([torch.cos(phi_), torch.sin(phi_)], 1), elev(nu_)).cpu().numpy()

    fixed = None
    if amb_init_mode == 'lo':
        fixed = 0.0
    elif amb_init_mode == 'hi':
        with torch.no_grad():
            amb0 = float(objective()[4])
        fixed = 2 * amb0 if amb0 > 0 else 1.0
    elif amb_init_mode is not None:
        raise ValueError(amb_init_mode)

    Pphi = torch.eye(T_, dtype=torch.float64, device=device)
    Pnu = Pphi.clone()

    def state():
        return phi.detach().clone(), nu.detach().clone()

    def restore(st):
        with torch.no_grad():
            phi.copy_(st[0])
            nu.copy_(st[1])

    hist, losses, all_trace = [], [], []
    last_amb = [float('nan')]

    def checkpoint(stage, step):
        np.savez_compressed(dest / 'checkpoint_latest.npz', phi=phi.detach().cpu().numpy(), nu=nu.detach().cpu().numpy(), amb=last_amb[0], stage=stage, step=step)
        write(dest / 'history_partial.json', dict(adam=hist, adam_losses=losses, optimization=all_trace))

    def progress(**kw):
        write(out / 'progress.json', dict(pid=os.getpid(), run=label, **kw, updated_utc=utc()))

    def say(obj):
        if not quiet:
            print(json.dumps(obj), flush=True)

    # ---- T-A Adam ----
    opt = torch.optim.Adam([dict(params=[phi], lr=LR_PHI), dict(params=[nu], lr=LR_NU)])
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, data, p, e, amb, _ = objective(override=fixed if (fixed is not None and step <= AINIT_STEPS) else None)
        if not torch.isfinite(loss):
            raise RuntimeError('非有限loss(Adam)')
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        last_amb[0] = float(amb.detach())
        if step in DROPS:
            for g in opt.param_groups:
                g['lr'] *= .3
        if step % 50 == 0 or step == 1:
            entry = dict(stage='TA', step=step, loss=losses[-1], data=float(data.detach()), amb=last_amb[0], amb_fixed=bool(fixed is not None and step <= AINIT_STEPS),
                         **{k: float(v.detach()) for k, v in p.items()}, e_min_deg=float(torch.rad2deg(e.detach().min())), e_max_deg=float(torch.rad2deg(e.detach().max())))
            hist.append(entry)
            checkpoint('TA', step)
            say(dict(run=label, **entry))
            progress(**entry)

    # ---- T-B L-BFGS + 证书 ----
    def run_lbfgs(max_outer, start=0):
        x0p, x0n = phi.detach().clone(), nu.detach().clone()
        qp = torch.nn.Parameter(torch.zeros_like(x0p))
        qn = torch.nn.Parameter(torch.zeros_like(x0n))
        lb = torch.optim.LBFGS([qp, qn], lr=1, history_size=50, line_search_fn='strong_wolfe', max_iter=LB_ITER, tolerance_grad=1e-12, tolerance_change=1e-15)
        eff = lambda: (x0p + Pphi @ qp, x0n + Pnu @ qn)

        def closure():
            lb.zero_grad()
            l_ = objective(*eff())[0]
            l_.backward()
            return l_

        trace, dirs = [], []

        def sync():
            with torch.no_grad():
                pe, ne = eff()
                phi.copy_(pe)
                nu.copy_(ne)

        for k in range(start, max_outer):
            lb.step(closure)
            with torch.no_grad():
                pe, ne = eff()
                res = objective(pe, ne)
            l_ = res[0]
            if not torch.isfinite(l_):
                raise FloatingPointError('非有限loss(LBFGS)')
            last_amb[0] = float(res[4])
            trace.append(float(l_))
            dirs.append(direction(pe, ne))
            all_trace.append(dict(stage='TB', outer=k + 1, loss=float(l_), amb=last_amb[0]))
            if len(trace) % 10 == 0:
                sync()
                checkpoint('TB', k + 1)
                say(dict(run=label, stage='TB', outer=k + 1, loss=trace[-1], amb=last_amb[0]))
                progress(stage='TB', outer=k + 1, loss=trace[-1], amb=last_amb[0])
            if len(trace) > WIN:
                rel = abs(trace[-1 - WIN] - trace[-1]) / max(abs(trace[-1 - WIN]), 1e-300)
                mx = float(ang_deg(dirs[-1 - WIN], dirs[-1]).max())
                if rel < 1e-9 and mx < .005:
                    sync()
                    return trace, dirs, True, rel, mx
        sync()
        rel = abs(trace[-1 - WIN] - trace[-1]) / max(abs(trace[-1 - WIN]), 1e-300)
        mx = float(ang_deg(dirs[-1 - WIN], dirs[-1]).max())
        return trace, dirs, False, rel, mx

    def probe():
        base = direction()
        st = state()
        lb0 = float(objective()[0].detach())
        o = torch.optim.Adam([dict(params=[phi], lr=1e-4), dict(params=[nu], lr=1e-3)])
        for _ in range(500):
            o.zero_grad()
            l_ = objective()[0]
            if not torch.isfinite(l_):
                restore(st)
                raise FloatingPointError('非有限loss(probe)')
            l_.backward()
            o.step()
        d1 = direction()
        l1 = float(objective()[0].detach())
        disp = float(ang_deg(base, d1).max())
        drop = (lb0 - l1) / max(abs(lb0), 1e-300)
        restore(st)
        return disp, drop

    cert = dict(path='lbfgs')
    try:
        trace, dirs, ok, rel, mx = run_lbfgs(outer)
        ext = False
        if not ok:
            ext = True
            t2, d2, ok, rel, mx = run_lbfgs(outer_ext, len(trace))
            trace += t2
            dirs += d2
        cert.update(C1_rel=rel, C1_ok=bool(np.isfinite(rel) and rel < 1e-9), C2_max_deg=mx, C2_ok=bool(np.isfinite(mx) and mx < .005), extended=ext, outer_iters=len(trace))
        disp, drop = probe()
        cert.update(C3_disp_deg=disp, C3_drop=drop, C3_ok=bool(np.isfinite(disp) and np.isfinite(drop) and disp < .1 and drop < 1e-4))
        cert['converged'] = bool(cert['C1_ok'] and cert['C2_ok'] and cert['C3_ok'])
        final_losses = trace
        if (not cert['converged']) or force_tc:
            pre = dict(cert)
            x0p, x0n = phi.detach().clone(), nu.detach().clone()
            qp = torch.nn.Parameter(torch.zeros_like(x0p))
            qn = torch.nn.Parameter(torch.zeros_like(x0n))
            oc = torch.optim.Adam([qp, qn], lr=TC_LR)
            sc = torch.optim.lr_scheduler.CosineAnnealingLR(oc, TC_STEPS, eta_min=0)
            for k in range(TC_STEPS):
                oc.zero_grad()
                res = objective(x0p + Pphi @ qp, x0n + Pnu @ qn)
                l_ = res[0]
                if not torch.isfinite(l_):
                    raise FloatingPointError('非有限loss(LBFGS-TC)')
                l_.backward()
                oc.step()
                sc.step()
                last_amb[0] = float(res[4].detach())
                all_trace.append(dict(stage='TC', step=k + 1, loss=float(l_.detach()), amb=last_amb[0]))
                if k % 500 == 0:
                    with torch.no_grad():
                        phi.copy_(x0p + Pphi @ qp)
                        nu.copy_(x0n + Pnu @ qn)
                    checkpoint('TC', k + 1)
                    say(dict(run=label, stage='TC', step=k + 1, loss=float(l_.detach()), amb=last_amb[0]))
                    progress(stage='TC', step=k + 1, loss=float(l_.detach()), amb=last_amb[0])
            with torch.no_grad():
                phi.copy_(x0p + Pphi @ qp)
                nu.copy_(x0n + Pnu @ qn)
            t3, d3, ok3, rel3, mx3 = run_lbfgs(TC_OUTER)
            trace += t3
            disp, drop = probe()
            cert = dict(path='lbfgs+TC', pre_TC=pre, C1_rel=rel3, C1_ok=bool(np.isfinite(rel3) and rel3 < 1e-9), C2_max_deg=mx3, C2_ok=bool(np.isfinite(mx3) and mx3 < .005),
                        extended=ext, outer_iters=len(trace), TC_steps=TC_STEPS, TC_lr=TC_LR, C3_disp_deg=disp, C3_drop=drop,
                        C3_ok=bool(np.isfinite(disp) and np.isfinite(drop) and disp < .1 and drop < 1e-4))
            cert['converged'] = bool(cert['C1_ok'] and cert['C2_ok'] and cert['C3_ok'])
            final_losses = trace
    except (FloatingPointError, RuntimeError) as exc:
        if 'LBFGS' not in str(exc):
            raise
        restore((torch.as_tensor(phi0, dtype=torch.float64, device=device), torch.as_tensor(np.broadcast_to(nu0, (T_,)).copy(), dtype=torch.float64, device=device)))
        o = torch.optim.Adam([dict(params=[phi], lr=LR_PHI), dict(params=[nu], lr=LR_NU)])
        N = 6000 + steps
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(o, N, eta_min=0)
        fl = []
        for fallback_step in range(N):
            o.zero_grad()
            res = objective()
            l_ = res[0]
            if not torch.isfinite(l_):
                raise FloatingPointError('非有限loss(Adam fallback)')
            l_.backward()
            o.step()
            sched.step()
            fl.append(float(l_.detach()))
            last_amb[0] = float(res[4].detach())
            all_trace.append(dict(stage='fallback', step=fallback_step + 1, loss=fl[-1], amb=last_amb[0]))
            if fallback_step % 500 == 0:
                checkpoint('fallback', fallback_step + 1)
                progress(stage='fallback', step=fallback_step + 1, loss=fl[-1], amb=last_amb[0])
        rel = abs(fl[-501] - fl[-1]) / max(abs(fl[-501]), 1e-300)
        disp, drop = probe()
        cert = dict(path='adam_fallback', reason=str(exc), C1_rel=rel, C3_disp_deg=disp, C3_drop=drop, converged=bool(rel < 1e-7 and disp < .1 and drop < 1e-4))
        final_losses = fl

    with torch.no_grad():
        loss, data, p, e, amb, rho = objective()
        end = direction()
    pp = {k: float(v) for k, v in p.items()}
    eld = np.rad2deg(e.detach().cpu().numpy())
    info = dict(penalties=pp, orbit_sign=s, final_loss=float(loss), data=float(data), certificate=cert, converged=cert['converged'], elev_deg=eld.tolist(),
                amb=float(amb), amb_fixed_first_steps=fixed)
    np.save(dest / 'direction.npy', end)
    np.savez_compressed(dest / 'parameters.npz', phi=phi.detach().cpu().numpy(), nu=nu.detach().cpu().numpy(), amb=np.float64(amb), rho=rho.detach().cpu().numpy())
    write(dest / 'history.json', dict(adam=hist, adam_losses=losses, optimization=all_trace, final_losses=final_losses))
    write(dest / 'certificate.json', cert)
    write(dest / 'config.json', dict(arm=arm, lam_a=0., lam_e=0., lam_rev=LAM_REV, adam_steps=steps, lr_phi=LR_PHI, lr_nu=LR_NU,
                                     drops=list(DROPS), e_center_deg=45., e_halfwidth_deg=5., device=device, dtype='float64', support_count=y.shape[1], final_loss=float(loss),
                                     final_amb=float(amb), amb_init_mode=amb_init_mode, amb_fixed_value=fixed, amb_fixed_steps=AINIT_STEPS if fixed is not None else 0,
                                     certificate=cert, penalties=pp, gt_read=False))
    say(dict(run=label, stage='DONE', final_loss=float(loss), amb=float(amb), certificate=cert))
    return end, info


# ------------------------------------------------------------------ 有效性与分歧
def endpoint_valid(l, rho, amb):
    """规划 6.3：有限、单位范数误差 <=1e-10、仰角在 [0,90]、rho in [-1e-12,1+1e-12]、amb >= -1e-12。"""
    l = np.asarray(l)
    if l.shape != (150, 3) and l.ndim != 2:
        return False
    if not (np.isfinite(l).all() and np.isfinite(rho).all() and np.isfinite(amb)):
        return False
    e = elev_deg(l)
    return bool(np.max(abs(np.linalg.norm(l, axis=1) - 1)) <= 1e-10 and np.min(e) >= -1e-8 and np.max(e) <= 90 + 1e-8
                and np.min(rho) >= -1e-12 and np.max(rho) <= 1 + 1e-12 and amb >= -1e-12)


def chord_deg(a, b):
    """数值精确的小角度（见 r6_env.chord_angle_deg）。"""
    return np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1) / 2, 0, 1)))


def choose(valid, arm):
    """第 6.3 节：不回退、不替补。"""
    return {'selected': f'{arm}/full' if valid else None, 'status': 'VALID' if valid else 'NO_VALID_PREDICTION', 'reference_only': 'E4'}


def final_status(full_valid, invalid_aux):
    return 'NO_VALID_PREDICTION' if not full_valid else ('INCOMPLETE_DIAGNOSTICS' if invalid_aux else 'AWAITING_REVIEW')


def repeat_status(full_dir, repeat_dir):
    if repeat_dir is None or full_dir is None:
        return 'UNAVAILABLE'
    return 'DETERMINISTIC' if float(chord_deg(full_dir, repeat_dir).max()) <= 1e-4 else 'NONDETERMINISTIC'


# ------------------------------------------------------------------ 正则审计（S9 同款合成断言，运行时写入 regularization_audit.json）
def regularization_audit_probe():
    """在合成小问题上断言：total == data + 0.1*rev（torch.equal），总梯度逐位等于 data+0.1*rev 的梯度（即 L_coef/a1/a2/e1/e2 的梯度贡献严格为 0），
    参数清单 == {phi, nu}，预条件为单位阵（fit_traj_ua 中 Pphi=Pnu=I）。"""
    import inspect
    rng = np.random.default_rng(SEED)
    T, G = 150, 9
    n = unit(rng.normal(size=(G, 3)))
    y = rng.random((T, G, 3))
    w = rng.random((T, G))
    ctx = make_ctx(n, y, w, 'cpu')
    phi = torch.tensor(np.linspace(0, 7, T) + rng.normal(0, .03, T), dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(rng.normal(size=T), dtype=torch.float64, requires_grad=True)
    res = {}
    for arm in ('A0', 'A1'):
        total, data, pen, e, amb, rho = objective_ua(phi, nu, ctx, arm, 1.0)
        ref = data + LAM_REV * pen['rev']
        assert torch.equal(total, ref), arm
        g1 = torch.autograd.grad(total, (phi, nu), retain_graph=True)
        g2 = torch.autograd.grad(ref, (phi, nu), retain_graph=True)
        assert all(torch.equal(u, v) for u, v in zip(g1, g2)), arm
        res[arm] = dict(total_equals_data_plus_0p1_rev=True, grad_equals_reference=True, raw_penalties={k: float(v.detach()) for k, v in pen.items()})
    src = inspect.getsource(objective_ua)
    for token in ("'a1'", "'a2'", "'e1'", "'e2'", 'square().mean()', 'kkt', 'sh9', 'env_terms', 'get_features', 'eval_sh'):
        assert token not in src, token
    res['parameters'] = dict(phi=[T], nu=[T])
    res['preconditioner'] = 'identity (Pphi=Pnu=I)'
    return res
