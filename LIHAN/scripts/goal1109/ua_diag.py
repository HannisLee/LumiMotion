"""1109-01 T01 无 GT 诊断（K-A1..K-A8）与 RGB 拟合质量（规划第 7 节）。所有记录带 gt_read=false。

诊断一律在 CPU float64 上进行，输入是冻结观测子集 + 拟合终点，不接触任何 GT 文件。
"""
import numpy as np
import torch

from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r6_env as r6
from LIHAN.scripts.goal1109 import ua_core as C

TOL_ACTIVE = 1e-12
NZ_BINS5 = [-1.0, -.5, 0.0, .5, .85, 1.0 + 1e-9]
NZ_BINS4 = [(-.85, -.5), (-.5, 0.0), (0.0, .5), (.5, .85 + 1e-9)]


# ------------------------------------------------------------------ 诊断上下文
def make_dctx(n, y, rgb, inten, w):
    """n:(T,G,3)，y:(T,G,3) 线性 RGB（无效 NaN），rgb:(T,G,3) sRGB（无效 NaN），inten:(T,G) luminance，w:(T,G)。返回 CPU float64 张量上下文。"""
    valid = np.isfinite(y).all(-1)
    ww = np.where(valid, w, 0.0)
    return dict(n=np.asarray(n), nt=torch.as_tensor(n, dtype=torch.float64), yt=torch.as_tensor(np.nan_to_num(y), dtype=torch.float64),
                rgb=np.nan_to_num(rgb), I=np.nan_to_num(inten), ww=torch.as_tensor(ww, dtype=torch.float64), w_np=ww, valid=valid)


def shade_t(dc, l):
    return torch.relu(torch.einsum('tgc,tc->tg', dc['nt'], torch.as_tensor(l, dtype=torch.float64)))


def stats_at(dc, l):
    x = shade_t(dc, l)
    return x, C.suff_stats(x, dc['yt'], dc['ww'])


def slice_S(S, mask=None, ch=None):
    out = {}
    for k, v in S.items():
        if mask is not None:
            v = v[mask]
        if ch is not None and v.ndim == 2:
            v = v[:, ch:ch + 1]
        out[k] = v
    return out


# ------------------------------------------------------------------ K-A1
def count_minima(F, tol_rel=1e-10):
    """离散曲线的严格局部极小个数（含端点）。显著度 = min(左侧邻域最大值, 右侧邻域最大值) - F；端点外侧视为无穷高；显著度 > tol_rel*|F| 才计数。"""
    F = np.asarray(F)
    n = len(F)
    found = []
    i = 0
    while i < n:
        j = i
        while j < n - 1 and F[j + 1] == F[i]:
            j += 1
        left = F[i - 1] if i > 0 else np.inf
        right = F[j + 1] if j < n - 1 else np.inf
        if F[i] < left and F[i] < right:
            lmax = np.inf if i == 0 else -np.inf
            k = i - 1
            while k >= 0 and F[k] >= F[i]:
                lmax = max(lmax, F[k])
                k -= 1
            rmax = np.inf if j == n - 1 else -np.inf
            k = j + 1
            while k < n and F[k] >= F[i]:
                rmax = max(rmax, F[k])
                k += 1
            prom = min(lmax, rmax) - F[i]
            if prom > tol_rel * abs(F[i]):
                found.append(i)
        i = j + 1
    return found


def ka1_profile(dc, l, amb_star):
    x, S = stats_at(dc, l)
    hi = max(4.0, 2 * amb_star)
    grid = np.linspace(0.0, hi, 4001)
    F = C.profile_F(torch.as_tensor(grid), S).numpy()
    mins = count_minima(F)
    flags = []
    if len(mins) > 1:
        flags.append('A_PROFILE_MULTIMODAL')
    if amb_star == 0.0:
        flags.append('A_AT_ZERO')
    return dict(n_local_minima=len(mins), minima_amb=[float(grid[i]) for i in mins], minima_F=[float(F[i]) for i in mins], grid_max=float(hi), flags=flags, gt_read=False), grid, F


# ------------------------------------------------------------------ K-A2
def als_amb(S, amb0, max_rounds=20000, tol=1e-12):
    """固定方向：交替 rho（闭式 [0,1]）与 amb=max(0, sum w rho (y-rho x)/sum w rho^2)；只用充分统计量。"""
    amb = float(amb0)
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        D = S['Sxx'] + 2 * amb * S['Sx'] + amb * amb * S['S1']
        N = S['Sxy'] + amb * S['Sy']
        rho = (N / (D.unsqueeze(-1) + C.DAMP)).clamp(0, 1)
        num = (rho * (S['Sy'] - rho * S['Sx'].unsqueeze(-1))).sum()
        den = (rho * rho * S['S1'].unsqueeze(-1)).sum()
        new = max(0.0, float(num / den)) if float(den) > 0 else amb
        done = abs(new - amb) < tol
        amb = new
        if done:
            break
    return amb, rounds


def ka2_als(dc, l, amb_star):
    x, S = stats_at(dc, l)
    Fstar = float(C.profile_F(amb_star, S)[0])
    starts = [0.0, .05, .2, .5, 1.0, 2.0, 4.0]
    rows = []
    for a0 in starts:
        a_end, rounds = als_amb(S, a0)
        rows.append(dict(start=a0, amb_end=a_end, rounds=rounds, F_end=float(C.profile_F(a_end, S)[0])))
    flags = []
    if any(r['F_end'] < Fstar - 1e-12 * Fstar for r in rows):
        flags.append('A_SEARCH_MISSED')
    ends = np.array([r['amb_end'] for r in rows])
    Fs = np.array([r['F_end'] for r in rows])
    if (ends.max() - ends.min()) > 1e-6 * max(1.0, amb_star) and (Fs.max() - Fs.min()) <= 1e-12 * max(abs(Fstar), 1e-300):
        flags.append('A_PROFILE_FLAT')
    return dict(F_star=Fstar, amb_star=amb_star, starts=rows, flags=flags, gt_read=False)


# ------------------------------------------------------------------ K-A3 / K-A3'
def ka3_stability(amb_by_flow, dirs, full='full'):
    af = amb_by_flow[full]
    den = max(af, 0.01)
    deltas = {k: abs(amb_by_flow[k] - af) / den for k in ['A', 'B', 'boot103108', 'boot103109', 'hi', 'lo'] if k in amb_by_flow}
    flags = ['A_UNSTABLE'] if deltas and max(deltas.values()) > 0.10 else []
    init = {}
    for k in ['ainit_lo', 'ainit_hi']:
        if k in amb_by_flow:
            d = abs(amb_by_flow[k] - af) / den
            m = float(C.chord_deg(dirs[k], dirs[full]).mean()) if k in dirs else None
            init[k] = dict(delta=d, mean_angle_deg=m)
            if d > 1e-3 or (m is not None and m > 0.1):
                flags.append('A_INIT_SENSITIVE')
    return dict(amb_by_flow=amb_by_flow, delta_vs_full=deltas, max_delta=max(deltas.values()) if deltas else None, ainit=init, flags=sorted(set(flags)), gt_read=False)


# ------------------------------------------------------------------ Fisher：K-A4 / K-A5
def fisher_ua(nt, yt, ww, l, amb, rho, include_a=True):
    """箭头阵 Schur：先消去活跃 rho_gc（对角块），再消去 amb。θ=(phi_1..T, e_1..T)(弧度)。

    活跃 rho：1e-12 < rho < 1-1e-12（被截断到 0/1 的 rho 不是自由参数）。返回 A（仅消去 rho，amb 固定）、Atilde（再消去 amb）、s_a、r_a、
    每通道 corr(amb, mean active rho)。
    """
    T, G, Cn = yt.shape
    lt = torch.as_tensor(l, dtype=torch.float64)
    rho = torch.as_tensor(rho, dtype=torch.float64)
    x = torch.relu(torch.einsum('tgc,tc->tg', nt, lt))
    A = amb + x
    W = ww.sum()
    v = ww / (Cn * W)                                           # (T,G) 每个通道的行权
    pp, pe = r6.theta_partials(nt, lt)
    r2 = rho.square()
    r2s = r2.sum(1)                                              # (G,)
    ar = torch.arange(T)
    Ath = torch.zeros(2 * T, 2 * T, dtype=torch.float64)
    Ath[ar, ar] = torch.einsum('tg,g,tg->t', v, r2s, pp * pp)
    Ath[T + ar, T + ar] = torch.einsum('tg,g,tg->t', v, r2s, pe * pe)
    cross = torch.einsum('tg,g,tg->t', v, r2s, pp * pe)
    Ath[ar, T + ar] = cross
    Ath[T + ar, ar] = cross
    vA = v * A
    Fthr = torch.cat([((vA * pp).unsqueeze(-1) * rho.unsqueeze(0)).reshape(T, G * Cn), ((vA * pe).unsqueeze(-1) * rho.unsqueeze(0)).reshape(T, G * Cn)], 0)
    Frr = ((v * A * A).sum(0).unsqueeze(-1).expand(G, Cn)).reshape(G * Cn)
    act = (rho.reshape(G * Cn) > TOL_ACTIVE) & (rho.reshape(G * Cn) < 1 - TOL_ACTIVE) & (Frr > 1e-300)
    if not include_a:
        Fur = Fthr[:, act]
        Sth = Ath - (Fur / Frr[act][None]) @ Fur.T
        return dict(A=Sth, Atilde=Sth, a_included=False, active_rho=int(act.sum()), total_rho=int(G * Cn))
    Fta = torch.cat([torch.einsum('tg,tg,g->t', v, pp, r2s), torch.einsum('tg,tg,g->t', v, pe, r2s)])
    Faa = torch.einsum('tg,g->', v, r2s)
    Far = (rho * (v * A).sum(0).unsqueeze(-1)).reshape(G * Cn)
    Fuu = torch.zeros(2 * T + 1, 2 * T + 1, dtype=torch.float64)
    Fuu[:2 * T, :2 * T] = Ath
    Fuu[:2 * T, 2 * T] = Fta
    Fuu[2 * T, :2 * T] = Fta
    Fuu[2 * T, 2 * T] = Faa
    Fur = torch.cat([Fthr[:, act], Far[act][None]], 0)
    Sfull = Fuu - (Fur / Frr[act][None]) @ Fur.T
    Sth = Sfull[:2 * T, :2 * T]
    Sta = Sfull[:2 * T, 2 * T]
    s_a = float(Sfull[2 * T, 2 * T])
    Atilde = Sth - torch.outer(Sta, Sta) / s_a if s_a > 0 else Sth
    # 每通道 corr(amb, mean active rho)
    act2 = act.reshape(G, Cn)
    b2 = Far.reshape(G, Cn)
    D2 = Frr.reshape(G, Cn)
    corr = []
    for c in range(Cn):
        m = act2[:, c]
        if int(m.sum()) == 0 or s_a <= 0:
            corr.append(None)
            continue
        sb = float((b2[m, c] / D2[m, c]).sum())
        s1 = float((1.0 / D2[m, c]).sum())
        corr.append(float(-sb / np.sqrt(s_a * s1 + sb * sb)))
    return dict(A=Sth, Atilde=Atilde, a_included=True, s_a=s_a, F_aa=float(Faa), r_a=s_a / float(Faa), corr_amb_rhobar=corr,
                active_rho=int(act.sum()), total_rho=int(G * Cn))


def mode_vectors(T):
    z = np.zeros(T)
    o = np.ones(T) / np.sqrt(T)
    modes = {'u_e0': np.concatenate([z, o]), 'u_phi0': np.concatenate([o, z])}
    for k, u in r6.dct_modes(T).items():
        modes[f'e_{k}'] = np.concatenate([z, u])
        modes[f'phi_{k}'] = np.concatenate([u, z])
    return modes


def retention(A, Atilde, modes, tau):
    """R(u)=info_after/info_before（数据项 + 纯数据岭 tau）。"""
    I2 = torch.eye(A.shape[0], dtype=torch.float64)
    out = {}
    for name, u in modes.items():
        before = r6.mode_info(A + tau * I2, u)
        after = r6.mode_info(Atilde + tau * I2, u)
        out[name] = dict(R=after / before, info_before=before, info_after=after)
    return out


def ka4_ka5(dc, l, amb, rho, arm, refs, inten_sub, w_sub):
    """K-A4/K-A5。refs：dict(E4=l_e4, C01=l_c01)，在各自冻结方向处用 fisher_e4（luminance）计算 R_b。"""
    T = dc['yt'].shape[0]
    modes = mode_vectors(T)
    F = fisher_ua(dc['nt'], dc['yt'], dc['ww'], l, amb, rho, include_a=(arm == 'A0'))
    tau = 1e-10 * float(torch.trace(F['A'])) / (2 * T)
    use_a = arm == 'A0' and amb > 0 and F.get('s_a', 0) > 0
    Atil = F['Atilde'] if use_a else F['A']
    ret = retention(F['A'], Atil, modes, tau)
    res = dict(arm=arm, a_eliminated=bool(use_a), tau=tau, modes={k: v for k, v in ret.items()}, active_rho=F['active_rho'], total_rho=F['total_rho'], gt_read=False)
    flags = []
    if arm == 'A0':
        res.update(s_a=F.get('s_a'), F_aa=F.get('F_aa'), r_a=F.get('r_a'), corr_amb_rhobar=F.get('corr_amb_rhobar'))
        if (F.get('r_a') is not None and F['r_a'] < .05) or any(c is not None and abs(c) > .95 for c in F.get('corr_amb_rhobar', [])):
            flags.append('A_RHO_CONFOUNDED')
    if use_a and ret['u_e0']['R'] < .1:
        flags.append('ELEV_AMB_CONFOUNDED')
    # 对照：E4 / C01 冻结方向处的 R_b（消去 rho,b 相对只消去 rho）
    ctx_nt = dict(nt=dc['nt'])
    ref_out = {}
    for name, lr in refs.items():
        E = r6.fisher_e4(ctx_nt, inten_sub, w_sub, lr)
        taur = 1e-10 * float(torch.trace(E['A_rho'])) / (2 * T)
        rr = retention(E['A_rho'], E['A_both'], modes, taur)
        ref_out[name] = dict(modes=rr, tau=taur, active_rho=E['active_rho'], active_b=E['active_b'], data=E['data'])
    res['refs'] = ref_out
    res['flags'] = flags
    return res


# ------------------------------------------------------------------ K-A5'
def shift_elevation(l, delta_deg):
    l = np.asarray(l)
    phi = np.arctan2(l[:, 1], l[:, 0])
    e = np.arcsin(np.clip(l[:, 2], -1, 1)) + np.deg2rad(delta_deg)
    return np.column_stack([np.cos(e) * np.cos(phi), np.cos(e) * np.sin(phi), np.sin(e)])


def _curv(deltas, L):
    """中心差分曲率（h=1°）与 |Δ|<=2° 的二次最小二乘曲率（d^2L/dΔ^2，单位 1/deg^2）。"""
    d = np.array(deltas)
    Lv = np.array(L)
    i0, ip, im = list(d).index(0.0), list(d).index(1.0), list(d).index(-1.0)
    cd = float(Lv[ip] + Lv[im] - 2 * Lv[i0])
    m = np.abs(d) <= 2
    co = np.polyfit(d[m], Lv[m], 2)
    return dict(central_h1=cd, lsq_abs_le_2=float(2 * co[0])), co


def ka5p_profile(dc, l, amb_star, arm, refs, inten_sub, w_sub, deltas=(-4, -3, -2, -1, -.5, 0, .5, 1, 2, 3, 4)):
    deltas = [float(d) for d in deltas]
    La, Lb, a_re = [], [], []
    for d in deltas:
        x, S = stats_at(dc, shift_elevation(l, d))
        Lb.append(float(C.profile_F(amb_star, S)[0]))
        if arm == 'A0':
            r = C.search_amb(S)
            La.append(r['F'])
            a_re.append(r['amb'])
    out = dict(deltas_deg=deltas, gt_read=False)
    i0 = deltas.index(0.0)
    out['b_fixed_amb'] = dict(L=Lb, dL=[v - Lb[i0] for v in Lb])
    cb, co_b = _curv(deltas, Lb)
    out['b_fixed_amb'].update(curvature=cb)
    if arm == 'A0':
        out['a_reprofile'] = dict(L=La, dL=[v - La[i0] for v in La], amb=a_re)
        ca, co_a = _curv(deltas, La)
        out['a_reprofile'].update(curvature=ca)
        # 极小点：网格极小 + 抛物线顶点（二次 LSQ）
        out['a_reprofile']['argmin_grid_deg'] = float(deltas[int(np.argmin(La))])
        out['a_reprofile']['argmin_parabola_deg'] = float(-co_a[1] / (2 * co_a[0])) if co_a[0] > 0 else None
        out['curvature_ratio_a_over_b'] = {k: (ca[k] / cb[k] if cb[k] != 0 else None) for k in ca}
    # (c) E4 与 C01 方向处用 kkt 重解（luminance 数据项，无系数正则）
    yt = torch.as_tensor(np.nan_to_num(inten_sub), dtype=torch.float64)
    wt = torch.as_tensor(np.where(np.isfinite(inten_sub), w_sub, 0), dtype=torch.float64)
    refc = {}
    for name, lr in refs.items():
        Ls = []
        for d in deltas:
            xx = torch.relu(torch.einsum('tgc,tc->tg', dc['nt'], torch.as_tensor(shift_elevation(lr, d), dtype=torch.float64)))
            rr, bb = r3.kkt(xx, yt, wt)
            Ls.append(float((wt * .5 * (xx * rr + bb - yt).square()).sum() / wt.sum()))
        cr, _ = _curv(deltas, Ls)
        refc[name] = dict(L=Ls, dL=[v - Ls[i0] for v in Ls], curvature=cr)
    out['refs_kkt_luminance'] = refc
    return out


# ------------------------------------------------------------------ K-A6
def _wls2(z1, z2, r, wt):
    """无截距二元加权最小二乘，返回 (coef, se)。"""
    Zw = np.stack([z1, z2], 1)
    G = (Zw * wt[:, None]).T @ Zw
    b = (Zw * wt[:, None]).T @ r
    coef = np.linalg.solve(G, b)
    e = r - Zw @ coef
    dof = max(len(r) - 2, 1)
    s2 = float((wt * e * e).sum() / dof)
    cov = s2 * np.linalg.inv(G)
    return coef, np.sqrt(np.diag(cov))


def ka6_uniformity(dc, l, amb_star, rho, coverage_only=False):
    x, S = stats_at(dc, l)
    xn = x.numpy()
    wn = dc['w_np']
    yn = dc['yt'].numpy()
    nz = dc['n'][..., 2]
    rho_n = np.asarray(rho)
    T, G, Cn = yn.shape
    a_ref = max(float(amb_star), 0.01)
    flags = []
    out = dict(a_ref=a_ref, note='a*=0 时相对阈值使用 max(a*,0.01)（与 K-A3 一致）', gt_read=False)
    # ① 背光/受光回归 y - rho*(amb[+x]) = rho*(delta + g*nz)
    for name, mask in ([] if coverage_only else [('backlit', (xn <= 0) & (wn > 0)), ('lit', (xn > 0) & (wn > 0))]):
        tt, gg = np.nonzero(mask)
        if len(tt) < 10:
            out[name] = dict(status='N/A', n=int(len(tt)))
            continue
        z1 = np.repeat(rho_n[gg], 1, axis=0)                                   # (n,3)
        base = amb_star + (xn[tt, gg] if name == 'lit' else 0.0)
        r = yn[tt, gg] - rho_n[gg] * (base if np.ndim(base) == 0 else base[:, None])
        zz1 = z1.reshape(-1)
        zz2 = (rho_n[gg] * nz[tt, gg][:, None]).reshape(-1)
        wt = np.repeat(wn[tt, gg], Cn)
        coef, se = _wls2(zz1, zz2, r.reshape(-1), wt)
        out[name] = dict(delta=float(coef[0]), g=float(coef[1]), delta_se=float(se[0]), g_se=float(se[1]), n_obs=int(len(tt)), weight_sum=float(wn[mask].sum()))
    if 'g' in out.get('backlit', {}) and abs(out['backlit']['g']) > 0.2 * a_ref:
        flags.append('AMBIENT_VERTICAL_GRADIENT')
    # ② 法线分箱的局部 amb
    sw = np.maximum(wn.sum(0), 1e-300)
    zbar = (wn * nz).sum(0) / sw
    bins = []
    for lo, hi in ([] if coverage_only else NZ_BINS4):
        m = (zbar >= lo) & (zbar < hi) & (wn.sum(0) > 0)
        if int(m.sum()) >= 50:
            r = C.search_amb(slice_S(S, mask=torch.as_tensor(m)))
            bins.append(dict(lo=lo, hi=min(hi, .85), n_gaussians=int(m.sum()), amb=r['amb'], flags=r['flags']))
        else:
            bins.append(dict(lo=lo, hi=min(hi, .85), n_gaussians=int(m.sum()), amb=None, note='<50 个高斯，不计入'))
    vals = [b['amb'] for b in bins if b['amb'] is not None]
    out['bins'] = bins
    out['bin_range'] = float(max(vals) - min(vals)) if len(vals) >= 2 else None
    if out['bin_range'] is not None and out['bin_range'] > 0.5 * a_ref:
        flags.append('AMBIENT_NONUNIFORM')
    # ③ 覆盖统计
    lit = (xn > 0) & (wn > 0)
    back = (xn <= 0) & (wn > 0)
    hist_lit, hist_back = [], []
    for i in range(len(NZ_BINS5) - 1):
        m = (nz >= NZ_BINS5[i]) & (nz < NZ_BINS5[i + 1])
        hist_lit.append(float(wn[lit & m].sum()))
        hist_back.append(float(wn[back & m].sum()))
    tot = float(wn.sum())
    per_frame_back = (wn * back).sum(1) / np.maximum(wn.sum(1), 1e-300)
    info_g = ((lit.sum(0) >= 3) & (back.sum(0) >= 3))
    support_g = (wn > 0).any(0)
    frac_info = float(info_g.sum() / max(support_g.sum(), 1))
    back_share = float(wn[back].sum() / tot)
    out['coverage'] = dict(nz_bin_edges=NZ_BINS5[:-1] + [1.0], lit_weight_hist=hist_lit, back_weight_hist=hist_back, back_weight_share=back_share,
                           per_frame_back_fraction=per_frame_back.tolist(), truncation_informative_gaussian_fraction=frac_info)
    if back_share < .05 or frac_info < .2:
        flags.append('LOW_BACKLIT_COVERAGE')
    out['flags'] = flags
    return out


# ------------------------------------------------------------------ K-A7
def ka7_chroma(dc, l, amb_star):
    x, S = stats_at(dc, l)
    a_c = [C.search_amb(slice_S(S, ch=c))['amb'] for c in range(S['Sy'].shape[-1])]
    a_ref = max(float(amb_star), 0.01)
    spread = float(max(a_c) - min(a_c))
    flags = ['AMBIENT_CHROMA_MISMATCH'] if spread > 0.2 * a_ref else []
    return dict(amb_c=a_c, spread=spread, a_ref=a_ref, flags=flags, gt_read=False)


# ------------------------------------------------------------------ K-A8
def qs(a):
    a = np.asarray(a)
    return dict(zip(['min', 'p1', 'p10', 'p50', 'p90', 'p99', 'max'], map(float, np.quantile(a, [0, .01, .1, .5, .9, .99, 1]))))


def ka8_energy(dc, l, amb, rho, compute_noub=True):
    x, S = stats_at(dc, l)
    xn = x.numpy()
    wn = dc['w_np']
    rho = np.asarray(rho)
    yhat = rho[None] * (amb + xn[..., None])
    amb_part = np.broadcast_to(rho[None] * amb, yhat.shape)
    v = wn[..., None]
    lit = (xn > 0)[..., None]
    eta = {}
    for name, m in [('all', np.ones_like(lit)), ('lit', lit), ('back', ~lit)]:
        vm = v * m
        num = (vm * amb_part).sum((0, 1))
        den = (vm * yhat).sum((0, 1))
        eta[name] = [float(a / b) if b > 0 else None for a, b in zip(num, den)]

    def tmean(z):
        return (v * z).sum(0, keepdims=True) / np.maximum(v.sum(0, keepdims=True), 1e-300)
    dev_a, dev_y = amb_part - tmean(amb_part), yhat - tmean(yhat)
    vden = float((v * dev_y ** 2).sum())
    eta_var = float((v * dev_a ** 2).sum() / vden) if vden > 0 else 0.0
    assert abs(eta_var) < 1e-12, eta_var
    prof = {}
    flags = []
    for c in range(rho.shape[1]):
        r = rho[:, c]
        f0, f1 = float((r <= TOL_ACTIVE).mean()), float((r >= 1 - TOL_ACTIVE).mean())
        prof[f'ch{c}'] = dict(**qs(r), mean=float(r.mean()), std=float(r.std()), frac_zero=f0, frac_one=f1)
        if f1 > .01 and 'RHO_UPPER_ACTIVE' not in flags:
            flags.append('RHO_UPPER_ACTIVE')
        if f0 > .05 and 'RHO_ZERO_ACTIVE' not in flags:
            flags.append('RHO_ZERO_ACTIVE')
    amb_noub = C.search_amb(S, upper=None)['amb'] if compute_noub else None
    return dict(eta_lin=eta, eta_var=eta_var, rho=prof, amb_noUB=amb_noub, amb=amb, flags=flags, gt_read=False)


# ------------------------------------------------------------------ 7.3 RGB 拟合质量
def oetf(u):
    u = np.clip(u, 0.0, 1.0)
    return np.where(u <= .0031308, 12.92 * u, 1.055 * np.power(np.maximum(u, 1e-300), 1 / 2.4) - .055)


def psnr_from_mse(m):
    return float(-10 * np.log10(max(m, 1e-300)))


def ssim_pts(p, o, w):
    """每帧每通道的全局点集 SSIM（w 加权）；frames 的 w>0 点集；150x3 等权平均。p,o:(T,G,3)，w:(T,G)。"""
    C1, C2 = .01 ** 2, .03 ** 2
    sw = w.sum(1)
    ok = sw > 0
    wn = w / np.where(sw > 0, sw, 1)[:, None]
    mp = (wn[..., None] * p).sum(1)
    mo = (wn[..., None] * o).sum(1)
    vp = (wn[..., None] * (p - mp[:, None]) ** 2).sum(1)
    vo = (wn[..., None] * (o - mo[:, None]) ** 2).sum(1)
    cv = (wn[..., None] * (p - mp[:, None]) * (o - mo[:, None])).sum(1)
    s = (2 * mp * mo + C1) * (2 * cv + C2) / ((mp ** 2 + mo ** 2 + C1) * (vp + vo + C2))
    return float(s[ok].mean()), s


def quality(pred_lin, y_lin, rgb, inten, w):
    """观测层面的 RGB 拟合质量（Gaussian 投影采样点，权重 w，只用 w>0）。"""
    w = np.asarray(w)
    m = w > 0
    W = w.sum()
    r_lin = pred_lin - y_lin
    mse_lin = float((w[..., None] * r_lin ** 2).sum() / (3 * W))
    ps = oetf(pred_lin)
    r_s = ps - rgb
    mse_s = float((w[..., None] * r_s ** 2).sum() / (3 * W))
    ssim, _ = ssim_pts(ps, rgb, w)
    yy = pred_lin @ C.KAPPA
    mse_y = float((w * (yy - inten) ** 2).sum() / W)
    return dict(PSNR_lin=psnr_from_mse(mse_lin), PSNR_sRGB=psnr_from_mse(mse_s), SSIM_pts=ssim, PSNR_Y=psnr_from_mse(mse_y),
                PSNR_sRGB_per_channel=[psnr_from_mse(float((w * r_s[..., c] ** 2).sum() / W)) for c in range(3)],
                channel_rms_lin=[float(np.sqrt((w * r_lin[..., c] ** 2).sum() / W)) for c in range(3)], n_obs=int(m.sum()), weight_sum=float(W),
                note='观测层面（Gaussian 投影采样点）；SSIM_pts 为点集全局 SSIM，不可与图像 SSIM 对比')


def predict_ua(dc, l, amb, rho):
    x = shade_t(dc, l).numpy()
    return np.asarray(rho)[None] * (amb + x[..., None])


def insample_quality(dc, l, amb, rho):
    pred = predict_ua(dc, l, amb, rho)
    q = quality(pred, dc['yt'].numpy(), dc['rgb'], dc['I'], dc['w_np'])
    rms = q['channel_rms_lin']
    q['channel_rms_ratio_max_over_min'] = float(max(rms) / max(min(rms), 1e-300))
    return q


def heldout_quality(dc, l, amb):
    """固定 full 的 amb 与方向：奇数帧（1-based，即 0-based 偶数）闭式求 rho，偶数帧评估；返回指标与逐通道截断 MAE(0.2)。"""
    yt, wn = dc['yt'], dc['w_np']
    Tn = yt.shape[0]
    x = shade_t(dc, l)
    cal = torch.as_tensor((np.arange(Tn)[:, None] % 2 == 0) & (wn > 0))
    wc = torch.where(cal, dc['ww'], torch.zeros_like(dc['ww']))
    A = amb + x
    rho = C.rho_box(A, yt, wc).numpy()
    gate = (cal.sum(0).numpy() >= 3) & ((wc * A * A).sum(0).numpy() > 1e-12)
    test = (~cal.numpy()) & (wn > 0) & gate[None]
    wt = np.where(test, wn, 0.0)
    pred = rho[None] * (amb + x.numpy())[..., None]
    q = quality(pred, yt.numpy(), dc['rgb'], dc['I'], wt)
    r = pred - yt.numpy()
    q['trunc_mae_0p2_per_channel'] = [float((wt * np.minimum(abs(r[..., c]), .2)).sum() / wt.sum()) for c in range(3)]
    q['n_gaussians_used'] = int(gate.sum())
    return q


def ref_luminance_quality(dc, l, inten_sub, w_sub):
    """E4 / C01 口径：luminance kkt 拟合（rho x + b）在 body_fit 上的 PSNR_Y（线性域，权重 w）。"""
    yt = torch.as_tensor(np.nan_to_num(inten_sub), dtype=torch.float64)
    wt = torch.as_tensor(np.where(np.isfinite(inten_sub), w_sub, 0), dtype=torch.float64)
    x = torch.relu(torch.einsum('tgc,tc->tg', dc['nt'], torch.as_tensor(l, dtype=torch.float64)))
    rr, bb = r3.kkt(x, yt, wt)
    mse = float((wt * (x * rr + bb - yt).square()).sum() / wt.sum())
    return dict(PSNR_Y=psnr_from_mse(mse), mse=mse)
