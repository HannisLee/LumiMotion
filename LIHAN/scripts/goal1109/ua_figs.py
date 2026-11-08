"""1109-01 T01 拟合阶段图（四类必选 + 两张）。残差热图对 NaN 用 set_bad 显示为灰色；每个面板登记有限数据占比，供 S14 检查。"""
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BAND = (40.0, 50.0)


def _reg(ax, arr):
    a = np.asarray(arr, dtype=float)
    ax._fin = getattr(ax, '_fin', []) + [float(np.isfinite(a).mean()) if a.size else 0.0]


def line(ax, x, y, **kw):
    _reg(ax, y)
    ax.plot(x, y, **kw)


def scatter(ax, x, y, **kw):
    _reg(ax, y)
    ax.scatter(x, y, **kw)


def hist(ax, v, **kw):
    v = np.asarray(v)
    _reg(ax, v)
    ax.hist(v[np.isfinite(v)], **kw)


def bar(ax, names, vals, **kw):
    vals = [np.nan if v is None else v for v in vals]
    _reg(ax, vals)
    ax.bar(names, np.nan_to_num(vals), **kw)


def image(ax, M, **kw):
    M = np.asarray(M, dtype=float)
    _reg(ax, M)
    cm = copy.copy(plt.get_cmap(kw.pop('cmap', 'coolwarm')))
    cm.set_bad('lightgray')
    return ax.imshow(np.ma.masked_invalid(M), cmap=cm, aspect='auto', **kw)


def check_figure(fig):
    """每个面板：登记的有限数据占比最大值 >0 且渲染像素标准差 >0。"""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
    H = buf.shape[0]
    out = []
    for ax in fig.axes:
        if not hasattr(ax, '_fin'):
            continue           # colorbar 等辅助轴
        bb = ax.get_window_extent()
        x0, x1 = int(max(bb.x0, 0)), int(bb.x1)
        y0, y1 = int(max(H - bb.y1, 0)), int(H - bb.y0)
        patch = buf[y0:y1, x0:x1]
        out.append(dict(title=ax.get_title(), finite_fraction=max(ax._fin), pixel_std=float(patch.std()) if patch.size else 0.0))
    return out


def _az(l):
    return np.rad2deg(np.unwrap(np.arctan2(l[:, 1], l[:, 0])))


def _el(l):
    return np.rad2deg(np.arcsin(np.clip(l[:, 2], -1, 1)))


def figures(out, arm, FD):
    """FD 键：frames, res_mean(T,3), res_q10(T,3), res_q90(T,3), res_heat(G,T), support_count(T), dirs{flow:(T,3)}, refs{name:(T,3)}, rho(G,3),
    eta{all,lit,back}->[3], a_bin[list of (label,value)], hist{flow:dict(final_losses, amb_traj)}, rel_angle{flow:(T,)}, profile(grid,F,amb_star,als_ends)|None, back_scatter(x,y,(delta,g,rho_mean))|None"""
    checks = {}
    T = len(FD['frames'])
    fr = np.arange(1, T + 1)
    cols = ['tab:red', 'tab:green', 'tab:blue']
    # ---- 01 ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for c in range(3):
        line(ax[0], fr, FD['res_mean'][:, c], c=cols[c], label=f'mean ch{c}')
        ax[0].fill_between(fr, FD['res_q10'][:, c], FD['res_q90'][:, c], color=cols[c], alpha=.15)
    ax[0].axhline(0, c='k', lw=.6)
    ax[0].set_title(f'{arm}: per-frame residual (pred-obs), mean and q10-q90')
    ax[0].set_xlabel('frame')
    ax[0].legend()
    line(ax[1], fr, FD['support_count'])
    ax[1].set_title('support points per frame (w>0)')
    ax[1].set_xlabel('frame')
    im = image(ax[2], FD['res_heat'], vmin=-.2, vmax=.2)
    ax[2].set_title('channel-mean residual heatmap (gray = no obs)')
    ax[2].set_xlabel('frame')
    ax[2].set_ylabel('gaussian (sorted by mean nz)')
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout()
    checks['01_support_residual.png'] = check_figure(fig)
    fig.savefig(out / '01_support_residual.png', dpi=130)
    plt.close(fig)
    # ---- 02 ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k in ['full', 'A', 'B']:
        if k in FD['dirs']:
            line(ax[0], fr, _az(FD['dirs'][k]), label=f'{arm}/{k}')
            line(ax[1], fr, _el(FD['dirs'][k]), label=f'{arm}/{k}')
    for k, ls in FD['refs'].items():
        line(ax[0], fr, _az(ls), ls='--', label=f'{k} (frozen)')
        line(ax[1], fr, _el(ls), ls='--', label=f'{k} (frozen)')
    for v in BAND:
        ax[1].axhline(v, ls=':', c='k')
    ax[0].set_title('azimuth (unwrapped, deg)')
    ax[1].set_title('elevation (deg), dotted = 40/50 band')
    for a_ in ax:
        a_.legend(fontsize=7)
        a_.grid(alpha=.2)
        a_.set_xlabel('frame')
    fig.tight_layout()
    checks['02_trajectories.png'] = check_figure(fig)
    fig.savefig(out / '02_trajectories.png', dpi=130)
    plt.close(fig)
    # ---- 03 ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for c in range(3):
        hist(ax[0], FD['rho'][:, c], bins=50, range=(0, 1), alpha=.45, color=cols[c], label=f'ch{c}')
    ax[0].axvline(0, c='k', ls=':')
    ax[0].axvline(1, c='k', ls=':')
    ax[0].set_title('rho histogram (dotted = bounds 0 and 1)')
    ax[0].legend()
    w = .27
    for j, key in enumerate(['all', 'lit', 'back']):
        vals = [np.nan if v is None else v for v in FD['eta'][key]]
        _reg(ax[1], vals)
        ax[1].bar(np.arange(3) + (j - 1) * w, np.nan_to_num(vals), w, label=key)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(['R', 'G', 'B'])
    ax[1].set_title('ambient linear energy share (eta_lin)')
    ax[1].legend()
    if FD['a_bin']:
        bar(ax[2], [b[0] for b in FD['a_bin']], [b[1] for b in FD['a_bin']])
        ax[2].axhline(FD['amb'], c='k', ls='--')
        ax[2].set_title('local amb per mean-nz bin (dashed = global amb)')
    else:
        line(ax[2], [0, 1], [FD['amb']] * 2)
        ax[2].set_title(f'global amb = {FD["amb"]:.4g} (no bins)')
    fig.tight_layout()
    checks['03_rho_ambient.png'] = check_figure(fig)
    fig.savefig(out / '03_rho_ambient.png', dpi=130)
    plt.close(fig)
    # ---- 04 ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for k, v in FD['rel_angle'].items():
        line(ax[0], fr, v, label=k)
    ax[0].set_title('angle to full (deg)')
    ax[0].legend(fontsize=7)
    for k, h in FD['hist'].items():
        if k in ('full', 'hi', 'lo'):
            line(ax[1], np.arange(len(h['final_losses'])), h['final_losses'], label=k)
    ax[1].set_yscale('log')
    ax[1].set_title('final-stage loss trace')
    ax[1].legend()
    for k, h in FD['hist'].items():
        line(ax[2], np.arange(len(h['amb_traj'])), h['amb_traj'], label=k)
    ax[2].set_title('amb trajectory (A1: identically 0)')
    ax[2].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=.2)
    fig.tight_layout()
    checks['04_uncertainty_convergence.png'] = check_figure(fig)
    fig.savefig(out / '04_uncertainty_convergence.png', dpi=130)
    plt.close(fig)
    # ---- 05 / 06 （仅 A0）----
    if FD.get('profile') is not None:
        g, F, a_star, als = FD['profile']
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        line(ax[0], g, F)
        ax[0].axvline(a_star, c='r', ls='--', label=f'amb*={a_star:.4g}')
        for a0 in als:
            ax[0].axvline(a0, c='g', ls=':', alpha=.5)
        ax[0].legend()
        ax[0].set_title('profile F(amb) (green dotted = ALS endpoints)')
        ax[0].set_xlabel('amb')
        i = int(np.argmin(F))
        lo, hi = max(0, i - 200), min(len(g), i + 200)
        line(ax[1], g[lo:hi], F[lo:hi])
        ax[1].axvline(a_star, c='r', ls='--')
        ax[1].set_title('zoom around minimum')
        ax[1].set_xlabel('amb')
        fig.tight_layout()
        checks['05_ambient_profile.png'] = check_figure(fig)
        fig.savefig(out / '05_ambient_profile.png', dpi=130)
        plt.close(fig)
    if FD.get('back_scatter') is not None:
        x, y, fit = FD['back_scatter']
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
        scatter(ax, x, y, s=1, alpha=.2)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 20)
        line(ax, xs, fit[0] * fit[2] + fit[1] * xs, c='r', label=f'g_hat={fit[1]:.3g}, delta_hat={fit[0]:.3g}')
        ax.legend()
        ax.set_xlabel('rho * nz')
        ax.set_ylabel('y - rho*amb  (backlit)')
        ax.set_title('backlit residual vs rho*nz')
        fig.tight_layout()
        checks['06_backlit_residual_nz.png'] = check_figure(fig)
        fig.savefig(out / '06_backlit_residual_nz.png', dpi=130)
        plt.close(fig)
    return checks
