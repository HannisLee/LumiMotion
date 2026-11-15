"""P0(d) 补充：b/rhoY 含 rho_y=0 时的 Spearman 稳健性（主脚本 b/rho_y 触发除零警告）。只读 p0d 产物与T01 prior，无GT。
运行（仓库根）：python LIHAN/scripts/goal1113_t02/p0d_supplement.py <p0d输出目录>"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path.cwd().resolve(); D = Path(sys.argv[1]).resolve()
r = json.load(open(D/'p0d_result.json'))
prior = np.load(ROOT/'output/1113-01-LH-static-ELDL-V3-T01-BguidedEnvMap/inputs/prior.npz')
with np.load(D/'V_fit.npz') as z: V = z['V'].astype(np.float64); fi = z['body_fit']
with np.load(ROOT/'output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz') as z:
    w = z['weights'].astype(np.float64)[:, fi]; n = z['normals'].astype(np.float64)[:, fi]
rho, b = prior['rho_y'][fi], prior['b'][fi]
q = w/w.sum(0, keepdims=True)
th = (np.arange(8)+.5)*np.pi/8; ph = (np.arange(16)+.5)*2*np.pi/16
T, P_ = np.meshgrid(th, ph, indexing='ij')
om = np.stack((np.sin(T)*np.cos(P_), np.sin(T)*np.sin(P_), np.cos(T)), -1).reshape(-1, 3)
so = ((2*np.pi/16)*(np.cos(np.arange(8)*np.pi/8)-np.cos((np.arange(8)+1)*np.pi/8)))[:, None].repeat(16, 1).reshape(-1)
nl = np.maximum(np.einsum('tgc,jc->tgj', n, om), 0)*so
vbar = np.einsum('tg,tgj,tgj->g', q, nl, V)/np.einsum('tg,tgj->g', q, nl)
out = dict(n_fit=len(fi), n_rho_y_zero=int((rho == 0).sum()), n_rho_y_lt1e6=int((rho < 1e-6).sum()), n_b_zero=int((b == 0).sum()),
           n_b_neg=int((b < 0).sum()), n_nonfinite_b_over_rho=int((~np.isfinite(b/np.where(rho == 0, np.nan, rho))).sum()))
ok = rho > 1e-6
out['spearman_cos_primary_excluding_rho_y_le_1e-6'] = dict(zip(['rho', 'p'], map(float, spearmanr((b/rho)[ok], vbar[ok]))))
out['spearman_cos_primary_main_script_value'] = r['spearman_b_over_rhoY_vs_Vbar']['cos_weighted_primary']
out['spearman_b_vs_Vbar_all'] = dict(zip(['rho', 'p'], map(float, spearmanr(b, vbar))))
out['n_used_excluding'] = int(ok.sum())
(D/'p0d_supplement.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n'); print(json.dumps(out, ensure_ascii=False, indent=1))
