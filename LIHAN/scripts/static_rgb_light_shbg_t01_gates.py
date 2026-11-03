"""T01 候选门槛的纯数值判定。"""
import numpy as np
from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r5b as r5
from LIHAN.scripts.static_rgb_light_shbg_t01_scoring import bg_rms_ratio, correlations


def gate_candidate(ends, infos, templates, indices, n, y, w, relative, e3_score, e4_score,
                   ols_score, bg_score, e4_bg_score, bg_count, ols_count):
    keys=('full','A','B','boot103108','boot103109','hi','lo')
    if any(k not in ends or k not in infos or k not in templates for k in keys):
        return {'G0':False},None
    full=ends['full']; split=r3.compare(ends['A'],ends['B']); hilo=r3.compare(ends['hi'],ends['lo'])
    boot=np.mean([r3.compare(ends[k],full)['mean_deg'] for k in ('boot103108','boot103109')])
    el_mae=float(np.mean(np.abs(r5.elev_deg(ends['A'])-r5.elev_deg(ends['B']))))
    saturation=float((np.abs(r5.elev_deg(full)-45)>=4.75).mean())
    sign=r5.orbit_sign(full); reverse=int((sign*np.diff(np.unwrap(np.arctan2(full[:,1],full[:,0])))<0).sum())
    fi=indices['full']; yy=y[:,fi]; ww=w[:,fi]; rel=relative[:,fi]
    counts=(ww>0).sum(1)
    if (counts==0).any(): view=np.inf
    else:
        dark=((rel<.15)&(ww>0)).sum(1)/counts; lo=dark<=np.quantile(dark,.3); hi=dark>=np.quantile(dark,.7)
        view=float(abs(full[lo,2].mean()-full[hi,2].mean()))
    corr={}
    for key in keys:
        idx=indices[key]; nn=n[:,idx] if n.ndim==3 else np.broadcast_to(n[idx],(len(y),len(idx),3))
        corr[key]=correlations(nn,ends[key],templates[key]['q'],w[:,idx],templates[key]['valid'])
    gamma=infos['full']['gamma']; ratio=bg_rms_ratio(templates['full']['q'],gamma,yy,ww)
    gates={
        'G0':all(infos[k]['converged'] and templates[k]['nondegenerate_fraction']>=.2 for k in keys),
        'G1':split['mean_deg']<=3.5 and split['p90_deg']<=6,
        'G2':hilo['mean_deg']<=.5 and abs(infos['hi']['gamma']-infos['lo']['gamma'])<=.05,
        'G3':boot<=2,
        'G4a':ols_score<=1.015*e3_score and ols_score<=e4_score,
        'G4b':np.isfinite(bg_score) and bg_score<=.995*e4_bg_score and bg_count>=.9*ols_count,
        'G5':saturation<=.2,
        'G6':reverse<=2,
        'G7':view<=.03,
        'G8':all(max(corr[k])<=.2 for k in keys) and ratio<=.25,
    }
    metrics=dict(split_mean_deg=split['mean_deg'],split_p90_deg=split['p90_deg'],split_el_mae_deg=el_mae,
                 hilo_mean_deg=hilo['mean_deg'],bootstrap_mean_deg=float(boot),saturation=saturation,reverse_steps=reverse,
                 view_dz=view,correlations=corr,bg_rms_ratio=ratio,G4a_mae=float(ols_score),G4b_mae=float(bg_score),
                 J=split['mean_deg']+.5*split['p90_deg']+el_mae+boot+hilo['mean_deg'])
    return {k:bool(v) for k,v in gates.items()},metrics
