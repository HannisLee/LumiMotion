"""T01 无 GT 留出评分、后验门槛和固定选择规则。"""
import numpy as np
import torch
from LIHAN.scripts.static_rgb_light_fit_r3 import kkt
from LIHAN.scripts.static_rgb_light_shbg_t01_core import design


def x_for(n,l):
    return np.maximum(np.einsum('tgc,tc->tg',n,l),0)


def locked_mask(n,y,w,e4):
    x=x_for(n,e4); valid=np.isfinite(y)&(w>0); cal=valid.copy(); cal[1::2]=False
    weights=np.where(cal,w,0); sw=weights.sum(0); mean=(weights*x).sum(0)/np.maximum(sw,1e-30)
    variance=(weights*(x-mean)**2).sum(0)/np.maximum(sw,1e-30)
    return (cal.sum(0)>=8)&(variance>1e-12)


def heldout_score(n,y,w,l,mask,H=None,gamma=0):
    x=x_for(n,l); valid=np.isfinite(y)&(w>0)&mask[None,:]; cal=valid.copy(); cal[1::2]=False
    test=valid.copy(); test[::2]=False
    ww=np.where(cal,w,0); yy=np.nan_to_num(y); H=np.zeros_like(x) if H is None else H
    with torch.no_grad():
        rho,b=kkt(torch.tensor(x),torch.tensor(yy-gamma*H),torch.tensor(ww)); rho=rho.numpy(); b=b.numpy()
    scoreweights=np.where(test,w,0)
    if scoreweights.sum()<=0 or np.any(scoreweights[1::2].sum(1)<=0): return np.inf,int(test.sum())
    err=np.minimum(np.abs(x*rho+b+gamma*H-yy),.2)
    return float((scoreweights*err).sum()/scoreweights.sum()),int(test.sum())


def weighted_p90(values,weights):
    if not len(values): return 0.
    v=np.asarray(values); w=np.asarray(weights); order=np.argsort(v); v=v[order]; w=w[order]
    return float(v[np.searchsorted(np.cumsum(w),.9*w.sum(),side='left')])


def correlations(n,l,q,w,valid):
    d=design(n,l); results=[]
    for k in (1,2,3):
        vals=[]; masses=[]
        for g in range(q.shape[1]):
            m=valid[:,g] & (w[:,g]>0); wt=w[m,g]; total=wt.sum()
            if total<=0: continue
            a=q[m,g]; b=d[m,g,k]; ac=a-(wt*a).sum()/total; bc=b-(wt*b).sum()/total
            va=(wt*ac*ac).sum(); vb=(wt*bc*bc).sum()
            if va<=0 or vb<=0: continue
            vals.append(abs((wt*ac*bc).sum()/np.sqrt(va*vb))); masses.append(total)
        results.append(weighted_p90(vals,masses))
    return results


def bg_rms_ratio(q,gamma,y,w):
    valid=np.isfinite(y)&(w>0); ww=np.where(valid,w,0); yy=np.nan_to_num(y); s=ww.sum(0)
    means=(ww*yy).sum(0)/np.maximum(s,1e-30)
    num=(ww*(gamma*q)**2).sum(); den=(ww*(yy-means)**2).sum()
    return float(np.sqrt(num/den)) if den>0 else (0. if num==0 else np.inf)


def select(rows):
    passed=[r for r in rows if r['all_gates']]
    if not passed: return None, 'FALLBACK_E4'
    best=min(r['J'] for r in passed)
    tied=[r for r in passed if r['J']<=best+.02]
    return min(tied,key=lambda r:(r['G4b_mae'],r['G4a_mae'],r['L'],-r['lambda_bg'],r['name'])), 'SELECTED_SH'
