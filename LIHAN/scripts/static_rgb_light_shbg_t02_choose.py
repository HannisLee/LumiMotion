"""T02 固定留出掩码、门槛和无 GT 选择。"""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts import static_rgb_light_fit_r5b as r5
from LIHAN.scripts.static_rgb_light_shbg_t01_choose import ols_fixed, json_safe
from LIHAN.scripts.static_rgb_light_shbg_t01_scoring import select
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters, design, final_eta
from LIHAN.scripts.static_rgb_light_shbg_t02_templates import load_support, template, save_template
from LIHAN.scripts.static_rgb_light_shbg_t02_fit import CANDIDATES, RUNS


def calibration_mask(n,y,w,direction):
    x=np.maximum(np.einsum('tgc,tc->tg',n[::2],direction[::2]),0)
    yy=y[::2]; ww=w[::2]; valid=np.isfinite(yy)&np.isfinite(ww)&(ww>0)
    weights=np.where(valid,ww,0.); mass=weights.sum(0); mean=(weights*x).sum(0)/np.maximum(mass,1e-30)
    variance=(weights*(x-mean)**2).sum(0)/np.maximum(mass,1e-30)
    return (valid.sum(0)>=24)&(variance>1e-12)


def heldout_score(n,y,w,direction,mask,H=None,gamma=0.):
    x=np.maximum(np.einsum('tgc,tc->tg',n,direction),0)
    H=np.zeros_like(x) if H is None else H
    odd_y,odd_w=y[::2],w[::2]
    valid=np.isfinite(odd_y)&np.isfinite(odd_w)&(odd_w>0)&mask[None,:]
    weights=np.where(valid,odd_w,0.); target=np.where(valid,odd_y,0.)-gamma*H[::2]
    if not np.isfinite(x).all() or not np.isfinite(H).all(): return np.inf,0
    with torch.no_grad():
        rho,b=r3.kkt(torch.tensor(x[::2]),torch.tensor(target),torch.tensor(weights)); rho=rho.numpy(); b=b.numpy()
    if not np.isfinite(rho).all() or not np.isfinite(b).all(): return np.inf,0
    even_y,even_w=y[1::2],w[1::2]; test=np.isfinite(even_y)&np.isfinite(even_w)&(even_w>0)&mask[None,:]
    scoreweights=np.where(test,even_w,0.); count=int(test.sum())
    if scoreweights.sum()<=0 or np.any(scoreweights.sum(1)<=0): return np.inf,count
    error=np.minimum(np.abs(x[1::2]*rho+b+gamma*H[1::2]-np.where(test,even_y,0.)),.2)
    return float((scoreweights*error).sum()/scoreweights.sum()),count


def read_data(observations, baseline):
    indices, val, anchors = load_support(baseline, observations)
    with np.load(observations, allow_pickle=False) as d:
        n, y, w = (d[k].astype(np.float64) for k in ('normals','intensity','weights'))
        ids, frames = d['gaussian_ids'].copy(), d['frames'].copy()
    nv = n[:,val] if n.ndim == 3 else np.broadcast_to(n[val], (150,len(val),3))
    return indices, val, anchors, n, y, w, ids, frames, nv


def baseline_scores(observations, baseline, output):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    _, val, anchors, n, y, w, ids, frames, nv = read_data(observations,baseline)
    yy, ww = y[:,val], w[:,val]; e4=anchors['full']
    e3=np.load(Path(baseline)/'fitting'/'runs'/'E3'/'direction.npy',allow_pickle=False)
    common=calibration_mask(nv,yy,ww,e4)
    even_valid=np.isfinite(yy[1::2])&np.isfinite(ww[1::2])&(ww[1::2]>0)
    count=int((even_valid&common).sum())
    e3s=ols_fixed(nv,yy,ww,e3,common); e4s=ols_fixed(nv,yy,ww,e4,common)
    e4bg,bgcount=heldout_score(nv,yy,ww,e4,common)
    np.savez_compressed(out/'score_masks.npz',gaussian_ids=ids[val],frames=frames,common=common,score_valid=even_valid&common)
    rec=dict(E3_OLS=e3s,E4_OLS=e4s,E4_KKT=e4bg,OLS_count=count,E4_KKT_count=bgcount,gt_read=False)
    (out/'baseline_scores.json').write_text(json.dumps(json_safe(rec),indent=2,allow_nan=False))
    return rec


def eta_record(n, direction, a, rec, w):
    phi=np.unwrap(np.arctan2(direction[:,1],direction[:,0]))
    d=design(n,phi,a)
    eta=final_eta(rec['q'],d,w,rec['cal'])
    eligible=rec['eligible']; values=eta[eligible]; mass=np.where(rec['cal'],w,0).sum(0)[eligible]
    if len(values):
        order=np.argsort(values); v=values[order]; weights=mass[order]; cum=np.cumsum(weights)
        quant=lambda q: float(v[min(np.searchsorted(cum,q*cum[-1]),len(v)-1)])
        summary=dict(p50=quant(.5),p90=quant(.9),max=float(v.max()),exceeded=int((v>.2).sum()))
    else: summary=dict(p50=None,p90=None,max=None,exceeded=0)
    return eta,summary


def bg_statistics(q,gamma,y,w):
    valid=np.isfinite(y)&np.isfinite(w)&(w>0); ww=np.where(valid,w,0.); yy=np.where(valid,y,0.)
    mass=ww.sum(0); bg=gamma*q
    ym=(ww*yy).sum(0)/np.maximum(mass,1e-30); bm=(ww*bg).sum(0)/np.maximum(mass,1e-30)
    num=(ww*(bg-bm)**2).sum(); den=(ww*(yy-ym)**2).sum(); total=ww.sum()
    rms=float(np.sqrt(num/total)) if total else 0.
    ratio=float(np.sqrt(num/den)) if den>0 else (0. if num==0 else np.inf)
    return rms,ratio


def choose(observations,baseline,sh_values,fit_output,output):
    dest=Path(output); dest.mkdir(parents=True,exist_ok=False); fit_output=Path(fit_output)
    indices,val,anchors,n,y,w,ids,frames,nv=read_data(observations,baseline)
    with np.load(sh_values,allow_pickle=False) as s:
        if not np.array_equal(ids,s['gaussian_ids']) or not np.array_equal(frames,s['frames']): raise ValueError('SH 与观测错位')
        h={L:s[f'h_L{L}'].copy() for L in (1,2)}
    with np.load(fit_output.parent/'baseline_scoring'/'score_masks.npz',allow_pickle=False) as m:
        if not np.array_equal(m['gaussian_ids'],ids[val]): raise ValueError('评分点错位')
        common=m['common'].copy()
    scores=json.loads((fit_output.parent/'baseline_scoring'/'baseline_scores.json').read_text())
    scores={k:(np.inf if v is None else v) for k,v in scores.items()}
    raw=json.loads((fit_output/'fit_results.json').read_text())
    dirs={}; infos={}; params={}
    for name in ['C00L']+[x[0] for x in CANDIDATES]:
        dirs[name]={};infos[name]={};params[name]={}
        for key in RUNS:
            run=fit_output/'runs'/name/key
            if (run/'direction.npy').is_file() and (run/'parameters.npz').is_file():
                dirs[name][key]=np.load(run/'direction.npy',allow_pickle=False)
                with np.load(run/'parameters.npz',allow_pickle=False) as p: params[name][key]={k:p[k].copy() for k in p.files}
                infos[name][key]=raw[name][key]
    control_ok=all(infos['C00L'].get(k,{}).get('converged',False) for k in RUNS)
    yy=y[:,val]; ww=w[:,val]; ols=lambda l: ols_fixed(nv,yy,ww,l,common)
    if 'full' in dirs['C00L']:
        c_ols=ols(dirs['C00L']['full']); c_bg,_=heldout_score(nv,yy,ww,dirs['C00L']['full'],common)
    else: c_ols=c_bg=np.inf
    with np.errstate(invalid='ignore',divide='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning); q20=np.nanquantile(y,.2,axis=0); q95=np.nanquantile(y,.95,axis=0)
        relative=np.clip((y-q20)/(q95-q20+.001),0,1.5); relative[:,~((np.isfinite(y).sum(0)>=8)&(q95-q20>.015))]=np.nan
    rows=[]
    for name,L,lam in CANDIDATES:
        row=dict(name=name,L=L,lambda_bg=lam,all_gates=False,reason=raw[name].get('failure'),gates={'G0':False})
        if all(k in dirs[name] for k in RUNS):
            full=dirs[name]['full']; split=r3.compare(dirs[name]['A'],dirs[name]['B']); hilo=r3.compare(dirs[name]['hi'],dirs[name]['lo'])
            boot=float(np.mean([r3.compare(dirs[name][k],full)['mean_deg'] for k in ('boot103108','boot103109')]))
            elmae=float(np.mean(np.abs(r5.elev_deg(dirs[name]['A'])-r5.elev_deg(dirs[name]['B']))))
            recs={}; eta={}; summaries={}; template_ok=True
            for key in RUNS:
                idx=indices[key]; rec=template(h[L],n,y,w,idx,anchors[key]); recs[key]=rec
                template_ok &= rec['nondegenerate_fraction']>=.2
                nn=n[:,idx] if n.ndim==3 else np.broadcast_to(n[idx],(150,len(idx),3))
                eta[key],summaries[key]=eta_record(nn,dirs[name][key],params[name][key]['a'],rec,w[:,idx])
            hold=template(h[L],n,y,w,val,anchors['full'],True)
            if not (dest/f'L{L}_holdout.npz').exists(): save_template(dest/f'L{L}_holdout.npz',h[L],val,ids,frames,hold,L)
            eta['val'],summaries['val']=eta_record(nv,full,params[name]['full']['a'],hold,ww)
            np.savez_compressed(dest/f'{name}_eta.npz',**eta)
            gamma=float(params[name]['full']['gamma']); bs,bc=heldout_score(nv,yy,ww,full,common,hold['H'],gamma); os=ols(full)
            rms,ratio=bg_statistics(recs['full']['q'],gamma,y[:,indices['full']],w[:,indices['full']])
            elev=r5.elev_deg(full); saturation=float((np.abs(elev-45)>=4.75).mean())
            reverse=int((infos[name]['full']['orbit_sign']*np.diff(np.unwrap(np.arctan2(full[:,1],full[:,0])))<0).sum())
            fi=indices['full']; cnt=(w[:,fi]>0).sum(1)
            if (cnt==0).any(): view=np.inf
            else:
                dark=((relative[:,fi]<.15)&(w[:,fi]>0)).sum(1)/cnt; low=dark<=np.quantile(dark,.3); high=dark>=np.quantile(dark,.7)
                view=float(abs(full[low,2].mean()-full[high,2].mean()))
            bases=[scores['E3_OLS'],scores['E4_OLS'],scores['E4_KKT'],c_ols,c_bg]
            g4valid=bool(common.any() and scores['OLS_count']>0 and np.isfinite(bases).all() and bc>=.9*scores['OLS_count'])
            gates=dict(G0=control_ok and template_ok and all(infos[name][k]['converged'] for k in RUNS),G1=split['mean_deg']<=3.5 and split['p90_deg']<=6,G2=hilo['mean_deg']<=.5 and abs(float(params[name]['hi']['gamma'])-float(params[name]['lo']['gamma']))<=.05,G3=boot<=2,G4a=g4valid and np.isfinite(os) and os<=1.015*scores['E3_OLS'] and os<=scores['E4_OLS'] and os<=c_ols,G4b=g4valid and np.isfinite(bs) and bs<=.995*scores['E4_KKT'] and bs<=.995*c_bg,G5=saturation<=.2,G6=reverse<=2,G7=view<=.03,G8=all(v['max'] is not None and v['max']<=.2 for v in summaries.values()) and ratio<=.25 and rms>=1e-5)
            row.update(gates={k:bool(v) for k,v in gates.items()},all_gates=all(gates.values()),G4a_mae=os,G4b_mae=bs,bg_rms=rms,bg_rms_ratio=ratio,eta=summaries,split_mean_deg=split['mean_deg'],split_p90_deg=split['p90_deg'],split_elevation_mae_deg=elmae,bootstrap_mean_deg=boot,hilo_mean_deg=hilo['mean_deg'],saturation=saturation,reverse_steps=reverse,view_dz=view,J=split['mean_deg']+.5*split['p90_deg']+elmae+boot+hilo['mean_deg'])
        rows.append(row)
    selected,status=select(rows)
    if selected is None:
        source=Path(baseline)/'fitting'/'runs'/'a1e+00_e1e+01_full'/'direction.npy'; shutil.copyfile(source,dest/'direction.npy')
        from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest
        if digest(source)!=digest(dest/'direction.npy'): raise ArithmeticError('回退方向文件字节不一致')
    else: np.save(dest/'direction.npy',dirs[selected['name']]['full'])
    result=dict(status=status,selected=None if selected is None else selected['name'],baseline=scores,C00L_OLS=c_ols,C00L_KKT=c_bg,candidates=rows,gt_read=False)
    (dest/'candidate_table.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False))
    return result
