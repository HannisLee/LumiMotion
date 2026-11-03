"""T01 共同评分掩码和候选选择组件（尚未完成独立静态审查）。"""
from pathlib import Path
import json
import numpy as np
from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts.static_rgb_light_shbg_t01_templates import load_support,template,save_template
from LIHAN.scripts.static_rgb_light_shbg_t01_scoring import locked_mask,heldout_score,select,x_for
from LIHAN.scripts.static_rgb_light_shbg_t01_gates import gate_candidate
from LIHAN.scripts.static_rgb_light_shbg_t01_fit import CANDIDATES,RUNS


def ols_mask(n,y,w,directions):
    valid=np.isfinite(y)&(w>0); cal=valid.copy(); cal[1::2]=False
    ww=np.where(cal,w,0); s=ww.sum(0); mask=cal.sum(0)>=3
    for l in directions:
        x=x_for(n,l); mean=(ww*x).sum(0)/np.maximum(s,1e-30)
        mask &= (ww*(x-mean)**2).sum(0)>1e-12
    return mask


def choose(observations,baseline,sh_values,fit_output,output):
    dest=Path(output); dest.mkdir(parents=True,exist_ok=False); fit_output=Path(fit_output)
    indices,val,anchors=load_support(baseline,observations)
    with np.load(observations,allow_pickle=False) as d:
        n=d['normals'].astype(np.float64); y=d['intensity'].astype(np.float64); w=d['weights'].astype(np.float64); ids=d['gaussian_ids'].copy(); frames=d['frames'].copy()
    with np.load(sh_values,allow_pickle=False) as s:
        if not np.array_equal(ids,s['gaussian_ids']) or not np.array_equal(frames,s['frames']): raise ValueError('SH 与观察数据未对齐')
        h={L:s[f'h_L{L}'].copy() for L in (1,2)}
    e3=np.load(Path(baseline)/'fitting'/'runs'/'E3'/'direction.npy',allow_pickle=False); e4=anchors['full']
    raw=json.loads((fit_output/'fit_results.json').read_text()); dirs={}; infos={}
    for name,L,lam in CANDIDATES:
        dirs[name]={}; infos[name]={}
        for key in RUNS:
            run=fit_output/'runs'/name/key
            if (run/'direction.npy').exists() and (run/'parameters.npz').exists():
                dirs[name][key]=np.load(run/'direction.npy',allow_pickle=False)
                with np.load(run/'parameters.npz',allow_pickle=False) as p: gamma=float(p['gamma'])
                infos[name][key]={'converged':bool(raw[name][key]['converged']),'gamma':gamma}
    nv=n[:,val] if n.ndim==3 else np.broadcast_to(n[val],(len(y),len(val),3)); yv=y[:,val]; wv=w[:,val]
    eligible_dirs=[dirs[name]['full'] for name,_,_ in CANDIDATES if 'full' in dirs[name]]
    common=ols_mask(nv,yv,wv,[e3,e4]+eligible_dirs)
    if not common.any(): raise ValueError('共同 OLS 留出掩码为空')
    def ols(l): return r3.ols_score(nv[:,common],yv[:,common],wv[:,common],l)
    e3s=ols(e3); e4s=ols(e4); valid=np.isfinite(yv)&(wv>0); ols_count=int((valid[1::2]&common).sum())
    locked=locked_mask(nv,yv,wv,e4); e4bg,count=heldout_score(nv,yv,wv,e4,locked)
    np.savez_compressed(dest/'score_masks.npz',gaussian_ids=ids[val],ols_common=common,kkt_locked=locked)
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning); q20=np.nanquantile(y,.2,axis=0); q95=np.nanquantile(y,.95,axis=0)
    relative=np.clip((y-q20)/(q95-q20+.001),0,1.5); relative[:,~((np.isfinite(y).sum(0)>=8)&(q95-q20>.015))]=np.nan
    rows=[]
    for name,L,lam in CANDIDATES:
        row=dict(name=name,L=L,lambda_bg=lam,all_gates=False,reason=raw[name].get('failure'))
        if all(k in dirs[name] for k in RUNS):
            recs={}
            for key in RUNS:
                idx=indices[key]
                recs[key]=template(h[L],n,y,w,idx,anchors[key])
            hold=template(h[L],n,y,w,val,e4,holdout=True)
            if not (dest/f'L{L}_holdout.npz').exists(): save_template(dest/f'L{L}_holdout.npz',h[L],val,ids,frames,hold,L)
            bs,bc=heldout_score(nv,yv,wv,dirs[name]['full'],locked,hold['H'],infos[name]['full']['gamma'])
            gates,metrics=gate_candidate(dirs[name],infos[name],recs,indices,n,y,w,relative,e3s,e4s,ols(dirs[name]['full']),bs,e4bg,bc,ols_count)
            row.update(metrics or {}); row['gates']=gates; row['all_gates']=all(gates.values())
        rows.append(row)
    chosen,status=select(rows); direction=e4 if chosen is None else dirs[chosen['name']]['full']
    np.savez_compressed(dest/'predictions.npz',frames=frames,E4=e4,selected=direction)
    result=dict(status=status,selected=None if chosen is None else chosen['name'],baseline={'E3_OLS':e3s,'E4_OLS':e4s,'E4_KKT':e4bg,'OLS_count':ols_count,'E4_KKT_count':count},candidates=rows,gt_read=False)
    (dest/'candidate_table.json').write_text(json.dumps(result,indent=2)); return result
