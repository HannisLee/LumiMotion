"""T01 拟合阶段：当前仍在开发，不得用于真实实验。"""
from pathlib import Path
import json
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t01_templates import load_support,template,save_template
from LIHAN.scripts.static_rgb_light_shbg_t01_optimizer import fit

CANDIDATES=(('C01',1,1e-4),('C02',1,1e-3),('C03',2,1e-4),('C04',2,1e-3))
RUNS=('full','A','B','boot103108','boot103109','hi','lo')


def fit_candidates(observations,baseline,sh_values,extraction_metadata,output,smoke=False):
    """拟合各候选；不计算选择。尚未完成完整门槛与冻结链。"""
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    indices,val,anchors=load_support(baseline,observations)
    with np.load(observations,allow_pickle=False) as d:
        n=d['normals'].astype(np.float64); y=d['intensity'].astype(np.float64); w=d['weights'].astype(np.float64); ids=d['gaussian_ids'].copy(); frames=d['frames'].copy()
    with np.load(sh_values,allow_pickle=False) as s:
        if not np.array_equal(ids,s['gaussian_ids']) or not np.array_equal(frames,s['frames']): raise ValueError('SH 与观察数据未对齐')
        h={L:s[f'h_L{L}'].copy() for L in (0,1,2)}
    meta=json.loads(Path(extraction_metadata).read_text()); diag=meta['diagnostics']
    l0=template(h[0],n,y,w,indices['full'],anchors['full'])
    if np.abs(l0['q']).max()>1e-8: raise ArithmeticError('L0 退化自检失败')
    results={}; cache={}
    for name,L,lam in (CANDIDATES[:1] if smoke else CANDIDATES):
        results[name]={}; failures=[]
        if diag[str(L)]['invalid_clipping']:
            results[name]['failure']='SH_CLIPPING'; continue
        for key in (('full',) if smoke else RUNS):
            tag=(L,key)
            if tag not in cache:
                cache[tag]=template(h[L],n,y,w,indices[key],anchors[key]); rec=cache[tag]
                save_template(out/'templates'/f'L{L}_{key}.npz',h[L],indices[key],ids,frames,rec)
            rec=cache[tag]
            if rec['nondegenerate_fraction']<.2:
                results[name][key]={'converged':False,'reason':'SH_DEGENERATE'}; failures.append(key); continue
            idx=indices[key]; anchor=anchors[key]
            a=np.arctanh(np.clip((np.arcsin(np.clip(anchor[:,2],-1,1))-np.deg2rad(45))/np.deg2rad(5),-.999999,.999999))
            g=.5
            if key=='hi': a=np.full(len(a),np.arctanh(.9)); g=.9
            if key=='lo': a=np.full(len(a),np.arctanh(-.9)); g=.1
            nn=n[:,idx] if n.ndim==3 else np.broadcast_to(n[idx],(len(y),len(idx),3))
            try:
                cert=fit(nn,y[:,idx],w[:,idx],rec['H'],anchor,g,a,lam,out/'runs'/name/key)
            except Exception as exc:
                cert={'converged':False,'reason':'INITIALIZATION_ERROR','exception':repr(exc)}
            results[name][key]=cert
            if not cert['converged']: failures.append(key)
            (out/'progress.json').write_text(json.dumps(results,indent=2))
        results[name]['failure']=None if not failures else failures
    (out/'fit_results.json').write_text(json.dumps(results,indent=2))
    return results
