#!/usr/bin/env python3
"""T02 只读无 GT 核算；不调用拟合、选档或评测入口。"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/home/han.li/reproduce/LumiMotion-stage2-perlight')
R=ROOT/'output/1107-01-LH-static-ELDL-V3-T01仰角正则消融'
PLAN=ROOT/'LIHAN/DOC/规划/1107-02-T02无GT选档准则与仰角精度诊断规划.md'
NAMES=[f'C{i:02d}' for i in range(7)]
KEYS=['split_mean_deg','split_p90_deg','split_el_mae_deg','boot_mean_deg','hilo_mean_deg']
WEIGHT=np.array([1.,.5,1.,1.,1.])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def angle(a,b): return np.rad2deg(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=1),-1,1)))
def elevation(a): return np.rad2deg(np.arcsin(np.clip(a[:,2],-1,1)))
def score(z):
    split=angle(z['A'],z['B']); hi=angle(z['hi'],z['lo']); el=np.abs(elevation(z['A'])-elevation(z['B'])).mean()
    boot=[float(angle(z[k],z['full']).mean()) for k in ('boot103108','boot103109')]
    vals=np.array([split.mean(),np.quantile(split,.9),el,np.mean(boot),hi.mean()]); return vals,split,hi,boot

def selftest():
    v=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    assert np.allclose(angle(v,2*v),0,atol=1e-6)
    assert np.allclose(angle(v,np.roll(v,1,axis=0)),90)
    assert np.isclose(np.mean([0,30,60]),30) and np.isclose(np.quantile([0,30,60],.9),54)
    t=np.array([[1.,0.,0.],[0.,1.,0.]]); z={k:t for k in ('full','A','B','hi','lo','boot103108','boot103109')}
    values,*_=score(z); assert np.allclose(values,0,atol=1e-6)
    a=np.deg2rad(10); rot=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    b=np.array([[0.,0.,1.],[1.,0.,0.]]); assert np.allclose(angle(t,b),angle(t@rot.T,b@rot.T),atol=1e-6)
    assert np.isclose(np.sum(values*WEIGHT),np.dot(values,WEIGHT))
    return {'same_trajectory':values.tolist(),'rotation_10_deg_before':angle(t,b).tolist(),'rotation_10_deg_after':angle(t@rot.T,b@rot.T).tolist(),'status':'PASS'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.self_test:
        (args.output/'synthetic.json').write_text(json.dumps(selftest(),indent=2,ensure_ascii=False)+'\n');print('synthetic PASS');return
    manifest=json.loads((R/'freeze_manifest.json').read_text()); frozen={x['path']:x['sha256'] for x in manifest['files']}
    rel=['fitting/candidate_table.json']+[f'fitting/runs/{n}_ends.npz' for n in NAMES]
    for n in NAMES:
        for flow in ('hi','lo'):
            rel += [f'fitting/runs/{n}_{flow}/certificate.json',f'fitting/runs/{n}_{flow}/config.json']
    evidence={}
    for item in rel:
        pth=R/item; actual=sha(pth); expected=frozen.get(item); evidence[item]={'sha256':actual,'frozen_sha256':expected,'matches':actual==expected}
    for pth in [PLAN,Path(__file__),ROOT/'LIHAN/scripts/static_rgb_light_fit_goal1107_t01.py',ROOT/'LIHAN/scripts/static_rgb_light_fit_r3.py',ROOT/'LIHAN/scripts/static_rgb_light_fit_r5b.py',R/'freeze_manifest.json']:
        evidence[str(pth)]={'sha256':sha(pth)}
    (args.output/'input_hashes.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
    if not all(evidence[x]['matches'] for x in rel): raise RuntimeError('冻结输入缺失或哈希不符')
    rows=json.loads((R/'fitting/candidate_table.json').read_text())['candidates']; out=[]
    for row in rows:
        n=row['name']; z=np.load(R/f'fitting/runs/{n}_ends.npz'); vals,sp,hi,boot=score(z)
        errors={k:float(vals[i]-row[k]) for i,k in enumerate(KEYS)}; errj=float(np.dot(vals,WEIGHT)-row['J'])
        if max(abs(v) for v in [*errors.values(),errj])>1e-8: raise RuntimeError(f'{n} 与存档不符: {errors}, J={errj}')
        elev=elevation(z['hi'])-elevation(z['lo']); az=np.rad2deg(np.angle(np.exp(1j*(np.arctan2(z['hi'][:,1],z['hi'][:,0])-np.arctan2(z['lo'][:,1],z['lo'][:,0])))))
        d={'name':n,'G2':{'mean_deg':float(hi.mean()),'p50_deg':float(np.median(hi)),'p90_deg':float(np.quantile(hi,.9)),'max_deg':float(hi.max()),'threshold_deg':.5,'signed_margin_deg':float(hi.mean()-.5),'ratio':float(hi.mean()/.5),'over_frame_fraction':float((hi>.5).mean()),'elevation_abs_mean_deg':float(np.abs(elev).mean()),'elevation_signed_mean_deg':float(elev.mean()),'azimuth_wrap_abs_mean_deg':float(np.abs(az).mean()),'historical_gate':row['gates']['G2'],'recomputed_gate':bool(hi.mean()<=.5),'hi_certificate':row['certificates']['hi'],'lo_certificate':row['certificates']['lo']},'J':{'terms':dict(zip(KEYS,vals.tolist())),'weighted_contributions':dict(zip(KEYS,(vals*WEIGHT).tolist())),'J':float(np.dot(vals,WEIGHT)),'boot_each_mean_deg':boot,'archive_term_errors':errors,'archive_J_error':errj},'historical_gate_values':{k:row[k] for k in ('converged','split_mean_deg','split_p90_deg','hilo_mean_deg','boot_mean_deg','body_heldout_mae','E3_body_heldout_mae','saturation_frac','reversal_count','view_dz_abs','gates','data_loss','full_penalties')}}
        if d['G2']['historical_gate']!=d['G2']['recomputed_gate']: raise RuntimeError(f'{n} G2 不一致')
        out.append(d)
    base=np.array(list(out[0]['J']['weighted_contributions'].values()))
    for i in (1,6):
        x=out[i]; delta=np.array(list(x['J']['weighted_contributions'].values()))-base; dj=x['J']['J']-out[0]['J']['J']; assert abs(delta.sum()-dj)<=1e-8
        x['J']['delta_weighted_vs_C00']=dict(zip(KEYS,delta.tolist()));x['J']['delta_J_vs_C00']=float(dj);x['J']['improvement_contribution_percent']=(dict(zip(KEYS,(-100*delta/(-dj)).tolist())) if abs(dj)>1e-12 else None)
    (args.output/'diagnostics.json').write_text(json.dumps({'candidates':out,'status':'PASS'},indent=2,ensure_ascii=False)+'\n');print('无 GT 重算 PASS；7 个候选，归档差 ≤ 1e-8°')
if __name__=='__main__': main()
