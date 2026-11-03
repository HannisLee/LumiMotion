"""T02 合成集成检查：假拟合只产生可审查的合成结果，不访问真实数据。"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t01_extract import digest
from LIHAN.scripts import static_rgb_light_shbg_t02_fit as fitter
from LIHAN.scripts import static_rgb_light_shbg_t02_choose as chooser
from LIHAN.scripts import static_rgb_light_shbg_t02_figures as drawer
from LIHAN.scripts.static_rgb_light_shbg_t02_freeze import freeze
from LIHAN.scripts.static_rgb_light_shbg_t02_core import anchor_parameters


def main():
    rng=np.random.default_rng(919); T=150; G=12; ids=np.arange(G); frames=np.arange(1,T+1)
    t=np.arange(T)*2*np.pi/T; anchor=np.stack((np.cos(t)/np.sqrt(2),np.sin(t)/np.sqrt(2),np.full(T,1/np.sqrt(2))),axis=1)
    normals=rng.normal(size=(T,G,3)); normals/=np.linalg.norm(normals,axis=2,keepdims=True)
    x=np.maximum(np.einsum('tgc,tc->tg',normals,anchor),0); y=.2+.7*x; w=np.ones_like(y)
    full=np.arange(8); val=np.arange(8,12); indices=dict(full=full,A=full[:4],B=full[4:],boot103108=full,boot103109=full,hi=full,lo=full); anchors={k:anchor for k in indices}
    def support(*unused): return indices,val,anchors
    def fake_fit(n,y,w,H,l,lam,dest,run='full',control=False):
        dest=Path(dest); dest.mkdir(parents=True,exist_ok=False); phi,a=anchor_parameters(l); np.save(dest/'direction.npy',l)
        np.savez_compressed(dest/'parameters.npz',U=np.zeros(5),V=np.zeros(5),phi=phi,a=a,gamma=0. if control else .5,rho=np.ones(y.shape[1]),b=np.zeros(y.shape[1]))
        cert=dict(converged=False,reason='NOT_CONVERGED',orbit_sign=1.); (dest/'certificate.json').write_text(json.dumps(cert)); (dest/'history.json').write_text('[]'); (dest/'checkpoint.pt').write_bytes(b'synthetic')
        return cert
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); baseline=root/'baseline'; run=baseline/'fitting/runs/a1e+00_e1e+01_full'; run.mkdir(parents=True); np.save(run/'direction.npy',anchor)
        e3=baseline/'fitting/runs/E3'; e3.mkdir(parents=True); np.save(e3/'direction.npy',anchor)
        obs=root/'observations.npz'; np.savez_compressed(obs,normals=normals,intensity=y,weights=w,gaussian_ids=ids,frames=frames)
        sh=root/'sh.npz'; np.savez_compressed(sh,gaussian_ids=ids,frames=frames,h_L0=np.full((T,G),.3),h_L1=rng.normal(size=(T,G)),h_L2=rng.normal(size=(T,G)))
        meta=root/'extract.json'; meta.write_text(json.dumps({'diagnostics':{'1':{'invalid_clipping':False},'2':{'invalid_clipping':False}}}))
        out=root/'out'; out.mkdir()
        with patch.object(fitter,'load_support',support),patch.object(fitter,'fit',fake_fit),patch.object(chooser,'load_support',support),patch.object(drawer,'load_support',support):
            baseline_result=chooser.baseline_scores(obs,baseline,out/'baseline_scoring'); assert baseline_result['OLS_count']==75*len(val)
            result=fitter.fit_candidates(obs,baseline,sh,meta,out/'fit'); assert len(result)==5 and all(len(result[k])==8 for k in result)
            choice=chooser.choose(obs,baseline,sh,out/'fit',out/'selection'); assert choice['status']=='FALLBACK_E4' and digest(run/'direction.npy')==digest(out/'selection/direction.npy')
            drawer.figures(obs,baseline,out/'fit',out/'selection',out/'figures')
        (out/'extraction').mkdir(); (out/'extraction/extraction.json').write_text('{}'); (out/'extraction/sh_values.npz').write_bytes(sh.read_bytes())
        (out/'input_lock.json').write_text(json.dumps({'files':{str(obs):{'sha256':digest(obs),'size':obs.stat().st_size}}}))
        (out/'README.md').write_text('source model iteration 命令 日志 目检 新渲染不适用 PASS FAILED')
        images=['01_sh_bg_observation_residual.png','02_direction_azimuth_elevation_split.png','03_gates_and_holdout.png','04_initialization_bootstrap_convergence.png']
        (out/'inspection.json').write_text(json.dumps(dict(inspected_at_utc='2000-01-01T00:00:00Z',figures={k:'合成检查' for k in images})))
        missing=out/'fit/runs/C00L/full/checkpoint.pt'; content=missing.read_bytes(); missing.unlink()
        try: freeze(out)
        except FileNotFoundError: pass
        else: raise AssertionError('未拒绝缺失 checkpoint')
        missing.write_bytes(content); manifest=freeze(out); assert manifest['status']=='FROZEN_AWAITING_INDEPENDENT_EVALUATION' and manifest['selection_status']=='FALLBACK_E4'
    print('PASS：合成流程调度、共同掩码、全部失败回退、绘图和冻结缺失文件拒绝。')

if __name__=='__main__': main()
