"""合成数据的拟合组件联调；使用假的优化器，不代表收敛证书自检。"""
import json
from pathlib import Path
import tempfile
import numpy as np
from LIHAN.scripts import static_rgb_light_shbg_t01_fit as fitter
from LIHAN.scripts import static_rgb_light_shbg_t01_choose as chooser
from LIHAN.scripts.static_rgb_light_shbg_t01_figures import figures
from LIHAN.scripts.static_rgb_light_shbg_t01_templates import combine_templates
from LIHAN.scripts.static_rgb_light_shbg_t01_freeze import freeze


def main():
    rng=np.random.default_rng(1104); T=150; G=20; phi=np.linspace(0,2*np.pi,T); l=np.column_stack([np.cos(phi)/np.sqrt(2),np.sin(phi)/np.sqrt(2),np.full(T,1/np.sqrt(2))]); n=rng.normal(size=(T,G,3)); n/=np.linalg.norm(n,axis=2,keepdims=True)
    x=np.maximum(np.einsum('tgc,tc->tg',n,l),0); h1=.4+rng.normal(0,.025,(T,G)); h2=.4+rng.normal(0,.025,(T,G)); y=.2+.6*x+.05*h1; w=np.ones_like(y)
    idx=np.arange(15); val=np.arange(15,20); indices={k:idx for k in fitter.RUNS}; anchors={k:l for k in fitter.RUNS}
    def support(*args): return indices,val,anchors
    def fakefit(nn,yy,ww,H,anchor,g,a,lam,dest):
        dest=Path(dest); dest.mkdir(parents=True); np.save(dest/'direction.npy',anchor); np.savez_compressed(dest/'parameters.npz',gamma=g,rho=np.full(yy.shape[1],.6),b=np.full(yy.shape[1],.2))
        (dest/'history.json').write_text(json.dumps([{'stage':'fake','step':1,'loss':.01,'gamma':g}])); return {'converged':True}
    fitter.load_support=support; chooser.load_support=support; fitter.fit=fakefit
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); obs=root/'obs.npz'; sh=root/'sh.npz'; base=root/'base'; e3=base/'fitting'/'runs'/'E3'; e3.mkdir(parents=True); np.save(e3/'direction.npy',l)
        np.savez_compressed(obs,normals=n,intensity=y,weights=w,gaussian_ids=np.arange(G),frames=np.arange(1,T+1))
        np.savez_compressed(sh,gaussian_ids=np.arange(G),frames=np.arange(1,T+1),h_L0=np.full((T,G),.4),h_L1=h1,h_L2=h2)
        meta=root/'extract.json'; meta.write_text(json.dumps({'diagnostics':{str(i):{'invalid_clipping':False} for i in (0,1,2)}}))
        fitter.fit_candidates(obs,base,sh,meta,root/'fit'); result=chooser.choose(obs,base,sh,root/'fit',root/'selection')
        assert len(result['candidates'])==4 and result['status'] in ('SELECTED_SH','FALLBACK_E4')
        figures(obs,root/'fit',root/'selection',root/'figures'); assert len(list((root/'figures').glob('*.png')))==4
        combine_templates(root/'fit',root/'selection',root/'sh_templates.npz')
        with np.load(root/'sh_templates.npz',allow_pickle=False) as z: assert 'L1_full__q' in z.files and 'L1_holdout__L' in z.files
        from datetime import datetime,timezone
        (root/'input_lock.json').write_text('{}'); (root/'extraction').mkdir(); (root/'extraction'/'extraction.json').write_text('{}'); (root/'extraction'/'sh_values.npz').write_bytes(b'synthetic')
        (root/'README.md').write_text('source model iteration 命令 日志 目检 新渲染不适用 PASS FAILED；合成自检占位')
        (root/'inspection.json').write_text(json.dumps({'inspected_at_utc':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),'figures':{p.name:'合成测试占位' for p in (root/'figures').glob('*.png')}}))
        frozen=freeze(root); assert frozen['status']=='FROZEN_AWAITING_INDEPENDENT_EVALUATION' and 'selection/predictions.npz' in frozen['files']
        try: freeze(root); raise AssertionError('重复冻结未被拒绝')
        except FileExistsError: pass
    print('合成联调通过；优化器为模拟，未读取真实数据或 GT。')

if __name__=='__main__': main()
