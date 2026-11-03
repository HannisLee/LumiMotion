"""T01 的支撑集与各流程正交模板构造。"""
import json
from pathlib import Path
import numpy as np
from LIHAN.scripts.static_rgb_light_shbg_t01_core import design, project_template


def load_support(baseline, observations):
    fitting=Path(baseline)/'fitting'
    with np.load(fitting/'support.npz',allow_pickle=False) as s:
        ids=s['gaussian_ids'].copy(); full=s['body_fit'].copy(); val=s['body_val'].copy(); A=s['split_A'].copy(); B=s['split_B'].copy()
    with np.load(observations,allow_pickle=False) as obs:
        if not np.array_equal(ids,obs['gaussian_ids']) or not np.array_equal(obs['frames'],np.arange(1,151)): raise ValueError('支撑 Gaussian ID 或帧不一致')
    if tuple(map(len,(full,val,A,B)))!=(2683,648,1450,1233): raise ValueError('支撑集数目不符')
    if not np.array_equal(np.sort(np.concatenate((A,B))),np.sort(full)) or np.intersect1d(full,val).size: raise ValueError('支撑集分区不符')
    runs={'full':full,'A':A,'B':B,'hi':full,'lo':full}
    for seed in (103108,103109):
        pick=np.load(fitting/f'bootstrap_{seed}_sample.npy',allow_pickle=False)
        if len(pick)!=len(full) or not np.isin(pick,full).all(): raise ValueError('bootstrap 索引无效')
        if not np.array_equal(pick,np.random.default_rng(seed).choice(full,len(full),replace=True)): raise ValueError('bootstrap 索引与种子不符')
        runs[f'boot{seed}']=pick
    anchor={}
    for key in ('full','A','B','boot103108','boot103109'):
        a=np.load(fitting/'runs'/f'a1e+00_e1e+01_{key}'/'direction.npy',allow_pickle=False)
        if a.shape!=(150,3) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-10): raise ValueError('锚点方向无效')
        anchor[key]=a
    anchor['hi']=anchor['lo']=anchor['full']; return runs,val,anchor


def template(h,n,y,w,idx,anchor,holdout=False):
    nn=n[:,idx] if n.ndim==3 else np.broadcast_to(n[idx],(len(y),len(idx),3))
    yy=y[:,idx]; ww=w[:,idx]; hh=h[:,idx]; valid=np.isfinite(yy)&np.isfinite(ww)&(ww>0)
    d=design(nn,anchor); q,H,beta,cal,errors=project_template(hh,d,ww,valid,holdout=holdout)
    mass=np.where(valid,ww,0).sum(0); rms=np.sqrt((np.where(valid,ww,0)*q*q).sum(0)/np.maximum(mass,1e-30))
    eligible=mass>0; fraction=float((rms[eligible]>=1e-4).mean()) if eligible.any() else 0.
    return dict(q=q,H=H,beta=beta,cal=cal,error=errors,valid=valid,nondegenerate_fraction=fraction,eligible=eligible)


def save_template(path, h, idx, ids, frames, record, L):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,L=L,frames=frames,gaussian_ids=ids[idx],indices=idx,h=h[:,idx],q=record['q'],H=record['H'],beta=record['beta'],calibration_mask=record['cal'],orthogonality_error=record['error'])
    (path.with_suffix('.json')).write_text(json.dumps({'nondegenerate_fraction':record['nondegenerate_fraction'],'eligible_points':int(record['eligible'].sum())},indent=2))


def combine_templates(fit_dir,selection_dir,output):
    """流式合并各流程与留出模板，不占用多个模板的额外内存。"""
    import shutil
    import zipfile
    inputs=sorted((Path(fit_dir)/'templates').glob('*.npz'))+sorted(Path(selection_dir).glob('L*_holdout.npz'))
    if not inputs: raise ValueError('没有模板可冻结')
    path=Path(output)
    with zipfile.ZipFile(path,mode='x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as dest:
        for source in inputs:
            with zipfile.ZipFile(source) as src:
                for item in src.infolist():
                    if not item.filename.endswith('.npy'): raise ValueError('模板成员格式错误')
                    with src.open(item) as inp, dest.open(f'{source.stem}__{item.filename}',mode='w',force_zip64=True) as out:
                        shutil.copyfileobj(inp,out,1024*1024)
