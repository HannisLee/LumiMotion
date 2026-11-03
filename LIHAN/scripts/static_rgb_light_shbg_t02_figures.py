"""T02 四类无 GT 可视化。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LIHAN.scripts.static_rgb_light_shbg_t02_templates import load_support


def figures(observations,baseline,fit_dir,selection_dir,output):
    fit_dir,sel,out=map(Path,(fit_dir,selection_dir,output)); out.mkdir(parents=True,exist_ok=False)
    table=json.loads((sel/'candidate_table.json').read_text()); selected=table['selected']; name=selected or 'C01'
    indices,_,anchors=load_support(baseline,observations)
    with np.load(observations,allow_pickle=False) as d: y=d['intensity'].copy(); w=d['weights'].copy(); n=d['normals'].copy(); ids=d['gaussian_ids'].copy()
    pred=np.load(sel/'direction.npy',allow_pickle=False); frame=np.arange(1,151); folder=fit_dir/'runs'/name/'full'
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    level=next(row['L'] for row in table['candidates'] if row['name']==name)
    path=fit_dir/'templates'/f'L{level}_full.npz'
    if path.is_file() and (folder/'parameters.npz').is_file():
        with np.load(path,allow_pickle=False) as t: h,H,idx=t['h'],t['H'],t['indices']
        with np.load(folder/'parameters.npz',allow_pickle=False) as p: gamma,rho,b=float(p['gamma']),p['rho'],p['b']
        l=np.load(folder/'direction.npy',allow_pickle=False); nn=n[:,idx] if n.ndim==3 else np.broadcast_to(n[idx],(150,len(idx),3))
        obs=y[:,idx]; ww=w[:,idx]; valid=np.isfinite(obs)&(ww>0); residual=np.maximum(np.einsum('tgc,tc->tg',nn,l),0)*rho+b+gamma*H-obs
        for ax,z,title in zip(axes.flat,(h,gamma*H,obs,residual),('SH','background','observation','residual')):
            ax.plot(frame,np.nanmean(np.where(valid,z,np.nan),axis=1)); ax.set_title(title)
            for t in (1,38,75,112,150): ax.axvline(t,color='gray',alpha=.2)
        bins=np.quantile(obs[valid],np.linspace(0,1,6)); vals=[]
        for i in range(5):
            mask=valid&(obs>=bins[i])&(obs<=bins[i+1] if i==4 else obs<bins[i+1]); vals.append(np.average(np.abs(residual[mask]),weights=ww[mask]) if mask.any() else np.nan)
        axes[2,0].bar(range(5),vals); axes[2,0].set_title('Residual by brightness'); axes[2,1].axis('off')
    else:
        for ax in axes.flat: ax.text(.1,.5,'No available SH fit'); ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'01_sh_bg_observation_residual.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    directions={'E4':anchors['full'],'selected':pred}
    for key in ('C00L/full',name+'/A',name+'/B'):
        f=fit_dir/'runs'/key/'direction.npy'
        if f.is_file(): directions[key]=np.load(f,allow_pickle=False)
    for label,l in directions.items():
        axes[0].plot(frame,np.rad2deg(np.unwrap(np.arctan2(l[:,1],l[:,0]))),label=label)
        axes[1].plot(frame,np.rad2deg(np.arcsin(np.clip(l[:,2],-1,1))),label=label)
        for j in range(3): axes[2].plot(frame,l[:,j],label=f'{label} {"xyz"[j]}')
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'02_direction_azimuth_elevation_split.png',dpi=160); plt.close(fig)
    rows=table['candidates']; gates=['G0','G1','G2','G3','G4a','G4b','G5','G6','G7','G8']; fig,axes=plt.subplots(2,2,figsize=(13,9))
    axes[0,0].imshow([[int(r.get('gates',{}).get(g,False)) for g in gates] for r in rows],vmin=0,vmax=1,cmap='RdYlGn',aspect='auto'); axes[0,0].set_xticks(range(len(gates)),gates,rotation=45); axes[0,0].set_yticks(range(4),[r['name'] for r in rows])
    for i,k in enumerate(('G4a_mae','G4b_mae')): axes[0,1].bar(np.arange(4)+.35*i,[r.get(k) if r.get(k) is not None else np.nan for r in rows],.35,label=k)
    axes[0,1].legend(); ef=sel/f'{name}_eta.npz'
    if ef.is_file():
        with np.load(ef,allow_pickle=False) as z:
            for k in z.files: axes[1,0].hist(z[k],bins=30,alpha=.25,label=k)
            vals=z['full']; idx=indices['full']; order=np.lexsort((ids[idx],-vals))[:5]
            axes[1,1].bar([str(i) for i in ids[idx][order]],vals[order]); axes[1,1].set_title('Highest eta points')
        axes[1,0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'03_gates_and_holdout.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for key in ('full','hi','lo','boot103108','boot103109'):
        d=fit_dir/'runs'/name/key
        if (d/'parameters.npz').is_file():
            with np.load(d/'parameters.npz',allow_pickle=False) as p:
                axes[0,0].plot(p['U'],label=key+' U'); axes[0,1].plot(p['V'],label=key+' V'); axes[1,0].scatter(key,float(p['gamma']))
        if (d/'history.json').is_file():
            hist=json.loads((d/'history.json').read_text()); axes[1,1].plot([x['loss'] for x in hist],label=key+' loss'); axes[1,1].plot([x['gamma'] for x in hist],linestyle='--',alpha=.5,label=key+' gamma')
    axes[0,0].set_title('U'); axes[0,1].set_title('V'); axes[1,0].set_title('gamma'); axes[1,1].set_title('Objective and gamma')
    for ax in (axes[0,0],axes[0,1],axes[1,1]): ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'04_initialization_bootstrap_convergence.png',dpi=160); plt.close(fig)
