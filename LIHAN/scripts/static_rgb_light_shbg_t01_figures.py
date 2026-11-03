"""T01 四类无 GT 可视化。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def figures(observations,fit_dir,selection_dir,output):
    fit_dir=Path(fit_dir); selection_dir=Path(selection_dir); out=Path(output); out.mkdir(parents=True,exist_ok=False)
    table=json.loads((selection_dir/'candidate_table.json').read_text()); selected=table['selected']
    with np.load(selection_dir/'predictions.npz',allow_pickle=False) as p: e4=p['E4']; pred=p['selected']
    with np.load(observations,allow_pickle=False) as d: y=d['intensity']; weights=d['weights']; normals=d['normals']
    name=selected if selected is not None else 'C01'; folder=fit_dir/'runs'/name/'full'
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    fig.suptitle(f'Full-run diagnostics: {name}; selection: {table["status"]}')
    if (folder/'parameters.npz').exists():
        L=next(r['L'] for r in table['candidates'] if r['name']==name)
        with np.load(fit_dir/'templates'/f'L{L}_full.npz',allow_pickle=False) as t:
            h=t['h']; H=t['H']; idx=t['indices']
        with np.load(folder/'parameters.npz',allow_pickle=False) as p: gamma=float(p['gamma']); rho=p['rho']; b=p['b']
        nn=normals[:,idx] if normals.ndim==3 else np.broadcast_to(normals[idx],(len(y),len(idx),3))
        xx=np.maximum(np.einsum('tgc,tc->tg',nn,pred if selected else np.load(folder/'direction.npy')),0)
        obs=y[:,idx]; w=weights[:,idx]; valid=np.isfinite(obs)&(w>0); resid=xx*rho+b+gamma*H-obs
        for ax,v,title in zip(axes.flat,(h,gamma*H,obs,resid),('SH appearance','background proxy','observation','residual')):
            z=np.where(valid,v,np.nan); ax.plot(np.arange(1,len(y)+1),np.nanmean(z,axis=1)); ax.set_title(title); ax.set_xlabel('frame')
            for fr in (1,38,75,112,150): ax.axvline(fr,color='gray',alpha=.15)
        bins=np.quantile(obs[valid],np.linspace(0,1,6)); scores=[]
        for k in range(5):
            m=valid&(obs>=bins[k])&((obs<=bins[k+1]) if k==4 else (obs<bins[k+1]))
            scores.append(float(np.average(np.abs(resid[m]),weights=w[m])) if m.any() else np.nan)
        axes[2,0].bar(range(1,6),scores); axes[2,0].set_title('absolute residual by brightness quintile'); axes[2,1].axis('off')
    else:
        for ax in axes.flat: ax.text(.1,.5,'No converged full fit available'); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out/'01_sh_bg_observation_residual.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,10)); fig.suptitle(f'Selection: {table["status"]}; split traces: {name}'); frame=np.arange(1,len(e4)+1)
    for label,l in (('E4',e4),('selected',pred)):
        axes[0].plot(frame,np.rad2deg(np.unwrap(np.arctan2(l[:,1],l[:,0]))),label=label)
        axes[1].plot(frame,np.rad2deg(np.arcsin(np.clip(l[:,2],-1,1))),label=label)
        for j,component in enumerate('xyz'): axes[2].plot(frame,l[:,j],label=f'{label} {component}')
    for key in ('A','B'):
        f=fit_dir/'runs'/name/key/'direction.npy'
        if f.exists():
            l=np.load(f,allow_pickle=False); axes[0].plot(frame,np.rad2deg(np.unwrap(np.arctan2(l[:,1],l[:,0]))),label=key,alpha=.6)
            axes[1].plot(frame,np.rad2deg(np.arcsin(np.clip(l[:,2],-1,1))),label=key,alpha=.6)
    axes[0].set_ylabel('azimuth (deg)'); axes[1].set_ylabel('elevation (deg)'); axes[2].set_ylabel('direction components')
    for ax in axes: ax.legend(); ax.set_xlabel('frame')
    fig.tight_layout(); fig.savefig(out/'02_direction_azimuth_elevation_split.png',dpi=160); plt.close(fig)
    rows=table['candidates']; gates=['G0','G1','G2','G3','G4a','G4b','G5','G6','G7','G8']; mat=np.array([[int(r.get('gates',{}).get(g,False)) for g in gates] for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].imshow(mat,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto'); axes[0].set_xticks(range(len(gates)),gates,rotation=45); axes[0].set_yticks(range(len(rows)),[r['name'] for r in rows]); axes[0].set_title('gates')
    xx=np.arange(len(rows)); axes[1].bar(xx-.2,[r.get('G4a_mae') if r.get('G4a_mae') is not None else np.nan for r in rows],.4,label='OLS'); axes[1].bar(xx+.2,[r.get('G4b_mae') if r.get('G4b_mae') is not None else np.nan for r in rows],.4,label='background KKT'); axes[1].set_xticks(xx,[r['name'] for r in rows]); axes[1].legend(); axes[1].set_title('holdout MAE')
    fig.tight_layout(); fig.savefig(out/'03_gates_and_holdout.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for key in ('hi','lo','boot103108','boot103109'):
        f=fit_dir/'runs'/name/key/'direction.npy'
        if f.exists():
            l=np.load(f,allow_pickle=False); deg=np.rad2deg(np.arccos(np.clip((l*pred).sum(1),-1,1))); axes[0,0].plot(frame,deg,label=key)
    axes[0,0].set_title('difference to selected'); axes[0,0].legend()
    for key in ('full','hi','lo','boot103108','boot103109'):
        f=fit_dir/'runs'/name/key/'history.json'
        if f.exists():
            hist=json.loads(f.read_text()); axes[0,1].plot([a['loss'] for a in hist],label=key); axes[1,0].plot([a['gamma'] for a in hist],label=key)
    axes[0,1].set_title('objective history'); axes[1,0].set_title('gamma history'); axes[0,1].legend(); axes[1,0].legend()
    axes[1,1].text(.05,.8,f"Selection: {table['status']}\nCandidate: {selected}"); axes[1,1].set_axis_off()
    fig.tight_layout(); fig.savefig(out/'04_initialization_bootstrap_convergence.png',dpi=160); plt.close(fig)
