"""第五轮：环绕轨迹先验 + 带宽受限时变仰角 + 轨迹平滑（不读取GT）。E0–E3逐字复用第三轮。"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts.static_rgb_light_fit_r3 import sha, write, unit, sub, compare, ols_score, initial, kkt

PREREG=Path('LIHAN/DOC/讨论/1031-08-未知方向光第五轮环绕先验与时变仰角预注册.md')
PREREG_HASH='5d6398e9ae572475e2de9a652d7db9670c99a5cbe103f5978396e3a268fd0314'
OBS=r3.OBS;OBS_HASH=r3.OBS_HASH
R3_PRED=Path('output/1031-06-LH-static-ELDL-V3-第三轮z偏置校正方向光初始化/fitting/predictions.npz')
E_C=np.deg2rad(45.);E_MAX=np.deg2rad(5.);LAM_REV=.1
LAM_A=[1e-4,1e-3,1e-2];LAM_E=[1e-4,1e-3,1e-2,1e-1]
STEPS=400;LR=.015;DROP=250;BOOT=[103108,103109]

def elev(a):return E_C+E_MAX*torch.tanh(a)
def compose(u,e):return torch.cat([torch.cos(e)[:,None]*u,torch.sin(e)[:,None]],1)
def dphi(u):return torch.atan2(u[:-1,0]*u[1:,1]-u[:-1,1]*u[1:,0],(u[:-1]*u[1:]).sum(1))

def penalties(u,e,s):
    d=dphi(u);de=e[1:]-e[:-1]
    return dict(a1=(d-d.mean()).square().mean(),a2=(d[1:]-d[:-1]).square().mean(),e1=de.square().mean(),e2=(de[1:]-de[:-1]).square().mean(),rev=torch.relu(-s*d).square().mean())

def orbit_sign(l):
    d=np.arctan2(l[:-1,0]*l[1:,1]-l[:-1,1]*l[1:,0],(l[:-1,:2]*l[1:,:2]).sum(1))
    return 1. if d.sum()>=0 else -1.

def a_from_z(z):return float(np.arctanh(np.clip((np.arcsin(z)-E_C)/E_MAX,-.9,.9)))

def fit_traj(n,y,w,init,a0,lam_a,lam_e,out,label,steps=STEPS):
    """Stage T：同时优化逐帧方位u与带宽内仰角a；L2 KKT变量投影 + 预注册平滑项。"""
    dest=out/'runs'/label;dest.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    nt=torch.as_tensor(n,dtype=torch.float64,device=device);yt=torch.as_tensor(np.nan_to_num(y),dtype=torch.float64,device=device)
    ww=torch.as_tensor(np.where(np.isfinite(y),w,0),dtype=torch.float64,device=device)
    rawu=torch.nn.Parameter(torch.as_tensor(unit(init[:,:2]),dtype=torch.float64,device=device).clone())
    a=torch.nn.Parameter(torch.as_tensor(np.broadcast_to(a0,(len(init),)).copy(),dtype=torch.float64,device=device))
    s=orbit_sign(init);opt=torch.optim.Adam([rawu,a],lr=LR);hist=[];losses=[]
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True);u=F.normalize(rawu,dim=1);e=elev(a);l=compose(u,e)
        x=torch.relu(torch.einsum('tgc,tc->tg',nt,l) if nt.ndim==3 else l@nt.T)
        rho,b=kkt(x,yt,ww);res=x*rho+b-yt
        data=(ww*.5*res.square()).sum()/ww.sum()+1e-5*(rho.square().mean()+b.square().mean())
        p=penalties(u,e,s);loss=data+lam_a*(p['a1']+p['a2'])+lam_e*(p['e1']+p['e2'])+LAM_REV*p['rev']
        if not torch.isfinite(loss):raise RuntimeError('非有限loss')
        loss.backward();opt.step();losses.append(float(loss.detach()))
        if step==DROP:
            for group in opt.param_groups:group['lr']*=.3
        if step%50==0 or step==1:
            entry=dict(stage='T',step=step,loss=losses[-1],data=float(data.detach()),**{k:float(v.detach()) for k,v in p.items()},e_min_deg=float(torch.rad2deg(e.min())),e_max_deg=float(torch.rad2deg(e.max())));hist.append(entry)
            print(json.dumps(dict(run=label,**entry)),flush=True)
            write(out/'progress.json',dict(pid=os.getpid(),run=label,**entry,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    with torch.no_grad():
        u=F.normalize(rawu,dim=1);e=elev(a);end=compose(u,e).cpu().numpy();p={k:float(v) for k,v in penalties(u,e,s).items()}
    info=dict(penalties=p,orbit_sign=s,final_loss=losses[-1],last50_rel_change=float(abs(losses[-1]-losses[-51])/abs(losses[-51])),elev_deg=np.rad2deg(e.cpu().numpy()).tolist())
    np.save(dest/'direction.npy',end);write(dest/'history.json',hist)
    write(dest/'config.json',dict(lam_a=lam_a,lam_e=lam_e,lam_rev=LAM_REV,steps=steps,lr=LR,lr_drop_step=DROP,e_center_deg=45.,e_halfwidth_deg=5.,device=device,dtype='float64',support_count=y.shape[1],**{k:v for k,v in info.items() if k!='elev_deg'}))
    return end,info

def elev_deg(l):return np.rad2deg(np.arcsin(np.clip(l[:,2],-1,1)))
def traj_stats(l,s):
    phi=np.arctan2(l[:,1],l[:,0]);d=np.angle(np.exp(1j*np.diff(phi)));e=elev_deg(l)
    return dict(saturation_frac=float((abs(e-45.)>=4.75).mean()),reversal_count=int((s*d<0).sum()),elev_min_deg=float(e.min()),elev_max_deg=float(e.max()),elev_std_deg=float(e.std()),max_dphi_dev_deg=float(np.rad2deg(abs(d-d.mean()).max())),max_delev_deg=float(abs(np.diff(e)).max()),mean_dphi_deg=float(np.rad2deg(d.mean())))

GATES=dict(G1=lambda r:r['split_mean_deg']<=3.5 and r['split_p90_deg']<=6.,G2=lambda r:r['hilo_mean_deg']<=.5,G3=lambda r:r['boot_mean_deg']<=2.,G4=lambda r:r['body_heldout_mae']<=1.01*r['E3_body_heldout_mae'],G5=lambda r:r['saturation_frac']<=.2,G6=lambda r:r['reversal_count']<=2,G7=lambda r:r['view_dz_abs']<=.03)

def select(rows):
    def pick(pool):
        if not pool:return None
        m=min(r['J'] for r in pool);tied=[r for r in pool if r['J']<=m+.02]
        return min(tied,key=lambda r:(r['body_heldout_mae'],-r['lam_e'],-r['lam_a']))
    t1=pick([r for r in rows if all(r['gates'].values())])
    if t1:return t1,'tier1'
    t2=pick([r for r in rows if all(r['gates'][g] for g in ['G4','G5','G6','G7'])])
    if t2:return t2,'tier2_stability_gates_not_all_passed'
    return None,'tier3_E3_fallback'

def tests(out):
    from scipy.optimize import nnls
    rng=np.random.default_rng(103108);x=rng.random((60,40));y=rng.random((60,40));w=rng.random((60,40))
    y[:,0]=.4-.2*x[:,0];y[:,1]=2*x[:,1]-.5;y[:,2]=.2+.5*x[:,2];x[:,3]=.5;w[:,4]=0
    tx=torch.tensor(x,dtype=torch.float64,requires_grad=True);r,b=kkt(tx,torch.tensor(y),torch.tensor(w));maxerr=0.
    for g in range(40):
        design=np.column_stack([x[:,g],np.ones(60)])*np.sqrt(w[:,g,None]);target=y[:,g]*np.sqrt(w[:,g])
        maxerr=max(maxerr,float(np.max(abs(nnls(design,target)[0]-np.array([r[g].item(),b[g].item()])))))
    assert maxerr<2e-5,maxerr
    th=torch.arange(150,dtype=torch.float64)*.04;u=torch.stack([th.cos(),th.sin()],1);e=torch.full((150,),.8,dtype=torch.float64)
    zero={k:float(v) for k,v in penalties(u,e,1.).items()};assert all(v<1e-24 for v in zero.values()),zero
    rev=penalties(u.flip(0),e,1.)['rev'];assert rev>1e-4
    band=torch.rad2deg(elev(torch.tensor([-1e3,-3.,0.,3.,1e3],dtype=torch.float64)));assert band.min()>=40-1e-9 and band.max()<=50+1e-9 and abs(band[2]-45)<1e-12
    # 合成：环绕一周 + 带宽内平滑非周期仰角（线性斜坡+单峰），初值为恒定45°与扰动方位。
    T=60;nn=unit(rng.normal(size=(300,3)));t=np.arange(T);phi=np.linspace(0,2*np.pi,T,endpoint=False)
    et=np.deg2rad(41.5+7*t/(T-1)-2.5*np.exp(-((t-35)/8.)**2));truth=np.column_stack([np.cos(et)*np.cos(phi),np.cos(et)*np.sin(phi),np.sin(et)])
    sy=np.maximum(truth@nn.T,0)*rng.uniform(.3,.8,300)+rng.uniform(.03,.12,300);sw=np.ones_like(sy)
    init=np.column_stack([unit(truth[:,:2]+rng.normal(0,.05,(T,2)))*np.cos(E_C),np.full(T,np.sin(E_C))])
    end,info=fit_traj(nn,sy,sw,init,0.,1e-3,1e-3,out,'synthetic')
    err=compare(end,truth);assert err['mean_deg']<1.,err
    end2,_=fit_traj(nn,sy,sw,init,0.,1e-3,1e-3,out,'synthetic_repeat');assert np.allclose(end,end2,atol=1e-10)
    gx=torch.zeros(150,2,dtype=torch.float64,requires_grad=True);ga=torch.zeros(150,dtype=torch.float64,requires_grad=True)
    tot=sum(penalties(F.normalize(u+gx,dim=1),elev(ga),1.).values());tot.backward();assert torch.isfinite(gx.grad).all() and torch.isfinite(ga.grad).all()
    write(out/'numeric_tests.json',dict(status='PASS',nnls_max_parameter_difference=maxerr,uniform_orbit_penalties=zero,reversal_penalty=float(rev),band_deg=band.tolist(),synthetic=err,synthetic_elev_range_deg=[float(np.rad2deg(et.min())),float(np.rad2deg(et.max()))],deterministic_repeat=True,gt_read=False))
    print('R5_SELF_TEST_PASS',json.dumps(err),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=Path,default=OBS);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(103108);np.random.seed(103108)
    if a.self_test:tests(out);return
    assert sha(PREREG)==PREREG_HASH;assert sha(a.observations)==OBS_HASH
    write(out/'launch.json',dict(pid=os.getpid(),argv=sys.argv,prereg_sha256=sha(PREREG),code_sha256=sha(__file__),r3_code_sha256=sha(r3.__file__),input_sha256=sha(a.observations),gt_read=False))
    # ---- 支撑集：与第三轮main逐字等价 ----
    d=np.load(a.observations);n=d['normals'].astype(np.float64);y=d['intensity'].astype(np.float64);w=d['weights'].astype(np.float64);xyz=d['xyz'];ids=d['gaussian_ids'];frames=d['frames']
    assert np.array_equal(frames,np.arange(1,151)) and y.shape==w.shape==(150,len(ids))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);q20=np.nanquantile(y,.2,axis=0);q95=np.nanquantile(y,.95,axis=0)
    eligible=(np.isfinite(y).sum(0)>=8)&(q95-q20>.015)
    mn=n.mean(0) if n.ndim==3 else n;body=(xyz[:,2]>=.05)&(abs(mn[:,2])<=.85)
    cells=np.floor(xyz*20).astype(np.int64);hval=cells[:,0]*73856093 ^ cells[:,1]*19349663 ^ cells[:,2]*83492791
    held=hval%5==0;fi=np.flatnonzero(eligible&body&~held);vi=np.flatnonzero(eligible&body&held);allvi=np.flatnonzero(eligible&held)
    A=fi[(hval[fi]//5)%2==0];B=fi[(hval[fi]//5)%2!=0]
    assert (len(fi),len(vi),len(A),len(B))==(2683,648,1450,1233),(len(fi),len(vi),len(A),len(B))
    relative=np.clip((y-q20)/(q95-q20+.001),0,1.5);relative[:,~eligible]=np.nan
    np.savez_compressed(out/'support.npz',body_fit=fi,body_val=vi,all_val=allvi,split_A=A,split_B=B,gaussian_ids=ids,hval=hval,body_mask=body)
    write(out/'support_manifest.json',dict(support_sha256=sha(out/'support.npz'),body_fit=2683,body_val=648,split_A=1450,split_B=1233))
    scoredata=(sub(n,vi),y[:,vi],w[:,vi]);results={'E0':np.tile([0.,0.,1.],(150,1))};voteinfo={}
    for k in ['E1','E2']:results[k],voteinfo[k]=initial(sub(n,fi),y[:,fi],w[:,fi],relative[:,fi],k,scoredata)
    results['E3'],_,_=r3.fit(sub(n,fi),y[:,fi],w[:,fi],results['E2'],False,.03,0.,out,'E3',False)
    with np.load(R3_PRED) as old:
        repro=dict(E1=float(abs(results['E1']-old['E1']).max()),E2=float(abs(results['E2']-old['E2']).max()),E3=float(abs(results['E3']-old['E3']).max()))
    write(out/'r3_reproduction.json',dict(max_abs_diff=repro,tolerance=1e-6,source=str(R3_PRED),gt_read=False))
    if max(repro.values())>1e-6:raise RuntimeError(f'第三轮E1–E3未复现：{repro}')
    e3mae=ols_score(*scoredata,results['E3'])
    # ---- 各支撑集共享的E2→Stage1 ----
    supports={'full':(fi,results['E3'])};
    for key,idx in [('A',A),('B',B)]:
        init,voteinfo[key]=initial(sub(n,idx),y[:,idx],w[:,idx],relative[:,idx],'E2')
        supports[key]=(idx,r3.fit(sub(n,idx),y[:,idx],w[:,idx],init,False,.03,0.,out,f'stage1_{key}',False)[0])
    for seed in BOOT:
        pick=np.random.default_rng(seed).choice(fi,len(fi),replace=True);np.save(out/f'bootstrap_{seed}_sample.npy',pick)
        init,voteinfo[f'boot{seed}']=initial(sub(n,pick),y[:,pick],w[:,pick],relative[:,pick],'E2')
        supports[f'boot{seed}']=(pick,r3.fit(sub(n,pick),y[:,pick],w[:,pick],init,False,.03,0.,out,f'stage1_boot{seed}',False)[0])
    dark=((relative[:,fi]<.15)&(w[:,fi]>0)).sum(1)/(w[:,fi]>0).sum(1);front=dark<=np.quantile(dark,.3);side=dark>=np.quantile(dark,.7)
    rows=[];ends_all={};infos_all={}
    for lam_a,lam_e in itertools.product(LAM_A,LAM_E):
        name=f'a{lam_a:.0e}_e{lam_e:.0e}';ends={};infos={}
        for key,(idx,s1) in supports.items():
            ends[key],infos[key]=fit_traj(sub(n,idx),y[:,idx],w[:,idx],s1,a_from_z(s1[:,2].mean()),lam_a,lam_e,out,f'{name}_{key}')
        s1=results['E3']
        for key,a0 in [('hi',np.arctanh(.9)),('lo',-np.arctanh(.9))]:
            ends[key],infos[key]=fit_traj(sub(n,fi),y[:,fi],w[:,fi],s1,a0,lam_a,lam_e,out,f'{name}_{key}')
        full=ends['full'];sp=compare(ends['A'],ends['B']);hl=compare(ends['hi'],ends['lo'])
        row=dict(name=name,lam_a=lam_a,lam_e=lam_e,lam_rev=LAM_REV,split_mean_deg=sp['mean_deg'],split_median_deg=sp['median_deg'],split_p90_deg=sp['p90_deg'],split_el_mae_deg=float(abs(elev_deg(ends['A'])-elev_deg(ends['B'])).mean()),hilo_mean_deg=hl['mean_deg'],boot_each_mean_deg=[compare(ends[f'boot{s}'],full)['mean_deg'] for s in BOOT],view_dz_abs=float(abs(full[front,2].mean()-full[side,2].mean())),body_heldout_mae=ols_score(*scoredata,full),all_heldout_mae=ols_score(sub(n,allvi),y[:,allvi],w[:,allvi],full),E3_body_heldout_mae=e3mae,full_penalties=infos['full']['penalties'],last50_rel_change={k:v['last50_rel_change'] for k,v in infos.items()},**traj_stats(full,infos['full']['orbit_sign']))
        row['boot_mean_deg']=float(np.mean(row['boot_each_mean_deg']))
        row['J']=row['split_mean_deg']+.5*row['split_p90_deg']+row['split_el_mae_deg']+row['boot_mean_deg']+row['hilo_mean_deg']
        row['gates']={g:bool(f(row)) for g,f in GATES.items()};row['all_gates']=all(row['gates'].values())
        rows.append(row);ends_all[name]=ends;infos_all[name]=infos
        np.savez_compressed(out/'runs'/f'{name}_ends.npz',**ends)
        write(out/'candidate_table.json',dict(E3_body_heldout_mae=e3mae,E2_body_heldout_mae=ols_score(*scoredata,results['E2']),candidates=rows));print('CANDIDATE',json.dumps(row),flush=True)
    selected,tier=select(rows)
    results['E4']=ends_all[selected['name']]['full'].copy() if selected else results['E3'].copy()
    selected_name=selected['name'] if selected else 'E3_fallback'
    for k,v in results.items():assert v.shape==(150,3) and np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-10)
    np.savez_compressed(out/'predictions.npz',frames=frames,**results)
    config=dict(final_method='E4',selected=selected_name,selection_tier=tier,selected_row=selected,gt_read=False,coordinate_system='world',direction_convention='surface_to_light',directional_intensity=1,prereg_sha256=PREREG_HASH,input_sha256=OBS_HASH,fit_gaussians=len(fi),heldout_gaussians=len(vi),elevation_center_deg=45.,elevation_halfwidth_deg=5.,stageT=dict(steps=STEPS,lr=LR,lr_drop_step=DROP,lr_drop=.3,loss='L2 KKT varpro + 1e-5 coef reg',lam_a_grid=LAM_A,lam_e_grid=LAM_E,lam_rev=LAM_REV),J='split_mean+0.5*split_p90+split_el_mae+boot_mean+hilo_mean',gates='G1 split_mean<=3.5&split_p90<=6; G2 hilo<=0.5; G3 boot<=2; G4 body<=1.01*E3; G5 sat<=0.2; G6 rev<=2; G7 view_dz<=0.03',tie_rule='J within 0.02 -> body_mae, -lam_e, -lam_a',r3_reproduction=repro,votes=voteinfo,photometric_mae={k:ols_score(*scoredata,l) for k,l in results.items()},information_limitations='稳定性、轨迹平滑与留出残差均为无GT经验准则，不证明GT精度')
    write(out/'method_config.json',config)
    figures(out,d,n,y,w,relative,fi,results,voteinfo,rows,ends_all,infos_all,selected_name)
    write(out/'fit_complete.json',dict(status='R5_FIT_COMPLETE_AWAITING_REVIEW',selected=selected_name,tier=tier,pid=os.getpid(),gt_read=False))
    print('R5_FIT_COMPLETE',selected_name,tier,flush=True)

def figures(out,d,n,y,w,relative,fi,results,voteinfo,rows,ends_all,infos_all,selected_name):
    l=results['E4'];x=np.maximum(r3.shade(sub(n,fi),l),0);yy=np.nan_to_num(y[:,fi]);ww=w[:,fi]
    with torch.no_grad():rho,b=kkt(torch.tensor(x),torch.tensor(yy),torch.tensor(ww));rho=rho.numpy();b=b.numpy()
    residual=x*rho+b-y[:,fi]
    np.savez_compressed(out/'photometric_diagnostics.npz',residual=residual,rho=rho,b=b,body_fit=fi,diagnostic_only=True)
    uv=d['projection_uv'][:,fi] if 'projection_uv' in d else None
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for col,t in enumerate([0,74,149]):
        ok=(ww[t]>0)&np.isfinite(y[t,fi]);pix=(uv[t]+1)*200-.5 if uv is not None else d['xyz'][fi,:2]
        high=ok&(relative[t,fi]>=np.quantile(relative[t,fi][ok],1-voteinfo['E2']['selected']['top']))
        axs[0,col].scatter(pix[ok,0],pix[ok,1],s=2,c='lightgray');axs[0,col].scatter(pix[high,0],pix[high,1],s=5,c=relative[t,fi][high],cmap='viridis');axs[0,col].set_title(f'Frame {t+1}: E2 body vote')
        sc=axs[1,col].scatter(pix[ok,0],pix[ok,1],s=4,c=residual[t,ok],cmap='coolwarm',vmin=-.15,vmax=.15);axs[1,col].set_title(f'E4 ({selected_name}) KKT residual');fig.colorbar(sc,ax=axs[1,col])
        for ax in axs[:,col]:ax.set_aspect('equal');ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'01_percentile_residual.png',dpi=150);plt.close(fig)
    shown=selected_name if selected_name in ends_all else min(rows,key=lambda r:r['J'])['name'];ends=ends_all[shown]
    fig,axs=plt.subplots(2,2,figsize=(13,8));fig.suptitle(f'Final E4: {selected_name}; split/boot diagnostics: {shown}')
    for k in ['E1','E2','E3','E4']:
        ll=results[k];axs[0,0].plot(np.rad2deg(np.unwrap(np.arctan2(ll[:,1],ll[:,0]))),label=k);axs[0,1].plot(elev_deg(ll),label=k)
    for k in ['A','B']:axs[1,0].plot(np.rad2deg(np.unwrap(np.arctan2(ends[k][:,1],ends[k][:,0]))),label=k);axs[1,1].plot(elev_deg(ends[k]),label=k)
    for ax in axs[:,1]:ax.axhline(40,c='k',ls=':');ax.axhline(50,c='k',ls=':')
    axs[0,1].set_ylim(35,75)
    for ax in axs.flat:ax.legend();ax.set_xlabel('Frame index')
    axs[0,0].set_ylabel('Unwrapped azimuth (deg)');axs[0,1].set_ylabel('Elevation (deg)');axs[1,0].set_ylabel('Split azimuth');axs[1,1].set_ylabel('Split elevation')
    fig.tight_layout();fig.savefig(out/'02_direction_split.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(17,8));names=[r['name'] for r in rows]
    for ax,key in zip(axs.flat,['body_heldout_mae','split_mean_deg','boot_mean_deg','saturation_frac','view_dz_abs','J']):
        ax.bar(names,[r[key] for r in rows],color=['tab:green' if r['all_gates'] else 'tab:red' for r in rows]);ax.tick_params(axis='x',rotation=60,labelsize=7);ax.set_ylabel(key)
    axs[0,0].axhline(1.01*rows[0]['E3_body_heldout_mae'],c='k',ls='--');axs[0,1].axhline(3.5,c='k',ls='--');axs[0,2].axhline(2,c='k',ls='--');axs[1,0].axhline(.2,c='k',ls='--');axs[1,1].axhline(.03,c='k',ls='--')
    fig.suptitle('Candidates (green = all gates G1-G7 pass; dashed = gate)');fig.tight_layout();fig.savefig(out/'03_holdout_stability.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8));fig.suptitle(f'hi/lo and bootstrap diagnostics: {shown}')
    for key in ['full','hi','lo']+[f'boot{s}' for s in BOOT]:
        axs[0,0].plot(elev_deg(ends[key]),label=key);axs[0,1].plot(np.rad2deg(np.unwrap(np.arctan2(ends[key][:,1],ends[key][:,0]))),label=key)
    for key in ['full','hi','lo']:
        h=json.loads((out/'runs'/f'{shown}_{key}'/'history.json').read_text());axs[1,0].plot([e['step'] for e in h],[e['loss'] for e in h],label=key)
    for r in rows:axs[1,1].plot(r['lam_e'],r['saturation_frac'],'o',label=None)
    axs[1,1].set_xscale('log');axs[1,1].set_xlabel('lam_e');axs[1,1].set_ylabel('saturation frac (all lam_a)')
    axs[0,0].axhline(40,c='k',ls=':');axs[0,0].axhline(50,c='k',ls=':');axs[0,0].set_ylabel('Elevation (deg)');axs[0,1].set_ylabel('Azimuth (deg)');axs[1,0].set_ylabel('Stage T loss');axs[1,0].set_yscale('log')
    for ax in axs.flat[:3]:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'04_hilo_convergence.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
