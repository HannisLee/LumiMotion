"""第五轮修订版(r5b)：φ角参数化 + Adam热启动 + L-BFGS抛光 + 收敛证书（不读取GT）。E0–E3逐字复用第三轮。预注册 讨论/1031-09。"""
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

PREREG=Path('LIHAN/DOC/讨论/1031-09-未知方向光第五轮修订版环绕先验与时变仰角预注册.md')
PREREG_HASH='4ad3c1b8ebaed20111294a41652f320ef191a15827ef82400d70c346421d5646'
ADDENDUM=Path('LIHAN/DOC/讨论/1031-10-第五轮修订版预注册附录-预条件与自检判据.md')
ADDENDUM_HASH='9b904e3ac8377a3083f0c7ba7f2d2fc2cb1e241556cad196cd56d6558c434293'
ADDENDUM2=Path('LIHAN/DOC/讨论/1031-11-第五轮修订版预注册附录二-C3降幅阈值.md')
ADDENDUM2_HASH='9196cd4e598071a53053bfa484b1a5fd48741dbffac8da44509e7f8498e931d0'
ADDENDUM3=Path('LIHAN/DOC/讨论/1031-12-第五轮修订版预注册附录三-L-BFGS停步后预条件Adam续跑.md')
ADDENDUM4=Path('LIHAN/DOC/讨论/1031-13-第五轮修订版预注册附录四-C3阈值按门槛分辨率标定.md')
ADDENDUM4_HASH='175187b641a0457a862795e2139e69cfbe0b6c1e65c4dfcd7fe70fe48fbc6f19'
ADDENDUM3_HASH='a124e436aef3e538db695b381d792a1d45107bce56eeeceaca65f9ef9cbb6ae9'
OBS=r3.OBS;OBS_HASH=r3.OBS_HASH
R3_PRED=Path('output/1031-06-LH-static-ELDL-V3-第三轮z偏置校正方向光初始化/fitting/predictions.npz')
E_C=np.deg2rad(45.);E_MAX=np.deg2rad(5.);LAM_REV=.1
LAM_A=[1e-2,1e-1,1.];LAM_E=[1e-2,1e-1,1.,10.]
STEPS=1000;LR_PHI=.01;LR_A=.11;DROPS=(600,850);BOOT=[103108,103109]
OUTER=200;OUTER_EXT=400;LB_ITER=20;WIN=5
TC_STEPS=6000;TC_LR=3e-3;TC_OUTER=60

def elev(a):return E_C+E_MAX*torch.tanh(a)
def compose(u,e):return torch.cat([torch.cos(e)[:,None]*u,torch.sin(e)[:,None]],1)
def dphi_of(phi):return phi[1:]-phi[:-1]

def penalties(phi,e,s):
    d=dphi_of(phi);de=e[1:]-e[:-1]
    return dict(a1=(d-d.mean()).square().mean(),a2=(d[1:]-d[:-1]).square().mean(),e1=de.square().mean(),e2=(de[1:]-de[:-1]).square().mean(),rev=torch.relu(-s*d).square().mean())

def orbit_sign(l):
    d=np.diff(np.unwrap(np.arctan2(l[:,1],l[:,0])))
    return 1. if d.sum()>=0 else -1.

def a_from_z(z):return float(np.arctanh(np.clip((np.arcsin(z)-E_C)/E_MAX,-.9,.9)))

def ang_deg(l1,l2):return np.rad2deg(np.arccos(np.clip((l1*l2).sum(1),-1,1)))

def fit_traj(n,y,w,init,a0,lam_a,lam_e,out,label,steps=STEPS,outer=OUTER,outer_ext=OUTER_EXT,force_tc=False):
    """Stage T(修订版)：逐帧φ与带宽内仰角a；L2 KKT变量投影 + 平滑项；Adam热启动→L-BFGS抛光→收敛证书C1–C3。"""
    dest=out/'runs'/label;dest.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    nt=torch.as_tensor(n,dtype=torch.float64,device=device);yt=torch.as_tensor(np.nan_to_num(y),dtype=torch.float64,device=device)
    ww=torch.as_tensor(np.where(np.isfinite(y),w,0),dtype=torch.float64,device=device)
    phi0=np.unwrap(np.arctan2(init[:,1],init[:,0]));s=orbit_sign(init)
    if abs(np.diff(phi0).sum())<np.pi:raise RuntimeError(f'环绕先验不成立：|sum dphi|={abs(np.diff(phi0).sum())}')
    phi=torch.nn.Parameter(torch.as_tensor(phi0,dtype=torch.float64,device=device).clone())
    a=torch.nn.Parameter(torch.as_tensor(np.broadcast_to(a0,(len(init),)).copy(),dtype=torch.float64,device=device))
    def objective(phi_=None,a_=None):
        phi_=phi if phi_ is None else phi_;a_=a if a_ is None else a_
        e=elev(a_);u=torch.stack([torch.cos(phi_),torch.sin(phi_)],1);l=compose(u,e)
        x=torch.relu(torch.einsum('tgc,tc->tg',nt,l) if nt.ndim==3 else l@nt.T)
        rho,b=kkt(x,yt,ww);res=x*rho+b-yt
        data=(ww*.5*res.square()).sum()/ww.sum()+1e-5*(rho.square().mean()+b.square().mean())
        p=penalties(phi_,e,s);return data+lam_a*(p['a1']+p['a2'])+lam_e*(p['e1']+p['e2'])+LAM_REV*p['rev'],data,p,e
    def direction(phi_=None,a_=None):
        phi_=phi if phi_ is None else phi_;a_=a if a_ is None else a_
        with torch.no_grad():return compose(torch.stack([torch.cos(phi_),torch.sin(phi_)],1),elev(a_)).cpu().numpy()
    T_=len(init);D1=torch.diff(torch.eye(T_,dtype=torch.float64,device=device),dim=0);D2=torch.diff(D1,dim=0) if T_>2 else D1
    def precond(lam,w):
        # 预条件（仅优化器细节，不改模型/目标）：P=(I+L²D1ᵀD1+L⁴D2ᵀD2)^(-1/2)，L²=λ/(T·κ·w)，κ=1e-4（预注册A6平滑长度标定）
        L2=lam/(T_*1e-4*w);M=torch.eye(T_,dtype=torch.float64,device=device)+L2*D1.T@D1+L2**2*D2.T@D2
        ev,U=torch.linalg.eigh(M);return U@torch.diag(ev.rsqrt())@U.T
    Pphi=precond(lam_a,.5);Pa=precond(lam_e,1.)
    def state():return phi.detach().clone(),a.detach().clone()
    def restore(st):
        with torch.no_grad():phi.copy_(st[0]);a.copy_(st[1])
    hist=[];losses=[]
    # ---- T-A Adam ----
    opt=torch.optim.Adam([dict(params=[phi],lr=LR_PHI),dict(params=[a],lr=LR_A)])
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True);loss,data,p,e=objective()
        if not torch.isfinite(loss):raise RuntimeError('非有限loss(Adam)')
        loss.backward();opt.step();losses.append(float(loss.detach()))
        if step in DROPS:
            for g in opt.param_groups:g['lr']*=.3
        if step%50==0 or step==1:
            entry=dict(stage='TA',step=step,loss=losses[-1],data=float(data.detach()),**{k:float(v.detach()) for k,v in p.items()},e_min_deg=float(torch.rad2deg(e.min())),e_max_deg=float(torch.rad2deg(e.max())));hist.append(entry)
            print(json.dumps(dict(run=label,**entry)),flush=True);write(out/'progress.json',dict(pid=os.getpid(),run=label,**entry,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    # ---- T-B L-BFGS + 证书 ----
    def run_lbfgs(max_outer,start=0):
        x0p=phi.detach().clone();x0a=a.detach().clone()
        qp=torch.nn.Parameter(torch.zeros_like(x0p));qa=torch.nn.Parameter(torch.zeros_like(x0a))
        lb=torch.optim.LBFGS([qp,qa],lr=1,history_size=50,line_search_fn='strong_wolfe',max_iter=LB_ITER,tolerance_grad=1e-12,tolerance_change=1e-15)
        eff=lambda:(x0p+Pphi@qp,x0a+Pa@qa)
        def closure():
            lb.zero_grad();l=objective(*eff())[0];l.backward();return l
        trace=[];dirs=[]
        def sync():
            with torch.no_grad():
                pe,ae=eff();phi.copy_(pe);a.copy_(ae)
        for k in range(start,max_outer):
            lb.step(closure)
            with torch.no_grad():pe,ae=eff();l=objective(pe,ae)[0]
            if not torch.isfinite(l):raise FloatingPointError('非有限loss(LBFGS)')
            trace.append(float(l));dirs.append(direction(pe,ae))
            if len(trace)%10==0:print(json.dumps(dict(run=label,stage='TB',outer=k+1,loss=trace[-1])),flush=True)
            if len(trace)>WIN:
                rel=abs(trace[-1-WIN]-trace[-1])/abs(trace[-1-WIN]);mx=float(ang_deg(dirs[-1-WIN],dirs[-1]).max())
                if rel<1e-9 and mx<.005:sync();return trace,dirs,True,rel,mx
        sync();rel=abs(trace[-1-WIN]-trace[-1])/abs(trace[-1-WIN]);mx=float(ang_deg(dirs[-1-WIN],dirs[-1]).max());return trace,dirs,False,rel,mx
    def probe():
        base=direction();st=state();lb0=float(objective()[0].detach());o=torch.optim.Adam([dict(params=[phi],lr=1e-4),dict(params=[a],lr=1e-3)])
        for _ in range(500):
            o.zero_grad();l=objective()[0];l.backward();o.step()
        d1=direction();l1=float(objective()[0].detach());disp=float(ang_deg(base,d1).max());drop=(lb0-l1)/abs(lb0);restore(st);return disp,drop
    cert=dict(path='lbfgs');fallback=False
    try:
        trace,dirs,ok,rel,mx=run_lbfgs(outer)
        ext=False
        if not ok:
            ext=True;t2,d2,ok,rel,mx=run_lbfgs(outer_ext,len(trace));trace+=t2;dirs+=d2
        cert.update(C1_rel=rel,C1_ok=bool(rel<1e-9),C2_max_deg=mx,C2_ok=bool(mx<.005),extended=ext,outer_iters=len(trace))
        disp,drop=probe();cert.update(C3_disp_deg=disp,C3_drop=drop,C3_ok=bool(disp<.1 and drop<1e-4))
        cert['converged']=bool(cert['C1_ok'] and cert['C2_ok'] and cert['C3_ok']);final_losses=trace
        if (not cert['converged']) or force_tc:
            # T-C（附录三）：预条件Adam(q空间)长程续跑+学习率余弦衰减，再短L-BFGS抛光，证书C1–C3重新判定（阈值不变）
            pre=dict(cert);x0p=phi.detach().clone();x0a=a.detach().clone()
            qp=torch.nn.Parameter(torch.zeros_like(x0p));qa=torch.nn.Parameter(torch.zeros_like(x0a))
            oc=torch.optim.Adam([qp,qa],lr=TC_LR);sc=torch.optim.lr_scheduler.CosineAnnealingLR(oc,TC_STEPS,eta_min=0)
            for k in range(TC_STEPS):
                oc.zero_grad();l=objective(x0p+Pphi@qp,x0a+Pa@qa)[0]
                if not torch.isfinite(l):raise FloatingPointError('非有限loss(LBFGS-TC)')
                l.backward();oc.step();sc.step()
                if k%500==0:print(json.dumps(dict(run=label,stage='TC',step=k,loss=float(l.detach()))),flush=True)
            with torch.no_grad():phi.copy_(x0p+Pphi@qp);a.copy_(x0a+Pa@qa)
            t3,d3,ok3,rel3,mx3=run_lbfgs(TC_OUTER);trace+=t3
            disp,drop=probe()
            cert=dict(path='lbfgs+TC',pre_TC=pre,C1_rel=rel3,C1_ok=bool(rel3<1e-9),C2_max_deg=mx3,C2_ok=bool(mx3<.005),extended=ext,outer_iters=len(trace),TC_steps=TC_STEPS,TC_lr=TC_LR,C3_disp_deg=disp,C3_drop=drop,C3_ok=bool(disp<.1 and drop<1e-4))
            cert['converged']=bool(cert['C1_ok'] and cert['C2_ok'] and cert['C3_ok']);final_losses=trace
    except (FloatingPointError,RuntimeError) as exc:
        if 'LBFGS' not in str(exc):raise
        fallback=True;restore((torch.as_tensor(phi0,dtype=torch.float64,device=device),torch.as_tensor(np.broadcast_to(a0,(len(init),)).copy(),dtype=torch.float64,device=device)))
        o=torch.optim.Adam([dict(params=[phi],lr=LR_PHI),dict(params=[a],lr=LR_A)]);N=6000+steps;sched=torch.optim.lr_scheduler.CosineAnnealingLR(o,N,eta_min=0)
        fl=[]
        for _ in range(N):
            o.zero_grad();l=objective()[0];l.backward();o.step();sched.step();fl.append(float(l.detach()))
        d_last=direction();rel=abs(fl[-501]-fl[-1])/abs(fl[-501]);disp,drop=probe()
        cert=dict(path='adam_fallback',reason=str(exc),C1_rel=rel,C3_disp_deg=disp,C3_drop=drop,converged=bool(rel<1e-7 and disp<.1 and drop<1e-4));final_losses=fl
    with torch.no_grad():
        loss,data,p,e=objective();end=direction();pp={k:float(v) for k,v in p.items()};eld=np.rad2deg(e.cpu().numpy())
        x=torch.relu(torch.einsum('tgc,tc->tg',nt,torch.as_tensor(end,device=device)) if nt.ndim==3 else torch.as_tensor(end,device=device)@nt.T)
        rho,b=kkt(x,yt,ww)
        np.savez_compressed(dest/'parameters.npz',phi=phi.detach().cpu().numpy(),a=a.detach().cpu().numpy(),rho=rho.cpu().numpy(),b=b.cpu().numpy())
    info=dict(penalties=pp,orbit_sign=s,final_loss=float(loss),data_loss=float(data),certificate=cert,converged=cert['converged'],elev_deg=eld.tolist())
    np.save(dest/'direction.npy',end);write(dest/'history.json',hist);write(dest/'certificate.json',cert)
    write(dest/'config.json',dict(lam_a=lam_a,lam_e=lam_e,lam_rev=LAM_REV,adam_steps=steps,lr_phi=LR_PHI,lr_a=LR_A,drops=list(DROPS),e_center_deg=45.,e_halfwidth_deg=float(np.rad2deg(E_MAX)),device=device,dtype='float64',support_count=y.shape[1],final_loss=float(loss),certificate=cert,penalties=pp))
    print(json.dumps(dict(run=label,stage='DONE',final_loss=float(loss),certificate=cert)),flush=True)
    return end,info

def elev_deg(l):return np.rad2deg(np.arcsin(np.clip(l[:,2],-1,1)))
def traj_stats(l,s):
    phi=np.arctan2(l[:,1],l[:,0]);d=np.angle(np.exp(1j*np.diff(phi)));e=elev_deg(l)
    return dict(saturation_frac=float((abs(e-45.)>=4.75).mean()),reversal_count=int((s*d<0).sum()),elev_min_deg=float(e.min()),elev_max_deg=float(e.max()),elev_std_deg=float(e.std()),max_dphi_dev_deg=float(np.rad2deg(abs(d-d.mean()).max())),max_delev_deg=float(abs(np.diff(e)).max()),mean_dphi_deg=float(np.rad2deg(d.mean())))

GATES=dict(G0=lambda r:r['converged'],G1=lambda r:r['split_mean_deg']<=3.5 and r['split_p90_deg']<=6.,G2=lambda r:r['hilo_mean_deg']<=.5,G3=lambda r:r['boot_mean_deg']<=2.,G4=lambda r:r['body_heldout_mae']<=1.015*r['E3_body_heldout_mae'],G5=lambda r:r['saturation_frac']<=.2,G6=lambda r:r['reversal_count']<=2,G7=lambda r:r['view_dz_abs']<=.03)

def select(rows):
    def pick(pool):
        if not pool:return None
        m=min(r['J'] for r in pool);tied=[r for r in pool if r['J']<=m+.02]
        return min(tied,key=lambda r:(r['body_heldout_mae'],-r['lam_e'],-r['lam_a'],r['name']))
    t1=pick([r for r in rows if all(r['gates'].values())])
    if t1:return t1,'tier1'
    t2=pick([r for r in rows if all(r['gates'][g] for g in ['G0','G1','G3','G4','G5','G6','G7']) and r['hilo_mean_deg']<=.5*r['split_mean_deg']])
    if t2:return t2,'tier2_G2_absolute_not_passed(G2prime_used)'
    return None,'tier3_E3_fallback'+('_NOT_CONVERGED' if not any(r['converged'] for r in rows) else '')


CANDIDATES=[("C00",10.,5.),("C01",3.,5.),("C02",1.,5.),("C03",0.,5.),("C04",10.,5./np.sqrt(.3)),("C05",10.,5./np.sqrt(.1)),("C06",10.,45.)]
BASE=Path('output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角/fitting')
PLAN=Path('LIHAN/DOC/规划/1107-01-T01减弱仰角正则化机械执行规划.md')

def select_t01(rows,delta=.02,tau=.02,el_weight=1.):
    def score(r):return r['split_mean_deg']+.5*r['split_p90_deg']+el_weight*r['split_el_mae_deg']+r['boot_mean_deg']+r['hilo_mean_deg']
    base=score(rows[0]);pool=[r for r in rows[1:] if r['all_gates'] and np.isfinite(score(r)) and np.isfinite(r['body_heldout_mae']) and score(r)<base-delta]
    if not pool or not np.isfinite(base):return None
    best=min(score(r) for r in pool);return min((r for r in pool if score(r)<=best+tau),key=lambda r:(r['body_heldout_mae'],r['name']))

def sensitivity(rows):
    result=[]
    for axis,values in [('delta',[0,.01,.02,.05,.10]),('tau',[0,.01,.02,.05]),('el_weight',[.5,1.,2.])]:
        for value in values:
            opts=dict(delta=.02,tau=.02,el_weight=1.);opts[axis]=value;chosen=select_t01(rows,**opts)
            score=lambda r:r['split_mean_deg']+.5*r['split_p90_deg']+opts['el_weight']*r['split_el_mae_deg']+r['boot_mean_deg']+r['hilo_mean_deg']
            result.append(dict(axis=axis,value=value,selected=chosen['name'] if chosen else 'E4_fallback_no_strict_noGT_improvement',score_difference=score(chosen)-score(rows[0]) if chosen else None))
    return result

def fit_or_record(n,y,w,init,a0,lam_e,out,label):
    try:return fit_traj(n,y,w,init,a0,1.,lam_e,out,label)
    except (RuntimeError,FloatingPointError) as exc:
        if '非有限' not in str(exc) and 'non-finite' not in str(exc).lower():raise
        dest=out/'runs'/label;dest.mkdir(parents=True,exist_ok=True)
        cert=dict(converged=False,path='numerical_failure',reason=str(exc));write(dest/'certificate.json',cert)
        write(dest/'config.json',dict(lam_a=1.,lam_e=lam_e,lam_rev=LAM_REV,e_halfwidth_deg=float(np.rad2deg(E_MAX)),status='NONFINITE',support_count=y.shape[1]))
        if not (dest/'history.json').exists():write(dest/'history.json',[])
        return np.full_like(init,np.nan),dict(certificate=cert,converged=False,final_loss=float('nan'),data_loss=float('nan'),penalties={},orbit_sign=orbit_sign(init))

def main():
    global E_MAX
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(103108);np.random.seed(103108)
    if a.self_test:self_test(out);return
    expected={Path('LIHAN/scripts/static_rgb_light_fit_r5b.py'):'ea6a90084d61d88ee87dbcd5230793de65f56e8b1f562b2125a68d90dd6d12e2',Path(r3.__file__):'0359461d5c4436698d4eee4d417a564ca273edf0091bfa87153f1ef0eb834517',BASE/'predictions.npz':'e888f8758301ded62d075d78de719a8ba1da61761a440d6275a1c2f3fec52757',BASE/'method_config.json':'0e43568348a0036fcd7f290c4b37dad8f3167462e9da398abf56f9edf517de04',OBS:OBS_HASH,PREREG:PREREG_HASH,ADDENDUM:ADDENDUM_HASH,ADDENDUM2:ADDENDUM2_HASH,ADDENDUM3:ADDENDUM3_HASH,ADDENDUM4:ADDENDUM4_HASH}
    for path,digest in expected.items():
        if sha(path)!=digest:raise RuntimeError(f'输入指纹不匹配: {path}')
    if sha(PLAN)!='7495cca5ac9daaad0e2a6f228caf8a0534107466b9b5144410f3466afe6a2976':raise RuntimeError('规划哈希不一致')
    model=Path('output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp');frozen=[model/'point_cloud/iteration_35000/point_cloud.ply',model/'deform/iteration_35000/deform.pth',R3_PRED,BASE/'runs/a1e+00_e1e+01_ends.npz',BASE/'candidate_table.json',Path(__file__)]
    a.observations=OBS
    write(out/'launch.json',dict(pid=os.getpid(),argv=sys.argv,plan_sha256=sha(PLAN),code_sha256=sha(__file__),inputs={**{str(k):v for k,v in expected.items()},**{str(k):sha(k) for k in frozen}},gt_read=False))
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
        repro={key:float(abs(results[key]-old[key]).max()) for key in ('E1','E2','E3')}
    with np.load(BASE/'predictions.npz') as old:
        repro_r5b={key:float(abs(results[key]-old[key]).max()) for key in ('E1','E2','E3')}
    write(out/'r3_reproduction.json',dict(max_abs_diff=repro,r5b_max_abs_diff=repro_r5b,tolerance=1e-6,source=str(R3_PRED),gt_read=False))
    if max(*repro.values(),*repro_r5b.values())>1e-6:
        write(out/'baseline_reproduction.json',dict(status='BASELINE_REPRODUCTION_FAILED',r3_reproduction=repro,r5b_reproduction=repro_r5b))
        raise RuntimeError(f'第三轮E1–E3未复现：{repro}')
    e3mae=ols_score(*scoredata,results['E3'])
    # ---- 各支撑集共享的E2→Stage1 ----
    supports={'full':(fi,results['E3'])};
    for key,idx in [('A',A),('B',B)]:
        init,voteinfo[key]=initial(sub(n,idx),y[:,idx],w[:,idx],relative[:,idx],'E2')
        supports[key]=(idx,r3.fit(sub(n,idx),y[:,idx],w[:,idx],init,False,.03,0.,out,f'stage1_{key}',False)[0])
    for seed in BOOT:
        pick=np.random.default_rng(seed).choice(fi,len(fi),replace=True);np.save(out/f'bootstrap_{seed}_sample.npy',pick)
        if not np.array_equal(pick,np.load(BASE/f'bootstrap_{seed}_sample.npy')):raise RuntimeError('bootstrap 索引与原档不符')
        init,voteinfo[f'boot{seed}']=initial(sub(n,pick),y[:,pick],w[:,pick],relative[:,pick],'E2')
        supports[f'boot{seed}']=(pick,r3.fit(sub(n,pick),y[:,pick],w[:,pick],init,False,.03,0.,out,f'stage1_boot{seed}',False)[0])
    dark=((relative[:,fi]<.15)&(w[:,fi]>0)).sum(1)/(w[:,fi]>0).sum(1);front=dark<=np.quantile(dark,.3);side=dark>=np.quantile(dark,.7)
    rows=[];ends_all={};infos_all={}
    for name,lam_e,h in CANDIDATES:
        E_MAX=np.deg2rad(h);ends={};infos={}
        for key,(idx,s1) in supports.items():
            original=a_from_z(s1[:,2].mean()) if h==5 else float(np.arctanh((np.deg2rad(5.)*np.tanh(float(np.arctanh(np.clip((np.arcsin(s1[:,2].mean())-E_C)/np.deg2rad(5.),-.9,.9)))))/E_MAX))
            ends[key],infos[key]=fit_or_record(sub(n,idx),y[:,idx],w[:,idx],s1,original,lam_e,out,f'{name}_{key}')
        for key,offset in [('hi',4.5),('lo',-4.5)]:
            a0=float(np.arctanh(offset/h));ends[key],infos[key]=fit_or_record(sub(n,fi),y[:,fi],w[:,fi],results['E3'],a0,lam_e,out,f'{name}_{key}')
        if any(not np.isfinite(v).all() for v in ends.values()):
            row=dict(name=name,lam_a=1.,lam_e=lam_e,halfwidth_deg=h,split_mean_deg=float('nan'),split_p90_deg=float('nan'),split_el_mae_deg=float('nan'),hilo_mean_deg=float('nan'),boot_mean_deg=float('nan'),body_heldout_mae=float('nan'),E3_body_heldout_mae=e3mae,view_dz_abs=float('nan'),saturation_frac=float('nan'),J=float('nan'),gates={f'G{i}':False for i in range(8)},all_gates=False,converged=False,certificates={k:v['certificate'] for k,v in infos.items()},failure='NONFINITE')
            rows.append(row);ends_all[name]=ends;infos_all[name]=infos;np.savez_compressed(out/'runs'/f'{name}_ends.npz',**ends);write(out/'candidate_table.json',dict(E3_body_heldout_mae=e3mae,candidates=rows))
            if name=='C00':write(out/'baseline_reproduction.json',dict(status='BASELINE_REPRODUCTION_FAILED',reason='NONFINITE'));raise RuntimeError('BASELINE_REPRODUCTION_FAILED')
            continue
        full=ends['full'];sp=compare(ends['A'],ends['B']);hl=compare(ends['hi'],ends['lo']);e=elev_deg(full);de=np.diff(e)
        ea=elev_deg(ends['A']);eb=elev_deg(ends['B']);corr=float(np.corrcoef(ea,eb)[0,1]) if ea.std()>1e-12 and eb.std()>1e-12 else None
        mask=abs(e-45)>=.95*h;longest=max(map(len, ''.join('1' if x else '0' for x in mask).split('0')))
        row=dict(name=name,lam_a=1.,lam_e=lam_e,halfwidth_deg=h,lam_rev=LAM_REV,split_mean_deg=sp['mean_deg'],split_p90_deg=sp['p90_deg'],split_el_mae_deg=float(abs(ea-eb).mean()),hilo_mean_deg=hl['mean_deg'],boot_each_mean_deg=[compare(ends[f'boot{s}'],full)['mean_deg'] for s in BOOT],view_dz_abs=float(abs(full[front,2].mean()-full[side,2].mean())),body_heldout_mae=ols_score(*scoredata,full),E3_body_heldout_mae=e3mae,full_penalties=infos['full']['penalties'],data_loss=infos['full']['data_loss'],converged=all(v['converged'] for v in infos.values()),certificates={k:v['certificate'] for k,v in infos.items()},final_loss={k:v['final_loss'] for k,v in infos.items()},r_AB=corr,quantiles_deg=np.quantile(e,[0,.05,.25,.5,.75,.95,1]).tolist(),rms_de_deg=float(np.sqrt(np.mean(de**2))),actual_boundary_frac=float(mask.mean()),actual_boundary_longest=longest,view_elevation_correlation=(float(np.corrcoef(dark,e)[0,1]) if dark.std()>1e-12 and e.std()>1e-12 else None),**traj_stats(full,infos['full']['orbit_sign']))
        row['boot_mean_deg']=float(np.mean(row['boot_each_mean_deg']));row['J']=row['split_mean_deg']+.5*row['split_p90_deg']+row['split_el_mae_deg']+row['boot_mean_deg']+row['hilo_mean_deg']
        row['gates']={g:bool(f(row)) for g,f in GATES.items()};row['all_gates']=all(row['gates'].values()) and all(np.isfinite(row[k]) for k in ['J','body_heldout_mae','view_dz_abs']) and all(np.isfinite(v).all() for v in ends.values()) and all(np.isfinite(v['final_loss']) and np.isfinite(v['data_loss']) and all(np.isfinite(x) for x in v['penalties'].values()) for v in infos.values())
        rows.append(row);ends_all[name]=ends;infos_all[name]=infos;np.savez_compressed(out/'runs'/f'{name}_ends.npz',**ends)
        write(out/'candidate_table.json',dict(E3_body_heldout_mae=e3mae,candidates=rows))
        if name=='C00':
            old=np.load(BASE/'runs/a1e+00_e1e+01_ends.npz');oldpred=np.load(BASE/'predictions.npz');oldrows=json.loads((BASE/'candidate_table.json').read_text())['candidates'];oldrow=next(r for r in oldrows if r['name']=='a1e+00_e1e+01')
            differences={k:float(abs(ends[k]-old[k]).max()) for k in ends};differences['prediction']=float(abs(full-oldpred['E4']).max())
            baseline_ok=max(differences.values())<=1e-6 and row['gates']==oldrow['gates'] and row['all_gates']==oldrow['all_gates'] and abs(row['J']-4.576831321876217)<=1e-4 and abs(row['body_heldout_mae']-.030837243629347505)<=1e-8
            write(out/'baseline_reproduction.json',dict(status='PASS' if baseline_ok else 'BASELINE_REPRODUCTION_FAILED',differences=differences,old_gates=oldrow['gates'],new_gates=row['gates'],J=row['J'],body_heldout_mae=row['body_heldout_mae']))
            if not baseline_ok:raise RuntimeError('BASELINE_REPRODUCTION_FAILED')
    chosen=select_t01(rows);name=chosen['name'] if chosen else 'E4_fallback_no_strict_noGT_improvement'
    with np.load(BASE/'predictions.npz') as old:results['E4']=ends_all[name]['full'].copy() if chosen else old['E4'].copy()
    for key,value in results.items():
        if value.shape!=(150,3) or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value,axis=1),1.,rtol=0,atol=1e-10):raise RuntimeError(f'无效预测: {key}')
    np.savez_compressed(out/'predictions.npz',frames=frames,**results)
    sens=sensitivity(rows);write(out/'sensitivity.json',dict(main_selection=name,rows=sens));ranking=[r['name'] for r in sorted((r for r in rows[1:] if r['all_gates'] and np.isfinite(r['J']) and r['J']<rows[0]['J']-.02),key=lambda r:(r['J'],r['body_heldout_mae'],r['name']))];write(out/'selection.json',dict(selected=name,selected_row=chosen,candidates=rows,eligible_by_J=ranking,rule='G0-G7; J < C00 - .02; window .02 then body MAE and candidate ID'))
    write(out/'method_config.json',dict(final_method='E4',selected=name,fit_gaussians=len(fi),heldout_gaussians=len(vi),gt_read=False,coordinate_system='world',direction_convention='surface_to_light',directional_intensity=1,candidates=CANDIDATES,plan_sha256=sha(PLAN),input_sha256=sha(OBS),r3_reproduction=repro,votes=voteinfo))
    make_figures(out,d,n,y,w,relative,fi,results,voteinfo,rows,ends_all,sens,name)
    write(out/'fit_complete.json',dict(status='T01_FIT_COMPLETE_AWAITING_REVIEW',selected=name,gt_read=False))

def make_figures(out,d,n,y,w,relative,fi,results,voteinfo,rows,ends_all,sens,name):
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
        sc=axs[1,col].scatter(pix[ok,0],pix[ok,1],s=4,c=residual[t,ok],cmap='coolwarm',vmin=-.15,vmax=.15);axs[1,col].set_title(f'E4 ({name}) KKT residual');fig.colorbar(sc,ax=axs[1,col])
        for ax in axs[:,col]:ax.set_aspect('equal');ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'01_percentile_residual.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row in rows:
        key=row['name'];ends=ends_all[key]
        for col,fn in enumerate((lambda z:np.rad2deg(np.unwrap(np.arctan2(z[:,1],z[:,0]))),elev_deg)):
            axes[0,col].plot(fn(ends['full']),label=key)
            for subset in ('A','B'):axes[1,col].plot(fn(ends[subset]),label=f'{key} {subset}',alpha=.7)
    for col,fn in enumerate((lambda z:np.rad2deg(np.unwrap(np.arctan2(z[:,1],z[:,0]))),elev_deg)):axes[0,col].plot(fn(results['E4']),label='selected',lw=2,ls='--')
    for ax in axes.flat:ax.legend(fontsize=6,ncol=3);ax.set_xlabel('frame');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('azimuth');axes[0,1].set_ylabel('elevation');axes[1,0].set_ylabel('A/B azimuth');axes[1,1].set_ylabel('A/B elevation');fig.tight_layout();fig.savefig(out/'02_direction_split.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9));ids=[r['name'] for r in rows]
    gates=np.array([[int(r['gates'][f'G{i}']) for i in range(8)] for r in rows]);axes[0,0].imshow(gates,aspect='auto',vmin=0,vmax=1);axes[0,0].set_yticks(range(len(ids)),ids);axes[0,0].set_xticks(range(8),[f'G{i}' for i in range(8)])
    axes[0,1].bar(ids,[r['body_heldout_mae'] for r in rows]);axes[0,1].set_ylabel('body heldout MAE')
    axes[1,0].bar(ids,[r['J'] for r in rows]);axes[1,0].set_ylabel('J')
    labels=[f"{x['axis']}={x['value']}" for x in sens];vals=[ids.index(x['selected']) if x['selected'] in ids else -1 for x in sens];axes[1,1].scatter(range(len(vals)),vals);axes[1,1].set_xticks(range(len(vals)),labels,rotation=90,fontsize=6);axes[1,1].set_yticks([-1]+list(range(len(ids))),['fallback']+ids)
    fig.tight_layout();fig.savefig(out/'03_holdout_stability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row in rows:
        key=row['name'];ends=ends_all[key]
        if key==(name if name in ends_all else 'C00'):
            for tag in ('full','hi','lo','boot103108','boot103109'):axes[0,0].plot(elev_deg(ends[tag]),label=f'{key}-{tag}')
        axes[0,1].bar(key,row['hilo_mean_deg']);axes[1,0].bar(key,row['boot_mean_deg'])
    cert=np.array([[int(rows[i]['certificates'][k]['converged']) for k in ['full','A','B','boot103108','boot103109','hi','lo']] for i in range(len(rows))]);axes[1,1].imshow(cert,aspect='auto',vmin=0,vmax=1);axes[1,1].set_yticks(range(len(ids)),ids);axes[1,1].set_xticks(range(7),['full','A','B','boot1','boot2','hi','lo'],rotation=45)
    axes[0,0].legend(fontsize=7);axes[0,0].set_ylabel('elevation');axes[0,1].set_ylabel('hi/lo mean');axes[1,0].set_ylabel('bootstrap mean');fig.tight_layout();fig.savefig(out/'04_hilo_convergence.png',dpi=150);plt.close(fig)

def self_test(out):
    from scipy.optimize import nnls
    rng=np.random.default_rng(103108);x=rng.random((60,40));y=rng.random((60,40));w=rng.random((60,40));x[:,3]=.5;w[:,4]=0
    tx=torch.tensor(x,dtype=torch.float64,requires_grad=True);rho,b=kkt(tx,torch.tensor(y),torch.tensor(w));err=0.
    for g in range(40):
        design=np.column_stack([x[:,g],np.ones(60)])*np.sqrt(w[:,g,None]);target=y[:,g]*np.sqrt(w[:,g]);err=max(err,float(np.max(abs(nnls(design,target)[0]-np.array([rho[g].item(),b[g].item()])))))
    assert err<2e-5
    th=torch.linspace(0,2*np.pi,150,dtype=torch.float64);e=torch.deg2rad(45+3*torch.sin(th));p=penalties(th,e,1.);data=torch.tensor(.123,dtype=torch.float64);strong=data+p['a1']+p['a2']+10*(p['e1']+p['e2'])+.1*p['rev'];weak=data+p['a1']+p['a2']+.1*p['rev'];assert torch.allclose(strong-weak,10*(p['e1']+p['e2']),rtol=0,atol=1e-15)
    eye=torch.eye(150,dtype=torch.float64);D1=torch.diff(eye,dim=0);D2=torch.diff(D1,dim=0);L2=0./(150*1e-4);M=eye+L2*D1.T@D1+L2**2*D2.T@D2;ev,U=torch.linalg.eigh(M);Pa=U@torch.diag(ev.rsqrt())@U.T;assert torch.allclose(Pa,eye,rtol=0,atol=1e-14)
    ga=torch.zeros(150,dtype=torch.float64,requires_grad=True);gp=th.detach().clone().requires_grad_();q=penalties(gp,E_C+np.deg2rad(5.)*torch.tanh(ga),1.);z=q['a1']+q['a2']+0*(q['e1']+q['e2'])+.1*q['rev'];z.backward();assert torch.isfinite(ga.grad).all() and torch.isfinite(gp.grad).all()
    global E_MAX
    for _,_,h in CANDIDATES:
        E_MAX=np.deg2rad(h);a=torch.atanh((e-E_C)/E_MAX).detach().requires_grad_();restored=elev(a);assert torch.max(abs(restored-e))<1e-14
        loss=sum(penalties(th,restored,1.).values());loss.backward();assert torch.isfinite(a.grad).all()
        extreme=torch.rad2deg(elev(torch.tensor([-1000.,1000.],dtype=torch.float64)));assert extreme[0]>=45-h-1e-10 and extreme[1]<=45+h+1e-10
        for start in [-4.5,4.5]:assert abs(float(torch.rad2deg(elev(torch.tensor(np.arctanh(start/h),dtype=torch.float64))))-45-start)<1e-10
    E_MAX=np.deg2rad(5.)
    def row(name,j,mae,ok=True):return dict(name=name,split_mean_deg=j,split_p90_deg=0.,split_el_mae_deg=0.,boot_mean_deg=0.,hilo_mean_deg=0.,body_heldout_mae=mae,all_gates=ok)
    rows=[row('C00',2.,1.),row('C01',1.98,.8),row('C02',1.9,.9),row('C03',1.91,.7)]
    assert select_t01(rows)['name']=='C03';assert select_t01(rows[:2]) is None
    assert select_t01([rows[0],row('C01',float('nan'),1.)]) is None
    before=select_t01(rows)['name'];sensitivity(rows);assert select_t01(rows)['name']==before
    # 同物理轨迹跨带宽的数据与方位正则一致。
    normals=torch.tensor(rng.normal(size=(20,3)),dtype=torch.float64);normals=normals/torch.linalg.vector_norm(normals,dim=1)[:,None]
    reference=None
    for _,_,h in CANDIDATES:
        E_MAX=np.deg2rad(h);aa=torch.atanh((e-E_C)/E_MAX);ll=compose(torch.stack([torch.cos(th),torch.sin(th)],1),elev(aa));shade=torch.relu(ll@normals.T);r,b=kkt(shade,torch.ones_like(shade)*.3,torch.ones_like(shade));dat=((shade*r+b-.3)**2).mean();q=penalties(th,elev(aa),1.);pair=np.array([float(dat),float(q['a1']+q['a2']+.1*q['rev'])])
        if reference is None:reference=pair
        assert np.max(abs(reference-pair))<1e-12
    # 小规模完整执行，并对 C00 比较原 r5b 路径。
    import LIHAN.scripts.static_rgb_light_fit_r5b as orig
    global TC_STEPS,TC_OUTER
    TC_STEPS=orig.TC_STEPS=2;TC_OUTER=orig.TC_OUTER=6;E_MAX=np.deg2rad(5.)
    tt=12;phi=np.linspace(0,2*np.pi,tt);init=np.column_stack([np.cos(phi)*np.cos(E_C),np.sin(phi)*np.cos(E_C),np.full(tt,np.sin(E_C))]);nn=rng.normal(size=(20,3));nn/=np.linalg.norm(nn,axis=1)[:,None];sy=np.maximum(init@nn.T,0)*.5+.1;sw=np.ones_like(sy)
    kwargs=dict(n=nn,y=sy,w=sw,init=init,a0=0.,lam_a=1.,out=out,steps=2,outer=6,outer_ext=12)
    a0,_=orig.fit_traj(lam_e=10.,label='original',**kwargs);a1,_=fit_traj(lam_e=10.,label='c00',**kwargs);parity=float(abs(a0-a1).max());assert parity<=1e-12
    zero,info=fit_traj(lam_e=0.,label='zero_smoothing',**kwargs);assert np.isfinite(zero).all() and np.isfinite(info['final_loss'])
    write(out/'numeric_tests.json',dict(status='PASS',nnls_max_parameter_difference=err,c00_max_abs_difference=parity,zero_smoothing_finite=True,gt_read=False))

if __name__=='__main__':main()
