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
    info=dict(penalties=pp,orbit_sign=s,final_loss=float(loss),certificate=cert,converged=cert['converged'],elev_deg=eld.tolist())
    np.save(dest/'direction.npy',end);write(dest/'history.json',hist);write(dest/'certificate.json',cert)
    write(dest/'config.json',dict(lam_a=lam_a,lam_e=lam_e,lam_rev=LAM_REV,adam_steps=steps,lr_phi=LR_PHI,lr_a=LR_A,drops=list(DROPS),e_center_deg=45.,e_halfwidth_deg=5.,device=device,dtype='float64',support_count=y.shape[1],final_loss=float(loss),certificate=cert,penalties=pp))
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

def tests(out):
    from scipy.optimize import nnls
    rng=np.random.default_rng(103108);x=rng.random((60,40));y=rng.random((60,40));w=rng.random((60,40))
    y[:,0]=.4-.2*x[:,0];y[:,1]=2*x[:,1]-.5;y[:,2]=.2+.5*x[:,2];x[:,3]=.5;w[:,4]=0
    tx=torch.tensor(x,dtype=torch.float64,requires_grad=True);r,b=kkt(tx,torch.tensor(y),torch.tensor(w));maxerr=0.
    for g in range(40):
        design=np.column_stack([x[:,g],np.ones(60)])*np.sqrt(w[:,g,None]);target=y[:,g]*np.sqrt(w[:,g])
        maxerr=max(maxerr,float(np.max(abs(nnls(design,target)[0]-np.array([r[g].item(),b[g].item()])))))
    assert maxerr<2e-5,maxerr
    th=torch.arange(150,dtype=torch.float64)*.04;e=torch.full((150,),.8,dtype=torch.float64)
    zero={k:float(v) for k,v in penalties(th,e,1.).items()};assert all(v<1e-24 for v in zero.values()),zero
    rev=penalties(th.flip(0),e,1.)['rev'];assert rev>1e-4
    band=torch.rad2deg(elev(torch.tensor([-1e3,-3.,0.,3.,1e3],dtype=torch.float64)));assert band.min()>=40-1e-9 and band.max()<=50+1e-9 and abs(band[2]-45)<1e-12
    # 合成：150帧变速环绕(平均2.4°/帧) + 带宽内非周期平滑随机游走仰角(±4°内)
    T=150;nn=unit(rng.normal(size=(600,3)));speed=2.4*(1+.3*np.sin(np.linspace(0,3,T)));phi=np.deg2rad(np.cumsum(speed))
    walk=np.cumsum(rng.normal(0,.15,T));walk=np.convolve(walk,np.ones(9)/9,'same');walk=walk-walk.mean();walk=3.9*walk/abs(walk).max()
    et=np.deg2rad(45+walk);truth=np.column_stack([np.cos(et)*np.cos(phi),np.cos(et)*np.sin(phi),np.sin(et)])
    sy=np.maximum(truth@nn.T,0)*rng.uniform(.3,.8,600)+rng.uniform(.03,.12,600);sw=np.ones_like(sy)
    init=np.column_stack([unit(truth[:,:2]+rng.normal(0,.03,(T,2)))*np.cos(E_C),np.full(T,np.sin(E_C))])
    end,info=fit_traj(nn,sy,sw,init,0.,1e-2,1e-2,out,'synthetic');err=compare(end,truth);assert err['mean_deg']<1.,err
    assert info['converged'],info['certificate']
    hi,ihi=fit_traj(nn,sy,sw,init,np.arctanh(.9),1e-2,1e-2,out,'synthetic_hi');lo,ilo=fit_traj(nn,sy,sw,init,-np.arctanh(.9),1e-2,1e-2,out,'synthetic_lo')
    assert ihi['converged'] and ilo['converged'],(ihi['certificate'],ilo['certificate'])
    hl=compare(hi,lo);assert hl['mean_deg']<.1,hl
    tcend,itc=fit_traj(nn,sy,sw,init,np.arctanh(.9),1e-2,1e-2,out,'synthetic_tc',force_tc=True);assert itc['converged'] and itc['certificate']['path']=='lbfgs+TC',itc['certificate']
    assert compare(tcend,end)['mean_deg']<.1,compare(tcend,end)
    ez=np.rad2deg(np.arcsin(end[:,2]));assert ez.min()>40 and ez.max()<50
    end2,_=fit_traj(nn,sy,sw,init,0.,1e-2,1e-2,out,'synthetic_repeat');assert np.allclose(end,end2,atol=1e-8)
    gp=torch.zeros(150,dtype=torch.float64,requires_grad=True);ga=torch.zeros(150,dtype=torch.float64,requires_grad=True)
    tot=sum(penalties(th+gp,elev(ga),1.).values());tot.backward();assert torch.isfinite(gp.grad).all() and torch.isfinite(ga.grad).all()
    write(out/'numeric_tests.json',dict(status='PASS',nnls_max_parameter_difference=maxerr,uniform_orbit_penalties=zero,reversal_penalty=float(rev),band_deg=band.tolist(),synthetic=err,synthetic_hilo=hl,synthetic_certificate=info['certificate'],synthetic_tc_certificate=itc['certificate'],synthetic_elev_range_deg=[float(np.rad2deg(et.min())),float(np.rad2deg(et.max()))],deterministic_repeat=True,gt_read=False))
    print('R5B_SELF_TEST_PASS',json.dumps(err),json.dumps(hl),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=Path,default=OBS);p.add_argument('--self-test',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(103108);np.random.seed(103108)
    if a.self_test:tests(out);return
    assert sha(PREREG)==PREREG_HASH,sha(PREREG);assert sha(ADDENDUM)==ADDENDUM_HASH,sha(ADDENDUM);assert sha(ADDENDUM2)==ADDENDUM2_HASH,sha(ADDENDUM2);assert sha(ADDENDUM3)==ADDENDUM3_HASH,sha(ADDENDUM3);assert sha(ADDENDUM4)==ADDENDUM4_HASH,sha(ADDENDUM4);assert sha(a.observations)==OBS_HASH
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
    if a.smoke:
        # P3 收敛冒烟：只在 full 支撑上跑候选 a1e-01_e1e+00，只记录证书/迭代数/耗时，不计算任何G1–G7量
        t0=time.time();_,inf=fit_traj(sub(n,fi),y[:,fi],w[:,fi],results['E3'],a_from_z(results['E3'][:,2].mean()),.1,1.,out,'smoke_full')
        write(out/'smoke_result.json',dict(certificate=inf['certificate'],converged=inf['converged'],final_loss=inf['final_loss'],seconds=time.time()-t0,gt_read=False));print('R5B_SMOKE',inf['converged'],flush=True);return
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
        row=dict(name=name,lam_a=lam_a,lam_e=lam_e,lam_rev=LAM_REV,split_mean_deg=sp['mean_deg'],split_median_deg=sp['median_deg'],split_p90_deg=sp['p90_deg'],split_el_mae_deg=float(abs(elev_deg(ends['A'])-elev_deg(ends['B'])).mean()),hilo_mean_deg=hl['mean_deg'],boot_each_mean_deg=[compare(ends[f'boot{s}'],full)['mean_deg'] for s in BOOT],view_dz_abs=float(abs(full[front,2].mean()-full[side,2].mean())),body_heldout_mae=ols_score(*scoredata,full),all_heldout_mae=ols_score(sub(n,allvi),y[:,allvi],w[:,allvi],full),E3_body_heldout_mae=e3mae,full_penalties=infos['full']['penalties'],converged=all(v['converged'] for v in infos.values()),certificates={k:v['certificate'] for k,v in infos.items()},final_loss={k:v['final_loss'] for k,v in infos.items()},r_AB=float(np.corrcoef(elev_deg(ends['A'])-elev_deg(ends['A']).mean(),elev_deg(ends['B'])-elev_deg(ends['B']).mean())[0,1]),**traj_stats(full,infos['full']['orbit_sign']))
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
    config=dict(final_method='E4',selected=selected_name,selection_tier=tier,selected_row=selected,gt_read=False,coordinate_system='world',direction_convention='surface_to_light',directional_intensity=1,prereg_sha256=PREREG_HASH,input_sha256=OBS_HASH,fit_gaussians=len(fi),heldout_gaussians=len(vi),elevation_center_deg=45.,elevation_halfwidth_deg=5.,stageT=dict(adam_steps=STEPS,lr_phi=LR_PHI,lr_a=LR_A,drops=list(DROPS),lbfgs=dict(outer=OUTER,ext=OUTER_EXT,iter=LB_ITER,win=WIN),loss='L2 KKT varpro + 1e-5 coef reg',lam_a_grid=LAM_A,lam_e_grid=LAM_E,lam_rev=LAM_REV),J='split_mean+0.5*split_p90+split_el_mae+boot_mean+hilo_mean',gates='G0 converged; G1 split_mean<=3.5&split_p90<=6; G2 hilo<=0.5; G3 boot<=2; G4 body<=1.015*E3; G5 sat<=0.2; G6 rev<=2; G7 view_dz<=0.03',tie_rule='J within 0.02 -> body_mae, -lam_e, -lam_a',r3_reproduction=repro,votes=voteinfo,photometric_mae={k:ols_score(*scoredata,l) for k,l in results.items()},information_limitations='稳定性、轨迹平滑与留出残差均为无GT经验准则，不证明GT精度')
    write(out/'method_config.json',config)
    figures(out,d,n,y,w,relative,fi,results,voteinfo,rows,ends_all,infos_all,selected_name)
    write(out/'fit_complete.json',dict(status='R5B_FIT_COMPLETE_AWAITING_REVIEW',selected=selected_name,tier=tier,pid=os.getpid(),gt_read=False))
    print('R5B_FIT_COMPLETE',selected_name,tier,flush=True)

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
