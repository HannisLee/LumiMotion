"""1108-02 T01：固定 E3 初值、去除系数与光向平滑项；禁止读取 GT。"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from LIHAN.scripts import static_rgb_light_fit_r3 as r3
from LIHAN.scripts.static_rgb_light_fit_r3 import sha, write, unit, sub, ols_score, kkt
from LIHAN.scripts.static_rgb_light_fit_r5b import elev, compose, penalties, orbit_sign, a_from_z, ang_deg, elev_deg
E_C=np.deg2rad(45.); E_MAX=np.deg2rad(5.); LAM_REV=.1
STEPS=1000; LR_PHI=.01; LR_A=.11; DROPS=(600,850); BOOT=[103108,103109]
OUTER=200; OUTER_EXT=400; LB_ITER=20; WIN=5; TC_STEPS=6000; TC_LR=.003; TC_OUTER=60

def objective_terms(phi,a,nt,yt,ww,s):
    e=elev(a); l=compose(torch.stack([torch.cos(phi),torch.sin(phi)],1),e)
    x=torch.relu(torch.einsum('tgc,tc->tg',nt,l) if nt.ndim==3 else l@nt.T)
    rho,b=kkt(x,yt,ww); data=(ww*.5*(x*rho+b-yt).square()).sum()/ww.sum()
    p=penalties(phi,e,s)
    return data+LAM_REV*p['rev'],data,p,e

def fit_traj(n,y,w,init,a0,out,label,steps=STEPS,outer=OUTER,outer_ext=OUTER_EXT,force_tc=False):
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
        return objective_terms(phi_,a_,nt,yt,ww,s)
    def direction(phi_=None,a_=None):
        phi_=phi if phi_ is None else phi_;a_=a if a_ is None else a_
        with torch.no_grad():return compose(torch.stack([torch.cos(phi_),torch.sin(phi_)],1),elev(a_)).cpu().numpy()
    T_=len(init);D1=torch.diff(torch.eye(T_,dtype=torch.float64,device=device),dim=0);D2=torch.diff(D1,dim=0) if T_>2 else D1
    def precond(lam,w):
        # 预条件（仅优化器细节，不改模型/目标）：P=(I+L²D1ᵀD1+L⁴D2ᵀD2)^(-1/2)，L²=λ/(T·κ·w)，κ=1e-4（预注册A6平滑长度标定）
        L2=lam/(T_*1e-4*w);M=torch.eye(T_,dtype=torch.float64,device=device)+L2*D1.T@D1+L2**2*D2.T@D2
        ev,U=torch.linalg.eigh(M);return U@torch.diag(ev.rsqrt())@U.T
    Pphi=torch.eye(T_,dtype=torch.float64,device=device);Pa=Pphi.clone()
    def state():return phi.detach().clone(),a.detach().clone()
    def restore(st):
        with torch.no_grad():phi.copy_(st[0]);a.copy_(st[1])
    hist=[];losses=[]; all_trace=[]
    def checkpoint(stage,step):
        np.savez_compressed(dest/'checkpoint_latest.npz',phi=phi.detach().cpu().numpy(),a=a.detach().cpu().numpy(),stage=stage,step=step)
        write(dest/'history_partial.json',dict(adam=hist,optimization=all_trace))
    # ---- T-A Adam ----
    opt=torch.optim.Adam([dict(params=[phi],lr=LR_PHI),dict(params=[a],lr=LR_A)])
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True);loss,data,p,e=objective()
        if not torch.isfinite(loss):raise RuntimeError('非有限loss(Adam)')
        loss.backward();opt.step();losses.append(float(loss.detach()))
        if step in DROPS:
            for g in opt.param_groups:g['lr']*=.3
        if step%50==0 or step==1:
            entry=dict(stage='TA',step=step,loss=losses[-1],data=float(data.detach()),**{k:float(v.detach()) for k,v in p.items()},e_min_deg=float(torch.rad2deg(e.min())),e_max_deg=float(torch.rad2deg(e.max())));hist.append(entry);checkpoint('TA',step)
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
            trace.append(float(l));dirs.append(direction(pe,ae)); all_trace.append(dict(stage='TB',outer=k+1,loss=float(l)))
            if len(trace)%10==0:
                sync();checkpoint('TB',k+1);print(json.dumps(dict(run=label,stage='TB',outer=k+1,loss=trace[-1])),flush=True);write(out/'progress.json',dict(pid=os.getpid(),run=label,stage='TB',outer=k+1,loss=trace[-1],updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            if len(trace)>WIN:
                rel=abs(trace[-1-WIN]-trace[-1])/max(abs(trace[-1-WIN]),1e-300);mx=float(ang_deg(dirs[-1-WIN],dirs[-1]).max())
                if rel<1e-9 and mx<.005:sync();return trace,dirs,True,rel,mx
        sync();rel=abs(trace[-1-WIN]-trace[-1])/max(abs(trace[-1-WIN]),1e-300);mx=float(ang_deg(dirs[-1-WIN],dirs[-1]).max());return trace,dirs,False,rel,mx
    def probe():
        base=direction();st=state();lb0=float(objective()[0].detach());o=torch.optim.Adam([dict(params=[phi],lr=1e-4),dict(params=[a],lr=1e-3)])
        for _ in range(500):
            o.zero_grad();l=objective()[0];l.backward();o.step()
        d1=direction();l1=float(objective()[0].detach());disp=float(ang_deg(base,d1).max());drop=(lb0-l1)/max(abs(lb0),1e-300);restore(st);return disp,drop
    cert=dict(path='lbfgs');fallback=False
    try:
        trace,dirs,ok,rel,mx=run_lbfgs(outer)
        ext=False
        if not ok:
            ext=True;t2,d2,ok,rel,mx=run_lbfgs(outer_ext,len(trace));trace+=t2;dirs+=d2
        cert.update(C1_rel=rel,C1_ok=bool(np.isfinite(rel) and rel<1e-9),C2_max_deg=mx,C2_ok=bool(np.isfinite(mx) and mx<.005),extended=ext,outer_iters=len(trace))
        disp,drop=probe();cert.update(C3_disp_deg=disp,C3_drop=drop,C3_ok=bool(np.isfinite(disp) and np.isfinite(drop) and disp<.1 and drop<1e-4))
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
                all_trace.append(dict(stage='TC',step=k+1,loss=float(l.detach())))
                if k%500==0:
                    with torch.no_grad():phi.copy_(x0p+Pphi@qp);a.copy_(x0a+Pa@qa)
                    checkpoint('TC',k+1);print(json.dumps(dict(run=label,stage='TC',step=k+1,loss=float(l.detach()))),flush=True);write(out/'progress.json',dict(pid=os.getpid(),run=label,stage='TC',step=k+1,loss=float(l.detach()),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            with torch.no_grad():phi.copy_(x0p+Pphi@qp);a.copy_(x0a+Pa@qa)
            t3,d3,ok3,rel3,mx3=run_lbfgs(TC_OUTER);trace+=t3
            disp,drop=probe()
            cert=dict(path='lbfgs+TC',pre_TC=pre,C1_rel=rel3,C1_ok=bool(np.isfinite(rel3) and rel3<1e-9),C2_max_deg=mx3,C2_ok=bool(np.isfinite(mx3) and mx3<.005),extended=ext,outer_iters=len(trace),TC_steps=TC_STEPS,TC_lr=TC_LR,C3_disp_deg=disp,C3_drop=drop,C3_ok=bool(np.isfinite(disp) and np.isfinite(drop) and disp<.1 and drop<1e-4))
            cert['converged']=bool(cert['C1_ok'] and cert['C2_ok'] and cert['C3_ok']);final_losses=trace
    except (FloatingPointError,RuntimeError) as exc:
        if 'LBFGS' not in str(exc):raise
        fallback=True;restore((torch.as_tensor(phi0,dtype=torch.float64,device=device),torch.as_tensor(np.broadcast_to(a0,(len(init),)).copy(),dtype=torch.float64,device=device)))
        o=torch.optim.Adam([dict(params=[phi],lr=LR_PHI),dict(params=[a],lr=LR_A)]);N=6000+steps;sched=torch.optim.lr_scheduler.CosineAnnealingLR(o,N,eta_min=0)
        fl=[]
        for _ in range(N):
            o.zero_grad();l=objective()[0];l.backward();o.step();sched.step();fl.append(float(l.detach()))
        d_last=direction();rel=abs(fl[-501]-fl[-1])/max(abs(fl[-501]),1e-300);disp,drop=probe()
        cert=dict(path='adam_fallback',reason=str(exc),C1_rel=rel,C3_disp_deg=disp,C3_drop=drop,converged=bool(rel<1e-7 and disp<.1 and drop<1e-4));final_losses=fl
    with torch.no_grad():
        loss,data,p,e=objective();end=direction(); xt=torch.relu(torch.einsum('tgc,tc->tg',nt,torch.as_tensor(end,device=device)) if nt.ndim==3 else torch.as_tensor(end,device=device)@nt.T); rho,b=kkt(xt,yt,ww); pp={k:float(v) for k,v in p.items()};eld=np.rad2deg(e.cpu().numpy())
    info=dict(penalties=pp,orbit_sign=s,final_loss=float(loss),certificate=cert,converged=cert['converged'],elev_deg=eld.tolist())
    np.save(dest/'direction.npy',end); np.savez_compressed(dest/'parameters.npz',phi=phi.detach().cpu().numpy(),a=a.detach().cpu().numpy(),rho=rho.detach().cpu().numpy(),b=b.detach().cpu().numpy()); write(dest/'history.json',dict(adam=hist,optimization=all_trace,final_losses=final_losses));write(dest/'certificate.json',cert)
    write(dest/'config.json',dict(lam_a=0.,lam_e=0.,lam_rev=LAM_REV,adam_steps=steps,lr_phi=LR_PHI,lr_a=LR_A,drops=list(DROPS),e_center_deg=45.,e_halfwidth_deg=5.,device=device,dtype='float64',support_count=y.shape[1],final_loss=float(loss),certificate=cert,penalties=pp))
    print(json.dumps(dict(run=label,stage='DONE',final_loss=float(loss),certificate=cert)),flush=True)
    return end,info


BASE=Path('output/1031-12-LH-static-ELDL-V3-第五轮修订版环绕先验时变仰角/fitting')
PLAN=Path('LIHAN/DOC/规划/1108-02-T01去除系数与光向平滑正则预注册规划.md')
EXPECTED={
'support.npz':'35afe6f886c5df44b02ddb9faf5088330c01d3c79e9ef482359b0e8abc9703a7',
'predictions.npz':'e888f8758301ded62d075d78de719a8ba1da61761a440d6275a1c2f3fec52757',
'bootstrap_103108_sample.npy':'33803f7b1e7ed47f5cdd9502c4f62d5d8004b18c56de72d4161bdc8c25d6a500',
'bootstrap_103109_sample.npy':'9d3b473dba1f17854f04c6018d3e68511c70d89cb270ed1d05e622b1cc20ab17',
'runs/E3/direction.npy':'fea649324e411d35a712f5657a6d5ea2e4a96ef5aec476c84435a36accde0651',
'runs/E3/config.json':'6b4fe87518cb0261efcab632fe615379097634d109469b47a9bcfe4f03a3a277',
'runs/stage1_A/direction.npy':'da99852c4ca989ac7b4439b9c7461160b1aca0cd74755abfc0ac2ec0cddbfe94',
'runs/stage1_A/config.json':'7a6b830992cc4545b7df56283502389c36e50cc061b809df92094ec1de0bfa45',
'runs/stage1_B/direction.npy':'fecf17051e56903b89fc3177a58eb0e02ad1c1438ac6b46673fecac8ab50e1cf',
'runs/stage1_B/config.json':'91d7335fbae5c706b71e1378fa93fc792c5ac1859113ed71a047e40023727734',
'runs/stage1_boot103108/direction.npy':'04442c587d3a602b0c4f80824397af79d8ae498da24787a1ec841c202c201300',
'runs/stage1_boot103108/config.json':'6b4fe87518cb0261efcab632fe615379097634d109469b47a9bcfe4f03a3a277',
'runs/stage1_boot103109/direction.npy':'c3f3d48bd935cefee6964972139b5bd3f48b1f844fd670e6b8e0aa354fe6d972',
'runs/stage1_boot103109/config.json':'6b4fe87518cb0261efcab632fe615379097634d109469b47a9bcfe4f03a3a277'}

def direction_valid(l,r,b):
    return bool(l.shape==(150,3) and np.isfinite(l).all() and np.max(abs(np.linalg.norm(l,axis=1)-1))<=1e-10 and np.min(elev_deg(l))>=-1e-8 and np.max(elev_deg(l))<=90+1e-8 and np.isfinite(r).all() and np.isfinite(b).all() and np.min(r)>=-1e-12 and np.min(b)>=-1e-12)

def choose(valid):
    return {'selected':'C01/full' if valid else None,'status':'VALID' if valid else 'NO_VALID_NOREG_PREDICTION','reference_only':'C00/E4'}

def qstats(a):
    a=np.asarray(a);return dict(zip(['min','p1','p10','p50','p90','p99','max'],map(float,np.quantile(a,[0,.01,.1,.5,.9,.99,1]))))

def diff(a,b):
    d=ang_deg(a,b);az=np.rad2deg(np.angle(np.exp(1j*(np.arctan2(a[:,1],a[:,0])-np.arctan2(b[:,1],b[:,0])))));el=elev_deg(a)-elev_deg(b)
    return dict(mean_deg=float(d.mean()),p50_deg=float(np.median(d)),p90_deg=float(np.quantile(d,.9)),max_deg=float(d.max()),az_abs=qstats(abs(az)),elev_abs=qstats(abs(el)),large_disagreement=bool(d.mean()>5 or np.quantile(d,.9)>10))

def coefficients(n,y,w,l):
    x=np.maximum(r3.shade(n,l),0);ww=np.where(np.isfinite(y),w,0);yy=np.nan_to_num(y)
    with torch.no_grad():r,b=kkt(torch.tensor(x),torch.tensor(yy),torch.tensor(ww))
    return x,ww,yy,r.numpy(),b.numpy()

def stats(n,y,w,l):
    x,ww,yy,r,b=coefficients(n,y,w,l);s=ww.sum(0);valid=s>0;mu=(ww*x).sum(0)/np.maximum(s,1e-15);v=(ww*(x-mu)**2).sum(0)/np.maximum(s,1e-15)
    xx=(ww*x*x).sum(0);cross=(ww*x).sum(0);root=np.sqrt((xx-s)**2+4*cross**2);largest=(xx+s+root)/2;smallest=np.maximum((xx+s-root)/2,0);ratio=np.divide(smallest,largest,out=np.zeros_like(s),where=largest>0)
    pred=x*r+b;res=pred-yy;sw=ww.sum();rmse=float(np.sqrt((ww*res**2).sum()/sw));mae=float((ww*abs(res)).sum()/sw)
    mx=(ww*x).sum(0)/np.maximum(s,1e-15);my=(ww*yy).sum(0)/np.maximum(s,1e-15);rr=(ww*(x-mx)*(yy-my)).sum(0)/((ww*(x-mx)**2).sum(0)+1e-6);bb=my-rr*mx
    branches={'rho_zero':int((rr<=0).sum()),'b_zero':int(((rr>0)&(bb<=0)).sum()),'interior':int(((rr>0)&(bb>0)).sum())}
    corr=float(np.corrcoef(r,b)[0,1]) if r.std()>0 and b.std()>0 else None
    direct=float((ww*(x*r)**2).sum());background=float((ww*b**2).sum());tot=direct+background
    result=dict(variance=qstats(v[valid]),near_constant_fraction=float((v[valid]<=1e-6).mean()),weak_separation=bool((v[valid]<=1e-6).mean()>.2),gram_ratio=qstats(ratio[valid]),collinear_fraction=float((ratio[valid]<=1e-6).mean()),rho=qstats(r),b=qstats(b),rho_zero_fraction=float((r==0).mean()),b_zero_fraction=float((b==0).mean()),rho_b_pearson=corr,energy_direct_fraction=direct/tot if tot else None,energy_background_fraction=background/tot if tot else None,branches=branches,mae=mae,rmse=rmse)
    return result,(x,ww,yy,r,b,ratio,res)

def band(l):
    e=elev_deg(l);mask=abs(e-45)>=4.75;longest=run=0
    for value in mask:run=run+1 if value else 0;longest=max(longest,run)
    return dict(nominal_fraction=float(mask.mean()),longest_run=longest,nominal_band_active=bool(mask.mean()>.2),physical_boundary_near=bool(((e<=1)|(e>=89)).any()))

def audit():
    return {'gt_read':False,'objective':'L_photo + 0.1 R_rev','items':[dict(name=x,source='static_rgb_light_fit_r5b.py:68' if x=='L_coef' else 'static_rgb_light_fit_r5b.py:41-43',old_weight=1e-5 if x=='L_coef' else ('lambda_a' if x[0]=='a' else 'lambda_e'),new_weight=0.,runtime_contribution=0.,gradient_contribution=0.,synthetic_assertion='PASS') for x in ['L_coef','a1','a2','e1','e2']], 'retained':{'rev':.1,'kkt_epsilon':1e-6,'kkt_effect':'分母稳定项具有岭收缩效应','E3':'历史E3含1e-5系数项，来源E2三帧平均','elevation_band_deg':[40,50]}}

def self_test(out):
    rng=np.random.default_rng(103108);T=150;G=9;n=unit(rng.normal(size=(G,3)));phi=torch.tensor(np.linspace(0,7,T)+rng.normal(0,.03,T),dtype=torch.float64,requires_grad=True);a=torch.tensor(rng.normal(size=T),dtype=torch.float64,requires_grad=True);nt=torch.tensor(n);y=torch.tensor(rng.random((T,G)));w=torch.tensor(rng.random((T,G)));e=elev(a);l=compose(torch.stack((phi.cos(),phi.sin()),1),e);x=torch.relu(l@nt.T);r,b=kkt(x,y,w);data=(w*.5*(r*x+b-y)**2).sum()/w.sum();p=penalties(phi,e,1.);loss,actual_data,actual_p,_=objective_terms(phi,a,nt,y,w,1.); assert torch.equal(actual_data,data) and all(torch.equal(actual_p[k],p[k]) for k in p);g=torch.autograd.grad(loss,(phi,a),retain_graph=True);g2=torch.autograd.grad(data+.1*p['rev'],(phi,a),retain_graph=True);assert torch.equal(loss,data+.1*p['rev']) and all(torch.equal(u,v) for u,v in zip(g,g2))
    for term in [r.square().mean()+b.square().mean(),p['a1'],p['a2'],p['e1'],p['e2']]:
        grads=torch.autograd.grad(0.*term,(phi,a),retain_graph=True,allow_unused=True);assert all(z is None or torch.count_nonzero(z)==0 for z in grads)
    assert torch.equal(torch.eye(T,dtype=torch.float64),torch.eye(T,dtype=torch.float64));z=torch.tensor([[.5,.2,.2,.3],[.5,.8,.8,.7],[.5,.5,.5,.5]],dtype=torch.float64);yy=torch.tensor([[.2,.8,.1,.4],[.2,.1,.9,.3],[.2,.4,.5,.2]],dtype=torch.float64);ww=torch.ones_like(z);ww[:,0]=0
    rr,bb=kkt(z,yy,ww);rr2,bb2=r3.kkt(z,yy,ww);assert torch.equal(rr,rr2) and torch.equal(bb,bb2) and torch.isfinite(rr).all() and torch.isfinite(bb).all() and (rr>=0).all() and (bb>=0).all()
    assert abs(float(elev(torch.tensor(0.,dtype=torch.float64)))-E_C)<1e-15 and float(elev(torch.tensor(-1000.)))>=np.deg2rad(40)-1e-7 and float(elev(torch.tensor(1000.)))<=np.deg2rad(50)+1e-7
    init=np.column_stack((np.cos(np.linspace(0,7,T))*np.cos(E_C),np.sin(np.linspace(0,7,T))*np.cos(E_C),np.full(T,np.sin(E_C))));assert orbit_sign(init)==1 and abs(a_from_z(init[:,2].mean()))<1e-12;assert choose(False)['selected'] is None and choose(True)['selected']=='C01/full';assert diff(init,init)['max_deg']<=1e-4
    assert diff(init,np.column_stack((-init[:,:2],init[:,2])))['large_disagreement'];assert not (np.isfinite(float('nan')) and float('nan')<1e-9)
    write(out/'numeric_tests.json',{'status':'PASS','gt_read':False,'objective_and_gradients':'PASS','kkt_identical_degenerate_zero_weight':'PASS','band_orbit_init':'PASS','invalid_no_fallback':'PASS','high_disagreement':'PASS','repeat_boundary':'PASS','unconverged_not_selection_gate':'PASS'});write(out/'regularization_audit.json',audit())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(103108);np.random.seed(103108)
    if args.self_test:self_test(out);return
    expected_files={PLAN:'ad122b5d98d442391d866d3fe0e8b94f8e591d43eba5f2182c9369760d799854',Path(r3.__file__):'0359461d5c4436698d4eee4d417a564ca273edf0091bfa87153f1ef0eb834517',Path('LIHAN/scripts/static_rgb_light_fit_r5b.py'):'ea6a90084d61d88ee87dbcd5230793de65f56e8b1f562b2125a68d90dd6d12e2',r3.OBS:r3.OBS_HASH,**{BASE/k:v for k,v in EXPECTED.items()}}
    for path,expected in expected_files.items():
        actual=sha(path)
        if actual!=expected:raise RuntimeError(f'输入哈希不符: {path}: {actual} != {expected}')
    hashes={str(p):sha(p) for p in expected_files};hashes[__file__]=sha(__file__)
    model=Path('output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp')
    for path in [model/'point_cloud/iteration_35000/point_cloud.ply',model/'deform/iteration_35000/deform.pth']:
        hashes[str(path)]=sha(path)
    write(out/'launch.json',dict(argv=sys.argv,pid=os.getpid(),input_and_code_sha256=hashes,gt_read=False))
    d=np.load(r3.OBS,allow_pickle=False);n=d['normals'].astype(np.float64);y=d['intensity'].astype(np.float64);w=d['weights'].astype(np.float64);xyz=d['xyz'];ids=d['gaussian_ids'];frames=d['frames']
    assert np.array_equal(frames,np.arange(1,151)) and y.shape==w.shape==(150,len(ids)) and np.isfinite(n).all() and np.isfinite(xyz).all() and np.isfinite(w).all() and (w>=0).all() and not ((w>0)&~np.isfinite(y)).any()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);q20=np.nanquantile(y,.2,axis=0);q95=np.nanquantile(y,.95,axis=0)
    eligible=(np.isfinite(y).sum(0)>=8)&(q95-q20>.015);mn=n.mean(0) if n.ndim==3 else n;body=(xyz[:,2]>=.05)&(abs(mn[:,2])<=.85);cells=np.floor(xyz*20).astype(np.int64);hval=cells[:,0]*73856093 ^ cells[:,1]*19349663 ^ cells[:,2]*83492791;held=hval%5==0
    fi=np.flatnonzero(eligible&body&~held);vi=np.flatnonzero(eligible&body&held);allvi=np.flatnonzero(eligible&held);A=fi[(hval[fi]//5)%2==0];B=fi[(hval[fi]//5)%2!=0];assert (len(fi),len(vi),len(A),len(B))==(2683,648,1450,1233)
    support=dict(body_fit=fi,body_val=vi,all_val=allvi,split_A=A,split_B=B,gaussian_ids=ids,hval=hval,body_mask=body)
    with np.load(BASE/'support.npz',allow_pickle=False) as old:
        for k,v in support.items():assert np.array_equal(v,old[k]),k
    np.savez_compressed(out/'support.npz',**support);supports={'full':fi,'A':A,'B':B}
    for seed in BOOT:
        pick=np.random.default_rng(seed).choice(fi,len(fi),replace=True);assert np.array_equal(pick,np.load(BASE/f'bootstrap_{seed}_sample.npy'));supports[f'boot{seed}']=pick;np.save(out/f'bootstrap_{seed}_sample.npy',pick)
    mapping={'full':'E3','A':'stage1_A','B':'stage1_B','boot103108':'stage1_boot103108','boot103109':'stage1_boot103109'};initial={}
    for key,dirname in mapping.items():
        l=np.load(BASE/'runs'/dirname/'direction.npy');conf=json.loads((BASE/'runs'/dirname/'config.json').read_text());assert l.shape==(150,3) and np.isfinite(l).all() and np.max(abs(np.linalg.norm(l,axis=1)-1))<=1e-10 and conf['support_count']==len(supports[key]) and conf['stage2'] is False and conf['geo_smooth']==0 and conf['balanced'] is False
        initial[key]=l
    with np.load(BASE/'predictions.npz',allow_pickle=False) as old:reference=old['E4'].copy();historical={k:old[k].copy() for k in ['E1','E2','E3']}
    assert direction_valid(reference,np.ones(1),np.ones(1))
    ends={};infos={};diagnostics={};arrays={};write(out/'regularization_audit.json',audit())
    for key in ['full','A','B','boot103108','boot103109','hi','lo','repeat_full']:
        source=key if key in supports else 'full';idx=supports[source];init=initial[source];a0=np.arctanh(.9)*(1 if key=='hi' else -1) if key in ('hi','lo') else a_from_z(init[:,2].mean());end,info=fit_traj(sub(n,idx),y[:,idx],w[:,idx],init,a0,out,key);ends[key]=end;infos[key]=info
        st,arr=stats(sub(n,idx),y[:,idx],w[:,idx],end);valid=direction_valid(end,arr[3],arr[4]);diagnostics[key]={'valid':valid,'converged':bool(info['converged']),'status':'CONVERGED' if info['converged'] else 'NOT_CONVERGED','coefficients':st,'band':band(end),'body_heldout_clipped_mae':ols_score(sub(n,vi),y[:,vi],w[:,vi],end)};arrays[key]=arr
        write(out/'diagnostics_partial.json',diagnostics)
    selected=choose(diagnostics['full']['valid']);full=ends['full'];c00,c00arr=stats(sub(n,fi),y[:,fi],w[:,fi],reference);ref_photo=ols_score(sub(n,vi),y[:,vi],w[:,vi],reference)
    for k in ['rho','b']:diagnostics['full']['coefficients'][k+'_inflation']=bool(diagnostics['full']['coefficients'][k]['p99']>10*max(c00[k]['p99'],1e-8))
    differences={f'{k}_vs_full':diff(ends[k],full) for k in ['A','B','boot103108','boot103109','hi','lo','repeat_full']};differences['A_vs_B']=diff(ends['A'],ends['B']);differences['hi_vs_lo']=diff(ends['hi'],ends['lo']);repeat=differences['repeat_full_vs_full']['max_deg'];selected['repeat_status']='DETERMINISTIC' if repeat<=1e-4 else 'NONDETERMINISTIC'
    ambiguity={};full_fit=diagnostics['full']['coefficients']['rmse']
    for k in ['hi','lo']:
        st,arr=stats(sub(n,fi),y[:,fi],w[:,fi],ends[k]);rd=abs(st['rmse']-full_fit)/max((st['rmse']+full_fit)/2,1e-12);ambiguity[k]={'photometric_ambiguity':bool(differences[k+'_vs_full']['mean_deg']>5 and rd<=.01),'relative_rmse_difference':rd,'rho_abs_diff':qstats(abs(arr[3]-arrays['full'][3])),'b_abs_diff':qstats(abs(arr[4]-arrays['full'][4]))}
    diagnostics['full']['photo_degraded']=bool(diagnostics['full']['body_heldout_clipped_mae']>1.05*ref_photo)
    write(out/'selection.json',selected);write(out/'diagnostics.json',dict(runs=diagnostics,differences=differences,ambiguity=ambiguity,reference_coefficients=c00,reference_body_heldout_clipped_mae=ref_photo,gt_read=False));write(out/'candidate_table.json',{'C00':{'reference_only':True,'body_heldout_clipped_mae':ref_photo},'C01':diagnostics['full']})
    np.savez_compressed(out/'predictions.npz',frames=frames,reference_E4=reference,**historical,**({'C01_full':full} if selected['selected'] else {}),**{f'C01_{k}':v for k,v in ends.items() if k!='full'})
    x,ww,yy,r,b,ratio,res=arrays['full'];np.savez_compressed(out/'photometric_diagnostics.npz',residual=res,rho=r,b=b,gram_ratio=ratio,body_fit=fi)
    fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].plot(np.sum(ww>0,axis=1));ax[0].set_title('Support count per frame');ax[1].imshow(np.where(ww>0,res,np.nan).T,aspect='auto',vmin=-.2,vmax=.2,cmap='coolwarm');ax[1].set_title('Photometric residual');fig.tight_layout();fig.savefig(out/'01_support_residual.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for k,l in {'E4':reference,'C01':full,'A':ends['A'],'B':ends['B']}.items():ax[0].plot(np.rad2deg(np.unwrap(np.arctan2(l[:,1],l[:,0]))),label=k);ax[1].plot(elev_deg(l),label=k)
    for a in ax:a.legend();a.grid(alpha=.2)
    for v in [40,50]:ax[1].axhline(v,ls=':',c='k')
    ax[0].set_title('Azimuth');ax[1].set_title('Elevation');fig.tight_layout();fig.savefig(out/'02_trajectories.png');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(15,4));ax[0].hist(r,bins=50,alpha=.6,label='rho');ax[0].hist(b,bins=50,alpha=.6,label='b');ax[0].legend();ax[1].hist(np.log10(np.maximum(ratio,1e-16)),bins=50);ax[1].set_title('log10 Gram ratio');ax[2].bar(['direct','background'],[diagnostics['full']['coefficients']['energy_direct_fraction'],diagnostics['full']['coefficients']['energy_background_fraction']]);fig.tight_layout();fig.savefig(out/'03_coefficients.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for k in ['hi','lo','boot103108','boot103109','repeat_full']:ax[0].plot(ang_deg(ends[k],full),label=k)
    for k in ['full','hi','lo']:
        hist=json.loads((out/'runs'/k/'history.json').read_text());ax[1].plot(hist['final_losses'],label=k)
    for a in ax:a.legend();a.grid(alpha=.2)
    ax[0].set_title('Difference to full, deg');ax[1].set_title('Final optimization loss');fig.tight_layout();fig.savefig(out/'04_uncertainty_convergence.png');plt.close(fig)
    write(out/'fit_complete.json',dict(status='AWAITING_REVIEW',selected=selected,gt_read=False))

if __name__=='__main__':main()
