"""第三轮：冻结几何/RGB观测的白化投票、KKT变量投影与机械稳定性选择（不读取GT）。"""
import argparse
import hashlib
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

PREREG=Path('LIHAN/DOC/讨论/1031-07-未知方向光第三轮z偏置诊断与预注册方案.md')
PREREG_HASH='dbfe6cbbac3076b89680ce7b6d77d9f449e8fb07637cda6773165be24c417426'
OBS=Path('output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz')
OBS_HASH='eaef94e0c60ad0c2bfb73630609597e81ca92f8000b635cad40d3e1a04382c47'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)

def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def sub(n,ids):return n[:,ids] if n.ndim==3 else n[ids]
def shade(n,l):return np.einsum('tgc,tc->tg',n,l) if n.ndim==3 else l@n.T

def compare(a,b):
    deg=np.rad2deg(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=1),-1,1)))
    return dict(mean_deg=float(deg.mean()),median_deg=float(np.median(deg)),p90_deg=float(np.quantile(deg,.9)),dz=float(abs(a[:,2].mean()-b[:,2].mean())))

def ols_score(n,y,w,l):
    """1-based奇数帧无约束OLS标定，偶数帧加权截断MAE；留出不优化方向。"""
    x=np.maximum(shade(n,l),0); yy=np.nan_to_num(y); valid=np.isfinite(y)&(w>0)
    cal=valid&(np.arange(len(y))[:,None]%2==0); test=valid&~cal
    ww=np.where(cal,w,0);s=ww.sum(0);sx=(ww*x).sum(0);sy=(ww*yy).sum(0)
    xc=x-sx/np.maximum(s,1e-12);yc=yy-sy/np.maximum(s,1e-12)
    sxx=(ww*xc*xc).sum(0);sxy=(ww*xc*yc).sum(0)
    rho=np.divide(sxy,sxx,out=np.zeros_like(sxy),where=sxx>1e-12)
    b=(sy-rho*sx)/np.maximum(s,1e-12)
    residual=x*rho+b-yy
    weights=np.where(test&(cal.sum(0)[None]>=3)&(sxx[None]>1e-12),w,0)
    assert (weights.sum(1)>0).all() if len(y)<4 else (weights.sum(1)>0).sum()>=len(y)//2
    assert weights.sum()>0
    return float((weights*np.minimum(abs(residual),.2)).sum()/weights.sum())

def voting(n,y,w,relative,kind,p):
    """每个调用仅用自身支撑点，白化水平票与自身terminator仰角。"""
    val=y if kind=='E1' else relative; us=[];counts=[]
    for t in range(len(y)):
        nt=n[t] if n.ndim==3 else n;ok=np.isfinite(val[t])&(w[t]>0)
        if ok.sum()<32:raise RuntimeError(f'第{t+1}帧投票少于32点')
        wt=w[t,ok];nt=nt[ok];v=val[t,ok];sw=wt.sum()
        mn=(nt*wt[:,None]).sum(0)/sw;mv=(v*wt).sum()/sw
        centered=nt[:,:2]-mn[:2];cov=(centered.T*wt)@centered/sw+.05*np.eye(2)
        high=v>=np.quantile(v,1-p)
        vec=np.linalg.solve(cov,(centered[high]*(wt[high]*np.maximum(v[high]-mv,0))[:,None]).sum(0))
        if np.linalg.norm(vec)<1e-12:raise RuntimeError('白化票退化')
        us.append(unit(vec));counts.append(int(high.sum()))
    u=np.asarray(us)
    if kind=='E2':u=unit(np.stack([u[max(0,t-1):min(len(u),t+2)].mean(0) for t in range(len(u))]))
    nx=shade(n,np.column_stack([u,np.zeros(len(u))]));nz=n[...,2] if n.ndim==3 else np.broadcast_to(n[:,2],y.shape)
    locus=(w>0)&np.isfinite(relative)&(relative>=.10)&(relative<=.25)&(nx<-.15)&(nz>.25)
    if locus.sum()<1:raise RuntimeError('交界线支撑不足')
    cot=np.median((-nx/np.maximum(nz,1e-12))[locus]);z=float(np.clip(cot/np.sqrt(1+cot*cot),.25,.90))
    l=np.column_stack([np.sqrt(1-z*z)*u,np.full(len(u),z)])
    return l,dict(top=p,terminator_count=int(locus.sum()),terminator_cot=float(cot),z_est=z,vote_counts=counts)

def initial(n,y,w,relative,kind,scoredata=None):
    candidates=[]
    for p in [.10,.20,.30]:
        l,info=voting(n,y,w,relative,kind,p)
        sn,sy,sw=scoredata if scoredata is not None else (n,y,w)
        info['selection_mae']=ols_score(sn,sy,sw,l);candidates.append((l,info))
    best=min(candidates,key=lambda c:(c[1]['selection_mae'],c[1]['top']))
    return best[0],dict(selected=best[1],candidates=[c[1] for c in candidates])

def kkt(x,y,w):
    """严格执行预注册1e-6分母稳定项与三个非负KKT活跃分支。"""
    s=w.sum(0).clamp_min(1e-15);mx=(w*x).sum(0)/s;my=(w*y).sum(0)/s
    xc=x-mx;yc=y-my
    rho=(w*xc*yc).sum(0)/((w*xc.square()).sum(0)+1e-6)
    b=my-rho*mx
    rb=((w*x*y).sum(0)/((w*x.square()).sum(0)+1e-6)).clamp_min(0)
    r=torch.where(rho<=0,torch.zeros_like(rho),torch.where(b<=0,rb,rho))
    bb=torch.where(rho<=0,my.clamp_min(0),torch.where(b<=0,torch.zeros_like(b),b))
    return r,bb

def balance_factor(n):
    # 平均模型法线的12 azimuth × 6 normal-z球面等面积分箱；平均归一后截断4。
    mean_n=unit(n.mean(0)) if n.ndim==3 else n
    az=np.arctan2(mean_n[:,1],mean_n[:,0]);zz=mean_n[:,2]
    bins=np.clip(((az+np.pi)/(2*np.pi)*12).astype(int),0,11)+12*np.clip(((zz+1)/2*6).astype(int),0,5)
    counts=np.bincount(bins,minlength=72);factor=1/np.sqrt(counts[bins]);factor/=factor.mean()
    return np.minimum(factor,4.)

def geo_penalty(u):
    v=u[:-1];v2=u[1:];d=torch.atan2(v[:,0]*v2[:,1]-v[:,1]*v2[:,0],(v*v2).sum(1))
    return (d[1:]-d[:-1]).square().mean()

def fit(n,y,w,init,balanced,delta,geo,out,label,stage2=True,steps=250):
    dest=out/'runs'/label;dest.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    nt=torch.as_tensor(n,dtype=torch.float64,device=device);yt=torch.as_tensor(np.nan_to_num(y),dtype=torch.float64,device=device)
    ww=torch.as_tensor(np.where(np.isfinite(y),w,0),dtype=torch.float64,device=device)
    if balanced:ww=ww*torch.as_tensor(balance_factor(n),dtype=torch.float64,device=device)
    rawu=torch.nn.Parameter(torch.as_tensor(unit(init[:,:2]),dtype=torch.float64,device=device).clone())
    z0=float(init[:,2].mean());rawz=torch.nn.Parameter(torch.tensor(np.log(z0/(1-z0)),dtype=torch.float64,device=device))
    hist=[];irw=ww;stage1_l=None
    for stage in [1,2] if stage2 else [1]:
        if stage==2:zfixed=torch.sigmoid(rawz.detach())
        opt=torch.optim.Adam([rawu,rawz] if stage==1 else [rawu],lr=.03 if stage==1 else .015)
        for step in range(1,steps+1):
            opt.zero_grad(set_to_none=True);u=F.normalize(rawu,dim=1);z=torch.sigmoid(rawz) if stage==1 else zfixed
            l=torch.cat([torch.sqrt(1-z*z)*u,z.expand(len(u),1)],1)
            x=torch.relu(torch.einsum('tgc,tc->tg',nt,l) if nt.ndim==3 else l@nt.T)
            rho,b=kkt(x,yt,ww if stage==1 else irw);res=x*rho+b-yt
            if stage==2 and step in [60,120]:
                irw=(ww*torch.minimum(torch.ones_like(res),delta/res.detach().abs().clamp_min(1e-15))).detach()
                rho,b=kkt(x,yt,irw);res=x*rho+b-yt
            reg=1e-5*(rho.square().mean()+b.square().mean())
            if stage==1:loss=(ww*.5*res.square()).sum()/ww.sum()+reg
            else:
                huber=torch.where(res.abs()<=delta,.5*res.square(),delta*(res.abs()-.5*delta))
                loss=(ww*huber).sum()/ww.sum()+reg+geo*geo_penalty(u)
            if not torch.isfinite(loss):raise RuntimeError('非有限loss')
            loss.backward();opt.step()
            if step==150:
                for group in opt.param_groups:group['lr']*=.3
            if step%25==0 or step==1:
                entry=dict(stage=stage,step=step,loss=float(loss.detach()),z=float(z.detach()));hist.append(entry)
                print(json.dumps(dict(run=label,**entry)),flush=True)
                write(out/'progress.json',dict(pid=os.getpid(),run=label,**entry,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        with torch.no_grad():
            u=F.normalize(rawu,dim=1);z=torch.sigmoid(rawz) if stage==1 else zfixed
            end=torch.cat([torch.sqrt(1-z*z)*u,z.expand(len(u),1)],1).cpu().numpy()
        np.savez_compressed(dest/f'stage{stage}_checkpoint.npz',direction=end,rawu=rawu.detach().cpu().numpy(),rawz=rawz.detach().cpu().numpy())
        write(dest/'history.json',hist)
        if stage==1:stage1_l=end.copy()
    np.save(dest/'direction.npy',end)
    write(dest/'config.json',dict(balanced=balanced,delta=delta,geo_smooth=geo,steps=steps,stage2=stage2,device=device,dtype='float64',support_count=y.shape[1]))
    return end,stage1_l,hist

def admission(row,e2,e3):
    checks=dict(hilo_mean=row['hilo_init_mean_deg']<=.50,hilo_dz=row['hilo_init_dz']<=.005,view_dz=row['view_dz_abs']<=.020,split_mean=row['split_half_mean_deg']<=3.50,split_p90=row['split_half_p90_deg']<=6.,body_vs_E2=row['body_heldout_mae']<=.60*e2,body_vs_E3=row['body_heldout_mae']<=1.015*e3)
    return checks,all(checks.values())

def select(rows):
    good=[r for r in rows if r['admitted']]
    if not good:return None
    m=min(r['J_stab'] for r in good);tied=[r for r in good if r['J_stab']<=m+.02]
    return min(tied,key=lambda r:(r['split_half_mean_deg'],r['body_heldout_mae'],-r['delta'],r['name']))

def tests(out):
    from scipy.optimize import nnls
    rng=np.random.default_rng(103107);x=rng.random((60,40));y=rng.random((60,40));w=rng.random((60,40))
    # 包括rho=0、b=0、内点、近常量与零权重表面。
    y[:,0]=.4-.2*x[:,0];y[:,1]=2*x[:,1]-.5;y[:,2]=.2+.5*x[:,2];x[:,3]=.5;w[:,4]=0
    tx=torch.tensor(x,dtype=torch.float64,requires_grad=True);ty=torch.tensor(y,dtype=torch.float64);tw=torch.tensor(w,dtype=torch.float64)
    r,b=kkt(tx,ty,tw);maxerr=0.
    for g in range(40):
        design=np.column_stack([x[:,g],np.ones(60)])*np.sqrt(w[:,g,None]);target=y[:,g]*np.sqrt(w[:,g])
        expected=nnls(design,target)[0];actual=np.array([r[g].item(),b[g].item()]);maxerr=max(maxerr,float(np.max(abs(expected-actual))))
    assert maxerr<2e-5,maxerr
    ((tx*r+b-ty).square()*tw).sum().backward();assert torch.isfinite(tx.grad).all()
    th=torch.arange(150,dtype=torch.float64)*.04;u=torch.stack([th.cos(),th.sin()],1);assert geo_penalty(u)<1e-24
    # 内点数值梯度验证，避开活跃集与ReLU边界。
    xx=torch.tensor(rng.random((12,3)),dtype=torch.float64,requires_grad=True);yy=.5*xx.detach()+.2;ww=torch.ones_like(xx)
    assert torch.autograd.gradcheck(lambda v: torch.stack(kkt(v,yy,ww)),(xx,),eps=1e-6,atol=1e-4)
    n=unit(rng.normal(size=(250,3)));theta=np.linspace(0,2*np.pi,30,endpoint=False);truth=np.column_stack([np.sqrt(1-.6**2)*np.cos(theta),np.sqrt(1-.6**2)*np.sin(theta),np.full(30,.6)])
    sy=np.maximum(truth@n.T,0)*rng.uniform(.3,.8,250)+rng.uniform(.03,.12,250);sw=np.ones_like(sy)
    init=truth.copy();init[:,:2]=unit(init[:,:2]+rng.normal(0,.05,(30,2)))*np.sqrt(1-.5**2);init[:,2]=.5
    end,_,_=fit(n,sy,sw,init,False,.03,.002,out,'synthetic',True)
    error=compare(end,truth);assert error['mean_deg']<.5,error
    write(out/'numeric_tests.json',dict(status='PASS',nnls_max_parameter_difference=maxerr,gradient_finite=True,gradcheck=True,constant_rotation_geo=float(geo_penalty(u)),synthetic=error))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=Path,default=OBS);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(103107);np.random.seed(103107)
    if a.self_test:tests(out);return
    assert sha(PREREG)==PREREG_HASH;assert sha(a.observations)==OBS_HASH
    write(out/'launch.json',dict(pid=os.getpid(),argv=sys.argv,prereg_sha256=sha(PREREG),code_sha256=sha(__file__),input_sha256=sha(a.observations),gt_read=False))
    d=np.load(a.observations);n=d['normals'].astype(np.float64);y=d['intensity'].astype(np.float64);w=d['weights'].astype(np.float64);xyz=d['xyz'];ids=d['gaussian_ids'];frames=d['frames']
    assert np.array_equal(frames,np.arange(1,151)) and y.shape==w.shape==(150,len(ids));assert np.isfinite(n).all() and np.isfinite(xyz).all();assert np.allclose(np.linalg.norm(n,axis=-1),1,atol=1e-4);assert np.isfinite(w).all() and (w>=0).all();assert not ((w>0)&~np.isfinite(y)).any()
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
    write(out/'support_manifest.json',dict(support_sha256=sha(out/'support.npz'),body_fit=2683,body_val=648,split_A=1450,split_B=1233,discrepancy='formal formula raw body_val=648; prereg text=599; formula retained without new filtering'))
    scoredata=(sub(n,vi),y[:,vi],w[:,vi]);results={'E0':np.tile([0.,0.,1.],(150,1))};voteinfo={}
    for k in ['E1','E2']:results[k],voteinfo[k]=initial(sub(n,fi),y[:,fi],w[:,fi],relative[:,fi],k,scoredata)
    # 半集完全独立：分位数候选用各自支撑点的奇偶帧光度评分；不接触另一半集/全量初值。
    splitinit={}
    for key,idx in [('A',A),('B',B)]:splitinit[key],voteinfo[key]=initial(sub(n,idx),y[:,idx],w[:,idx],relative[:,idx],'E2')
    results['E3'],_,_=fit(sub(n,fi),y[:,fi],w[:,fi],results['E2'],False,.03,0.,out,'E3',False)
    e2=ols_score(*scoredata,results['E2']);e3=ols_score(*scoredata,results['E3'])
    dark=((relative[:,fi]<.15)&(w[:,fi]>0)).sum(1)/(w[:,fi]>0).sum(1);front=dark<=np.quantile(dark,.3);side=dark>=np.quantile(dark,.7)
    rows=[];directions={};details={}
    for b,delta,geo in itertools.product([False,True],[.03,.05],[.001,.002]):
        name=f'b{int(b)}_d{delta:.2f}_g{geo:.3f}';ends={};traces={}
        for key,idx in [('full',fi),('A',A),('B',B),('hi',fi),('lo',fi)]:
            init=splitinit[key].copy() if key in splitinit else results['E2'].copy()
            if key in ['hi','lo']:
                z=.85 if key=='hi' else .35;init=np.column_stack([unit(init[:,:2])*np.sqrt(1-z*z),np.full(150,z)])
            ends[key],_,traces[key]=fit(sub(n,idx),y[:,idx],w[:,idx],init,b,delta,geo,out,name+'_'+key)
        sp=compare(ends['A'],ends['B']);hl=compare(ends['hi'],ends['lo']);full=ends['full']
        row=dict(name=name,balanced=b,delta=delta,geo_smooth=geo,split_half_mean_deg=sp['mean_deg'],split_half_median_deg=sp['median_deg'],split_half_p90_deg=sp['p90_deg'],split_half_dz=sp['dz'],hilo_init_mean_deg=hl['mean_deg'],hilo_init_dz=hl['dz'],view_dz_abs=float(abs(full[front,2].mean()-full[side,2].mean())),body_heldout_mae=ols_score(*scoredata,full),all_heldout_mae=ols_score(sub(n,allvi),y[:,allvi],w[:,allvi],full),mean_z=float(full[:,2].mean()))
        row['J_stab']=row['split_half_mean_deg']+.5*row['split_half_p90_deg']+10*row['split_half_dz']+row['hilo_init_mean_deg'];row['gate_checks'],row['admitted']=admission(row,e2,e3)
        rows.append(row);directions[name]=full;details[name]=(ends,traces)
        write(out/'ablation_stability_table.json',dict(E2_body_heldout_mae=e2,E3_body_heldout_mae=e3,candidates=rows));print('CANDIDATE',json.dumps(row),flush=True)
    selected=select(rows);results['E4']=directions[selected['name']].copy() if selected else results['E3'].copy()
    selected_name=selected['name'] if selected else 'E3_fallback';boot=[]
    for seed in [103108,103109]:
        pick=np.random.default_rng(seed).choice(fi,len(fi),replace=True)
        init,info=initial(sub(n,pick),y[:,pick],w[:,pick],relative[:,pick],'E2')
        lb,_,_=fit(sub(n,pick),y[:,pick],w[:,pick],init,selected['balanced'] if selected else False,selected['delta'] if selected else .03,selected['geo_smooth'] if selected else 0.,out,f'bootstrap_{seed}',selected is not None)
        np.save(out/f'bootstrap_{seed}.npy',lb);np.save(out/f'bootstrap_{seed}_sample.npy',pick)
        boot.append(dict(seed=seed,**compare(lb,results['E4']),initial=info))
    for k,v in results.items():assert v.shape==(150,3) and np.isfinite(v).all() and np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-10)
    np.savez_compressed(out/'predictions.npz',frames=frames,**results)
    config=dict(final_method='E4',selected=selected_name,selected_config=selected,gt_read=False,coordinate_system='world',direction_convention='surface_to_light',directional_intensity=1,frame_intensity_parameters=False,prereg_sha256=PREREG_HASH,input_sha256=OBS_HASH,fit_gaussians=len(fi),heldout_gaussians=len(vi),steps_per_stage=250,stage1_lr=.03,stage2_lr=.015,lr_drop_after_step=150,IRLS_update_steps=[60,120],KKT_denominator_epsilon=1e-6,dtype='float64',Huber='standard: 0.5*r^2 inside; delta*(abs(r)-0.5*delta) outside; fixed before formal grid',OLS='unconstrained rho,b; odd calibration/even validation; no clipping coefficients; abs residual clipped at 0.2',percentile_selection='full E1/E2 body_val odd/even OLS; independent half/bootstrap within-own-support odd/even OLS',normal_balance='mean model normal 12 azimuth x 6 equal normal-z bins; inverse sqrt counts normalized mean 1 then max4',J_stab='split_mean + 0.5*split_p90 + 10*split_dz + hilo_mean (formal prereg 4.2; GOAL summary conflict overruled)',tie_rule='within 0.02 of minimum J, then split_mean, body_MAE, -delta, fixed candidate name',votes=voteinfo,bootstrap_stability=boot,photometric_mae={k:ols_score(*scoredata,l) for k,l in results.items()},information_limitations='stability and shared-z are empirical/model constraints, not proof of unbiasedness, identifiability or GT accuracy')
    write(out/'method_config.json',config)
    figures(out,d,n,y,w,relative,fi,vi,results,voteinfo,rows,details,selected_name,dark)
    write(out/'fit_complete.json',dict(status='R3_FIT_COMPLETE_AWAITING_REVIEW',selected=selected_name,pid=os.getpid(),gt_read=False))
    print('R3_FIT_COMPLETE',selected_name,flush=True)


def figures(out,d,n,y,w,relative,fi,vi,results,voteinfo,rows,details,selected_name,dark):
    l=results['E4'];x=np.maximum(shade(sub(n,fi),l),0);yy=np.nan_to_num(y[:,fi]);ww=w[:,fi]
    with torch.no_grad():rho,b=kkt(torch.tensor(x),torch.tensor(yy),torch.tensor(ww));rho=rho.numpy();b=b.numpy()
    residual=x*rho+b-y[:,fi]
    np.savez_compressed(out/'photometric_diagnostics.npz',residual=residual,rho=rho,b=b,body_fit=fi,dark_fraction=dark,diagnostic_only=True)
    uv=d['projection_uv'][:,fi] if 'projection_uv' in d else None
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for col,t in enumerate([0,74,149]):
        ok=(ww[t]>0)&np.isfinite(y[t,fi]);pix=(uv[t]+1)*200-.5 if uv is not None else d['xyz'][fi,:2]
        high=ok&(relative[t,fi]>=np.quantile(relative[t,fi][ok],1-voteinfo['E2']['selected']['top']))
        axs[0,col].scatter(pix[ok,0],pix[ok,1],s=2,c='lightgray');axs[0,col].scatter(pix[high,0],pix[high,1],s=5,c=relative[t,fi][high],cmap='viridis');axs[0,col].set_title(f'Frame {t+1}: actual E2 body vote')
        sc=axs[1,col].scatter(pix[ok,0],pix[ok,1],s=4,c=residual[t,ok],cmap='coolwarm',vmin=-.15,vmax=.15);axs[1,col].set_title('Frozen E4 KKT residual');fig.colorbar(sc,ax=axs[1,col])
        for ax in axs[:,col]:ax.set_aspect('equal');ax.invert_yaxis()
    fig.tight_layout();fig.savefig(out/'01_percentile_residual.png',dpi=150);plt.close(fig)
    diagnostic_name=selected_name if selected_name in details else rows[0]['name']
    shown=details[diagnostic_name][0]
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    fig.suptitle(f'Final: {selected_name}; split diagnostics: {diagnostic_name}')
    for k in ['E1','E2','E3','E4']:
        ll=results[k];axs[0,0].plot(np.rad2deg(np.unwrap(np.arctan2(ll[:,1],ll[:,0]))),label=k);axs[0,1].plot(ll[:,2],label=k)
    for k in ['A','B']:axs[1,0].plot(np.rad2deg(np.unwrap(np.arctan2(shown[k][:,1],shown[k][:,0]))),label=k);axs[1,1].plot(shown[k][:,2],label=k)
    for ax in axs.flat:ax.legend();ax.set_xlabel('Frame index')
    axs[0,0].set_ylabel('Unwrapped azimuth (deg)');axs[0,1].set_ylabel('World z');axs[1,0].set_ylabel('Independent split azimuth');axs[1,1].set_ylabel('Independent split z')
    fig.tight_layout();fig.savefig(out/'02_direction_split.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,8));names=[r['name'] for r in rows]
    for ax,key in zip(axs.flat,['body_heldout_mae','split_half_mean_deg','split_half_p90_deg','J_stab']):
        ax.bar(names,[r[key] for r in rows],color=['tab:green' if r['admitted'] else 'tab:red' for r in rows]);ax.tick_params(axis='x',rotation=45);ax.set_ylabel(key)
    fig.tight_layout();fig.savefig(out/'03_holdout_stability.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    fig.suptitle(f'Final: {selected_name}; convergence diagnostics: {diagnostic_name}')
    for key in ['full','hi','lo','A','B']:
        hist=details[selected_name if selected_name in details else rows[0]['name']][1][key]
        xx=[h['step']+250*(h['stage']-1) for h in hist];axs[0,0].plot(xx,[h['z'] for h in hist],label=key);axs[0,1].plot(xx,[h['loss'] for h in hist],label=key)
    for key in ['hi','lo']:axs[1,0].plot(np.rad2deg(np.unwrap(np.arctan2(shown[key][:,1],shown[key][:,0]))),label=key);axs[1,1].plot(shown[key][:,2],label=key)
    for ax in axs.flat:ax.legend();ax.grid(alpha=.2)
    axs[0,0].set_ylabel('Shared z');axs[0,1].set_ylabel('Objective (stage objectives differ)');axs[1,0].set_ylabel('Final azimuth deg');axs[1,1].set_ylabel('Final z')
    fig.tight_layout();fig.savefig(out/'04_hilo_convergence.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
