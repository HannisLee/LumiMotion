"""固定Stage1几何的E0—E4：无GT方向估计、光度留出选择与冻结。"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from LIHAN.scripts.static_rgb_light_prepare import write, sha


def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def subset(n,ids): return n[:,ids] if n.ndim==3 else n[ids]

def shading(n,l): return np.einsum('tgc,tc->tg',n,l) if n.ndim==3 else l@n.T

def vote(n,y,w,p):
    out=np.tile([0.,0.,1.],(len(y),1)); count=[]
    for t in range(len(y)):
        ok=np.isfinite(y[t])&(w[t]>0)
        if ok.sum()<16: count.append(0); continue
        threshold=np.quantile(y[t,ok],1-p)
        sel=ok&(y[t]>=threshold)
        weight=w[t,sel]*np.maximum(y[t,sel],0)
        v=((n[t,sel] if n.ndim==3 else n[sel])*weight[:,None]).sum(0)
        if np.linalg.norm(v)>1e-8: out[t]=unit(v)
        count.append(int(sel.sum()))
    return out,count


def nuisance_score(n,y,w,l):
    """留出Gaussian以1-based奇数帧标定rho/b、偶数帧验证；不调整灯向。"""
    x=np.maximum(shading(n,l),0)
    train=np.isfinite(y)&(np.arange(len(y))[:,None]%2==0)
    score=np.isfinite(y)&~train
    ww=np.where(train,w,0); yy=np.nan_to_num(y)
    s=ww.sum(0); sx=(ww*x).sum(0); sy=(ww*yy).sum(0)
    sxx=(ww*x*x).sum(0); sxy=(ww*x*yy).sum(0)
    rho=np.clip((s*sxy-sx*sy)/np.maximum(s*sxx-sx*sx,1e-7),.01,3)
    b=np.maximum((sy-rho*sx)/np.maximum(s,1e-7),0)
    res=x*rho[None]+b[None]-yy
    valid=score&(s[None]>=3)
    ww=np.where(valid,w,0)
    if (ww.sum(0)>0).sum()<16 or (ww.sum(1)>0).sum()<40 or ww.sum()<100:
        raise RuntimeError('空间留出标定/验证覆盖不足，禁止以零残差选择候选')
    return float((ww*np.minimum(np.abs(res),.2)).sum()/np.maximum(ww.sum(),1e-8))


def fit(n,y,w,initial,delta,balanced=False,shadow=False,steps=1600):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    nn=torch.tensor(n,dtype=torch.float32,device=device)
    yy=torch.tensor(np.nan_to_num(y),dtype=torch.float32,device=device)
    ww=torch.tensor(np.where(np.isfinite(y),w,0),dtype=torch.float32,device=device)
    if balanced:
        az=np.arctan2(n[...,1],n[...,0]); zz=n[...,2]
        bins=np.clip(((az+np.pi)/(2*np.pi)*12).astype(int),0,11)+12*np.clip(((zz+1)/2*6).astype(int),0,5)
        if n.ndim==3:
            factor=np.stack([1/np.sqrt(np.maximum(np.bincount(row,minlength=72)[row],1)) for row in bins])
            factor/=factor.mean(axis=1,keepdims=True)
        else:
            counts=np.bincount(bins,minlength=72)
            factor=1/np.sqrt(np.maximum(counts[bins],1));factor/=factor.mean()
        ww=ww*torch.tensor(np.minimum(factor,5),device=device)
    direction=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float32,device=device))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        q20=np.nanquantile(y,.2,axis=0); q95=np.nanquantile(y,.95,axis=0)
    rho0=np.maximum((q95-q20)*1.5,.03); b0=np.maximum(q20*.8,.001)
    rawrho=torch.nn.Parameter(torch.tensor(np.log(np.expm1(rho0)),dtype=torch.float32,device=device))
    rawb=torch.nn.Parameter(torch.tensor(np.log(np.expm1(b0)),dtype=torch.float32,device=device))
    opt=torch.optim.Adam([{'params':[direction],'lr':.025},{'params':[rawrho,rawb],'lr':.04}])
    history=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        l=F.normalize(direction,dim=1)
        rho=F.softplus(rawrho); b=F.softplus(rawb)
        pred=torch.relu(torch.einsum('tgc,tc->tg',nn,l) if nn.ndim==3 else l@nn.T)*rho[None]+b[None]
        residual=pred-yy
        huber=torch.where(residual.abs()<=delta,.5*residual.square()/delta,residual.abs()-.5*delta)
        effective=ww
        if shadow: effective=effective*torch.where(residual.detach()>delta,0.2,1.)
        loss=(huber*effective).sum()/effective.sum().clamp_min(1)
        # 仅抑制极端材质/环境自由度，无时间方向先验或逐帧强度。
        loss=loss+1e-5*(rho.square().mean()+b.square().mean())
        loss.backward(); opt.step()
        if step in [800,1200]:
            for group in opt.param_groups: group['lr']*=.3
        if step%200==0 or step==steps-1:
            history.append(dict(step=step,loss=float(loss.detach())))
            print('fit',balanced,shadow,delta,step,float(loss.detach()),flush=True)
    return F.normalize(direction,dim=1).detach().cpu().numpy(),history


def figures(data,n,y,w,relative,results,choices,out,fitids):
    """只用RGB/模型预测生成百分位与残差目检，不读取GT。"""
    x=np.maximum(shading(n,results['E4']),0); yy=np.nan_to_num(y); ww=np.where(np.isfinite(y),w,0)
    s=ww.sum(0);sx=(ww*x).sum(0);sy=(ww*yy).sum(0);sxx=(ww*x*x).sum(0);sxy=(ww*x*yy).sum(0)
    rho=np.clip((s*sxy-sx*sy)/np.maximum(s*sxx-sx*sx,1e-7),.01,3)
    b=np.maximum((sy-rho*sx)/np.maximum(s,1e-7),0)
    residual=x*rho[None]+b[None]-y
    np.savez_compressed(out/'photometric_diagnostics.npz',rho=rho,b=b,residual=residual,
                        diagnostic_only=True,nuisance_calibration='固定方向后全观测OLS重新标定，不是优化器内部Huber残差')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(residual[np.isfinite(residual)],bins=100,range=(-.3,.3));axes[0].set(xlabel='OLS prediction minus observed inverse-sRGB luminance',ylabel='Count')
    axes[1].plot(np.arange(1,151),np.nanmedian(np.abs(residual),axis=1));axes[1].set(xlabel='Frame',ylabel='Median absolute OLS diagnostic residual')
    fig.tight_layout();fig.savefig(out/'photometric_residual_distribution.png',dpi=150);plt.close(fig)
    if 'projection_uv' not in data:return
    uv=data['projection_uv']
    fit_surface=np.zeros(y.shape[1],dtype=bool);fit_surface[fitids]=True
    for t in [0,29,59,89,119,149]:
        valid=np.isfinite(y[t])&(w[t]>0)&np.isfinite(uv[t]).all(1)
        if valid.sum()<16:continue
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        pix=(uv[t]+1)*200-.5
        for ax,k,values in [(axes[0],'E1',y[t]),(axes[1],'E2',relative[t])]:
            ok=valid&np.isfinite(values)&fit_surface
            high=np.zeros(y.shape[1],dtype=bool)
            if ok.sum()>=16:
                cut=np.quantile(values[ok],1-choices[k]['selected_top'])
                high=ok&(values>=cut)
            ax.scatter(pix[valid,0],pix[valid,1],s=1,c='lightgray')
            ax.scatter(pix[high,0],pix[high,1],s=4,c=values[high],cmap='viridis')
            ax.set(title=f'{k} actual voting surfaces / frame {t+1}',xlim=(0,400),ylim=(400,0),aspect='equal')
        m=axes[2].scatter(pix[valid,0],pix[valid,1],s=3,c=residual[t,valid],cmap='coolwarm',vmin=-.15,vmax=.15)
        axes[2].set(title='E4 fixed-direction / OLS residual',xlim=(0,400),ylim=(400,0),aspect='equal');fig.colorbar(m,ax=axes[2],shrink=.7)
        fig.tight_layout();fig.savefig(out/f'percentile_residual_{t+1:04d}.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1600);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    d=np.load(a.observations);n=d['normals'];y=d['intensity'];w=d['weights'];ids=d['gaussian_ids'];xyz=d['xyz']
    assert np.array_equal(d['frames'],np.arange(1,151)), '观测帧必须完整且有序'
    assert y.shape==w.shape==(150,len(ids)) and xyz.shape==(len(ids),3)
    assert n.shape in [(len(ids),3),(150,len(ids),3)]
    assert len(np.unique(ids))==len(ids) and np.isfinite(n).all() and np.isfinite(xyz).all()
    assert np.allclose(np.linalg.norm(n,axis=-1),1,atol=1e-4)
    assert np.isfinite(w).all() and (w>=0).all() and (w<=1.0001).all()
    assert not np.any((w>0)&~np.isfinite(y)), '有效权重须有真实RGB观测'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        q20=np.nanquantile(y,.2,axis=0);q95=np.nanquantile(y,.95,axis=0)
    eligible=(np.isfinite(y).sum(0)>=8)&(q95-q20>.015)&(np.isfinite(n).all((0,2)) if n.ndim==3 else np.isfinite(n).all(1))
    # 空间块划分，非图像像素或GT标签；固定验证点不参与灯向拟合。
    cells=np.floor(xyz*20).astype(np.int64)
    held=((cells[:,0]*73856093 ^ cells[:,1]*19349663 ^ cells[:,2]*83492791)%5)==0
    fitids=np.flatnonzero(eligible&~held);validids=np.flatnonzero(eligible&held)
    rng=np.random.default_rng(1030)
    if len(fitids)>12000: fitids=np.sort(rng.choice(fitids,12000,replace=False))
    if len(validids)>6000:validids=np.sort(rng.choice(validids,6000,replace=False))
    if len(fitids)<32 or len(validids)<16:raise RuntimeError('几何可靠跨帧对应不足，不能执行可信光度拟合')
    nn=subset(n,fitids);yy=y[:,fitids];ww=w[:,fitids]
    frame_eigenvalues=[]
    for t,row in enumerate(ww):
        nt=nn[t] if nn.ndim==3 else nn
        covariance=(nt.T*row)@nt/max(float(row.sum()),1e-8)
        frame_eigenvalues.append(np.linalg.eigvalsh(covariance).tolist())
    relative=(y-q20[None])/(q95-q20+.001)[None];relative=np.clip(relative,0,1.5)
    relative[:,~eligible]=np.nan
    results={'E0':np.tile([0.,0.,1.],(150,1))}; choices={};votes={}
    for method,values in [('E1',y),('E2',relative)]:
        candidates=[]
        for top in [.1,.2,.3]:
            l,count=vote(nn,values[:,fitids],ww,top)
            score=nuisance_score(subset(n,validids),y[:,validids],w[:,validids],l)
            candidates.append((score,top,l,count))
        best=min(candidates,key=lambda x:x[0]);results[method]=best[2];votes[method]=best[3]
        choices[method]=dict(selected_top=best[1],validation=[dict(top=c[1],mae=c[0]) for c in candidates])
    histories={}
    cached_e3=[]
    for method in ['E3','E4']:
        candidates=[]
        variants=[('robust_only',False,False)] if method=='E3' else [
            ('robust_only',False,False),('normal_balance',True,False),
            ('shadow_downweight',False,True),('normal_balance_and_shadow',True,True)]
        for variant,balanced,shadow in variants:
            if method=='E4' and variant=='robust_only':
                candidates.extend(cached_e3)
                continue
            for delta in [.02,.05]:
                l,history=fit(nn,yy,ww,results['E2'],delta,balanced,shadow,a.steps)
                score=nuisance_score(subset(n,validids),y[:,validids],w[:,validids],l)
                candidates.append(dict(score=score,delta=delta,l=l,history=history,variant=variant,
                                       balanced=balanced,shadow=shadow))
        if method=='E3':cached_e3=candidates.copy()
        best=min(candidates,key=lambda c:(c['score'],int(c['balanced'])+int(c['shadow']),c['delta']))
        results[method]=best['l'];histories[method]=best['history']
        choices[method]=dict(selected_delta=best['delta'],selected_variant=best['variant'],
                            selected_balanced=best['balanced'],selected_shadow=best['shadow'],
                            validation=[dict(delta=c['delta'],variant=c['variant'],clipped_mae=c['score']) for c in candidates])
    # 完整150帧保证，缺失对应保留E0并明确标记。
    missing=(np.isfinite(yy).sum(1)<16)
    for k in results:results[k][missing]=[0,0,1]
    np.savez_compressed(out/'predictions.npz',frames=d['frames'],**results)
    scores={k:nuisance_score(subset(n,validids),y[:,validids],w[:,validids],l) for k,l in results.items()}
    figures(d,n,y,w,relative,results,choices,out,fitids)
    bootstrap=[]
    for seed in [1031,1032]:
        pick=np.random.default_rng(seed).choice(len(fitids),len(fitids),replace=True)
        l,history=fit(subset(nn,pick),yy[:,pick],ww[:,pick],results['E4'],choices['E4']['selected_delta'],
                      choices['E4']['selected_balanced'],choices['E4']['selected_shadow'],min(a.steps,800))
        angle=np.degrees(np.arccos(np.clip((l*results['E4']).sum(1),-1,1)))
        bootstrap.append(dict(seed=seed,mean_degrees=float(angle.mean()),median_degrees=float(np.median(angle)),p90_degrees=float(np.quantile(angle,.9))))
        np.save(out/f'bootstrap_{seed}.npy',l)
    # E4是预注册最终改进方案；不得在看到GT后从E0—E4挑选。
    config=dict(final_method='E4',gt_read=False,coordinate_system='world',direction_convention='surface_to_light',
          input_sha256=sha(a.observations),spatial_split='floor(xyz*20)的固定三素数哈希mod5=0留出',
          fit_gaussians=len(fitids),heldout_gaussians=len(validids),eligible_gaussians=int(eligible.sum()),
          steps=a.steps,directional_intensity=1,frame_intensity_parameters=False,
          ambient='每表面固定非负偏置b，跨帧共享；等价rho*A；非逐帧环境',
          E4_selection_rule='仅现有normal_balance/shadow_downweight四组合、delta[0.02,0.05]、同E2初始化与1600步；空间留出clipped MAE最小；全关最优则明确回退E3，不强制更差改进',
          percentile_visualization='高亮为真实fit Gaussian集及其投票阈值；灰点为全部可靠观测',
          residual_visualization='固定最终光向后，对全部观测逐表面OLS重新标定rho/b；仅诊断，不等于优化器内部Huber残差或CV留出残差',
          choices=choices,heldout_photometric_clipped_mae=scores,score_clipping_threshold=.2,
          missing_frames=(np.flatnonzero(missing)+1).tolist(),
          histories=histories,votes=votes,bootstrap_stability=bootstrap,
          normal_covariance_eigenvalues=np.linalg.eigvalsh(nn.reshape(-1,3).T@nn.reshape(-1,3)/len(nn.reshape(-1,3))).tolist(),
          per_frame_normal_eigenvalues=frame_eigenvalues,
          normal_covariance_condition_number=float(np.linalg.cond(nn.reshape(-1,3).T@nn.reshape(-1,3)/len(nn.reshape(-1,3)))))
    write(out/'method_config.json',config)
    fig,ax=plt.subplots(figsize=(10,4))
    for k,l in results.items():ax.plot(d['frames'],l[:,2],label=k)
    ax.set(xlabel='Frame',ylabel='World direction z',title='Frozen geometry / RGB-only estimates');ax.legend();fig.tight_layout();fig.savefig(out/'directions_no_gt.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(list(scores),list(scores.values()));ax.set(ylabel='Held-out photometric MAE');fig.tight_layout();fig.savefig(out/'photometric_holdout.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(final_method='E4',scores=scores,missing_frames=config['missing_frames'])),flush=True)


if __name__=='__main__':main()
