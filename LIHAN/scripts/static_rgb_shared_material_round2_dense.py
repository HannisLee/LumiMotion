"""第二轮预注册RGB共享材质拟合；此模块没有GT读取或评测接口。"""
import argparse, hashlib, json, os, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

PLAN=Path('LIHAN/DOC/讨论/1031-05-未知方向光第二轮RGB盲估计预注册.md')
PLAN_HASH='67f130da51517fd268050ed0831c4022d258143905fac37bf1d9b8c18f7940e7'
OBS=Path('output/1030-01-LH-static-ELDL-V3-RGB方向光初始化/extraction/observations.npz')
MODEL=Path('output/1019-02-LH-static-jumpingjacks-ELDL-V3-B00原生Stage1/model_mlp')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def angle(a,b):return np.degrees(np.arccos(np.clip((unit(a)*unit(b)).sum(-1),-1,1)))
def chroma(rgb,valid):
    z=rgb/(rgb.sum(-1,keepdims=True)+1e-6);z=np.where(valid[...,None],z,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);x=np.nanmedian(z,axis=0)
    return x/(x.sum(-1,keepdims=True)+1e-6)
def vote(n,y,w):
    out=np.tile([0.,0.,1.],(len(y),1));counts=[]
    for t in range(len(y)):
        ok=(w[t]>0)&np.isfinite(y[t]);s=np.zeros(len(ok),bool)
        if ok.sum():s=ok&(y[t]>=np.quantile(y[t,ok],.8))
        counts.append(int(s.sum()))
        if s.sum()>=16:
            v=(n[t,s]*(w[t,s]*np.maximum(y[t,s],0))[:,None]).sum(0)
            if np.linalg.norm(v)>1e-8:out[t]=unit(v)
    return out,counts

def sparse(n,rgb,w,labels):
    t,g=np.where(w>0)
    return tuple(torch.tensor(a,device='cuda',dtype=d) for a,d in [(t,torch.long),(n[t,g],torch.float32),(rgb[t,g],torch.float32),(w[t,g],torch.float32),(labels[g],torch.long)])
def fit(n,rgb,w,labels,K,initial,out,steps=1600,fixed=False,multiplicity=None):
    """仅在训练观测拟合，逐帧等权；终点训练目标包含共享参数正则。"""
    torch.manual_seed(103105)
    nn=torch.tensor(n,device='cuda',dtype=torch.float32)
    yy=torch.tensor(np.nan_to_num(rgb),device='cuda',dtype=torch.float32)
    ww=torch.tensor(w,device='cuda',dtype=torch.float32)
    lab=torch.tensor(labels,device='cuda',dtype=torch.long)
    den=ww.sum(1).clamp_min(1e-12)
    rho_values=[]
    for k in range(K):
        mask=w[:,labels==k]>0;values=rgb[:,labels==k][mask]
        if multiplicity is not None:
            repeats=np.broadcast_to(multiplicity[labels==k][None],mask.shape)[mask]
            values=np.repeat(values,repeats,axis=0)
        rho_values.append(np.median(values,axis=0))
    rho0=np.stack(rho_values)/.7
    rho0=np.maximum(rho0,.01)
    rawrho=torch.nn.Parameter(torch.tensor(np.log(np.expm1(rho0)),device='cuda',dtype=torch.float32))
    rawa=torch.nn.Parameter(torch.full((3,),float(np.log(np.expm1(.1))),device='cuda'))
    l=torch.nn.Parameter(torch.tensor(initial,device='cuda',dtype=torch.float32),requires_grad=not fixed)
    groups=[{'params':[rawrho,rawa],'lr':.04}]
    if not fixed:groups.insert(0,{'params':[l],'lr':.025})
    opt=torch.optim.Adam(groups);history=[]
    def objective():
        rho=F.softplus(rawrho);a=F.softplus(rawa)
        pred=rho[lab][None]*(a[None,None]+torch.relu((nn*F.normalize(l,dim=1)[:,None]).sum(2))[...,None])
        r=pred-yy
        h=torch.where(r.abs()<=.03,.5*r.square()/.03,r.abs()-.015).mean(2)
        by=(h*ww).sum(1)/den
        return by.mean()+1e-5*(rho.square().mean()+a.square().mean())
    for step in range(steps):
        if step in [800,1200]:
            for gr in opt.param_groups:gr['lr']*=.3
        opt.zero_grad(set_to_none=True);loss=objective()
        if not torch.isfinite(loss):raise RuntimeError('非有限训练目标')
        loss.backward();opt.step()
        with torch.no_grad():l.copy_(F.normalize(l,dim=1))
        if step%200==0 or step==steps-1:
            history.append({'step':step,'objective':float(loss.detach())});print(out.name,history[-1],flush=True)
    with torch.no_grad():
        result=dict(direction=F.normalize(l,dim=1).cpu().numpy(),rho=F.softplus(rawrho).cpu().numpy(),ambient=F.softplus(rawa).cpu().numpy())
        obj=float(objective())
    np.savez_compressed(out,**result)
    write(out.with_suffix('.json'),dict(objective=obj,history=history,steps=steps,fixed_direction=fixed))
    return result,obj

def predict(n,labels,result):
    return result['rho'][labels][None]*(result['ambient'][None,None]+np.maximum(np.einsum('tgc,tc->tg',n,result['direction']),0)[...,None])
def score(n,rgb,w,labels,result):
    residual=predict(n,labels,result)-rgb;per=[];raw=[];frames=[]
    for t in range(1,150,2):
        ok=w[t]>0
        if ok.sum()<16:continue
        ww=w[t,ok];e=np.abs(residual[t,ok]);per.append(float((np.minimum(e,.2).mean(1)*ww).sum()/ww.sum()));raw.append(float((e.mean(1)*ww).sum()/ww.sum()));frames.append(t+1)
    return dict(clipped_mae=float(np.mean(per)),mae=float(np.mean(raw)),se=float(np.std(per,ddof=1)/np.sqrt(len(per))),per_frame=per,unclipped_per_frame=raw,frames=frames)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--synthetic-check',action='store_true');args=ap.parse_args()
    assert sha(PLAN)==PLAN_HASH
    args.output.mkdir(parents=True,exist_ok=False);out=args.output
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if args.synthetic_check:
        rng=np.random.default_rng(103105);n=unit(rng.normal(size=(150,64,3))).astype('float32');l=unit(rng.normal(size=(150,3))).astype('float32');lab=np.arange(64)%4
        rho=rng.uniform(.2,.8,(4,3));ambient=np.array([.1,.15,.2]);rgb=predict(n,lab,dict(direction=l,rho=rho,ambient=ambient));w=np.ones((150,64),dtype='float32')
        r,obj=fit(n,rgb,w,lab,4,l,out/'synthetic.npz',steps=1600)
        err=float(np.abs(predict(n,lab,r)-rgb).mean());assert err<.002,(err,obj)
        # 留出评分只用偶数帧；修改奇数帧不会影响固定标签后的评分。
        s=score(n,rgb,w,lab,r);rgb2=rgb.copy();rgb2[::2]=123;assert score(n,rgb2,w,lab,r)==s
        write(out/'synthetic_check.json',dict(status='PASS',rgb_mae=err,objective=obj,gt='仅程序生成合成方向，未读取任何真实GT'));return
    inputs=[PLAN,OBS,OBS.with_name('extraction.json'),Path(__file__),MODEL/'point_cloud/iteration_35000/point_cloud.ply',MODEL/'deform/iteration_35000/deform.pth']
    write(out/'input_manifest.json',dict(gt_read=False,command=' '.join(sys.argv),hostname=os.uname().nodename,conda=os.environ.get('CONDA_DEFAULT_ENV'),torch=torch.__version__,numpy=np.__version__,inputs={str(p):sha(p) for p in inputs},seed=103105,precision='float32',huber='平滑L1：0.5*r²/delta；与既有实现一致'))
    d=np.load(OBS);n=d['normals'];w=d['weights'];xyz=d['xyz'];srgb=d['rgb'];rgb=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
    assert n.shape==(150,30000,3) and np.array_equal(d['frames'],np.arange(1,151));assert np.isfinite(rgb[w>0]).all()
    valid=(w>0)&np.isfinite(rgb).all(-1);w=np.where(valid,w,0)
    eligible=valid.sum(0)>=8;cells=np.floor(xyz*20).astype(np.int64);held=((cells[:,0]*73856093^cells[:,1]*19349663^cells[:,2]*83492791)%5)==0
    train=np.flatnonzero(eligible&~held);hold_before=np.flatnonzero(eligible&held);hold=hold_before[valid[::2,hold_before].sum(0)>=4]
    rng=np.random.default_rng(103105)
    if len(train)>12000:train=np.sort(rng.choice(train,12000,replace=False))
    if len(hold)>6000:hold=np.sort(rng.choice(hold,6000,replace=False))
    np.savez_compressed(out/'split.npz',train=train,holdout=hold,gaussian_ids=d['gaussian_ids'],cells=cells)
    coverage=dict(eligible=int(eligible.sum()),train=len(train),holdout=len(hold),excluded_odd_insufficient=len(hold_before)-int((valid[::2,hold_before].sum(0)>=4).sum()),train_per_frame=valid[:,train].sum(1).tolist(),holdout_per_frame=valid[:,hold].sum(1).tolist())
    write(out/'coverage.json',coverage)
    if min(coverage['train_per_frame'])<32 or sum(valid[t,hold].sum()>=16 for t in range(1,150,2))<60:
        write(out/'selection.json',dict(state='STOP_INSUFFICIENT_COVERAGE',gt_read=False));return
    nn=n[:,train];rr=rgb[:,train];ww=w[:,train];hn=n[:,hold];hr=rgb[:,hold];hw=w[:,hold]
    c=chroma(rr,ww>0);hvalid=(hw>0)&(np.arange(150)[:,None]%2==0);hc=chroma(hr,hvalid)
    lum=rr@np.array([.2126,.7152,.0722]);lum=np.where(ww>0,lum,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);q20=np.nanquantile(lum,.2,axis=0);q95=np.nanquantile(lum,.95,axis=0)
    rel=np.clip((lum-q20)/(q95-q20+.001),0,1.5);rel[:,q95-q20<=.015]=np.nan
    initial={};votes={}
    for name,y in [('absolute',lum),('relative',rel)]:initial[name],votes[name]=vote(nn,y,ww)
    np.savez_compressed(out/'initializations.npz',**initial);write(out/'vote_counts.json',votes)
    candidates={};labels={};scores={};runs=0;invalid={};ambiguity={}
    for K in [1,4,8]:
        km=KMeans(n_clusters=K,init='k-means++',random_state=103105,n_init=1,max_iter=100,tol=1e-6,algorithm='lloyd').fit(c)
        lab=km.labels_;hl=((hc[:,None]-km.cluster_centers_[None])**2).sum(-1).argmin(1)
        counts=np.bincount(lab,minlength=K);observations=np.array([(ww[:,lab==k]>0).sum() for k in range(K)])
        np.savez_compressed(out/f'labels_K{K}.npz',train_labels=lab,holdout_labels=hl,centers=km.cluster_centers_,gaussian_counts=counts,observation_counts=observations)
        if min(counts)<10 or min(observations)<200:invalid[K]=dict(gaussians=counts.tolist(),observations=observations.tolist());continue
        labels[K]=(lab,hl);ends=[]
        for init in ['absolute','relative']:
            r,obj=fit(nn,rr,ww,lab,K,initial[init],out/f'K{K}_{init}.npz');runs+=1;ends.append((obj,init,r))
        ends.sort(key=lambda x:(x[0],x[1]));obj,name,r=ends[0];candidates[K]=(r,obj,name);scores[K]=score(hn,hr,hw,hl,r)
        diff=angle(ends[0][2]['direction'],ends[1][2]['direction']);gap=abs(ends[0][0]-ends[1][0])/max(abs(ends[0][0]),1e-12)
        ambiguity[K]=dict(relative_objective_gap=gap,mean_degrees=float(diff.mean()),p90_degrees=float(np.quantile(diff,.9)),near_equivalent_multiple_solutions=bool(gap<=.01 and diff.mean()>5),per_frame_degrees=diff.tolist())
    if not candidates:write(out/'selection.json',dict(state='STOP_NO_VALID_K',invalid=invalid,optimizations=runs,gt_read=False));return
    best=min(scores,key=lambda k:(scores[k]['clipped_mae'],k));upper=scores[best]['clipped_mae']+scores[best]['se'];chosen=min(k for k in scores if scores[k]['clipped_mae']<=upper)
    result,obj,initname=candidates[chosen];baseK=max(candidates);lab,hl=labels[baseK];baselines={}
    for init in ['absolute','relative']:
        r,bo=fit(nn,rr,ww,lab,baseK,initial[init],out/f'baseline_{init}_K{baseK}.npz',fixed=True);runs+=1;baselines[init]=score(hn,hr,hw,hl,r)
    base=min(baselines,key=lambda k:baselines[k]['clipped_mae']);improvement=1-scores[chosen]['clipped_mae']/baselines[base]['clipped_mae'];reasons=[]
    if improvement<.02 or scores[chosen]['mae']>baselines[base]['mae']:reasons.append('STOP_NO_RGB_IMPROVEMENT')
    if ambiguity[chosen]['near_equivalent_multiple_solutions']:reasons.append('STOP_NONIDENTIFIABLE')
    boot=[]
    # 已满足停止条件则禁止扩展；不再运行bootstrap。
    if not reasons:
        blocks,inverse=np.unique(cells[train],axis=0,return_inverse=True)
        for seed in [103106,103107]:
            rng=np.random.default_rng(seed);mult=np.bincount(rng.choice(len(blocks),len(blocks),replace=True),minlength=len(blocks));bw=ww*mult[inverse][None]
            np.save(out/f'bootstrap_block_multiplicity_{seed}.npy',mult)
            r,bo=fit(nn,rr,bw,labels[chosen][0],chosen,result['direction'],out/f'bootstrap_{seed}.npz',multiplicity=mult[inverse]);runs+=1
            a=angle(r['direction'],result['direction']);boot.append(dict(seed=seed,mean_degrees=float(a.mean()),p90_degrees=float(np.quantile(a,.9)),per_frame_degrees=a.tolist()))
        if any(x['mean_degrees']>5 or x['p90_degrees']>10 for x in boot):reasons.append('STOP_UNSTABLE')
    # 固定共享材质条件下的切平面信息；不用于删除帧或调整优化。
    lab,hl=labels[chosen];information=[]
    for t in range(150):
        l=result['direction'][t];axis=np.eye(3)[np.argmin(abs(l))];u=unit(np.cross(l,axis));v=np.cross(l,u);basis=np.stack([u,v],axis=1)
        lit=(ww[t]>0)&((nn[t]*l).sum(1)>0);x=nn[t,lit]@basis;mass=ww[t,lit]*np.mean(result['rho'][lab[lit]]**2,axis=1);mat=(x.T*mass)@x/max(float(mass.sum()),1e-12);eig=np.linalg.eigvalsh(mat);ratio=float(eig[0]/max(eig[1],1e-12))
        information.append(dict(frame=t+1,lit_count=int(lit.sum()),eigenvalues=eig.tolist(),ratio=ratio,condition=float(eig[1]/max(eig[0],1e-12)),insufficient=bool(lit.sum()<16 or ratio<1e-3)))
    if any(x['insufficient'] for x in information):reasons.append('STOP_INSUFFICIENT_INFORMATION')
    np.savez_compressed(out/'final_predictions.npz',frames=np.arange(1,151),direction=result['direction'],rho=result['rho'],ambient=result['ambient'],train_labels=lab,holdout_labels=hl)
    selection=dict(state=reasons[0] if reasons else 'FROZEN_READY',stop_reasons=reasons,gt_read=False,chosen_K=chosen,chosen_initialization=initname,chosen_train_objective=obj,best_score_K=best,one_se_upper=upper,scores=scores,invalid_candidates=invalid,baseline_K=baseK,baselines=baselines,baseline_better=base,relative_rgb_improvement=improvement,initialization_stability=ambiguity,bootstrap=boot,information=information,optimizations=runs,coordinate_system='world',direction_convention='surface_to_light',final_refit=False)
    write(out/'selection.json',selection)
    # 代表帧所有可视化均仅使用原RGB观测、冻结几何及实际共享参数直接预测。
    pred=predict(hn,hl,result);uv=d['projection_uv'][:,hold];tuv=d['projection_uv'][:,train]
    for t in [0,29,59,89,119,149]:
        fig,axes=plt.subplots(1,4,figsize=(17,4));ok=hw[t]>0;p=(uv[t]+1)*200-.5
        for ax,val,title in [(axes[0],hr[t],'Observed linear RGB'),(axes[1],np.clip(pred[t],0,1),'Direct shared-material prediction')]:
            ax.scatter(p[ok,0],p[ok,1],c=val[ok],s=8);ax.set_title(title)
        m=axes[2].scatter(p[ok,0],p[ok,1],c=np.mean(abs(pred[t,ok]-hr[t,ok]),1),s=8,vmin=0,vmax=.25,cmap='magma');fig.colorbar(m,ax=axes[2]);axes[2].set_title('Direct RGB MAE / held-out')
        pp=(tuv[t]+1)*200-.5;axes[3].scatter(pp[ww[t]>0,0],pp[ww[t]>0,1],c='lightgray',s=2)
        for name,col in [('absolute','red'),('relative','blue')]:
            yy=lum[t] if name=='absolute' else rel[t];ok2=(ww[t]>0)&np.isfinite(yy);s=ok2&(yy>=np.quantile(yy[ok2],.8));axes[3].scatter(pp[s,0],pp[s,1],c=col,s=3,alpha=.5,label=name)
        axes[3].legend();axes[3].set_title('Actual top20 voting samples')
        for ax in axes:ax.set(xlim=(0,400),ylim=(400,0),aspect='equal')
        fig.tight_layout();fig.savefig(out/f'direct_rgb_diagnostic_{t+1:04d}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,s in scores.items():axes[0].plot(s['frames'],s['per_frame'],label=f'K={k}')
    axes[0].legend();axes[0].set(xlabel='Frame',ylabel='Held-out clipped RGB MAE')
    for c in range(3):axes[1].plot(np.arange(1,151),result['direction'][:,c],label='xyz'[c])
    axes[1].legend();axes[1].set(xlabel='Frame',ylabel='Frozen direction');fig.tight_layout();fig.savefig(out/'no_gt_summary.png',dpi=150);plt.close(fig)
    write(out/'frozen_manifest.json',dict(state=selection['state'],gt_read=False,plan_sha256=PLAN_HASH,files={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='run.log'}))
    print(json.dumps({k:selection[k] for k in ['state','chosen_K','optimizations','relative_rgb_improvement','stop_reasons']}),flush=True)
if __name__=='__main__':main()
