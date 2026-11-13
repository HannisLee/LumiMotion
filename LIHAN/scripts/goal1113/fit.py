"""T01 固定 A/B/C 执行器。"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from . import inputs as I, physics as P, optimizer as O


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def setup(device):
    # 必须经原有相机读取接口，rgba 自动覆盖已由 whitelist 禁止。
    from LIHAN.scripts.static_rgb_light_extract import cameras
    hashes=I.whitelist()
    src=I.load_observations()
    cams=cameras(I.RUNTIME,'all')
    if [int(c.image_name_train_light.split('_')[-1]) for c in cams]!=list(range(1,151)):
        raise ValueError('相机帧号不符')
    centers=np.stack([c.camera_center.detach().cpu().numpy() for c in cams]).astype(np.float64)
    view=centers[:,None,:]-src['raw']['frame_xyz']
    length=np.linalg.norm(view,axis=-1,keepdims=True)
    if not np.isfinite(length).all() or (length<=1e-8).any(): raise ValueError('视线退化')
    view/=length
    fi=src['support']['body_fit']; w=src['w'][:,fi]
    if (w.sum(0)<=0).any(): raise ValueError('fit 高斯无有效观测')
    dev=torch.device(device)
    def t(x): return torch.as_tensor(x,dtype=torch.float64,device=dev)
    data=dict(n=t(src['n'][:,fi]),v=t(view[:,fi]),target=t(src['target'][:,fi]),w=t(w),b=t(src['b'][fi]))
    rho_y=src['rho_y'][fi]; K=max(1.,float(rho_y.max()))
    l=src['direction']; phi=np.unwrap(np.arctan2(l[:,1],l[:,0])); span=phi[-1]-phi[0]
    if abs(span)<math.pi: raise ValueError('初始绕行不足π')
    sign=float(np.sign(span))
    z=(np.arcsin(np.clip(l[:,2],-1,1))-math.pi/4)/(math.pi/36)
    if (abs(z)>=1).any(): raise ValueError('初始仰角不在可表示范围')
    params=dict(rho=t(np.repeat((rho_y/K)[:,None],3,axis=1)).clone(),env=t(np.ones((128,3))).clone(),
                phi=t(phi).clone(),a=t(np.arctanh(z)).clone())
    with torch.no_grad():
        initial=P.direction(params['phi'],params['a'])
        dynamic=K*params['rho'][None]*torch.einsum('tgc,tc->tg',data['n'],initial).clamp_min(0)[...,None]
        expected=t(rho_y)[None,:,None]*torch.einsum('tgc,tc->tg',data['n'],t(l)).clamp_min(0)[...,None]
        diff=float((dynamic-expected).abs().max())
        if diff>1e-10: raise ValueError('E4 动态代数不符')
        # 依据固定几何预计算 BRDF 基；只搬运块，保持 128 高斯求积精确。
        bases=[P.brdf_basis(data['n'][:,j:j+128],data['v'][:,j:j+128]) for j in range(0,len(fi),128)]
        data['basis']=(torch.cat([x[0] for x in bases],1),torch.cat([x[1] for x in bases],1))
        env=t(np.ones((128,3))); vals=[]
        for j in range(0,len(fi),128):
            sl=slice(j,min(j+128,len(fi))); basis=(data['basis'][0][:,sl],data['basis'][1][:,sl])
            _,s=P.components(data['n'][:,sl],data['v'][:,sl],params['rho'][sl],env,initial,K,basis)
            vals.append((data['w'][:,sl]*(s@s.new_tensor(P.Y))).sum(0)/data['w'][:,sl].sum(0))
        u=torch.cat(vals); b=data['b']; c0=max(float((u*b).sum()/(u.square().sum()+1e-12)),1e-6)
        params['env'].fill_(c0)
    scale=max(float((data['w']*t(src['intensity'][:,fi])).sum()/data['w'].sum()),1e-3)
    e0=float((w*src['b'][fi][None]).sum()/(w*(rho_y[None]*np.maximum(np.einsum('tgc,tc->tg',src['n'][:,fi],l),0)+src['b'][fi][None])).sum())
    return src,data,params,dict(K=K,c0=c0,scale=scale,e0=e0,sign=sign,equivalence_error=diff,hashes=hashes)


@torch.no_grad()
def diagnostic(data, params, meta):
    n,v,w=data['n'],data['v'],data['w']; l=P.direction(params['phi'],params['a'])
    if (l.norm(dim=-1)-1).abs().max()>1e-8 or params['rho'].min() < -1e-10 or params['rho'].max()>1+1e-10 or params['env'].min() < -1e-10:
        raise FloatingPointError('方向或参数违反数值不变量')
    sums=torch.zeros(n.shape[0],dtype=n.dtype,device=n.device); den=sums.clone(); sb=[]; residual=[]
    for j in range(0,n.shape[1],128):
        sl=slice(j,min(j+128,n.shape[1])); basis=(data['basis'][0][:,sl],data['basis'][1][:,sl])
        d,s=P.components(n[:,sl],v[:,sl],params['rho'][sl],params['env'],l,meta['K'],basis)
        sums+=(w[:,sl,None]*s).sum((1,2)); den+=(w[:,sl,None]*(d+s)).sum((1,2))
        sb.append((w[:,sl]*(s@s.new_tensor(P.Y))).sum(0)/w[:,sl].sum(0))
        residual.append((w[:,sl,None]*(d+s-data['target'][:,sl]).square()).sum((0,1)))
    b=torch.cat(sb); r=(b-data['b']).abs()/meta['scale']; signed=(b-data['b'])/meta['scale']
    en=float(sums.sum()/den.sum()); quant=torch.quantile(r,torch.tensor([.5,.9,.95],device=r.device,dtype=r.dtype))
    ok,status=P.guard(en,meta['e0'],float(r.mean()),float(quant[1]))
    return dict(energy=en,energy_frames=(sums/den.clamp_min(1e-30)).cpu().tolist(),b_mean=float(r.mean()),
                b_p50=float(quant[0]),b_p90=float(quant[1]),b_p95=float(quant[2]),b_max=float(r.max()),
                b_signed_mean=float(signed.mean()),rms=(torch.stack(residual).sum(0)/w.sum()).sqrt().cpu().tolist(),
                invalid_nv_fraction=float((((n*v).sum(-1)<=1e-6)*w).sum()/w.sum()),
                rho_boundary_fraction=float(((params['rho']<=1e-10)|(params['rho']>=1-1e-10)).double().mean()),
                env_max=float(params['env'].max()),env_mean=float(params['env'].mean()),
                pass_guard=ok,status=status)


def run(root, device, resume=None):
    torch.manual_seed(111301); np.random.seed(111301)
    root=Path(root); src,data,params,meta=setup(device)
    state=None
    if resume is None:
        (root/'inputs').mkdir(exist_ok=False,parents=True)
        write(root/'inputs'/'manifest.json',dict(original_source=str(I.SOURCE),runtime_source=str(I.RUNTIME),gt_read=False,
              hashes=meta['hashes'],missing=src['missing'],valid_target_change_max=src['valid_target_change_max']))
        write(root/'inputs'/'config.json',{k:v for k,v in meta.items() if k!='hashes'})
        np.savez_compressed(root/'inputs'/'prior.npz',rho_y=src['rho_y'],b=src['b'],direction=src['direction'])
        (root/'fitting').mkdir(exist_ok=True)
        a=root/'fitting'/'A'; a.mkdir(exist_ok=False)
        torch.save({'params':{k:v.detach().cpu() for k,v in params.items()},'meta':{k:v for k,v in meta.items() if k!='hashes'}},a/'checkpoint.pt')
        write(a/'diagnostic.json',diagnostic(data,params,meta))
        if max(0.,meta['e0']-.2)>min(.65,meta['e0']+.2):
            write(root/'fitting'/'status.json',dict(selected=None,status='PRIOR_GUARD_CONFLICT')); return
    else:
        if (root/'fitting'/'status.json').exists(): raise ValueError('终态禁止续算')
        saved=json.loads((root/'inputs'/'manifest.json').read_text())
        if saved['hashes']!=meta['hashes']: raise ValueError('输入哈希变化')
        checkpoint_path=Path(resume).resolve()
        if root.resolve() not in checkpoint_path.parents: raise ValueError('续算路径不在本运行目录')
        state=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if state['stage'] not in O.STAGES or checkpoint_path.parent.name!=state['stage']:
            raise ValueError('续算阶段不符')
        latest=sorted(checkpoint_path.parent.glob('checkpoint_*.pt'))[-1].resolve()
        if latest!=checkpoint_path: raise ValueError('只能续算最后完整 checkpoint')
        print('RESUME',str(checkpoint_path),flush=True)
    waiting=state is not None
    for stage in O.STAGES:
        if waiting and stage!=state['stage']: continue
        out=root/'fitting'/stage
        opt=O.make_optimizer(stage,params,meta['c0'])
        if waiting:
            O.restore(state,params,opt); start=state['step']; waiting=False
            history=json.loads((out/'history.json').read_text()) if start else []
            if len(history)!=start: raise ValueError('历史与 checkpoint 不匹配')
        else:
            out.mkdir(exist_ok=False); history=[]; start=0
            torch.save(O.checkpoint(params,opt,stage,0),out/'checkpoint_0000.pt')
        for step in range(start+1,O.STAGES[stage][0]+1):
            O.step(data,params,opt,stage,step,meta['K'],meta['scale'],meta['sign'])
            with torch.no_grad():
                values=O.objective_backward(data,params,meta['K'],meta['scale'],meta['sign'],backward=False)
            history.append(values)
            if step%100==0 or step==O.STAGES[stage][0]:
                torch.save(O.checkpoint(params,opt,stage,step),out/f'checkpoint_{step:04d}.pt')
                write(out/'history.json',history)
                print(stage,step,values,flush=True)
        result=diagnostic(data,params,meta)
        tail=np.asarray([x['total'] for x in history[-100:]])
        result['convergence_relative_difference']=float(abs(tail[50:].mean()-tail[:50].mean())/max(abs(tail[:50].mean()),1e-12))
        result['converged']=result['convergence_relative_difference']<=1e-3
        write(out/'diagnostic.json',result)
        if not result['pass_guard']:
            write(root/'fitting'/'status.json',dict(selected=None,status=result['status'],stage=stage)); return
    write(root/'fitting'/'status.json',dict(selected='C3',status='FIT_COMPLETE'))
