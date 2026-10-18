"""固定TRAIN几何上的连续删失EL+DL初始化；不使用GT或黑暗标签。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def residual(pred,target,saturated):
    delta=pred-target
    return torch.where(saturated,delta.clamp_max(0),delta)


def temporal_path(cost,directions,times):
    # 数据项逐时间减最小值/候选跨度归一，避免继承DLv2损失量纲。
    c=cost.cpu().numpy();d=directions.cpu().numpy();t=times.cpu().numpy()
    spread=np.maximum(np.quantile(c,.9,axis=1)-c.min(1),1e-6)
    c=(c-c.min(1,keepdims=True))/spread[:,None]
    angle_cost=.05*(1-np.clip(d@d.T,-1,1))
    dt=np.median(np.diff(t));now=c[0];parents=[]
    for i,row in enumerate(c[1:],1):
        total=now[:,None]+angle_cost*dt/max(t[i]-t[i-1],1e-6)
        parent=total.argmin(0);now=row+total[parent,np.arange(len(d))];parents.append(parent)
    path=[int(now.argmin())]
    for parent in reversed(parents):path.append(int(parent[path[-1]]))
    return torch.tensor(list(reversed(path)),device=directions.device)


def interpolate(times,known_times,known):
    right=torch.searchsorted(known_times,times).clamp(1,len(known_times)-1);left=right-1
    w=((times-known_times[left])/(known_times[right]-known_times[left])).clamp(0,1)
    return F.normalize(torch.lerp(known[left],known[right],w[:,None]),dim=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.resolve(),a.cache.parent/'manifest.json',out])
    meta=json.loads((a.cache.parent/'manifest.json').read_text())
    assert not meta['gt_read'] and sha(a.cache)==meta['cache_sha256']
    data=torch.load(a.cache,map_location='cpu',weights_only=False)
    device='cuda';fit=data['fit'].to(device);times=data['times'].to(device)
    y=data['rgb'].to(device);sat=data['saturated'].to(device);valid=data['valid'].to(device)
    directions=data['directions'].to(device);normal=data['normal'].to(device)
    vis=data['visibility'].to(device);q=vis*(directions@normal).clamp_min(0)[:,None]
    active=valid[fit].sum(0)>=10
    valid &= active[None]
    yf,sf,vf=y[fit],sat[fit],valid[fit]
    supported_frames=vf.sum(1)>=100
    if not supported_frames.all():
        raise ValueError('fit帧不足100个稳定接收点；保留缓存，不猜灯向')
    denom=vf.sum().clamp_min(1)*3
    all_results=[]
    for index,ratio in enumerate((1/3,1.,3.)):
        directory=out/f'init_{index+1}';directory.mkdir()
        path=torch.full((int(fit.sum()),),int((directions@normal).argmax()),device=device,dtype=torch.long)
        initial=(yf*vf[...,None]).sum(0)/vf.sum(0).clamp_min(1)[:,None]/(ratio+1)
        albedo_logits=torch.nn.Parameter(torch.logit(initial.clamp(.001,.95)))
        log_environment=torch.nn.Parameter(torch.full((3,),math.log(math.pi*ratio),device=device))
        optimizer=torch.optim.Adam([albedo_logits,log_environment],lr=.03)
        rounds=[]
        for outer in range(5):
            for step in range(200):
                albedo=albedo_logits.sigmoid()
                environment=log_environment.exp()
                pred=albedo[None]*(environment[None,None]/math.pi+q[path][...,None])
                loss=(residual(pred,yf,sf).square()*vf[...,None]).sum()/denom
                optimizer.zero_grad();loss.backward();optimizer.step()
                with torch.no_grad():
                    log_environment.clamp_(math.log(math.pi*.01),math.log(math.pi*30.))
                    albedo_logits.clamp_(torch.logit(torch.tensor(.0001)).item(),torch.logit(torch.tensor(.97)).item())
            with torch.no_grad():
                albedo=albedo_logits.sigmoid();environment=log_environment.exp()
                candidate_pred=albedo[None]*(environment[None,None]/math.pi+q[...,None])
                costs=[]
                for t in range(len(yf)):
                    error=residual(candidate_pred,yf[t][None],sf[t][None]).square()
                    costs.append((error*vf[t][None,:,None]).sum((1,2))/(vf[t].sum().clamp_min(1)*3))
                costs=torch.stack(costs)
                new_path=temporal_path(costs,directions,times[fit])
                change=torch.rad2deg(torch.acos((directions[path]*directions[new_path]).sum(-1).clamp(-1,1)))
                rounds.append(dict(round=outer+1,fit_mse=float(loss),ambient_rgb=environment.tolist(),
                                   direction_change_mean_deg=float(change.mean()),direction_change_p90_deg=float(torch.quantile(change,.9))))
                path=new_path
                print(json.dumps(dict(initial_ratio=ratio,**rounds[-1])),flush=True)
        # 最后方向确定后仅用fit再次拟合能量/共享材质，使所有统计对应该固定路径。
        for step in range(200):
            albedo=albedo_logits.sigmoid();environment=log_environment.exp()
            pred=albedo[None]*(environment[None,None]/math.pi+q[path][...,None])
            loss=(residual(pred,yf,sf).square()*vf[...,None]).sum()/denom
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():
                log_environment.clamp_(math.log(math.pi*.01),math.log(math.pi*30.))
                albedo_logits.clamp_(torch.logit(torch.tensor(.0001)).item(),torch.logit(torch.tensor(.97)).item())
        with torch.no_grad():
            albedo=albedo_logits.sigmoid();environment=log_environment.exp()
            smooth=interpolate(times,times[fit],directions[path])
            # held方向插值后用最近候选透射率只作初始化诊断；正式输出须再做精确BVH验证。
            nearest=(smooth@directions.T).argmax(-1)
            predicted=albedo[None]*(environment[None,None]/math.pi+q[nearest][...,None])
            per=(residual(predicted,y,sat).square()*valid[...,None]).sum((1,2))/(valid.sum(1).clamp_min(1)*3)
            enough=valid.sum(1)>=100
            metrics={name:float(per[mask&enough].mean()) for name,mask in [('fit',fit),('held',~fit)]}
            # 常量表面颜色对照仅拟合fit，逐通道闭式均值后对删失通道再优化。
            constant=torch.nn.Parameter(((yf*vf[...,None]).sum(0)/vf.sum(0).clamp_min(1)[:,None]).clone())
        opt0=torch.optim.Adam([constant],lr=.03)
        for step in range(200):
            loss0=(residual(constant[None],yf,sf).square()*vf[...,None]).sum()/denom
            opt0.zero_grad();loss0.backward();opt0.step()
            with torch.no_grad():constant.clamp_(0,30)
        with torch.no_grad():
            null_per=(residual(constant[None],y,sat).square()*valid[...,None]).sum((1,2))/(valid.sum(1).clamp_min(1)*3)
            metrics.update({f'constant_{name}':float(null_per[mask&enough].mean()) for name,mask in [('fit',fit),('held',~fit)]})
            result=dict(gt_read=False,initial_ratio=ratio,metrics=metrics,rounds=rounds,
                        ambient_rgb=environment.tolist(),direct_intensity=math.pi,directions=smooth.tolist(),
                        fit_directions=directions[path].tolist(),held_supported_frames=int(((~fit)&enough).sum()),
                        bound_active=bool(((environment/math.pi<.0101)|(environment/math.pi>29.9)).any()),
                        absolute_energy_gauge='固定白DL强度π；估计EL/DL相对能量，不声称Blender绝对功率',
                        held_visibility='nearest_candidate_approximation_pending_exact_BVH')
            torch.save(dict(albedo=albedo.cpu(),environment=environment.cpu(),directions=smooth.cpu(),
                            path=path.cpu(),prediction=predicted.cpu(),per_frame_mse=per.cpu()),directory/'fit.pt')
            (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));all_results.append(result)
    # 只按fit选择数值初值；held仅作被冻结方案的可靠性验证。
    chosen=min(range(3),key=lambda i:all_results[i]['metrics']['fit'])
    arrays=torch.tensor([r['fit_directions'] for r in all_results])
    agreement=[]
    for i in range(3):
        for j in range(i):
            angle=torch.rad2deg(torch.acos((arrays[i]*arrays[j]).sum(-1).clamp(-1,1)))
            agreement.append(dict(pair=[j+1,i+1],median_deg=float(angle.median()),p90_deg=float(torch.quantile(angle,.9))))
    selected=all_results[chosen]
    support_ok=selected['held_supported_frames']>=12
    preliminary=support_ok and not selected['bound_active'] and selected['metrics']['held']<selected['metrics']['constant_held'] and all(x['median_deg']<15 for x in agreement)
    summary=dict(status='PENDING_EXACT_BVH_VALIDATION' if preliminary else 'FAILED_INTERNAL_VALIDATION',
                 gt_read=False,selected_by='fit_censored_mse_only',selected_init=chosen+1,
                 selected=selected,three_start_agreement=agreement,preliminary_checks_passed=preliminary,
                 internal_validation_passed=False,cache_sha256=sha(a.cache),
                 note='不导出训练灯光，必须先对真实插值方向重算BVH并检查held/偏置稳定性与代表fit图')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
