"""由训练表面的明暗轨迹与预测物体投影估计主光；不读取GT资产。"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def project_shadow(points, directions, origin, normal, basis, lower, span, resolution=128):
    height=(points-origin)@normal
    projected=(points-origin)[None]-height[None,:,None]/(directions@normal)[:,None,None].clamp_min(.001)*directions[:,None]
    xy=projected@basis.T
    index=((xy-lower)/span*resolution).floor().long()
    inside=((index>=0)&(index<resolution)).all(-1)
    flat=index[...,1].clamp(0,resolution-1)*resolution+index[...,0].clamp(0,resolution-1)
    grid=torch.zeros(len(directions),resolution*resolution,device=points.device)
    grid.scatter_add_(1,flat,inside.float())
    return F.max_pool2d(grid.clamp_max(1).view(-1,1,resolution,resolution),3,stride=1,padding=1)[:,0].flatten(1)


def sample_shadow(grid, points, origin, basis, lower, span, resolution=128):
    xy=(points-origin)@basis.T
    index=((xy-lower)/span*resolution).floor().long()
    inside=((index>=0)&(index<resolution)).all(-1)
    flat=index[:,1].clamp(0,resolution-1)*resolution+index[:,0].clamp(0,resolution-1)
    return grid[:,flat],inside


def interpolate(times, train_times, directions):
    right=torch.searchsorted(train_times,times).clamp(1,len(train_times)-1);left=right-1
    w=((times-train_times[left])/(train_times[right]-train_times[left])).clamp(0,1)
    return F.normalize(torch.lerp(directions[left],directions[right],w[:,None]),dim=-1)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--baseline-result',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=1015)
    p.add_argument('--split-checkpoint',required=True)
    p.add_argument('--geometry-state',choices=['dynamic','frozen'],default='dynamic')
    p.add_argument('--brightness-reference',choices=['canonical','global'],default='canonical')
    args=p.parse_args();started=time.monotonic();torch.manual_seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('禁止覆盖已有结果。')
    data=torch.load(args.cache,weights_only=False,map_location='cpu')
    baseline=json.loads(Path(args.baseline_result).read_text())
    cache_hash=hashlib.sha256(Path(args.cache).read_bytes()).hexdigest()
    assert baseline['cache_sha256']==cache_hash and not baseline.get('gt_direction_reference',False)
    xyz=data['canonical_xyz'].cuda();positions=data['positions'].cuda();normals=data['normals'].cuda()
    valid=data['valid'].cuda();times=data['times'].cuda();rgb=data['rgb'].cuda()
    split=torch.load(args.split_checkpoint,map_location='cpu',weights_only=False)
    assert torch.equal(split['times'],data['times'])
    train=torch.zeros(len(times),dtype=torch.bool,device='cuda');train[split['train_frames']]=True
    held=torch.zeros_like(train);held[split['held_frames']]=True
    assert not (train&held).any() and train.sum()==45 and held.sum()==10
    support=valid[train].sum(0)>=10
    cloud=xyz[support]
    if len(cloud)<100:raise ValueError('平面拟合支持不足。')
    extent=float((cloud.amax(0)-cloud.amin(0)).norm())
    generator=torch.Generator(device='cuda').manual_seed(args.seed)
    triples=cloud[torch.randint(len(cloud),(512,3),device='cuda',generator=generator)]
    candidate_n=F.normalize(torch.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0],dim=-1),dim=-1)
    distance=(cloud@candidate_n.T-(triples[:,0]*candidate_n).sum(-1)[None]).abs()
    scores=(distance<extent*.005).sum(0)
    scores[candidate_n.norm(dim=-1)<.9]=0
    best=int(scores.argmax());inliers=distance[:,best]<extent*.005
    origin=cloud[inliers].mean(0)
    _,_,vectors=torch.linalg.svd(cloud[inliers]-origin,full_matrices=False)
    normal=vectors[-1]
    if ((cloud-origin)@normal).mean()<0:normal=-normal
    first=F.normalize(vectors[0]-(vectors[0]@normal)*normal,dim=0)
    basis=torch.stack([first,F.normalize(torch.cross(normal,first,dim=0),dim=0)])
    motion=(positions[train]-positions[train].mean(0)[None]).square().sum(-1).mean(0).sqrt()
    receiver=support&(((xyz-origin)@normal).abs()<extent*.01)&(motion<extent*.01)
    if receiver.sum()<100:raise ValueError('稳定平面接收区域不足100个点。')
    uv=(xyz[receiver]-origin)@basis.T
    lower=uv.amin(0);span=(uv.amax(0)-lower).clamp_min(.01)
    lower=lower-.05*span;span=span*1.1
    luminance=rgb.mean(-1)
    reference=torch.nanquantile(torch.where(valid[train],luminance[train],torch.nan),.9,dim=0)
    if args.brightness_reference=='global':
        reference=torch.full_like(reference,float(torch.quantile(luminance[train][valid[train]&receiver[None]],.9)))
    receiver &= torch.isfinite(reference)&(reference>.05)
    ratio=luminance/reference.clamp_min(.05)[None]
    eligible=valid&receiver[None]&(((positions-origin)@normal).abs()<extent*.015)
    positive=eligible&(ratio<.2);negative=eligible&(ratio>.65)
    usable=(positive.sum(-1)>=8)&(negative.sum(-1)>=32)
    if (usable&train).sum()<10:raise ValueError('明暗支持足够的训练时间不足10。')
    k=torch.arange(512,device='cuda');z=.05+.95*(k+.5)/512;phi=k*math.pi*(3-math.sqrt(5));r=(1-z*z).sqrt()
    candidates=(r*phi.cos())[:,None]*basis[0]+(r*phi.sin())[:,None]*basis[1]+z[:,None]*normal
    occluder=((xyz-origin)@normal)>extent*.015
    if occluder.sum()<32:raise ValueError('离开接收平面的遮挡物支持不足。')
    costs=[]
    occluder_positions=positions if args.geometry_state=='dynamic' else positions[:1].expand_as(positions)
    for index in range(len(times)):
        grid=project_shadow(occluder_positions[index,occluder],candidates,origin,normal,basis,lower,span)
        pred,inside=sample_shadow(grid,positions[index],origin,basis,lower,span)
        pos=positive[index]&inside;neg=negative[index]&inside
        usable[index]=(pos.sum()>=8)&(neg.sum()>=32)
        cost=.5*((1-pred)*pos[None]).sum(-1)/pos.sum().clamp_min(1)+.5*(pred*neg[None]).sum(-1)/neg.sum().clamp_min(1)
        costs.append(cost if usable[index] else torch.zeros_like(cost))
        positive[index]=pos;negative[index]=neg
        if index%20==0:print(json.dumps({'frame':index,'positive':int(pos.sum()),'negative':int(neg.sum()),'min_cost':float(cost.min())}),flush=True)
    costs=torch.stack(costs)
    # 只用拟合时间的数据项；保留时间观测不参与独立选择或动态规划。
    fit_cost=costs[train].cpu().numpy();candidates_np=candidates.cpu().numpy()
    transition=.05*(1-np.clip(candidates_np@candidates_np.T,-1,1))
    current=fit_cost[0].copy();back=[]
    for row in fit_cost[1:]:
        total=current[:,None]+transition
        parent=total.argmin(0);back.append(parent);current=row+total[parent,np.arange(512)]
    path=[int(current.argmin())]
    for parent in reversed(back):path.append(int(parent[path[-1]]))
    path=list(reversed(path))
    smooth=interpolate(times,times[train],candidates[torch.tensor(path,device='cuda')])
    supported_train=train&usable
    if supported_train.sum()<2:raise ValueError('有效训练时刻不足以插值。')
    independent=interpolate(times,times[supported_train],candidates[costs[supported_train].argmin(-1)])
    metrics={};predicted={}
    for name,directions in [('independent',independent),('smooth',smooth)]:
        all_pred=[]
        for i in range(len(times)):
            grid=project_shadow(occluder_positions[i,occluder],directions[i:i+1],origin,normal,basis,lower,span)
            pred,_=sample_shadow(grid,positions[i],origin,basis,lower,span);all_pred.append(pred[0])
        pred=torch.stack(all_pred);predicted[name]=pred.cpu()
        frame_error=.5*((1-pred)*positive).sum(-1)/positive.sum(-1).clamp_min(1)+.5*(pred*negative).sum(-1)/negative.sum(-1).clamp_min(1)
        metrics[name]={split:float(frame_error[mask&usable].mean()) for split,mask in [('fit',train),('held_time',held)]}
    common=dict(frames=[f['name'] for f in data['frames']],times=times.tolist(),train_indices=torch.where(train)[0].tolist(),held_indices=torch.where(held)[0].tolist(),cache_sha256=cache_hash)
    for name,directions in [('independent',independent),('smooth',smooth)]:
        result=dict(**common,directions=directions.tolist(),config=vars(args),metrics=metrics,
                    receivers=int(receiver.sum()),occluders=int(occluder.sum()),usable_train=int((usable&train).sum()),usable_held=int((usable&held).sum()),
                    plane_normal=normal.tolist(),plane_origin=origin.tolist(),scene_extent=extent,
                    seconds=time.monotonic()-started,gt_read=False,method=name,
                    scope='仅45训练时刻定义支持、亮度参考、候选选择与路径；10保留时刻不参与拟合；共用既有Stage1并非端到端未见帧',split_checkpoint_sha256=hashlib.sha256(Path(args.split_checkpoint).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (out/('result.json' if name=='smooth' else 'independent_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    torch.save(dict(candidates=candidates.cpu(),costs=costs.cpu(),train=train.cpu(),held=held.cpu(),usable=usable.cpu(),brightness_reference=reference.cpu(),lower=lower.cpu(),span=span.cpu(),
                    positive=positive.cpu(),negative=negative.cpu(),receiver=receiver.cpu(),occluder=occluder.cpu(),
                    predicted=predicted,plane_origin=origin.cpu(),plane_normal=normal.cpu(),basis=basis.cpu(),
                    smooth=smooth.cpu(),independent=independent.cpu(),cache_sha256=cache_hash),out/'checkpoint.pt')
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
