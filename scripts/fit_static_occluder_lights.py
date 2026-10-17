"""冻结原T05观测、平面与标签，只替换无GT静态模型遮挡几何后重新估光。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from plyfile import PlyData
import torch

from scripts.fit_dl_shadow_tracks import project_shadow, sample_shadow, interpolate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--baseline-result', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, default=16000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--transition-weight',type=float,choices=[.05,.80],default=.05,
                   help='显式固定粗DP时间权重；本接口不选择或扫描权重。')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial = json.loads((args.initial/'result.json').read_text())
    baseline = json.loads(args.baseline_result.read_text())
    old = torch.load(args.initial/'checkpoint.pt',map_location='cuda',weights_only=False)
    data = torch.load(args.cache,map_location='cuda',weights_only=False)
    assert initial['cache_sha256'] == baseline['cache_sha256'] == old['cache_sha256'] == sha(args.cache)
    assert not initial['gt_read'] and not baseline['gt_read']
    marker_path = args.model/'unknown_light_training.json'
    marker = json.loads(marker_path.read_text())
    assert marker['gt_supervision'] is False and marker['actual_entrypoint']=='scripts/train_static_unknown_light.py'
    deform_path = args.model/f'deform/iteration_{args.iteration}/deform.pth'
    deformation = torch.load(deform_path,map_location='cpu',weights_only=True)
    for head in ('gaussian_warp','gaussian_rotation','gaussian_scaling'):
        for suffix in ('weight','bias'):
            assert torch.count_nonzero(deformation[f'{head}.{suffix}'])==0, '必须严格静态。'
    ply_path = args.model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    vertices = PlyData.read(ply_path)['vertex']
    new_xyz = np.stack([vertices[k] for k in ('x','y','z')],-1)
    opacity_logits = np.asarray(vertices['opacity'])
    assert np.isfinite(new_xyz).all() and np.isfinite(opacity_logits).all()
    # 与旧观测缓存相同的64体素代表规则，不能按GT语义筛遮挡物。
    cells = np.floor((new_xyz-new_xyz.min(0))/np.maximum(np.ptp(new_xyz,axis=0),1e-6)*63).astype(np.int64)
    keys = cells[:,0]+64*cells[:,1]+4096*cells[:,2]
    priority = np.argsort(-opacity_logits,kind='stable')
    _, first = np.unique(keys[priority],return_index=True)
    ids = priority[first]
    if len(ids)>8192:
        ids = np.random.default_rng(1015).choice(ids,8192,replace=False)
    selected = torch.tensor(new_xyz[np.sort(ids)],device='cuda')
    xyz, positions, valid = (data[k] for k in ('canonical_xyz','positions','valid'))
    times, train, usable = data['times'],old['train'],old['usable']
    origin, normal, basis = (old[k] for k in ('plane_origin','plane_normal','basis'))
    positive, negative, old_occluder = (old[k] for k in ('positive','negative','occluder'))
    support = valid[train].sum(0)>=10
    extent = float((xyz[support].amax(0)-xyz[support].amin(0)).norm())
    motion = (positions[train]-positions[train].mean(0)[None]).square().sum(-1).mean(0).sqrt()
    receiver = support & (((xyz-origin)@normal).abs()<extent*.01) & (motion<extent*.01)
    uv = (xyz[receiver]-origin)@basis.T
    lower, span = uv.amin(0),(uv.amax(0)-uv.amin(0)).clamp_min(.01)
    lower, span = lower-.05*span,span*1.1
    blockers = selected[((selected-origin)@normal)>extent*.015]
    assert len(blockers)>=32
    candidates = old['candidates']

    def predict(index,directions,new_geometry):
        points = blockers if new_geometry else positions[index,old_occluder]
        grid = project_shadow(points,directions,origin,normal,basis,lower,span)
        prediction,inside = sample_shadow(grid,positions[index],origin,basis,lower,span)
        assert not (positive[index]&~inside).any() and not (negative[index]&~inside).any()
        pos,neg = positive[index],negative[index]
        error = .5*((1-prediction)*pos[None]).sum(-1)/pos.sum().clamp_min(1) \
                +.5*(prediction*neg[None]).sum(-1)/neg.sum().clamp_min(1)
        return error,prediction

    costs=[]
    for index in range(len(times)):
        error,_ = predict(index,candidates,True)
        costs.append(error if usable[index] else torch.zeros_like(error))
        if index%20==0:
            print(json.dumps({'frame':index,'minimum_cost':float(error.min()),'new_occluders':len(blockers)}),flush=True)
    costs=torch.stack(costs)
    fit_cost=costs[train].cpu().numpy(); c=candidates.cpu().numpy()
    transition=args.transition_weight*(1-np.clip(c@c.T,-1,1))
    current=fit_cost[0].copy();parents=[]
    for row in fit_cost[1:]:
        total=current[:,None]+transition;parent=total.argmin(0)
        current=row+total[parent,np.arange(len(c))];parents.append(parent)
    path=[int(current.argmin())]
    for parent in reversed(parents):path.append(int(parent[path[-1]]))
    path.reverse()
    smooth=interpolate(times,times[train],candidates[torch.tensor(path,device='cuda')])
    previous=torch.tensor(baseline['directions'],device='cuda')
    cross={};predicted={}
    for geom in (False,True):
        for name,directions in [('baseline_fixed',previous),('new_coarse',smooth)]:
            errors=[];predictions=[]
            for index in range(len(times)):
                error,pred=predict(index,directions[index:index+1],geom)
                errors.append(error[0]);predictions.append(pred[0])
            errors=torch.stack(errors)
            cross[('new_geometry/' if geom else 'old_geometry/')+name]={
                split:float(errors[mask&usable].mean()) for split,mask in [('fit',train),('held_time',~train)]}
            if geom and name=='new_coarse':predicted['smooth']=torch.stack(predictions).cpu()
    source_paths=[marker_path,ply_path,deform_path,args.baseline_result,args.initial/'checkpoint.pt',Path(__file__)]
    sources={str(path.resolve()):sha(path) for path in source_paths}
    output=dict(initial,directions=smooth.tolist(),method='fixed_observation_new_static_occluder',
                temporal_weight=args.transition_weight,
                config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                metrics=cross['new_geometry/new_coarse'],cross_geometry_metrics=cross,
                cross_geometry_scope='old_geometry为原缓存轨迹遮挡点；不是T03/T08静态几何四格对照。',
                occluders=len(blockers),original_occluders=int(old_occluder.sum()),
                new_geometry_total_points=len(new_xyz),new_geometry_representatives=len(selected),
                all_observations_plane_labels_grid_frozen=True,gt_read=False,
                geometry_sources=sources,seconds=time.monotonic()-started,
                scope='只替换遮挡几何及其固定体素采样表示；粗候选后仍需同6度/1度细化及正式几何评测')
    (args.output/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    checkpoint={k:v.cpu() if torch.is_tensor(v) else v for k,v in old.items()}
    checkpoint.update(costs=costs.cpu(),smooth=smooth.cpu(),predicted=predicted,
                      occluder_positions_override=blockers.cpu(),geometry_sources=sources,
                      transition_weight=args.transition_weight)
    torch.save(checkpoint,args.output/'checkpoint.pt')
    print(json.dumps(cross),flush=True)


if __name__=='__main__':
    main()
