"""固定T05几何与明暗标签，仅细化方向离散化；保留时间不参与选择。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from scripts.fit_dl_shadow_tracks import project_shadow, sample_shadow, interpolate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = json.loads((args.initial/'result.json').read_text())
    checkpoint = torch.load(args.initial/'checkpoint.pt', map_location='cuda', weights_only=False)
    data = torch.load(args.cache, map_location='cuda', weights_only=False)
    assert not result['gt_read'] and result['cache_sha256'] == sha(args.cache) == checkpoint['cache_sha256']
    xyz, positions, valid = data['canonical_xyz'], data['positions'], data['valid']
    times, train, usable = data['times'], checkpoint['train'], checkpoint['usable']
    origin, normal, basis = (checkpoint[k] for k in ('plane_origin','plane_normal','basis'))
    positive, negative, occluder = (checkpoint[k] for k in ('positive','negative','occluder'))
    # 原栅格边界来自亮度过滤之前的接收点，严格复现，不能直接用最终receiver。
    support = valid[train].sum(0) >= 10
    extent = float((xyz[support].amax(0)-xyz[support].amin(0)).norm())
    motion = (positions[train]-positions[train].mean(0)[None]).square().sum(-1).mean(0).sqrt()
    receiver = support & (((xyz-origin)@normal).abs() < extent*.01) & (motion < extent*.01)
    uv = (xyz[receiver]-origin)@basis.T
    lower, span = uv.amin(0), (uv.amax(0)-uv.amin(0)).clamp_min(.01)
    lower, span = lower-.05*span, span*1.1

    def score(index, directions):
        grid = project_shadow(positions[index,occluder], directions, origin, normal, basis, lower, span)
        pred, inside = sample_shadow(grid, positions[index], origin, basis, lower, span)
        assert not (positive[index] & ~inside).any() and not (negative[index] & ~inside).any()
        pos, neg = positive[index], negative[index]
        error = .5*((1-pred)*pos[None]).sum(-1)/pos.sum().clamp_min(1) \
                +.5*(pred*neg[None]).sum(-1)/neg.sum().clamp_min(1)
        return error, pred

    indices = torch.where(train)[0].tolist()
    seeds = checkpoint['smooth'][train]
    nearest = checkpoint['candidates']@checkpoint['candidates'].T
    nearest.fill_diagonal_(-1)
    spacing = float(torch.rad2deg(nearest.max(-1).values.clamp(-1,1).acos()).median())
    radius = math.ceil(spacing)
    assert radius == 6, '当前预注册只支持原512候选约5.9度间距。'
    offsets = torch.tensor([(x,y) for x in range(-radius,radius+1)
                            for y in range(-radius,radius+1) if x*x+y*y <= radius*radius],
                           dtype=seeds.dtype, device='cuda')*math.pi/180
    all_candidates, all_costs, baseline_costs = [], [], []
    max_reproduction_error = 0.
    for index, center in zip(indices,seeds):
        old_error, old_pred = score(index,center[None])
        expected = checkpoint['predicted']['smooth'][index].to('cuda')
        difference = float((old_pred[0]-expected).abs().max())
        max_reproduction_error = max(max_reproduction_error,difference)
        if difference != 0:
            raise AssertionError('未精确复现T05投影网格/标签。')
        first = F.normalize(basis[0]-(basis[0]@center)*center,dim=0)
        second = F.normalize(torch.cross(center,first,dim=0),dim=0)
        tangent = offsets[:,0,None]*first+offsets[:,1,None]*second
        angle = tangent.norm(dim=-1,keepdim=True)
        candidates = angle.cos()*center + torch.sinc(angle/math.pi)*tangent
        candidates = F.normalize(candidates,dim=-1)
        candidates = torch.cat([center[None],candidates],dim=0)
        candidates = candidates[(candidates@normal)>=.05]
        error, _ = score(index,candidates)
        all_candidates.append(candidates.cpu().numpy())
        all_costs.append(error.cpu().numpy() if usable[index] else np.zeros(len(error),dtype=np.float32))
        baseline_costs.append(float(old_error[0]) if usable[index] else 0.)
        if len(all_costs)%20==0:
            print(json.dumps({'training_frames':len(all_costs),'candidates':len(candidates),
                              'baseline_cost':float(old_error[0]),'local_min_cost':float(error.min())}),flush=True)
    current, parents = all_costs[0].astype(np.float64), []
    for row in range(1,len(indices)):
        transition = .05*(1-np.clip(all_candidates[row-1]@all_candidates[row].T,-1,1))
        total = current[:,None]+transition
        parent = total.argmin(0)
        current = all_costs[row]+total[parent,np.arange(len(parent))]
        parents.append(parent)
    path = [int(current.argmin())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    path.reverse()
    selected = np.stack([candidates[index] for candidates,index in zip(all_candidates,path)])
    seed_np = seeds.cpu().numpy()
    baseline_objective = sum(baseline_costs)+float((.05*(1-(seed_np[:-1]*seed_np[1:]).sum(-1).clip(-1,1))).sum())
    refined_objective = float(current.min())
    assert refined_objective <= baseline_objective+1e-5, '局部DP没有保留原路径可行性。'
    accepted = refined_objective < baseline_objective-1e-6
    chosen = torch.tensor(selected,device='cuda') if accepted else seeds
    refined = interpolate(times,times[train],chosen)
    frame_errors = []
    for index in range(len(times)):
        error,_ = score(index,refined[index:index+1]);frame_errors.append(error[0])
    errors = torch.stack(frame_errors)
    metrics = {split:float(errors[mask&usable].mean()) for split,mask in [('fit',train),('held_time',~train)]}
    config = {'radius_deg':radius,'step_deg':1,'transition_weight':.05,
              'selection':'只用120训练时刻数据与原时序转移目标；保留时刻仅插值',
              'geometry_labels_grid_frozen':True,'gt_read':False}
    output = dict(result,directions=refined.tolist(),method='local_spherical_grid_dp',
                  config=config,metrics=metrics,accepted_by_training_objective=accepted,
                  baseline_train_path_objective=baseline_objective,refined_train_path_objective=refined_objective,
                  original_prediction_max_abs=max_reproduction_error,
                  candidate_nearest_neighbor_median_deg=spacing,
                  initial_result_sha256=sha(args.initial/'result.json'),
                  initial_checkpoint_sha256=sha(args.initial/'checkpoint.pt'),
                  script_sha256=sha(__file__),seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    torch.save({'directions':refined.cpu(),'train':train.cpu(),'candidate_offsets_rad':offsets.cpu(),
                'path':path,'costs':all_costs,'candidates':all_candidates,'config':config},args.output/'checkpoint.pt')
    print(json.dumps({'accepted':accepted,'baseline_objective':baseline_objective,
                      'refined_objective':refined_objective,'metrics':metrics,'gt_read':False}),flush=True)


if __name__=='__main__':
    main()
