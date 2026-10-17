"""冻结T03阴影几何，仅以原训练内15验证帧选择时间正则，零模型更新。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from scripts.fit_dl_shadow_tracks import project_shadow,sample_shadow,interpolate


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--baseline-local',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    original=torch.load(args.initial/'checkpoint.pt',map_location='cuda',weights_only=False)
    initial_result=json.loads((args.initial/'result.json').read_text())
    data=torch.load(args.cache,map_location='cuda',weights_only=False)
    assert original['cache_sha256']==initial_result['cache_sha256']==sha(args.cache)
    assert not initial_result['gt_read'] and 'occluder_positions_override' in original
    source_paths=[args.cache,args.initial/'checkpoint.pt',args.initial/'result.json',args.baseline_local,
                  Path(__file__),Path(__file__).with_name('refine_static_shadow_temporal.py')]
    sources={str(p.resolve()):sha(p) for p in source_paths}
    train,usable=original['train'],original['usable'];times=data['times']
    xyz,pos,valid=data['canonical_xyz'],data['positions'],data['valid']
    origin,normal,basis=[original[k] for k in ['plane_origin','plane_normal','basis']]
    support=valid[train].sum(0)>=10;extent=float((xyz[support].amax(0)-xyz[support].amin(0)).norm())
    motion=(pos[train]-pos[train].mean(0)[None]).square().sum(-1).mean(0).sqrt()
    receiver=support&(((xyz-origin)@normal).abs()<extent*.01)&(motion<extent*.01)
    uv=(xyz[receiver]-origin)@basis.T;lower=uv.amin(0);span=(uv.amax(0)-lower).clamp_min(.01)
    lower=lower-.05*span;span=span*1.1
    def errors(directions):
        values=[];predictions=[]
        for i in range(len(times)):
            grid=project_shadow(original['occluder_positions_override'],directions[i:i+1],origin,normal,basis,lower,span)
            pred,inside=sample_shadow(grid,pos[i],origin,basis,lower,span)
            positive,negative=original['positive'][i],original['negative'][i]
            assert not (positive&~inside).any() and not (negative&~inside).any()
            values.append(.5*((1-pred[0])*positive).sum()/positive.sum().clamp_min(1)+.5*(pred[0]*negative).sum()/negative.sum().clamp_min(1))
            predictions.append(pred[0])
        return torch.stack(values),torch.stack(predictions)
    candidates=original['candidates'];candidate_np=candidates.cpu().numpy()
    costs=original['costs'][train].cpu().numpy()
    weights=[.05,.20,.80]
    plan={'weights':weights,'geometry':'原T04方向来源的固定T03@16000预测遮挡点，保持同一基线',
          'split':'原135 TRAIN中的120拟合/15调参验证；不是多折CV、不是全管线未见数据',
          'selection':'15验证帧均衡shadow proxy均值最小，数值差<=1e-8时取较小权重',
          'coarse':'相同512候选完整DP，已冻结候选代价只取120拟合行；非复用旧局部中心',
          'local':'每权重粗路径重新6度/1度细化，同权重贯穿两阶段',
          'external_15_test':'完全隔离，不读取图像/GT灯向','gt_read':False,'optimizer_steps':0}
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    results=[]
    for weight in weights:
        folder=args.output/f'weight_{weight:.2f}';folder.mkdir();coarse=folder/'coarse';coarse.mkdir()
        transition=weight*(1-np.clip(candidate_np@candidate_np.T,-1,1))
        current=costs[0].copy();parents=[]
        for row in costs[1:]:
            total=current[:,None]+transition;parent=total.argmin(0)
            current=row+total[parent,np.arange(len(candidate_np))];parents.append(parent)
        path=[int(current.argmin())]
        for parent in reversed(parents):path.append(int(parent[path[-1]]))
        path.reverse();directions=interpolate(times,times[train],candidates[torch.tensor(path,device='cuda')])
        if weight==.05:
            assert torch.equal(directions,original['smooth']), '0.05粗路径未逐元素复现旧基线。'
        error,predicted=errors(directions)
        checkpoint={k:v.cpu() if torch.is_tensor(v) else v for k,v in original.items()}
        checkpoint.update(smooth=directions.cpu(),predicted={'smooth':predicted.cpu()})
        torch.save(checkpoint,coarse/'checkpoint.pt')
        result=dict(initial_result,directions=directions.tolist(),temporal_weight=weight,gt_read=False,
                    metrics={'fit':float(error[train&usable].mean()),'held_time':float(error[~train&usable].mean())})
        result.pop('cross_geometry_metrics',None)
        result['config']=dict(initial_result.get('config',{}),output=str(coarse.resolve()),temporal_weight=weight)
        result['method']='cached_fixed_geometry_temporal_dp'
        (coarse/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        argv=[sys.executable,'-u','-m','scripts.refine_static_shadow_temporal','--cache',str(args.cache.resolve()),
              '--initial',str(coarse.resolve()),'--output',str((folder/'local').resolve()),'--transition-weight',str(weight)]
        (folder/'local_command.json').write_text(json.dumps({'argv':argv,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))
        with (folder/'local.log').open('w') as log:
            rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT).returncode
        (folder/'local_exit.json').write_text(json.dumps({'exit_code':rc}))
        assert rc==0
        local=json.loads((folder/'local/result.json').read_text());direction=torch.tensor(local['directions'],device='cuda')
        if weight==.05:
            old=np.asarray(json.loads(args.baseline_local.read_text())['directions'])
            assert np.array_equal(old,np.asarray(local['directions'])), '0.05局部结果未逐元素复现原T04方向。'
        error,_=errors(direction);fit=error[train&usable].cpu().numpy();validation=error[~train&usable].cpu().numpy()
        direction_np=direction.cpu().numpy();dt=np.diff(times.cpu().numpy())
        angle=np.degrees(np.arccos(np.clip(np.sum(direction_np[:-1]*direction_np[1:],axis=-1),-1,1)))
        stats={'weight':weight,'fit_data_mean':float(fit.mean()),'fit_data_sum':float(fit.sum()),
               'fit_full_objective':local['refined_train_path_objective'],
               'fit_transition':local['refined_train_path_objective']-float(fit.astype(np.float64).sum()),
               'validation_mean':float(validation.mean()),'validation_median':float(np.median(validation)),
               'validation_p90':float(np.quantile(validation,.9)),'validation_max':float(validation.max()),
               'validation_frames':[data['frames'][i]['name'] for i in torch.where(~train&usable)[0].tolist()],
               'validation_per_frame':validation.tolist(),'direction_step_deg':{'mean':float(angle.mean()),'p90':float(np.quantile(angle,.9)),'max':float(angle.max())},
               'angular_velocity_deg_per_time':{'mean':float((angle/dt).mean()),'max':float((angle/dt).max())},
               'local_result_sha256':sha(folder/'local/result.json')}
        results.append(stats);print(json.dumps(stats,ensure_ascii=False),flush=True)
    best=min(x['validation_mean'] for x in results)
    selected=min(x['weight'] for x in results if x['validation_mean']<=best+1e-8)
    baseline=np.array(results[0]['validation_per_frame'])
    for result in results:
        change=np.array(result['validation_per_frame'])-baseline
        result['validation_difference_vs_005']=change.tolist()
        result['validation_improved_frames']=int((change < -1e-8).sum())
        positive=np.sort(np.maximum(-change,0))[::-1]
        result['top2_share_of_positive_gain']=float(positive[:2].sum()/max(positive.sum(),1e-12))
    assert all(sha(Path(path))==expected for path,expected in sources.items())
    summary={'status':'LOCKED_DIAGNOSTIC_ONLY','selected_weight':selected,'candidates':results,'plan':plan,
             'sources':sources,'gt_read':False,'optimizer_steps':0,'formal_training_started':False,
             'baseline_005_directions_exact_reproduction':True,
             'limits':'内部15已用于选权重，不再是独立泛化证据；没有原15测试GT灯误差和新法线结果。正式训练必须等待T06完整局部及全局验收。'}
    (args.output/'selection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
