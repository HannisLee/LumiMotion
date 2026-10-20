"""仅TRAIN的V3动态T05投影阴影估光；冻结后另进程评价。"""
import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel, load_train_cameras
from LIHAN.scripts.dynamic_v3_tracks import build_observations
from scripts.fit_dl_shadow_tracks import project_shadow, sample_shadow, interpolate
from utils.no_gt_guard import install_no_gt_guard


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def path_dp(costs, candidates, frames):
    """候选可以逐帧不同；仅使用有支持的TRAIN时刻。"""
    current=costs[0].astype(np.float64); back=[]
    for i in range(1,len(costs)):
        transition=.05*(1-np.clip(candidates[i-1]@candidates[i].T,-1,1))/(frames[i]-frames[i-1])
        total=current[:,None]+transition
        parent=total.argmin(0); current=costs[i]+total[parent,np.arange(len(parent))]
        back.append(parent)
    path=[int(current.argmin())]
    for parent in reversed(back):path.append(int(parent[path[-1]]))
    path.reverse()
    return np.stack([c[p] for c,p in zip(candidates,path)]),float(current.min()),path


@torch.no_grad()
def run(args):
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(1015)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert out.is_relative_to(Path.cwd()/'output') and not (out/'freeze.json').exists()
    assert not (out/'observations').exists(), '禁止覆盖先前实验'
    source=args.source.resolve();parent=args.parent.resolve()
    manifest=json.loads((source/'no_gt_manifest.json').read_text())
    assert len(manifest['train_frames'])==117 and not manifest['gt_read']
    assert not set(manifest['train_frames'])&set(manifest['dev_frames']+manifest['test_frames'])
    install_no_gt_guard([source,parent,out,args.stats.resolve()])
    inputs=[Path(__file__),Path('scripts/fit_dl_shadow_tracks.py'),Path('LIHAN/scripts/dynamic_v3_surface.py'),
            Path('LIHAN/scripts/dynamic_v3_tracks.py'),source/'no_gt_manifest.json',source/'transforms_train.json',
            source.parent/'input_audit_v1/radiometry_v1.json',args.stats,
            parent/'cfg_args',parent/'point_cloud/iteration_35000/point_cloud.ply',
            parent/'deform/iteration_35000/deform.pth']
    input_hashes={str(p.resolve()):sha(p) for p in inputs}
    stats=json.loads(args.stats.read_text());cameras=load_train_cameras(source)
    frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras]
    assert frames==manifest['train_frames']
    times=torch.tensor([float(c.fid) for c in cameras],device='cuda')
    dummy=dict(times=times,directions=torch.tensor([[0.,0.,1.]],device='cuda').repeat(len(times),1),intensity=math.pi)
    model=DynamicSurfaceModel(parent,35000,dummy,stats,'A')
    orientation=model.initialize_orientation(cameras);write(out/'orientation.json',orientation)
    data=build_observations(model,cameras,out/'observations',maximum=16384,transport=False)
    xyz=data['canonical_xyz'].cuda();positions=data['positions'].cuda();valid=data['valid'].cuda()
    support=valid.sum(0)>=10;cloud=xyz[support]
    assert len(cloud)>=100, '平面拟合支持不足'
    extent=float((cloud.amax(0)-cloud.amin(0)).norm())
    generator=torch.Generator(device='cuda').manual_seed(1015)
    triples=cloud[torch.randint(len(cloud),(512,3),device='cuda',generator=generator)]
    normals=F.normalize(torch.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0],dim=-1),dim=-1)
    distance=(cloud@normals.T-(triples[:,0]*normals).sum(-1)[None]).abs()
    scores=(distance<extent*.005).sum(0);scores[normals.norm(dim=-1)<.9]=0
    inliers=distance[:,int(scores.argmax())]<extent*.005;origin=cloud[inliers].mean(0)
    _,_,vectors=torch.linalg.svd(cloud[inliers]-origin,full_matrices=False);normal=vectors[-1]
    if ((cloud-origin)@normal).mean()<0:normal=-normal
    first=F.normalize(vectors[0]-(vectors[0]@normal)*normal,dim=0)
    basis=torch.stack([first,F.normalize(torch.cross(normal,first,dim=0),dim=0)])
    motion=(positions-positions.mean(0)[None]).square().sum(-1).mean(0).sqrt()
    receiver=support&(((xyz-origin)@normal).abs()<extent*.01)&(motion<extent*.01)
    assert receiver.sum()>=100, '稳定平面不足100点'
    uv=(xyz[receiver]-origin)@basis.T;lower=uv.amin(0);span=(uv.amax(0)-lower).clamp_min(.01)
    lower=lower-.05*span;span=span*1.1
    # 遮挡点来自完整当前预测几何，不能只用相机可见的接收者/轨迹。
    blockers=[];counts=[]
    for camera in cameras:
        camera.load2device();geometry=model.effective(camera)
        chosen=(((geometry.get_xyz-origin)@normal)>extent*.015)&(geometry.get_opacity[:,0]>.1)
        points=geometry.get_xyz[chosen].detach().cpu();blockers.append(points);counts.append(len(points))
        camera.load2device('cpu')
    assert min(counts)>=32
    geometry_cache=dict(blockers=blockers,origin=origin.cpu(),normal=normal.cpu(),basis=basis.cpu(),
        lower=lower.cpu(),span=span.cpu(),receiver=receiver.cpu(),motion=motion.cpu(),extent=extent,
        frames=frames,opacity_cutoff=.1)
    torch.save(geometry_cache,out/'geometry.pt')
    lum=data['observed'].cuda().mean(-1)
    quantiles=torch.nanquantile(torch.where(valid,lum,torch.nan),torch.tensor([.1,.9],device='cuda'),dim=0)
    low,high=quantiles
    eligible=valid&receiver[None]&(((positions-origin)@normal).abs()<extent*.015)
    variants={}
    for name in ('original','ambient_contrast'):
        good=torch.isfinite(high)&(high>.05)
        if name=='original':ratio=lum/high.clamp_min(.05)[None]
        else:
            amplitude=high-low;good &= amplitude>torch.maximum(torch.full_like(high,.03),.1*high)
            ratio=(lum-low[None])/amplitude.clamp_min(1e-6)[None]
        positive=eligible&good[None]&(ratio<.2);negative=eligible&good[None]&(ratio>.65)
        usable=(positive.sum(-1)>=8)&(negative.sum(-1)>=32)
        variants[name]=dict(positive=positive,negative=negative,usable=usable,
            positive_counts=positive.sum(-1).tolist(),negative_counts=negative.sum(-1).tolist(),
            good_receivers=int((receiver&good).sum()))
    write(out/'support.json',dict(frames=frames,receivers=int(receiver.sum()),blocker_counts=counts,
        plane_normal=normal.tolist(),plane_origin=origin.tolist(),scene_extent=extent,
        variants={k:{x:y for x,y in v.items() if not isinstance(y,torch.Tensor)}|
                  dict(usable_train=int(v['usable'].sum())) for k,v in variants.items()}))
    k=torch.arange(512,device='cuda');z=.05+.95*(k+.5)/512;phi=k*math.pi*(3-math.sqrt(5));r=(1-z*z).sqrt()
    candidates=(r*phi.cos())[:,None]*basis[0]+(r*phi.sin())[:,None]*basis[1]+z[:,None]*normal

    def score(index,directions,variant,return_grid=False):
        grids=[]
        for chunk in directions.split(32):
            grids.append(project_shadow(blockers[index].cuda(),chunk,origin,normal,basis,lower,span))
        grid=torch.cat(grids);pred,inside=sample_shadow(grid,positions[index],origin,basis,lower,span)
        pos=variant['positive'][index]&inside;neg=variant['negative'][index]&inside
        error=.5*((1-pred)*pos[None]).sum(-1)/pos.sum().clamp_min(1)+.5*(pred*neg[None]).sum(-1)/neg.sum().clamp_min(1)
        return (error,grid,pred) if return_grid else error

    all_cost={name:[] for name in variants}
    for index in range(len(frames)):
        # 投影一次，共享两个标签版本；两者都不读取灯向GT。
        _,grid,pred=score(index,candidates,variants['ambient_contrast'],True)
        _,inside=sample_shadow(grid,positions[index],origin,basis,lower,span)
        for name,v in variants.items():
            v['positive'][index]&=inside;v['negative'][index]&=inside
            v['usable'][index]=(v['positive'][index].sum()>=8)&(v['negative'][index].sum()>=32)
            p,n=v['positive'][index],v['negative'][index]
            error=.5*((1-pred)*p[None]).sum(-1)/p.sum().clamp_min(1)+.5*(pred*n[None]).sum(-1)/n.sum().clamp_min(1)
            all_cost[name].append(error.cpu())
        if index%10==0:print(json.dumps(dict(projected_frame=frames[index],blockers=counts[index])),flush=True)
    results={};checkpoints={};coarse_np=candidates.cpu().numpy()
    for name,v in variants.items():
        cost=torch.stack(all_cost[name]);indices=torch.where(v['usable'])[0].tolist()
        folder=out/name;folder.mkdir()
        common=dict(gt_read=False,method=name,train_frames=frames,times=times.tolist(),usable_train_frames=[frames[i] for i in indices],
                    receivers=int(receiver.sum()),config=dict(seed=1015,candidates=512,transition_weight=.05,
                    transition_divided_by_frame_gap=True,grid_resolution=128,grid_dilation=3))
        if len(indices)<10:
            result=dict(common,status='FAILED_INSUFFICIENT_SHADOW_SUPPORT');write(folder/'result.json',result);results[name]=result
            torch.save(dict(costs=cost,**{k:v[k].cpu() for k in ['positive','negative','usable']}),folder/'checkpoint.pt')
            continue
        supported_frames=[frames[i] for i in indices]
        seeds,objective,path=path_dp([cost[i].numpy() for i in indices],[coarse_np]*len(indices),supported_frames)
        chosen=torch.tensor(seeds,device='cuda');coarse=interpolate(times,times[indices],chosen)
        result=dict(common,status='PASS_FIT_FROZEN_PENDING_GT',directions=coarse.tolist(),train_objective=objective)
        write(folder/'coarse_result.json',result)
        refine_info={}
        if name=='ambient_contrast':
            offsets=torch.tensor([(x,y) for x in range(-6,7) for y in range(-6,7) if x*x+y*y<=36],
                                 device='cuda',dtype=torch.float32)*math.pi/180
            local_candidates=[];local_costs=[]
            for i,center in zip(indices,chosen):
                tangent1=F.normalize(basis[0]-(basis[0]@center)*center,dim=0)
                tangent2=F.normalize(torch.cross(center,tangent1,dim=0),dim=0)
                tangent=offsets[:,0,None]*tangent1+offsets[:,1,None]*tangent2
                angle=tangent.norm(dim=-1,keepdim=True)
                local=F.normalize(angle.cos()*center+torch.sinc(angle/math.pi)*tangent,dim=-1)
                local=torch.cat([center[None],local]);local=local[(local@normal)>=.05]
                local_candidates.append(local.cpu().numpy());local_costs.append(score(i,local,v).cpu().numpy())
            refined,new_objective,local_path=path_dp(local_costs,local_candidates,supported_frames)
            assert new_objective<=objective+1e-5
            accepted=new_objective<objective-1e-6
            if accepted:chosen=torch.tensor(refined,device='cuda')
            refine_info=dict(radius_deg=6,step_deg=1,accepted_by_train=accepted,coarse_objective=objective,refined_objective=new_objective)
            torch.save(dict(candidates=local_candidates,costs=local_costs,path=local_path),folder/'local_grid.pt')
        prediction=interpolate(times,times[indices],chosen)
        final_errors=torch.stack([score(i,prediction[i:i+1],v)[0] for i in indices])
        result=dict(result,directions=prediction.tolist(),refinement=refine_info,
                    train_balanced_shadow_error=float(final_errors.mean()),primary=name=='ambient_contrast')
        write(folder/'result.json',result);results[name]=result
        torch.save(dict(candidates=candidates.cpu(),costs=cost,path=path,directions=prediction.cpu(),
                        **{k:v[k].cpu() for k in ['positive','negative','usable']}),folder/'checkpoint.pt')
        checkpoints[name]=dict(prediction=prediction,indices=indices)
        print(json.dumps(dict(variant=name,usable=len(indices),train_shadow_error=float(final_errors.mean()),refinement=refine_info)),flush=True)

    # 四类目检中的前三类：固定TRAIN帧，所有结果确定之后生成，不能据图再调参。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    visual=out/'visualizations';visual.mkdir()
    for index in [0,29,58,87,116]:
        fig,axes=plt.subplots(1,4,figsize=(18,4))
        image=cameras[index].original_image_train_light.permute(1,2,0).cpu().numpy()
        axes[0].imshow(image);u,v=data['pixels'][index].T.numpy();ok=data['valid'][index].numpy()&receiver.cpu().numpy()
        axes[0].scatter(u[ok],v[ok],s=1,c='lime');axes[0].set_title(f'TRAIN {frames[index]} observed / receiver')
        world=blockers[index].numpy();xy=(world-origin.cpu().numpy())@basis.cpu().numpy().T
        axes[1].scatter(xy[:,0],xy[:,1],s=.1,c='black');axes[1].set_title('current occluder / plane XY');axes[1].set_aspect('equal')
        for j,name in enumerate(['original','ambient_contrast']):
            ax=axes[j+2];variant=variants[name]
            if name in checkpoints:
                _,grid,_=score(index,checkpoints[name]['prediction'][index:index+1],variant,True)
                lo=lower.cpu().numpy();sp=span.cpu().numpy()
                ax.imshow(grid[0].reshape(128,128).cpu(),origin='lower',extent=[lo[0],lo[0]+sp[0],lo[1],lo[1]+sp[1]],cmap='Greys',alpha=.7)
            xy=((positions[index]-origin)@basis.T).cpu().numpy()
            p=variant['positive'][index].cpu().numpy();n=variant['negative'][index].cpu().numpy()
            ax.scatter(xy[n,0],xy[n,1],s=3,c='gold',label='bright');ax.scatter(xy[p,0],xy[p,1],s=6,c='red',label='dark')
            ax.set_title(f'{name}: dark {p.sum()} bright {n.sum()}');ax.legend(fontsize=6);ax.set_aspect('equal')
        fig.tight_layout();fig.savefig(visual/f'train_{frames[index]:04d}.png',dpi=150);plt.close(fig)
    for p,digest in input_hashes.items():assert sha(p)==digest,p
    frozen=[p for p in out.rglob('*') if p.is_file() and p.name!='run.log']
    write(out/'freeze.json',dict(status='FROZEN_BEFORE_GT',primary='ambient_contrast/result.json',gt_read=False,
        input_sha256=input_hashes,artifact_sha256={str(p.relative_to(out)):sha(p) for p in frozen},
        command=shlex.join([sys.executable]+sys.argv),seconds=time.monotonic()-started,
        limitations=['每点加性底亮近似，动态点灯与饱和仍可能误标。','投影栅格不是完整BVH/BRDF渲染。','无Stage2重训练或几何优化。']))
    print(json.dumps(dict(status='FROZEN_BEFORE_GT',seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','parent','stats','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
