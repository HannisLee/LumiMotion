"""新TRAIN阴影与完整当前几何联合估计平滑方向光；无旧灯向或逐帧曝光。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras,sh2_basis
from LIHAN.scripts.dynamic_v3_tracks import build_observations
from utils.no_gt_guard import install_no_gt_guard


def fibonacci_directions(count,device):
    index=torch.arange(count,device=device,dtype=torch.float32)
    z=1-2*(index+.5)/count;phi=index*math.pi*(3-math.sqrt(5))
    radius=(1-z*z).sqrt()
    return torch.stack([radius*phi.cos(),radius*phi.sin(),z],-1)


def censored_frame_loss(prediction,target,saturated,mask):
    residual=prediction-target
    residual=torch.where(saturated,residual.clamp_max(0),residual)
    robust=(residual.square()+.02**2).sqrt()-.02
    return (robust*mask[...,None]).sum((-1,-2))/(3*mask.sum(-1)).clamp_min(1)


def smooth_path(cost,candidates,weight=.01):
    """全序列动态规划；不以相机运动构造光照方向。"""
    penalty=weight*(candidates[:,None]-candidates[None]).square().sum(-1)
    previous=cost[0];back=[]
    for row in cost[1:]:
        best,index=(previous[:,None]+penalty).min(0)
        previous=row+best;back.append(index)
    indices=[int(previous.argmin())]
    for table in reversed(back):indices.append(int(table[indices[-1]]))
    return torch.tensor(list(reversed(indices)),device=cost.device)


def run(args):
    out=args.output.resolve()
    root=Path(os.environ.get('DYNAMIC_V3_REPO_ROOT',Path(__file__).resolve().parents[2])).resolve()
    assert out.is_relative_to(root/'output'),'估光输出必须在本仓库output内'
    out.mkdir(parents=True,exist_ok=False)
    source,parent=args.source.resolve(),args.parent.resolve()
    manifest=json.loads((source/'no_gt_manifest.json').read_text())
    assert manifest['dataset']=='LH-dynamic-jumpingjacks-ELDL-V3' and not manifest['gt_read']
    assert args.smoke==('smoke_test' in out.parts)
    assert args.iteration==35000 or args.smoke
    assert not set(manifest['train_frames'])&set(manifest['dev_frames']+manifest['test_frames'])
    install_no_gt_guard([source,parent,out,args.stats.resolve()])
    torch.set_num_threads(4);torch.manual_seed(1019)
    stats=json.loads(args.stats.read_text())
    cameras=load_train_cameras(source)
    if args.smoke:cameras=[c for c in cameras if int(c.image_name_train_light.split('_')[-1]) in (1,21,41,81,101,121,141,149)]
    times=torch.tensor([float(c.fid) for c in cameras])
    initial=dict(times=times,directions=torch.tensor([[0.,0.,1.]]).repeat(len(times),1),intensity=math.pi)
    model=DynamicSurfaceModel(parent,args.iteration,initial,stats,'A',allow_smoke_parent=args.smoke)
    started=time.monotonic();orientation=model.initialize_orientation(cameras)
    observations=build_observations(model,cameras,out/'observations',maximum=128 if args.smoke else 2048,transport=True)
    device=model.pc.get_xyz.device;ids=observations['ids'].to(device)
    candidate_count=32 if args.smoke else 256
    candidates=fibonacci_directions(candidate_count,device)
    cache=[]
    with torch.no_grad():
        for index,camera in enumerate(cameras):
            camera.load2device();geometry=model.effective(camera);model.visibility.build(geometry)
            parts=[]
            for start in range(0,candidate_count,16):
                parts.append(model.visibility.trace(geometry.get_xyz[ids],candidates[start:start+16]).T.cpu())
            cache.append(torch.cat(parts))
            print(json.dumps(dict(visibility_frame=observations['frames'][index]['frame'],candidate_count=candidate_count)),flush=True)
            camera.load2device('cpu')
    visibility=torch.stack(cache).to(device)
    normals=observations['world_normals'].to(device)
    cosine=torch.einsum('tni,ki->tkn',normals,candidates).clamp_min(0)
    shading=visibility*cosine
    target=observations['observed'].to(device);saturated=observations['saturated'].to(device)
    valid=observations['valid'].to(device).float()
    count=len(ids);raw_rho=torch.nn.Parameter(torch.zeros(count,3,device=device))
    intensity_raw=torch.nn.Parameter(torch.tensor(math.log((math.pi-.1)/(20-math.pi)),device=device))
    coefficients=torch.nn.Parameter(torch.zeros(9,3,device=device))
    env_scale=float(model.radiance.environment_scale)
    basis=sh2_basis(normals)
    indices=torch.argmax((shading*valid[:,None]).sum(-1),dim=-1)
    history=[]
    def appearances():
        rho=.03+.94*(raw_rho-raw_rho.median().detach()).sigmoid()
        intensity=.1+19.9*intensity_raw.sigmoid()
        environment=env_scale*F.softplus(basis@(.5*coefficients.tanh())+math.log(math.expm1(1.)))
        return rho,intensity,environment
    def optimize_appearance(selected,steps,extra=None):
        parameters=[raw_rho,intensity_raw,coefficients]+([] if extra is None else [extra])
        optimizer=torch.optim.Adam(parameters,lr=.01)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            rho,intensity,environment=appearances()
            direct=selected() if callable(selected) else selected
            prediction=rho[None]/math.pi*(environment+intensity*direct[...,None])
            photo=censored_frame_loss(prediction,target,saturated,valid).mean()
            regular=.0001*(raw_rho.square().mean()+coefficients.square().mean())
            if extra is not None:
                directions=F.normalize(extra,dim=-1)
                regular=regular+.01*(directions[1:]-directions[:-1]).square().sum(-1).mean()
            loss=photo+regular
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
        return float(photo.detach())
    rounds=2 if args.smoke else 4;appearance_steps=20 if args.smoke else 200
    for block in range(rounds):
        selected=shading[torch.arange(len(cameras),device=device),indices]
        photo=optimize_appearance(selected,appearance_steps)
        with torch.no_grad():
            rho,intensity,environment=appearances();cost=[]
            for t in range(len(cameras)):
                prediction=rho[None]/math.pi*(environment[t][None]+intensity*shading[t,:,:,None])
                cost.append(censored_frame_loss(prediction,target[t][None],saturated[t][None],valid[t][None]))
            cost=torch.stack(cost);indices=smooth_path(cost,candidates)
        history.append(dict(block=block+1,full_train_censored=photo,selected_indices=indices.tolist()))
        print(json.dumps(history[-1]),flush=True)
    coarse=candidates[indices].detach()
    direction_raw=torch.nn.Parameter(coarse.clone())
    selected_visibility=visibility[torch.arange(len(cameras),device=device),indices].detach()
    def continuous():
        directions=F.normalize(direction_raw,dim=-1)
        return selected_visibility*(normals*directions[:,None]).sum(-1).clamp_min(0)
    approximate_loss=optimize_appearance(continuous,20 if args.smoke else 200,direction_raw)
    with torch.no_grad():
        estimated=F.normalize(direction_raw,dim=-1).detach();actual=[]
        for t,camera in enumerate(cameras):
            camera.load2device();geometry=model.effective(camera);model.visibility.build(geometry)
            actual.append(model.visibility.trace(geometry.get_xyz[ids],estimated[t:t+1])[:,0].cpu())
            camera.load2device('cpu')
        actual_visibility=torch.stack(actual).to(device)
        actual_shading=actual_visibility*(normals*estimated[:,None]).sum(-1).clamp_min(0)
    actual_loss=optimize_appearance(actual_shading,10 if args.smoke else 100)
    with torch.no_grad():
        rho,intensity,environment=appearances()
        without_direct=rho[None]/math.pi*environment
        baseline_loss=float(censored_frame_loss(without_direct,target,saturated,valid).mean())
        direction_margin=[]
        for row in cost:
            values=row.sort().values
            direction_margin.append(float(values[1]-values[0]))
    assets=dict(format_version=1,status='TRAIN_ESTIMATED_NOT_GT',source_split='TRAIN',gt_read=False,
        source=str(source),parent=str(parent),parent_iteration=args.iteration,
        train_frames=[r['frame'] for r in observations['frames']],times=times.tolist(),
        directions=estimated.cpu().tolist(),intensity=float(intensity.detach()),
        world_fixed=False,global_intensity=True,per_frame_exposure=False,
        candidate_count=candidate_count,appearance_rounds=rounds,appearance_steps=appearance_steps,
        continuous_steps=20 if args.smoke else 200,actual_visibility_refit_steps=10 if args.smoke else 100,
        visibility='完整当前G_t每帧重建；候选全BVH，连续细化固定V近似，终点重算真实V并固定方向重拟合',
        environment_role='仅共同灯向估计的受限共享2阶SH辅助；B正式训练不得读取此环境或扣除它',
        initial_direction_precision_claim=False,smoke=args.smoke,
        orientation=orientation,history=history,approximate_photo=approximate_loss,actual_photo=actual_loss,
        without_direction_photo=baseline_loss,candidate_second_best_margin=direction_margin,
        valid_observations_per_frame=observations['valid'].sum(-1).tolist(),elapsed_seconds=time.monotonic()-started,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'no_gt_manifest.json',source.parent/'input_audit_v1/radiometry_v1.json',parent/f'point_cloud/iteration_{args.iteration}/point_cloud.ply',
             parent/f'deform/iteration_{args.iteration}/deform.pth',args.stats,Path(__file__)]})
    (out/'lighting.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    torch.save(dict(rho=rho.detach().cpu(),environment_coefficients=coefficients.detach().cpu(),
        ids=ids.cpu(),candidate_directions=candidates.cpu(),candidate_visibility=visibility.cpu(),
        final_actual_visibility=actual_visibility.cpu(),initial_indices=indices.cpu()),out/'lighting_diagnostics.pt')
    print(json.dumps(dict(status=assets['status'],intensity=assets['intensity'],actual_photo=actual_loss,
                         without_direction_photo=baseline_loss),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--iteration',type=int,default=35000);parser.add_argument('--stats',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args)


if __name__=='__main__':main()
