"""同起点同预算比较旧方向与已通过FIT筛查的固定新方向；只更新材质和EL。"""
from __future__ import annotations
import argparse
import copy
import math
import socket
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import calibrate_eldl_v3_raster as f


def install_registered_guard(a, extra_files):
    """所有叠加护栏登记相同的精确额外文件，不开放整个L00目录。"""
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.parent,a.fit,a.model,a.source,a.output,*extra_files])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','fit','model','source','refinement','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    p.add_argument('--case',choices=['old','refined'],required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==v.PYTHON.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    assert a.checkpoint == a.cache.parent.parent/'raster_calibration_v2_anneal/init_3/checkpoint_epoch_30.pt'
    assert a.refinement == a.cache.parent.parent/'raster_direction_refinement_v2/result'
    registered_inputs=[a.checkpoint,a.checkpoint.parent/'configuration.json',
        a.checkpoint.parent/'fit_epoch_30.json',a.checkpoint.parent/'fixed_foreground.pt',
        a.refinement/'summary.json',a.refinement/'configuration.json',a.refinement/'direction_state.pt']
    install_registered_guard(a,registered_inputs)
    original=v.read(a.checkpoint.parent/'configuration.json')
    assert not original['gt_read'] and original['geometry_frozen'] and original['directions_frozen']
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert saved['epoch']==30 and saved['cumulative_epoch']==90 and saved['constant'] is False
    parent_hash=v.sha(a.checkpoint)
    parent_score=v.read(a.checkpoint.parent/'fit_epoch_30.json')
    screen=v.read(a.refinement/'summary.json')
    refinement_config=v.read(a.refinement/'configuration.json')
    assert screen['status']=='PASS_FIT_SCREEN_ONLY' and screen['eligible_for_equal_budget_calibration']
    assert screen['relative_fit_improvement']>=.01 and screen['held_scored'] is False
    assert screen['gt_read'] is False and screen['training_asset_exported'] is False
    assert screen['held_used_for_selection'] is False
    assert screen['geometry_bitexact_unchanged'] and screen['material_environment_unchanged']
    assert len(screen['history'])==6 and screen['history'][-1]['round']==6
    assert [row['frame'] for row in screen['history'][-1]['per_frame']]==v.FIT
    assert v.sha(a.refinement/'direction_state.pt')==screen['direction_state_sha256']
    assert v.sha(a.refinement/'configuration.json')==screen['configuration_sha256']
    assert refinement_config['source_checkpoint_sha256']==parent_hash
    refined=torch.load(a.refinement/'direction_state.pt',map_location='cpu',weights_only=False)
    assert refined['frames']==v.TRAIN and refined['fit_frames']==v.FIT and refined['gt_read'] is False
    assert refined['held_rgb_used'] is False
    assert refined['source_checkpoint_sha256']==parent_hash
    extra_files=[*registered_inputs,Path(f.__file__).resolve(),ROOT/'scene/eldl_v3_photometric.py',
        ROOT/'gaussian_renderer/__init__.py',ROOT/'scene/stage2_photometric.py']
    bindings={str(path):v.sha(path) for path in extra_files}
    original_install_guard=v.install_guard
    v.install_guard=lambda args:install_registered_guard(args,registered_inputs)
    try:
        pc,cameras,pipe,data,state,hashes=f.load(a)
    finally:
        v.install_guard=original_install_guard
    assert torch.equal(saved['directions'],state['directions'])
    shared=set(hashes).intersection(original['input_sha256'])
    assert shared and all(hashes[path]==original['input_sha256'][path] for path in shared)
    hashes.update(bindings)
    v.check_hashes(hashes)  # 首次BVH/渲染前验证叠加护栏允许全部登记输入重读。
    fit_indices=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
    assert torch.equal(refined['directions'][fit_indices],refined['fit_directions'])
    interpolated=v.interpolate_frozen(torch.tensor(v.TRAIN,dtype=torch.float64),
        torch.tensor(v.FIT,dtype=torch.float64),refined['fit_directions'].double()).float()
    interpolated[fit_indices]=refined['fit_directions']
    assert torch.equal(interpolated,refined['directions'])
    direction=(saved['directions'] if a.case=='old' else refined['directions']).cuda()
    assert direction.shape==(135,3) and torch.isfinite(direction).all()
    assert torch.allclose(direction.norm(dim=-1),torch.ones(135,device='cuda'),atol=2e-6,rtol=0)
    from scene.static_dl_photometric import StaticVisibility
    from utils.general_utils import build_rotation
    normal=build_rotation(pc.get_rotation)[:,:,2]
    trace=StaticVisibility(bias=float(data['shadow_bias']),update_interval=1000000)
    fit_indices=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
    q={}
    with torch.no_grad():
        for i in fit_indices:
            visibility=trace(pc,direction[i:i+1],0)
            facing=torch.where((normal*(cameras[i].camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.)
            q[i]=(((normal*facing)@direction[i:i+1].T).clamp_min(0)*visibility).detach()
    constant=False
    from scene.stage2_photometric import srgb_to_linear
    lo=float(srgb_to_linear(torch.tensor(.03)));hi=float(srgb_to_linear(torch.tensor(.97)))
    rho=torch.nn.Parameter(saved['rho'].cuda().clone());loge=torch.nn.Parameter(saved['log_e'].cuda().clone(),requires_grad=True)
    optimizer=torch.optim.Adam([rho] if constant else [rho,loge],lr=.003);optimizer.load_state_dict(saved['optimizer'])
    gen=np.random.default_rng();gen.bit_generator.state=copy.deepcopy(saved['rng'])
    masks=torch.stack([(c.gt_alpha_mask>.5).cpu() for c in cameras])
    old_masks=torch.load(a.checkpoint.parent/'fixed_foreground.pt',map_location='cpu',weights_only=False)
    assert old_masks['frames']==v.TRAIN and torch.equal(masks,old_masks['foreground'])
    assert v.sha(a.checkpoint.parent/'fixed_foreground.pt')==original['foreground_sha256']
    torch.save(dict(frames=v.TRAIN,foreground=masks),a.output/'fixed_foreground.pt')
    counts=masks.flatten(1).sum(1).double();fit_indices=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
    geometry={name:getattr(pc,name).clone() for name in ('_xyz','_rotation','_scaling','_opacity')}
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__))
    from scene.pbr_rasterizer import native_module
    configuration=dict(gt_read=False,external_test_read=False,geometry_fit_frames=v.FIT,held_frames=v.HELD,geometry_frozen=True,directions_frozen=True,
        initial_ratio=saved['initial_ratio'],constant=constant,epochs=30,updates=3510,cumulative_epochs=120,cumulative_updates=14040,case=a.case,held_scored=False,held_used_for_selection=False,train_containers_loaded=135,development_containers_loaded=18,
        parent_checkpoint=str(a.checkpoint),parent_checkpoint_sha256=parent_hash,parent_status='FIT_BUDGET_PASS',refinement=str(a.refinement),refinement_summary_sha256=v.sha(a.refinement/'summary.json'),
        rho_linear_bounds=[lo,hi],environment_over_pi_bounds=[.01,30],constant_color_bounds=[0,30],
        loss='linear_censored_mse_original_alpha_gt_0p5_fixed_denominator',optimizer='Adam',
        optimizer_schedule=[dict(first=1,last=10,lr=.003),dict(first=11,last=20,lr=.001),dict(first=21,last=30,lr=.0001)],
        seed='两组继承同一v2@30/累计90 checkpoint及Adam/RNG' ,foreground_sha256=v.sha(a.output/'fixed_foreground.pt'),backend_provenance=native_module().provenance,
        input_sha256=hashes,selection='两组固定末次30；均25→30变化<=1%且refined优于old至少1%；不使用开发帧评分' ,internal_validation_passed=False)
    v.write(a.output/'configuration.json',configuration);history=[]
    def score():
        with torch.no_grad():
            values=[]
            for i in fit_indices:
                colors=rho if constant else rho*(loge.exp()/math.pi+q[i])
                values.append(float(f.objective(f.render(cameras[i],pc,pipe,colors)['render'],cameras[i])))
            return dict(frame_mean=sum(values)/117,pixel_mean=sum(x*float(counts[i]) for x,i in zip(values,fit_indices))/float(counts[fit_indices].sum()),per_frame=values)
    def save(epoch):
        v.write(a.output/f'fit_epoch_{epoch}.json',score())
        torch.save(dict(rho=rho.detach().cpu(),environment=loge.detach().exp().cpu(),log_e=loge.detach().cpu(),epoch=epoch,cumulative_epoch=90+epoch,constant=constant,initial_ratio=saved['initial_ratio'],directions=direction.cpu(),optimizer=optimizer.state_dict(),rng=gen.bit_generator.state,history=history),a.output/f'checkpoint_epoch_{epoch}.pt')
    save(0)
    restart_score=v.read(a.output/'fit_epoch_0.json')
    expected=parent_score['per_frame'] if a.case=='old' else [row['mse'] for row in screen['history'][-1]['per_frame']]
    assert len(expected)==len(restart_score['per_frame'])==len(v.FIT)
    assert np.isfinite(expected).all()
    delta=max(abs(x-y) for x,y in zip(expected,restart_score['per_frame']))
    assert delta<1e-7,delta
    v.write(a.output/'restart_parity.json',dict(status='PASS_FIT_FORWARD_RESTART',max_per_frame_mse_difference=delta,environment_log_roundtrip=True,geometry_unchanged=True))
    for epoch in range(1,31):
        lr=.003 if epoch<=10 else (.001 if epoch<=20 else .0001)
        for group in optimizer.param_groups:group['lr']=lr
        total=0.
        for i in gen.permutation(fit_indices):
            colors=rho if constant else rho*(loge.exp()/math.pi+q[i]);loss=f.objective(f.render(cameras[i],pc,pipe,colors)['render'],cameras[i])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():rho.clamp_(0 if constant else lo,30 if constant else hi);loge.clamp_(math.log(math.pi*.01),math.log(math.pi*30))
            total+=float(loss.detach())
        row=dict(epoch=epoch,cumulative_epoch=90+epoch,updates=epoch*117,lr=lr,param_group_lrs=[group['lr'] for group in optimizer.param_groups],stochastic_fit_mse=total/117,ambient_rgb=loge.detach().exp().tolist());history.append(row);print(row,flush=True)
        if epoch in (10,20,25,30):save(epoch)
    assert all(torch.equal(value,getattr(pc,name)) for name,value in geometry.items())
    v.check_hashes(hashes)
    assert torch.equal(direction.cpu(),saved['directions'] if a.case=='old' else refined['directions'])
    v.write(a.output/'train_epoch_30.json',dict(status='FIT_COMPLETE_HELD_NOT_EVALUATED',gt_read=False,internal_validation_passed=False,epochs=30,cumulative_epochs=120,case=a.case,held_scored=False,geometry_bitexact_unchanged=True,directions_bitexact_unchanged=bool(torch.equal(direction.cpu(),saved['directions'] if a.case=='old' else refined['directions'])),history=history))


if __name__=='__main__':main()
