"""只用fit证据预登记的退火校准：保留v1失败，仅修改优化日程。"""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','fit','model','source','raster-cache','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==v.PYTHON.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    assert a.checkpoint.is_relative_to(a.cache.parent.parent/'raster_calibration_v1')
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.parent.parent,a.fit,a.model,a.source,a.output])
    original=v.read(a.checkpoint.parent/'configuration.json')
    assert not original['gt_read'] and original['geometry_frozen'] and original['directions_frozen']
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert saved['epoch']==60
    parent_hash=v.sha(a.checkpoint)
    cachemeta=v.read(a.raster_cache.parent/'summary.json');assert v.sha(a.raster_cache)==cachemeta['cache_sha256']
    assert cachemeta['gt_read'] is False and cachemeta['status']=='PASS_FORWARD_CACHE_NOT_LIGHT_ACCEPTANCE'
    packed=torch.load(a.raster_cache,map_location='cpu',weights_only=False)
    pc,cameras,pipe,data,state,hashes=f.load(a)
    assert torch.equal(saved['directions'],state['directions']) and torch.equal(packed['directions'],state['directions'])
    q=packed['q'].cuda();constant=bool(saved['constant'])
    from scene.stage2_photometric import srgb_to_linear
    lo=float(srgb_to_linear(torch.tensor(.03)));hi=float(srgb_to_linear(torch.tensor(.97)))
    rho=torch.nn.Parameter(saved['rho'].cuda().clone());loge=torch.nn.Parameter(saved['environment'].cuda().log(),requires_grad=not constant)
    optimizer=torch.optim.Adam([rho] if constant else [rho,loge],lr=.003);optimizer.load_state_dict(saved['optimizer'])
    gen=np.random.default_rng();gen.bit_generator.state=copy.deepcopy(saved['rng'])
    masks=torch.stack([(c.gt_alpha_mask>.5).cpu() for c in cameras]);torch.save(dict(frames=v.TRAIN,foreground=masks),a.output/'fixed_foreground.pt')
    counts=masks.flatten(1).sum(1).double();fit_indices=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
    geometry={name:getattr(pc,name).clone() for name in ('_xyz','_rotation','_scaling','_opacity')}
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__))
    from scene.pbr_rasterizer import native_module
    configuration=dict(gt_read=False,external_test_read=False,geometry_fit_frames=v.FIT,held_frames=v.HELD,geometry_frozen=True,directions_frozen=True,
        initial_ratio=saved['initial_ratio'],constant=constant,epochs=30,updates=3510,cumulative_epochs=90,cumulative_updates=10530,
        parent_checkpoint=str(a.checkpoint),parent_checkpoint_sha256=parent_hash,parent_status='OPTIMIZATION_BUDGET_INSUFFICIENT',
        rho_linear_bounds=[lo,hi],environment_over_pi_bounds=[.01,30],constant_color_bounds=[0,30],
        loss='linear_censored_mse_original_alpha_gt_0p5_fixed_denominator',optimizer='Adam',
        optimizer_schedule=[dict(first=1,last=10,lr=.003),dict(first=11,last=20,lr=.001),dict(first=21,last=30,lr=.0001)],
        seed='继承v1@60的相同RNG状态',foreground_sha256=v.sha(a.output/'fixed_foreground.pt'),backend_provenance=native_module().provenance,
        input_sha256=hashes,selection='四条固定末次30，按fit选初值；不使用held',internal_validation_passed=False)
    v.write(a.output/'configuration.json',configuration);history=[]
    def score():
        with torch.no_grad():
            values=[]
            for i in fit_indices:
                colors=rho if constant else rho*(loge.exp()/math.pi+q[1,i])
                values.append(float(f.objective(f.render(cameras[i],pc,pipe,colors)['render'],cameras[i])))
            return dict(frame_mean=sum(values)/117,pixel_mean=sum(x*float(counts[i]) for x,i in zip(values,fit_indices))/float(counts[fit_indices].sum()),per_frame=values)
    def save(epoch):
        v.write(a.output/f'fit_epoch_{epoch}.json',score())
        torch.save(dict(rho=rho.detach().cpu(),environment=loge.detach().exp().cpu(),epoch=epoch,cumulative_epoch=60+epoch,constant=constant,initial_ratio=saved['initial_ratio'],directions=state['directions'],optimizer=optimizer.state_dict(),rng=gen.bit_generator.state,history=history),a.output/f'checkpoint_epoch_{epoch}.pt')
    save(0)
    for epoch in range(1,31):
        lr=.003 if epoch<=10 else (.001 if epoch<=20 else .0001)
        for group in optimizer.param_groups:group['lr']=lr
        total=0.
        for i in gen.permutation(fit_indices):
            colors=rho if constant else rho*(loge.exp()/math.pi+q[1,i]);loss=f.objective(f.render(cameras[i],pc,pipe,colors)['render'],cameras[i])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():rho.clamp_(0 if constant else lo,30 if constant else hi);loge.clamp_(math.log(math.pi*.01),math.log(math.pi*30))
            total+=float(loss.detach())
        row=dict(epoch=epoch,cumulative_epoch=60+epoch,updates=epoch*117,lr=lr,param_group_lrs=[group['lr'] for group in optimizer.param_groups],stochastic_fit_mse=total/117,ambient_rgb=loge.detach().exp().tolist());history.append(row);print(row,flush=True)
        if epoch in (10,20,25,30):save(epoch)
    assert all(torch.equal(value,getattr(pc,name)) for name,value in geometry.items())
    v.check_hashes(hashes)
    v.write(a.output/'train_epoch_30.json',dict(status='FIT_COMPLETE_HELD_NOT_EVALUATED',gt_read=False,internal_validation_passed=False,epochs=30,cumulative_epochs=90,geometry_bitexact_unchanged=True,history=history))


if __name__=='__main__':main()
