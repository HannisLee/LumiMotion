"""冻结预登记整轨迹±10度；同起点同预算重拟合材质的TRAIN局部方向反证。"""
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
    for key in ('cache','fit','model','source','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    p.add_argument('--case',choices=['center','u_minus','u_plus','v_minus','v_plus'],required=True)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==v.PYTHON.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    # checkpoint是校准无GT产物，核验父配置再读取；不得将任意外部模型传入。
    assert any(a.checkpoint.is_relative_to(a.cache.parent.parent/name) for name in ('raster_calibration_v1','raster_calibration_v2_anneal'))
    meta=v.read(a.checkpoint.parent/'configuration.json')
    assert not meta['gt_read'] and meta['geometry_frozen'] and meta['directions_frozen'] and not meta['constant']
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.parent.parent,a.fit,a.model,a.source,a.output])
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    checkpoint_hash=v.sha(a.checkpoint)
    pc,cameras,pipe,data,state,hashes=f.load(a)
    assert torch.equal(saved['directions'],state['directions'])
    from scene.static_dl_photometric import StaticVisibility
    from scene.stage2_photometric import srgb_to_linear
    from utils.general_utils import build_rotation
    direction=state['directions'].cuda();matrix=torch.eye(3,device='cuda')
    if a.case!='center':
        axis=data['basis'][0 if a.case[0]=='u' else 1].cuda()
        axis=axis/axis.norm();angle=math.radians(-10 if a.case.endswith('minus') else 10)
        x,y,z=axis;zero=axis.new_zeros(())
        skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
        matrix=math.cos(angle)*matrix+(1-math.cos(angle))*axis[:,None]*axis[None]+math.sin(angle)*skew
        direction=direction@matrix.T
    normal=build_rotation(pc.get_rotation)[:,:,2]
    trace=StaticVisibility(bias=float(data['shadow_bias']),update_interval=1000000)
    fit_indices=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
    q={}
    with torch.no_grad():
        for i in fit_indices:
            vis=trace(pc,direction[i:i+1],0)
            facing=torch.where((normal*(cameras[i].camera_center-pc.get_xyz)).sum(-1,keepdim=True)>=0,1.,-1.)
            q[i]=(((normal*facing)@direction[i:i+1].T).clamp_min(0)*vis).detach()
    rho=torch.nn.Parameter(saved['rho'].cuda().clone());loge=torch.nn.Parameter(saved.get('log_e',saved['environment'].log()).cuda().clone())
    optimizer=torch.optim.Adam([rho,loge],lr=.03);optimizer.load_state_dict(saved['optimizer'])
    gen=np.random.default_rng();gen.bit_generator.state=copy.deepcopy(saved['rng'])
    lo=float(srgb_to_linear(torch.tensor(.03)));hi=float(srgb_to_linear(torch.tensor(.97)))
    geometry={name:getattr(pc,name).clone() for name in ('_xyz','_rotation','_scaling','_opacity')}
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__))
    config=dict(gt_read=False,held_used=False,case=a.case,epochs=20,updates=2340,rotation_matrix=matrix.cpu().tolist(),checkpoint=str(a.checkpoint),checkpoint_sha256=checkpoint_hash,
      source_checkpoint_epoch=saved['epoch'],input_sha256=hashes,internal_validation_passed=False,local_profile_only=True)
    v.write(a.output/'configuration.json',config)
    def score():
        with torch.no_grad():
            rows=[]
            for i in fit_indices:
                pred=f.render(cameras[i],pc,pipe,rho*(loge.exp()/math.pi+q[i]))['render']
                rows.append(dict(frame=v.TRAIN[i],mse=float(f.objective(pred,cameras[i]))))
            return dict(frame_mean=sum(r['mse'] for r in rows)/len(rows),per_frame=rows,
                time_blocks=[sum(r['mse'] for r in rows if left<=r['frame']<=right)/sum(left<=r['frame']<=right for r in rows) for left,right in ((1,50),(51,100),(101,150))])
    v.write(a.output/'fit_epoch_0.json',score())
    for epoch in range(1,21):
        total=0.
        for i in gen.permutation(fit_indices):
            pred=f.render(cameras[i],pc,pipe,rho*(loge.exp()/math.pi+q[i]))['render'];loss=f.objective(pred,cameras[i])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():rho.clamp_(lo,hi);loge.clamp_(math.log(math.pi*.01),math.log(math.pi*30))
            total+=float(loss.detach())
        print(dict(case=a.case,epoch=epoch,fit_stochastic=total/117),flush=True)
        if epoch in (10,20):
            v.write(a.output/f'fit_epoch_{epoch}.json',score())
            torch.save(dict(rho=rho.detach().cpu(),environment=loge.detach().exp().cpu(),directions=direction.cpu(),epoch=epoch,optimizer=optimizer.state_dict(),rng=gen.bit_generator.state),a.output/f'checkpoint_epoch_{epoch}.pt')
    assert all(torch.equal(value,getattr(pc,name)) for name,value in geometry.items())
    v.check_hashes(hashes)
    v.write(a.output/'result.json',dict(status='PROFILE_COMPLETE_NOT_LIGHT_ACCEPTANCE',gt_read=False,held_used=False,internal_validation_passed=False,case=a.case,
        fit=v.read(a.output/'fit_epoch_20.json'),ambient_rgb=loge.detach().exp().tolist(),albedo_bound_fraction=float(((rho<=lo+1e-6)|(rho>=hi-1e-6)).float().mean()),geometry_bitexact_unchanged=True))


if __name__=='__main__':main()
