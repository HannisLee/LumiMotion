"""生产全图固定几何/方向校准，训练集拟合和内部held一次冻结验收。"""
from __future__ import annotations
import argparse
import json
import math
import socket
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts import calibrate_eldl_v3_raster as forward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','fit','model','source','raster-cache','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=16000)
    p.add_argument('--initial-ratio',type=float,choices=[1/3,1.,3.],default=1.)
    p.add_argument('--constant',action='store_true')
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--resume',type=int,default=0)
    p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if socket.gethostname()!='ushas' or Path(sys.executable).resolve()!=v.PYTHON.resolve():raise RuntimeError('需要ushas环境')
    a.output.mkdir(parents=True,exist_ok=bool(a.resume or a.evaluate_only))
    # forward.load的白名单含cache.parent；raster cache必须在同L00目录。
    assert a.raster_cache.is_relative_to(a.cache.parent.parent)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.cache.parent.parent,a.fit,a.model,a.source,a.output])
    # load将再次装更窄白名单，因此先载入显式核验的同批全图缓存。
    meta=v.read(a.raster_cache.parent/'summary.json')
    assert meta['status']=='PASS_FORWARD_CACHE_NOT_LIGHT_ACCEPTANCE' and not meta['gt_read']
    assert v.sha(a.raster_cache)==meta['cache_sha256']
    packed=torch.load(a.raster_cache,map_location='cpu',weights_only=False)
    pc,cameras,pipe,data,state,hashes=forward.load(a)
    assert packed['frames']==v.TRAIN and torch.equal(packed['directions'],state['directions'])
    q=packed['q'].cuda();assert q.shape==(3,135,len(pc.get_xyz),1)
    from scene.stage2_photometric import srgb_to_linear,linear_to_srgb
    lo=float(srgb_to_linear(torch.tensor(.03)));hi=float(srgb_to_linear(torch.tensor(.97)))
    # 所有初值同一材质：线性0.5，不用held、GT或旧稀疏rho初始化。
    rho=torch.nn.Parameter(torch.full_like(pc.get_xyz,.5))
    log_e=torch.nn.Parameter(torch.full((3,),math.log(math.pi*a.initial_ratio),device='cuda'),requires_grad=not a.constant)
    optimizer=torch.optim.Adam([rho] if a.constant else [rho,log_e],lr=.03)
    fit_indices=[i for i,n in enumerate(v.TRAIN) if n in v.FIT]
    gen=np.random.default_rng(1019);history=[]
    masks=torch.stack([(c.gt_alpha_mask>.5).cpu() for c in cameras])
    counts=masks.flatten(1).sum(1).double()
    if not a.resume and not a.evaluate_only:
        torch.save(dict(frames=v.TRAIN,foreground=masks),a.output/'fixed_foreground.pt')
    from scene.pbr_rasterizer import native_module
    backend_provenance=native_module().provenance
    fit_mean=float(counts[fit_indices].mean())
    start=0
    if a.resume or a.evaluate_only:
        saved=torch.load(a.output/f'checkpoint_epoch_{a.resume or a.epochs}.pt',map_location='cuda',weights_only=False)
        assert saved['constant']==a.constant and saved['initial_ratio']==a.initial_ratio
        assert torch.equal(saved['directions'].cpu(),state['directions'])
        rho.data.copy_(saved['rho']);log_e.data.copy_(saved['environment'].log())
        optimizer.load_state_dict(saved['optimizer']);gen.bit_generator.state=saved['rng']
        start=saved['epoch'];history=saved['history']
    hashes[str(Path(__file__).resolve())]=v.sha(Path(__file__))
    hashes[str(ROOT/'scripts/calibrate_eldl_v3_raster.py')]=v.sha(ROOT/'scripts/calibrate_eldl_v3_raster.py')
    hashes[str(a.raster_cache)]=meta['cache_sha256']
    # 固定几何校验只读摘要；真实颜色优化不改变几何参数。
    geometry={name:value.detach().clone() for name,value in vars(pc).items() if isinstance(value,torch.Tensor) and name in ('_xyz','_rotation','_scaling','_opacity')}
    configuration=dict(gt_read=False,external_test_read=False,geometry_fit_frames=v.FIT,held_frames=v.HELD,
        geometry_frozen=True,directions_frozen=True,initial_ratio=a.initial_ratio,constant=a.constant,epochs=a.epochs,updates=a.epochs*117,
        rho_initial=.5,rho_linear_bounds=[lo,hi],environment_over_pi_bounds=[.01,30],constant_color_bounds=[0,30],
        loss='linear_censored_mse_original_alpha_gt_0p5_fixed_denominator',foreground_sha256=v.sha(a.output/'fixed_foreground.pt'),backend_provenance=backend_provenance,optimizer='Adam',lr=.03,seed=1019,input_sha256=hashes,
        selection='固定末次；不得按held选迭代',internal_validation_passed=False)
    if start==0:v.write(a.output/'configuration.json',configuration)
    else:
        original=v.read(a.output/'configuration.json')
        assert original['initial_ratio']==a.initial_ratio and original['constant']==a.constant
        assert original['input_sha256']==hashes
    def complete_fit():
        with torch.no_grad():
            values=[]
            for index in fit_indices:
                colors=rho if a.constant else rho*(log_e.exp()/math.pi+q[1,index])
                values.append(float(forward.objective(forward.render(cameras[index],pc,pipe,colors)['render'],cameras[index])))
            return dict(frame_mean=sum(values)/117,pixel_mean=sum(x*float(counts[i]) for x,i in zip(values,fit_indices))/float(counts[fit_indices].sum()),per_frame=values)
    if start==0:v.write(a.output/'fit_epoch_0.json',complete_fit())
    for epoch in range(start+1,a.epochs+1) if not a.evaluate_only else []:
        train_loss=0.
        for index in gen.permutation(fit_indices):
            colors=rho if a.constant else rho*(log_e.exp()/math.pi+q[1,index])
            package=forward.render(cameras[index],pc,pipe,colors)
            loss=forward.objective(package['render'],cameras[index])
            if not torch.isfinite(loss):raise RuntimeError('非有限loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():
                rho.clamp_(0 if a.constant else lo,30 if a.constant else hi)
                log_e.clamp_(math.log(math.pi*.01),math.log(math.pi*30))
            train_loss+=float(loss.detach())
        record=dict(epoch=epoch,updates=epoch*117,stochastic_fit_mse=train_loss/117,ambient_rgb=log_e.detach().exp().tolist())
        history.append(record);print(json.dumps(record),flush=True)
        if epoch%10==0 or epoch==a.epochs:
            v.write(a.output/f'fit_epoch_{epoch}.json',complete_fit())
            torch.save(dict(rho=rho.detach().cpu(),environment=log_e.detach().exp().cpu(),epoch=epoch,constant=a.constant,initial_ratio=a.initial_ratio,directions=state['directions'],optimizer=optimizer.state_dict(),rng=gen.bit_generator.state,history=history),a.output/f'checkpoint_epoch_{epoch}.pt')
    if not a.evaluate_only:
        assert all(torch.equal(value,getattr(pc,name)) for name,value in geometry.items())
        v.check_hashes({k:x for k,x in hashes.items() if k!=str(a.raster_cache)})
        v.write(a.output/f'train_epoch_{a.epochs}.json',dict(status='FIT_COMPLETE_HELD_NOT_EVALUATED',gt_read=False,internal_validation_passed=False,epochs=a.epochs,geometry_bitexact_unchanged=True,history=history))
        return
    metrics={};visuals=[]
    import cv2
    def png(path,x):
        arr=x.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
        if arr.shape[-1]==1:arr=np.repeat(arr,3,-1)
        cv2.imwrite(str(path),(arr[...,::-1]*255+.5).astype(np.uint8))
    with torch.no_grad():
        for si,scale in enumerate((.5,1.,2.)):
            if a.constant and scale!=1.:continue
            records=[]
            for i,camera in enumerate(cameras):
                colors=rho if a.constant else rho*(log_e.exp()/math.pi+q[si,i])
                package=forward.render(camera,pc,pipe,colors);value=float(forward.objective(package['render'],camera))
                records.append(dict(frame=v.TRAIN[i],split='fit' if v.TRAIN[i] in v.FIT else 'held',mse=value,pixels=int(counts[i])))
                if v.TRAIN[i] in (1,81,149,61,71,79):
                    directory=a.output/f'visuals/bias_{v.scale_key(scale)}/frame_{v.TRAIN[i]:04d}';directory.mkdir(parents=True)
                    pred=linear_to_srgb(package['render'].clamp_min(0));observed=camera.original_image_train_light.cuda()
                    png(directory/'RGB.png',torch.cat([observed,pred,(pred-observed).abs()*3],dim=2))
                    png(directory/'Alpha.png',torch.cat([camera.gt_alpha_mask.cuda(),package['rend_alpha']],dim=2))
                    normal=package['rend_normal'];normal=torch.nn.functional.normalize(normal,dim=0)
                    png(directory/'Normal.png',(normal+1)/2*package['rend_alpha'])
                    if a.constant:
                        png(directory/'Separation.png',torch.cat([pred,torch.zeros_like(pred)],dim=2))
                    else:
                        env=forward.render(camera,pc,pipe,rho*log_e.exp()/math.pi)['render']
                        direct=forward.render(camera,pc,pipe,rho*q[si,i])['render']
                        png(directory/'Separation.png',torch.cat([linear_to_srgb(env.clamp_min(0)),linear_to_srgb(direct.clamp_min(0))],dim=2))
                    visuals.append(str(directory))
            bysplit={}
            for split in ('fit','held'):
                rows=[r for r in records if r['split']==split]
                bysplit[split]=dict(frame_mean=sum(r['mse'] for r in rows)/len(rows),pixel_mean=sum(r['mse']*r['pixels'] for r in rows)/sum(r['pixels'] for r in rows),frames=len(rows),pixels=sum(r['pixels'] for r in rows))
            metrics[v.scale_key(scale)]=dict(**bysplit,per_frame=records)
        assert all(torch.equal(value,getattr(pc,name)) for name,value in geometry.items())
    # load_inputs历史源码核验与当前源码；外部光栅数据缓存另已绑定SHA。
    v.check_hashes({k:x for k,x in hashes.items() if k!=str(a.raster_cache)})
    v.write(a.output/'result.json',dict(status='FITTED_PENDING_CROSS_START_AND_MANUAL_VALIDATION',gt_read=False,external_test_read=False,internal_validation_passed=False,
        constant=a.constant,initial_ratio=a.initial_ratio,epochs=a.epochs,metrics=metrics,history=history,ambient_rgb=log_e.detach().exp().tolist(),
        environment_bound_active=bool(((log_e.exp()/math.pi<.0101)|(log_e.exp()/math.pi>29.9)).any()),
        albedo_bound_fraction=float(((rho<=lo+1e-6)|(rho>=hi-1e-6)).float().mean()) if not a.constant else None,
        directions_independently_refit=False,geometry_bitexact_unchanged=True,visuals=visuals,input_sha256=hashes))


if __name__=='__main__':main()
