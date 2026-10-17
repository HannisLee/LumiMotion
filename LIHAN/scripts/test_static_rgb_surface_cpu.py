"""新Median三维表面项的CPU独立数值梯度/缺失/原相机权重核验。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import random
import runpy
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'output/1017-01-静态DL未知光总控'
MODULE=ROOT/'scene/static_sparse_rgb_surface.py'
ASSET=C/'1017-07-TRAIN固定Median邻域资产/anchors.json'


def main():
    torch.set_num_threads(1)
    ns=runpy.run_path(str(MODULE));core=ns['pixel_surface_loss']
    indices=torch.tensor([[80200,80201,80600,80601,0,0,0,0],
                          [80202,80203,80602,80603,0,0,0,0]])
    member=torch.zeros((2,8),dtype=torch.bool);member[:,:4]=True
    rays=torch.zeros((2,8,3),dtype=torch.double)
    for i in range(2):
        for j in range(4):
            y,x=divmod(int(indices[i,j]),400)
            rays[i,j]=torch.tensor([(x+.5-200)/500,(y+.5-200)/500,1.],dtype=torch.double)
    packed=dict(indices=indices,member=member,rays=rays,origin=torch.zeros(3,dtype=torch.double),
                targets=torch.tensor([[.003,.003,2.11],[.018,.009,3.05]],dtype=torch.double),
                inverse_views=torch.tensor([1/4,1/7],dtype=torch.double))
    initial=torch.tensor([2.,2.04,2.08,2.12,3.,3.02,3.06,3.08],dtype=torch.double,requires_grad=True)
    def objective(v):
        base=torch.zeros(400*400,dtype=torch.double).scatter(0,indices[:,:4].reshape(-1),v).reshape(400,400)
        return core(base,packed,4.29)[0]
    assert torch.autograd.gradcheck(objective,(initial,),eps=1e-5,atol=2e-7,rtol=2e-5)
    value=objective(initial);gradient=torch.autograd.grad(value,initial)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)==2
    points=initial.detach().reshape(2,4,1)*rays[:,:4]
    manual=torch.sqrt((points-packed['targets'][:,None]).square().sum(-1)+1e-12).min(1).values
    manual=(135/115)*(manual/4.29*packed['inverse_views']).sum()
    assert abs(float(manual-value.detach()))<1e-14
    zeros=torch.zeros((400,400),dtype=torch.double,requires_grad=True)
    missing,diag=core(zeros,packed,4.29)
    assert diag['missing_observations']==2 and abs(float(missing)-(135/115)*(1/4+1/7))<1e-14
    assert torch.count_nonzero(torch.autograd.grad(missing,zeros)[0])==0
    distant=torch.full((400,400),100.,dtype=torch.double,requires_grad=True)
    clipped,diag=core(distant,packed,4.29)
    assert diag['capped_observations']==2 and abs(float(clipped-missing.detach()))<1e-14
    assert torch.count_nonzero(torch.autograd.grad(clipped,distant)[0])==0
    # 使用完整固定资产/135原相机；纯零合成深度，不读取真实模型或缓存。
    py0=random.getstate();np0=np.random.get_state();torch0=torch.get_rng_state().clone()
    surface=ns['FixedSparseRGBSurface'](ASSET,4.290023326873779)
    values=[];missing_total=0
    for name,frame in surface.asset['frames'].items():
        K=frame['K400']
        camera=SimpleNamespace(image_name_train_light=name,image_width=400,image_height=400,
                 world_view_transform=torch.tensor(frame['E'],dtype=torch.float32).T,
                 FoVx=2*math.atan(400/(2*K[0][0])),FoVy=2*math.atan(400/(2*K[1][1])))
        loss,diag=surface(camera,torch.zeros((1,400,400),dtype=torch.double))
        values.append(float(loss));missing_total+=int(diag['missing_observations'])
        if not frame['fit_view']:assert float(loss)==0 and diag['observations']==0
    assert abs(np.mean(values)-1)<1e-13 and missing_total==600
    assert random.getstate()==py0 and all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(np0,np.random.get_state()))
    assert torch.equal(torch0,torch.get_rng_state())
    result=dict(status='PASS_CPU_LOSS_GRADCHECK_AND_ALL135_CAMERA_WEIGHTING',
           module_sha256=hashlib.sha256(MODULE.read_bytes()).hexdigest(),asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
           finite_difference_gradcheck=True,manual_distance_and_weighting_exact=True,
           selected_depth_gradient_nonzero_count=2,all_missing_fixed_penalty_zero_gradient=True,
           distant_clipped_penalty_zero_gradient=True,all135_original_camera_pose_and_intrinsics_exact=True,
           uniform135_missing_penalty_mean=float(np.mean(values)),fixed_missing_observations=missing_total,
           zero_held_anchor=True,Python_NumPy_TorchCPU_RNG_unchanged=True,GT_read=False,GPU_used=False,model_or_actual_cache_read=False)
    out=C/'T14_sparse_surface_CPU_loss_validation_v1.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
