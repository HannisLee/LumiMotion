"""独立PS已知控制：刚性、非刚性、退化方向、阴影与饱和不放大。"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from LIHAN.scripts.dynamic_v3_ps_math import estimate_normals,transport_normal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT/'output/smoke_test')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(1019)
    t,n=32,8
    phi=torch.linspace(0,2*math.pi,t+1)[:-1]
    directions=F.normalize(torch.stack([.8*phi.cos(),.8*phi.sin(),torch.ones(t)],-1),dim=-1)
    target=F.normalize(torch.tensor([[.2,-.1,1.]])+torch.randn(n,3)*.05,dim=-1)
    material=.2+.5*torch.rand(n,3)
    visibility=.3+.7*torch.rand(t,n)
    reports={}
    for mode in ('rigid','affine'):
        maps=torch.eye(3)[None,None].repeat(t,n,1,1)
        if mode=='affine':
            maps[:,:,0,0]=.7+.6*torch.sin(phi)[:,None].square()
            maps[:,:,1,1]=.8+.5*torch.cos(phi)[:,None].square()
            maps[:,:,0,2]=.15*torch.sin(phi)[:,None]
        world=transport_normal(target,maps)
        observed=material[None]*visibility[...,None]*(world*directions[:,None]).sum(-1).clamp_min(0)[...,None]
        # 近零阴影与满码观测删失，不应通过除以V制造巨大伪观测。
        visibility[0,:]=0
        observed[0,:]=0
        saturated=torch.zeros_like(observed,dtype=torch.bool);saturated[1,:,:]=True
        result=estimate_normals(observed,directions,maps,visibility,torch.ones(t,n,dtype=torch.bool),saturated,
                                math.pi,steps=500)
        error=torch.rad2deg(torch.acos((result['normal']*target).sum(-1).clamp(-1,1)))
        assert result['reliable'].all() and float(error.max())<2.,(mode,error.tolist())
        assert not result['normal'].requires_grad and not result['confidence'].requires_grad
        reports[mode]=dict(max_angle_deg=float(error.max()),mean_angle_deg=float(error.mean()),
                           minimum_singular_ratio=float(result['effective_singular_ratio'].min()),
                           relative_error_max=float(result['relative_photometric_error'].max()),
                           valid_observations=result['valid_observations'].tolist(),targets_detached=True)
        torch.save(result,out/f'{mode}_result.pt')
    flat_dirs=torch.tensor([[0.,0.,1.]]).repeat(t,1)
    flat_maps=torch.eye(3)[None,None].repeat(t,n,1,1)
    observed=material[None].expand(t,-1,-1).clone()
    degenerate=estimate_normals(observed,flat_dirs,flat_maps,torch.ones(t,n),
        torch.ones(t,n,dtype=torch.bool),torch.zeros_like(observed,dtype=torch.bool),math.pi,steps=20)
    assert not degenerate['reliable'].any() and not degenerate['confidence'].any()
    reports['degenerate_directions']=dict(all_rejected=True,maximum_confidence=0)
    empty=estimate_normals(observed,directions,flat_maps,torch.zeros(t,n),
        torch.ones(t,n,dtype=torch.bool),torch.zeros_like(observed,dtype=torch.bool),math.pi,steps=5)
    assert not empty['reliable'].any() and torch.isfinite(empty['normal']).all()
    reports['all_shadow']=dict(all_rejected=True,finite=True)
    result=dict(status='PASS_SYNTHETIC_PS_MATH' ,gt_scene_read=False,reports=reports,
                scope='独立已知数值控制；不证明真实对应、动态遮挡、GS梯度或PS收益')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
