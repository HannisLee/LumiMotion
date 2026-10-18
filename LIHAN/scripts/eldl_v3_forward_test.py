"""独立混合前向、颜色和有限差分验证，不能替代真实光栅几何梯度验收。"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    path=Path(__file__).resolve().parents[2]/'scene/eldl_v3_photometric.py'
    spec=importlib.util.spec_from_file_location('eldl_forward',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    dtype=torch.float64
    albedo=torch.tensor([[.2,.4,.7],[.5,.6,.8]],dtype=dtype)
    normal=torch.tensor([[0.,0.,1.],[0.,0.,-1.]],dtype=dtype)
    light=torch.tensor([[0.,0.,1.]],dtype=dtype)
    direct=torch.tensor([math.pi],dtype=dtype);ambient=torch.tensor([.2,.3,.5],dtype=dtype)
    visible=torch.ones(2,1,dtype=dtype);hidden=torch.zeros_like(visible)
    e,d,c=m.diffuse_components(albedo,normal,light,direct,ambient,visible)
    eh,dh,ch=m.diffuse_components(albedo,normal,light,direct,ambient,hidden)
    checks=dict(environment_independent_of_dl_visibility=bool(torch.equal(e,eh)),
                dl_hidden_is_zero=bool(torch.count_nonzero(dh)==0),
                backface_keeps_environment=bool(torch.equal(c[1],e[1]) and (e[1]>0).all()),
                white_unit_lambertian=bool(torch.allclose(d[0],albedo[0],atol=1e-12)),
                component_additivity=bool(torch.equal(c,e+d)))
    theta=torch.tensor(.4,dtype=dtype,requires_grad=True)
    def f(t):
        n=torch.stack([t.sin(),t*0,t.cos()]).reshape(1,3)
        return m.diffuse_components(albedo[:1],n,light,direct,ambient,visible[:1])[2].sum()
    grad=torch.autograd.grad(f(theta),theta)[0];step=1e-6;fd=(f(theta.detach()+step)-f(theta.detach()-step))/(2*step)
    checks['normal_direction_finite_difference']=bool(torch.allclose(grad,fd,atol=1e-8,rtol=1e-7))
    pred=torch.tensor([[[1.2,.8]],[[.4,.7]],[[.5,.9]]],dtype=dtype,requires_grad=True)
    target=torch.tensor([[[1.,1.]],[[.5,.5]],[[.5,.5]]],dtype=dtype)
    loss,_=m.censored_symmetric_loss(pred,target,torch.ones(1,1,2,dtype=dtype))
    g=torch.autograd.grad(loss,pred)[0]
    checks['saturated_satisfied_zero_gradient']=bool(g[0,0,0]==0)
    checks['saturated_unsatisfied_increases_prediction']=bool(g[0,0,1]<0)
    checks['unsaturated_channels_retained']=bool(g[1,0,0]<0 and g[1,0,1]>0 and g[2,0,1]>0)
    result=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks,
                normal_autograd=float(grad),normal_finite_difference=float(fd),
                limitation='本测试不声称已验证CUDA XYZ/rotation/scale或真实遮挡几何梯度')
    (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False));assert all(checks.values())


if __name__=='__main__':main()
