"""只使用合成数据验证已实现的物理核心。"""
import math
import torch
from . import physics as P


def run():
    torch.manual_seed(111301)
    dtype = torch.float64
    omega, solid = P.grid('cpu', dtype)
    assert abs(solid.sum().item()-4*math.pi) < 1e-12
    # 绕 z 旋转一个 texel 的对称性以及非负有限 GGX。
    n = torch.tensor([[[0., 0., 1.], [0.6, 0., .8]]], dtype=dtype)
    v = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]], dtype=dtype)
    diffuse, spec = P.brdf_basis(n, v)
    assert torch.isfinite(diffuse).all() and torch.isfinite(spec).all()
    assert (diffuse >= 0).all() and (spec >= 0).all()
    # 恒定环境的漫反射近似积分，减去已明确计算的 Fresnel 权重。
    assert .90 < diffuse[0, 0].sum().item() < 1.05
    angle = 2*math.pi/16
    rot = torch.tensor([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]], dtype=dtype)
    d2, s2 = P.brdf_basis(n @ rot.T, v @ rot.T)
    assert torch.allclose(diffuse.sum(-1), d2.sum(-1), atol=1e-12, rtol=0)
    assert torch.allclose(spec.sum(-1), s2.sum(-1), atol=1e-12, rtol=0)
    # 时间变化的法线产生不同静态项，但其时间均值是 b。
    n = torch.tensor([[[0.,0.,1.],[0.,0.,1.]], [[.6,0.,.8],[0.,.6,.8]]], dtype=dtype)
    v = torch.tensor([0.,0.,1.], dtype=dtype).expand_as(n)
    rho = torch.tensor([[.3,.4,.5],[.5,.4,.3]], dtype=dtype, requires_grad=True)
    env = torch.full((128,3), .7, dtype=dtype, requires_grad=True)
    phi = torch.tensor([0.,1.], dtype=dtype)
    a = torch.zeros(2, dtype=dtype)
    l = P.direction(phi,a)
    K = 2.
    direct, static = P.components(n,v,rho,env,l,K)
    assert not torch.allclose(static[0], static[1])
    b = (static.detach() @ torch.tensor(P.Y, dtype=dtype)).mean(0)
    w = torch.ones((2,2),dtype=dtype)
    target = (direct+static).detach()+.1
    rgb, sl, sbar = P.loss_parts(direct,static,target,w,b,.5)
    assert torch.allclose(sbar,b) and sl.item() < 1e-14
    gd = torch.autograd.grad(direct.sum(),rho,retain_graph=True)[0]
    gs = torch.autograd.grad(static.sum(),rho,retain_graph=True)[0]
    assert gd.abs().sum() > 0 and gs.abs().sum() > 0
    assert torch.autograd.grad(rgb,env,retain_graph=True)[0].abs().sum() > 0
    _, sl2, _ = P.loss_parts(direct,static,target,w,b+.1,.5)
    assert torch.autograd.grad(sl2,env,retain_graph=True)[0].abs().sum() > 0
    # E4 动态代数：中性 RGB rhoY/K 映射。
    rho_y = torch.tensor([.3,1.2],dtype=dtype)
    rr = (rho_y/K)[:,None].expand(-1,3)
    dd, _ = P.components(n,v,rr,env,l,K)
    expected = rho_y[None,:,None]*torch.einsum('tgc,tc->tg',n,l).clamp_min(0)[...,None]
    assert (dd-expected).abs().max() < 1e-10
    # 边界按闭区间，失败不替补由调用方实现。
    assert P.guard(.3,.3,.1,.25)[0]
    assert not P.guard(.3,.3,.10001,.25)[0]
    assert P.ALPHA == .5
    return {'固角': 'PASS', '固定粗糙度与有限非负BRDF': 'PASS', '旋转对称': 'PASS',
            '共享rho梯度': 'PASS', 'RGB与b环境梯度': 'PASS', '时间均值': 'PASS',
            'E4动态代数': 'PASS', '门槛边界': 'PASS'}
