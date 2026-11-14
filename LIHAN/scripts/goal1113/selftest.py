"""只使用合成数据验证已实现的物理核心。"""
import math
import json
from pathlib import Path
import torch
from . import physics as P, inputs as I, freeze
from . import optimizer as O


def run(out=None):
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
    # 分块与整块总损失及全部梯度一致。
    params = {'rho': rho.detach().clone().requires_grad_(), 'env': env.detach().clone().requires_grad_(),
              'phi': phi.clone().requires_grad_(), 'a': a.clone().requires_grad_()}
    data = dict(n=n, v=v, target=target, w=w, b=b+.1)
    full = O.objective_backward(data,params,K,.5,1.,chunk=2)
    grads = {k:p.grad.clone() for k,p in params.items()}
    for p in params.values(): p.grad = None
    split = O.objective_backward(data,params,K,.5,1.,chunk=1)
    assert abs(full['total']-split['total']) < 1e-12
    assert all(torch.allclose(grads[k],p.grad,atol=1e-12,rtol=1e-12) for k,p in params.items())
    # 阶段冻结计划、学习率衰减与包含优化器状态的恢复一致。
    for stage, expected in [('B',{'env'}),('C1',{'rho'}),('C2',{'rho','env'}),('C3',{'rho','env','phi','a'})]:
        opt = O.make_optimizer(stage,params,.7)
        assert {k for k,p in params.items() if p.requires_grad} == expected
    opt = O.make_optimizer('C3',params,.7)
    O.step(data,params,opt,'C3',1,K,.5,1.,1)
    saved = O.checkpoint(params,opt,'C3',1)
    O.step(data,params,opt,'C3',2,K,.5,1.,1)
    want = {k:p.detach().clone() for k,p in params.items()}
    O.restore(saved,params,opt)
    O.step(data,params,opt,'C3',2,K,.5,1.,1)
    assert all(torch.equal(want[k],p) for k,p in params.items())
    try:
        I.sha(Path('/synthetic/lights.json'))
    except ValueError:
        pass
    else: raise AssertionError('未拒绝 GT 路径')
    if out is not None:
        out=Path(out)
        # 序列化检查点后恢复，验证磁盘恢复也保持优化器状态。
        path=out/'synthetic_checkpoint.pt'; torch.save(saved,path)
        disk=torch.load(path,map_location='cpu',weights_only=False)
        O.restore(disk,params,opt); O.step(data,params,opt,'C3',2,K,.5,1.,1)
        assert all(torch.equal(want[k],p) for k,p in params.items())
        # 合成冻结与篡改检测；失败状态不允许替补阶段为主预测。
        root=out/'freeze_fixture'
        for name in ('inputs','fitting','render','figures'): (root/name).mkdir(parents=True)
        (root/'fitting/status.json').write_text(json.dumps({'selected':None,'status':'INIT_GUARD_FAILED'}))
        for name in ('manifest.json','config.json','prior.npz'): (root/'inputs'/name).write_bytes(b'synthetic')
        for name in ('summary.json','total.mp4'): (root/'render'/name).write_bytes(b'synthetic')
        for i in range(1,151): (root/'render'/f'linear_{i:04d}.npz').write_bytes(b'synthetic')
        for name in ('01-rgb','02-components','03-env-b','04-trajectory'): (root/'figures'/f'{name}.png').write_bytes(b'synthetic')
        m=freeze.run(root); assert m['selected'] is None and freeze.verify(root)['selected'] is None
        (root/'render/linear_0001.npz').write_bytes(b'tampered')
        try: freeze.verify(root)
        except ValueError: pass
        else: raise AssertionError('冻结篡改未检出')
    return {'固角': 'PASS', '固定粗糙度与有限非负BRDF': 'PASS', '旋转对称': 'PASS',
            '共享rho梯度': 'PASS', 'RGB与b环境梯度': 'PASS', '时间均值': 'PASS',
            'E4动态代数': 'PASS', '门槛边界': 'PASS', '分块梯度': 'PASS',
            '阶段冻结': 'PASS', '恢复优化器': 'PASS', 'GT路径拒绝':'PASS',
            '磁盘恢复和冻结篡改': 'PASS' if out is not None else '未执行',
            '失败不替补': 'PASS' if out is not None else '未执行'}
