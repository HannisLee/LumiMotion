"""逐点独立形变的世界位置Jacobian对角度量；不含GT或优化器。"""
import torch


def diagonal_metric(x, world):
    # 当前DeformNetwork为逐行MLP，无点间聚合，因此求和VJP逐行给出Jacobian。
    columns = torch.zeros_like(x)
    for axis in range(3):
        row = torch.autograd.grad(world[:, axis].sum(), x, retain_graph=axis < 2)[0]
        columns.add_(row.detach().square())
    if not torch.isfinite(columns).all():
        raise RuntimeError('非有限形变Jacobian度量')
    return columns


def affine_check(device):
    x=torch.tensor([[.2,.3,.4],[.4,-.2,.1]],dtype=torch.float64,device=device,requires_grad=True)
    a=torch.tensor([[2.,1.,0.],[0.,3.,0.],[1.,0.,4.]],dtype=torch.float64,device=device)
    measured=diagonal_metric(x,x@a.T)
    expected=a.square().sum(0).expand_as(x)
    error=float((measured-expected).abs().max())
    assert error==0
    return dict(status='PASS',max_error=error,scope='已知逐点仿射映射的解析Jacobian；非native梯度审计')
