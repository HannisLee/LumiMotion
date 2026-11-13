"""固定训练阶段与全局归一化分块梯度。"""
import copy
import torch
from . import physics as P

STAGES = {
    'B': (2000, (1200, 1700), {'env': .02}),
    'C1': (500, (), {'rho': .003}),
    'C2': (1500, (900, 1275), {'rho': .001, 'env': .005}),
    'C3': (1000, (600, 850), {'rho': .0003, 'env': .002, 'phi': .001, 'a': .005}),
}


def make_optimizer(stage, params, c0):
    _, _, rates = STAGES[stage]
    for name, tensor in params.items():
        tensor.requires_grad_(name in rates)
    groups = [dict(params=[params[name]], lr=rate * (max(c0, .001) if name == 'env' else 1.), name=name)
              for name, rate in rates.items()]
    return torch.optim.Adam(groups, betas=(.9,.999), eps=1e-8, weight_decay=0)


def schedule(optimizer, stage, step):
    if step in STAGES[stage][1]:
        for group in optimizer.param_groups:
            group['lr'] *= .3


def objective_backward(data, params, K, scale, sign, chunk=128, backward=True):
    """每块只累加全局分母归一化的贡献，故与整块梯度相同。"""
    n, v, target, w, b = (data[k] for k in ('n','v','target','w','b'))
    T, G = w.shape
    total_w = w.sum()
    if total_w <= 0 or (w.sum(0) <= 0).any():
        raise ValueError('零权重')
    totals = [0., 0.]
    for start in range(0, G, chunk):
        sl = slice(start, min(start+chunk, G))
        # 每块重建方向计算图，避免反复 backward 同一图。
        l = P.direction(params['phi'], params['a'])
        basis = None
        if 'basis' in data:
            basis = (data['basis'][0][:,sl], data['basis'][1][:,sl])
        direct, static = P.components(n[:,sl], v[:,sl], params['rho'][sl], params['env'], l, K, basis)
        ww = w[:,sl]
        rgb = (ww[...,None]*(direct+static-target[:,sl]).square()).sum()/(3*scale**2*total_w)
        sbar = (ww*(static @ static.new_tensor(P.Y))).sum(0)/ww.sum(0)
        static_loss = (sbar-b[sl]).abs().sum()/(G*scale)
        piece = rgb+static_loss
        if not torch.isfinite(piece):
            raise FloatingPointError('非有限损失')
        if backward and piece.requires_grad:
            piece.backward()
        totals[0] += rgb.item()
        totals[1] += static_loss.item()
    rev = P.reverse_penalty(params['phi'],sign)
    if backward and rev.requires_grad:
        (.1*rev).backward()
    return dict(rgb=totals[0], static=totals[1], reverse=rev.item(), total=totals[0]+totals[1]+.1*rev.item())


def step(data, params, optimizer, stage, number, K, scale, sign, chunk=128):
    schedule(optimizer, stage, number)
    optimizer.zero_grad(set_to_none=True)
    parts = objective_backward(data, params, K, scale, sign, chunk)
    for p in params.values():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError('非有限梯度')
    optimizer.step()
    with torch.no_grad():
        params['rho'].clamp_(0,1)
        params['env'].clamp_(min=0)
    for p in params.values():
        if not torch.isfinite(p).all():
            raise FloatingPointError('非有限参数')
    return parts


def checkpoint(params, optimizer, stage, number):
    return {'params': {k:v.detach().clone() for k,v in params.items()},
            'optimizer': copy.deepcopy(optimizer.state_dict()), 'stage':stage, 'step':number}


def restore(state, params, optimizer):
    with torch.no_grad():
        for k,v in state['params'].items():
            params[k].copy_(v)
    optimizer.load_state_dict(state['optimizer'])
