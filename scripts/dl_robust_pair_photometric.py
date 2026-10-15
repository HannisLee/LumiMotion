"""共同反射率消元后的逐对径向Cauchy损失；不删除对应。"""
import torch


def pair_errors(p,D,E,train=True):
    m=p['train'] if train else ~p['train'];a=p['a'][m];b=p['b'][m]
    sa=E[a]*((p['na'][m]*D[a]).sum(-1).clamp_min(0)+.1)
    sb=E[b]*((p['nb'][m]*D[b]).sum(-1).clamp_min(0)+.1)
    ya=p['ya'][m];yb=p['yb'][m]
    rho=(sa[:,None]*ya+sb[:,None]*yb)/(sa.square()+sb.square()).clamp_min(1e-12)[:,None]
    return ((sa[:,None]*rho-ya).double().square()+(sb[:,None]*rho-yb).double().square()).mean(-1)/2


def scale_from_training(p,D,E):
    return max(float(pair_errors(p,D,E,True).sqrt().median()),1e-4)


def robust_loss(p,D,E,scale,mode):
    e=pair_errors(p,D,E,True)
    if mode=='mse':return e.mean()
    assert mode=='robust' and scale>0
    return (scale**2*torch.log1p(e/scale**2)).mean()
