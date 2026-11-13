"""冻结配置的方向光及环境光渲染方程；不访问真实数据。"""
import math
import torch

ALPHA = .5
EPS = 1e-8
Y = (.2126, .7152, .0722)


def grid(device, dtype=torch.float64):
    i = torch.arange(8, device=device, dtype=dtype)
    j = torch.arange(16, device=device, dtype=dtype)
    theta = (i + .5) * math.pi / 8
    phi = (j + .5) * 2 * math.pi / 16
    t, p = torch.meshgrid(theta, phi, indexing='ij')
    omega = torch.stack((t.sin()*p.cos(), t.sin()*p.sin(), t.cos()), -1).reshape(-1, 3)
    solid = ((2*math.pi/16)*(torch.cos(i*math.pi/8)-torch.cos((i+1)*math.pi/8)))[:, None].expand(8, 16).reshape(-1)
    return omega, solid


def direction(phi, a):
    e = math.pi/4 + math.pi/36 * torch.tanh(a)
    return torch.stack((e.cos()*phi.cos(), e.cos()*phi.sin(), e.sin()), -1)


def brdf_basis(n, v):
    """返回漫反射 rho 系数及独立于 rho 的镜面系数，均含 cos 和固角。"""
    omega, solid = grid(n.device, n.dtype)
    nv = (n*v).sum(-1).clamp(0, 1)
    nl = torch.einsum('...c,jc->...j', n, omega).clamp(0, 1)
    h0 = v[..., None, :] + omega
    hn = torch.linalg.vector_norm(h0, dim=-1, keepdim=True)
    h = h0 / hn.clamp_min(EPS)
    nh = (n[..., None, :]*h).sum(-1).clamp(0, 1)
    vh = (v[..., None, :]*h).sum(-1).clamp(0, 1)
    fresnel = .04 + .96*(1-vh).pow(5)
    aa = ALPHA**2
    ggx = aa / (math.pi*((nh.square()*(aa-1)+1).square()))
    def g1(c):
        return 2*c/(c + torch.sqrt(aa+(1-aa)*c.square())).clamp_min(EPS)
    geom = g1(nv)[..., None]*g1(nl)
    spec = ggx*geom*fresnel/(4*nv[..., None].clamp_min(EPS)*nl.clamp_min(EPS))
    mask = (nv[..., None] > 1e-6) & (nl > 0) & (hn[..., 0] > EPS)
    factor = nl*solid
    diffuse = torch.where(mask, (1-fresnel)*factor/math.pi, 0.)
    spec = torch.where(mask, spec*factor, 0.)
    return diffuse, spec


def components(n, v, rho, env, l, K, basis=None):
    """n/v 为 T×G×3，rho 为 G×3，env 为 128×3。"""
    dif, spec = brdf_basis(n, v) if basis is None else basis
    static = torch.einsum('tgj,jc->tgc', dif, env)*rho[None] + torch.einsum('tgj,jc->tgc', spec, env)
    direct = K*rho[None]*torch.einsum('tgc,tc->tg', n, l).clamp_min(0)[..., None]
    return direct, static


def loss_parts(direct, static, target, weight, b, scale):
    """权重与完整支持集的归一化；用于整块数值核验。"""
    total_w = weight.sum()
    if not bool(total_w > 0) or bool((weight.sum(0) <= 0).any()):
        raise ValueError('权重总和或高斯时间权重为零')
    rgb = (weight[..., None]*(direct+static-target).square()).sum()/(3*scale**2*total_w)
    luminance = static @ static.new_tensor(Y)
    sbar = (weight*luminance).sum(0)/weight.sum(0)
    static_loss = (sbar-b).abs().mean()/scale
    return rgb, static_loss, sbar


def reverse_penalty(phi, sign):
    return torch.relu(-sign*torch.diff(phi)).square().mean()


def guard(energy, e0, mean, p90):
    lo, hi = max(0., e0-.20), min(.65, e0+.20)
    if lo > hi:
        return False, 'PRIOR_GUARD_CONFLICT'
    if not (lo <= energy <= hi and mean <= .10 and p90 <= .25):
        return False, 'INIT_GUARD_FAILED'
    return True, 'PASS'
