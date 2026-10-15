"""未知远场主光与共享表面响应；纯 Torch，不读取任何外部资产。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SurfaceLightFit(nn.Module):
    def __init__(self, times, initial_directions, patches):
        super().__init__()
        if len(times) < 2 or not (times[1:] > times[:-1]).all():
            raise ValueError('拟合时间须严格递增且至少两个。')
        self.register_buffer('times', times.clone())
        self.raw_directions = nn.Parameter(F.normalize(initial_directions, dim=-1))
        self.log_energy = nn.Parameter(torch.zeros_like(times))
        self.raw_albedo = nn.Parameter(torch.full((patches, 3), math.log(math.expm1(.5)), device=times.device))

    def lighting(self, times):
        if (self.raw_directions.norm(dim=-1) < 1e-6).any():
            raise RuntimeError('方向接近零，拒绝无效归一化。')
        directions = F.normalize(self.raw_directions, dim=-1)
        # 训练时间节点上的平均 E/pi 恒为1；共享有效 albedo 采用相同尺度。
        energy = torch.softmax(self.log_energy, dim=0) * len(self.times)
        right = torch.searchsorted(self.times, times.contiguous()).clamp(1, len(self.times)-1)
        left = right - 1
        weight = ((times-self.times[left])/(self.times[right]-self.times[left])).clamp(0, 1)
        vector = torch.lerp(directions[left], directions[right], weight[:, None])
        # 对跖点插值无唯一解，采用较近端点并记录为模型局限。
        fallback = torch.where((weight < .5)[:, None], directions[left], directions[right])
        vector = torch.where((vector.norm(dim=-1) < 1e-6)[:, None], fallback, vector)
        return F.normalize(vector, dim=-1), torch.lerp(energy[left], energy[right], weight)

    def forward(self, times, normals):
        directions, energy = self.lighting(times)
        cosine = (normals * directions[:, None]).sum(-1).clamp_min(0)
        return F.softplus(self.raw_albedo)[None] * cosine[..., None] * energy[:, None, None]

    def regularizer(self):
        directions, energy = self.lighting(self.times)
        return (directions[1:] - directions[:-1]).square().mean() + .1 * (energy[1:].log()-energy[:-1].log()).square().mean()


def observation_loss(prediction, target, saturated, valid):
    residual = torch.where(saturated, (prediction-target).clamp_max(0), prediction-target)
    # 固定观测掩膜、固定分母；稳健损失减少未建模阴影与镜面的影响。
    return ((residual.square()+1e-6).sqrt()*valid[..., None]).sum() / (valid.sum().clamp_min(1)*3)


@torch.no_grad()
def initialize_directions(normals, target, valid, saturated, method, seed):
    generator = torch.Generator(device=normals.device).manual_seed(seed)
    if method == 'random':
        return F.normalize(torch.randn(len(normals), 3, generator=generator, device=normals.device), dim=-1)
    # 固定256个球面候选；逐帧轮廓余弦与亮度拟合正尺度，仅使用未饱和观测。
    k = torch.arange(256, device=normals.device)
    z = 1 - 2*(k+.5)/256
    phi = k*(math.pi*(3-math.sqrt(5)))
    radius = (1-z*z).sqrt()
    candidates = torch.stack([radius*phi.cos(), radius*phi.sin(), z], -1)
    output = []
    for n, y, v, s in zip(normals, target, valid, saturated):
        weight = v & ~s.any(-1)
        if weight.sum() < 16:
            raise ValueError('候选估光未饱和观测不足16个，不能伪造初值。')
        h = (n[weight] @ candidates.T).clamp_min(0)
        intensity = y[weight].mean(-1, keepdim=True)
        scale = (h*intensity).sum(0) / h.square().sum(0).clamp_min(1e-8)
        residual = (h*scale[None]-intensity).square().mean(0)
        output.append(candidates[residual.argmin()])
    return torch.stack(output)
