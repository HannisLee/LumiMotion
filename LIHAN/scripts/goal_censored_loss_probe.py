"""满码单边损失的CPU数值正负对照；不读取训练图片或更新模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from scene.static_dl_censored import censored_photometric_loss, loss_for_mode, saturated_channels
from scene.static_dl_photometric import robust_photometric_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'output/smoke_test'):
        parser.error('数值冒烟必须位于output/smoke_test。')
    output.mkdir(parents=True, exist_ok=False)
    alpha = torch.ones(1, 1, 1, dtype=torch.float64)
    records, checks = {}, {}
    cases = [('满码低估', 1., .8, -1), ('满码超界', 1., 1.3, 0),
             ('非满码高估', .6, 1.3, 1), ('非满码低估', .6, .5, -1)]
    for name, y, p, sign in cases:
        value = torch.tensor([[[p]]], dtype=torch.float64, requires_grad=True)
        target = torch.full_like(value, y)
        loss = censored_photometric_loss(value, target, alpha)[0]
        gradient = float(torch.autograd.grad(loss, value)[0])
        epsilon = 1e-5
        # 各点都远离权重阈值与截断折点，可检验局部有限差分。
        plus = censored_photometric_loss(value.detach()+epsilon, target, alpha)[0]
        minus = censored_photometric_loss(value.detach()-epsilon, target, alpha)[0]
        fd = float((plus-minus)/(2*epsilon))
        records[name] = dict(loss=float(loss), gradient=gradient, finite_difference=fd)
        checks[name] = (gradient == 0 if sign == 0 else gradient*sign > 0) and abs(gradient-fd) < 1e-8

    value = torch.tensor([[[1.3]]], dtype=torch.float64, requires_grad=True)
    target = torch.full_like(value, .6)
    wrong = robust_photometric_loss(value.clamp_max(1.), target, alpha)[0]
    wrong_gradient = float(torch.autograd.grad(wrong, value)[0])
    records['全图截断负控'] = dict(gradient=wrong_gradient)
    checks['全图截断负控'] = wrong_gradient == 0 and records['非满码高估']['gradient'] > 0

    value = torch.tensor([1.3, 1.2, .2], dtype=torch.float64).reshape(3, 1, 1).requires_grad_(True)
    target = torch.tensor([1., .4, .7], dtype=torch.float64).reshape(3, 1, 1)
    grad = torch.autograd.grad(censored_photometric_loss(value, target, alpha)[0], value)[0].flatten()
    records['逐通道混合'] = dict(gradient=grad.tolist())
    checks['逐通道混合'] = bool(grad[0] == 0 and grad[1] > 0 and grad[2] < 0)

    # 上游非平凡映射也必须完全保留所有参数导数，而不只检查loss值。
    parameter = torch.tensor([.3, .8, 1.2], dtype=torch.float64, requires_grad=True)
    image = (parameter.square()+.1*parameter.roll(1)).reshape(3, 1, 1)
    target = torch.tensor([.2, .5, .9], dtype=torch.float64).reshape(3, 1, 1)
    old = robust_photometric_loss(image, target, alpha)[0]
    new = censored_photometric_loss(image, target, alpha)[0]
    old_grad = torch.autograd.grad(old, parameter, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new, parameter)[0]
    records['非满码完全同义'] = dict(loss_difference=float(new-old),
        parameter_gradient_max_difference=float((old_grad-new_grad).abs().max()))
    checks['非满码完全同义'] = bool(torch.equal(old, new) and torch.equal(old_grad, new_grad))
    checks['双边模式调用旧函数'] = loss_for_mode('two_sided') is robust_photometric_loss

    mask_target = torch.tensor([1., 250./255, 254./255], dtype=torch.float64).reshape(3, 1, 1).requires_grad_(True)
    mask_alpha = alpha.clone().requires_grad_(True)
    mask = saturated_channels(mask_target, mask_alpha)
    checks['仅精确满码且掩膜停止梯度'] = mask.flatten().tolist() == [True, False, False] and not mask.requires_grad
    checks['背景保持双边'] = not bool(saturated_channels(torch.ones_like(mask_target), alpha*.5).any())
    sources = [Path(__file__), ROOT/'scene/static_dl_censored.py', ROOT/'scene/static_dl_photometric.py']
    result = dict(status='PASS' if all(checks.values()) else 'FAILED', checks=checks, cases=records,
        device='cpu', dtype='float64', training_images_loaded=0, parameter_updates=0,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (output/'run.log').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    command = shlex.join([sys.executable, str(Path(__file__).resolve())]+sys.argv[1:])
    (output/'README.md').write_text('# 满码光度损失CPU数值审计\n\n完整命令：\n\n```bash\n'+command+
        '\n```\n\nCPU float64标量、逐通道、上游参数导数和错误全图截断负控；无图像、模型、GT或参数更新。'+
        '\n\n日志：`run.log`；数值和源码SHA：`result.json`。source/model/iteration、渲染命令和四类目检不适用，本测试不进行渲染。'+
        '\n\n结论：**'+result['status']+'**；不代表真实模型或Goal已通过。\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] != 'PASS':
        raise RuntimeError('标量损失审计失败，保留全部输出。')


if __name__ == '__main__':
    main()
