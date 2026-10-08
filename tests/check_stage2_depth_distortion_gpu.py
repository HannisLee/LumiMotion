"""独立CUDA畸变核：成对双精度参考、有限差分与原输出回归；无GT输入。"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch
from utils.graphics_utils import getProjectionMatrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-path', default='')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.module_path:
        sys.path.insert(0, str(Path(args.module_path).resolve()))
    stable = importlib.import_module('diff_surfel_rasterization_stable')
    old = importlib.import_module('diff_surfel_rasterization')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'producer_snapshot.py').write_text(Path(__file__).read_text())
    records = []
    size = 17
    multiplier = 1e8

    def render(module, xyz, opacity):
        settings = module.GaussianRasterizationSettings(
            image_height=size, image_width=size, tanfovx=.5, tanfovy=.5,
            bg=torch.tensor([.1, .2, .3], device='cuda'), scale_modifier=1.,
            viewmatrix=torch.eye(4, device='cuda'),
            projmatrix=getProjectionMatrix(.2, 100., 2*math.atan(.5),
                                           2*math.atan(.5)).T.cuda(),
            sh_degree=0, campos=torch.zeros(3, device='cuda'),
            prefiltered=False, debug=False)
        return module.GaussianRasterizer(settings)(
            means3D=xyz, means2D=torch.zeros_like(xyz), opacities=opacity,
            colors_precomp=torch.tensor([[.3, .7, .5], [.6, .2, .8], [.4, .6, .1]],
                                       device='cuda'),
            scales=torch.full((3, 2), .4, device='cuda'),
            rotations=torch.tensor([[1., 0., 0., 0.]]*3, device='cuda'))

    def reference(z, opacity):
        weights = opacity * torch.cat((opacity.new_ones(1), (1-opacity[:-1]).cumprod(0)))
        return sum(weights[i]*weights[j]*(100*.2/(100-.2)*(z[j]-z[i])/(z[i]*z[j]))**2
                   for i in range(3) for j in range(i+1, 3))*multiplier

    for depths in ([3., 3.001, 3.007], [8., 8.0001, 8.0008], [8., 8., 8.]):
        xyz = torch.tensor([[0., 0., z] for z in depths], device='cuda', requires_grad=True)
        opacity = torch.tensor([[.3], [.45], [.5]], device='cuda', requires_grad=True)
        rgb, radii, maps = render(stable, xyz, opacity)
        old_rgb, old_radii, old_maps = render(old, xyz, opacity)
        torch.testing.assert_close(rgb, old_rgb, atol=0, rtol=0)
        torch.testing.assert_close(radii, old_radii, atol=0, rtol=0)
        for channel in [0, 1, 2, 3, 4, 5]:
            torch.testing.assert_close(maps[channel], old_maps[channel], atol=0, rtol=0)
        assert bool(torch.isfinite(maps).all())
        loss = maps[6, size//2, size//2]*multiplier
        actual_grad = torch.autograd.grad(loss, (xyz, opacity), retain_graph=True)
        z = xyz[:, 2].detach().double().requires_grad_()
        a = opacity[:, 0].detach().double().requires_grad_()
        expected = reference(z, a)
        expected_grad = torch.autograd.grad(expected, (z, a))
        torch.testing.assert_close(loss.double(), expected, atol=1e-9, rtol=1e-3)
        torch.testing.assert_close(actual_grad[0][:, 2].double(), expected_grad[0], atol=1e-7, rtol=2e-3)
        torch.testing.assert_close(actual_grad[1][:, 0].double(), expected_grad[1], atol=1e-9, rtol=2e-3)
        differences = {}
        for label, index, step in [('depth', 0, 5e-5), ('opacity', 1, 1e-3)]:
            # Exact ties change the alpha-compositing order under individual
            # depth perturbations. Check their zero gradient and common shift
            # below; finite differences here require an unchanged sort order.
            if index == 0 and depths[0] == depths[-1]:
                with torch.no_grad():
                    shifted = xyz.detach().clone()
                    shifted[:, 2] += step
                    assert float(render(stable, shifted, opacity.detach())[2][6].abs().max()) == 0.
                continue
            fd = []
            for point in range(3):
                plus_xyz, minus_xyz = xyz.detach().clone(), xyz.detach().clone()
                plus_a, minus_a = opacity.detach().clone(), opacity.detach().clone()
                plus, minus = ((plus_xyz, minus_xyz) if index == 0 else (plus_a, minus_a))
                component = 2 if index == 0 else 0
                plus[point, component] += step
                minus[point, component] -= step
                denom = (plus[point, component]-minus[point, component]).double()
                with torch.no_grad():
                    lp = render(stable, plus_xyz, plus_a)[2][6, size//2, size//2].double()*multiplier
                    lm = render(stable, minus_xyz, minus_a)[2][6, size//2, size//2].double()*multiplier
                fd.append((lp-lm)/denom)
            fd = torch.stack(fd)
            analytical = actual_grad[index][:, 2 if index == 0 else 0].double()
            torch.testing.assert_close(fd, analytical, atol=1e-5 if index == 0 else 1e-7, rtol=4e-2)
            differences[label] = {'finite_difference':fd.tolist(), 'analytical':analytical.tolist()}
        old_regular = old_rgb.mean()+old_maps[:6].mean()
        new_regular = rgb.mean()+maps[:6].mean()
        old_grad = torch.autograd.grad(old_regular, (xyz, opacity))
        new_grad = torch.autograd.grad(new_regular, (xyz, opacity))
        for x, y in zip(old_grad, new_grad):
            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
        if depths[0] == depths[-1]:
            assert float(loss) == 0.
            assert float(actual_grad[0].abs().max()) == 0.
            assert float(actual_grad[1].abs().max()) == 0.
        records.append({'float32_depths':xyz[:, 2].tolist(), 'scaled_distortion':float(loss),
                        'double_pairwise_reference':float(expected),
                        'old_scaled_distortion':float(old_maps[6, size//2, size//2]*multiplier),
                        'gradients':differences})
        print(records[-1], flush=True)
    result = {'passed':True, 'gt_normal_inputs':False, 'gt_depth_inputs':False,
              'old_regular_forward_bit_exact':True, 'old_regular_gradient_close':True,
              'near_coincident_depth_and_opacity_gradients_verified':True, 'records':records}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'producer_snapshot.py').write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
