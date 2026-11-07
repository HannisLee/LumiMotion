"""T01(1107-03) I5：Stage3 完整管线独立评测（TEST15 / DEV18 / 训练 3 帧冒烟）。

运行方式（经 snapshot 的 bootstrap，保证用的是冻结 snapshot 代码）：
  python <S>/LIHAN/scripts/bootstrap_1011.py <S> LIHAN.scripts.pipeline1107_evaluate \
      --model <输出>/model_mlp --iteration 55000 --split test|dev|fit|train3 --seed 0 \
      --source <总控>/test_source --directions <总控>/e4_test_directions.json --output <新目录>

口径：400x400 黑底全浮点 sRGB；PSNR(float64，并用 NumPy 独立复算校验)、SSIM(utils.loss_utils.ssim 11x11 sigma1.5)、
LPIPS(vgg，权重不可用时记 N/A 并写明原因)；前景 PSNR、前景包围框 SSIM/LPIPS。torch.manual_seed(seed)。
评测只读 checkpoint，不改任何文件，不读 GT 材质/几何(TEST 目标是 RGB 图像)；输出目录禁止覆盖。

--dump-leak 额外为泄漏监测保存逐像素量(环境可见度、SUN 直射项、残差、分量、法线、albedo)。
分量(环境/SUN/间接)通过包装 rendering_equation 取得，复刻 stage3_el_eldl_mixed.equation 的合成逻辑，
合成结果与原函数逐项相同，探针只记录。
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
VIS_FRAMES = dict(test=[10, 80, 150], dev=[61, 69, 79])


# ---------------------------------------------------------------- 纯函数(可 CPU 自检)
def psnr_from_mse(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def independent_psnr(rgb, target):
    """NumPy float64 独立复算（与 torch 路径无共享代码）。"""
    diff = np.asarray(rgb, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return psnr_from_mse(np.mean(diff * diff))


def aggregate(rows, keys):
    out = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(values)) if len(values) == len(rows) and rows else None
    return out


def foreground_bbox(mask2d):
    ys, xs = torch.where(mask2d > 0)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


# ---------------------------------------------------------------- 分量探针
class ComponentProbe:
    """记录混合方程各分支(环境/SUN/间接)的逐像素量。每次 render 前 reset。"""

    def __init__(self):
        self.parts = []

    def reset(self):
        self.parts = []

    def collect(self):
        if not self.parts:
            return None
        keys = self.parts[0].keys()
        return {k: torch.cat([p[k] for p in self.parts], 0) for k in keys}


def install_probe(module, original_equation, probe):
    """用探针版本替换 module.rendering_equation（须在 install_evaluation() 之后调用）。

    非混合调用(E0，无方向光)原样转发；混合调用与 stage3_el_eldl_mixed.equation 的数值合成一致。
    """
    def equation(*args, **kwargs):
        if kwargs.get('directional_lights') is None or kwargs.get('envlight') is None:
            result = original_equation(*args, **kwargs)
            probe.parts.append(dict(env=(result['diffuse'] + result['specular']).detach().float().cpu(),
                                    ind=result['light_indirect'].detach().float().cpu(),
                                    env_vis=result['visibility'].detach().float().cpu().reshape(-1)))
            return result
        environment = dict(kwargs)
        environment['directional_lights'] = None
        env = original_equation(*args, **environment)
        direct = dict(kwargs)
        direct['envlight'] = None
        positional = list(args)
        pipe = copy.copy(positional[6] if len(positional) > 6 else direct['pipe'])
        pipe.wo_indirect = True
        if len(positional) > 6:
            positional[6] = pipe
        else:
            direct['pipe'] = pipe
        sun = original_equation(*positional, **direct)
        normals = args[2]
        directions = kwargs['directional_lights'][0].to(normals)
        cosine = (normals[:, None] * torch.nn.functional.normalize(directions, dim=-1)[None]).sum(-1).clamp_min(0).amax(1)
        sun_vis = sun['visibility'].reshape(len(normals), -1).mean(1)
        probe.parts.append(dict(
            env=(env['diffuse'] + env['specular']).detach().float().cpu(),
            sun=(sun['diffuse'] + sun['specular']).detach().float().cpu(),
            ind=env['light_indirect'].detach().float().cpu(),
            env_vis=env['visibility'].detach().float().cpu().reshape(-1),
            sun_vis=sun_vis.detach().float().cpu(),
            s=(cosine * sun_vis).detach().float().cpu()))
        return {k: (env[k] + sun[k] if k in {'diffuse', 'specular', 'light_direct', 'light', 'light_blocked'} else env[k])
                for k in env}
    module.rendering_equation = equation


def scatter(values, mask, shape_hw):
    """把前景像素序列(N,C)按布尔 mask(H,W) 还原成 (C,H,W)。"""
    channels = values.shape[1] if values.ndim == 2 else 1
    image = torch.zeros(channels, *shape_hw)
    image[:, mask.cpu().bool()] = values.reshape(len(values), -1).T
    return image


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--split', choices=['dev', 'test', 'train3', 'fit'], required=True)
    p.add_argument('--source', required=True, help='dev_source / test_source / train_source')
    p.add_argument('--directions', default='', help='E0 留空；E1/E2 用对应 split 的 e4_*_directions.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--bias', type=float, default=.1)
    p.add_argument('--vis-frames', type=int, nargs='*')
    p.add_argument('--dump-leak', action='store_true')
    p.add_argument('--no-lpips', action='store_true')
    a = p.parse_args()

    import gaussian_renderer.render_ir as render_module
    original_equation = render_module.rendering_equation       # 在 install_evaluation 之前取原始函数
    from LIHAN.scripts.stage3_el_eldl_mixed import install_evaluation
    install_evaluation()
    probe = ComponentProbe()
    install_probe(render_module, original_equation, probe)
    from scripts.render_stage2 import Stage2RenderRun, save_image
    from scripts.stage2_runtime import FrozenScene, supervision, file_sha256
    from scene.stage2_photometric import frame_key, srgb_to_linear
    from utils.loss_utils import ssim
    from utils.graphics_utils import rgb_to_srgb

    lpips = None
    lpips_note = None
    if not a.no_lpips:
        try:
            from lpipsPyTorch import lpips as lpips_function
            lpips = lpips_function
        except Exception as error:
            lpips_note = 'LPIPS 不可用：' + repr(error)
    else:
        lpips_note = '按参数 --no-lpips 跳过'

    a.output.mkdir(parents=True, exist_ok=False)
    run = Stage2RenderRun(a.model, a.iteration)
    run.args.source_path = a.source
    run.args.diffuse_sample_num = a.samples
    run.args.light_t_min = a.bias
    run.args.pbr_mixed_direct_multiplier = 1.
    if a.directions:
        run.args.pbr_mixed_lights_path = a.directions
    elif getattr(run.args, 'pbr_mixed_lights_path', ''):
        raise ValueError('模型使用混合方向光，必须给 --directions。')
    run.context = FrozenScene(run.args, a.model, a.iteration)
    ctx = run.context
    cameras = list(ctx.train_cameras) if a.split in ('train3', 'fit') else list(ctx.test_cameras)
    if a.split == 'train3':
        cameras = [cameras[i] for i in [0, len(cameras) // 2, len(cameras) - 1]]
    if not cameras:
        raise ValueError('评测相机为空。')
    default_vis = (VIS_FRAMES[a.split] if a.split in VIS_FRAMES else
                   [int(frame_key(c.image_name_train_light)) for c in cameras] if a.split == 'train3' else [])
    vis = set(a.vis_frames if a.vis_frames is not None else default_vis)
    for name in ['rgb', 'gt', 'arrays', 'alpha', 'normal', 'separation', 'albedo', 'components'] + (['leak'] if a.dump_leak else []):
        (a.output / name).mkdir()

    rows = []
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    started = time.time()
    with torch.no_grad():
        for cam in cameras:
            key = frame_key(cam.image_name_train_light)
            probe.reset()
            frame_start = time.time()
            pkg = run.render(cam)
            target, mask = supervision(cam, ctx.background)
            image = pkg['render'].clamp(0, 1)
            if not torch.isfinite(image).all():
                raise FloatingPointError('渲染含非有限值：' + key)
            count = float(mask.sum())
            if count <= 0:
                raise ValueError('前景为空：' + key)
            mse = float((image - target).square().mean())
            fg_mse = float(((image - target).square() * mask).sum() / (count * 3))
            y0, y1, x0, x1 = foreground_bbox(mask[0])
            crop_i, crop_t = (image * mask)[:, y0:y1, x0:x1], (target * mask)[:, y0:y1, x0:x1]
            arrays = dict(rgb=image.cpu().numpy(), target=target.cpu().numpy(), mask=mask.cpu().numpy())
            row = dict(frame=key, psnr=psnr_from_mse(mse), psnr_foreground=psnr_from_mse(fg_mse),
                       ssim=float(ssim(image, target)), ssim_foreground_bbox=float(ssim(crop_i, crop_t)))
            independent = independent_psnr(arrays['rgb'], arrays['target'])
            if abs(independent - row['psnr']) > 1e-4:
                raise AssertionError(f'PSNR 独立复算不一致：{key} {independent} vs {row["psnr"]}')
            row['psnr_numpy_check'] = independent
            if lpips is not None:
                try:
                    row['lpips'] = float(lpips(image, target, net_type='vgg'))
                    row['lpips_foreground_bbox'] = float(lpips(crop_i, crop_t, net_type='vgg'))
                except Exception as error:
                    lpips, lpips_note = None, 'LPIPS 计算失败：' + repr(error)
                    row['lpips'] = row['lpips_foreground_bbox'] = None
            else:
                row['lpips'] = row['lpips_foreground_bbox'] = None
            row['seconds'] = time.time() - frame_start
            np.savez_compressed(a.output / 'arrays' / f'{key}.npz', **arrays)
            save_image(a.output / 'rgb' / f'{key}.png', image)
            save_image(a.output / 'gt' / f'{key}.png', target)

            parts = probe.collect()
            fg = pkg['mask'].detach()
            if parts is not None:
                row['energy_env'] = float(parts['env'].mean())
                row['energy_sun'] = float(parts['sun'].mean()) if 'sun' in parts else None
                row['energy_indirect'] = float(parts['ind'].mean())
                row['energy_indirect_max'] = float(parts['ind'].max())
            if int(key) in vis:
                alpha = pkg['rend_alpha']
                save_image(a.output / 'alpha' / f'{key}.png', alpha.expand(3, -1, -1))
                save_image(a.output / 'normal' / f'{key}.png',
                           (torch.nn.functional.normalize(pkg['rend_normal'], dim=0) * .5 + .5) * alpha)
                save_image(a.output / 'albedo' / f'{key}.png', pkg['base_color'])
                from gaussian_renderer import render as gaussian_render
                d = ctx.deformation(cam)
                f = ctx.pc.get_binary_feature().reshape(-1, 1)
                colors = torch.cat([1 - f, f, torch.zeros_like(f)], dim=1)
                separation = gaussian_render(cam, ctx.pc, run.args, ctx.background, override_color=colors,
                                             **{k: d[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity']})['render']
                save_image(a.output / 'separation' / f'{key}.png', separation)
                if parts is not None:
                    hw = tuple(fg.shape)
                    for name in [n for n in ('env', 'sun', 'ind') if n in parts]:
                        linear = scatter(parts[name], fg, hw).to(image.device)
                        save_image(a.output / 'components' / f'{key}_{name}.png', rgb_to_srgb(linear.clamp_min(0)))
            if a.dump_leak and parts is not None:
                fgc = fg.cpu().bool()
                render_lin = srgb_to_linear(image.detach().cpu())[:, fgc].T
                target_lin = srgb_to_linear(target.detach().cpu())[:, fgc].T
                normal = torch.nn.functional.normalize(pkg['rend_normal'].detach(), dim=0).cpu()[:, fgc].T
                np.savez_compressed(
                    a.output / 'leak' / f'{key}.npz', fg=fgc.numpy(),
                    env_vis=parts['env_vis'].numpy(), sun_vis=parts.get('sun_vis', torch.zeros(0)).numpy(),
                    s=parts.get('s', torch.zeros(0)).numpy(), env=parts['env'].numpy(),
                    sun=parts.get('sun', torch.zeros(0, 3)).numpy(), ind=parts['ind'].numpy(),
                    albedo_lin=(pkg['base_color_linear'] / pkg['rend_alpha'].clamp_min(1e-3)).detach().cpu()[:, fgc].T.numpy(),  # 去 alpha 预乘(黑底)
                    resid_lin=(render_lin - target_lin).numpy(), normal=normal.numpy())
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            ctx.unload_camera(cam)

    if any(file_sha256(Path(path)) != digest for path, digest in ctx.source_files.items()):
        raise AssertionError('评测期间 checkpoint 文件发生变化。')
    keys = ['psnr', 'ssim', 'lpips', 'psnr_foreground', 'ssim_foreground_bbox', 'lpips_foreground_bbox']
    report = dict(model=str(a.model), iteration=a.iteration, split=a.split, source=a.source, seed=a.seed,
                  samples=a.samples, light_t_min=a.bias, direction_source=a.directions or None,
                  wo_indirect=bool(getattr(run.args, 'wo_indirect', False)),
                  metrics=aggregate(rows, keys), frames=rows, lpips_note=lpips_note,
                  total_seconds=time.time() - started, checkpoint_unchanged=True,
                  torch_max_memory_allocated_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                  protocol='400x400黑底全浮点sRGB，逐帧计算后均值；SSIM11x11 sigma1.5；PSNR以float64并由NumPy独立复算；'
                           'LPIPS(vgg)权重不可用时N/A；前景指标为遮罩后包围框')
    (a.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['metrics'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
