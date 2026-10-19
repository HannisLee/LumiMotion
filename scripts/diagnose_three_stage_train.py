"""固定135TRAIN完整前向，核验机制、亮度分段及相邻保存点的变化。"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scene.three_stage_pipeline import ThreeStageModel, sha, write
from scene.three_stage_lighting import brightness_weights, censored_error
from scene.stage2_photometric import srgb_to_linear
from scripts.train_three_stage import load_cameras, install_guard, preview


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)), p95=float(np.quantile(values, .95)), max=float(values.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iterations', type=int, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(os.environ.get('THREE_STAGE_REPO_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
    a.model, a.output = a.model.resolve(), a.output.resolve()
    if not a.output.is_relative_to(root/'output'):
        raise ValueError('诊断输出必须位于output。')
    a.output.mkdir(parents=True, exist_ok=False)
    marker = json.loads((a.model/'three_stage_training.json').read_text())
    if marker['smoke']:
        if 'smoke_test' not in a.output.parts:
            raise ValueError('冒烟模型只能在smoke_test诊断。')
    binding = marker['binding']; source = Path(binding['source'])
    if marker['gt_supervision'] is not False or marker['train_frames'] != [n for n in range(1, 151) if n % 10]:
        raise ValueError('必须本轮135TRAIN模型。')
    manifest = json.loads((source/'no_gt_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(source/name) != digest:
            raise ValueError('TRAIN源SHA变化。')
    install_guard(source, a.output, a.model, binding)
    cameras = load_cameras(source)
    endpoints = []
    first_geometry = None
    with torch.no_grad():
        for iteration in a.iterations:
            model = ThreeStageModel.load(a.model, iteration)
            geometry = {k: getattr(model.pc, k).detach().cpu().clone() for k in ('_xyz', '_rotation', '_scaling', '_opacity')}
            if first_geometry is None:
                first_geometry = geometry
            xyz_delta = (geometry['_xyz']-first_geometry['_xyz']).norm(dim=-1).numpy()
            initial, direction = model.lights.initial.double(), model.lights.directions().double()
            angle = torch.rad2deg(torch.atan2(torch.linalg.cross(initial, direction, dim=-1).norm(dim=-1),
                                            (initial*direction).sum(-1))).cpu().numpy()
            rows = []
            for camera in cameras:
                camera.load2device()
                package = model.render(camera, iteration, components=True)
                if float(package['component_add_max_error']) >= 2e-5:
                    raise AssertionError('分量加和失败。')
                target = camera.original_image_train_light.cuda()
                alpha = camera.gt_alpha_mask.cuda()
                observed = srgb_to_linear(target)
                y, _, _ = brightness_weights(observed, model.stats)
                fg = alpha > .5
                residual = package['render_linear']-observed
                censored = censored_error(package['render_linear'], observed, camera.observed_saturation_mask.cuda())
                row = dict(frame=int(camera.image_name_train_light.split('_')[-1]),
                    alpha_bce=float(F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6, 1-1e-6), alpha)),
                    psnr_srgb_black=-10*math.log10(max(float((package['render'].clamp(0, 1)-target).square().mean()), 1e-20)),
                    visibility_mean=float(package['visibility_mean']))
                for name, mask in [('foreground', fg), ('dark', fg & (y < model.stats['q50'])),
                                   ('body', fg & (y <= model.stats['q_base_value'])),
                                   ('bright', fg & (y >= model.stats['q_bright_value']))]:
                    count = int(mask.sum())
                    row[name] = dict(pixels=count, linear_observed_mae=float((residual.abs()*mask).sum()/(3*count)) if count else None,
                        linear_censored_charbonnier=float((censored*mask).sum()/(3*count)) if count else None,
                        base_mean=float((package['base_linear']*mask).sum()/(3*count)) if count else None,
                        residual_mean=float((package['residual_linear']*mask).sum()/(3*count)) if count else None)
                rows.append(row)
                if row['frame'] in (1, 75, 149):
                    preview(a.output/f'iteration_{iteration}/{camera.image_name_train_light}', package)
                camera.load2device('cpu')
            summary = {k: float(np.mean([r[k] for r in rows])) for k in ('alpha_bce', 'psnr_srgb_black', 'visibility_mean')}
            for segment in ('foreground', 'dark', 'body', 'bright'):
                summary[segment] = {field: float(np.mean([r[segment][field] for r in rows if r[segment][field] is not None]))
                    for field in ('linear_observed_mae', 'linear_censored_charbonnier', 'base_mean', 'residual_mean')}
            endpoint = dict(iteration=iteration, rows=rows, summary=summary,
                xyz_drift_from_first_endpoint=summarize(xyz_delta), direction_deviation_deg=summarize(angle),
                checkpoint_sha256=sha(a.model/f'three_stage/iteration_{iteration}/state.pt'))
            endpoints.append(endpoint)
            write(a.output/f'endpoint_{iteration}.json', endpoint)
            print(json.dumps(dict(iteration=iteration, summary=summary), ensure_ascii=False), flush=True)
            del model
    changes = []
    for before, after in zip(endpoints[:-1], endpoints[1:]):
        change = dict(before=before['iteration'], after=after['iteration'])
        for segment in ('foreground', 'dark', 'body', 'bright'):
            x = before['summary'][segment]['linear_censored_charbonnier']
            y = after['summary'][segment]['linear_censored_charbonnier']
            change[segment] = dict(before=x, after=y, relative_change=(y-x)/max(x, 1e-9))
        changes.append(change)
    write(a.output/'receipt.json', dict(status='PASS_TRAIN_DIAGNOSTIC_EXECUTION', model=str(a.model),
        source=str(source), source_manifest_sha256=sha(source/'no_gt_manifest.json'), iterations=a.iterations,
        changes=changes, gt_read=False, external_test_read=False,
        evaluation_scope='135已见TRAIN，固定原Alpha分母与共同阈值，无曝光拟合；不证明外部几何精度',
        convergence='待结合末段完整TRAIN变化和几何机制判读，不自动声称收敛',
        note='xyz漂移相对本命令第一个保存点；若需B00转换后的实际起点，iterations首项包含0'))


if __name__ == '__main__':
    main()
