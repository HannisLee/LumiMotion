#!/usr/bin/env python3
"""解析小场景的完整光栅梯度审计：不读数据集、不优化、不产生训练结果。"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.static_dl_shadow_split import render_split_photometric, geometry_regularization, require_native_backend
from scene.static_dl_photometric import render_photometric, StaticVisibility, robust_photometric_loss
from utils.sh_utils import RGB2SH

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--expected_gpu_uuid', required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT/'output/smoke_test') or (output/'result.json').exists():
        parser.error('只允许新的 output/smoke_test 结果目录，失败不能覆盖。')
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'training': False, 'dataset_images_read': False, 'tests': {}}
    try:
        if socket.gethostname() != 'ushas' or 'lumimotion-ushas' not in sys.executable:
            raise RuntimeError('必须使用 ushas/lumimotion-ushas。')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != args.expected_gpu_uuid:
            raise RuntimeError('必须用约定的单卡 UUID。')
        provenance = require_native_backend()
        report['rasterizer_provenance'] = {k:v for k,v in provenance.items() if k != 'source_files'}
        from argparse import Namespace
        torch.set_num_threads(1)
        torch.manual_seed(1011)
        pc = GaussianModel(0, no_binary_separation=True)
        def leaf(value):
            return torch.tensor(value, device='cuda', dtype=torch.float32, requires_grad=True)
        xyz = [[x, y, 2.+.07*x-.04*y] for y in (-.5, 0., .5) for x in (-.5, 0., .5)]
        xyz.append([.23, .02, 1.25])
        pc._xyz = leaf(xyz)
        pc._rotation = leaf([[1., .05, -.07, .03]]*9+[[1., -.10, .06, 0.]])
        pc._scaling = leaf(np.log([[.34, .31]]*9+[[.31, .27]]))
        pc._opacity = leaf(np.log(np.array([[.85]]*9+[[.70]])/(1-np.array([[.85]]*9+[[.70]]))))
        pc._albedo_dc = leaf(RGB2SH(np.array([[[.65, .55, .45]]]*10)))
        pc._albedo_rest = torch.zeros(10, 0, 3, device='cuda')
        log_intensity = leaf([np.log(3.5)])
        directions = leaf([[-.2, .1, -1.]])
        directions.data.copy_(F.normalize(directions.data, dim=-1))
        receivers = pc._xyz.detach().clone().requires_grad_(True)
        parameters = dict(xyz=pc._xyz, rotation=pc._rotation, scale=pc._scaling,
                          opacity=pc._opacity, material=pc._albedo_dc, light=log_intensity,
                          direction=directions, receiver_origin=receivers)
        camera = Camera(0, np.eye(3), np.zeros(3), 1.05, 1.05,
                        torch.zeros(3, 64, 64), torch.ones(1, 64, 64), 'analytic', '', 0, fid=0.)
        pipe = Namespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=1., white_background=False)
        background = torch.zeros(3, device='cuda')
        target = torch.full((3, 64, 64), .25, device='cuda')
        mask = torch.ones(1, 64, 64, device='cuda')
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                              '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True).stdout
        own = [s for s in gpu.splitlines() if s.split(',')[0].strip() == str(os.getpid())]
        if len(own) != 1 or args.expected_gpu_uuid not in own[0]:
            raise RuntimeError('实际 GPU 归属与请求不同。')
        report['gpu_runtime'] = own

        def gradient(loss):
            values = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
            arrays = {k: None if v is None else v.detach().cpu().numpy()
                      for k, v in zip(parameters, values)}
            summary = {k: {'is_none': v is None, 'norm': 0. if v is None else float(np.linalg.norm(v)),
                           'max_abs': 0. if v is None else float(np.abs(v).max()),
                           'finite': v is None or bool(np.isfinite(v).all())} for k, v in arrays.items()}
            return arrays, summary

        def render(route, detach_camera=True):
            return render_split_photometric(camera, pc, pipe, background, directions,
                                            log_intensity.exp(), .03, route=route,
                                            receiver_positions=receivers, detach_camera=detach_camera)

        def photo(package):
            return robust_photometric_loss(package['render'], target, mask)[0]

        def clean(summary, keys):
            return all(summary[k]['norm'] == 0 and summary[k]['finite'] for k in keys)

        def nonzero(summary, keys):
            return all(summary[k]['norm'] > 0 and summary[k]['finite'] for k in keys)

        def difference(a, b):
            result = {}
            for k in parameters:
                va = np.zeros(tuple(parameters[k].shape)) if a[k] is None else a[k]
                vb = np.zeros(tuple(parameters[k].shape)) if b[k] is None else b[k]
                result[k] = float(np.max(np.abs(va-vb)))
            return result

        packages = {route: render(route) for route in ('control', 'split', 'occluder')}
        forward_diff = {k: float((packages['control'][k]-packages['split'][k]).abs().max())
                        for k in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth')}
        gs = {route: gradient(photo(package)) for route, package in packages.items()}
        control, split, shadow = (gs[k][0] for k in ('control', 'split', 'occluder'))
        cs, ss, os_ = (gs[k][1] for k in ('control', 'split', 'occluder'))
        common_diff = difference(control, split)
        expected_xyz = (np.zeros_like(split['xyz']) if control['xyz'] is None else control['xyz'])+shadow['xyz']
        xyz_error = float(np.max(np.abs(split['xyz']-expected_xyz)))
        xyz_relative = float(np.linalg.norm(split['xyz']-expected_xyz)/max(np.linalg.norm(expected_xyz), 1e-20))
        checks = [all(v == 0 for v in forward_diff.values()),
                  clean(cs, ('xyz', 'scale', 'opacity', 'direction', 'receiver_origin')),
                  nonzero(cs, ('rotation', 'material', 'light')),
                  clean(os_, ('rotation', 'scale', 'opacity', 'material', 'light', 'direction', 'receiver_origin')),
                  nonzero(os_, ('xyz',)),
                  all(common_diff[k] <= 1e-7 for k in ('rotation', 'material', 'light')),
                  xyz_relative <= 1e-5, all(v['finite'] for v in ss.values())]
        report['tests']['route_identity'] = dict(status='PASS' if all(checks) else 'FAILED',
            forward_max_abs_difference=forward_diff, gradients={k:v[1] for k,v in gs.items()},
            control_split_gradient_difference=common_diff, split_xyz_equals_control_plus_shadow_max_error=xyz_error,
            split_xyz_equals_control_plus_shadow_relative_error=xyz_relative)

        # 原生相机 Alpha/depth/normal 合成与 override_color 链分开审计。
        frozen_camera = render('split')
        alpha_maps = frozen_camera['rend_alpha'].mean()+frozen_camera['surf_depth'].mean()+frozen_camera['rend_normal'].square().mean()
        _, map_summary = gradient(alpha_maps)
        negative = render('control', detach_camera=False)
        _, neg_summary = gradient(photo(negative))
        report['tests']['camera_bypass'] = dict(status='PASS' if clean(map_summary, parameters) and
            nonzero(neg_summary, ('xyz', 'rotation')) else 'FAILED', detached_maps=map_summary,
            camera_xyz_q_enabled_negative_control=neg_summary,
            screen_proxy='原生独立屏幕代理可能有导数；新训练入口不优化它且永不收集增密统计。')

        terms, _ = geometry_regularization(camera, pc, pipe, background, mask)
        geometry_loss = .1*terms['normal']+1000*terms['distortion']+.001*terms['alpha']
        _, geo_summary = gradient(geometry_loss)
        report['tests']['real_geometry_regularizer'] = dict(status='PASS' if
            nonzero(geo_summary, ('xyz', 'rotation')) and clean(geo_summary, ('scale', 'opacity', 'material', 'light', 'direction', 'receiver_origin'))
            else 'FAILED', values={k:float(v.detach()) for k,v in terms.items()}, gradients=geo_summary)
        totals = {}
        for label, route in (('control', 'control'), ('split', 'split'), ('repeat_control', 'control')):
            package = render(route)
            terms, _ = geometry_regularization(camera, pc, pipe, background, mask)
            loss = photo(package)+.1*terms['normal']+1000*terms['distortion']+.001*terms['alpha']
            totals[label] = gradient(loss)[0]
        total_difference = difference(totals['control'], totals['split'])
        repeat_difference = difference(totals['control'], totals['repeat_control'])
        relative_common = {k:float(np.linalg.norm(totals['control'][k]-totals['split'][k])/
                                  max(np.linalg.norm(totals['control'][k]), 1e-20))
                           for k in ('rotation', 'material', 'light')}
        total_xyz_delta = totals['split']['xyz']-totals['control']['xyz']
        total_error = float(np.linalg.norm(total_xyz_delta-shadow['xyz'])/np.linalg.norm(shadow['xyz']))
        report['tests']['one_photo_loss_with_common_geometry'] = dict(status='PASS' if
            all(total_difference[k] <= 2e-6 and relative_common[k] <= 1e-5
                for k in ('rotation', 'material', 'light')) and total_error < 1e-4
            else 'FAILED', common_gradient_max_abs_difference=total_difference,
            common_gradient_relative_l2_difference=relative_common,
            identical_control_repeat_max_abs_difference=repeat_difference,
            tolerance={'common_max_abs':2e-6, 'common_relative_l2':1e-5, 'xyz_relative_l2':1e-4},
            xyz_difference_equals_shadow_relative_error=total_error,
            note='原生CUDA浮点归约不要求重复backward逐位一致；另记同一control重复的数值差。')

        class Lights:
            def for_camera(self, _camera):
                return directions, log_intensity.exp()
        old = render_photometric(camera, pc, pipe, background, Lights(), StaticVisibility(.03, 1), 1)
        current = render('split')
        parity = {k: float((old[k]-current[k]).abs().max().detach())
                  for k in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth')}
        report['tests']['existing_forward_parity'] = dict(status='PASS' if max(parity.values()) < 1e-6 else 'FAILED', max_abs=parity)
        # 验证正式入口实际调用的审计helper；此处仍只使用解析张量，不加载checkpoint。
        from scene.static_dl_shadow_split_audit import audit_checkpoint_routes
        camera.original_image_train_light.fill_(.25)
        audit_lights = Lights()
        audit_lights.log_intensity = log_intensity
        settings = Namespace(shadow_bias=.03, shadow_weight=.25, highlight_weight=.5,
                             residual_threshold=.08, normal_weight=.1, dist_weight=1000., alpha_weight=.001)
        report['tests']['checkpoint_audit_helper_integration'] = audit_checkpoint_routes(
            camera, pc, pipe, background, audit_lights, settings)
        report['visibility'] = {'min':float(current['light_visibility'].min()),
                                'mean':float(current['light_visibility'].mean()),
                                'max':float(current['light_visibility'].max())}
        report['sources'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), ROOT/'scene/static_dl_shadow_split.py', ROOT/'scene/static_dl_shadow_split_audit.py',
             ROOT/'scene/static_dl_photometric.py',
             ROOT/'scene/gaussian_model.py', ROOT/'gaussian_renderer/__init__.py')}
        report['limitations'] = ['解析小场景局部梯度审计，无训练、模型优化或 MAE；正式checkpoint仍需相同审计。',
            '沿用 .97 饱和与 bias 限制；BVH命中边界不可用本局部审计保证。',
            '本例无 deform/共享网络，正式入口冻结并复制零几何头，不调用其前向。',
            '形式上阻断RGB相机旁路不保证优化有效，真正结果仍由全15帧MAE与覆盖率判定。']
        report['status'] = 'PASS' if all(t['status']=='PASS' for t in report['tests'].values()) else 'FAILED'
    except Exception:
        report['status'] = 'FAILED'
        report['exception'] = traceback.format_exc()
        traceback.print_exc()
    report['duration_seconds'] = time.time()-started
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
