#!/usr/bin/env python3
"""独立解析Gaussian数值探针：只验证遮挡XYZ梯度，不训练任何模型。"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--expected_gpu_uuid', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT/'output/smoke_test'):
        parser.error('数值探针输出必须位于本仓库output/smoke_test。')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        parser.error('已有结果不可覆盖，请使用新的日期计数目录。')
    os.chdir(ROOT)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    full_command = 'CUDA_VISIBLE_DEVICES='+shlex.quote(os.environ.get('CUDA_VISIBLE_DEVICES', ''))+' '+command
    (output/'command.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+full_command+'\n')
    started = time.time()
    report = {'status': 'RUNNING', 'source': '解析构造的接收点与遮挡Gaussian，不读取数据集',
              'model': None, 'iteration': None, 'training_or_optimization': False,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'command': full_command,
              'server': socket.gethostname(), 'python': sys.executable,
              'gpu_env': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid(), 'tests': {}}
    try:
        if socket.gethostname() != 'ushas' or 'lumimotion-ushas' not in sys.executable:
            raise RuntimeError('探针要求ushas与lumimotion-ushas环境。')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != args.expected_gpu_uuid:
            raise RuntimeError('必须显式用约定GPU UUID绑定单卡。')
        import numpy as np
        import torch
        import torch.nn.functional as F
        import surfel_tracer
        import surfel_tracer._C as native_extension
        from scene.gaussian_model import GaussianModel
        from utils.general_utils import build_rotation

        wrapper = Path(inspect.getfile(surfel_tracer.GaussianTracer))
        repository_wrapper = ROOT/'submodules/surfel_tracer/surfel_tracer/raytracer.py'
        if sha(wrapper) != sha(repository_wrapper):
            raise RuntimeError('实际trace Python wrapper与仓库源码不同，拒绝混用。')
        report['provenance'] = {
            'wrapper_path': str(wrapper), 'wrapper_sha256': sha(wrapper),
            'repository_wrapper_sha256': sha(repository_wrapper),
            'extension_path': native_extension.__file__, 'extension_sha256': sha(native_extension.__file__),
            'extension_scope': '当前正式训练同一环境实际加载的OptiX扩展；未重新编译，不宣称独立重建源码对应',
            'gaussian_model_sha256': sha(ROOT/'scene/gaussian_model.py'),
            'probe_sha256': sha(Path(__file__)), 'torch': torch.__version__, 'cuda': torch.version.cuda}
        torch.set_num_threads(1)
        torch.manual_seed(1011)
        pc = GaussianModel(0, no_binary_separation=True)
        torch.cuda.synchronize()
        gpu_query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True).stdout
        own = [line.strip() for line in gpu_query.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        if len(own) != 1 or args.expected_gpu_uuid not in own[0]:
            raise RuntimeError('实际GPU进程归属核验失败：'+repr(own))
        report['gpu_runtime'] = own

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device='cuda', requires_grad=True)

        def evaluate(center=(.42, -.04, 1.), opacity_value=.7, bias=.03, routed=True,
                     raw_trace=False, backward=True):
            # 输入方向略倾斜，使单射线对xyz三个坐标都有响应；沿射线的组合方向仍是零空间。
            params = dict(xyz=tensor([center]), ray_origin=tensor([[0., 0., 0.]]),
                          ray_direction=tensor([[.2, -.1, 1.]]),
                          occluder_rotation=tensor([[1., 0., 0., 0.]]),
                          scales=tensor([[.25, .20]]), opacity=tensor([[opacity_value]]),
                          receiver_rotation=tensor([[1., 0., 0., 0.]]),
                          material=tensor([.7]), intensity=tensor([1.3]))
            pc._xyz = params['xyz']
            pc._rotation = params['occluder_rotation']
            pc._scaling = params['scales'].log()
            pc._opacity = torch.logit(params['opacity'])
            pc._albedo_dc = torch.zeros(1, 1, 3, device='cuda')
            pc._albedo_rest = torch.zeros(1, 0, 3, device='cuda')
            # BVH索引构建离散，有限差分每次也以当前参数重建；不缓存不同扰动的包围盒。
            with torch.no_grad():
                pc.build_bvh(0., 0., 0.)
            used = {name: value.detach() if routed and name != 'xyz' else value
                    for name, value in params.items()}
            direction = F.normalize(used['ray_direction'], dim=-1)
            origin = used['ray_origin']+direction*bias
            q = F.normalize(used['occluder_rotation'], dim=-1)
            if raw_trace:
                rotation = build_rotation(q)
                ru = rotation[:, :, 0]/used['scales'][:, :1]
                rv = rotation[:, :, 1]/used['scales'][:, 1:2]
                alpha = pc.gaussian_tracer.trace(origin, direction, used['xyz'], used['opacity'], ru, rv,
                    rotation[:, :, 2], torch.zeros(1, 4, device='cuda'), pc.get_features,
                    pc.alpha_min, deg=0, back_culling=False)[4]
            else:
                alpha = pc.trace(origin, direction, features=torch.zeros(1, 4, device='cuda'),
                    xyz=used['xyz'], scales=used['scales'], rotation=q, opacity=used['opacity'],
                    use_zeros=True)['alpha']
            visibility = 1-alpha
            receiver_normal = build_rotation(F.normalize(used['receiver_rotation'], dim=-1))[:, :, 2]
            cosine = (receiver_normal*direction).sum(-1).clamp_min(0)
            response = (used['material']*used['intensity']*cosine*visibility).sum()
            gradients = None
            if backward:
                response.backward()
                gradients = {name: None if value.grad is None else value.grad.detach().cpu().tolist()
                             for name, value in params.items()}
            return {'response': float(response.detach()), 'visibility': float(visibility.detach().item()),
                    'alpha': float(alpha.detach().item()), 'gradients': gradients}

        smooth = evaluate()
        forbidden = {key: val for key, val in smooth['gradients'].items() if key != 'xyz'}
        forbidden_ok = all(value is None or np.count_nonzero(value) == 0 for value in forbidden.values())
        automatic = np.array(smooth['gradients']['xyz'][0], dtype=np.float64)
        # 单个轴向surfel的闭式表达独立于CUDA实现：交点为(.2*z, -.1*z, z)。
        center = np.array([.42, -.04, 1.], dtype=np.float64)
        uv = np.array([center[0]-.2*center[2], center[1]+.1*center[2]])
        sigma = np.array([.25, .20])
        alpha_expected = .7*np.exp(-.5*np.sum((uv/sigma)**2))
        factor = .7*1.3/np.sqrt(1.05)
        grad_xy = factor*alpha_expected*uv/sigma**2
        analytic = np.array([grad_xy[0], grad_xy[1], -.2*grad_xy[0]+.1*grad_xy[1]])
        differences = []
        for step in (.002, .001, .0005):
            gradient = []
            for axis in range(3):
                plus, minus = center.copy(), center.copy()
                plus[axis] += step
                minus[axis] -= step
                p = evaluate(tuple(plus), backward=False)['response']
                m = evaluate(tuple(minus), backward=False)['response']
                gradient.append((p-m)/(2*step))
            gradient = np.asarray(gradient)
            differences.append({'epsilon': step, 'gradient': gradient.tolist(),
                                'relative_l2_error': float(np.linalg.norm(gradient-automatic)/np.linalg.norm(automatic)),
                                'max_absolute_error': float(np.max(np.abs(gradient-automatic)))})
        analytic_error = float(np.linalg.norm(analytic-automatic)/np.linalg.norm(analytic))
        smooth_ok = bool(forbidden_ok and np.isfinite(automatic).all() and np.linalg.norm(automatic) > 0
                         and abs(smooth['alpha']-alpha_expected) < 2e-5 and analytic_error < .01
                         and all(row['relative_l2_error'] < .01 for row in differences))
        report['tests']['smooth_xyz_gradient'] = dict(status='PASS' if smooth_ok else 'FAILED',
            center=center.tolist(), ray_direction=[.2, -.1, 1.], scales=sigma.tolist(), opacity=.7,
            actual=smooth, forbidden_gradients_zero_or_none=forbidden_ok,
            expected_alpha=float(alpha_expected), analytic_gradient=analytic.tolist(),
            analytic_relative_error=analytic_error, finite_differences=differences,
            tolerance={'relative_gradient_l2': .01, 'absolute_alpha': 2e-5})

        # 负对照移除路由：证明输入实际参与计算，而非因输入从未使用而出现None。
        full = evaluate(routed=False)
        negative_control_ok = all(value is not None and np.isfinite(value).all() and np.linalg.norm(value) > 0
                                  for value in full['gradients'].values())
        report['tests']['unrouted_negative_control'] = dict(status='PASS' if negative_control_ok else 'FAILED', **full)

        movements = []
        for dx in (-.08, 0., .08):
            moved = center.copy(); moved[0] += dx
            movements.append({'dx': dx, **evaluate(tuple(moved), backward=False)})
        motion_ok = movements[0]['visibility'] < movements[1]['visibility'] < movements[2]['visibility']
        report['tests']['occluder_motion'] = {'status': 'PASS' if motion_ok else 'FAILED', 'measurements': movements}

        # pc.trace在alpha>=.97时输出1，真实扩展alpha与包装后的梯度分别记录。
        saturated_center = (.204, -.1, 1.)
        wrapped = evaluate(saturated_center, opacity_value=.99)
        raw = evaluate(saturated_center, opacity_value=.99, raw_trace=True)
        saturated_ok = (wrapped['alpha'] == 1. and np.count_nonzero(wrapped['gradients']['xyz']) == 0
                        and raw['alpha'] >= .97 and np.linalg.norm(raw['gradients']['xyz']) > 0)
        report['tests']['saturation_limit'] = dict(status='PASS' if saturated_ok else 'FAILED', wrapped=wrapped, raw=raw,
            conclusion='包装器透射率饱和区域的XYZ梯度为零；不能把全部阴影内部视为有效形状约束。')

        # 一个真实近邻遮挡物位于接收点前0.02m；过大的bias会直接越过它。
        near_center = (.034, -.002, .02)
        biases = [{'bias': bias, **evaluate(near_center, bias=bias, backward=False)} for bias in (.001, .03)]
        bias_ok = biases[0]['visibility'] < .5 and biases[1]['visibility'] > .999
        report['tests']['bias_limit'] = dict(status='PASS' if bias_ok else 'FAILED', measurements=biases,
            conclusion='bias可绕过距离接收面很近的真实遮挡；本探针未证明实际场景0.03m最佳，也未验证完整自遮挡消除。')
        torch.cuda.synchronize()
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['limitations'] = [
            '只验证单个Gaussian平滑衰减区的局部导数，不包含相机Alpha合成、depth、增密或共享MLP旁路。',
            '同一单射线不能约束沿射线的遮挡物平移，需多方向与其他几何约束。',
            'BVH相交集合、alpha_min截断、0.99单点上限和0.03透射率早停处不保证光滑有限差分。',
            '透明度、旋转、尺度在本路由中冻结，不表示这些物理参数普遍不应由阴影约束。',
            '没有读取任何训练/测试图像、GT几何或光强；没有优化器、训练迭代、checkpoint或MAE。']
        report['status'] = 'PASS' if all(item['status'] == 'PASS' for item in report['tests'].values()) else 'FAILED'
    except Exception:
        report['status'] = 'FAILED'
        report['exception'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['duration_seconds'] = time.time()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output/'result.json', report)
        text = ('# 静态DL原生Trace遮挡XYZ数值验证\n\n'
                '本目录为解析数值单元测试，不是新训练实验。source为解析接收点和Gaussian；model/iteration为不适用。\n\n'
                '完整命令见`command.sh`；输出为`result.json`；日志为`probe.log`。原始失败结果禁止覆盖。\n\n'
                '不生成图像或视频，RGB／Alpha／Normal／分离四类目检均不适用；数值结果不能代替正式全15帧MAE验收。\n\n'
                '定量梯度、有限差分、可见度变化、饱和和bias边界见result.json。\n\n'
                f'最终结论：**{report["status"]}**。仅代表数值探针判定，不代表Goal达标。\n')
        (output/'README.md').write_text(text)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
