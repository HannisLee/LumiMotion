"""旧FIT白色DL能量的有限条件凸剖面；不导出、不训练、不读取开发图像。"""
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_eldl_v3_material_bounds as b


def solve_profile(a, d, y, saturated, weights):
    """一次断点排序及后缀扫描求非负一维分段二次目标的全局最小。"""
    a, d, y, weights = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (a, d, y, weights)]
    saturated = np.asarray(saturated, dtype=bool).reshape(-1)
    b.require(all(x.shape == a.shape for x in (d, y, weights, saturated)), '剖面数组长度不同。')
    b.require(all(np.isfinite(x).all() for x in (a, d, y, weights)) and (d >= 0).all() and (weights > 0).all(), '非有限/负DL基底/分母无效。')
    unsat = ~saturated
    denominator = np.sum(weights[unsat] * d[unsat] ** 2)
    numerator = np.sum(weights[unsat] * d[unsat] * (y[unsat] - a[unsat]))
    active = saturated & (d > 0) & (y > a)
    thresholds = (y[active] - a[active]) / d[active]
    order = np.argsort(thresholds, kind='stable')
    thresholds = thresholds[order]
    quadratic = (weights[active] * d[active] ** 2)[order]
    linear = (weights[active] * d[active] * (y[active] - a[active]))[order]
    den = denominator + np.r_[np.cumsum(quadratic[::-1])[::-1], 0.]
    num = numerator + np.r_[np.cumsum(linear[::-1])[::-1], 0.]
    low, high = np.r_[0., thresholds], np.r_[thresholds, np.inf]
    if num[0] < 0:
        value, interval, unique = 0., [0., 0.], True
    else:
        flat = (den == 0) & (num == 0)
        if flat.any():
            index = int(np.flatnonzero(flat)[0])
            left, right = float(low[index]), float(high[index])
            value = max(left, min(1., right))
            interval, unique = [left, right if np.isfinite(right) else None], False
        else:
            roots = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
            tolerance = 1e-12 * np.maximum(1., np.abs(roots))
            valid = (den > 0) & (roots >= low - tolerance) & (roots <= high + tolerance)
            b.require(valid.any(), '断点扫描未找到有限最优解，停止。')
            index = int(np.flatnonzero(valid)[0])
            value = float(np.clip(roots[index], low[index], high[index]))
            interval, unique = [value, value], True
    residual = np.where(saturated, np.minimum(a + value * d - y, 0), a + value * d - y)
    derivative = float(2 * np.sum(weights * d * residual))
    scale = float(2 * np.sum(weights * np.abs(d * residual)))
    tolerance = 1e-10 + 1e-8 * scale
    b.require(np.isfinite(value) and value >= 0 and (derivative >= -tolerance if value == 0 else abs(derivative) <= tolerance), '剖面KKT核验失败。')
    loss = float(np.sum(weights * residual ** 2))
    original = np.where(saturated, np.minimum(a + d - y, 0), a + d - y)
    baseline = float(np.sum(weights * original ** 2))
    b.require(loss <= baseline + 1e-10, '全局剖面差于原a=1，停止。')
    return dict(coefficient=value, minimizer_interval=interval, unique=unique,
                loss=loss, baseline=baseline, derivative=derivative,
                active_saturation_breakpoints=len(thresholds), nonnegative_boundary=value == 0.)


def score(a, d, target, saturated, value):
    residual = a + value * d - target
    residual = np.where(saturated, np.minimum(residual, 0), residual)
    return float(np.mean(residual ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    b.require(out.is_relative_to(b.CONTROL) and not out.exists(), '只允许总控新输出，不覆盖。')
    b.require(b.socket.gethostname() == 'ushas' and Path(sys.executable).resolve() == Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve(), '需要ushas对应环境。')
    cfg, ck, fits, hashes = b.inputs(out)
    hashes[str(Path(__file__).resolve())] = b.sha(Path(__file__).resolve())
    hashes[str(Path(b.__file__))] = b.sha(Path(b.__file__))
    protocol = ROOT / 'LIHAN/DOC/报告/1019-57-材质边界分支结束与白色DL能量条件剖面建议.md'
    hashes[str(protocol)] = b.sha(protocol)
    out.mkdir()
    b.CREATED_OUTPUT = out
    b.write(out / 'input_binding.json', dict(input_sha256=hashes, scope='旧FIT来源，固定白DL幅度条件剖面，禁止导出。'))
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from utils.general_utils import build_rotation
    from scripts.probe_dynamic_dl_selfcal import read_config
    from scripts.calibrate_eldl_v3_raster import render
    from scene.static_dl_photometric import StaticVisibility
    from scene.pbr_rasterizer import native_module
    from scene.stage2_photometric import srgb_to_linear
    import diff_surfel_rasterization as backend
    import surfel_tracer.raytracer as tracer_module
    actual = native_module().provenance
    b.require(all(actual[k] == cfg['backend_provenance'][k] for k in ('source_sha256', 'extension_sha256', 'extension_path')), '原生来源不符。')
    b.require(backend.provenance['source_sha256'] == actual['source_sha256'] and b.sha(Path(backend._C.__file__)) == actual['extension_sha256'], '实际光栅alias不同。')
    tracer = Path(tracer_module._C.__file__).resolve()
    b.require(b.sha(tracer) == '5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e', '实际tracer不同。')
    hashes[str(tracer)] = b.sha(tracer)
    hashes[actual['extension_path']] = actual['extension_sha256']
    loaded = read_config(b.MODEL / 'cfg_args')
    loaded.resolution = 2
    pc = GaussianModel(loaded.sh_degree, no_binary_separation=loaded.no_binary_separation, fea_dim=loaded.hyper_dim)
    pc.load_ply(str(b.MODEL / 'point_cloud/iteration_16000/point_cloud.ply'))
    for value in vars(pc).values():
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
    geometry = {k: getattr(pc, k).clone() for k in ('_xyz', '_scaling', '_rotation', '_opacity')}
    weights = torch.load(b.MODEL / 'deform/iteration_16000/deform.pth', map_location='cpu', weights_only=True)
    b.require(all(torch.count_nonzero(weights[f'{h}.{s}']) == 0 for h in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling') for s in ('weight', 'bias')), 'I00几何头非零。')
    cameras = []
    for i, info in enumerate(readCamerasFromTransforms(str(b.SOURCE), 'transforms_train.json', 'images', False, no_bg=False)):
        b.require(int(info.image_name_train_light.split('_')[-1]) == b.FIT[i], '相机次序不符。')
        uid = b.TRAIN.index(b.FIT[i])
        cameras.append(loadCam(loaded, uid, info._replace(uid=uid), 1.))
    b.require(len(cameras) == 117, 'FIT不完整。')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.)
    rho, environment, directions = ck['rho'].cuda(), ck['log_e'].cuda().exp(), ck['directions'].cuda()
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    fixed = torch.load(b.PARENT / 'fixed_foreground.pt', map_location='cpu', weights_only=False)
    b.require(b.sha(b.PARENT / 'fixed_foreground.pt') == cfg['foreground_sha256'] and fixed['frames'] == b.TRAIN, '固定mask来源不同。')
    cache_path = b.L00 / 'raster_forward_cache_v1/raster_cache.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    meta_path = cache_path.parent / 'summary.json'
    b.require(b.sha(cache_path) == b.read(meta_path)['cache_sha256'] and torch.equal(cache['directions'], ck['directions']), 'q缓存不符。')
    for path in (cache_path, meta_path, b.PARENT / 'fixed_foreground.pt'):
        hashes[str(path)] = b.sha(path)
    bases, rows, count = {}, [], 0
    traces = {scale: StaticVisibility(cache['bias'] * scale, 1000000) for scale in b.SCALES}

    def q_for(camera, frame, scale):
        index = b.TRAIN.index(frame)
        light = directions[index:index + 1]
        vis = traces[scale](pc, light, 0)
        facing = torch.where((normal * (camera.camera_center - pc.get_xyz)).sum(-1, keepdim=True) >= 0, 1., -1.)
        q = ((normal * facing) @ light.T).clamp_min(0) * vis
        b.require(float((q.cpu() - cache['q'][cache['scales'].index(scale), index]).abs().max()) < 2e-6, '实际q与原缓存不同。')
        return q

    with torch.no_grad():
        for scale in b.SCALES:
            for i, (frame, camera) in enumerate(zip(b.FIT, cameras)):
                camera.load2device()
                q = q_for(camera, frame, scale)
                total = render(camera, pc, pipe, rho * (environment / math.pi + q))['render']
                direct = render(camera, pc, pipe, rho * q)['render']
                ambient = total - direct
                count += 2
                mask = (camera.gt_alpha_mask.cuda() > .5).expand_as(total)
                b.require(torch.equal(mask[:1].cpu(), fixed['foreground'][b.TRAIN.index(frame)]), '原mask分母不同。')
                b.require(torch.isfinite(total).all() and torch.isfinite(direct).all() and float(direct.min()) >= 0 and float(ambient.min()) >= -2e-5, '线性基底无效。')
                target = srgb_to_linear(camera.original_image_train_light.cuda())
                saturated = camera.observed_saturation_mask.cuda().bool()
                a, d, y = [x[mask].cpu().numpy().astype(np.float64) for x in (ambient, direct, target)]
                s = saturated[mask].cpu().numpy()
                inner = (camera.gt_alpha_mask.cuda() > .99).expand_as(total)[mask].cpu().numpy()
                baseline = score(a, d, y, s, 1.)
                if scale == 1.:
                    b.require(abs(baseline - fits['per_frame'][i]) < 1e-7, '中心逐帧目标复现失败。')
                bases[(scale, frame)] = dict(a=a, d=d, y=y, s=s, inner=inner)
                if frame in (1, 81, 149):
                    bases[(scale, frame)]['full'] = (ambient.cpu(), direct.cpu())
                row = dict(scale=scale, frame=frame, baseline=baseline, foreground_channels=len(a))
                rows.append(row)
                b.append(out / 'base_progress.jsonl', row)
                camera.load2device('cpu')
            print('basis', scale, 'complete', flush=True)
    solutions = {}
    shared_arrays = [[] for _ in range(5)]
    for frame in b.FIT:
        base = bases[(1., frame)]
        arrays = [base[k] for k in ('a', 'd', 'y', 's')]
        weight = np.full_like(base['a'], 1. / len(base['a']))
        solutions[str(frame)] = solve_profile(*arrays, weight)
        b.append(out / 'profile_progress.jsonl', dict(frame=frame, **solutions[str(frame)]))
        for bucket, value in zip(shared_arrays, [*arrays, weight / 117]):
            bucket.append(value)
    shared = solve_profile(*[np.concatenate(values) for values in shared_arrays])
    del shared_arrays
    b.write(out / 'conditional_solutions.json', dict(shared=shared, per_frame=solutions, export_permitted=False))
    results = []
    for scale in b.SCALES:
        for frame in b.FIT:
            base = bases[(scale, frame)]
            for case, value in [('baseline', 1.), ('shared', shared['coefficient']), ('per_frame_oracle', solutions[str(frame)]['coefficient'])]:
                residual = base['a'] + value * base['d'] - base['y']
                residual = np.where(base['s'], np.minimum(residual, 0), residual)
                parts = {name: float(np.sum(residual[mask] ** 2) / len(residual)) for name, mask in [('saturated', base['s']), ('unsaturated', ~base['s']), ('inner', base['inner']), ('edge', ~base['inner'])]}
                results.append(dict(scale=scale, frame=frame, case=case, coefficient=value, loss=float(np.mean(residual ** 2)), loss_parts=parts, foreground_channels=len(base['a'])))
    b.write(out / 'per_frame.json', results)
    parity = []
    with torch.no_grad():
        for scale in b.SCALES:
            for frame in (1, 81, 149):
                camera = cameras[b.FIT.index(frame)]
                camera.load2device()
                q = q_for(camera, frame, scale)
                ambient, direct = bases[(scale, frame)]['full']
                for case, value in [('shared', shared['coefficient']), ('per_frame_oracle', solutions[str(frame)]['coefficient'])]:
                    actual_image = render(camera, pc, pipe, rho * (environment / math.pi + value * q))['render']
                    predicted = ambient.cuda() + value * direct.cuda()
                    error = float((actual_image - predicted).abs().max())
                    count += 1
                    row = dict(scale=scale, frame=frame, case=case, coefficient=value, max_abs_error=error)
                    parity.append(row)
                    b.append(out / 'parity_progress.jsonl', row)
                    b.require(math.isfinite(error) and error <= 2e-5, '实际native组合与基底剖面不一致，停止。')
                camera.load2device('cpu')
    b.require(count == 720, '前向预算不完整。')
    b.require(all(torch.equal(getattr(pc, k), value) for k, value in geometry.items()), '冻结几何改变。')
    for path, value in hashes.items():
        b.require(b.sha(Path(path)) == value, '输入/源码改变。')
    means = {case: {str(scale): sum(r['loss'] for r in results if r['case'] == case and r['scale'] == scale) / 117 for scale in b.SCALES} for case in ('baseline', 'shared', 'per_frame_oracle')}
    parts = {case: {str(scale): {name: sum(r['loss_parts'][name] for r in results if r['case'] == case and r['scale'] == scale) / 117 for name in ('saturated', 'unsaturated', 'inner', 'edge')} for scale in b.SCALES} for case in means}
    blocks = {case: {str(scale): {str(block): dict(frame_count=sum(1 for r in results if r['case'] == case and r['scale'] == scale and (r['frame'] - 1) // 30 == block), contribution_to_full_117_mean=sum(r['loss'] for r in results if r['case'] == case and r['scale'] == scale and (r['frame'] - 1) // 30 == block) / 117) for block in range(5)} for scale in b.SCALES} for case in means}
    b.require(abs(means['shared']['1.0'] - shared['loss']) < 1e-12, '共享系数求解和逐帧目标权重不一致。')
    benefit = 1 - means['per_frame_oracle']['1.0'] / means['baseline']['1.0']
    b.write(out / 'per_frame.json', results)
    b.write(out / 'summary.json', dict(status='PASS_CONDITIONAL_PROFILE_EXECUTION', gt_read=False, development_images_decoded=False, forward_count=count, backward_count=0, optimizer_steps=0, scalar_problems=118, input_sha256=hashes, means=means, loss_parts=parts, fixed_time_blocks=blocks, shared_solution=shared, per_frame_solutions=solutions, parity=parity, conditional_center_relative_benefit=benefit, limited_one_percent_hypothesis_nominally_falsified=benefit < .01, export_permitted=False, limitation='同帧RGB条件oracle，仅检验固定旧rho/E/L/几何的白DL幅度解释能力；不识别真实D(t)，非独立验证，不导出或自动正式训练。'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        if b.CREATED_OUTPUT is not None:
            b.write(b.CREATED_OUTPUT / 'FAILED.json', dict(status='FAILED_ENGINEERING', traceback=traceback.format_exc(), preserve_all_existing_outputs=True))
        raise
