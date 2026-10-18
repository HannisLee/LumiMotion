"""冻结旧校准状态的材质边界梯度诊断；无优化、无开发集评分。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONTROL = ROOT / 'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控'
L00 = ROOT / 'output/1019-04-LH-static-jumpingjacks-ELDL-V3-L00连续混合估光'
PARENT = L00 / 'raster_calibration_v2_anneal/init_3'
MODEL = ROOT / 'output/1019-03-LH-static-jumpingjacks-ELDL-V3-I00内部静态SH/model_mlp'
SOURCE = CONTROL / 'data_fit_v2'
TRAIN = [i for i in range(1, 151) if i % 10]
FIT = [i for i in TRAIN if not 61 <= i <= 79]
SCALES = (1., .5, 2.)
CREATED_OUTPUT = None


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def append(path, data):
    with path.open('a') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(output):
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([SOURCE, MODEL, PARENT, output,
                        L00 / 'raster_forward_cache_v1',
                        CONTROL / 'data_train_v2/no_gt_manifest.json'])
    manifest = read(SOURCE / 'no_gt_manifest.json')
    require(manifest['train_frames'] == FIT and not manifest['gt_read'], '只接受原117 FIT。')
    original = read(CONTROL / 'data_train_v2/no_gt_manifest.json')
    require(sha(CONTROL / 'data_train_v2/no_gt_manifest.json') == manifest['parent_manifest_sha256'], '父来源不同。')
    hashes = {str(SOURCE / 'no_gt_manifest.json'): sha(SOURCE / 'no_gt_manifest.json'),
              str(CONTROL / 'data_train_v2/no_gt_manifest.json'): sha(CONTROL / 'data_train_v2/no_gt_manifest.json')}
    for name, value in manifest['files'].items():
        path = (SOURCE / name).resolve()
        require(path.is_relative_to(SOURCE) and sha(path) == value, 'FIT文件SHA不同。')
        if name.startswith('images/'):
            require(original['files'][name] == value, 'FIT图像不是原校准输入。')
        hashes[str(path)] = value
    cfg = read(PARENT / 'configuration.json')
    original_manifest_path = CONTROL / 'data_train_v2/no_gt_manifest.json'
    require(sha(original_manifest_path) == cfg['input_sha256'][str(original_manifest_path)], 'FIT父manifest不属于旧校准。')
    original_camera_path = CONTROL / 'data_train_v2/transforms_train.json'
    require(sha(original_camera_path) == cfg['input_sha256'][str(original_camera_path)], '原相机来源不同。')
    original_cameras = read(original_camera_path)
    fit_cameras = read(SOURCE / 'transforms_train.json')
    expected_frames = [f for f in original_cameras['frames'] if int(Path(f['file_path']).stem.split('_')[-1]) in FIT]
    require(fit_cameras['frames'] == expected_frames, 'FIT相机/time不同于原135相机对应帧。')
    require({k: v for k, v in fit_cameras.items() if k != 'frames'} == {k: v for k, v in original_cameras.items() if k != 'frames'}, 'FIT相机全局约定不同。')
    hashes[str(original_camera_path)] = sha(original_camera_path)
    checkpoint = PARENT / 'checkpoint_epoch_30.pt'
    require(sha(checkpoint) == '470e8505f04bb19f05ee89f3badcdb9cdeaa9d6cf89914e2888fbce8f2b29866', '旧90epoch父checkpoint改变。')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(ck['epoch'] == 30 and ck['cumulative_epoch'] == 90 and not ck['constant'], '父预算/身份不同。')
    require(cfg['geometry_frozen'] and cfg['directions_frozen'] and cfg['geometry_fit_frames'] == FIT, '父冻结条件不同。')
    require(cfg['loss'] == 'linear_censored_mse_original_alpha_gt_0p5_fixed_denominator', '父目标不同。')
    marker = read(MODEL / 'eldl_v3_training.json')
    require(marker['baseline'] == 'static_sh' and marker['train_frames'] == FIT and not marker['gt_supervision'] and not marker['init_weights_loaded'], 'I00无GT来源不同。')
    for name in ('cfg_args', 'point_cloud/iteration_16000/point_cloud.ply', 'deform/iteration_16000/deform.pth'):
        path = MODEL / name
        require(sha(path) == cfg['input_sha256'][str(path)], 'I00父几何改变。')
        hashes[str(path)] = sha(path)
    for path in (checkpoint, PARENT / 'configuration.json', PARENT / 'fit_epoch_30.json',
                 MODEL / 'eldl_v3_training.json', Path(__file__).resolve(),
                 ROOT / 'scripts/calibrate_eldl_v3_raster.py', ROOT / 'gaussian_renderer/__init__.py',
                 ROOT / 'scene/static_dl_photometric.py', ROOT / 'scene/gaussian_model.py',
                 ROOT / 'scene/dataset_readers.py', ROOT / 'utils/rgba16.py',
                 ROOT / 'scene/stage2_photometric.py', ROOT / 'utils/camera_utils.py',
                 ROOT / 'scene/cameras.py'):
        hashes[str(path)] = sha(path)
    fits = read(PARENT / 'fit_epoch_30.json')
    require(len(fits['per_frame']) == 117, '旧FIT不完整。')
    return cfg, ck, fits, hashes


def scalar_parts(pred, camera):
    from scene.stage2_photometric import srgb_to_linear
    target = srgb_to_linear(camera.original_image_train_light.cuda())
    alpha = camera.gt_alpha_mask.cuda()
    mask = (alpha > .5).expand_as(target)
    saturated = camera.observed_saturation_mask.cuda().bool()
    residual = torch.where(saturated, (pred - target).clamp_max(0), pred - target)
    denom = mask.sum()
    inner = (alpha > .99).expand_as(target)
    parts = {}
    for name, sub in [('saturated', saturated), ('unsaturated', ~saturated),
                      ('inner', inner), ('edge', ~inner)]:
        parts[name] = float((residual.square() * (mask & sub)).sum() / denom)
    return residual[mask].square().mean(), parts, target, mask, saturated, residual


def main():
    global CREATED_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    require(socket.gethostname() == 'ushas' and Path(sys.executable).resolve() == Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve(), '需要ushas对应环境。')
    require(output.is_relative_to(CONTROL) and not output.exists(), '只允许总控新目录，不覆盖。')
    cfg, ck, fits, hashes = inputs(output)
    output.mkdir()
    CREATED_OUTPUT = output
    write(output / 'input_binding.json', dict(input_sha256=hashes, scope='旧父checkpoint/I00/FIT源与相机CPU绑定；fixed_mask和raster_cache在执行阶段另验。'))
    if args.cpu_only:
        write(output / 'cpu_receipt.json', dict(status='PASS_CPU_INPUT_BINDING', input_sha256=hashes, cuda_initialized=torch.cuda.is_initialized(), gpu_forward=False))
        return
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from utils.general_utils import build_rotation
    from scripts.probe_dynamic_dl_selfcal import read_config
    from scripts.calibrate_eldl_v3_raster import render
    from scene.static_dl_photometric import StaticVisibility
    from scene.pbr_rasterizer import native_module
    backend = native_module().provenance
    require(all(backend[k] == cfg['backend_provenance'][k] for k in ('source_sha256', 'extension_sha256', 'extension_path')), 'native来源改变。')
    import diff_surfel_rasterization as actual_backend
    require(hasattr(actual_backend, 'provenance') and actual_backend.provenance['source_sha256'] == backend['source_sha256'], '实际光栅alias并非已验原生实现，必须通过bootstrap。')
    require(sha(Path(actual_backend._C.__file__).resolve()) == backend['extension_sha256'], '实际光栅alias二进制不同。')
    import surfel_tracer.raytracer as tracer_module
    tracer = Path(tracer_module._C.__file__).resolve()
    require(sha(tracer) == '5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e', 'BVH二进制改变。')
    hashes[str(tracer)] = sha(tracer)
    hashes[backend['extension_path']] = backend['extension_sha256']
    loaded = read_config(MODEL / 'cfg_args')
    loaded.resolution = 2
    pc = GaussianModel(loaded.sh_degree, no_binary_separation=loaded.no_binary_separation, fea_dim=loaded.hyper_dim)
    pc.load_ply(str(MODEL / 'point_cloud/iteration_16000/point_cloud.ply'))
    for value in vars(pc).values():
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
    geometry = {k: getattr(pc, k).clone() for k in ('_xyz', '_scaling', '_rotation', '_opacity')}
    weights = torch.load(MODEL / 'deform/iteration_16000/deform.pth', map_location='cpu', weights_only=True)
    require(all(torch.count_nonzero(weights[f'{h}.{s}']) == 0 for h in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling') for s in ('weight', 'bias')), 'I00几何头不是零。')
    cameras = []
    # 原校准使用no_bg=False；严格复现该线性FIT目标，不冒充正式sRGB目标。
    for i, info in enumerate(readCamerasFromTransforms(str(SOURCE), 'transforms_train.json', 'images', False, no_bg=False)):
        require(int(info.image_name_train_light.split('_')[-1]) == FIT[i], 'FIT相机序列改变。')
        original_uid = TRAIN.index(FIT[i])
        cameras.append(loadCam(loaded, original_uid, info._replace(uid=original_uid), 1.))
    require(len(cameras) == 117, '不完整FIT。')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.)
    rho = ck['rho'].cuda().clone().requires_grad_(True)
    environment = ck['log_e'].cuda().exp().detach()
    require(torch.allclose(environment.cpu(), ck['environment'], atol=1e-7, rtol=2e-7), 'EL保存值和logE不一致。')
    directions = ck['directions'].cuda()
    require(rho.shape == pc.get_xyz.shape and directions.shape == (135, 3), '参数维度不符。')
    lo, hi = cfg['rho_linear_bounds']
    lower, upper = rho.detach() <= lo + 1e-6, rho.detach() >= hi - 1e-6
    require(torch.isfinite(rho).all() and (rho >= lo - 1e-7).all() and (rho <= hi + 1e-7).all(), 'rho越界。')
    cache = torch.load(L00 / 'raster_forward_cache_v1/raster_cache.pt', map_location='cpu', weights_only=False)
    cache_meta = read(L00 / 'raster_forward_cache_v1/summary.json')
    require(sha(L00 / 'raster_forward_cache_v1/raster_cache.pt') == cache_meta['cache_sha256'] and torch.equal(cache['directions'], ck['directions']), '旧缓存绑定改变。')
    for path in (L00 / 'raster_forward_cache_v1/raster_cache.pt', L00 / 'raster_forward_cache_v1/summary.json'):
        hashes[str(path)] = sha(path)
    normal = build_rotation(pc.get_rotation)[:, :, 2]
    fixed_mask = torch.load(PARENT / 'fixed_foreground.pt', map_location='cpu', weights_only=False)
    require(sha(PARENT / 'fixed_foreground.pt') == cfg['foreground_sha256'] and fixed_mask['frames'] == TRAIN, '原固定mask绑定改变。')
    hashes[str(PARENT / 'fixed_foreground.pt')] = sha(PARENT / 'fixed_foreground.pt')
    for frame, camera in zip(FIT, cameras):
        require(torch.equal((camera.gt_alpha_mask > .5).cpu(), fixed_mask['foreground'][TRAIN.index(frame)]), 'FIT原Alpha分母改变。')
    del fixed_mask
    gradient = torch.zeros_like(rho, dtype=torch.float64)
    block_gradients = {str(i): torch.zeros_like(gradient) for i in range(5)}
    rows, base_images, per_frame_gradients, individual_gradients = [], {}, [], []
    forward_count = backward_count = 0
    traces = {scale: StaticVisibility(cache['bias'] * scale, 1000000) for scale in SCALES}

    def coefficient(camera, frame, scale):
        idx = TRAIN.index(frame)
        light = directions[idx:idx + 1]
        with torch.no_grad():
            vis = traces[scale](pc, light, 0)
            facing = torch.where((normal * (camera.camera_center - pc.get_xyz)).sum(-1, keepdim=True) >= 0, 1., -1.)
            q = ((normal * facing) @ light.T).clamp_min(0) * vis
            require(float((q.cpu() - cache['q'][cache['scales'].index(scale), idx]).abs().max()) < 2e-6, '实际BVH与旧校准q不同。')
        return environment / math.pi + q

    for scale in SCALES:
        for j, (frame, camera) in enumerate(zip(FIT, cameras)):
            camera.load2device()
            factor = coefficient(camera, frame, scale)
            with torch.set_grad_enabled(scale == 1.):
                pred = render(camera, pc, pipe, rho * factor)['render']
                loss, parts, target, mask, saturated, residual = scalar_parts(pred, camera)
                require(torch.isfinite(loss), '原目标非有限。')
                forward_count += 1
                if scale == 1.:
                    require(abs(float(loss) - fits['per_frame'][j]) < 1e-7, '旧中心FIT复现失败，停止。')
                    grad = torch.autograd.grad(loss / 117, rho)[0].detach().double()
                    require(torch.isfinite(grad).all(), '真实材质梯度非有限。')
                    gradient += grad
                    block_gradients[str((frame - 1) // 30)] += grad
                    individual_gradients.append(grad.cpu())
                    per_frame_gradients.append(dict(frame=frame, gradient_l1=float(grad.abs().sum()), gradient_squared=float(grad.square().sum()), lower_outward_l1=float(grad[lower].clamp_min(0).sum()), upper_outward_l1=float((-grad[upper]).clamp_min(0).sum())))
                    append(output / 'partial_gradient.jsonl', per_frame_gradients[-1])
                    backward_count += 1
            rows.append(dict(case='base', scale=scale, frame=frame, loss=float(loss), loss_parts=parts, foreground_channels=int(mask.sum())))
            append(output / 'partial_per_frame.jsonl', rows[-1])
            base_images[(scale, frame)] = pred.detach().cpu()
            camera.load2device('cpu')
        print('base', scale, 'complete', flush=True)
        if scale == 1.:
            torch.save(dict(gradient=gradient.cpu(), lower=lower.cpu(), upper=upper.cpu(), backward_count=backward_count), output / 'center_gradient_before_probe.pt')
    outward = torch.where(lower & (gradient > 0) | upper & (gradient < 0), gradient, 0.)
    bound = (torch.where(lower, gradient.clamp_min(0) * rho.detach(), 0.) + torch.where(upper, (-gradient).clamp_min(0) * (1 - rho.detach()), 0.)).sum()
    delta = (-.001 * outward / outward.abs().max().clamp_min(1e-300)).to(rho.dtype)
    probe = (rho.detach() + delta).clamp(0, 1)
    delta = probe - rho.detach()
    require(not torch.count_nonzero(delta[~(lower | upper)]), '反事实改动非贴边坐标。')
    global_bound_weights = (torch.where(lower & (gradient > 0), rho.detach(), 0.) - torch.where(upper & (gradient < 0), 1 - rho.detach(), 0.)).cpu()
    for row, grad in zip(per_frame_gradients, individual_gradients):
        row['signed_global_bound_contribution'] = float((grad * global_bound_weights).sum())
        row['probe_directional_derivative'] = float((grad * delta.cpu()).sum())
    require(abs(sum(r['signed_global_bound_contribution'] for r in per_frame_gradients) - float(bound)) < 1e-10, '逐帧全局上界分解不一致。')
    del individual_gradients
    directional_from_pixels = 0.
    for scale in SCALES:
        for frame, camera in zip(FIT, cameras):
            camera.load2device()
            with torch.no_grad():
                pred = render(camera, pc, pipe, probe * coefficient(camera, frame, scale))['render']
                loss, parts, target, mask, saturated, residual = scalar_parts(pred, camera)
                base = base_images.pop((scale, frame)).cuda()
                diff = pred - base
                before = torch.where(saturated, (base - target).clamp_max(0), base - target)
                derivative = 2 * before * diff
                response = {}
                inner = (camera.gt_alpha_mask.cuda() > .99).expand_as(target)
                for name, subset in [('saturated', saturated), ('unsaturated', ~saturated), ('inner', inner), ('edge', ~inner)]:
                    response[name] = dict(linear_response_abs_mean_fixed_denominator=float((diff.abs() * (mask & subset)).sum() / mask.sum()), directional_derivative=float((derivative * (mask & subset)).sum() / mask.sum()))
                if scale == 1.:
                    directional_from_pixels += float(derivative[mask].mean()) / 117
                require(torch.isfinite(loss), '反事实目标非有限。')
                rows.append(dict(case='probe', scale=scale, frame=frame, loss=float(loss), loss_parts=parts, foreground_channels=int(mask.sum()), response=response))
                append(output / 'partial_per_frame.jsonl', rows[-1])
                forward_count += 1
            camera.load2device('cpu')
        print('probe', scale, 'complete', flush=True)
    means = {case: {str(scale): sum(r['loss'] for r in rows if r['case'] == case and r['scale'] == scale) / 117 for scale in SCALES} for case in ('base', 'probe')}
    center = means['base']['1.0']
    dot_delta = float((gradient * delta).sum())
    derivative_difference = abs(dot_delta - directional_from_pixels)
    require(derivative_difference <= 1e-8 + 1e-3 * abs(dot_delta), '像素方向导数与真实VJP不一致，停止。')
    require(forward_count == 702 and backward_count == 117, '预算超出或不完整。')
    require(all(torch.equal(value, getattr(pc, k)) for k, value in geometry.items()), '冻结几何改变。')
    for path, value in hashes.items():
        require(sha(Path(path)) == value, '执行期间输入改变。')
    torch.save(dict(gradient=gradient.cpu(), outward_gradient=outward.cpu(), delta=delta.cpu(), block_gradients={k: value.cpu() for k, value in block_gradients.items()}, lower=lower.cpu(), upper=upper.cpu()), output / 'gradient_diagnostic.pt')
    write(output / 'per_frame.json', rows)
    write(output / 'per_frame_gradient.json', per_frame_gradients)
    write(output / 'summary.json', dict(status='PASS_FROZEN_GRADIENT_DIAGNOSTIC_EXECUTION', gt_read=False, development_images_decoded=False, optimizer_steps=0, forward_count=forward_count, backward_count=backward_count, loss_means=means, maximal_convex_gain_bound=float(bound), bound_over_center=float(bound) / center, limited_one_percent_nominal_bound_below_threshold=float(bound) / center < .01, lower_count_per_channel=lower.sum(0).tolist(), upper_count_per_channel=upper.sum(0).tolist(), outward_squared_gradient_fraction=float(outward.square().sum() / gradient.square().sum().clamp_min(1e-300)), gradient_dot_delta=dot_delta, directional_derivative_from_pixel_response=directional_from_pixels, directional_derivative_abs_difference=derivative_difference, input_sha256=hashes, limitation='只针对旧固定几何/灯/visibility的线性删失MSE及当前贴边坐标；目标梯度含残差抵消，不能当可见面积；float32计算近阈值需谨慎，不是精确数值证书；无科学精度或放宽材质许可。'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        if CREATED_OUTPUT is not None:
            write(CREATED_OUTPUT / 'FAILED.json', dict(status='FAILED_ENGINEERING', traceback=traceback.format_exc(), preserve_all_existing_outputs=True))
        raise
