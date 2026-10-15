"""从默认 Stage1 验证未知主光的真实光栅梯度；不读取任何 GT 光照资产。"""
import argparse
import ast
import hashlib
import json
import math
import shlex
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.pbr_rasterizer import native_module
from scene.stage2_photometric import DirectionalLightTable, srgb_to_linear, linear_to_srgb
from utils.camera_utils import loadCam
from utils.general_utils import build_rotation


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_image(path, tensor):
    array = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(path)


def read_config(path):
    node = ast.parse(path.read_text().strip(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
        raise ValueError('配置必须是纯字面量 Namespace。')
    return SimpleNamespace(**{kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})


def run(args, out):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = read_config(Path(args.model) / 'cfg_args')
    if cfg.white_background:
        raise ValueError('首轮探针仅支持黑背景，避免改变历史合成约定。')
    cfg.resolution = args.resolution
    cfg.load2gpu_on_the_fly = True
    model = Path(args.model).resolve()
    source = Path(args.source).resolve()
    ply = model / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    weights = model / f'deform/iteration_{args.iteration}/deform.pth'
    provenance = native_module().provenance
    manifest = {
        'args': vars(args), 'source': str(source), 'model': str(model),
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ply, weights, model / 'cfg_args', source / 'transforms_train.json', Path(__file__)]},
        'native': provenance, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'scope': '仅工程梯度探针：无阴影、无环境、无镜面；不是正式自标定精度实验',
        'gt_lighting_read': False, 'normal_transport': '沿用默认形变旋转头，尚未证明与切平面一致',
    }
    (out / 'runtime.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    pc.load_ply(str(ply))
    deform = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender,
                         hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    assert deform.load_weights(str(model), args.iteration)
    deform.deform.eval().requires_grad_(False)
    tensors = {name: val for name, val in vars(pc).items() if isinstance(val, torch.Tensor)}
    for val in tensors.values():
        val.requires_grad_(False)
    frozen = {**tensors, **{'deform.' + k: v for k, v in deform.deform.state_dict().items()}}
    before = {k: digest_tensor(v) for k, v in frozen.items()}
    # 只调用训练相机读取器；不经过会加载测试集／初始化点云的 Scene。
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', cfg.train_light_folder, False)
    indices = np.linspace(0, len(infos) - 1, args.frames, dtype=int).tolist()
    cameras = [loadCam(cfg, i, infos[i], 1.) for i in indices]
    cached = []
    bg = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=1.)
    with torch.no_grad():
        for camera in cameras:
            camera.load2device()
            ds = deform.step(pc.get_xyz, camera.fid.reshape(1, 1).expand(len(pc.get_xyz), 1),
                             feature=pc.get_binary_feature(), camera_center=camera.camera_center)
            ds = {k: ds[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color']}
            normals = build_rotation(pc.get_rotation_bias(ds['d_rotation']))[:, :, 2]
            view = camera.camera_center - (pc.get_xyz + ds['d_xyz'])
            normals *= torch.where((normals * view).sum(-1, keepdim=True) >= 0, 1., -1.)
            target = camera.original_image_train_light
            mask = camera.gt_alpha_mask.to('cuda') if camera.gt_alpha_mask is not None else torch.ones_like(target[:1])
            target = target * mask
            # 使用原图满码标记的保守下采样，避免缩图抹去饱和信息。
            original = np.array(Image.open(camera.image_path_train_light).convert('RGBA'))
            saturated = torch.from_numpy((original[..., :3] >= 254).astype(np.float32)).permute(2, 0, 1)[None].cuda()
            saturated = F.adaptive_max_pool2d(saturated, target.shape[-2:])[0].bool() & (mask > .99)
            native = render(camera, pc, pipe, bg, **ds)
            save_image(out / f'{camera.image_name_train_light}_stage1.png', native['render'])
            cached.append((camera, ds, normals, srgb_to_linear(target), saturated))
    times = torch.tensor([float(c.fid.item()) for c in cameras], device='cuda')
    if not (times[1:] > times[:-1]).all():
        raise ValueError('采样训练相机时间须严格递增。')
    directions = F.normalize(torch.randn(len(times), 3, device='cuda'), dim=-1)
    lights = DirectionalLightTable(times, directions)
    initial = lights.raw_directions.detach().clone()
    # 固定辐照度尺度 pi；全局共享的灰反照率只用于隔离灯向梯度。
    albedo = torch.full_like(pc.get_xyz, .5)

    def forward(index):
        camera, ds, normals, target, saturated = cached[index]
        direction = lights(camera.fid)
        colors = albedo * (normals @ direction).clamp_min(0)[:, None]
        package = render(camera, pc, pipe, bg, **ds, override_color=colors)
        residual = package['render'] - target
        # 饱和仅提供线性亮度下界；本探针固定 albedo/能量并不追求拟合饱和像素。
        residual = torch.where(saturated, residual.clamp_max(0), residual)
        return residual.square().mean(), package

    optimizer = torch.optim.Adam(lights.parameters(), lr=args.lr)
    loss, _ = forward(0)
    loss.backward()
    grad = lights.raw_directions.grad.detach().clone()
    coordinate = int(grad.abs().argmax())
    analytic = float(grad.flatten()[coordinate])
    if abs(analytic) < 1e-8:
        raise RuntimeError('真实灯向梯度为零，不能进入训练。')
    differences = []
    with torch.no_grad():
        original_value = lights.raw_directions.flatten()[coordinate].item()
        for epsilon in [1e-2, 3e-3, 1e-3]:
            lights.raw_directions.flatten()[coordinate] = original_value + epsilon
            plus = float(forward(0)[0])
            lights.raw_directions.flatten()[coordinate] = original_value - epsilon
            minus = float(forward(0)[0])
            lights.raw_directions.flatten()[coordinate] = original_value
            numerical = (plus - minus) / (2 * epsilon)
            relative = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-8)
            differences.append(dict(epsilon=epsilon, numerical=numerical, relative_error=relative))
        initial_losses = [float(forward(i)[0]) for i in range(len(cached))]
    gradient_ok = all(x['relative_error'] < .05 for x in differences)
    (out / 'gradient.json').write_text(json.dumps(dict(analytic=analytic, coordinate=coordinate,
                                                          differences=differences, passed=gradient_ok), indent=2))
    if not gradient_ok:
        raise RuntimeError('灯向有限差分不满足预注册 5% 相对误差。')
    history = []
    started = time.monotonic()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward(step % len(cached))
        if not torch.isfinite(loss):
            raise RuntimeError('损失非有限。')
        loss.backward()
        optimizer.step()
        history.append(dict(update=step + 1, loss=float(loss)))
        print(json.dumps(history[-1]), flush=True)
    final_losses = []
    with torch.no_grad():
        for index, (camera, ds, _, _, _) in enumerate(cached):
            loss, package = forward(index)
            final_losses.append(float(loss))
            name = camera.image_name_train_light
            save_image(out / f'{name}_rgb.png', linear_to_srgb(package['render'].clamp_min(0)))
            save_image(out / f'{name}_alpha.png', package['rend_alpha'])
            normal = F.normalize(package['rend_normal'], dim=0)
            save_image(out / f'{name}_normal.png', (normal * .5 + .5) * package['rend_alpha'])
            gate = pc.get_binary_feature()
            separation = torch.cat([gate, 1 - gate, torch.zeros_like(gate)], dim=1)
            sep = render(camera, pc, pipe, bg, **ds, override_color=separation)
            save_image(out / f'{name}_separation.png', sep['render'])
        after = {k: digest_tensor(v) for k, v in frozen.items()}
        unchanged = before == after
        delta = float((lights.raw_directions - initial).abs().max())
    torch.save(dict(lights=lights.state_dict(), optimizer=optimizer.state_dict(), updates=args.steps,
                    seed=args.seed, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    args=vars(args)), out / 'checkpoint.pt')
    passed = unchanged and delta > 0 and np.isfinite(final_losses).all() and np.mean(final_losses) < np.mean(initial_losses)
    result = dict(status='PASS' if passed else 'FAILED', scope='工程冒烟，科学目标未验收',
                  optimizer_updates=args.steps, gradient_ok=gradient_ok, frozen_unchanged=unchanged,
                  direction_parameter_delta=delta, initial_losses=initial_losses, final_losses=final_losses,
                  frames=[c.image_name_train_light for c in cameras], times=times.tolist(),
                  directions=F.normalize(lights.raw_directions.detach(), dim=-1).tolist(),
                  points=len(pc.get_xyz), elapsed_seconds=time.monotonic()-started,
                  max_memory_bytes=torch.cuda.max_memory_allocated(), history=history,
                  frozen_hashes=before)
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['history', 'frozen_hashes']}, ensure_ascii=False), flush=True)
    if not passed:
        raise RuntimeError('参数更新／冻结／损失下降验收失败，保留现场。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iteration', type=int, default=35000)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1015)
    parser.add_argument('--lr', type=float, default=.03)
    args = parser.parse_args()
    if args.frames < 1 or args.steps < 1:
        parser.error('frames 和 steps 必须为正。')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'runtime.json').exists() or (out / 'result.json').exists():
        raise FileExistsError('不得覆盖已执行探针。')
    (out / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    try:
        run(args, out)
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps({'status': 'FAILED', 'error': repr(error)}, ensure_ascii=False, indent=2))
        raise


if __name__ == '__main__':
    main()
