"""135个已见TRAIN的静态SH预热诊断；无GT/TEST、不优化、不选择终点。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FRAMES = [i for i in range(1, 151) if i % 10]
ITERATIONS = [1000, 12000, 13000, 14000, 15000, 16000]
REPRESENTATIVES = [1, 81, 149]
PYTHON = Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python')
DOC = ROOT / 'LIHAN/DOC/报告/1019-45-SH预热TRAIN诊断协议.md'
CORE = ['scene/gaussian_model.py', 'scene/deform_model.py', 'utils/time_utils.py',
        'gaussian_renderer/__init__.py', 'scene/cameras.py', 'scene/dataset_readers.py',
        'utils/rgba16.py', 'utils/camera_utils.py', 'utils/loss_utils.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def safe_location(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / 'output'), '数据与输出必须位于本仓库output')
    require('eval_only' not in path.parts and 'eval_source' not in path.parts,
            'TRAIN诊断禁止外部评测目录')
    return path


def input_files(model, source, iterations, complete=False):
    """仅元数据/来源SHA检查，CPU计划不解码图像或加载checkpoint。"""
    marker = read(model / 'eldl_v3_training.json')
    require(marker['actual_entrypoint'] == 'scripts/train_eldl_v3_sh_warmup.py' and
            marker['baseline'] == 'static_sh' and marker['gt_supervision'] is False and
            marker['external_test_read'] is False and marker['init_weights_loaded'] is False,
            '仅接受无GT共同135TRAIN静态SH预热入口')
    require(marker['checkpoint_semantics'] == 'post_optimizer_post_native_topology', '保存语义不匹配')
    cfg = marker['args']
    require(Path(cfg['source_path']).resolve() == source and Path(cfg['model_path']).resolve() == model,
            '模型/source记录不匹配')
    require(cfg['is_blender'] and cfg['deform_type'] == 'mlp' and cfg['resolution'] == 2 and
            cfg['gt_alpha_mask_as_scene_mask'] and not cfg['white_background'],
            '当前协议固定Blender/MLP/400/原Alpha/黑背景')
    manifest = read(source / 'no_gt_manifest.json')
    require(manifest['gt_read'] is False and manifest['init_weights_loaded'] is False and
            manifest['train_frames'] == FRAMES and marker['train_frames'] == FRAMES,
            '必须完整135已见TRAIN，禁止117内部FIT或TEST')
    require(sha(source / 'no_gt_manifest.json') == marker['input_manifest_sha256'], '训练源manifest已变')
    require({p.name for p in source.iterdir()} ==
            {'images', 'points3d.ply', 'transforms_train.json', 'no_gt_manifest.json'}, 'TRAIN源含额外数据')
    transforms = read(source / 'transforms_train.json')
    require(transforms.get('rgba16') is True and [r['source_frame'] for r in transforms['frames']] == FRAMES,
            '必须原135 TRAIN显式16位RGBA')
    files = {str(model / 'eldl_v3_training.json'): sha(model / 'eldl_v3_training.json'),
             str(model / 'cfg_args'): sha(model / 'cfg_args'),
             str(source / 'no_gt_manifest.json'): sha(source / 'no_gt_manifest.json'),
             str(Path(__file__).resolve()): sha(__file__), str(DOC): sha(DOC)}
    for relative, expected in manifest['files'].items():
        p = (source / relative).resolve()
        require(p.is_relative_to(source) and sha(p) == expected, 'TRAIN输入SHA不符：' + relative)
        files[str(p)] = expected
    listed = {str(p.relative_to(source)) for p in (source / 'images').iterdir()}
    require(listed == {n for n in manifest['files'] if n.startswith('images/')}, 'TRAIN图像白名单不完整')
    snapshot = Path(marker['source_code_root']).resolve()
    core = {}
    for relative in CORE:
        p, old = ROOT / relative, snapshot / relative
        core[relative] = sha(p) == sha(old)
        require(core[relative], '训练核心与诊断核心不同：' + relative)
        files[str(p)] = sha(p)
        files[str(old)] = sha(old)
    files[str(snapshot / 'scene/eldl_v3_sh_warmup.py')] = sha(snapshot / 'scene/eldl_v3_sh_warmup.py')
    files[str(snapshot / 'scripts/train_eldl_v3_sh_warmup.py')] = sha(snapshot / 'scripts/train_eldl_v3_sh_warmup.py')
    pending, endpoints = [], {}
    for step in iterations:
        folder = model / f'geometry_init/actual_{step}'
        paths = dict(receipt=folder / 'receipt.json', geometry=folder / 'geometry.pt',
                     ply=model / f'point_cloud/iteration_{step}/point_cloud.ply',
                     deform=model / f'deform/iteration_{step}/deform.pth')
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            pending += missing
            continue
        receipt = read(paths['receipt'])
        require(receipt['requested_step'] == step and receipt['completed_optimizer_calls'] == step and
                receipt['checkpoint_semantics'] == marker['checkpoint_semantics'] and
                receipt['gt_supervision'] is False and receipt['external_test_read'] is False and
                receipt['geometry_heads_zero'] and receipt['optimizer_audit']['iteration'] == step and
                receipt['optimizer_audit']['topology_precedes_optimizer'], 'post-step端点收据不匹配')
        for key in ('geometry', 'ply', 'deform'):
            require(sha(paths[key]) == receipt[key + '_sha256'], '端点SHA不匹配：' + key)
        files.update({str(p): sha(p) for p in paths.values()})
        endpoints[str(step)] = {k: str(p) for k, p in paths.items()}
    parent = model / 'geometry_parent_manifest.json'
    if parent.is_file():
        data = read(parent)
        require(data['gt_supervision'] is False and data['external_test_read'] is False and
                data['pure_geometry_only'] and data['train_frames'] == FRAMES and
                data['completed_optimizer_calls'] == 16000 and data['source'] == str(source) and
                data['model_record_sha256'] == sha(model / 'eldl_v3_training.json'), 'parent manifest不匹配')
        files[str(parent)] = sha(parent)
        for step, endpoint in endpoints.items():
            for key in ('geometry', 'ply', 'deform', 'receipt'):
                require(data['endpoints'][step][key + '_sha256'] == sha(endpoint[key]), 'parent端点绑定不匹配')
    else:
        pending.append(str(parent))
    if marker.get('completed_optimizer_calls') != 16000:
        pending.append('训练正式16000次optimizer调用尚未完成')
    if complete:
        require(not pending, '正式运行须等待完整16000/所有端点及根任务ready：' + str(pending))
    return dict(marker=marker, source_manifest=manifest, endpoints=endpoints, input_sha256=files,
                core_code_matches=core, pending=pending)


def change_statistics(old, current):
    """CPU tensor统计；法线不取abs，缺失始终留在原Alpha固定分母。"""
    import torch
    zcommon = old['depth_valid'] & current['depth_valid']
    ncommon = old['normal_valid'] & current['normal_valid']
    denominator = len(zcommon)
    delta = (old['depth'][zcommon] - current['depth'][zcommon]).abs()
    angles = ((old['normal'][:, ncommon] * current['normal'][:, ncommon]).sum(0)
              .clamp(-1, 1).acos() * (180 / math.pi))
    relative = (delta / old['depth'][zcommon].clamp_min(1e-6)).clamp(max=1)
    return dict(fixed_source_alpha_pixels=denominator, depth_common_pixels=int(zcommon.sum()),
        normal_common_pixels=int(ncommon.sum()), expected_depth_change_sum=float(delta.sum()),
        expected_depth_change_mae=float(delta.mean()) if delta.numel() else None,
        gs_normal_change_sum_deg=float(angles.sum()),
        gs_normal_change_mae_deg=float(angles.mean()) if angles.numel() else None,
        gs_normal_change_missing180_sum=float(angles.sum()) + 180 * (denominator - int(ncommon.sum())),
        relative_depth_change_missing1_sum=float(relative.sum()) + denominator - int(zcommon.sum()))


def aggregate_changes(rows):
    total = sum(r['fixed_source_alpha_pixels'] for r in rows)
    zn, nn = sum(r['depth_common_pixels'] for r in rows), sum(r['normal_common_pixels'] for r in rows)
    return dict(fixed_source_alpha_pixels=total, depth_common_pixels=zn, normal_common_pixels=nn,
        depth_common_coverage=zn / max(1, total), normal_common_coverage=nn / max(1, total),
        expected_depth_change_pixel_mae=sum(r['expected_depth_change_sum'] for r in rows) / zn if zn else None,
        expected_depth_change_frame_mae=mean_valid(rows, 'expected_depth_change_mae'),
        gs_normal_change_pixel_mae_deg=sum(r['gs_normal_change_sum_deg'] for r in rows) / nn if nn else None,
        gs_normal_change_frame_mae_deg=mean_valid(rows, 'gs_normal_change_mae_deg'),
        gs_normal_change_missing180_pixel_mean_deg=sum(r['gs_normal_change_missing180_sum'] for r in rows) / max(1, total),
        relative_depth_change_missing1_pixel_mean=sum(r['relative_depth_change_missing1_sum'] for r in rows) / max(1, total))


def mean_valid(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return sum(values) / len(values) if values else None


def cpu_self_check():
    import torch
    x = dict(depth=torch.tensor([1., 1., 1.]), normal=torch.tensor([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.]]),
             depth_valid=torch.tensor([True, True, True]), normal_valid=torch.tensor([True, True, True]))
    y = dict(depth=torch.tensor([1.1, 1., 0.]), normal=torch.tensor([[-1., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
             depth_valid=torch.tensor([True, True, False]), normal_valid=torch.tensor([True, True, False]))
    r = change_statistics(x, y)
    require(r['normal_common_pixels'] == 2 and abs(r['gs_normal_change_sum_deg'] - 180) < 1e-6 and
            abs(r['gs_normal_change_missing180_sum'] - 360) < 1e-6 and
            abs(r['relative_depth_change_missing1_sum'] - 1.1) < 1e-6, '有向/固定分母/缺失罚自检失败')
    require(abs(aggregate_changes([r])['gs_normal_change_missing180_pixel_mean_deg'] - 120) < 1e-6,
            '聚合自检失败')
    rejected = False
    try:
        safe_location(ROOT / 'output/x/eval_only/test')
    except ValueError:
        rejected = True
    require(rejected, '外部评测目录必须拒绝')
    print('PASS_CPU_SELF_CHECK：有向反向180°、缺失固定分母、独立深度/法线支持及路径边界；无GPU/GT。')


def install_guard(files, output, model, source):
    from utils.no_gt_guard import install_no_gt_guard
    exact = {Path(p).resolve() for p in files}
    install_no_gt_guard([*exact, output])
    opened = set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if 'eval_only' in p.parts:
            raise PermissionError('SH TRAIN诊断禁止外部评测读取：' + str(p))
        if p.is_relative_to(source) or p.is_relative_to(model):
            require(p in exact, '模型/TRAIN源仅允许精确登记文件：' + str(p))
            opened.add(str(p))
    sys.addaudithook(audit)
    return opened


def render_native(camera, pc, deform, cfg, step):
    """同一次生产render截获allmap；不改共享renderer或训练depth_ratio。"""
    import torch
    import gaussian_renderer as renderer
    capture = {}
    native = renderer.GaussianRasterizer
    class Capture:
        def __init__(self, *args, **kwargs):
            self.instance = native(*args, **kwargs)
        def __call__(self, **kwargs):
            result = self.instance(**kwargs)
            require('allmap' not in capture, '生产forward意外多次光栅')
            capture['allmap'] = result[2]
            return result
    n = len(pc.get_xyz)
    if step < cfg['warm_up']:
        delta = dict(d_xyz=0., d_rotation=0., d_scaling=0., d_opacity=None, d_color=None)
    else:
        delta = deform.step(pc.get_xyz, camera.fid.unsqueeze(0).expand(n, -1), iteration=step,
                            feature=pc.get_binary_feature(eval=False, T=.5), camera_center=camera.camera_center)
    for key in ('d_xyz', 'd_rotation', 'd_scaling'):
        v = delta[key]
        require(not torch.is_tensor(v) or torch.count_nonzero(v) == 0, '静态几何实际输出非零：' + key)
    require(delta['d_opacity'] is None, '静态SH opacity不允许MLP变化')
    try:
        renderer.GaussianRasterizer = Capture
        package = renderer.render(camera, pc, SimpleNamespace(**cfg), torch.zeros(3, device='cuda'),
                                  **{k: delta[k] for k in ('d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color')})
    finally:
        renderer.GaussianRasterizer = native
    allmap = capture['allmap']
    require(torch.equal(allmap[1:2], package['rend_alpha']), '同次native Alpha不一致')
    raw_expected = allmap[0:1] / allmap[1:2]
    expected_finite = torch.isfinite(raw_expected)
    expected = torch.nan_to_num(raw_expected, nan=0., posinf=0., neginf=0.)
    return package, expected, delta, expected_finite


def objective(package, delta, camera, pc, cfg, step):
    import torch
    from utils.loss_utils import l1_loss, ssim
    import torch.nn.functional as F
    target, alpha = camera.original_image_train_light, camera.gt_alpha_mask
    l1 = l1_loss(package['render'], target)
    structural = 1 - ssim(package['render'], target)
    image = (1 - cfg['lambda_dssim']) * l1 + cfg['lambda_dssim'] * structural
    # 与原SH训练一致的全图BCE，不加预测mask、不作物理删失或线性化。
    bce = F.binary_cross_entropy(package['rend_alpha'][:, None, None], alpha.unsqueeze(1).unsqueeze(1))
    normal = (.02 if step > cfg['start_normal_reg'] else 0.) * (
        1 - (package['rend_normal'] * package['surf_normal']).sum(0)).mean()
    dist = (cfg['lambda_dist'] if step > cfg['start_normal_reg'] else 0.) * package['rend_dist'].mean()
    xyz = torch.zeros_like(l1)
    color, separation = torch.zeros_like(l1), torch.zeros_like(l1)
    if step > cfg['warm_up']:
        xyz = delta['d_xyz'].square().mean() * cfg['d_xyz_loss_weight']
        if torch.is_tensor(delta['d_color']):
            color = delta['d_color'][:, :3].square().mean() * cfg['d_color_reg_loss_weight']
        if step > cfg['binarization_warm_up'] and not cfg['no_binary_separation']:
            separation = pc.get_binary_feature(eval=False, T=.5).mean() * cfg['lambda_separation']
    total = image + cfg['lambda_alpha_loss'] * bce + normal + dist + xyz + color + separation
    return dict(l1_srgb=float(l1), one_minus_ssim=float(structural), sh_image_objective=float(image),
        alpha_bce=float(bce), alpha_l1=float((package['rend_alpha'] - alpha).abs().mean()),
        weighted_normal_regularizer=float(normal), weighted_dist_regularizer=float(dist),
        weighted_zero_xyz_regularizer=float(xyz), weighted_d_color_regularizer=float(color),
        weighted_stochastic_separation_regularizer=float(separation), frozen_poststep_sh_total=float(total))


def pictures(out, step, frame, package, pc, camera, cfg):
    import numpy as np
    import torch
    import cv2
    from gaussian_renderer import render
    folder = out / 'figures' / f'iteration_{step}'
    folder.mkdir(parents=True, exist_ok=True)
    gate = render(camera, pc, SimpleNamespace(**cfg), torch.zeros(3, device='cuda'), 0., 0., 0.,
                  override_color=pc.get_binary_feature(eval=True).expand(-1, 3))['render']
    normal = torch.nn.functional.normalize(package['rend_normal'], dim=0)
    values = dict(rgb=package['render'], alpha=package['rend_alpha'].expand(3, -1, -1),
                  normal=normal * .5 + .5, separation=gate)
    paths = {}
    for kind, value in values.items():
        array = (value.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        path = folder / f'{kind}_{frame:04d}.png'
        require(cv2.imwrite(str(path), array[..., ::-1]), 'PNG写入失败')
        paths[str(path)] = sha(path)
    return paths


def run(a, inputs):
    import torch
    require(torch.cuda.is_available(), '正式run需要已协调释放的GPU')
    import diff_surfel_rasterization as backend
    require(backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer' and
            backend.provenance['source_sha256'] == inputs['marker']['rasterizer_provenance']['source_sha256'],
            '必须bootstrap绑定与训练同源原生CUDA')
    training_backend = inputs['marker']['rasterizer_provenance']
    actual_extension = Path(backend._C.__file__).resolve()
    require(str(actual_extension) == str(Path(training_backend['extension_path']).resolve()) and
            sha(actual_extension) == training_backend['extension_sha256'] == backend.provenance['extension_sha256'],
            '实际加载native扩展对象/二进制SHA与训练不一致')
    from scene.gaussian_model import GaussianModel
    from scene.deform_model import DeformModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    cfg = dict(inputs['marker']['args'])
    opened = install_guard(inputs['input_sha256'], a.output, a.model, a.source)
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / 'plan.json', read(a.plan))
    write(a.output / 'environment.json', dict(hostname=socket.gethostname(), python=sys.executable,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), device=torch.cuda.get_device_name(),
        torch_version=torch.__version__, backend=backend.provenance,
        actual_extension_object=dict(path=str(actual_extension), sha256=sha(actual_extension)),
        proc_self_maps=Path('/proc/self/maps').read_text(),
        threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        command=shlex.join([sys.executable, *sys.argv])))
    cameras, fixed_masks = [], {}
    transform_records = read(a.source / 'transforms_train.json')['frames']
    for index, info in enumerate(readCamerasFromTransforms(str(a.source), 'transforms_train.json', 'images', False, no_bg=True)):
        require(int(info.image_name_train_light.split('_')[-1]) == FRAMES[index], 'TRAIN相机顺序不符')
        cam = loadCam(SimpleNamespace(**cfg), index, info, 1.)
        require((cam.image_height, cam.image_width) == (400, 400), '固定400²')
        require(hasattr(cam, 'observed_saturation_mask'), '缺少原16位饱和通道记录')
        center = torch.tensor(transform_records[index]['transform_matrix'], dtype=torch.float32)[:3, 3]
        require(float((cam.camera_center.cpu() - center).abs().max()) < 1e-4, '原c2w相机中心不符')
        support = (cam.gt_alpha_mask[0] >= .5).cpu().numpy()
        fixed_masks[str(FRAMES[index])] = dict(pixels=int(support.sum()),
            sha256=hashlib.sha256(support.tobytes()).hexdigest())
        cameras.append(cam)
    require(len(cameras) == 135, '必须全部135已见TRAIN')
    results, previous, previous_step, figures = [], {}, None, {}
    with torch.no_grad():
        for step in a.iterations:
            ep = inputs['endpoints'][str(step)]
            pc = GaussianModel(cfg['sh_degree'], no_binary_separation=cfg['no_binary_separation'], fea_dim=cfg['hyper_dim'])
            pc.load_ply(ep['ply'])
            pc.active_sh_degree = min(pc.max_sh_degree, step // cfg['oneupSHdegree_step'])
            pc.no_binary_separation = cfg['no_binary_separation'] or step <= cfg['binarization_warm_up']
            geometry = torch.load(ep['geometry'], map_location='cpu', weights_only=True)
            for name in ('_xyz', '_scaling', '_rotation', '_opacity'):
                require(torch.equal(getattr(pc, name).cpu(), geometry[name]), '纯几何与实际PLY不一致：' + name)
            deform = DeformModel(deform_type='mlp', is_blender=cfg['is_blender'], hyper_dim=cfg['hyper_dim'], pred_color=cfg['pred_color'])
            deform.deform.load_state_dict(torch.load(ep['deform'], map_location='cpu', weights_only=True))
            for head in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
                for p in getattr(deform.deform, head).parameters():
                    require(torch.count_nonzero(p) == 0, '静态几何头权重非零')
            rows, changes = [], []
            for frame, camera in zip(FRAMES, cameras):
                camera.load2device()
                # 原训练binary(eval=False)为随机Concrete；仅诊断固定seed并恢复外部RNG，不伪称原随机batch loss。
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(step * 1000 + frame)
                    package, depth, delta, expected_finite = render_native(camera, pc, deform, cfg, step)
                    row = objective(package, delta, camera, pc, cfg, step)
                alpha = camera.gt_alpha_mask
                support = (alpha >= .5)[0]
                pred_mask = (package['rend_alpha'] >= .5)[0]
                z = depth[0]
                normal = torch.nn.functional.normalize(package['rend_normal'], dim=0)
                zv = pred_mask & expected_finite[0] & (z > 0)
                nv = pred_mask & torch.isfinite(normal).all(0) & (normal.norm(dim=0) > .5)
                truth, tp = int(support.sum()), int((pred_mask & support).sum())
                row.update(frame=frame, fixed_source_alpha_pixels=truth, alpha_true_positive=tp,
                    fixed_source_alpha_sha256=fixed_masks[str(frame)]['sha256'],
                    alpha_false_positive=int((pred_mask & ~support).sum()), alpha_false_negative=truth - tp,
                    alpha_iou=float((pred_mask & support).sum() / (pred_mask | support).sum().clamp_min(1)),
                    expected_depth_valid_pixels=int((zv & support).sum()), gs_normal_valid_pixels=int((nv & support).sum()),
                    raw_expected_nonfinite_source_pixels=int((~expected_finite[0] & support).sum()),
                    source_foreground_depth_coverage=float((zv & support).sum() / support.sum().clamp_min(1)),
                    source_foreground_normal_coverage=float((nv & support).sum() / support.sum().clamp_min(1)),
                    max_abs_geometry_output=0., saturation_channels=int(camera.observed_saturation_mask.sum()),
                    note='饱和仅计数；原SH全图sRGB L1/SSIM不使用物理删失')
                current = dict(depth=z[support].cpu(), normal=normal[:, support].cpu(),
                    depth_valid=zv[support].cpu(), normal_valid=nv[support].cpu())
                digest = hashlib.sha256()
                for key in ('depth', 'normal', 'depth_valid', 'normal_valid'):
                    digest.update(current[key].numpy().tobytes())
                row['predicted_geometry_sha256'] = digest.hexdigest()
                if previous_step is not None:
                    changes.append(dict(frame=frame, previous_iteration=previous_step, **change_statistics(previous[frame], current)))
                previous[frame] = current
                if frame in REPRESENTATIVES:
                    figures.update(pictures(a.output, step, frame, package, pc, camera, cfg))
                rows.append(row)
                camera.load2device('cpu')
                del package, depth, delta, expected_finite
            event = dict(post_optimizer_post_native_topology=True,
                native_topology_precedes_optimizer=step < cfg['densify_until_iter'],
                densify_and_prune_at_step=(cfg['densify_from_iter'] < step < cfg['densify_until_iter'] and step % cfg['densification_interval'] == 0),
                opacity_reset_at_step=(step < cfg['densify_until_iter'] and step % cfg['opacity_reset_interval'] == 0),
                warmup_boundary=step == cfg['warm_up'],
                note='12000/15000为reset后状态；端点间跨拓扑/点数变化，不作固定Gaussian ID匹配')
            numerical_keys=[k for k,v in rows[0].items() if isinstance(v, (int,float)) and k != 'frame']
            record = dict(iteration=step, points=len(pc.get_xyz), active_sh_degree=pc.active_sh_degree,
                events=event, rows=rows, frame_mean={k: mean_valid(rows,k) for k in numerical_keys},
                fixed_support_pixel_totals={k:sum(r[k] for r in rows) for k in
                    ('fixed_source_alpha_pixels','expected_depth_valid_pixels','gs_normal_valid_pixels','alpha_false_positive','alpha_false_negative')},
                checkpoint_receipt=read(ep['receipt']))
            if changes:
                record['geometry_change'] = dict(previous_iteration=previous_step, rows=changes, **aggregate_changes(changes))
            write(a.output / f'iteration_{step}.json', record)
            results.append({k:v for k,v in record.items() if k not in ('rows','checkpoint_receipt')})
            print(dict(iteration=step, points=record['points'], **record['frame_mean']), flush=True)
            previous_step = step
            del pc, deform, geometry
            torch.cuda.empty_cache()
    for p,h in inputs['input_sha256'].items():
        require(sha(p) == h, '运行中输入SHA变化：' + p)
    write(a.output / 'summary.json', dict(status='PASS_SEEN_TRAIN_SH_DIAGNOSTIC_EXECUTION',
        gt_read=False, external_test_read=False, all_135_frames_seen_in_training=True,
        development_held_evaluation=False, no_parameter_updates=True, accuracy_claim=False,
        physics_censored_objective=False, depth_statistic='expected_camera_Z', normal_statistic='signed_GS',
        input_sha256=inputs['input_sha256'], opened_whitelisted_inputs=sorted(opened),
        fixed_source_alpha_masks=fixed_masks,
        source=str(a.source), model=str(a.model), iterations=a.iterations, endpoints=results,
        figures_sha256=figures, manual_review='PENDING_ACTUAL_VIEW',
        stochastic_gate='原SH训练eval=False Concrete；诊断每step/frame固定seed且fork恢复RNG，非原训练batch loss复现',
        limitation='相邻预测稳定性不是GT精度；原Alpha固定分母缺失罚保留，reset/topology显式记录；不据此挑终点/调光'))
    (a.output / 'README.md').write_text('# 共同静态SH预热已见TRAIN诊断\n\n2026-10-19（Asia/Tokyo）。\n\n'+
        f'source：`{a.source}`；model：`{a.model}`；iterations：{a.iterations}；输出：`{a.output}`。\n\n'+
        '完整命令/环境见environment.json和plan.json；日志由supervisor保存在本目录。定量见summary.json与iteration_*.json；代表图figures/iteration_*/{rgb,alpha,normal,separation}_{0001,0081,0149}.png及SHA见summary.json。\n\n'+
        '最终PASS仅执行工程；四类实际目检PENDING，未声称精度PASS。135均训练已见，不作held；原SH sRGB L1/DSSIM/Alpha BCE与物理linear censored目标不同。12000/15000 reset及原topology前置语义已标记，变化不是GT准确性。全部产物保留，禁止调光/选终点。\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['plan', 'run'], default='plan')
    for name in ('model', 'source', 'output', 'plan'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--iterations', type=int, nargs='+', default=ITERATIONS)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check:
        cpu_self_check()
        return
    require(socket.gethostname() == 'ushas' and Path(sys.executable).resolve() == PYTHON.resolve(), '需要ushas对应Conda环境')
    require(all(getattr(a,n) for n in ('model','source','output','plan')), '需要model/source/output/plan')
    a.model, a.source, a.output, a.plan = [safe_location(getattr(a,n)) for n in ('model','source','output','plan')]
    require(a.iterations == ITERATIONS, '本轮端点已固定1000/12000/13000/14000/15000/16000，不据结果挑选')
    require(not a.output.exists() and not a.output.is_relative_to(a.model) and not a.output.is_relative_to(a.source),
            '禁止覆盖/写入模型或训练源')
    inputs = input_files(a.model, a.source, a.iterations, complete=a.action == 'run')
    plan = dict(status='PENDING_TRAINING_COMPLETION' if inputs['pending'] else 'READY_CPU_INPUT_CHECK_ONLY',
        model=str(a.model), source=str(a.source), output=str(a.output), iterations=a.iterations,
        source_frames=FRAMES, representatives=REPRESENTATIVES,
        pending=inputs['pending'], input_sha256=inputs['input_sha256'], core_code_matches=inputs['core_code_matches'],
        gpu_authorized=False, gt_read=False, test_read=False, no_parameter_updates=True,
        protocol_sha256=sha(DOC), script_sha256=sha(__file__))
    if a.action == 'plan':
        require(not a.plan.exists(), 'CPU计划禁止覆盖')
        a.plan.parent.mkdir(parents=True, exist_ok=True)
        write(a.plan, plan)
        print(plan['status'], 'PENDING=', len(inputs['pending']), 'SHA=', sha(a.plan))
    else:
        frozen = read(a.plan)
        require(frozen == plan and not plan['pending'], '必须重新冻结完整落盘后的CPU计划，且逐项完全一致')
        run(a, inputs)


if __name__ == '__main__':
    main()
