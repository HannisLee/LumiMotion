"""两登记SH起点的固定几何材质预适配；只导出DC，不继承优化器。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import socket
import sys
from types import SimpleNamespace

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/preadapt_eldl_v3_material.py'
FRAMES = [i for i in range(1, 151) if i % 10]
KEYS = ('_xyz', '_scaling', '_rotation', '_opacity')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--initial-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source, model, out = [getattr(args, k).resolve() for k in ('source', 'initial_model', 'output')]
    repo = source.parent.parent.parent
    require(socket.gethostname() == 'ushas' and Path(sys.executable).resolve() == Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve(), '需要ushas对应环境。')
    require(source == repo / 'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_train_eldl_raster_v1', '只允许原旧混合光135TRAIN源。')
    require(out.is_relative_to(repo / 'output') and not out.exists(), '新输出必须在output，不覆盖。')
    require(('smoke_test' in out.parts) == args.smoke, '正式/冒烟输出不匹配。')
    names = ['1019-18-ELDL-V3-短SH纯几何物理转换', '1019-19-ELDL-V3-长SH纯几何物理转换'] if args.smoke else ['1019-12-LH-static-jumpingjacks-ELDL-V3-M06短SH几何初始化24k', '1019-13-LH-static-jumpingjacks-ELDL-V3-M07长SH几何初始化24k']
    bases = [repo / ('output/smoke_test' if args.smoke else 'output') / name for name in names]
    require(model in [p / 'model_mlp' for p in bases], '未登记的初始模型。')
    index = [p / 'model_mlp' for p in bases].index(model)
    # bootstrap保留快照模块优先；仅从工作区取得独立来源审计助手。
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    from LIHAN.scripts.eldl_v3_geometry_record import fixed_geometry_record_files
    marker = read(model / 'eldl_v3_training.json')
    files = fixed_geometry_record_files(model, marker)
    require(marker['args']['geometry_step'] == ([3, 20] if args.smoke else [1000, 16000])[index], '父几何步数不符。')
    require(marker['args']['smoke_schedule'] == args.smoke and Path(marker['args']['source_path']).resolve() == source, '来源/冒烟标记不符。')
    manifest = read(source / 'no_gt_manifest.json')
    require(not manifest['gt_read'] and not manifest['init_weights_loaded'] and manifest['train_frames'] == FRAMES and manifest['dataset'] == 'LH-static-jumpingjacks-ELDL-V3', '需要原135TRAIN。')
    require({p.name for p in source.iterdir()} == {'images', 'points3d.ply', 'transforms_train.json', 'estimated_directions_train.json', 'estimated_eldl_train.json', 'no_gt_manifest.json'}, 'source存在额外输入。')
    approved = [*files, model / 'eldl_v3_training.json', model / 'static_dl_training.json', model / 'cfg_args', model / 'point_cloud/iteration_0/point_cloud.ply', model / 'optimizer_0.pt']
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([source, out, *approved])
    hashes = {str(path): sha(path) for path in approved}
    hashes[str(source / 'no_gt_manifest.json')] = sha(source / 'no_gt_manifest.json')
    for relative, expected in manifest['files'].items():
        path = (source / relative).resolve()
        require(path.is_relative_to(source) and sha(path) == expected, '原source SHA改变。')
        hashes[str(path)] = expected
    require(sha(source / 'estimated_eldl_train.json') == marker['lighting_sha256'], '旧灯资产改变。')
    initial = torch.load(model / 'optimizer_0.pt', map_location='cpu', weights_only=False)
    require(initial['iteration'] == 0 and not initial['gaussian_optimizer']['state'] and not initial['light_optimizer']['state'], '必须零步空Adam起点。')
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from utils.sh_utils import RGB2SH
    from scene.eldl_v3_photometric import FixedELDLLights, render_photometric, censored_symmetric_loss
    import diff_surfel_rasterization as backend
    expected = read(model / 'static_dl_training.json')['rasterizer_provenance']
    require(backend.provenance['source_sha256'] == expected['source_sha256'] and sha(Path(backend._C.__file__)) == expected['extension_sha256'], '实际native并非原控制。')
    hashes[str(Path(backend._C.__file__).resolve())] = expected['extension_sha256']
    cfg = SimpleNamespace(**marker['args'])
    require(cfg.albedo_lr == .01, '原控制DC学习率不是.01。')
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    pc.load_ply(str(model / 'point_cloud/iteration_0/point_cloud.ply'))
    effective = torch.load(model / 'effective_initial_geometry.pt', map_location='cpu', weights_only=True)
    require(set(effective) == set(KEYS), '有效几何包不是四张量。')
    require(all(torch.equal(getattr(pc, k).detach().cpu(), effective[k]) for k in KEYS), '零步生产PLY与登记几何不一致。')
    for value in vars(pc).values():
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
    require(float((pc.get_albedo - .5).abs().max()) < 1e-6, '初始材质不是共同灰值。')
    fixed = {k: v.detach().clone() for k, v in vars(pc).items() if isinstance(v, torch.Tensor) and k != '_albedo_dc'}
    pc._albedo_dc.requires_grad_(True)
    optimizer = torch.optim.Adam([pc._albedo_dc], lr=.01, eps=1e-15)
    lights = FixedELDLLights(source / 'estimated_eldl_train.json')
    frozen_lights = {k: v.detach().clone() for k, v in lights.state_dict().items()}
    cameras = []
    for i, info in enumerate(readCamerasFromTransforms(str(source), 'transforms_train.json', 'images', False, no_bg=True)):
        require(int(info.image_name_train_light.split('_')[-1]) == FRAMES[i], '相机序列改变。')
        cameras.append(loadCam(cfg, i, info, 1.))
    require(len(cameras) == 135, 'TRAIN不完整。')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=1.)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    generator = np.random.default_rng(0)
    original_code = Path(next(iter(marker['new_sources']))).parents[1]
    for relative in (ENTRY, 'scene/eldl_v3_photometric.py', 'scene/stage2_photometric.py', 'scene/gaussian_model.py', 'gaussian_renderer/__init__.py', 'scene/dataset_readers.py', 'scene/cameras.py', 'utils/camera_utils.py', 'utils/rgba16.py'):
        path = CODE_ROOT / relative
        hashes[str(path)] = sha(path)
        if relative != ENTRY:
            original = original_code / relative
            require(sha(original) == hashes[str(path)], '渲染或目标源码与原控制不同。')
            hashes[str(original)] = sha(original)
    out.mkdir()
    epochs, steps = (1, 20) if args.smoke else (30, 4050)
    save_steps = [0, 10, 20] if args.smoke else [0, 1350, 2700, 3375, 4050]
    baseline_maps, scores = {}, {}
    config = dict(actual_entrypoint=ENTRY, source=str(source), initial_model=str(model), initial_physical_step=0, parent_geometry_step=marker['args']['geometry_step'], geometry_binding=marker['geometry_initialization']['geometry_binding'], points=len(pc.get_xyz), train_frames=FRAMES, gt_supervision=False, external_test_read=False, smoke=args.smoke, optimizer='Adam', eps=1e-15, learning_rates=[.01, .001, .0001], steps=steps, full_epochs=epochs if not args.smoke else None, no_visibility=True, target='AREA(RGB*Alpha)，全图sRGB删失Charbonnier', only_trainable_tensor='_albedo_dc', input_sha256=hashes)
    write(out / 'configuration.json', config)

    def forward(camera):
        return render_photometric(camera, pc, pipe, torch.zeros(3, device='cuda'), lights, None, 0)

    @torch.no_grad()
    def save(step):
        require(all(torch.equal(v, getattr(pc, k)) for k, v in fixed.items()), '非DC状态改变。')
        require(all(torch.equal(v, lights.state_dict()[k]) for k, v in frozen_lights.items()), '冻结灯改变。')
        rows = []
        for frame, camera in zip(FRAMES, cameras):
            camera.load2device()
            package = forward(camera)
            photo, _ = censored_symmetric_loss(package['render'], camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda(), camera.observed_saturation_mask)
            require(torch.isfinite(photo), '回放非有限。')
            maps = {k: hashlib.sha256(package[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in ('rend_alpha', 'rend_normal', 'surf_depth')}
            if step == 0:
                baseline_maps[frame] = maps
            else:
                require(maps == baseline_maps[frame], '材质变化改动几何渲染图。')
            rows.append(dict(frame=frame, photo=float(photo), geometry_map_sha256=maps))
            camera.load2device('cpu')
        value = sum(r['photo'] for r in rows) / 135
        scores[str(step)] = value
        write(out / f'fit_step_{step}.json', dict(step=step, frame_mean=value, per_frame=rows, all_frames_seen=True, geometry_maps_bitexact=True))
        torch.save({'_albedo_dc': pc._albedo_dc.detach().cpu().clone()}, out / f'dc_step_{step}.pt')
        torch.save(dict(step=step, optimizer=optimizer.state_dict(), numpy_generator=generator.bit_generator.state, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), out / f'optimizer_step_{step}.pt')
        print(dict(stage='full_TRAIN', step=step, photo=value), flush=True)

    save(0)
    completed = 0
    for epoch in range(1, epochs + 1):
        order = generator.permutation(135)
        for position in order:
            if completed >= steps:
                break
            step = completed + 1
            phase = ((step - 1) // 7 if args.smoke else (epoch - 1) // 10)
            lr = [.01, .001, .0001][min(phase, 2)]
            optimizer.param_groups[0]['lr'] = lr
            camera = cameras[int(position)]
            camera.load2device()
            package = forward(camera)
            loss, _ = censored_symmetric_loss(package['render'], camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda(), camera.observed_saturation_mask)
            require(torch.isfinite(loss), '训练photo非有限。')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = pc._albedo_dc.grad
            require(grad is not None and torch.isfinite(grad).all(), 'DC真实梯度缺失/非有限。')
            require(all(getattr(pc, k).grad is None for k in KEYS), '冻结几何意外接收梯度。')
            optimizer.step()
            with torch.no_grad():
                pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            completed = step
            with (out / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step, epoch=epoch, frame=FRAMES[int(position)], lr=lr, photo=float(loss.detach()), dc_gradient_norm=float(grad.norm()))) + '\n')
            camera.load2device('cpu')
            if step in save_steps:
                save(step)
    require(completed == steps and int(optimizer.state[pc._albedo_dc]['step']) == steps, '实际Adam步数不同。')
    require(all(torch.equal(v, getattr(pc, k)) for k, v in fixed.items()), '末点非DC状态改变。')
    saved_dc = torch.load(out / f'dc_step_{steps}.pt', map_location='cpu', weights_only=True)
    require(set(saved_dc) == {'_albedo_dc'} and torch.equal(saved_dc['_albedo_dc'], pc._albedo_dc.detach().cpu()), '导出DC重载非逐值一致。')
    require(not torch.equal(saved_dc['_albedo_dc'], torch.load(out / 'dc_step_0.pt', map_location='cpu', weights_only=True)['_albedo_dc']), 'DC没有实际变化。')
    if args.smoke:
        with torch.no_grad():
            clone = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
            clone.load_ply(str(model / 'point_cloud/iteration_0/point_cloud.ply'))
            clone._albedo_dc.copy_(saved_dc['_albedo_dc'].cuda())
            require(all(torch.equal(v, getattr(clone, k)) for k, v in fixed.items()), '独立生产重载非DC属性不同。')
            rows = []
            for frame, camera in zip(FRAMES, cameras):
                if frame not in (1, 81, 149):
                    continue
                camera.load2device()
                actual = forward(camera)
                reloaded = render_photometric(camera, clone, pipe, torch.zeros(3, device='cuda'), lights, None, 0)
                differences = {k: float((actual[k] - reloaded[k]).abs().max()) for k in ('render', 'rend_alpha', 'rend_normal', 'surf_depth')}
                require(all(np.isfinite(v) and v < 1e-5 for v in differences.values()), '导出材质独立重载前向不同。')
                rows.append(dict(frame=frame, max_abs=differences))
                camera.load2device('cpu')
            require(len(rows) == 3, '材质重载三个固定TRAIN视角不全。')
            write(out / 'material_reload_audit.json', dict(status='PASS', rows=rows, extra_forward_calls=6, gt_read=False, optimizer_updates=0))
    for path, expected_sha in hashes.items():
        require(sha(Path(path)) == expected_sha, '执行期间输入改变。')
    change = None if args.smoke else abs(scores['4050'] - scores['3375']) / max(scores['3375'], 1e-12)
    eligible = not args.smoke and scores['4050'] < scores['0'] and change <= .01
    manifest = dict(status='COMPLETED_MATERIAL_PREADAPT_PENDING_PAIR_REVIEW', actual_entrypoint=ENTRY, gt_supervision=False, external_test_read=False, smoke=args.smoke, source=str(source), initial_model=str(model), geometry_binding=config['geometry_binding'], parent_geometry_step=config['parent_geometry_step'], points=len(pc.get_xyz), only_tensor='_albedo_dc', completed_optimizer_steps=steps, scores=scores, terminal_absolute_relative_change=change, budget_eligible=eligible, pair_eligible=False, formal_use_permitted=False, no_optimizer_rng_transfer=True, non_dc_bitexact=True, all_geometry_render_maps_bitexact=True, material=str(out / f'dc_step_{steps}.pt'), material_sha256=sha(out / f'dc_step_{steps}.pt'), configuration_sha256=sha(out / 'configuration.json'), input_sha256=hashes)
    write(out / 'material_manifest.json', manifest)
    write(out / 'training_finished.json', dict(status='TRAINING_FINISHED', actual_updates=completed, budget_eligible=eligible, scientific_accuracy_claim=False))
    print(dict(status='MATERIAL_PREADAPT_FINISHED', budget_eligible=eligible, terminal_relative_change=change), flush=True)


if __name__ == '__main__':
    main()
