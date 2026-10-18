"""只用已见135 TRAIN比较冻结检查点的预测几何变化，稳定性不代表准确性。"""
import argparse
import json
import hashlib
import math
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_eldl_v3_raster_lighting import sha, read, require


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--iterations', type=int, nargs='+', required=True)
    a = parser.parse_args()
    a.model, a.source, a.output = a.model.resolve(), a.source.resolve(), a.output.resolve()
    require(socket.gethostname() == 'ushas', '需要ushas及对应环境')
    require(not a.output.exists(), '禁止覆盖已有诊断')
    a.output.mkdir(parents=True)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([a.model, a.source, a.output])
    from scene.gaussian_model import GaussianModel
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    from scripts.probe_dynamic_dl_selfcal import read_config
    from scene.static_dl_photometric import StaticVisibility
    from scene.eldl_v3_photometric import FixedELDLLights, render_photometric, censored_symmetric_loss
    marker = read(a.model / 'eldl_v3_training.json')
    source = read(a.source / 'no_gt_manifest.json')
    frames = [i for i in range(1, 151) if i % 10]
    require(marker['gt_supervision'] is False and marker['lighting_frozen'] and
            marker['actual_entrypoint'] == 'scripts/train_eldl_v3.py', '必须冻结混合光无GT模型')
    require(source['gt_read'] is False and source['init_weights_loaded'] is False and
            source['train_frames'] == frames, '需要原135 TRAIN')
    require(Path(marker['args']['source_path']).resolve() == a.source, '模型源不匹配')
    hashes = {str(a.source / 'no_gt_manifest.json'): sha(a.source / 'no_gt_manifest.json'),
              str(Path(__file__).resolve()): sha(__file__)}
    for relative, expected in source['files'].items():
        path = (a.source / relative).resolve()
        require(path.is_relative_to(a.source) and sha(path) == expected, 'TRAIN源SHA变化')
    cfg = read_config(a.model / 'cfg_args')
    cfg.resolution = 2
    cfg.load2gpu_on_the_fly = True
    cameras = []
    for i, info in enumerate(readCamerasFromTransforms(str(a.source), 'transforms_train.json', 'images', False)):
        require(int(info.image_name_train_light.split('_')[-1]) == frames[i], '相机序列变化')
        cam = loadCam(cfg, i, info, 1.)
        cam.load2device()
        cameras.append(cam)
    lights = FixedELDLLights(a.source / 'estimated_eldl_train.json')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.)
    endpoints = []
    previous_geometry = {}
    previous_iteration = None
    for iteration in a.iterations:
        require(iteration >= 5000, '仅比较完整可见性物理阶段')
        ply = a.model / f'point_cloud/iteration_{iteration}/point_cloud.ply'
        deform = a.model / f'deform/iteration_{iteration}/deform.pth'
        saved_light = a.model / f'static_dl_lighting/iteration_{iteration}/lighting.json'
        hashes.update({str(p): sha(p) for p in (ply, deform, saved_light)})
        saved = read(saved_light)
        require(saved['frozen_light_check'] == 'PASS' and saved['lighting_asset'] == lights.asset, '灯资产不一致')
        weights = torch.load(deform, map_location='cpu', weights_only=True)
        for head in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
            for suffix in ('weight', 'bias'):
                require(torch.count_nonzero(weights[f'{head}.{suffix}']) == 0, '静态几何头非零')
        pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
        pc.load_ply(str(ply))
        trace = StaticVisibility(bias=marker['args']['static_dl_shadow_bias'], update_interval=1000000)
        rows = []
        changes = []
        for frame, cam in zip(frames, cameras):
            package = render_photometric(cam, pc, pipe, torch.zeros(3, device='cuda'), lights, trace, iteration)
            target, alpha = cam.original_image_train_light.cuda(), cam.gt_alpha_mask.cuda()
            photo, _ = censored_symmetric_loss(package['render'], target, alpha, cam.observed_saturation_mask)
            pred_mask, mask = package['rend_alpha'] >= .5, alpha >= .5
            depth = package['surf_depth']
            valid = pred_mask & torch.isfinite(depth) & (depth > 0)
            rows.append(dict(frame=frame, photo=float(photo), alpha_l1=float((package['rend_alpha']-alpha).abs().mean()),
                             alpha_iou=float((pred_mask & mask).sum()/(pred_mask | mask).sum().clamp_min(1)),
                             source_foreground_coverage=float((valid & mask).sum()/mask.sum().clamp_min(1)),
                             mean_visibility=float(package['mean_visibility'])))
            normal = torch.nn.functional.normalize(package['rend_normal'], dim=0)
            norm_valid = torch.isfinite(normal).all(0) & (normal.norm(dim=0) > .5)
            support = mask[0]
            current = dict(depth=depth[0][support].cpu(), normal=normal[:, support].cpu(),
                           valid=(valid[0] & norm_valid)[support].cpu())
            digest = hashlib.sha256()
            for key in ('depth', 'normal', 'valid'):
                digest.update(current[key].numpy().tobytes())
            rows[-1]['predicted_geometry_sha256'] = digest.hexdigest()
            if frame in previous_geometry:
                old = previous_geometry[frame]
                common = old['valid'] & current['valid']
                count, denominator = int(common.sum()), len(common)
                delta = (old['depth'][common] - current['depth'][common]).abs()
                angles = (old['normal'][:, common] * current['normal'][:, common]).sum(0).clamp(-1, 1).acos() * (180/math.pi)
                relative = (delta / old['depth'][common].clamp_min(1e-6)).clamp(max=1)
                changes.append(dict(frame=frame, previous_iteration=previous_iteration,
                    common_pixels=count, fixed_source_alpha_pixels=denominator,
                    common_coverage=count/max(1,denominator),
                    expected_depth_change_sum=float(delta.sum()),
                    expected_depth_change_mae=float(delta.mean()) if count else None,
                    gs_normal_change_sum_deg=float(angles.sum()),
                    gs_normal_change_mae_deg=float(angles.mean()) if count else None,
                    gs_normal_change_missing180_sum=float(angles.sum())+180*(denominator-count),
                    relative_depth_change_missing1_sum=float(relative.sum())+denominator-count))
            previous_geometry[frame] = current
        result = dict(iteration=iteration, points=len(pc.get_xyz), rows=rows,
                      frame_mean={k:sum(row[k] for row in rows)/len(rows) for k in rows[0] if k not in ('frame', 'predicted_geometry_sha256')})
        if changes:
            valid_count = sum(row['common_pixels'] for row in changes)
            denominator = sum(row['fixed_source_alpha_pixels'] for row in changes)
            result['geometry_change'] = dict(previous_iteration=previous_iteration, rows=changes,
                common_coverage=valid_count/max(1,denominator),
                expected_depth_change_pixel_mae=sum(row['expected_depth_change_sum'] for row in changes)/max(1,valid_count),
                gs_normal_change_pixel_mae_deg=sum(row['gs_normal_change_sum_deg'] for row in changes)/max(1,valid_count),
                gs_normal_change_missing180_pixel_mean_deg=sum(row['gs_normal_change_missing180_sum'] for row in changes)/max(1,denominator),
                relative_depth_change_missing1_pixel_mean=sum(row['relative_depth_change_missing1_sum'] for row in changes)/max(1,denominator))
        previous_iteration = iteration
        endpoints.append(result)
        (a.output / f'iteration_{iteration}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(dict(iteration=iteration, **result['frame_mean']), flush=True)
        del pc, trace
    summary = dict(status='PASS_TRAIN_GEOMETRY_STABILITY_EXECUTION', gt_read=False, external_test_read=False,
                   all_frames_seen_during_training=True, no_parameter_updates=True,
                   accuracy_claim=False, depth_statistic='expected_Z', normal_statistic='signed_GS',
                   limitation='相邻冻结预测的一致性不是GT精度；固定分母同时报告缺失，不能通过删掉不稳定像素改善统计',
                   source=str(a.source), model=str(a.model), input_sha256=hashes,
                   endpoints=[{k:v for k,v in row.items() if k!='rows'} for row in endpoints])
    (a.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
