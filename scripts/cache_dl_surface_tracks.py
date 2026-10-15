"""从默认 Stage1 和训练 RGB 构建可审计表面轨迹，不读取 GT 灯光或测试集。"""
import argparse
import hashlib
import json
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
from scene.stage2_photometric import srgb_to_linear
from utils.camera_utils import loadCam
from utils.general_utils import build_rotation
from scripts.probe_dynamic_dl_selfcal import read_config, save_image


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--iteration', type=int, default=35000)
    p.add_argument('--resolution', type=int, default=4)
    p.add_argument('--patches', type=int, default=8192)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'tracks.pt').exists():
        raise FileExistsError('禁止覆盖观测缓存。')
    model, source = Path(args.model), Path(args.source)
    cfg = read_config(model / 'cfg_args')
    cfg.resolution, cfg.load2gpu_on_the_fly = args.resolution, True
    if cfg.white_background:
        raise ValueError('首版要求黑背景。')
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    ply = model / f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    pc.load_ply(str(ply))
    deform = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender,
                         hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    assert deform.load_weights(str(model), args.iteration)
    deform.deform.eval()
    # 每个 canonical 体素仅保留一个高 opacity 代表；不按观察颜色或 GT 筛选。
    xyz = pc.get_xyz.cpu().numpy()
    lower, upper = xyz.min(0), xyz.max(0)
    cells = np.floor((xyz - lower) / np.maximum(upper - lower, 1e-6) * 63).astype(np.int64)
    keys = cells[:, 0] + 64 * cells[:, 1] + 4096 * cells[:, 2]
    priority = np.argsort(-pc.get_opacity[:, 0].cpu().numpy(), kind='stable')
    _, first = np.unique(keys[priority], return_index=True)
    ids = priority[first]
    if len(ids) > args.patches:
        ids = np.random.default_rng(1015).choice(ids, args.patches, replace=False)
    ids = torch.tensor(np.sort(ids), device='cuda')
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', cfg.train_light_folder, False)
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=1.)
    background = torch.zeros(3, device='cuda')
    records, frames, hashes = [], [], {}
    for index, info in enumerate(infos):
        camera = loadCam(cfg, index, info, 1.)
        camera.load2device()
        ds = deform.step(pc.get_xyz, camera.fid.reshape(1, 1).expand(len(xyz), 1),
                         feature=pc.get_binary_feature(), camera_center=camera.camera_center)
        ds = {k: ds[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color']}
        package = render(camera, pc, pipe, background, **ds)
        points = (pc.get_xyz + ds['d_xyz'])[ids]
        homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], -1)
        clip = homogeneous @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:].clamp_min(1e-8)
        h, w = camera.image_height, camera.image_width
        u = ((ndc[:, 0] + 1) * w / 2 - .5).round().long()
        v = ((ndc[:, 1] + 1) * h / 2 - .5).round().long()
        in_image = (clip[:, 3] > 0) & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        u, v = u.clamp(0, w - 1), v.clamp(0, h - 1)
        depth = (homogeneous @ camera.world_view_transform)[:, 2]
        reference_depth = package['surf_depth'][0, v, u]
        depth_error = (depth - reference_depth).abs() / depth.clamp_min(1e-6)
        normals = build_rotation(pc.get_rotation_bias(ds['d_rotation']))[ids, :, 2]
        facing = torch.where((normals * (camera.camera_center - points)).sum(-1, keepdim=True) >= 0, 1., -1.)
        normals = normals * facing
        image_normal = F.normalize(package['rend_normal'][:, v, u].T, dim=-1)
        normal_agreement = (normals * image_normal).sum(-1)
        original = np.array(Image.open(camera.image_path_train_light).convert('RGBA'))
        saturated = torch.from_numpy((original[..., :3] >= 254).astype(np.float32)).permute(2, 0, 1)[None].cuda()
        saturated = F.adaptive_max_pool2d(saturated, (h, w))[0, :, v, u].T.bool()
        alpha = camera.gt_alpha_mask.to('cuda')[0]
        # 3x3 全部在前景内，且预测 Alpha、深度与像素法线支持该表面。
        interior = -F.max_pool2d(-alpha[None, None], 3, stride=1, padding=1)[0, 0]
        valid = in_image & (interior[v, u] > .99) & (package['rend_alpha'][0, v, u] > .95)
        valid &= (depth_error < .01) & (normal_agreement > .8)
        before_unique = int(valid.sum())
        # 同一帧同一像素最多一条观测，优先选择预测深度最接近者。
        good = torch.where(valid)[0].cpu().numpy()
        order = good[np.argsort(depth_error[good].cpu().numpy(), kind='stable')]
        pixels = (v * w + u).cpu().numpy()
        _, unique = np.unique(pixels[order], return_index=True)
        valid.zero_()
        valid[torch.as_tensor(order[unique], device='cuda')] = True
        rgb = srgb_to_linear(camera.original_image_train_light[:, v, u].T)
        sh_rgb = srgb_to_linear(package['render'][:, v, u].T.clamp(0, 1))
        records.append(dict(normals=normals.cpu(), rgb=rgb.cpu(), sh_rgb=sh_rgb.cpu(),
                            valid=valid.cpu(), saturated=saturated.cpu(), depth_error=depth_error.cpu(),
                            normal_agreement=normal_agreement.cpu(), facing=facing[:, 0].cpu(),
                            positions=points.cpu(), pixels=torch.stack([u, v], -1).cpu()))
        frames.append(dict(name=camera.image_name_train_light, time=float(camera.fid.item()),
                           valid=int(valid.sum()), before_unique=before_unique,
                           saturated_valid=int((saturated.any(-1) & valid).sum())))
        image_path = Path(camera.image_path_train_light)
        hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if index in [0, len(infos)//2, len(infos)-1]:
            name = camera.image_name_train_light
            save_image(out / f'{name}_rgb.png', package['render'])
            save_image(out / f'{name}_alpha.png', package['rend_alpha'])
            save_image(out / f'{name}_normal.png', (F.normalize(package['rend_normal'], dim=0)*.5+.5)*package['rend_alpha'])
            gate = pc.get_binary_feature()
            sep = render(camera, pc, pipe, background, **ds,
                         override_color=torch.cat([gate, 1-gate, torch.zeros_like(gate)], -1))
            save_image(out / f'{name}_separation.png', sep['render'])
            image = camera.original_image_train_light.clone()
            image[:, v[valid], u[valid]] = image.new_tensor([1., 0., 0.])[:, None]
            save_image(out / f'{name}_samples.png', image)
        print(json.dumps(dict(index=index, **frames[-1])), flush=True)
        del camera, package, ds
    data = {k: torch.stack([r[k] for r in records]) for k in records[0]}
    data.update(times=torch.tensor([f['time'] for f in frames]), ids=ids.cpu(),
                canonical_xyz=pc.get_xyz[ids].cpu(), frames=frames, config=vars(args))
    torch.save(data, out / 'tracks.pt')
    for path in [ply, model/f'deform/iteration_{args.iteration}/deform.pth', model/'cfg_args', source/'transforms_train.json', Path(__file__)]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(config=vars(args), frames=frames, patches=len(ids), input_hashes=hashes,
                    native=native_module().provenance, stage1_saw_all_training_images=True,
                    gt_read=False, normal_transport='默认旋转头，需进一步核实与切平面一致性',
                    limitations='代表点并非已验证物质对应；预测可见性不包含光源阴影',
                    cache_sha256=hashlib.sha256((out/'tracks.pt').read_bytes()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
