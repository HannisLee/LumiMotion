"""只读source支持与原生合成贡献；不读取GT标签、不改变训练或评分。"""
from pathlib import Path
import sys, json, hashlib, math, datetime
import numpy as np
import torch
from plyfile import PlyData
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parent.parent
sys.path.insert(0, str(CWD))
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_rasterizer import rasterizer_classes
from scripts.stage2_runtime import supervision
from scene.pbr_geometry import quaternion_normal


def stats(array):
    array = np.asarray(array)
    return dict(count=len(array), mean=float(array.mean()),
                q0_q50_q90_q99_q100=np.quantile(array, [0, .5, .9, .99, 1]).tolist())


@torch.no_grad()
def main():
    dest = ROOT/'inspection/constraint_pixel_mass_40000'
    dest.mkdir(exist_ok=False)
    run = Stage2RenderRun(ROOT/'model_mlp', 40000)
    ctx = run.context
    pc = ctx.pc
    source = ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
    vertex = PlyData.read(source)['vertex'].data
    xyz = np.column_stack([vertex[k] for k in ['x', 'y', 'z']])
    source_q = np.column_stack([vertex[f'rot_{k}'] for k in range(4)])
    opacity = 1/(1+np.exp(-vertex['opacity']))
    static = vertex['fea_0'] < 0
    with np.load(ROOT/'model_mlp/pbr_plane_projection.npz') as data:
        arrays = {k: data[k].copy() for k in data.files}
    canonical_dist = np.abs(xyz@arrays['plane']+arrays['offset'])
    position_stable = ((opacity > .05) & (arrays['motion_span'] <= .01))
    bands = {}
    for band in [.01, .025, .05, .1, .2]:
        bands[str(band)] = (position_stable & (canonical_dist <= band) &
                            (arrays['max_plane_distance'] <= band))
    confidence = ctx.pbr_rgb_normal.confidence.cpu().numpy() > 0
    projected = arrays['support']
    assert len(xyz) == 99827 and int(confidence.sum()) == 29621 and int(projected.sum()) == 15187
    groups = dict(RGB_confidence=confidence, existing_plane=projected,
                  source_position_stable_plane_005=bands['0.05'])
    point_stats = {
        name: dict(points=int(mask.sum()), source_static_points=int((mask & static).sum()),
                   source_dynamic_points=int((mask & ~static).sum()),
                   original_opacity=stats(opacity[mask]),
                   motion_span=stats(arrays['motion_span'][mask]))
        for name, mask in groups.items()
    }
    normal = quaternion_normal(torch.tensor(source_q)).numpy()
    cosine = np.abs(normal@arrays['plane'])
    bands_report = {
        key: dict(points=int(mask.sum()), static_points=int((mask & static).sum()),
                  dynamic_points=int((mask & ~static).sum()),
                  overlapping_RGB_confidence=int((mask & confidence).sum()),
                  overlapping_existing_plane=int((mask & projected).sum()),
                  rejected_by_canonical_axis_cosine=int((mask & (cosine < .5)).sum()),
                  rejected_by_any_source_time_axis_cosine=int((mask & (arrays['min_axis_cosine'] < .5)).sum()))
        for key, mask in bands.items()
    }
    colors = torch.tensor(np.column_stack(list(groups.values())).astype(np.float32), device='cuda')
    Settings, Rasterizer = rasterizer_classes(run.args)
    rows = []
    for frame in ['0001', '0075', '0149']:
        camera = ctx.camera_by_key[frame]
        ctx.load_camera(camera)
        deformation = ctx.deformation(camera)
        means = pc.get_xyz+deformation['d_xyz']
        rotations = pc.get_rotation_bias(deformation['d_rotation'])
        scales = pc.get_scaling+deformation['d_scaling']
        actual_opacity = pc.get_opacity
        if deformation['d_opacity'] is not None:
            actual_opacity = actual_opacity+deformation['d_opacity']
        settings = Settings(
            image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx/2), tanfovy=math.tan(camera.FoVy/2),
            bg=torch.zeros(3, device='cuda'), scale_modifier=1.,
            viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=pc.active_sh_degree, campos=camera.camera_center,
            prefiltered=False, debug=False)
        image, _, maps = Rasterizer(settings)(
            means3D=means, means2D=torch.zeros_like(means), shs=None,
            colors_precomp=colors, opacities=actual_opacity,
            scales=scales, rotations=rotations, cov3D_precomp=None)
        alpha = maps[1]
        _, own_alpha = supervision(camera, ctx.background)
        foreground = own_alpha[0] >= .5
        visible = foreground & (alpha >= .5)
        assert torch.isfinite(image).all() and torch.isfinite(maps).all()
        assert (image >= -1e-6).all() and (image <= alpha[None]+2e-5).all()
        current = dict(frame=frame, own_RGBA_foreground_pixels=int(foreground.sum()),
                       visible_pixels=int(visible.sum()), group_mass={})
        for i, name in enumerate(groups):
            weight = image[i]
            ratio = weight/alpha.clamp_min(1e-6)
            current['group_mass'][name] = dict(
                alpha_weight_fraction_over_own_foreground=float(weight[foreground].sum()/alpha[foreground].sum()),
                mean_pixel_fraction_over_visible=float(ratio[visible].mean()),
                visible_pixel_fraction_with_majority_group=float((ratio[visible] >= .5).float().mean()))
            Image.fromarray((ratio.clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)).save(
                dest/(frame+'_'+name+'.png'))
        np.savez_compressed(dest/(frame+'_native_weights.npz'),
                            group_weight=image.cpu().numpy(), alpha=alpha.cpu().numpy(),
                            own_RGBA_foreground=foreground.cpu().numpy())
        rows.append(current)
        ctx.unload_camera(camera)
    ctx.verify_frozen()
    manifest = json.loads((ROOT/'code_manifest.json').read_text())
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest() == value for k, value in manifest.items())
    result = dict(
        status='PASS_READ_ONLY_NATIVE_PIXEL_CONTRIBUTIONS', iteration=40000,
        source_iteration=35000, created_at=datetime.datetime.now().astimezone().isoformat(),
        point_stats=point_stats, source_position_stable_band_counts=bands_report, rows=rows,
        actual_checkpoint_files=ctx.source_files, all_points_retained=True,
        GT_labels_read=False, optimizer_steps=0, training_prior_generated=False,
        production_support_changed=False, scoring_mask_generated=False, full_MAE=False,
        limitations='RGB贡献图仅诊断真实前后顺序alpha合成权重；三TRAIN自身Alpha不当GT法线评分。'
                    'source band集合是位置/全135TRAIN形变稳定候选，未施加新投影，也不是物体GT分类；'
                    '不能凭候选数量扩大生产支持或声称7度。')
    (dest/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
