"""B07 40k只读联合光栅贡献：来源near-plane未投影组；不读GT标签。"""
from pathlib import Path
import sys, json, math, hashlib
ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parent.parent
sys.path.insert(0, str(CWD))
import torch
import numpy as np
import torch.nn.functional as F
from plyfile import PlyData
from scripts.render_stage2 import Stage2RenderRun, save_image
from scene.pbr_geometry import quaternion_normal
from scene.pbr_rasterizer import rasterizer_classes
from gaussian_renderer.render_ir import render_ir


@torch.no_grad()
def main():
    run = Stage2RenderRun(ROOT/'model_mlp', 40000)
    ctx, pc = run.context, run.context.pc
    settings_type, rasterizer_type = rasterizer_classes(run.args)
    plane, offset = ctx.pbr_plane_projection.plane, ctx.pbr_plane_projection.offset
    source = PlyData.read(ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply')['vertex'].data
    xyz0 = torch.tensor(np.column_stack([source[k] for k in ['x', 'y', 'z']]), device='cuda')
    q0 = torch.tensor(np.column_stack([source['rot_%d' % i] for i in range(4)]), device='cuda')
    feature = torch.sigmoid(torch.tensor(source['fea_0'].copy(), device='cuda')*2)
    alpha0 = torch.sigmoid(torch.tensor(source['opacity'].copy(), device='cuda'))
    with np.load(ROOT/'model_mlp/pbr_plane_projection.npz', allow_pickle=False) as arrays:
        span = torch.tensor(arrays['motion_span'].copy(), device='cuda')
        distance = torch.tensor(arrays['max_plane_distance'].copy(), device='cuda')
        cosine = torch.tensor(arrays['min_axis_cosine'].copy(), device='cuda')
    near = ((alpha0 > .05) & ((xyz0@plane+offset).abs() <= run.args.pbr_plane_support_band) &
            (distance <= run.args.pbr_plane_support_band) &
            ((quaternion_normal(q0)@plane).abs() >= .5) & (cosine >= .5))
    projected = ctx.pbr_plane_projection.support
    static_unprojected = near & (feature < .5) & ~projected
    dynamic_near = near & (feature >= .5)
    other = ~(projected | static_unprojected | dynamic_near)
    names = ['existing_projected', 'source_static_near_unprojected', 'source_dynamic_near_unprojected', 'other']
    masks = [projected, static_unprojected, dynamic_near, other]
    assert torch.stack(masks).sum(0).eq(1).all()
    assert all(span[static_unprojected] > run.args.pbr_plane_motion_span)
    source_stats = {}
    for name, mask in zip(names, masks):
        values = span[mask]
        source_stats[name] = dict(points=int(mask.sum()),
            motion_span_percentiles={str(p): float(torch.quantile(values, p)) for p in [.0, .5, .9, .99, 1.]},
            motion_threshold_counts={str(t): int((values <= t).sum()) for t in [.01, .02, .04, .08, .16]})
    destination = ROOT/'inspection/unprojected_plane_groups_40000'
    destination.mkdir(exist_ok=False)
    rows = []
    colors = torch.stack(masks[:3], dim=1).float()
    other_colors = other[:, None].float().expand(-1, 3)
    for frame in ['0001', '0075', '0149']:
        camera = ctx.camera_by_key[frame]
        ctx.load_camera(camera)
        d = ctx.deformation(camera)
        xyz = pc.get_xyz+d['d_xyz']
        settings = settings_type(image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx*.5), tanfovy=math.tan(camera.FoVy*.5), bg=ctx.background,
            scale_modifier=1., viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=pc.active_sh_degree, campos=camera.camera_center, prefiltered=False, debug=False)
        rasterizer = rasterizer_type(raster_settings=settings)
        opacity = pc.get_opacity if d['d_opacity'] is None else pc.get_opacity+d['d_opacity']
        kwargs = dict(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
            scales=pc.get_scaling+d['d_scaling'], rotations=pc.get_rotation_bias(d['d_rotation']), cov3D_precomp=None)
        rgb, _, buffers = rasterizer(colors_precomp=colors, opacities=opacity, **kwargs)
        other_rgb, _, other_buffers = rasterizer(colors_precomp=other_colors, opacities=opacity, **kwargs)
        reference = render_ir(camera, pc, run.args, ctx.background,
            **{k: d[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color']}, material_only=True)
        world = (buffers[2:5].permute(1, 2, 0)@camera.world_view_transform[:3, :3].T).permute(2, 0, 1)
        weights = torch.cat([rgb, other_rgb[:1]], 0)
        errors = dict(alpha=float((buffers[1:2]-reference['rend_alpha']).abs().max()),
                      worldnormal=float((world-reference['rend_normal']).abs().max()),
                      expectedZ=float((buffers[0:1]/buffers[1:2].clamp_min(1e-6)-reference['surf_depth']).abs().max()),
                      second_pass_buffer=float((buffers-other_buffers).abs().max()),
                      group_sum_alpha=float((weights.sum(0)-buffers[1]).abs().max()))
        assert max(errors.values()) < 2e-5
        alpha = buffers[1]
        valid = alpha > .5
        dominant = weights.argmax(0)
        row = dict(frame=frame, buffer_max_errors=errors,
            joint_groups={name: dict(points=int(mask.sum()), accumulated_weight_sum=float(weights[i].sum()),
                    dominant_alpha05_pixels=int((valid & (dominant == i)).sum()))
                for i, (name, mask) in enumerate(zip(names, masks))})
        rows.append(row)
        palette = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [1., 1., 1.]], device='cuda')
        save_image(destination/(frame+'_joint_groups.png'), torch.einsum('ki,kxy->ixy', palette, weights))
        save_image(destination/(frame+'_normal.png'), (F.normalize(world, dim=0)*.5+.5)*alpha)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        ctx.unload_camera(camera)
    ctx.verify_frozen()
    manifest = json.loads((ROOT/'code_manifest.json').read_text())
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest() == h for k, h in manifest.items())
    result = dict(status='PASS_READ_ONLY_UNPROJECTED_PLANE_JOINT_AUDIT', iteration=40000,
        frames=['0001', '0075', '0149'], source_stats=source_stats, rows=rows, optimizer_steps=0,
        GT_geometry_normal_material_inputs=False, training_modified=False, final_mae=False,
        all_gaussian_opacity_retained=True, source_files=ctx.source_files,
        limits='source near-plane与static/dynamic是固定35k预测分类，不是真实平台或物体标签；只改诊断颜色，所有99827点/原opacity/geometry保留。无训练先验/新监督/评分mask输出。')
    path = ROOT/'inspection/unprojected_plane_groups.json'
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
