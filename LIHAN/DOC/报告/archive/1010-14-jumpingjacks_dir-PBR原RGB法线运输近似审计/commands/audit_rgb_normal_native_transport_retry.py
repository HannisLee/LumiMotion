"""原35k预测对应点/135 TRAIN RGB/GT方向的可观测性诊断，不输出训练法线资产。"""
from pathlib import Path
import sys, json, math, hashlib
ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parent.parent
sys.path.insert(0, str(CWD))
import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData
from types import SimpleNamespace
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_geometry import quaternion_normal, project_geometry
from scene.pbr_plane_projection import align_quaternion_normal
from scene.pbr_rasterizer import rasterizer_classes
from scene.pbr_directional import directional_lights_for_camera
from utils.general_utils import build_rotation
from utils.graphics_utils import srgb_to_rgb


@torch.no_grad()
def main():
    run = Stage2RenderRun(ROOT/'model_mlp', 40000)
    ctx = run.context
    st, rt = rasterizer_classes(run.args)
    source_path = ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
    source = PlyData.read(source_path)['vertex'].data
    xyz0 = torch.tensor(np.column_stack([source[k] for k in ['x', 'y', 'z']]), device='cuda')
    q0 = torch.tensor(np.column_stack([source['rot_%d' % i] for i in range(4)]), device='cuda')
    scales0 = torch.tensor(np.column_stack([source['scale_%d' % i] for i in range(2)]), device='cuda').exp()
    opacity0 = torch.sigmoid(torch.tensor(source['opacity'].copy(), device='cuda'))[:, None]
    feature0 = torch.sigmoid(torch.tensor(source['fea_0'].copy(), device='cuda')*2)[:, None]
    original = SimpleNamespace(get_xyz=xyz0, _xyz=xyz0, _rotation=q0)
    base_rotation = build_rotation(q0)
    source_axis = quaternion_normal(q0)
    n = len(xyz0)
    gram = torch.zeros((n, 3, 3), device='cuda')
    rhs = torch.zeros((n, 3), device='cuda')
    observations = torch.zeros(n, device='cuda', dtype=torch.int32)
    y_squared = torch.zeros(n, device='cuda')
    rows = []
    luminosity = torch.tensor([.2126, .7152, .0722], device='cuda')
    train_meta = json.loads((Path(run.args.source_path)/'transforms_train.json').read_text())
    training_keys = [str(int(f['source_frame'])).zfill(4) for f in train_meta['frames']]
    assert len(training_keys) == 135 and all(int(k) % 10 != 0 for k in training_keys)
    for key in training_keys:
        camera = ctx.camera_by_key[key]
        ctx.load_camera(camera)
        d = ctx.deform.step(xyz0, camera.fid.unsqueeze(0).expand(n, -1), iteration=35000,
                           feature=feature0, camera_center=camera.camera_center)
        d = ctx.pbr_plane_projection.apply(original, d)
        xyz = xyz0+d['d_xyz']
        q = q0+d['d_rotation']
        rotation = build_rotation(q)
        settings = st(image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx*.5), tanfovy=math.tan(camera.FoVy*.5), bg=ctx.background,
            scale_modifier=1., viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=run.args.sh_degree, campos=camera.camera_center, prefiltered=False, debug=False)
        rasterizer = rt(raster_settings=settings)
        _, _, buffers = rasterizer(means3D=xyz, means2D=torch.zeros_like(xyz), shs=None,
            colors_precomp=torch.ones_like(xyz), opacities=opacity0,
            scales=scales0+d['d_scaling'], rotations=F.normalize(q, dim=-1), cov3D_precomp=None)
        homogeneous = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1)
        clip = homogeneous@camera.full_proj_transform
        coordinates = clip[:, :2]/clip[:, 3:].clamp_min(1e-8)
        grid = coordinates.reshape(1, 1, n, 2)
        def sample(image):
            return F.grid_sample(image[None], grid, align_corners=False, padding_mode='zeros')[0, :, 0].T
        color = sample(srgb_to_rgb(camera.original_image_train_light.cuda()))
        input_alpha = sample(camera.gt_alpha_mask.to(xyz0))[:, 0]
        predicted_alpha = sample(buffers[1:2])[:, 0]
        expected_z = sample(buffers[:1]/buffers[1:2].clamp_min(1e-6))[:, 0]
        camera_z = (homogeneous@camera.world_view_transform)[:, 2]
        valid = ((clip[:, 3] > 0) & (coordinates.abs() <= 1).all(-1) &
                 (input_alpha >= .99) & (predicted_alpha > .9) &
                 ((expected_z-camera_z).abs() < .025) & torch.isfinite(color).all(-1))
        direction, energy = directional_lights_for_camera(run.args, camera)
        assert direction.shape == (1, 3)
        # 单SUN单位方向；不使用保存的RGB-fitted E，也不读原GT颜色/强度。
        light = F.normalize(direction[0], dim=-1)
        world_axis = rotation[:, :, 2]
        view_sign = torch.where((world_axis*(camera.camera_center-xyz)).sum(-1) >= 0, 1., -1.)
        transported_light = torch.einsum('nij,j->ni', rotation.transpose(1, 2), light)
        transported_light = torch.einsum('nij,nj->ni', base_rotation, transported_light)*view_sign[:, None]
        light_valid = transported_light[valid]
        luminance = color[valid]@luminosity
        gram[valid] += light_valid[:, :, None]*light_valid[:, None, :]
        rhs[valid] += light_valid*luminance[:, None]
        y_squared[valid] += luminance.square()
        observations[valid] += 1
        recovered=sample(camera.rays_d_hw_unnormalized.permute(2,0,1))*camera_z[:,None]+camera.camera_center
        projection_error=float((recovered[valid]-xyz[valid]).abs().max())
        transport_error=float(((transported_light*source_axis).sum(-1)-view_sign*(world_axis*light).sum(-1)).abs().max())
        assert projection_error<1e-4 and transport_error<1e-5
        row = dict(frame=key,own_rgb_observations=int(valid.sum()),projection_roundtrip_max_abs_error=projection_error,source_direction_transport_max_abs_error=transport_error)
        rows.append(row)
        ctx.unload_camera(camera)
    eigen = torch.linalg.eigvalsh(gram)
    conditioned = (observations >= 12) & (eigen[:, 0] > .01*eigen[:, 2].clamp_min(1e-8))
    ridge = gram+torch.eye(3, device='cuda')[None]*1e-5
    vector = torch.linalg.solve(ridge, rhs[:, :, None])[:, :, 0]
    residual = (y_squared-2*(vector*rhs).sum(-1)+torch.einsum('ni,nij,nj->n', vector, gram, vector)).clamp_min(0)
    relative_residual = (residual/y_squared.clamp_min(1e-8)).sqrt()
    reliable = conditioned & torch.isfinite(vector).all(-1) & (vector.norm(dim=-1) > .01) & (relative_residual < .3)
    source_difference = torch.rad2deg(torch.acos((F.normalize(vector, dim=-1)*source_axis).sum(-1).abs().clamp(0, 1)))
    groups = dict(all=torch.ones(n, dtype=torch.bool, device='cuda'), source_static=feature0[:, 0] < .5,
                  source_dynamic=feature0[:, 0] >= .5, projected_source=ctx.pbr_plane_projection.support)
    statistics = {}
    for name, group in groups.items():
        selected = group & reliable
        statistics[name] = dict(points=int(group.sum()), observations_ge12=int((group & (observations >= 12)).sum()),
            conditioned_points=int((group & conditioned).sum()), reliable_linear_points=int(selected.sum()),
            relative_residual_median=float(relative_residual[selected].median()) if selected.any() else None,
            unoriented_source_axis_difference_median_deg=float(source_difference[selected].median()) if selected.any() else None)
    assert torch.isfinite(gram).all() and torch.isfinite(rhs).all()
    # 只在RAM用原RGB拟合候选方向；不输出任何方向/四元数/训练mask资产。
    applicable = reliable & ~ctx.pbr_plane_projection.support
    canonical_target = F.normalize(vector, dim=-1)
    canonical_target *= torch.where((canonical_target*source_axis).sum(-1, keepdim=True) < 0, -1., 1.)
    canonical_target = torch.where(applicable[:,None], canonical_target, source_axis)
    candidate_q = align_quaternion_normal(q0, canonical_target)
    candidate_q = torch.where(applicable[:,None], candidate_q, q0)
    project_geometry(xyz0.clone(), candidate_q, xyz0, q0, .08, 60.)
    canonical_candidate = quaternion_normal(candidate_q)
    candidate_source_angle = torch.rad2deg(torch.acos((canonical_candidate.double()*source_axis.double()).sum(-1).abs().clamp(0,1)))
    angular_sum = torch.zeros(n, device='cuda', dtype=torch.float64)
    angular_max = torch.zeros_like(angular_sum)
    per_frame_transport = []
    proposed = SimpleNamespace(get_xyz=xyz0, _xyz=xyz0, _rotation=candidate_q)
    for key in training_keys:
        camera = ctx.camera_by_key[key]
        ctx.load_camera(camera)
        raw_d = ctx.deform.step(xyz0, camera.fid.unsqueeze(0).expand(n,-1), iteration=35000,
                              feature=feature0, camera_center=camera.camera_center)
        source_d = ctx.pbr_plane_projection.apply(original, raw_d)
        candidate_d = ctx.pbr_plane_projection.apply(proposed, raw_d)
        source_world_rotation = build_rotation(q0+source_d['d_rotation']).double()
        approximate = torch.einsum('nij,nj->ni', source_world_rotation,
                                  torch.einsum('nij,nj->ni', base_rotation.double().transpose(1,2), canonical_candidate.double()))
        approximate = F.normalize(approximate,dim=-1)
        exact = quaternion_normal((candidate_q+candidate_d['d_rotation']).double())
        bias = torch.rad2deg(torch.acos((approximate*exact).sum(-1).abs().clamp(0,1)))
        assert torch.isfinite(bias).all()
        angular_sum += bias
        angular_max = torch.maximum(angular_max,bias)
        per_frame_transport.append(dict(frame=key,candidate_points=int(applicable.sum()),
            mean_deg=float(bias[applicable].mean()),p99_deg=float(torch.quantile(bias[applicable],.99)),maximum_deg=float(bias[applicable].max())))
        ctx.unload_camera(camera)
    def quantiles(values):
        values = values.double()
        return dict(mean=float(values.mean()),median=float(values.median()),p90=float(torch.quantile(values,.9)),
                    p99=float(torch.quantile(values,.99)),maximum=float(values.max())) if len(values) else None
    exact_transport_statistics = {}
    for label, group in groups.items():
        selected = applicable & group
        exact_transport_statistics[label] = dict(candidate_points=int(selected.sum()),
            canonical_candidate_angle_deg=quantiles(candidate_source_angle[selected]),
            average_native_vs_rotation_transport_bias_deg=quantiles((angular_sum/len(training_keys))[selected]),
            worst_time_native_vs_rotation_transport_bias_deg=quantiles(angular_max[selected]))
    ctx.verify_frozen()
    manifests = json.loads((ROOT/'code_manifest.json').read_text())
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest() == h for k, h in manifests.items())
    result = dict(status='PASS_READ_ONLY_RGB_NORMAL_NATIVE_TRANSPORT_DIAGNOSTIC', source_iteration=35000,
        projection_and_source_transport_verified=True, camera_frames=135, geometry_from='original source35k plus existing source-only plane projection',
        camera_and_network_container_checkpoint=40000, learned_40k_xyz_quaternion_material_used=False,
        statistics=statistics, rows=rows, native_transport_statistics=exact_transport_statistics,
        per_frame_native_transport=per_frame_transport, candidate_points=int(applicable.sum()), candidate_plane_points=0,
        geometry_trust_angle_degrees=60., source_feature_not_learned_40k=True,
        optimizer_steps=0, gt_labels_read=False,
        gt_light_photometry_read=False, training_target_vectors_or_prior_saved=False,
        source_file_sha=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        limitations='只是Lambertian线性可观测性，不是原生GGX/影子模型拟合或GT normal验证。可见性用来源预测Z距.025与原RGBA/来源alpha，不是GT深度；固定来源旋转的切向运输与来源camera-facing符号只是近似。禁止从此统计宣称方向准确或7度达标。本诊断仅在RAM由原RGB生成可靠候选，已有硬plane支持不作为新旋转候选，60度canonical信赖域保持；比较真实raw quaternion加固定source MLP偏置与旧source旋转运输的近似误差，不是GT normal误差。40k learned feature/XYZ/q/material完全不用。不输出拟合法线/四元数/训练监督或评分mask资产。')
    path = ROOT/'inspection/rgb_normal_native_transport_diagnostic.json'
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
