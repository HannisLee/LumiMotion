"""逐点源NN形变导数诊断，0 optimizer、0 GT标签，不改生产或生成teacher。"""
from pathlib import Path
import sys, json, hashlib, datetime
import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent.parent
CWD = ROOT.parents[2]
sys.path.insert(0, str(CWD))
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_feature_gate import native_probability
from scene.pbr_geometry import quaternion_normal


def stats(value):
    value = value.detach().double()
    return dict(points=len(value), mean=float(value.mean()),
                q0_q50_q90_q99_q100=torch.quantile(
                    value, value.new_tensor([0., .5, .9, .99, 1.])).cpu().tolist())


def main():
    base = CWD/'output/1010-16-jumpingjacks_dir-PBR训练RGB来源局部法线约束'
    run = Stage2RenderRun(base/'model_mlp', 40000)
    ctx = run.context
    source = base/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
    mlp = base/'model_mlp/deform/iteration_35000/deform.pth'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529'
    assert hashlib.sha256(mlp.read_bytes()).hexdigest() == 'e9ddf76b59a1ef8a85caa16891473e3a60663d903a5ad3928ac3e6ca46892bf5'
    vertex = PlyData.read(source)['vertex'].data
    xyz = torch.tensor(np.column_stack([vertex[x] for x in ['x', 'y', 'z']]), device='cuda')
    q = torch.tensor(np.column_stack([vertex[f'rot_{i}'] for i in range(4)]), device='cuda')
    gate = native_probability(torch.tensor(vertex['fea_0'].copy(), device='cuda'))[:, None]
    opacity = torch.sigmoid(torch.tensor(vertex['opacity'].copy(), device='cuda'))
    axis = quaternion_normal(q)
    static = gate[:, 0] < .5
    with np.load(base/'model_mlp/pbr_plane_projection.npz') as data:
        arrays = {k: torch.tensor(data[k].copy(), device='cuda') for k in data.files}
    near = (~static & (opacity > .05) & ((xyz@arrays['plane']+arrays['offset']).abs() <= .2) &
            (arrays['max_plane_distance'] <= .2) & ((axis@arrays['plane']).abs() >= .5) &
            (arrays['min_axis_cosine'] >= .5))
    groups = dict(all=torch.ones_like(static), source_projected=arrays['support'],
                  source_dynamic_near_plane=near, source_other_dynamic=~static & ~near,
                  source_static_unprojected=static & ~arrays['support'])
    assert sum(int(groups[x].sum()) for x in groups if x != 'all') == len(xyz) == 99827
    assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
    rows = []
    for frame in ['0001', '0031', '0061', '0091', '0121', '0149']:
        camera = ctx.camera_by_key[frame]
        ctx.load_camera(camera)
        parts = {name: [] for name in ['jacobian', 'determinant', 'condition', 'normal', 'native_normal', 'valid']}
        for start in range(0, len(xyz), 8192):
            end = min(start+8192, len(xyz))
            point = xyz[start:end].detach().clone().requires_grad_(True)
            prediction = ctx.deform.step(
                point, camera.fid.unsqueeze(0).expand(len(point), -1), iteration=35000,
                feature=gate[start:end], camera_center=camera.camera_center, is_training=False)
            warp = prediction['d_xyz']
            derivative = torch.stack([
                torch.autograd.grad(warp[:, i].sum(), point, retain_graph=i < 2)[0]
                for i in range(3)], dim=1).detach()
            jacobian = derivative+torch.eye(3, device='cuda')[None]
            assert torch.isfinite(jacobian).all()
            determinant = torch.linalg.det(jacobian)
            singular = torch.linalg.svdvals(jacobian)
            condition = singular[:, 0]/singular[:, -1].clamp_min(1e-12)
            transformed, info = torch.linalg.solve_ex(jacobian.transpose(-1, -2), axis[start:end, :, None])
            transformed = transformed[..., 0]
            valid = ((info == 0) & torch.isfinite(transformed).all(-1) &
                     (determinant > 1e-4) & (condition < 100) & (transformed.norm(dim=-1) > 1e-6))
            normal = F.normalize(torch.nan_to_num(transformed), dim=-1)
            native_normal = quaternion_normal(q[start:end]+prediction['d_rotation'].detach())
            for key, value in dict(jacobian=jacobian, determinant=determinant, condition=condition,
                                   normal=normal, native_normal=native_normal, valid=valid).items():
                parts[key].append(value.detach())
            del prediction, warp, point
        values = {key: torch.cat(items) for key, items in parts.items()}
        delta = torch.rad2deg(torch.acos(
            (values['normal'].double()*values['native_normal'].double()).sum(-1).abs().clamp(0, 1)))
        group_rows = {}
        for name, group in groups.items():
            good = group & values['valid']
            group_rows[name] = dict(points=int(group.sum()), valid_geometry_J=int(good.sum()),
                                    folded_or_negative_det=int((group & (values['determinant'] <= 0)).sum()),
                                    condition_ge100=int((group & (values['condition'] >= 100)).sum()),
                                    determinant=stats(values['determinant'][group]),
                                    condition=stats(values['condition'][group]),
                                    axis_difference_native_vs_inverse_transpose_deg=stats(delta[good]))
        # 16固定索引central difference只审计导数，不以该子集替代全99827统计。
        sample = torch.arange(0, len(xyz), max(1, len(xyz)//16), device='cuda')[:16]
        numerical = []
        epsilon = 1e-5
        for axis_index in range(3):
            shift = torch.zeros(3, device='cuda')
            shift[axis_index] = epsilon
            with torch.no_grad():
                outputs = []
                for sign in [1., -1.]:
                    output = ctx.deform.step(
                        xyz[sample]+sign*shift, camera.fid.unsqueeze(0).expand(len(sample), -1),
                        iteration=35000, feature=gate[sample], camera_center=camera.camera_center,
                        is_training=False)['d_xyz']
                    outputs.append(output)
                numerical.append((outputs[0]-outputs[1])/(2*epsilon))
        numerical = torch.stack(numerical, -1)+torch.eye(3, device='cuda')[None]
        error = (numerical-values['jacobian'][sample]).abs()
        reference = values['jacobian'][sample].abs().max().clamp_min(1e-6)
        assert torch.isfinite(error).all()
        row = dict(frame=frame, groups=group_rows,
                   finite_difference_16_max_abs=float(error.max()),
                   finite_difference_16_max_relative_to_reference_max=float(error.max()/reference))
        rows.append(row)
        np.savez_compressed(ROOT/'inspection'/('source_J_'+frame+'.npz'),
                            **{k: v.cpu().numpy() for k, v in values.items()}, axis_difference_deg=delta.cpu().numpy())
        print(json.dumps(row, ensure_ascii=False), flush=True)
        ctx.unload_camera(camera)
    ctx.verify_frozen()
    manifest = json.loads((base/'code_manifest.json').read_text())
    assert len(manifest) == 476
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest() == v for k, v in manifest.items())
    result = dict(status='PASS_SOURCE_GEOMETRY_JACOBIAN_DIAGNOSTIC', points=len(xyz),
                  source_iteration=35000, TRAIN_frames_used=6, TEST_frames_used=0, rows=rows,
                  source_PLY=str(source), source_PLY_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_MLP=str(mlp), source_MLP_sha256=hashlib.sha256(mlp.read_bytes()).hexdigest(),
                  source_feature_probability_native_exact=True, feature_held_constant_in_spatial_J=True,
                  learned_40000_geometry_material_feature_used=False, optimizer_steps=0, GT_labels_read=False,
                  training_prior_generated=False, production_changed=False, formal_MAE=False,
                  created_at=datetime.datetime.now().astimezone().isoformat(),
                  limitations='仅源NN位置Jacobian与源native q运动一致性；J反射/病态仍完整保存，不删除训练点或评分像素。'
                              '独立轴误差不等于GT normal误差；源NN未保证体积warp的导数可靠，不能直接用此替换生产旋转。'
                              '6TRAIN不能宣称全135来源运输已通过，finite difference仅16固定点辅助检查。')
    (ROOT/'inspection/source_jacobian.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
