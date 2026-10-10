"""来源局部旋转与原训练RGB派生轴约束；不使用GT法线/材质/几何。"""
import json
import math
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scene.pbr_deformation import _source_assets, _sha
from scene.pbr_geometry import quaternion_normal
from scene.stage2_photometric import frame_key

ASSET = 'pbr_rgb_normal.npz'
METADATA = 'pbr_rgb_normal.json'
INPUT_ASSET = 'pbr_rgb_normal_input.npz'
INPUT_METADATA = 'pbr_rgb_normal_input.json'
METHOD = 'source_local_rotation_with_TRAIN_RGB_canonical_axis_constraint'
INPUT_METHOD = 'source_native_rotation_transported_TRAIN_RGB_linear_luminance'


def validate_rgb_normal(args):
    enabled = getattr(args, 'pbr_source_local_rotation', False)
    weight = getattr(args, 'lambda_pbr_rgb_normal_prior', 0.)
    path = getattr(args, 'pbr_rgb_normal_prior_path', '')
    if not enabled:
        if weight != 0 or path:
            raise ValueError('训练RGB轴约束须启用来源局部旋转。')
        return
    if (args.render_mode != 'original_ir' or args.pbr_light_mode != 'gt_directional' or
            not getattr(args, 'pbr_optimize_geometry', False) or
            not getattr(args, 'pbr_deformation_anchor', False) or
            not getattr(args, 'pbr_plane_projection', False) or
            not getattr(args, 'pbr_optimize_feature_gate', False) or
            getattr(args, 'pbr_rasterizer_backend', 'installed') != 'lumimotion' or
            not path or not math.isfinite(weight) or weight <= 0):
        raise ValueError('RGB轴约束需要完整来源锚定/plane/native/gate方向光PBR和正权重/资产路径。')


def quaternion_product(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def transport_quaternion(world, source, current):
    source = F.normalize(source, dim=-1)
    conjugate = source * source.new_tensor([1., -1., -1., -1.])
    local = quaternion_product(conjugate, F.normalize(current, dim=-1))
    return quaternion_product(F.normalize(world, dim=-1), local)*world.norm(dim=-1, keepdim=True)


class SourceLocalRGBNormal:
    def __init__(self, source, target, confidence):
        if (source.ndim != 2 or source.shape[1] != 4 or target.shape != (len(source), 3) or
                confidence.shape != (len(source),) or
                any(v.device != source.device or v.dtype != source.dtype for v in [target, confidence]) or
                any(not torch.isfinite(v).all() for v in [source, target, confidence]) or
                (source.norm(dim=-1) < 1e-8).any() or
                not torch.allclose(target.norm(dim=-1), torch.ones_like(confidence), atol=2e-6, rtol=0) or
                not ((confidence == 0) | (confidence == 1)).all() or not confidence.any()):
            raise ValueError('来源局部旋转/RGB轴约束数组无效。')
        self.source = source.detach().clone()
        self.target = F.normalize(target.detach(), dim=-1).clone()
        self.confidence = confidence.detach().clone()
        self.metadata_sha = None

    def apply(self, pc, prediction):
        current = pc._rotation
        raw = prediction['d_rotation']
        if (current.shape != self.source.shape or raw.shape != current.shape or
                current.dtype != self.source.dtype or current.device != self.source.device):
            raise ValueError('当前旋转与来源局部旋转模型shape/dtype/device不符。')
        baseline = self.source + raw
        moved = transport_quaternion(baseline, self.source, current)
        initial = transport_quaternion(baseline, self.source, self.source)
        out = dict(prediction)
        # current==source时逐bit返回raw；梯度仍传到真实canonical quaternion。
        out['d_rotation'] = raw + (self.source-current) + (moved-initial)
        return out

    def loss(self, pc):
        axis = quaternion_normal(pc._rotation)
        mismatch = 1-(axis*self.target).sum(-1).clamp(-1, 1)
        return (mismatch*self.confidence).sum()/self.confidence.sum()


def _source_rotation(model, iteration, deform):
    _, files, snapshots = _source_assets(model, iteration)
    v = PlyData.read(model/f'point_cloud/iteration_{iteration}/point_cloud.ply')['vertex'].data
    q = np.column_stack([v[f'rot_{i}'] for i in range(4)])
    if q.dtype != np.float32 or not np.isfinite(q).all() or (np.linalg.norm(q, axis=-1) < 1e-8).any():
        raise ValueError('来源canonical quaternion无效。')
    saved = torch.load(model/f'deform/iteration_{iteration}/deform.pth', map_location='cpu')
    if any(p.requires_grad for p in deform.deform.parameters()):
        raise ValueError('来源局部旋转禁止开放来源MLP梯度。')
    actual = deform.deform.state_dict()
    if set(saved) != set(actual) or any(not torch.equal(saved[k], actual[k].detach().cpu()) for k in saved):
        raise ValueError('来源局部旋转只支持完全冻结的来源MLP。')
    return q, files, snapshots


def _verify_input(asset, metadata, model, args, cameras, source_q, snapshots):
    meta = json.loads(metadata.read_text())
    keys = {frame_key(c.image_name_train_light) for c in cameras}
    if (meta.get('format_version') != 1 or meta.get('method') != INPUT_METHOD or
            meta.get('points') != len(source_q) or meta.get('TRAIN_frames') != len(keys) or
            set(meta.get('training_frame_keys', [])) != keys or
            set(meta.get('TRAIN_RGBA_manifest', {})) != keys or meta.get('TEST_frames_used') != 0 or
            meta.get('optimizer_steps') != 0 or meta.get('native_feature_probability_exact') is not True or
            any(meta.get(k) is not False for k in ['GT_labels_read', 'GT_light_photometry_read',
                'learned_40000_geometry_material_feature_used', 'production_pipeline_changed', 'full_stage2_started']) or
            _sha(asset) != meta.get('asset_sha256')):
        raise ValueError('训练RGB候选来源协议/帧/资产SHA无效。')
    manifest = meta['source_manifest']
    for name, item in manifest.items():
        if _sha(item['path']) != item['sha256']:
            raise ValueError('RGB候选来源文件SHA变化：'+name)
    if (manifest['PLY']['sha256'] != snapshots[f"point_cloud/iteration_{args.load_iter}/point_cloud.ply"] or
            manifest['MLP']['sha256'] != snapshots[f"deform/iteration_{args.load_iter}/deform.pth"] or
            manifest['TRAIN_cameras']['sha256'] != _sha(Path(args.source_path)/'transforms_train.json') or
            manifest['directions_only']['sha256'] != _sha(args.pbr_gt_lights_path)):
        raise ValueError('RGB候选与当前35k/相机/单位方向表不符。')
    from scene.pbr_directional import GTDirectionalLights
    table = GTDirectionalLights(args.pbr_gt_lights_path)
    if any(not torch.equal(energy, torch.ones_like(energy)) for _, energy in table.frames.values()):
        raise ValueError('训练RGB约束只接受白色单位GT方向表，不读取GT光强/颜色。')
    frames = json.loads((Path(args.source_path)/'transforms_train.json').read_text())['frames']
    for frame in frames:
        key = str(int(frame['source_frame'])).zfill(4)
        image = Path(args.source_path)/args.train_light_folder/frame['file_path']
        if key not in keys or _sha(image) != meta['TRAIN_RGBA_manifest'][key]['sha256']:
            raise ValueError('RGB候选观测并非当前原TRAIN RGBA。')
    expected_protocol = dict(min_observations=12, gram_eigenvalue_ratio=.01, min_coefficient_norm=.01,
        max_relative_residual=.3, source_expected_Z_distance_m=.025, own_alpha_min=.99, source_raster_alpha_min=.9)
    if meta.get('threshold_protocol') != expected_protocol:
        raise ValueError('未知RGB候选拟合/可靠性协议。')
    with np.load(asset, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    n = len(source_q)
    fields = {'canonical_axis', 'confidence', 'observations', 'gram_eigenvalues', 'relative_luminance_residual',
              'source_canonical_axis', 'gram', 'rhs', 'y_squared'}
    if (set(arrays) != fields or any(len(v) != n or not np.isfinite(v).all() for v in arrays.values()) or
            arrays['canonical_axis'].shape != (n, 3) or arrays['confidence'].shape != (n,) or
            arrays['gram'].shape != (n, 3, 3) or arrays['rhs'].shape != (n, 3) or
            arrays['canonical_axis'].dtype != np.float32 or arrays['confidence'].dtype != np.float32):
        raise ValueError('RGB候选必须包含全部来源点的有限数组。')
    with np.load(model/'pbr_plane_projection.npz', allow_pickle=False) as values:
        support = values['support'].copy()
    with np.load(manifest['plane_arrays']['path'], allow_pickle=False) as values:
        if not np.array_equal(support, values['support']):
            raise ValueError('RGB候选plane支持与当前来源预测不符。')
    gram = arrays['gram'].astype(np.float64)
    rhs = arrays['rhs'].astype(np.float64)
    eigen = np.linalg.eigvalsh(gram)
    vector = np.linalg.solve(gram+np.eye(3)[None]*1e-5, rhs[..., None])[..., 0]
    norm = np.linalg.norm(vector, axis=-1)
    residual = np.maximum(0, arrays['y_squared']-2*np.sum(vector*rhs, axis=-1)+np.einsum('ni,nij,nj->n', vector, gram, vector))
    relative = np.sqrt(residual/np.maximum(arrays['y_squared'], 1e-8))
    reliable = ((arrays['observations'] >= 12) & (eigen[:, 0] > .01*np.maximum(eigen[:, 2], 1e-8)) &
                (norm > .01) & (relative < .3) & ~support)
    source_axis = quaternion_normal(torch.from_numpy(source_q)).numpy()
    target = vector/np.maximum(norm[:, None], 1e-12)
    target *= np.where(np.sum(target*source_axis, axis=-1, keepdims=True) < 0, -1., 1.)
    target = np.where(reliable[:, None], target, source_axis)
    if (not np.array_equal(arrays['confidence'] > 0, reliable) or
            not np.isin(arrays['confidence'], [0, 1]).all() or
            np.max(np.abs(target-arrays['canonical_axis'])) > 2e-4 or
            np.max(np.abs(source_axis-arrays['source_canonical_axis'])) > 2e-6 or
            not reliable.any() or int(reliable.sum()) != meta['confidence_points']):
        raise ValueError('RGB候选轴/confidence与独立TRAIN最小二乘复算不符。')
    return arrays['canonical_axis'], arrays['confidence'], meta


def create_rgb_normal(model, pc, deform, cameras, args, source_iteration):
    validate_rgb_normal(args)
    model = Path(model)
    if any((model/k).exists() for k in [ASSET, METADATA, INPUT_ASSET, INPUT_METADATA]):
        raise FileExistsError('来源局部RGB轴模型禁止覆盖。')
    source, files, snapshots = _source_rotation(model, source_iteration, deform)
    if not np.array_equal(pc._rotation.detach().cpu().numpy(), source):
        raise ValueError('只能从原Stage1 quaternion初始化RGB轴模型。')
    origin = Path(args.pbr_rgb_normal_prior_path)
    target, confidence, provenance = _verify_input(origin/'training_rgb_normal_constraint.npz',
        origin/'training_rgb_normal_constraint.json', model, args, cameras, source, snapshots)
    shutil.copy2(origin/'training_rgb_normal_constraint.npz', model/INPUT_ASSET)
    shutil.copy2(origin/'training_rgb_normal_constraint.json', model/INPUT_METADATA)
    np.savez(model/ASSET, source_quaternion=source, canonical_axis=target, confidence=confidence)
    record = dict(format_version=1, method=METHOD, source_iteration=source_iteration, points=len(source),
        source_files=files, source_snapshots=snapshots, asset_sha256=_sha(model/ASSET),
        input_asset_sha256=_sha(model/INPUT_ASSET), input_metadata_sha256=_sha(model/INPUT_METADATA),
        constraint_weight=args.lambda_pbr_rgb_normal_prior, confidence_points=int(confidence.sum()),
        GT_normal_inputs=False, GT_material_inputs=False, GT_geometry_inputs=False, GT_light_photometry_inputs=False,
        observations_from='original_TRAIN_RGB_and_own_RGBA_alpha', source_MLP_frozen=True,
        world_rotation='R_source_world_current_gate_times_R_source_canonical_inverse_times_R_current_canonical')
    (model/METADATA).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    operator = SourceLocalRGBNormal(torch.as_tensor(source, device=pc._rotation.device),
        torch.as_tensor(target, device=pc._rotation.device), torch.as_tensor(confidence, device=pc._rotation.device))
    operator.metadata_sha = _sha(model/METADATA)
    return operator


def load_rgb_normal(model, pc, deform, cameras, args, iteration):
    validate_rgb_normal(args)
    model = Path(model)
    record = json.loads((model/METADATA).read_text())
    src = record.get('source_iteration')
    if (record.get('format_version') != 1 or record.get('method') != METHOD or type(src) is not int or
            src != args.load_iter or not 1 <= src <= iteration or
            record.get('constraint_weight') != args.lambda_pbr_rgb_normal_prior or
            any(record.get(k) is not False for k in ['GT_normal_inputs', 'GT_material_inputs',
                'GT_geometry_inputs', 'GT_light_photometry_inputs']) or record.get('source_MLP_frozen') is not True):
        raise ValueError('来源局部RGB轴模型协议与当前cfg不符。')
    source, files, snapshots = _source_rotation(model, src, deform)
    if record.get('source_files') != files or record.get('source_snapshots') != snapshots or record.get('points') != len(source):
        raise ValueError('来源局部RGB轴模型来源不符。')
    for key, name in [('asset_sha256', ASSET), ('input_asset_sha256', INPUT_ASSET), ('input_metadata_sha256', INPUT_METADATA)]:
        if _sha(model/name) != record.get(key):
            raise ValueError('来源局部RGB轴模型资产SHA不符：'+name)
    target, confidence, _ = _verify_input(model/INPUT_ASSET, model/INPUT_METADATA, model, args, cameras, source, snapshots)
    with np.load(model/ASSET, allow_pickle=False) as values:
        expected = dict(source_quaternion=source, canonical_axis=target, confidence=confidence)
        if set(values.files) != set(expected) or any(values[k].dtype != v.dtype or not np.array_equal(values[k], v) for k, v in expected.items()):
            raise ValueError('来源局部RGB轴资产不符合真实来源复算。')
    if iteration > src:
        marker = json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())
        if marker.get('rgb_normal_metadata_sha256') != _sha(model/METADATA):
            raise ValueError('当前checkpoint未绑定来源局部RGB轴模型。')
    operator = SourceLocalRGBNormal(torch.as_tensor(source, device=pc._rotation.device),
        torch.as_tensor(target, device=pc._rotation.device), torch.as_tensor(confidence, device=pc._rotation.device))
    operator.metadata_sha = _sha(model/METADATA)
    return operator
