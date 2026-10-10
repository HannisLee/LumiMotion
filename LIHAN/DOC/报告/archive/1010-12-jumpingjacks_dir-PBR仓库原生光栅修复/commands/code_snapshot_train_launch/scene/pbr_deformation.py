"""原生PBR可选的Stage1预测XYZ输入锚定；不改变实际渲染坐标。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from plyfile import PlyData

METHOD = 'stage1_prediction_XYZ_deformation_input_anchor'


def validate_deformation_anchor(args):
    if not getattr(args, 'pbr_deformation_anchor', False):
        return
    if (args.render_mode != 'original_ir' or args.pbr_light_mode != 'gt_directional' or
            not getattr(args, 'pbr_optimize_geometry', False)):
        raise ValueError('原生形变输入锚定仅支持方向光PBR几何微调。')
    if getattr(args, 'geometry_source_deformation_mode', 'dynamic') != 'dynamic':
        raise ValueError('原生XYZ输入锚定必须保留各帧来源形变，不能同时锚定相机/time。')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_assets(model, iteration):
    if type(iteration) is not int or iteration < 1:
        raise ValueError('形变输入锚定缺少有效来源iteration。')
    source = json.loads((model/'stage1_source.json').read_text())
    snapshots = {}
    for relative in ['point_cloud/iteration_{}/point_cloud.ply', 'deform/iteration_{}/deform.pth']:
        relative = relative.format(iteration)
        matches = [key for key in source if key.endswith('/'+relative)]
        if len(matches) != 1 or _sha(model/relative) != source[matches[0]]:
            raise ValueError('形变锚定的本地35k来源副本与记录SHA不符。')
        snapshots[relative] = source[matches[0]]
    v = PlyData.read(model/f'point_cloud/iteration_{iteration}/point_cloud.ply')['vertex'].data
    xyz = np.column_stack([v[key] for key in ['x','y','z']])
    if xyz.dtype != np.float32 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('来源XYZ必须是非空有限float32。')
    return xyz, source, snapshots


def deformation_input(current_xyz, anchor_xyz=None):
    """默认保持原native查询；启用时只替换MLP输入，渲染仍由调用方使用当前XYZ。"""
    if anchor_xyz is None:
        return current_xyz.detach()
    if (anchor_xyz.shape != current_xyz.shape or anchor_xyz.dtype != current_xyz.dtype or
            anchor_xyz.device != current_xyz.device or anchor_xyz.data_ptr() == current_xyz.data_ptr()):
        raise ValueError('来源形变XYZ输入须独立存储、同shape/dtype/device。')
    return anchor_xyz.detach()


@torch.no_grad()
def create_deformation_anchor(model, xyz, source_iteration):
    model = Path(model)
    asset, metadata = model/'pbr_deformation_anchor.npz', model/'pbr_deformation_anchor.json'
    if asset.exists() or metadata.exists():
        raise FileExistsError('来源形变锚定资产禁止覆盖。')
    expected, source, snapshots = _source_assets(model, source_iteration)
    actual = xyz.detach().cpu().numpy()
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError('只能用真实复制的Stage1 XYZ创建形变输入锚定。')
    np.savez(asset, xyz=expected)
    record = dict(format_version=1, method=METHOD, source_iteration=source_iteration,
                  points=len(expected), source_files=source, source_snapshots=snapshots,
                  asset_sha256=_sha(asset), gt_geometry_inputs=False,
                  gt_material_inputs=False, gt_normal_inputs=False)
    metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return xyz.detach().clone(), record


def load_deformation_anchor(model, current_xyz, iteration):
    model = Path(model)
    record = json.loads((model/'pbr_deformation_anchor.json').read_text())
    if (record.get('format_version') != 1 or record.get('method') != METHOD or
            any(record.get(key) is not False for key in ['gt_geometry_inputs','gt_material_inputs','gt_normal_inputs'])):
        raise ValueError('形变输入锚定来源协议无效。')
    source_iteration = record.get('source_iteration')
    expected, source, snapshots = _source_assets(model, source_iteration)
    if (iteration < source_iteration or record.get('source_files') != source or
            record.get('source_snapshots') != snapshots or record.get('points') != len(expected)):
        raise ValueError('形变输入锚定的来源/点数与当前模型不符。')
    asset = model/'pbr_deformation_anchor.npz'
    if _sha(asset) != record.get('asset_sha256'):
        raise ValueError('形变输入锚定资产SHA不符。')
    with np.load(asset, allow_pickle=False) as values:
        if set(values.files) != {'xyz'}:
            raise ValueError('形变输入锚定资产必须仅包含来源XYZ。')
        xyz = values['xyz'].copy()
    if xyz.shape != expected.shape or xyz.dtype != np.float32 or not np.array_equal(xyz, expected):
        raise ValueError('形变输入锚定XYZ与真实来源不一致。')
    if tuple(xyz.shape) != tuple(current_xyz.shape) or current_xyz.dtype != torch.float32:
        raise ValueError('形变输入锚定点数/dtype与当前checkpoint不符。')
    if iteration > source_iteration:
        marker = json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())
        if marker.get('iteration') != iteration or marker.get('deformation_anchor_sha256') != record['asset_sha256']:
            raise ValueError('当前PBR checkpoint没有绑定同一形变输入锚定资产。')
    return torch.as_tensor(xyz, device=current_xyz.device)
