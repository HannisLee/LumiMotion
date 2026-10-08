"""训练RGB推导的法线弱先验及Stage2初始化协议；GT法线没有输入接口。"""
import json
from pathlib import Path

import numpy as np
import torch


RGB_STATE_FIELDS = {'rgb_normal_targets', 'rgb_normal_valid'}
COHERENCE_STATE_FIELDS = {'coherence_neighbors', 'coherence_orientation', 'coherence_lengths',
    'coherence_weights', 'coherence_initialized', 'coherence_source_static'}


def initialize_geometry_from_stage2(geometry, state, refined_coherence=False):
    """严格复制学习参数；只有显式新分支可丢弃来源旧图并从当前真实表面重建。"""
    source = state['geometry']
    if 'translation' not in source:
        raise ValueError('Stage2初始化缺少位置参数。')
    present_rgb = RGB_STATE_FIELDS & source.keys()
    if present_rgb and present_rgb != RGB_STATE_FIELDS:
        raise ValueError('初始化RGB目标缓冲不完整。')
    if state['config'].get('lambda_geometry_rgb_normal_prior', 0.) > 0 and present_rgb != RGB_STATE_FIELDS:
        raise ValueError('真实来源启用了RGB先验但缺少完整缓存。')
    if present_rgb:
        normal, valid = source['rgb_normal_targets'], source['rgb_normal_valid']
        count = len(source['translation'])
        if (normal.shape != (count, 3) or normal.dtype != torch.float32 or
                valid.shape != (count,) or valid.dtype != torch.bool or not bool(valid.any()) or
                not torch.isfinite(normal).all() or
                not torch.allclose(normal[valid].norm(dim=-1), torch.ones_like(normal[valid, 0]), rtol=0, atol=1e-4)):
            raise ValueError('初始化RGB缓存的shape/dtype/有效点/单位法线不合法。')
    if state['config'].get('lambda_geometry_coherence', 0.) > 0:
        required = COHERENCE_STATE_FIELDS - {'coherence_source_static'}
        if state['config'].get('geometry_coherence_source_mask', ''):
            required = required | {'coherence_source_static'}
        if not required.issubset(source) or not bool(source['coherence_initialized']):
            raise ValueError('真实来源的邻域图缓存不完整或未初始化。')
    refreshed = set()
    if refined_coherence:
        if not geometry.coherence_enabled:
            raise ValueError('重建微调表面邻域必须启用邻域项。')
        refreshed = COHERENCE_STATE_FIELDS & geometry.state_dict().keys()
        source = {key:value for key,value in source.items() if key not in COHERENCE_STATE_FIELDS}
    allowed_missing = refreshed | (RGB_STATE_FIELDS - source.keys() if geometry.rgb_normal_enabled else set())
    missing, unexpected = geometry.load_state_dict(source, strict=False)
    if set(missing) != allowed_missing or unexpected:
        raise ValueError('初始化几何状态不完整或参数不兼容。')
    return dict(refined_coherence_graph_rebuilt=bool(refined_coherence),
                recomputed_buffer_names=sorted(refreshed), all_learned_parameters_copied=True)


def validate_stage2_initialization(state, args, source_files, fingerprint):
    if (state.get('diagnostic') or state.get('format_version') != 2 or
            state.get('iteration') != args.geometry_initial_stage2_iteration or
            state.get('source_iteration') != args.load_iter or
            state.get('source_files') != source_files or
            state.get('frozen_fingerprint') != fingerprint or 'geometry' not in state):
        raise ValueError('Stage2初始化必须来自同一冻结来源的真实训练检查点，不能续用诊断状态。')
    config = state['config']
    fields = {'geometry_subdivide':False, 'geometry_subdivision_side':2,
        'geometry_optimize_scale':False, 'geometry_optimize_opacity':False,
        'geometry_area_emitter':False, 'geometry_area_power_convention':'blender5',
        'geometry_tonemap':'srgb', 'geometry_shading':'screen',
        'photometric_material_mode':'per_gaussian', 'photometric_palette_size':4,
        'geometry_validation_stride':10, 'geometry_checkpoint_source_cache':False}
    for key, default in fields.items():
        if config.get(key, default) != getattr(args, key, default):
            raise ValueError('Stage2初始化的几何/材质/划分配置不匹配：'+key)
    for key in ['geometry', 'material', 'light']:
        if not all(not torch.is_tensor(v) or not v.is_floating_point() or torch.isfinite(v).all()
                   for v in state[key].values()):
            raise ValueError('Stage2初始化包含非有限张量：'+key)


def load_rgb_normal_prior(path, count, source_files, initialization_sha, keys):
    with np.load(path, allow_pickle=False) as asset:
        normal, valid = asset['normal'].copy(), asset['valid'].copy()
        metadata = json.loads(asset['metadata'].item())
    if (normal.dtype != np.float32 or normal.shape != (count, 3) or
            valid.dtype != np.bool_ or valid.shape != (count,) or
            not np.isfinite(normal).all() or not valid.any() or
            not np.allclose(np.linalg.norm(normal[valid], axis=-1), 1, rtol=0, atol=1e-4)):
        raise ValueError('RGB法线先验的shape/dtype/有效点或单位法线不合法。')
    if (metadata.get('format_version') != 1 or
            metadata.get('method') != 'training_rgb_refined_area_ps' or
            metadata.get('gt_normal_inputs') is not False or
            metadata.get('gt_depth_inputs') is not False or
            metadata.get('gt_material_inputs') is not False or
            metadata.get('original_test_used') is not False or
            metadata.get('optimization_keys') != keys or
            metadata.get('source_files') != source_files or
            metadata.get('stage2_checkpoint_sha256') != initialization_sha):
        raise ValueError('RGB法线先验的来源、Stage2初始化、训练划分或无GT协议不匹配。')
    return torch.from_numpy(normal), torch.from_numpy(valid), metadata
