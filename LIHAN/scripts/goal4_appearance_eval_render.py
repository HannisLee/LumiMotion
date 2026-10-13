#!/usr/bin/env python3
"""从精确独立外观state加载真实a0前向，训练/测试方向严格分离。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from goal4_appearance_eval_common import audit, checked, frozen_hashes, ENTRY, BASE, BASE_FILES
from goal_eval_protocol import FRAMES, read

def physical_renderer(run,iteration,eval_directions):
    import torch
    from argparse import Namespace
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    evidence=audit(run.model,iteration);marker=evidence['marker'];sources=evidence['sources'];cfg=marker['inherited_config']
    light_path=run.model/f'static_dl_lighting/iteration_{iteration}/lighting.json';lighting=read(light_path)
    if (lighting['actual_entrypoint']!=ENTRY or lighting['iteration']!=iteration or lighting['appearance_updates']!=iteration-16000
        or lighting['optimizer_updates']!=iteration-16000 or lighting['run_updates']!=iteration-16000-marker['appearance_updates_before']
        or lighting['loss_model']!=marker['loss_model'] or lighting['normal_depth_statistic']!='expected'
        or lighting['ambient_mode']!='zero' or lighting['ambient_fraction']!=0 or not lighting['geometry_frozen']
        or lighting['smoke']!=marker['smoke']):raise AssertionError('保存光强与appearance元数据冲突。')
    state_path=run.model/f'appearance_state/iteration_{iteration}.pth';state=torch.load(state_path,map_location='cpu',weights_only=True)
    expected_state=dict(format_version=1,actual_entrypoint=ENTRY,loss_model=marker['loss_model'],base_sources=marker['base_sources'],
        source_hashes=marker['source_hashes'],training_inputs=marker['training_inputs'],seed=1012,smoke=marker['smoke'],iteration=iteration,
        appearance_updates=iteration-16000,frozen_sha256=marker['initial_frozen_sha256'],point_count=67081)
    if any(state[k]!=v for k,v in expected_state.items()):raise AssertionError('精确state与训练来源不同。')
    groups=state['optimizer']['param_groups']
    if {g['name'] for g in groups}!={'albedo_dc','light'} or any(len(g['params'])!=1 for g in groups):raise AssertionError('精确state optimizer含额外路径。')
    for group in groups:
        if float(state['optimizer']['state'][group['params'][0]]['step'])!=iteration-16000:raise AssertionError('精确state Adam步数不同。')
    # save_ply正常保存stage1三字段，load_ply优先读原字段；只有历史缺字段才派生。
    # 本轮实际保存含字段时必须全部hash匹配，不能无条件排除stage1副本。
    ply_pc=run.pc;loaded_hashes=frozen_hashes(ply_pc);expected_hashes=marker['initial_frozen_sha256']
    derived='_albedo_dc_stage1'
    from plyfile import PlyData
    properties={p.name for p in PlyData.read(str(run.model/f'point_cloud/iteration_{iteration}/point_cloud.ply')).elements[0].properties}
    stage1_fields={f'albedo_dc_stage1_{i}' for i in range(3)}
    has_stage1=stage1_fields.issubset(properties)
    if properties&stage1_fields and not has_stage1:raise AssertionError('新PLY的stage1副本字段部分缺失。')
    compared_loaded=loaded_hashes if has_stage1 else {k:v for k,v in loaded_hashes.items() if k!=derived}
    compared_expected=expected_hashes if has_stage1 else {k:v for k,v in expected_hashes.items() if k!=derived}
    if compared_loaded!=compared_expected or len(ply_pc.get_xyz)!=67081:raise AssertionError('新PLY真实全部冻结字段改变。')
    from scene.gaussian_model import GaussianModel
    rebuilt=GaussianModel(cfg['sh_degree'],no_binary_separation=True,fea_dim=cfg['hyper_dim'])
    rebuilt.load_ply(str(BASE/'point_cloud/iteration_16000/point_cloud.ply'))
    checked(BASE/'point_cloud/iteration_16000/point_cloud.ply',sources,BASE_FILES['point_cloud/iteration_16000/point_cloud.ply'])
    if frozen_hashes(rebuilt)!=expected_hashes:raise AssertionError('真实T06重建全部起始冻结hash不符。')
    for p,h in state['assets'].items():checked(run.model/p,sources,h)
    if not torch.equal(ply_pc._albedo_dc.detach().cpu(),state['albedo_dc']):raise AssertionError('PLY与精确反照率state不同。')
    with torch.no_grad():rebuilt._albedo_dc.copy_(state['albedo_dc'].to('cuda'))
    for value in vars(rebuilt).values():
        if torch.is_tensor(value) and value.is_leaf:value.requires_grad_(False)
    run.pc=rebuilt
    train=DirectionOnlyLights(cfg['static_dl_directions']);evaluation=DirectionOnlyLights(eval_directions)
    from goal_eval_protocol import CONTROL
    if Path(eval_directions).resolve()!=CONTROL/'eval_directions_only_single.json':raise AssertionError('必须原固定完整150纯方向表。')
    keys={f'{n:04d}' for n in range(1,151) if n not in FRAMES}
    if (set(train.directions)!=keys or set(evaluation.directions)!={f'{n:04d}' for n in range(1,151)}
        or train.sha256!=lighting['directions_sha256'] or any(not torch.equal(train.directions[k],evaluation.directions[k]) for k in keys)):
        raise AssertionError('135TRAIN/150EVAL纯方向边界或共同方向不符。')
    saved=state['log_intensity'].to('cuda')
    if saved.shape!=evaluation.log_intensity.shape or not torch.isfinite(saved).all():raise AssertionError('精确光强shape或数值不符。')
    if not torch.equal(saved.exp().cpu(),torch.tensor(lighting['intensity_from_train_rgb'],dtype=saved.dtype)):raise AssertionError('light JSON与精确state不符。')
    with torch.no_grad():evaluation.log_intensity.copy_(saved)
    evaluation.requires_grad_(False)
    pipe=Namespace(**cfg);bg=torch.zeros(3,device='cuda');visibility=StaticVisibility(cfg['static_dl_shadow_bias'],cfg['static_dl_bvh_interval'])
    def render(camera):return render_photometric(camera,run.pc,pipe,bg,evaluation,visibility,iteration,0.,component_images=True)
    render.eval_lights=evaluation;render.eval_visibility=visibility
    ply_visibility=StaticVisibility(cfg['static_dl_shadow_bias'],cfg['static_dl_bvh_interval'])
    def render_saved_ply(camera):return render_photometric(camera,ply_pc,pipe,bg,evaluation,ply_visibility,iteration,0.,component_images=True)
    render.saved_ply_renderer=render_saved_ply
    for p in (Path(__file__),light_path,Path(cfg['static_dl_directions']),Path(eval_directions)):checked(p,sources)
    run.files.update(sources)
    return render,dict(actual_entrypoint=ENTRY,mode='lambertian_visibility_frozen_geometry',ambient_fraction=0.,geometry_frozen=True,
        loss_model=marker['loss_model'],precise_state=str(state_path),test_directions_used_only_for_evaluation=True,
        directions_only=True,intensity_from_train_rgb=lighting['intensity_from_train_rgb'],sources=sources,
        frozen_hashes_rebuilt_all=True,saved_ply_frozen_hashes_all=has_stage1,saved_ply_stage1_fields_present=has_stage1,
        derived_stage1_fallback_used=not has_stage1,
        derived_stage1_albedo_semantics='有stage1三字段严格验证全部hash；历史缺字段才显式派生fallback，另外从T06重建全hash，仅apply精确state。')
