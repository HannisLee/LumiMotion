#!/usr/bin/env python3
"""足部恢复新入口入口真实checkpoint前向重载；训练反向audit单独核查。"""
import argparse
import os
import shlex
import random
import math
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--load_iter', type=int, required=True)
    p.add_argument('--frame', type=int, default=1)
    p.add_argument('--output_dir', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from argparse import Namespace
    from scripts.eval_geometry_gt import GeometryRun
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility
    from scene.static_dl_ambient import render_photometric
    from goal3_densify_render import actual_sources, physical_renderer, ENTRY
    from goal_eval_protocol import FRAMES, CONTROL, read, write, digest
    from utils.geometry_eval_utils import unit_normal
    import diff_surfel_rasterization as rasterizer

    backend = getattr(rasterizer, 'provenance', None)
    if not backend or backend.get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise AssertionError('必须bootstrap绑定原生CUDA光栅。')
    if args.frame in FRAMES or args.frame not in range(1,151):
        raise AssertionError('fresh只取训练相机。')
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'output'):
        raise AssertionError('输出必须output内。')
    output.mkdir(parents=True, exist_ok=False)
    marker, sources = actual_sources(args.model_path)
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    run.files.update(sources)
    for path, sha in marker['initial_sources'].items():
        if digest(path) != sha:
            raise AssertionError('起点六资产变化。')
        run.files[path] = sha
    static = read(run.model / 'static_dl_training.json')
    config, trained = static['args'], static['rasterizer_provenance']
    finished_path = run.model / 'foot_densify_finished.json'
    finished = read(finished_path)
    if finished['status'] != 'TRAINING_FINISHED' or finished['iteration'] != 16000+marker['update_budget'] or finished['optimizer_updates'] != marker['update_budget'] or finished['opacity_resets_performed'] != 0 or not finished['initial_sources_unchanged']:
        raise AssertionError('必须真实完成原预算、无reset、起点来源不变。')
    for key in ('normal_opacity_route','densification_route','reset_policy','clock_start','clock_shift_diagnostic'):
        if finished[key] != marker[key]:
            raise AssertionError('真实finished和训练marker不一致：'+key)
    runtime_path = run.model.parent / 'actual_runtime_capture.json'
    runtime = read(runtime_path)
    if runtime['status'] != 'PASS_CAPTURED' or not runtime['process_alive_during_hash'] or runtime['actual_module'] != 'scripts.train_static_dl_foot_densify':
        raise AssertionError('实际训练存活期native/tracer未完整捕获。')
    run.files.update({str(p):digest(p) for p in (finished_path,runtime_path)})
    for key in ('method', 'source_sha256', 'extension_sha256', 'source_files'):
        if trained[key] != backend[key]:
            raise AssertionError('保存训练/native与fresh来源不同：' + key)
    if digest(backend['extension_path']) != trained['extension_sha256']:
        raise AssertionError('真实.so哈希不同。')
    # 只检查真实已产生的训练反向审计，不在fresh中重复构造或伪造其梯度证据。
    audits = {}
    expected_audit_counts = {'initial_checkpoint_gradient_audit.json':17,
                             'initial_normal_opacity_audit.json':18,
                             'initial_topology_operator_audit.json':40}
    for name in ('initial_checkpoint_gradient_audit.json', 'initial_normal_opacity_audit.json', 'initial_topology_operator_audit.json'):
        path = run.model / name
        evidence = read(path)
        if evidence['status'] != 'PASS' or len(evidence['checks']) != expected_audit_counts[name] or not all(item['passed'] for item in evidence['checks'].values()):
            raise AssertionError('真实训练内审计未通过：' + name)
        if name == 'initial_normal_opacity_audit.json' and (evidence['parameter_updates'] != 0 or evidence['no_gt_region_used'] is not True):
            raise AssertionError('normal route审计须零更新且无GT区域。')
        if name == 'initial_topology_operator_audit.json' and evidence['optimizer_updates'] != 0:
            raise AssertionError('拓扑算子等价审计必须隔离且零优化器更新。')
        audits[name] = dict(status=evidence['status'], path=str(path), sha256=digest(path), checks=evidence['checks'])
        run.files[str(path)] = digest(path)
    # 新增增密审计为{name: bool}，旧三审计仍为{name: {passed: bool}}。
    density_path = run.model / 'initial_densification_route_audit.json'
    density = read(density_path)
    mandatory = {'original_matches_native','original_invisible_unchanged','candidate_accumulation_exact',
                 'denominator_same_as_native','floor_not_smaller','candidate_parameter_gradients_unchanged',
                 'nonempty_parameter_gradient_audit'}
    if marker['densification_route'] == 'original':
        mandatory.add('original_full_stats_equal')
    if density['status'] != 'PASS' or set(density['checks']) != mandatory or not all(v is True for v in density['checks'].values()) or density['actual_updates'] != 0 or not density['parameter_gradient_names']:
        raise AssertionError('真实首步增密算子/参数梯度审计未通过。')
    audits[density_path.name] = dict(status=density['status'],path=str(density_path),sha256=digest(density_path),checks=density['checks'])
    run.files[str(density_path)] = digest(density_path)
    # 数据reader按文件名数值排序，以独立seed推导首相机，不复用marker结论。
    train_metadata_path = Path(config['source_path']) / 'transforms_train.json'
    train_records = sorted(read(train_metadata_path)['frames'],key=lambda r:int(Path(r['file_path']).stem.split('_')[-1]))
    if len(train_records) != 135 or {r['source_frame'] for r in train_records} != set(range(1,151))-set(FRAMES):
        raise AssertionError('首相机推导必须原135训练输入。')
    index = random.Random(marker['args']['seed']).randrange(len(train_records))
    name = Path(train_records[index]['file_path']).stem
    first_path = run.model / 'first_training_gradient_audit.json'
    first = read(first_path)
    if marker['expected_first_camera_index'] != index or marker['expected_first_camera'] != name or first['camera'] != name or first['iteration'] != 16001 or first['step'] != 1 or first['points'] != marker['initial_point_count']:
        raise AssertionError('seed推导/marker/真实16001首相机不一致。')
    if first['normal_opacity_route'] != 'detach_opacity' or not first['rgb_to_real_geometry'] or not first['visibility_detached']:
        raise AssertionError('真实首步物理/梯度route错误。')
    for k in ('xyz','rotation','scaling','opacity','albedo','light'):
        if not all(math.isfinite(float(first['gradients'][k][v])) for v in ('norm','max_abs')):
            raise AssertionError('真实首步参数梯度非有限。')
    scores_path = run.model / 'densification_scores.jsonl'
    scores = json.loads(scores_path.read_text().splitlines()[0])
    if scores['route'] != marker['densification_route'] or scores['iteration'] != 16001 or scores['camera'] != name or scores['step'] != 1:
        raise AssertionError('真实增密首步路线与相机错误。')
    if not 0 <= scores['floor_active_points'] <= scores['visible_points'] <= marker['initial_point_count'] or not scores['floor_norm_sum'] + 1e-12 >= max(scores['original_norm_sum'],scores['non_normal_norm_sum']):
        raise AssertionError('真实首步增密下限统计不成立。')
    run.files.update({str(p):digest(p) for p in (train_metadata_path,first_path,scores_path)})
    independent = dict(status='PASS',seed=marker['args']['seed'],first_camera_index=index,
        first_camera=name,first_training_iteration=first['iteration'],initial_adam_state_empty=marker['initial_adam_state_empty'],
        initial_density_statistics_zero=marker['initial_density_statistics_zero'],optimizer_state_restored=False,
        scope='实际冻结entry的零Adam/零统计断言、训练审计和seed首相机独立复核；无历史Adam恢复声明。')
    metadata = read(run.source / 'transforms_train.json')
    record = next(r for r in metadata['frames'] if r['source_frame'] == args.frame)
    camera, _, intrinsics, _ = run.camera(metadata, record, 2)
    lighting_path = run.model / f'static_dl_lighting/iteration_{args.load_iter}/lighting.json'
    lighting = read(lighting_path)
    lights = DirectionOnlyLights(config['static_dl_directions'])
    with torch.no_grad():
        lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    visibility = StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
    wrapper, provenance = physical_renderer(run, args.load_iter, CONTROL / 'eval_directions_only_single.json')
    with torch.no_grad():
        reference = render_photometric(camera, run.pc, Namespace(**config), torch.zeros(3,device='cuda'),
            lights, visibility, args.load_iter, ambient_fraction=0., component_images=True)
        actual = wrapper(camera)
        geometry = run.render(camera, np.asarray(record['transform_matrix']), intrinsics)
        deformation = run.deformation(camera)
    arrays, checks = {}, {}
    for key in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth', 'light_intensity',
                'mean_visibility', 'light_visibility', 'ambient_fraction', 'ambient_linear', 'direct_linear'):
        a,b = reference[key].detach().cpu().numpy(), actual[key].detach().cpu().numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('实际前向非有限：' + key)
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
        arrays['training_'+key],arrays['evaluation_'+key]=a,b
        checks[key]=dict(passed=True,max_abs=float(np.max(np.abs(a-b))))
    normal,valid=unit_normal(actual['rend_normal'].permute(1,2,0).cpu().numpy())
    np.testing.assert_allclose(normal[valid],geometry['normal'][valid],atol=2e-5,rtol=1e-5)
    np.testing.assert_allclose(actual['rend_alpha'].squeeze(0).cpu().numpy(),geometry['alpha'],atol=2e-5,rtol=1e-5)
    visible=geometry['alpha']>=.5
    np.testing.assert_allclose(actual['surf_depth'].squeeze(0).cpu().numpy()[visible],geometry['median'][visible],atol=2e-5,rtol=1e-5)
    for key in ('d_xyz','d_rotation','d_scaling'):
        if torch.count_nonzero(deformation[key]):
            raise AssertionError('时变几何输出非零。')
    state=torch.load(run.model/f'deform/iteration_{args.load_iter}/deform.pth',map_location='cpu',weights_only=True)
    heads={k:int(torch.count_nonzero(v)) for k,v in state.items() if any(h in k for h in ('gaussian_warp.','gaussian_rotation.','gaussian_scaling.'))}
    if len(heads)!=6 or any(heads.values()):
        raise AssertionError('保存几何头必须严格0。')
    optimizer_path = run.model / f'optimizer/iteration_{args.load_iter}.pth'
    saved_optimizer = torch.load(optimizer_path,map_location='cpu',weights_only=False)
    for key,value in {'iteration':args.load_iter,'optimizer_updates':args.load_iter-16000,
                      'reset_policy':'skip_all','densification_route':marker['densification_route'],
                      'clock_shift_diagnostic':False}.items():
        if saved_optimizer[key] != value:
            raise AssertionError('新保存optimizer的元数据与模型不一致：'+key)
    if not saved_optimizer['gaussian']['state'] or not saved_optimizer['light']['state']:
        raise AssertionError('实际更新后的新Adam状态为空。')
    for key in ('xyz_gradient_accum','denom','max_radii2D'):
        if len(saved_optimizer[key]) != len(run.pc.get_xyz) or not torch.isfinite(saved_optimizer[key]).all():
            raise AssertionError('新保存增密统计与checkpoint点数不一致。')
    run.files[str(optimizer_path)] = digest(optimizer_path)
    arrays.update(geometry)
    arrays['real_xyz']=run.pc.get_xyz.detach().cpu().numpy()
    arrays['real_quaternion']=run.pc.get_rotation.detach().cpu().numpy()
    np.savez_compressed(output/'render_parity_arrays.npz',**arrays)
    modules=sorted({line.split()[-1] for line in Path(f'/proc/{os.getpid()}/maps').read_text().splitlines()
        if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line) and line.split()[-1].endswith('.so')})
    if not any('surfel_tracer/' in path for path in modules):
        raise AssertionError('没有真实tracer运行映射。')
    if {path:digest(path) for path in modules} != runtime['actual_runtime_modules']:
        raise AssertionError('真实训练存活期与fresh native/tracer不同。')
    run.files[str(Path(__file__).resolve())]=digest(Path(__file__))
    for path,sha in run.files.items():
        if digest(path)!=sha:
            raise AssertionError('fresh来源变化：'+path)
    report=dict(status='PASS',model=str(run.model),source=str(run.source),iteration=args.load_iter,frame=args.frame,
        command=[sys.executable] + sys.argv,
        actual_entrypoint=ENTRY,normal_opacity_route=marker['normal_opacity_route'],reset_policy=marker['reset_policy'],densification_route=marker['densification_route'],clock_shift_diagnostic=marker['clock_shift_diagnostic'],ambient_fraction=0.,photo_depth_ratio=1,
        checks=checks,point_count=len(run.pc.get_xyz),initial_point_count=marker['initial_point_count'],saved_zero_heads=heads,
        geometry_maps='真实XYZ/q world normal/alpha/median-depth同原GeometryRun，三时变几何输出全零。',
        parameter_scope='Stage1 XYZ/scale/opacity/拓扑允许更新；不施加Stage2冻结项。',
        training_audit_evidence=audits,initialization_independent=independent,
        saved_optimizer=dict(path=str(optimizer_path),sha256=digest(optimizer_path),metadata_status='PASS',restored_for_fresh=False),
        forward_scope='同checkpoint两侧fresh/BVH，仅保存重载与实际前向；反向route由上述已真实运行训练audit证明，本fresh不重新证明科学效果。',
        effective_rasterizer=backend,training_rasterizer=trained,actual_runtime_modules={path:digest(path) for path in modules},
        runtime_pid=os.getpid(),physical_provenance=provenance,sources=run.files,goal_numeric='尚未评测')
    write(output/'parity.json',report)
    (output/'README.md').write_text('# 足部恢复实际来源与保存重载\n\n'
        +f'模型：`{run.model}`；source：`{run.source}`；iteration={args.load_iter}，训练相机{args.frame}。\n\n'
        +f'route={marker["normal_opacity_route"]}，实际ambient函数a0/photo1，11项fresh输出通过；原GeometryRun及六零头通过。\n\n'
        +'新12源码/6起点资产与真实native/tracer一致。实际训练photo/normal-opacity/拓扑/增密四审计PASS，seed首相机独立核对；新Adam与统计清零，不声明历史恢复。\n\n'
        +'单相机接入**PASS**；完整15指标及四类150图/视频未在本检查执行，另行验收。\n')
    with (output/'README.md').open('a') as stream:
        stream.write('\n完整内部命令：\n\n```bash\n'+shlex.join([sys.executable]+sys.argv)+'\n```\n\n'
            +f'输出：`{output}`；统计：`{output / "parity.json"}`；数组：`{output / "render_parity_arrays.npz"}`。\n')
    print('足部恢复实际fresh PASS：'+str(output),flush=True)


if __name__=='__main__':
    main()
