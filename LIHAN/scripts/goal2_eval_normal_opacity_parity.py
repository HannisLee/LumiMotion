#!/usr/bin/env python3
"""新normal-opacity入口真实checkpoint前向重载；训练反向audit单独核查。"""
import argparse
import os
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
    from goal2_eval_normal_opacity_render import actual_sources, physical_renderer, ENTRY
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
            raise AssertionError('起点五资产变化。')
        run.files[path] = sha
    static = read(run.model / 'static_dl_training.json')
    config, trained = static['args'], static['rasterizer_provenance']
    for key in ('method', 'source_sha256', 'extension_sha256', 'source_files'):
        if trained[key] != backend[key]:
            raise AssertionError('保存训练/native与fresh来源不同：' + key)
    if digest(backend['extension_path']) != trained['extension_sha256']:
        raise AssertionError('真实.so哈希不同。')
    # 只检查真实已产生的训练反向审计，不在fresh中重复构造或伪造其梯度证据。
    audits = {}
    for name in ('initial_checkpoint_gradient_audit.json', 'initial_normal_opacity_audit.json'):
        path = run.model / name
        evidence = read(path)
        if evidence['status'] != 'PASS' or not evidence['checks'] or not all(item['passed'] for item in evidence['checks'].values()):
            raise AssertionError('真实训练内审计未通过：' + name)
        if name == 'initial_normal_opacity_audit.json' and (evidence['parameter_updates'] != 0 or evidence['no_gt_region_used'] is not True):
            raise AssertionError('normal route审计须零更新且无GT区域。')
        audits[name] = dict(status=evidence['status'], path=str(path), sha256=digest(path), checks=evidence['checks'])
        run.files[str(path)] = digest(path)
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
    arrays.update(geometry)
    arrays['real_xyz']=run.pc.get_xyz.detach().cpu().numpy()
    arrays['real_quaternion']=run.pc.get_rotation.detach().cpu().numpy()
    np.savez_compressed(output/'render_parity_arrays.npz',**arrays)
    modules=sorted({line.split()[-1] for line in Path(f'/proc/{os.getpid()}/maps').read_text().splitlines()
        if ('lumimotion_pbr_cuda' in line or 'surfel_tracer/' in line) and line.split()[-1].endswith('.so')})
    if not any('surfel_tracer/' in path for path in modules):
        raise AssertionError('没有真实tracer运行映射。')
    run.files[str(Path(__file__).resolve())]=digest(Path(__file__))
    for path,sha in run.files.items():
        if digest(path)!=sha:
            raise AssertionError('fresh来源变化：'+path)
    report=dict(status='PASS',model=str(run.model),source=str(run.source),iteration=args.load_iter,frame=args.frame,
        actual_entrypoint=ENTRY,normal_opacity_route=marker['normal_opacity_route'],ambient_fraction=0.,photo_depth_ratio=1,
        checks=checks,point_count=len(run.pc.get_xyz),initial_point_count=marker['initial_point_count'],saved_zero_heads=heads,
        geometry_maps='真实XYZ/q world normal/alpha/median-depth同原GeometryRun，三时变几何输出全零。',
        parameter_scope='Stage1 XYZ/scale/opacity/拓扑允许更新；不施加Stage2冻结项。',
        training_audit_evidence=audits,forward_scope='同checkpoint两侧fresh/BVH，仅保存重载与实际前向；反向route由上述已真实运行训练audit证明，本fresh不重新证明科学效果。',
        effective_rasterizer=backend,training_rasterizer=trained,actual_runtime_modules={path:digest(path) for path in modules},
        runtime_pid=os.getpid(),physical_provenance=provenance,sources=run.files,goal_numeric='尚未评测')
    write(output/'parity.json',report)
    (output/'README.md').write_text('# normal-opacity实际来源与保存重载\n\n'
        +f'模型：`{run.model}`；source：`{run.source}`；iteration={args.load_iter}，训练相机{args.frame}。\n\n'
        +f'route={marker["normal_opacity_route"]}，实际ambient函数a0/photo1，11项fresh输出通过；原GeometryRun及六零头通过。\n\n'
        +'新10源码/5起点资产与真实native/tracer一致。实际训练photo/normal-opacity两审计PASS，SHA与反向检查结果独立留档；不把fresh前向当梯度路线或Goal成功证明。\n\n'
        +'单相机接入**PASS**；完整15指标及四类150图/视频未在本检查执行，另行验收。\n')
    print('normal-opacity实际fresh PASS：'+str(output),flush=True)


if __name__=='__main__':
    main()
