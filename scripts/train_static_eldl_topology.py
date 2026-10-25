"""静态ELDL-V3 F阶段（拓扑配对）：从DEV选定的G4@8100继续2000次新增更新。

F0 ：原增密路线（每100步；屏幕梯度阈值0.0002；min opacity 0.01；max_screen_size 20；percent_dense 0.01；跳过reset）。
F0c：同父、同预算、同相机序列、同LR的无增密对照（只在同一事件点记录“若执行将选中多少点”，不改拓扑）。
共同：保留G4的photo部分覆盖排除（GT Alpha∈(0,0.99)像素photo权重0）、full路由、LR×1日程重启
     （xyz 1e-6、rotation 5e-5、scaling 1e-5、opacity 1e-3按0.1^((f-1)/1999)衰减到0.1倍；albedo 2e-4、照明1e-5恒定）；
     新建零动量Adam（含lr=0的albedo_rest/roughness/feature组，仅为拓扑算子同步裁剪/拼接）；增密统计清零；
     屏幕梯度统计=photo包与normal副本的viewspace梯度先向量相加再累计一次范数与denom（scene/static_dl_foot_densify）。
每步顺序：forward/backward → 统计 → optimizer → （F0：stage_step%100==0且<2000时增密/裁剪）→ 保存。
终点（2000）不做拓扑事件。保存的 iteration = 8100 + stage_step，含 stage_step 次本阶段新增更新。
不修改 scene/static_eldl_lighting.py（评测包装器按其内容核对训练快照）。
"""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from arguments import OptimizationParams
from scene.dataset_readers import getNerfppNorm
from scene.static_dl_photometric import StaticVisibility
from scene.static_dl_foot_recovery import FootRecoveryGaussianModel, opacity_summary, audit_topology_equivalence
from scene.static_dl_foot_densify import accumulate_screen_statistics
from scene.static_eldl_lighting import (TrainDirections, MixedLighting, render_mixed, censored_loss, geometry_losses,
                                        component_stats, param_hashes, audit_normal_routes, audit_physical_forward,
                                        partial_alpha_keep_mask)
from scripts.train_static_eldl_physical import (FROZEN, sha, write_json, install_guard, load_cameras,
                                                audit_photo_pixel_keep)
from utils.sh_utils import RGB2SH

F_STEPS, PARENT_ITERATION = 2000, 8100
FROZEN_F = dict(
    steps=F_STEPS, saves=[500, 1000, 1500, 2000], xyz_lr=1e-6, rotation_lr=5e-5, scaling_lr=1e-5, opacity_lr=1e-3,
    final_factor=0.1, albedo_lr=2e-4, lighting_lr=1e-5, frozen_groups_lr0=['albedo_rest', 'roughness', 'feature'],
    densify=dict(interval=100, grad_threshold=2e-4, min_opacity=0.01, max_screen_size=None, percent_dense=0.01,
                 world_size_factor=0.1, last_event_step=1900, statistics_route='original', reset='skip_all',
                 note='冒烟核对后冻结（smoke_test/1025-14）：成熟父点云约5%点屏幕半径>20px（事件前8671/8327个）。原日程传20，'
                      '但原算法clone/split会把max_radii2D清零，屏幕裁剪实际失效；只有某事件恰无clone/split时才会突然删除'
                      '这些大点（多为平台）。故冻结为None（原算法densify_from≤步≤opacity_reset_interval的分支）：'
                      '凡有clone/split的事件结果与传20逐位相同；代价是世界尺度裁剪（>0.1·extent，父模型仅1点）同时关闭。'),
    photo_partial_alpha='exclude', route='full', loss=FROZEN['loss'])


def lr_at(step):
    f = FROZEN_F['final_factor'] ** ((step - 1) / (F_STEPS - 1))
    return dict(xyz=FROZEN_F['xyz_lr'] * f, rotation=FROZEN_F['rotation_lr'] * f, scaling=FROZEN_F['scaling_lr'] * f,
                opacity=FROZEN_F['opacity_lr'] * f)


def freeze_aux(pc):
    for name in ('_albedo_rest', '_roughness', 'feature'):
        getattr(pc, name).requires_grad_(False)


def build_optimizer(pc, lighting):
    for name in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc'):
        getattr(pc, name).requires_grad_(True)
    freeze_aux(pc)
    groups = [dict(params=[pc._xyz], lr=FROZEN_F['xyz_lr'], name='xyz'),
              dict(params=[pc._rotation], lr=FROZEN_F['rotation_lr'], name='rotation'),
              dict(params=[pc._scaling], lr=FROZEN_F['scaling_lr'], name='scaling'),
              dict(params=[pc._opacity], lr=FROZEN_F['opacity_lr'], name='opacity'),
              dict(params=[pc._albedo_dc], lr=FROZEN_F['albedo_lr'], name='albedo_dc'),
              dict(params=[pc._albedo_rest], lr=0., name='albedo_rest'),
              dict(params=[pc._roughness], lr=0., name='roughness'),
              dict(params=[pc.feature], lr=0., name='feature')]
    gopt = torch.optim.Adam(groups, lr=0., eps=1e-15)
    pc.optimizer = gopt  # 原生拓扑算子按group名同步裁剪/拼接参数与Adam状态
    for p in lighting.parameters():
        p.requires_grad_(True)
    lopt = torch.optim.Adam([dict(params=[lighting.log_dc, lighting.higher], lr=FROZEN_F['lighting_lr'], name='env_sh'),
                             dict(params=[lighting.beta], lr=FROZEN_F['lighting_lr'], name='direct_fraction')])
    return gopt, lopt


@torch.no_grad()
def hypothetical_event(pc, extent):
    """F0c：按原判据统计若执行增密将选中的点（不改动任何参数）。"""
    cfg = FROZEN_F['densify']
    grads = pc.xyz_gradient_accum / pc.denom
    grads[grads.isnan()] = 0.0
    g = torch.norm(grads, dim=-1) >= cfg['grad_threshold']
    big = pc.get_scaling.max(dim=1).values > cfg['percent_dense'] * extent
    return dict(would_clone=int((g & ~big).sum()), would_split=int((g & big).sum()),
                low_opacity=int((pc.get_opacity < cfg['min_opacity']).sum()),
                screen_over_20px=int((pc.max_radii2D > 20).sum()),
                world_over=int((pc.get_scaling.max(dim=1).values > cfg['world_size_factor'] * extent).sum()))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--densify', choices=['on', 'off'], required=True, help='on=F0原增密；off=F0c无增密对照')
    p.add_argument('--parent', type=Path, required=True, help='G4的model_mlp目录')
    p.add_argument('--parent_iteration', type=int, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--directions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--save_steps', type=int, nargs='*', default=None, help='仅冒烟可改')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    out = args.output.resolve(); model = out / 'model_mlp'
    parent, source, split_path, dpath = (args.parent.resolve(), args.source.resolve(), args.split.resolve(),
                                         args.directions.resolve())
    if 'output' not in out.parts or model.exists():
        p.error('输出须在output内且model_mlp尚不存在')
    if args.smoke:
        if 'smoke_test' not in out.parts or args.steps > 300:
            p.error('冒烟只能在smoke_test且≤300步')
    elif args.steps != F_STEPS or args.save_steps is not None:
        p.error('正式运行预算/保存点冻结，不可修改')
    save_steps = FROZEN_F['saves'] if args.save_steps is None else sorted(set(args.save_steps + [args.steps]))
    split = json.loads(split_path.read_text())

    # ---- 父模型核验：必须是正式G4配置（G阶段、full、LR×1、photo排除、linear、normal 0.1、7020次）
    fin = json.loads((parent / 'physical_finished.json').read_text())
    plan_parent = json.loads((parent / 'physical_plan.json').read_text())
    assert fin['status'] == 'TRAINING_FINISHED' and fin['stage'] == 'G' and fin['route'] == 'full'
    assert plan_parent['photo_partial_alpha'] == 'exclude' and plan_parent['geometry_lr_multiplier'] == 1.0
    assert plan_parent.get('opacity_lr_multiplier', 1.0) == 1.0 and plan_parent.get('composite', 'linear') == 'linear'
    assert plan_parent.get('normal_weight', 0.1) == 0.1
    assert fin['smoke'] is False and fin['new_optimizer_updates'] == 7020 and args.parent_iteration == PARENT_ITERATION
    ply = parent / f'point_cloud/iteration_{args.parent_iteration}/point_cloud.ply'
    deform = parent / f'deform/iteration_{args.parent_iteration}/deform.pth'
    parent_lighting = json.loads((parent / f'lighting/iteration_{args.parent_iteration}/lighting.json').read_text())
    state = torch.load(deform, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for s in ('weight', 'bias'):
            assert torch.count_nonzero(state[name + '.' + s]) == 0, '父模型不是严格静态零几何头'
    install_guard([source, split_path.parent, dpath.parent, parent, out, Path.cwd()])

    seed = FROZEN['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    camera_rng = random.Random(seed)
    torch.cuda.set_device(0)
    import diff_surfel_rasterization as backend
    assert backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer'
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, depth_ratio=1.0)
    pc = FootRecoveryGaussianModel(3, no_binary_separation=True, fea_dim=1)
    pc.load_ply(str(ply))
    if hasattr(pc, '_albedo_dc_stage1'):
        del pc._albedo_dc_stage1
    pc.percent_dense = FROZEN_F['densify']['percent_dense']
    loaded_hashes = param_hashes(pc)
    directions = TrainDirections(dpath)
    assert directions.frames == split['train_frames']
    lighting = MixedLighting(.5)
    lighting.load_state(parent_lighting['lighting'])
    cameras, image_sha = load_cameras(source, split)
    extent = float(getNerfppNorm(cameras)['radius'])
    gopt, lopt = build_optimizer(pc, lighting)
    vis = StaticVisibility(FROZEN['shadow_bias'], 1)
    background = torch.zeros(3, device='cuda')
    pc.max_radii2D = torch.zeros(len(pc.get_xyz), device='cuda')
    pc.xyz_gradient_accum = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    pc.denom = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    init_hashes = param_hashes(pc, lighting)
    preview = random.Random(seed)
    first_index = list(range(len(cameras))).pop(preview.randrange(len(cameras)))
    with torch.no_grad():
        ms = pc.get_scaling.max(dim=1).values
        parent_stats = dict(points=len(ms), camera_extent=extent, percent_dense_threshold=FROZEN_F['densify']['percent_dense'] * extent,
                            split_eligible_fraction=float((ms > FROZEN_F['densify']['percent_dense'] * extent).float().mean()),
                            world_prune_candidates=int((ms > .1 * extent).sum()),
                            max_scale_quantiles=torch.quantile(ms, torch.tensor([.1, .5, .9, .99, .999], device='cuda')).tolist(),
                            opacity=opacity_summary(pc))
    initialization = dict(
        stage='F', densify=args.densify, parent=str(parent), parent_iteration=args.parent_iteration,
        parent_ply_sha256=sha(ply), parent_deform_sha256=sha(deform), parent_lighting_sha256=sha(
            parent / f'lighting/iteration_{args.parent_iteration}/lighting.json'),
        loaded_parameter_hashes=loaded_hashes, initialized_parameter_hashes=init_hashes,
        inherited_identical=all(loaded_hashes[k] == init_hashes[k] for k in loaded_hashes),
        gaussian_adam_state_count=len(gopt.state), light_adam_state_count=len(lopt.state),
        adam='新建Adam，动量为零；不恢复父优化器', density_statistics='xyz_gradient_accum/denom/max_radii2D清零',
        expected_first_camera=cameras[first_index].image_name_train_light, parent_point_cloud_statistics=parent_stats)
    assert initialization['inherited_identical'] and not gopt.state and not lopt.state
    model.mkdir(parents=True)
    write_json(model / 'initialization_audit.json', initialization)
    weights = dict(alpha=FROZEN['loss']['alpha'], normal=FROZEN['loss']['normal'], distortion=FROZEN['loss']['distortion'])
    cam0 = cameras[first_index]
    cam0.load2device()
    audits = dict(physical=audit_physical_forward(cam0, pc, pipe, background, directions, lighting, vis, args.parent_iteration),
                  normal_route=audit_normal_routes(cam0, pc, pipe, background, directions, lighting, vis,
                                                   args.parent_iteration, weights),
                  photo_partial_alpha=audit_photo_pixel_keep(cam0, pc, pipe, background, directions, lighting, vis,
                                                             args.parent_iteration))
    cam0.load2device('cpu')
    opt_ns = Namespace(**vars(OptimizationParams(ArgumentParser())))
    opt_ns.sh_degree, opt_ns.hyper_dim = 3, 1
    audits['topology_operator'] = audit_topology_equivalence(ply, opt_ns, extent, model)
    for k, v in audits.items():
        write_json(model / f'initial_{k}_audit.json', v)
    if any(v['status'] != 'PASS' for v in audits.values()):
        raise RuntimeError('零更新审计失败，保留现场不训练：' + str({k: v['status'] for k, v in audits.items()}))
    assert param_hashes(pc, lighting) == init_hashes, '审计改变了训练模型'
    clock = args.parent_iteration
    root = Path.cwd()
    plan = dict(stage='F', densify=args.densify, route='full', photo_partial_alpha='exclude', args={k: str(v) for k, v in vars(args).items()},
                frozen=FROZEN, frozen_f=FROZEN_F, clock_start=clock, steps=args.steps, save_steps_stage_relative=save_steps,
                save_iterations_relative_total=[clock + s for s in save_steps], lr_first_step=lr_at(1),
                lr_schedule='几何四组按0.1^((f-1)/1999)衰减（G日程在F阶段重启）；albedo/照明恒定；lr=0组只为拓扑同步',
                topology=('每100步原增密/裁剪（最后一次1900），跳过reset' if args.densify == 'on' else '无拓扑；同事件点只记录假想选中数'),
                camera_sampling=f'独立random.Random({seed})不放回遍历117相机；F0/F0c序列相同',
                update_semantics='每步均更新（含末步）；顺序 forward/backward→统计→optimizer→拓扑→保存；终点无拓扑',
                visibility='StaticVisibility bias0.03 no_grad，每步重建，拓扑后invalidate',
                inputs=dict(source_manifest=sha(source / 'no_gt_manifest.json'), transforms=sha(source / 'transforms_train.json'),
                            split=sha(split_path), directions=directions.sha256, images=image_sha),
                gt_supervision='仅TRAIN帧GT主光单位方向（固定不训练）；拓扑与掩膜只用训练Alpha与预测量，无GT区域',
                sources={f: sha(root / f) for f in ('scripts/train_static_eldl_topology.py', 'scripts/train_static_eldl_physical.py',
                                                     'scene/static_eldl_lighting.py', 'scene/static_dl_foot_recovery.py',
                                                     'scene/static_dl_foot_densify.py', 'scene/static_dl_photometric.py',
                                                     'scene/gaussian_model.py', 'gaussian_renderer/__init__.py')},
                rasterizer=backend.provenance, smoke=args.smoke,
                # 以下字段供评测marker（register_static_eldl_eval_marker.py）沿用
                geometry_lr_multiplier=1.0, opacity_lr_multiplier=1.0, normal_weight=0.1, composite='linear')
    write_json(model / 'physical_plan.json', plan)
    cfg = dict(sh_degree=3, source_path=str(source), model_path=str(model), images='images', train_light_folder='images',
               test_light_folder='images', resolution=2, white_background=False, data_device='cpu', eval=True,
               load2gpu_on_the_fly=True, is_blender=True, deform_type='mlp', hyper_dim=1, pred_color=False,
               no_binary_separation=False, load_test_set_only=False, start_frame=0, end_frame=-1)
    (model / 'cfg_args').write_text(str(Namespace(**cfg)))
    shutil.copyfile(ply, model / 'input.ply')

    saves, stack, events = [], [], []

    def save(step):
        it = clock + step
        pc.save_ply(str(model / f'point_cloud/iteration_{it}/point_cloud.ply'))
        dst = model / f'deform/iteration_{it}/deform.pth'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deform, dst)
        write_json(model / f'lighting/iteration_{it}/lighting.json', dict(
            iteration_relative_total=it, stage='F', route='full', stage_step=step, densify=args.densify,
            new_optimizer_updates_this_stage=step, lineage_new_updates_since_S0=1080 + 7020 + step,
            directions_sha256=directions.sha256, lighting=lighting.state(), model='static_eldl_mixed_v1'))
        opath = model / f'optimizer/iteration_{it}.pth'
        opath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(gaussian=gopt.state_dict(), light=lopt.state_dict(), camera_rng=camera_rng.getstate(),
                        remaining=list(stack), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        iteration=it, stage_step=step), opath)
        saves.append(dict(iteration=it, stage_step=step, points=len(pc.get_xyz), hashes=param_hashes(pc, lighting),
                          ply_sha256=sha(model / f'point_cloud/iteration_{it}/point_cloud.ply'),
                          opacity=opacity_summary(pc)))
        write_json(model / 'saves.json', saves)

    n0 = len(pc.get_xyz)
    cfgd = FROZEN_F['densify']
    metrics = (model / 'physical_metrics.jsonl').open('a')
    for step in range(1, args.steps + 1):
        lrs_now = lr_at(step)
        for g in gopt.param_groups:
            if g['name'] in lrs_now:
                g['lr'] = lrs_now[g['name']]
        if not stack:
            stack = list(range(len(cameras)))
        cam = cameras[stack.pop(camera_rng.randrange(len(stack)))]
        if step == 1:
            assert cam.image_name_train_light == initialization['expected_first_camera']
        cam.load2device()
        log = step == 1 or step % 100 == 0 or step in save_steps
        pk = render_mixed(cam, pc, pipe, background, directions, lighting, vis, clock + step, components=log)
        target, alpha = cam.original_image_train_light.cuda(), cam.gt_alpha_mask.cuda()
        sat = cam.observed_saturation_mask.cuda()
        photo, pdiag = censored_loss(pk['render'], target, alpha, sat, FROZEN['loss']['shadow_weight'],
                                     FROZEN['loss']['highlight_weight'], FROZEN['loss']['threshold'],
                                     pixel_keep=partial_alpha_keep_mask(alpha))
        geo, gpk = geometry_losses(cam, pc, pipe, background, alpha, pk, 'full')
        loss = photo + weights['alpha'] * geo['alpha'] + weights['normal'] * geo['normal'] + weights['distortion'] * geo['distortion']
        if not torch.isfinite(loss):
            raise FloatingPointError('loss非有限')
        loss.backward()
        with torch.no_grad():
            sel = pk['visibility_filter']
            pc.max_radii2D[sel] = torch.maximum(pc.max_radii2D[sel], pk['radii'][sel].float())
            dens = accumulate_screen_statistics(pc, pk, gpk, 'original', audit=step == 1)
            if step == 1:
                write_json(model / 'initial_densification_statistics_audit.json', dens['audit'])
            lrs = {g['name']: g['lr'] for g in gopt.param_groups + lopt.param_groups}
            gopt.step(); lopt.step()
            gopt.zero_grad(set_to_none=True); lopt.zero_grad(set_to_none=True)
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lighting.log_dc.clamp_(-5, 5); lighting.beta.clamp_(-6, 6)
            event = None
            if step % cfgd['interval'] == 0 and step < args.steps:
                pre = hypothetical_event(pc, extent)
                if args.densify == 'on':
                    rec = pc.densify_and_prune_audited(cfgd['grad_threshold'], cfgd['min_opacity'], extent,
                                                       cfgd['max_screen_size'], model, step, clock + step)
                    freeze_aux(pc)
                    vis.invalidate()
                    event = dict(performed=True, precheck=pre, **{k: rec[k] for k in (
                        'before', 'cloned_parents', 'split_parents_removed', 'split_children', 'low_opacity', 'screen_size',
                        'world_size', 'prune_union', 'after')})
                else:
                    # 与原算法统计窗口同步：事件后统计清零（不改参数与拓扑）
                    pc.xyz_gradient_accum.zero_(); pc.denom.zero_(); pc.max_radii2D.zero_()
                    event = dict(performed=False, precheck=pre, points=len(pc.get_xyz))
                    assert len(pc.get_xyz) == n0
                event.update(step=step, iteration=clock + step)
                events.append(event)
                with (model / 'topology_summary.jsonl').open('a') as f:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
            if args.densify == 'off':
                assert len(pc.get_xyz) == n0, '无增密对照点数变化'
            if log:
                ra, fgm = pk['rend_alpha'].detach(), alpha >= .5
                part = (alpha > 0) & (alpha < .99)
                rec = dict(step=step, iteration=clock + step, camera=cam.image_name_train_light, densify=args.densify,
                           loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geo.items()}, **pdiag,
                           **component_stats(pk, alpha), mean_visibility=float(pk['mean_visibility']),
                           direct_fraction=float(lighting.direct_fraction()), lrs_used=lrs, points=len(pc.get_xyz),
                           opacity_below_min_fraction=float((pc.get_opacity < .01).float().mean()),
                           opacity_mean=float(pc.get_opacity.mean()),
                           train_view_alpha_missing_fraction=float((fgm & (ra < .5)).sum() / fgm.sum().clamp_min(1)),
                           train_view_partial_pred_alpha_mean=float(ra[part].mean()) if part.any() else None,
                           densification_visible_points=dens['visible_points'])
                metrics.write(json.dumps(rec, ensure_ascii=False) + '\n'); metrics.flush()
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if step in save_steps:
                save(step)
        cam.load2device('cpu')
    metrics.close()
    write_json(model / 'physical_finished.json', dict(
        status='TRAINING_FINISHED', stage='F', route='full', densify=args.densify, smoke=args.smoke, clock_start=clock,
        final_iteration_relative_total=clock + args.steps, new_optimizer_updates=args.steps,
        lineage_updates=dict(S0=34999, A0=1080, G=7020, F=args.steps), points_start=n0, points=len(pc.get_xyz),
        topology_events_performed=sum(e['performed'] for e in events), opacity_resets_performed=0,
        saves=saves, final_lighting=lighting.state(), photo_partial_alpha='exclude', geometry_lr_multiplier=1.0,
        normal_weight=0.1, composite='linear',
        evaluation='尚未评测；DEV/TEST由独立评测Agent使用对应帧GT方向'))


if __name__ == '__main__':
    main()
