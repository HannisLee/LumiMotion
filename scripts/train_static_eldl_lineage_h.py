"""静态ELDL-V3 H谱系物理几何阶段：父=H-A0@1080（S0@10000→A0式照明适配1080步，几何冻结）。

目的：检验成熟35k原生几何是否为人体误差瓶颈。从S0@10000（增密窗口仍开启的中间状态）切入物理着色，
按S0原生日程继续（全局等效步 g = 10000 + h，h=本阶段步1..20000）：
- 几何LR：xyz=原生expon(8e-4→8e-6, delay_mult 0.01, max_steps 30000)在g处取值；scaling 1e-2；opacity 0.05（S0原生值）；
  rotation按--rotation_mode：native=1e-3（S0原生，H1）或 g_level=5e-5（G4/G11的rotation水平，H2）——唯一差别；
  albedo 2e-4、照明1e-5（G阶段值）。辅助组albedo_rest/roughness/feature lr=0，仅供拓扑算子同步。
- 增密：S0原生窗口 g<20000（densify_until_iter=20000）且g%100==0（h=100..9900），阈值0.0002，min opacity 0.01，
  percent_dense 0.01，extent=S0记录4.3194；max_screen_size=None（理由见修改/1025-04第3节：原算法clone/split清零
  max_radii2D，屏幕裁剪只在无clone/split的事件生效；None在有clone/split时与20逐位相同）。
- reset：S0原生 g%3000==0 且 g<20000 → g=12000/15000/18000（h=2000/5000/8000）；同步骤先增密后reset（原生顺序）。
- normal→opacity：首次reset前full；首次reset后（h≥2001）仅detach（Rc/DL T06机制），其余路径不变。
- loss：同G11（censored photo＋GT Alpha∈(0,0.99)排除、阈值0.08、0.1 Alpha BCE、0.1 normal、1000 distortion）。
- 新建零动量Adam；增密统计清零；屏幕梯度统计=photo包与normal副本viewspace梯度先相加再累计（original）。
每步：forward/backward → 统计 → optimizer → 增密/裁剪 → reset（含路由切换审计）→ 保存。终点不做拓扑。
保存：h=3000/6000/9000/12000/16000/20000（相对总步1080+h）。
不修改 scene/static_eldl_lighting.py。
"""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from arguments import OptimizationParams
from scene.static_dl_photometric import StaticVisibility
from scene.static_dl_foot_recovery import FootRecoveryGaussianModel, opacity_summary, audit_topology_equivalence
from scene.static_dl_foot_densify import accumulate_screen_statistics
from scene.static_eldl_lighting import (TrainDirections, MixedLighting, render_mixed, censored_loss, geometry_losses,
                                        component_stats, param_hashes, audit_normal_routes, audit_physical_forward,
                                        partial_alpha_keep_mask)
from scripts.train_static_eldl_physical import FROZEN, sha, write_json, install_guard, load_cameras, audit_photo_pixel_keep
from utils.general_utils import get_expon_lr_func
from utils.sh_utils import RGB2SH

H_STEPS, A0_CLOCK, GLOBAL_START = 20000, 1080, 10000
FROZEN_H = dict(
    steps=H_STEPS, saves=[3000, 6000, 9000, 12000, 16000, 20000], global_start=GLOBAL_START,
    xyz=dict(lr_init=1.6e-4 * 5, lr_final=1.6e-6 * 5, lr_delay_mult=0.01, max_steps=30000, note='S0原生position_lr×spatial_lr_scale5'),
    scaling_lr=0.002 * 5, opacity_lr=0.05, rotation_lr=dict(native=1e-3, g_level=5e-5), albedo_lr=2e-4, lighting_lr=1e-5,
    densify=dict(interval=100, from_iter=500, until_iter=20000, grad_threshold=2e-4, min_opacity=0.01, percent_dense=0.01,
                 max_screen_size=None, extent=4.319371700286865, statistics_route='original'),
    reset=dict(interval=3000, until_iter=20000, rule='GaussianModel.reset_opacity（min(op,0.01)，opacity Adam动量清零）'),
    route=dict(before_first_reset='full', after_first_reset='detach_opacity'),
    loss=FROZEN['loss'], photo_partial_alpha='exclude', photo_threshold=0.08)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--rotation_mode', choices=['native', 'g_level'], required=True)
    p.add_argument('--parent', type=Path, required=True, help='H-A0的model_mlp')
    p.add_argument('--parent_iteration', type=int, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--directions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--global_start', type=int, default=GLOBAL_START, help='仅冒烟可改（如11900使h=100即遇reset）')
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
    elif args.steps != H_STEPS or args.save_steps is not None or args.global_start != GLOBAL_START:
        p.error('正式运行预算/保存点/全局起点冻结')
    save_steps = FROZEN_H['saves'] if args.save_steps is None else sorted(set(args.save_steps + [args.steps]))
    split = json.loads(split_path.read_text())

    fin = json.loads((parent / 'physical_finished.json').read_text())
    pp = json.loads((parent / 'physical_plan.json').read_text())
    assert fin['status'] == 'TRAINING_FINISHED' and fin['stage'] == 'A0' and args.parent_iteration == A0_CLOCK
    assert fin['new_optimizer_updates'] == 1080 or args.smoke
    if not args.smoke:
        assert fin['smoke'] is False and pp.get('s0_parent_iteration') == 10000, '父须为S0@10000的正式H-A0'
    ply = parent / f'point_cloud/iteration_{args.parent_iteration}/point_cloud.ply'
    deform = parent / f'deform/iteration_{args.parent_iteration}/deform.pth'
    lpath = parent / f'lighting/iteration_{args.parent_iteration}/lighting.json'
    state = torch.load(deform, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for s in ('weight', 'bias'):
            assert torch.count_nonzero(state[name + '.' + s]) == 0
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
    pc.percent_dense = FROZEN_H['densify']['percent_dense']
    directions = TrainDirections(dpath)
    lighting = MixedLighting(.5)
    lighting.load_state(json.loads(lpath.read_text())['lighting'])
    loaded = param_hashes(pc, lighting)
    parent_saved = [r for r in json.loads((parent / 'saves.json').read_text()) if r['iteration'] == args.parent_iteration][0]
    match = {k: loaded[k] == parent_saved['hashes'][k] for k in parent_saved['hashes']}
    assert all(match.values()) and sha(ply) == parent_saved['ply_sha256'], '父加载与保存记录不一致'
    cameras, image_sha = load_cameras(source, split)
    extent = FROZEN_H['densify']['extent']
    xyz_lr = get_expon_lr_func(lr_init=FROZEN_H['xyz']['lr_init'], lr_final=FROZEN_H['xyz']['lr_final'],
                               lr_delay_mult=FROZEN_H['xyz']['lr_delay_mult'], max_steps=FROZEN_H['xyz']['max_steps'])
    rot_lr = FROZEN_H['rotation_lr'][args.rotation_mode]
    for name in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc'):
        getattr(pc, name).requires_grad_(True)

    def freeze_aux():
        for name in ('_albedo_rest', '_roughness', 'feature'):
            getattr(pc, name).requires_grad_(False)
    freeze_aux()
    gopt = torch.optim.Adam([dict(params=[pc._xyz], lr=0., name='xyz'), dict(params=[pc._rotation], lr=rot_lr, name='rotation'),
                             dict(params=[pc._scaling], lr=FROZEN_H['scaling_lr'], name='scaling'),
                             dict(params=[pc._opacity], lr=FROZEN_H['opacity_lr'], name='opacity'),
                             dict(params=[pc._albedo_dc], lr=FROZEN_H['albedo_lr'], name='albedo_dc'),
                             dict(params=[pc._albedo_rest], lr=0., name='albedo_rest'),
                             dict(params=[pc._roughness], lr=0., name='roughness'),
                             dict(params=[pc.feature], lr=0., name='feature')], lr=0., eps=1e-15)
    pc.optimizer = gopt
    for prm in lighting.parameters():
        prm.requires_grad_(True)
    lopt = torch.optim.Adam([dict(params=[lighting.log_dc, lighting.higher], lr=FROZEN_H['lighting_lr'], name='env_sh'),
                             dict(params=[lighting.beta], lr=FROZEN_H['lighting_lr'], name='direct_fraction')])
    vis = StaticVisibility(FROZEN['shadow_bias'], 1)
    background = torch.zeros(3, device='cuda')
    pc.max_radii2D = torch.zeros(len(pc.get_xyz), device='cuda')
    pc.xyz_gradient_accum = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    pc.denom = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    preview = random.Random(seed)
    first_index = list(range(len(cameras))).pop(preview.randrange(len(cameras)))
    with torch.no_grad():
        ms = pc.get_scaling.max(dim=1).values
        parent_stats = dict(points=len(ms), extent=extent, split_eligible_fraction=float((ms > .01 * extent).float().mean()),
                            world_over=int((ms > .1 * extent).sum()), opacity=opacity_summary(pc),
                            max_scale_quantiles=torch.quantile(ms, torch.tensor([.1, .5, .9, .99], device='cuda')).tolist())
    model.mkdir(parents=True)
    write_json(model / 'initialization_audit.json', dict(
        stage='H', rotation_mode=args.rotation_mode, parent=str(parent), parent_iteration=args.parent_iteration,
        parent_ply_sha256=sha(ply), parent_saved_hashes_match=match, loaded_hashes=loaded, global_start=args.global_start,
        adam='新建Adam，动量为零', statistics='xyz_gradient_accum/denom/max_radii2D清零',
        expected_first_camera=cameras[first_index].image_name_train_light, parent_point_cloud_statistics=parent_stats))
    weights = dict(alpha=FROZEN['loss']['alpha'], normal=FROZEN['loss']['normal'], distortion=FROZEN['loss']['distortion'])
    cam0 = cameras[first_index]
    cam0.load2device()
    audits = dict(physical=audit_physical_forward(cam0, pc, pipe, background, directions, lighting, vis, A0_CLOCK),
                  normal_route=audit_normal_routes(cam0, pc, pipe, background, directions, lighting, vis, A0_CLOCK, weights),
                  photo_partial_alpha=audit_photo_pixel_keep(cam0, pc, pipe, background, directions, lighting, vis, A0_CLOCK))
    cam0.load2device('cpu')
    opt_ns = Namespace(**vars(OptimizationParams(ArgumentParser())))
    opt_ns.sh_degree, opt_ns.hyper_dim = 3, 1
    audits['topology_operator'] = audit_topology_equivalence(ply, opt_ns, extent, model)
    for k, v in audits.items():
        write_json(model / f'initial_{k}_audit.json', v)
    if any(v['status'] != 'PASS' for v in audits.values()):
        raise RuntimeError('零更新审计失败')
    assert param_hashes(pc, lighting) == loaded
    clock = A0_CLOCK
    gs = args.global_start
    dcfg, rcfg = FROZEN_H['densify'], FROZEN_H['reset']
    events_plan = dict(densify=[h for h in range(1, args.steps) if (gs + h) < dcfg['until_iter'] and (gs + h) > dcfg['from_iter']
                                and (gs + h) % dcfg['interval'] == 0],
                       reset=[h for h in range(1, args.steps) if (gs + h) < rcfg['until_iter'] and (gs + h) % rcfg['interval'] == 0])
    root = Path.cwd()
    write_json(model / 'physical_plan.json', dict(
        stage='H', rotation_mode=args.rotation_mode, route='full', post_reset_route='detach_opacity',
        args={k: str(v) for k, v in vars(args).items()}, frozen=FROZEN, frozen_h=FROZEN_H, clock_start=clock,
        global_start=gs, steps=args.steps, save_steps_stage_relative=save_steps,
        save_iterations_relative_total=[clock + s for s in save_steps],
        densify_steps=events_plan['densify'], reset_steps=events_plan['reset'],
        lr_first_step=dict(xyz=xyz_lr(gs + 1), rotation=rot_lr, scaling=FROZEN_H['scaling_lr'], opacity=FROZEN_H['opacity_lr']),
        camera_sampling=f'独立random.Random({seed})不放回遍历117相机；H1/H2序列相同',
        update_semantics='每步均更新（含末步）；forward/backward→统计→optimizer→增密/裁剪→reset→保存；终点无拓扑',
        inputs=dict(split=sha(split_path), directions=directions.sha256, images=image_sha),
        gt_supervision='仅TRAIN帧GT主光单位方向（固定不训练）',
        sources={f: sha(root / f) for f in ('scripts/train_static_eldl_lineage_h.py', 'scripts/train_static_eldl_physical.py',
                                             'scene/static_eldl_lighting.py', 'scene/static_dl_foot_recovery.py',
                                             'scene/static_dl_foot_densify.py', 'scene/gaussian_model.py',
                                             'gaussian_renderer/__init__.py', 'scene/static_dl_photometric.py')},
        rasterizer=backend.provenance, smoke=args.smoke,
        photo_partial_alpha='exclude', photo_threshold=0.08, normal_weight=0.1, composite='linear', seed=seed,
        geometry_lr_multiplier=None, rotation_lr_multiplier=None))
    cfg = dict(sh_degree=3, source_path=str(source), model_path=str(model), images='images', train_light_folder='images',
               test_light_folder='images', resolution=2, white_background=False, data_device='cpu', eval=True,
               load2gpu_on_the_fly=True, is_blender=True, deform_type='mlp', hyper_dim=1, pred_color=False,
               no_binary_separation=False, load_test_set_only=False, start_frame=0, end_frame=-1)
    (model / 'cfg_args').write_text(str(Namespace(**cfg)))
    shutil.copyfile(ply, model / 'input.ply')

    saves, stack, events = [], [], []
    route = 'full'
    resets_done = 0

    def save(step):
        it = clock + step
        pc.save_ply(str(model / f'point_cloud/iteration_{it}/point_cloud.ply'))
        dst = model / f'deform/iteration_{it}/deform.pth'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deform, dst)
        write_json(model / f'lighting/iteration_{it}/lighting.json', dict(
            iteration_relative_total=it, stage='H', rotation_mode=args.rotation_mode, route=route, stage_step=step,
            global_equivalent=gs + step, new_optimizer_updates_this_stage=step,
            lineage_updates=dict(S0=9999, A0=1080, H=step), directions_sha256=directions.sha256,
            lighting=lighting.state(), model='static_eldl_mixed_v1'))
        opath = model / f'optimizer/iteration_{it}.pth'
        opath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(gaussian=gopt.state_dict(), light=lopt.state_dict(), camera_rng=camera_rng.getstate(),
                        remaining=list(stack), iteration=it, stage_step=step), opath)
        saves.append(dict(iteration=it, stage_step=step, global_equivalent=gs + step, route=route, points=len(pc.get_xyz),
                          resets_done=resets_done, hashes=param_hashes(pc, lighting),
                          ply_sha256=sha(model / f'point_cloud/iteration_{it}/point_cloud.ply'), opacity=opacity_summary(pc)))
        write_json(model / 'saves.json', saves)

    metrics = (model / 'physical_metrics.jsonl').open('a')
    for step in range(1, args.steps + 1):
        g = gs + step
        lrs_now = dict(xyz=xyz_lr(g), rotation=rot_lr, scaling=FROZEN_H['scaling_lr'], opacity=FROZEN_H['opacity_lr'])
        for grp in gopt.param_groups:
            if grp['name'] in lrs_now:
                grp['lr'] = lrs_now[grp['name']]
        if not stack:
            stack = list(range(len(cameras)))
        cam = cameras[stack.pop(camera_rng.randrange(len(stack)))]
        if step == 1:
            assert cam.image_name_train_light == cameras[first_index].image_name_train_light
        cam.load2device()
        near_reset = any(0 <= step - r <= 500 for r in events_plan['reset'])
        log = step == 1 or step % 100 == 0 or step in save_steps or (near_reset and step % 20 == 0)
        pk = render_mixed(cam, pc, pipe, background, directions, lighting, vis, clock + step, components=log)
        target, alpha = cam.original_image_train_light.cuda(), cam.gt_alpha_mask.cuda()
        sat = cam.observed_saturation_mask.cuda()
        photo, pdiag = censored_loss(pk['render'], target, alpha, sat, FROZEN['loss']['shadow_weight'],
                                     FROZEN['loss']['highlight_weight'], FROZEN['loss']['threshold'],
                                     pixel_keep=partial_alpha_keep_mask(alpha))
        geo, gpk = geometry_losses(cam, pc, pipe, background, alpha, pk, route)
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
            lrs = {grp['name']: grp['lr'] for grp in gopt.param_groups + lopt.param_groups}
            gopt.step(); lopt.step()
            gopt.zero_grad(set_to_none=True); lopt.zero_grad(set_to_none=True)
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lighting.log_dc.clamp_(-5, 5); lighting.beta.clamp_(-6, 6)
            if step in events_plan['densify'] and step < args.steps:
                rec = pc.densify_and_prune_audited(dcfg['grad_threshold'], dcfg['min_opacity'], extent,
                                                   dcfg['max_screen_size'], model, step, clock + step)
                freeze_aux(); vis.invalidate()
                events.append(dict(kind='densify', step=step, global_equivalent=g, **{k: rec[k] for k in (
                    'before', 'cloned_parents', 'split_parents_removed', 'low_opacity', 'prune_union', 'after')}))
        if step in events_plan['reset'] and step < args.steps:
            before = opacity_summary(pc)
            pc.reset_opacity()
            freeze_aux(); vis.invalidate()
            resets_done += 1
            ev = dict(kind='reset', step=step, global_equivalent=g, before=before, after=opacity_summary(pc),
                      route_before=route)
            if route == 'full':
                route = FROZEN_H['route']['after_first_reset']
                cam0.load2device()
                ra = audit_normal_routes(cam0, pc, pipe, background, directions, lighting, vis, clock + step, weights)
                cam0.load2device('cpu')
                write_json(model / 'route_switch_audit.json', ra)
                ev['route_switch_audit'] = ra['status']
                if ra['status'] != 'PASS':
                    raise RuntimeError('路由切换审计失败')
            ev['route_after'] = route
            events.append(ev)
        if events and events[-1]['step'] == step:
            with (model / 'events.jsonl').open('a') as f:
                for e in [e for e in events if e['step'] == step]:
                    f.write(json.dumps(e, ensure_ascii=False) + '\n')
        with torch.no_grad():
            if log:
                ra_, fgm = pk['rend_alpha'].detach(), alpha >= .5
                rec = dict(step=step, iteration=clock + step, global_equivalent=g, camera=cam.image_name_train_light,
                           route=route, loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geo.items()}, **pdiag,
                           **component_stats(pk, alpha), lrs_used=lrs, points=len(pc.get_xyz), resets_done=resets_done,
                           opacity_mean=float(pc.get_opacity.mean()),
                           opacity_below_001_fraction=float((pc.get_opacity < .01).float().mean()),
                           train_view_alpha_missing_fraction=float((fgm & (ra_ < .5)).sum() / fgm.sum().clamp_min(1)),
                           train_view_pred_alpha_mean_fg=float(ra_[fgm].mean()))
                metrics.write(json.dumps(rec, ensure_ascii=False) + '\n'); metrics.flush()
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if step in save_steps:
                save(step)
        cam.load2device('cpu')
    metrics.close()
    write_json(model / 'physical_finished.json', dict(
        status='TRAINING_FINISHED', stage='H', rotation_mode=args.rotation_mode, route='full',
        post_reset_route='detach_opacity', smoke=args.smoke, clock_start=clock, global_start=gs,
        final_iteration_relative_total=clock + args.steps, new_optimizer_updates=args.steps,
        lineage_updates=dict(S0=9999, A0=1080, H=args.steps), points=len(pc.get_xyz),
        densify_events=sum(e['kind'] == 'densify' for e in events), opacity_resets_performed=resets_done,
        saves=saves, final_lighting=lighting.state(), evaluation='尚未评测'))


if __name__ == '__main__':
    main()
