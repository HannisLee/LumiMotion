"""静态ELDL-V3 R阶段（reset配对，GOAL第3节R0/R1的opacity lr 0.05另立版本）：父=DEV选定并冻结的G11@8100。

Ra：opacity lr 0.05，无reset（只改opacity LR的对照）。
Rb：opacity lr 0.05，在stage步1000的optimizer更新之后执行原生reset（opacity←min(opacity,0.01)，opacity的Adam动量清零），全程full。
Rc：同Rb，但stage步1001起仅normal→opacity切为detach（DL T06机制）；reset前为full。
共同：G11配置（photo部分覆盖排除、阈值0.08、normal 0.1、linear、seed1012不放回相机序列），固定拓扑（无增密/裁剪）；
     新建零动量Adam；增密统计不使用；几何LR按G日程在2000步内重启：xyz 1e-5、scaling 1e-4（×10）、rotation 5e-5（×1）
     按0.1^((r-1)/1999)衰减；opacity 0.05恒定（原生3DGS的opacity lr，三组相同）；albedo 2e-4、照明1e-5恒定。
每步：forward/backward → optimizer →（步1000：保存reset前 → reset → 保存reset后副本 → 路由切换零更新审计）→ 保存。
保存（相对总步=8100+stage步）：9000、9100（reset前；Rb/Rc另存post_reset_9100副本）、9200、9300、10100。
不修改 scene/static_eldl_lighting.py（评测包装器按其内容核对训练快照）。
"""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scene.gaussian_model import GaussianModel
from scene.static_dl_photometric import StaticVisibility
from scene.static_dl_foot_recovery import opacity_summary
from scene.static_eldl_lighting import (TrainDirections, MixedLighting, render_mixed, censored_loss, geometry_losses,
                                        component_stats, param_hashes, audit_normal_routes, audit_physical_forward,
                                        partial_alpha_keep_mask)
from scripts.train_static_eldl_physical import FROZEN, sha, write_json, install_guard, load_cameras, audit_photo_pixel_keep
from utils.sh_utils import RGB2SH

R_STEPS, PARENT_ITERATION, RESET_STEP = 2000, 8100, 1000
FROZEN_R = dict(steps=R_STEPS, saves=[900, 1000, 1100, 1200, 2000], reset_step=RESET_STEP,
                xyz_lr=1e-5, rotation_lr=5e-5, scaling_lr=1e-4, final_factor=0.1, opacity_lr=0.05,
                albedo_lr=2e-4, lighting_lr=1e-5, reset_rule='GaussianModel.reset_opacity：min(opacity,0.01)，Adam动量清零',
                prune_after_reset=False,
                prune_note='固定拓扑不裁剪；原生路线下一次增密事件会删除仍<0.01的点，本入口逐步记录该数量供核对',
                parent_config=dict(stage='G', route='full', photo_partial_alpha='exclude', geometry_lr_multiplier=10.0,
                                   rotation_lr_multiplier=1.0, opacity_lr_multiplier=1.0, normal_weight=0.1,
                                   composite='linear', photo_threshold=0.08, seed=1012))
VARIANTS = dict(Ra=dict(reset=False, post_reset_route='full'), Rb=dict(reset=True, post_reset_route='full'),
                Rc=dict(reset=True, post_reset_route='detach_opacity'))


def lr_at(step):
    f = FROZEN_R['final_factor'] ** ((step - 1) / (R_STEPS - 1))
    return dict(xyz=FROZEN_R['xyz_lr'] * f, rotation=FROZEN_R['rotation_lr'] * f, scaling=FROZEN_R['scaling_lr'] * f,
                opacity=FROZEN_R['opacity_lr'])


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=list(VARIANTS), required=True)
    p.add_argument('--parent', type=Path, required=True, help='G11的model_mlp')
    p.add_argument('--parent_iteration', type=int, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--directions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--reset_step', type=int, default=RESET_STEP, help='仅冒烟可改')
    p.add_argument('--save_steps', type=int, nargs='*', default=None, help='仅冒烟可改')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    var = VARIANTS[args.variant]
    out = args.output.resolve(); model = out / 'model_mlp'
    parent, source, split_path, dpath = (args.parent.resolve(), args.source.resolve(), args.split.resolve(),
                                         args.directions.resolve())
    if 'output' not in out.parts or model.exists():
        p.error('输出须在output内且model_mlp尚不存在')
    if args.smoke:
        if 'smoke_test' not in out.parts or args.steps > 300:
            p.error('冒烟只能在smoke_test且≤300步')
    elif args.steps != R_STEPS or args.save_steps is not None or args.reset_step != RESET_STEP:
        p.error('正式运行预算/保存点/reset步冻结')
    reset_step = args.reset_step
    save_steps = FROZEN_R['saves'] if args.save_steps is None else sorted(set(args.save_steps + [args.steps, reset_step]))
    split = json.loads(split_path.read_text())

    # ---- 父核验：正式G11配置@8100
    fin = json.loads((parent / 'physical_finished.json').read_text())
    pp = json.loads((parent / 'physical_plan.json').read_text())
    assert fin['status'] == 'TRAINING_FINISHED' and fin['stage'] == 'G' and fin['smoke'] is False
    assert fin['new_optimizer_updates'] == 7020 and args.parent_iteration == PARENT_ITERATION
    assert not pp.get('continue_from_g', False)
    got = dict(stage=pp['stage'], route=pp['route'], photo_partial_alpha=pp.get('photo_partial_alpha', 'keep'),
               geometry_lr_multiplier=pp.get('geometry_lr_multiplier', 1.0),
               rotation_lr_multiplier=pp.get('rotation_lr_multiplier'), opacity_lr_multiplier=pp.get('opacity_lr_multiplier', 1.0),
               normal_weight=pp.get('normal_weight', 0.1), composite=pp.get('composite', 'linear'),
               photo_threshold=pp.get('photo_threshold', 0.08), seed=pp.get('seed', 1012))
    assert got == FROZEN_R['parent_config'], f'父须为G11配置：{got}'
    parent_saved = [r for r in json.loads((parent / 'saves.json').read_text()) if r['iteration'] == args.parent_iteration][0]
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
    pc = GaussianModel(3, no_binary_separation=True, fea_dim=1)
    pc.load_ply(str(ply))
    if hasattr(pc, '_albedo_dc_stage1'):
        del pc._albedo_dc_stage1
    directions = TrainDirections(dpath)
    lighting = MixedLighting(.5)
    lighting.load_state(json.loads(lpath.read_text())['lighting'])
    loaded = param_hashes(pc, lighting)
    match = {k: loaded[k] == parent_saved['hashes'][k] for k in parent_saved['hashes']}
    assert all(match.values()) and sha(ply) == parent_saved['ply_sha256'], '父加载与保存记录不一致'
    cameras, image_sha = load_cameras(source, split)
    for name in ('_albedo_rest', '_roughness', 'feature'):
        getattr(pc, name).requires_grad_(False)
    for name in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc'):
        getattr(pc, name).requires_grad_(True)
    gopt = torch.optim.Adam([dict(params=[pc._xyz], lr=0., name='xyz'), dict(params=[pc._rotation], lr=0., name='rotation'),
                             dict(params=[pc._scaling], lr=0., name='scaling'), dict(params=[pc._opacity], lr=0., name='opacity'),
                             dict(params=[pc._albedo_dc], lr=FROZEN_R['albedo_lr'], name='albedo_dc')], lr=0., eps=1e-15)
    pc.optimizer = gopt  # 原生reset_opacity按group名替换opacity并清零其Adam动量
    for prm in lighting.parameters():
        prm.requires_grad_(True)
    lopt = torch.optim.Adam([dict(params=[lighting.log_dc, lighting.higher], lr=FROZEN_R['lighting_lr'], name='env_sh'),
                             dict(params=[lighting.beta], lr=FROZEN_R['lighting_lr'], name='direct_fraction')])
    vis = StaticVisibility(FROZEN['shadow_bias'], 1)
    background = torch.zeros(3, device='cuda')
    pc.max_radii2D = torch.zeros(len(pc.get_xyz), device='cuda')
    pc.xyz_gradient_accum = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    pc.denom = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    preview = random.Random(seed)
    first_index = list(range(len(cameras))).pop(preview.randrange(len(cameras)))
    model.mkdir(parents=True)
    write_json(model / 'initialization_audit.json', dict(
        stage='R', variant=args.variant, parent=str(parent), parent_iteration=args.parent_iteration,
        parent_ply_sha256=sha(ply), parent_saved_hashes_match=match, loaded_hashes=loaded,
        adam='新建Adam，动量为零；不恢复父优化器', statistics='xyz_gradient_accum/denom/max_radii2D清零（固定拓扑不使用）',
        expected_first_camera=cameras[first_index].image_name_train_light, opacity=opacity_summary(pc)))
    weights = dict(alpha=FROZEN['loss']['alpha'], normal=FROZEN['loss']['normal'], distortion=FROZEN['loss']['distortion'])
    cam0 = cameras[first_index]
    cam0.load2device()
    audits = dict(physical=audit_physical_forward(cam0, pc, pipe, background, directions, lighting, vis, args.parent_iteration),
                  normal_route=audit_normal_routes(cam0, pc, pipe, background, directions, lighting, vis,
                                                   args.parent_iteration, weights),
                  photo_partial_alpha=audit_photo_pixel_keep(cam0, pc, pipe, background, directions, lighting, vis,
                                                             args.parent_iteration))
    cam0.load2device('cpu')
    for k, v in audits.items():
        write_json(model / f'initial_{k}_audit.json', v)
    if any(v['status'] != 'PASS' for v in audits.values()):
        raise RuntimeError('零更新审计失败')
    assert param_hashes(pc, lighting) == loaded
    clock = args.parent_iteration
    root = Path.cwd()
    write_json(model / 'physical_plan.json', dict(
        stage='R', variant=args.variant, route='full', post_reset_route=var['post_reset_route'], reset=var['reset'],
        reset_step=reset_step, args={k: str(v) for k, v in vars(args).items()}, frozen=FROZEN, frozen_r=FROZEN_R,
        clock_start=clock, steps=args.steps, save_steps_stage_relative=save_steps,
        save_iterations_relative_total=[clock + s for s in save_steps], lr_first_step=lr_at(1),
        camera_sampling=f'独立random.Random({seed})不放回遍历117相机；Ra/Rb/Rc序列相同',
        update_semantics='每步均更新（含末步）；步reset_step：optimizer→保存reset前→reset→保存reset后副本→审计；之后继续',
        topology='固定拓扑（无增密/裁剪）', inputs=dict(split=sha(split_path), directions=directions.sha256, images=image_sha),
        gt_supervision='仅TRAIN帧GT主光单位方向（固定不训练）',
        sources={f: sha(root / f) for f in ('scripts/train_static_eldl_reset.py', 'scripts/train_static_eldl_physical.py',
                                             'scene/static_eldl_lighting.py', 'scene/gaussian_model.py',
                                             'gaussian_renderer/__init__.py', 'scene/static_dl_photometric.py')},
        rasterizer=backend.provenance, smoke=args.smoke,
        photo_partial_alpha='exclude', geometry_lr_multiplier=10.0, rotation_lr_multiplier=1.0, normal_weight=0.1,
        composite='linear', photo_threshold=0.08, seed=seed, opacity_lr=FROZEN_R['opacity_lr']))
    cfg = dict(sh_degree=3, source_path=str(source), model_path=str(model), images='images', train_light_folder='images',
               test_light_folder='images', resolution=2, white_background=False, data_device='cpu', eval=True,
               load2gpu_on_the_fly=True, is_blender=True, deform_type='mlp', hyper_dim=1, pred_color=False,
               no_binary_separation=False, load_test_set_only=False, start_frame=0, end_frame=-1)
    (model / 'cfg_args').write_text(str(Namespace(**cfg)))
    shutil.copyfile(ply, model / 'input.ply')

    saves, stack = [], []
    route = 'full'

    def save(step, tag=None):
        it = clock + step
        base = model if tag is None else model / tag
        pc.save_ply(str(base / f'point_cloud/iteration_{it}/point_cloud.ply'))
        dst = base / f'deform/iteration_{it}/deform.pth'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deform, dst)
        write_json(base / f'lighting/iteration_{it}/lighting.json', dict(
            iteration_relative_total=it, stage='R', variant=args.variant, route=route, stage_step=step, tag=tag,
            new_optimizer_updates_this_stage=step, lineage_new_updates_since_S0=1080 + 7020 + step,
            directions_sha256=directions.sha256, lighting=lighting.state(), model='static_eldl_mixed_v1'))
        if tag is None:
            opath = model / f'optimizer/iteration_{it}.pth'
            opath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(gaussian=gopt.state_dict(), light=lopt.state_dict(), camera_rng=camera_rng.getstate(),
                            remaining=list(stack), iteration=it, stage_step=step), opath)
        saves.append(dict(iteration=it, stage_step=step, tag=tag, route=route, hashes=param_hashes(pc, lighting),
                          ply_sha256=sha(base / f'point_cloud/iteration_{it}/point_cloud.ply'), opacity=opacity_summary(pc)))
        write_json(model / 'saves.json', saves)

    metrics = (model / 'physical_metrics.jsonl').open('a')
    reset_record = None
    for step in range(1, args.steps + 1):
        lrs_now = lr_at(step)
        for g in gopt.param_groups:
            if g['name'] in lrs_now:
                g['lr'] = lrs_now[g['name']]
        if not stack:
            stack = list(range(len(cameras)))
        cam = cameras[stack.pop(camera_rng.randrange(len(stack)))]
        if step == 1:
            assert cam.image_name_train_light == cameras[first_index].image_name_train_light
        cam.load2device()
        dense = reset_step - 10 <= step <= reset_step + 300
        log = step == 1 or step % 100 == 0 or step in save_steps or (dense and step % 10 == 0)
        pk = render_mixed(cam, pc, pipe, background, directions, lighting, vis, clock + step, components=log)
        target, alpha = cam.original_image_train_light.cuda(), cam.gt_alpha_mask.cuda()
        sat = cam.observed_saturation_mask.cuda()
        photo, pdiag = censored_loss(pk['render'], target, alpha, sat, FROZEN['loss']['shadow_weight'],
                                     FROZEN['loss']['highlight_weight'], FROZEN['loss']['threshold'],
                                     pixel_keep=partial_alpha_keep_mask(alpha))
        geo, _ = geometry_losses(cam, pc, pipe, background, alpha, pk, route)
        loss = photo + weights['alpha'] * geo['alpha'] + weights['normal'] * geo['normal'] + weights['distortion'] * geo['distortion']
        if not torch.isfinite(loss):
            raise FloatingPointError('loss非有限')
        loss.backward()
        with torch.no_grad():
            lrs = {g['name']: g['lr'] for g in gopt.param_groups + lopt.param_groups}
            gopt.step(); lopt.step()
            gopt.zero_grad(set_to_none=True); lopt.zero_grad(set_to_none=True)
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lighting.log_dc.clamp_(-5, 5); lighting.beta.clamp_(-6, 6)
            if log:
                ra, fgm = pk['rend_alpha'].detach(), alpha >= .5
                rec = dict(step=step, iteration=clock + step, camera=cam.image_name_train_light, variant=args.variant,
                           route=route, loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geo.items()}, **pdiag,
                           **component_stats(pk, alpha), lrs_used=lrs, opacity_mean=float(pc.get_opacity.mean()),
                           opacity_below_001_fraction=float((pc.get_opacity < .01).float().mean()),
                           opacity_below_0005_fraction=float((pc.get_opacity < .005).float().mean()),
                           train_view_alpha_missing_fraction=float((fgm & (ra < .5)).sum() / fgm.sum().clamp_min(1)),
                           train_view_pred_alpha_mean_fg=float(ra[fgm].mean()))
                metrics.write(json.dumps(rec, ensure_ascii=False) + '\n'); metrics.flush()
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if step in save_steps:
                save(step)
        if step == reset_step and var['reset']:
            before = opacity_summary(pc)
            st = gopt.state.get(pc._opacity, {})
            adam_before = {k: float(v.norm()) if torch.is_tensor(v) and v.ndim else (float(v) if torch.is_tensor(v) else v)
                           for k, v in st.items()}
            pc.reset_opacity()
            vis.invalidate()
            st = gopt.state.get(pc._opacity, {})
            adam_after = {k: float(v.norm()) if torch.is_tensor(v) and v.ndim else (float(v) if torch.is_tensor(v) else v)
                          for k, v in st.items()}
            route = var['post_reset_route']
            with torch.no_grad():
                save(step, tag='post_reset')
            cam0.load2device()
            post_audit = audit_normal_routes(cam0, pc, pipe, background, directions, lighting, vis, clock + step, weights)
            cam0.load2device('cpu')
            reset_record = dict(step=step, iteration=clock + step, before=before, after=opacity_summary(pc),
                                adam_opacity_before=adam_before, adam_opacity_after=adam_after,
                                max_opacity_after=float(pc.get_opacity.max()), post_reset_route=route,
                                route_switch_audit_status=post_audit['status'],
                                order='optimizer更新→保存reset前（标准iteration）→reset→保存post_reset副本→路由切换零更新审计→下一步')
            write_json(model / 'reset_event.json', reset_record)
            write_json(model / 'post_reset_route_audit.json', post_audit)
            if post_audit['status'] != 'PASS' or reset_record['max_opacity_after'] > .0100001:
                raise RuntimeError('reset后审计失败')
        cam.load2device('cpu')
    metrics.close()
    write_json(model / 'physical_finished.json', dict(
        status='TRAINING_FINISHED', stage='R', variant=args.variant, route='full', post_reset_route=var['post_reset_route'],
        smoke=args.smoke, clock_start=clock, final_iteration_relative_total=clock + args.steps,
        new_optimizer_updates=args.steps, lineage_updates=dict(S0=34999, A0=1080, G11=7020, R=args.steps),
        opacity_resets_performed=int(reset_record is not None), reset_event=reset_record, saves=saves,
        final_lighting=lighting.state(), evaluation='尚未评测；post_reset副本不用于评测'))


if __name__ == '__main__':
    main()
