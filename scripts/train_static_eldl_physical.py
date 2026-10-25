"""静态ELDL-V3 物理续训入口：A0照明适配（固定几何）与 G0/G1 固定拓扑几何优化。

A0：继承S0@35000，共1080次新增更新；1–540只学反照率DC与共享normal-SH环境（色度+形状），直射比例固定；
    541–1080再加入直射比例（环境/直射幅度共同拟合）。几何（xyz/rotation/scaling/opacity）全程冻结。
G ：继承A0@1080（相对总步1080），各7020次新增更新至相对总步8100；固定拓扑、无增密/裁剪/reset；
    --route full（G0）或 detach_opacity（G1，仅normal副本对opacity停止梯度）。
损失：censored鲁棒RGB(0.08/0.25/0.5，满码单边) + 0.1 Alpha BCE + 0.1 expected-depth法线一致性 + 1000 distortion；
A0几何冻结，只有RGB项对可训练参数有梯度（其余三项只记录）。
每步顺序：forward/backward → optimizer → （无拓扑）→ 保存；保存的 iteration k 含 k 次本谱系新增更新（A0）或 k-1080 次（G）。
"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.static_dl_photometric import StaticVisibility
from scene.static_eldl_lighting import (TrainDirections, MixedLighting, render_mixed, censored_loss, geometry_losses,
                                        component_stats, param_hashes, audit_normal_routes, audit_physical_forward,
                                        partial_alpha_keep_mask)
from scene.static_eldl_composite import render_mixed_srgb, audit_composite, COMPOSITES
from scene.stage2_photometric import frame_key
from utils.camera_utils import cameraList_from_camInfos
from utils.sh_utils import RGB2SH

A0_STEPS, A0_PHASE1, G_STEPS, G_CLOCK_START = 1080, 540, 7020, 1080
FROZEN = dict(
    A0=dict(albedo_lr=1e-3, lighting_lr=1e-3, initial_direct_fraction=0.5, geometry_frozen=True,
            bvh_interval=10 ** 9, saves=[0, 540, 1080]),
    G=dict(xyz_lr=1e-6, rotation_lr=5e-5, scaling_lr=1e-5, opacity_lr=1e-3, geometry_lr_final_factor=0.1,
           albedo_lr=2e-4, lighting_lr=1e-5, bvh_interval=1, saves_relative_total=[4050, 8100]),
    loss=dict(shadow_weight=.25, highlight_weight=.5, threshold=.08, alpha=.1, normal=.1, distortion=1000.),
    seed=1012, shadow_bias=.03, resolution=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def install_guard(allowed):
    allowed = [Path(a).resolve() for a in allowed]

    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if any(path.is_relative_to(a) for a in allowed):
            return
        if path.suffix.lower() in {'.ply', '.pth', '.pt', '.exr', '.blend', '.png', '.jpg', '.jpeg', '.npz'}:
            raise PermissionError('训练禁止读取白名单外模型/图像/GT：' + str(path))
        if path.name in {'lights.json', 'object_pose.json', 'camera.json', 'bone_pose.json'}:
            raise PermissionError('训练禁止读取受限真值：' + str(path))
    sys.addaudithook(guard)


def load_cameras(source, split):
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', 'images', False, no_bg=True)
    cameras = cameraList_from_camInfos(infos, 1., Namespace(resolution=FROZEN['resolution'], data_device='cpu',
                                                            load2gpu_on_the_fly=True))
    frames = sorted(int(frame_key(c.image_name_train_light)) for c in cameras)
    assert frames == split['train_frames'] and len(cameras) == 117
    assert all(hasattr(c, 'observed_saturation_mask') for c in cameras)
    return cameras, {str(Path(i.image_path_train_light)): sha(i.image_path_train_light) for i in infos}


def build_optimizers(stage, pc, lighting, phase):
    for name in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc', '_albedo_rest', '_roughness', 'feature'):
        if hasattr(pc, name) and isinstance(getattr(pc, name), torch.nn.Parameter):
            getattr(pc, name).requires_grad_(False)
    pc._albedo_dc.requires_grad_(True)
    if stage == 'A0':
        groups = [dict(params=[pc._albedo_dc], lr=FROZEN['A0']['albedo_lr'], name='albedo_dc')]
        light_groups = [dict(params=[lighting.log_dc, lighting.higher], lr=FROZEN['A0']['lighting_lr'], name='env_sh')]
        lighting.beta.requires_grad_(phase == 2)
        lighting.log_dc.requires_grad_(True); lighting.higher.requires_grad_(True)
    else:
        cfg = FROZEN['G']
        for name in ('_xyz', '_rotation', '_scaling', '_opacity'):
            getattr(pc, name).requires_grad_(True)
        groups = [dict(params=[pc._xyz], lr=cfg['xyz_lr'], name='xyz'),
                  dict(params=[pc._rotation], lr=cfg['rotation_lr'], name='rotation'),
                  dict(params=[pc._scaling], lr=cfg['scaling_lr'], name='scaling'),
                  dict(params=[pc._opacity], lr=cfg['opacity_lr'], name='opacity'),
                  dict(params=[pc._albedo_dc], lr=cfg['albedo_lr'], name='albedo_dc')]
        for p in lighting.parameters():
            p.requires_grad_(True)
        light_groups = [dict(params=[lighting.log_dc, lighting.higher], lr=cfg['lighting_lr'], name='env_sh'),
                        dict(params=[lighting.beta], lr=cfg['lighting_lr'], name='direct_fraction')]
    gopt = torch.optim.Adam(groups, lr=0., eps=1e-15)
    lopt = torch.optim.Adam(light_groups)
    return gopt, lopt


def geometry_lr(step, multiplier=1.0, opacity_multiplier=1.0, rotation_multiplier=None):
    """multiplier只作用于xyz/rotation/scaling；opacity_multiplier（1025补充，1=G0，0=冻结opacity）只作用于opacity。"""
    cfg = FROZEN['G']
    t = (step - 1) / (G_STEPS - 1)
    f = cfg['geometry_lr_final_factor'] ** t
    rot = multiplier if rotation_multiplier is None else rotation_multiplier  # 1025 G9：None=跟随multiplier（现状）
    return dict(xyz=cfg['xyz_lr'] * f * multiplier, rotation=cfg['rotation_lr'] * f * rot,
                scaling=cfg['scaling_lr'] * f * multiplier, opacity=cfg['opacity_lr'] * f * opacity_multiplier)


def audit_photo_threshold(camera, pc, pipe, background, directions, lighting, visibility, iteration, threshold, keep_partial):
    """1025 G10零更新审计：阈值下调只改变 threshold<|signed|≤0.08 且Alpha>0.5 像素的权重；其余像素photo梯度与0.08逐元素相同。"""
    before = param_hashes(pc, lighting)
    target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
    sat = camera.observed_saturation_mask.cuda()
    pk = render_mixed(camera, pc, pipe, background, directions, lighting, visibility, iteration)
    image = pk['render'].detach().requires_grad_(True)
    keep = partial_alpha_keep_mask(alpha) if keep_partial else None
    base, _ = censored_loss(image, target, alpha, sat, threshold=FROZEN['loss']['threshold'], pixel_keep=keep)
    new, _ = censored_loss(image, target, alpha, sat, threshold=threshold, pixel_keep=keep)
    gb = torch.autograd.grad(base, image)[0]
    gn = torch.autograd.grad(new, image)[0]
    signed = (image.detach() - target).mean(0, keepdim=True).abs()
    band = ((signed > threshold) & (signed <= FROZEN['loss']['threshold']) & (alpha > .5)).expand_as(gb)
    checks = dict(band_nonempty=dict(passed=bool(band.any()), fraction=float(band[:1].float().mean())),
                  outside_band_gradient_identical=dict(passed=bool(torch.equal(gb[~band], gn[~band]))),
                  band_gradient_changed=dict(passed=bool((gb[band] != gn[band]).any())),
                  loss_not_increased=dict(passed=bool(new <= base), base=float(base), new=float(new)),
                  parameters_unchanged=dict(passed=before == param_hashes(pc, lighting)))
    return dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'FAILED', checks=checks,
                threshold=threshold, camera=camera.image_name_train_light, parameter_updates=0)


def audit_photo_pixel_keep(camera, pc, pipe, background, directions, lighting, visibility, iteration):
    """1025零更新审计：部分覆盖掩膜只把被排除像素的photo对渲染图梯度置0，保留像素梯度与G0逐元素相同。"""
    before = param_hashes(pc, lighting)
    target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
    sat = camera.observed_saturation_mask.cuda()
    pk = render_mixed(camera, pc, pipe, background, directions, lighting, visibility, iteration)
    image = pk['render'].detach().requires_grad_(True)
    keep = partial_alpha_keep_mask(alpha)
    base, _ = censored_loss(image, target, alpha, sat)
    masked, diag = censored_loss(image, target, alpha, sat, pixel_keep=keep)
    gb = torch.autograd.grad(base, image)[0]
    gm = torch.autograd.grad(masked, image)[0]
    k = keep.bool().expand_as(gb)
    checks = dict(
        excluded_pixels_nonzero=dict(passed=bool((~keep.bool()).any()), fraction=diag['photo_pixel_excluded_fraction']),
        excluded_are_partial_alpha=dict(passed=bool(((alpha[~keep.bool()] > 0) & (alpha[~keep.bool()] < .99)).all())),
        kept_gradient_identical=dict(passed=bool(torch.equal(gb[k], gm[k]))),
        excluded_gradient_zero=dict(passed=bool(torch.count_nonzero(gm[~k]) == 0)),
        baseline_had_gradient_on_excluded=dict(passed=bool(gb[~k].abs().sum() > 0), abs_sum=float(gb[~k].abs().sum())),
        loss_not_increased=dict(passed=bool(masked <= base), base=float(base), masked=float(masked)),
        parameters_unchanged=dict(passed=before == param_hashes(pc, lighting)))
    return dict(status='PASS' if all(c['passed'] for c in checks.values()) else 'FAILED', checks=checks,
                camera=camera.image_name_train_light, parameter_updates=0)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['A0', 'G'], required=True)
    p.add_argument('--parent', type=Path, required=True, help='父model_mlp目录')
    p.add_argument('--parent_iteration', type=int, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--directions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='实验目录；模型写入其下model_mlp')
    p.add_argument('--route', choices=['full', 'detach_opacity'])
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--save_steps', type=int, nargs='*', default=None, help='仅冒烟可改：本阶段相对新增步')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--geometry_lr_multiplier', type=float, default=1.0, choices=[1.0, 3.0, 10.0, 30.0],
                   help='G阶段xyz/rotation/scaling学习率倍率（预登记：1=G0/G1，10=G2/G3，3=1025第三轮G7）')
    p.add_argument('--s0_parent_iteration', type=int, default=35000, choices=[35000, 10000],
                   help='1026 H谱系：A0父为S0的哪个保存点；35000=主线（现状），10000=H谱系（S0中间保存点，9999次更新）')
    p.add_argument('--continue_from_g', action='store_true',
                   help='1025 G12：父为正式G阶段模型@8100（同配置续训7020步，新Adam，G日程重启）；默认关闭=父须为A0（现状）')
    p.add_argument('--seed', type=int, default=FROZEN['seed'], choices=[1012, 2024, 3036],
                   help='1025 种子复验：相机采样与全局RNG seed；1012=G0–G10（FROZEN）；2024/3036仅用于G4-s2/s3噪声估计')
    p.add_argument('--photo_threshold', type=float, default=FROZEN['loss']['threshold'], choices=[0.08, 0.036],
                   help='1025 G10：censored photo符号阈值；0.08=G0–G9（FROZEN）；0.036=由A0@1080 TRAIN残差冻结（2·1.4826·median|r|）')
    p.add_argument('--rotation_lr_multiplier', type=float, default=None, choices=[0.3, 1.0, 3.0, 10.0],
                   help='1025 G9：rotation单独倍率；默认None=跟随--geometry_lr_multiplier（与G0–G8逐位相同）')
    p.add_argument('--normal_weight', type=float, default=FROZEN['loss']['normal'], choices=[0.1, 0.3],
                   help='1025第三轮：expected-depth法线一致性权重；0.1=G0（FROZEN），0.3=G6')
    p.add_argument('--composite', choices=list(COMPOSITES), default='linear',
                   help='1025第三轮：linear=sRGB(线性合成)（G0）；srgb_per_gaussian=逐Gaussian先sRGB再合成（G8）')
    p.add_argument('--photo_partial_alpha', choices=['keep', 'exclude'], default='keep',
                   help='1025：GT Alpha部分覆盖像素(0<a<0.99)的photo项；keep=G0，exclude=权重置0（分母不变）')
    p.add_argument('--opacity_lr_multiplier', type=float, default=1.0, choices=[1.0, 0.0],
                   help='1025：G阶段opacity学习率倍率；1=G0，0=冻结opacity（每个保存点断言opacity逐位不变）')
    p.add_argument('--allow_parent_any', action='store_true', help='仅冒烟：允许非S0@35000父（如B00或S0中间点）')
    args = p.parse_args()
    out = args.output.resolve(); model = out / 'model_mlp'
    parent, source, split_path, dpath = (args.parent.resolve(), args.source.resolve(), args.split.resolve(),
                                         args.directions.resolve())
    if 'output' not in out.parts or model.exists():
        p.error('输出须在output内且model_mlp尚不存在')
    stage_steps = A0_STEPS if args.stage == 'A0' else G_STEPS
    if args.smoke:
        if 'smoke_test' not in out.parts or args.steps > 300:
            p.error('冒烟只能在smoke_test且≤300步')
    else:
        if args.steps != stage_steps or args.save_steps is not None or args.allow_parent_any:
            p.error('正式运行预算/保存点冻结，不可修改')
    if args.stage == 'G' and args.route is None:
        p.error('G阶段必须指定route')
    if args.stage == 'G' and args.s0_parent_iteration != 35000:
        p.error('--s0_parent_iteration只用于A0')
    if args.stage == 'A0' and args.route is not None:
        p.error('A0不使用route')
    if args.stage == 'A0' and (args.geometry_lr_multiplier != 1.0 or args.rotation_lr_multiplier is not None or
                               args.photo_threshold != FROZEN['loss']['threshold'] or args.seed != FROZEN['seed'] or
                               args.continue_from_g):
        p.error('A0几何冻结，不使用几何LR倍率')
    if args.stage == 'A0' and (args.photo_partial_alpha != 'keep' or args.opacity_lr_multiplier != 1.0 or
                               args.normal_weight != FROZEN['loss']['normal'] or args.composite != 'linear'):
        p.error('A0不使用1025的photo部分覆盖/opacity倍率/normal权重/合成域选项')
    clock = 0 if args.stage == 'A0' else args.parent_iteration
    phase1 = A0_PHASE1 if not args.smoke else min(A0_PHASE1, args.steps // 2)
    if args.save_steps is None:
        save_steps = (FROZEN['A0']['saves'] if args.stage == 'A0'
                      else [x - G_CLOCK_START for x in FROZEN['G']['saves_relative_total']])
    else:
        save_steps = sorted(set(args.save_steps + [args.steps]))
    split = json.loads(split_path.read_text())

    # ---- 父模型核验
    ply = parent / f'point_cloud/iteration_{args.parent_iteration}/point_cloud.ply'
    deform = parent / f'deform/iteration_{args.parent_iteration}/deform.pth'
    parent_record = {}
    if args.stage == 'A0':
        fin = parent / 'training_finished.json'
        if fin.exists():
            parent_record = json.loads(fin.read_text())
        if not args.allow_parent_any:
            assert parent_record.get('stage') == 'S0静态原生Stage1' and parent_record['iterations'] == 35000
            assert parent_record['actual_optimizer_updates'] == 34999
            assert args.parent_iteration == args.s0_parent_iteration
            saved = [r for r in parent_record['saves'] if r['iteration'] == args.parent_iteration]
            assert len(saved) == 1 and saved[0]['optimizer_updates_applied'] == args.parent_iteration - 1
            assert parent_record['geometry_heads_final_all_zero'] and parent_record['deform_state_unchanged']
            assert parent_record['static_config']['pred_color'] is False
        parent_lighting = None
    else:
        fin = parent / 'physical_finished.json'
        parent_record = json.loads(fin.read_text())
        assert parent_record['status'] == 'TRAINING_FINISHED'
        if not args.continue_from_g:
            assert parent_record['stage'] == 'A0'
            if not args.smoke:
                assert args.parent_iteration == G_CLOCK_START and parent_record['new_optimizer_updates'] == A0_STEPS
                assert parent_record['smoke'] is False
        else:
            # 续训：父须为正式G（7020次、@8100、非冒烟），且训练配置与本次逐项相同
            assert parent_record['stage'] == 'G' and parent_record['smoke'] is False
            assert parent_record['new_optimizer_updates'] == G_STEPS and args.parent_iteration == G_CLOCK_START + G_STEPS
            pp = json.loads((parent / 'physical_plan.json').read_text())
            want = dict(route=args.route, photo_partial_alpha=args.photo_partial_alpha,
                        geometry_lr_multiplier=args.geometry_lr_multiplier, rotation_lr_multiplier=args.rotation_lr_multiplier,
                        opacity_lr_multiplier=args.opacity_lr_multiplier, normal_weight=args.normal_weight,
                        composite=args.composite, photo_threshold=args.photo_threshold, seed=args.seed)
            got = dict(route=pp['route'], photo_partial_alpha=pp.get('photo_partial_alpha', 'keep'),
                       geometry_lr_multiplier=pp.get('geometry_lr_multiplier', 1.0),
                       rotation_lr_multiplier=pp.get('rotation_lr_multiplier'),
                       opacity_lr_multiplier=pp.get('opacity_lr_multiplier', 1.0), normal_weight=pp.get('normal_weight', 0.1),
                       composite=pp.get('composite', 'linear'), photo_threshold=pp.get('photo_threshold', 0.08),
                       seed=pp.get('seed', 1012))
            assert want == got, f'续训配置须与父G相同：{want} vs {got}'
            parent_saved = [r for r in json.loads((parent / 'saves.json').read_text())
                            if r['iteration'] == args.parent_iteration][0]
        parent_lighting = json.loads((parent / f'lighting/iteration_{args.parent_iteration}/lighting.json').read_text())
    state = torch.load(deform, map_location='cpu', weights_only=True)
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        for s in ('weight', 'bias'):
            assert torch.count_nonzero(state[name + '.' + s]) == 0, '父模型不是严格静态零几何头'
    assert not any(k.startswith('gaussian_color') for k in state), '父模型含时变颜色头'
    install_guard([source, split_path.parent, dpath.parent, parent, out, Path.cwd()])

    seed = args.seed
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
    loaded_hashes = param_hashes(pc)
    directions = TrainDirections(dpath)
    assert directions.frames == split['train_frames']
    lighting = MixedLighting(FROZEN['A0']['initial_direct_fraction'])
    if parent_lighting is not None:
        lighting.load_state(parent_lighting['lighting'])
    cameras, image_sha = load_cameras(source, split)
    phase = 1 if args.stage == 'A0' else 0
    gopt, lopt = build_optimizers(args.stage, pc, lighting, phase if args.stage == 'A0' else 0)
    vis = StaticVisibility(FROZEN['shadow_bias'], FROZEN[args.stage]['bvh_interval'])
    background = torch.zeros(3, device='cuda')
    pc.max_radii2D = torch.zeros(len(pc.get_xyz), device='cuda')
    pc.xyz_gradient_accum = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    pc.denom = torch.zeros((len(pc.get_xyz), 1), device='cuda')
    init_hashes = param_hashes(pc, lighting)
    preview = random.Random(seed)
    first_index = list(range(len(cameras))).pop(preview.randrange(len(cameras)))
    initialization = dict(
        stage=args.stage, parent=str(parent), parent_iteration=args.parent_iteration, parent_ply_sha256=sha(ply),
        parent_deform_sha256=sha(deform), loaded_parameter_hashes=loaded_hashes, initialized_parameter_hashes=init_hashes,
        inherited_identical=all(loaded_hashes[k] == init_hashes[k] for k in loaded_hashes),
        gaussian_adam_state_count=len(gopt.state), light_adam_state_count=len(lopt.state),
        adam='新建Adam，动量为零；不恢复父优化器', density_statistics='xyz_gradient_accum/denom/max_radii2D清零（固定拓扑不使用）',
        points=len(pc.get_xyz), expected_first_camera=cameras[first_index].image_name_train_light,
        lighting_initial=lighting.state(), lighting_inherited=parent_lighting is not None)
    assert initialization['inherited_identical'] and not gopt.state and not lopt.state
    if args.stage == 'G' and args.continue_from_g:
        initialization['continue_from_g'] = True
        initialization['parent_saved_hashes_match'] = {k: init_hashes[k] == parent_saved['hashes'][k]
                                                       for k in parent_saved['hashes']}
        initialization['parent_ply_sha_matches_saves'] = sha(ply) == parent_saved['ply_sha256']
        assert all(initialization['parent_saved_hashes_match'].values()) and initialization['parent_ply_sha_matches_saves'], \
            '续训父加载与父保存时哈希不一致'
    model.mkdir(parents=True)
    write_json(model / 'initialization_audit.json', initialization)
    weights = dict(alpha=FROZEN['loss']['alpha'], normal=args.normal_weight, distortion=FROZEN['loss']['distortion'])
    render_fn = render_mixed if args.composite == 'linear' else render_mixed_srgb
    cam0 = cameras[first_index]
    cam0.load2device()
    phys = audit_physical_forward(cam0, pc, pipe, background, directions, lighting, vis, clock)
    write_json(model / 'initial_physical_forward_audit.json', phys)
    route_audit = None
    mask_audit = None
    if args.stage == 'G':
        route_audit = audit_normal_routes(cam0, pc, pipe, background, directions, lighting, vis, clock, weights)
        write_json(model / 'initial_normal_route_audit.json', route_audit)
        if args.photo_partial_alpha == 'exclude':
            mask_audit = audit_photo_pixel_keep(cam0, pc, pipe, background, directions, lighting, vis, clock)
            write_json(model / 'initial_photo_partial_alpha_audit.json', mask_audit)
        if args.photo_threshold != FROZEN['loss']['threshold']:
            th_audit = audit_photo_threshold(cam0, pc, pipe, background, directions, lighting, vis, clock,
                                             args.photo_threshold, args.photo_partial_alpha == 'exclude')
            write_json(model / 'initial_photo_threshold_audit.json', th_audit)
            if th_audit['status'] != 'PASS':
                raise RuntimeError('photo阈值零更新审计失败，保留现场不训练')
        if args.composite != 'linear':
            comp_audit = audit_composite(cam0, pc, pipe, background, directions, lighting, vis, clock)
            write_json(model / 'initial_composite_audit.json', comp_audit)
            if comp_audit['status'] != 'PASS':
                raise RuntimeError('合成域零更新审计失败，保留现场不训练')
    cam0.load2device('cpu')
    if phys['status'] != 'PASS' or (route_audit is not None and route_audit['status'] != 'PASS') or (
            mask_audit is not None and mask_audit['status'] != 'PASS'):
        raise RuntimeError('零更新审计失败，保留现场不训练')
    root = Path.cwd()
    plan = dict(stage=args.stage, route=args.route, args={k: str(v) for k, v in vars(args).items()}, frozen=FROZEN,
                geometry_lr_multiplier=args.geometry_lr_multiplier,
                rotation_lr_multiplier=args.rotation_lr_multiplier, photo_threshold=args.photo_threshold,
                continue_from_g=args.continue_from_g, s0_parent_iteration=args.s0_parent_iteration,
                geometry_lr_first_step=(geometry_lr(1, args.geometry_lr_multiplier, args.opacity_lr_multiplier,
                                                    args.rotation_lr_multiplier)
                                        if args.stage == 'G' else None),
                photo_partial_alpha=args.photo_partial_alpha, opacity_lr_multiplier=args.opacity_lr_multiplier,
                normal_weight=args.normal_weight, composite=args.composite,
                options_1025_round3=('normal_weight 0.1=G0；composite linear=G0（render_mixed），srgb_per_gaussian='
                                     'scene/static_eldl_composite.render_mixed_srgb；geometry_lr_multiplier可取3'),
                options_1025=('默认keep/1.0时与G0逐式相同；exclude：GT Alpha∈(0,0.99)像素photo权重置0、分母仍为全部像素；'
                              'opacity_lr_multiplier=0：opacity组lr恒为0（仍计算梯度，Adam不改变参数）'),
                clock_start=clock, steps=args.steps, a0_phase1_steps=phase1 if args.stage == 'A0' else None, save_steps_stage_relative=save_steps,
                save_iterations_relative_total=[clock + s for s in save_steps],
                topology_window=[], densify=False, prune=False, opacity_reset=False,
                camera_sampling=f'独立random.Random({seed})不放回遍历117相机；G0/G1序列相同', seed=seed,
                update_semantics='每步均执行optimizer更新（含末步）；无拓扑；保存在更新后',
                lr_schedule=('A0：反照率/照明恒定；beta仅541起加入（新param group，零动量）' if args.stage == 'A0' else
                             'G：几何四组lr按0.1^((g-1)/(7020-1))对数线性衰减；albedo/照明恒定'),
                visibility='StaticVisibility bias0.03，no_grad；A0几何冻结只建一次BVH，G每步重建',
                saturation='camera.observed_saturation_mask（原分辨率任一子像素满码，AREA缩放后为下界）∧Alpha>0.5',
                inputs=dict(source_manifest=sha(source / 'no_gt_manifest.json'), transforms=sha(source / 'transforms_train.json'),
                            split=sha(split_path), directions=directions.sha256, images=image_sha),
                gt_supervision='仅TRAIN帧GT主光单位方向（固定不训练）；无GT法线/深度/材质/mesh/光强/点光/环境',
                sources={f: sha(root / f) for f in ('scripts/train_static_eldl_physical.py', 'scene/static_eldl_lighting.py', 'scene/static_eldl_composite.py',
                                                     'scene/static_dl_photometric.py', 'scene/gaussian_model.py',
                                                     'gaussian_renderer/__init__.py', 'scene/stage2_photometric.py',
                                                     'utils/camera_utils.py', 'scene/dataset_readers.py', 'utils/rgba16.py')},
                rasterizer=backend.provenance, smoke=args.smoke)
    write_json(model / 'physical_plan.json', plan)
    cfg = dict(sh_degree=3, source_path=str(source), model_path=str(model), images='images', train_light_folder='images',
               test_light_folder='images', resolution=2, white_background=False, data_device='cpu', eval=True,
               load2gpu_on_the_fly=True, is_blender=True, deform_type='mlp', hyper_dim=1, pred_color=False,
               no_binary_separation=False, load_test_set_only=False, start_frame=0, end_frame=-1)
    (model / 'cfg_args').write_text(str(Namespace(**cfg)))
    shutil.copyfile(ply, model / 'input.ply')
    start = {n: getattr(pc, n).detach().clone() for n in ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_dc')}

    saves = []

    def save(step):
        it = clock + step
        pc.save_ply(str(model / f'point_cloud/iteration_{it}/point_cloud.ply'))
        dst = model / f'deform/iteration_{it}/deform.pth'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(deform, dst)
        write_json(model / f'lighting/iteration_{it}/lighting.json', dict(
            iteration_relative_total=it, stage=args.stage, route=args.route, stage_step=step,
            new_optimizer_updates_this_stage=step, lineage_new_updates_since_S0=it,
            directions_sha256=directions.sha256, lighting=lighting.state(), model='static_eldl_mixed_v1',
            **({} if args.composite == 'linear' else dict(composite=args.composite))))
        opath = model / f'optimizer/iteration_{it}.pth'
        opath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(gaussian=gopt.state_dict(), light=lopt.state_dict(), camera_rng=camera_rng.getstate(),
                        remaining=list(stack), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        iteration=it, stage_step=step), opath)
        h = param_hashes(pc, lighting)
        rec = dict(iteration=it, stage_step=step, points=len(pc.get_xyz), hashes=h,
                   ply_sha256=sha(model / f'point_cloud/iteration_{it}/point_cloud.ply'))
        if args.stage == 'A0':
            rec['geometry_unchanged_vs_parent'] = all(h[k] == loaded_hashes[k] for k in
                                                      ('_xyz', '_rotation', '_scaling', '_opacity', '_albedo_rest'))
            assert rec['geometry_unchanged_vs_parent'], 'A0几何发生变化'
        if args.stage == 'G' and args.opacity_lr_multiplier == 0.0:
            rec['opacity_unchanged_vs_parent'] = h['_opacity'] == loaded_hashes['_opacity']
            assert rec['opacity_unchanged_vs_parent'], 'opacity冻结但发生变化'
        saves.append(rec)
        write_json(model / 'saves.json', saves)

    stack = []
    if 0 in save_steps:
        save(0)
    n0 = len(pc.get_xyz)
    metrics = (model / 'physical_metrics.jsonl').open('a')
    for step in range(1, args.steps + 1):
        if args.stage == 'A0' and step == phase1 + 1:
            lighting.beta.requires_grad_(True)
            lopt.add_param_group(dict(params=[lighting.beta], lr=FROZEN['A0']['lighting_lr'], name='direct_fraction'))
            write_json(model / 'phase2_switch.json', dict(step=step, beta_before=float(lighting.beta),
                                                          note='直射比例加入优化（新param group，零动量）'))
        if args.stage == 'G':
            for g in gopt.param_groups:
                if g['name'] in ('xyz', 'rotation', 'scaling', 'opacity'):
                    g['lr'] = geometry_lr(step, args.geometry_lr_multiplier, args.opacity_lr_multiplier,
                                          args.rotation_lr_multiplier)[g['name']]
        if not stack:
            stack = list(range(len(cameras)))
        cam = cameras[stack.pop(camera_rng.randrange(len(stack)))]
        if step == 1:
            assert cam.image_name_train_light == initialization['expected_first_camera']
        cam.load2device()
        log = step == 1 or step % 100 == 0 or step in save_steps
        pk = render_fn(cam, pc, pipe, background, directions, lighting, vis, clock + step, components=log)
        target, alpha = cam.original_image_train_light.cuda(), cam.gt_alpha_mask.cuda()
        sat = cam.observed_saturation_mask.cuda()
        keep = partial_alpha_keep_mask(alpha) if args.photo_partial_alpha == 'exclude' else None
        photo, pdiag = censored_loss(pk['render'], target, alpha, sat, FROZEN['loss']['shadow_weight'],
                                     FROZEN['loss']['highlight_weight'], args.photo_threshold, pixel_keep=keep)
        if args.stage == 'G':
            geo, _ = geometry_losses(cam, pc, pipe, background, alpha, pk, args.route)
        else:
            with torch.no_grad():
                geo, _ = geometry_losses(cam, pc, pipe, background, alpha, pk, 'full')
        loss = photo + weights['alpha'] * geo['alpha'] + weights['normal'] * geo['normal'] + weights['distortion'] * geo['distortion']
        if not torch.isfinite(loss):
            raise FloatingPointError('loss非有限')
        loss.backward()
        if step == 1:
            grads = {}
            for name, t in [('xyz', pc._xyz), ('rotation', pc._rotation), ('scaling', pc._scaling), ('opacity', pc._opacity),
                            ('albedo_dc', pc._albedo_dc), ('env_log_dc', lighting.log_dc), ('env_higher', lighting.higher),
                            ('direct_fraction', lighting.beta)]:
                grads[name] = None if t.grad is None else dict(norm=float(t.grad.norm()), finite=bool(torch.isfinite(t.grad).all()))
            write_json(model / 'first_step_gradients.json', dict(camera=cam.image_name_train_light, gradients=grads))
        with torch.no_grad():
            lrs = {g['name']: g['lr'] for g in gopt.param_groups + lopt.param_groups}
            gopt.step(); lopt.step()
            gopt.zero_grad(set_to_none=True); lopt.zero_grad(set_to_none=True)
            pc._albedo_dc.clamp_(float(RGB2SH(.031)), float(RGB2SH(.969)))
            lighting.log_dc.clamp_(-5, 5); lighting.beta.clamp_(-6, 6)
            assert len(pc.get_xyz) == n0, '固定拓扑点数变化'
            if log:
                disp = dict(xyz_mean_abs_displacement=float((pc._xyz - start['_xyz']).norm(dim=-1).mean()),
                            xyz_max_displacement=float((pc._xyz - start['_xyz']).norm(dim=-1).max()),
                            rotation_mean_abs_change=float((pc._rotation - start['_rotation']).abs().mean()),
                            scaling_mean_abs_change=float((pc._scaling - start['_scaling']).abs().mean()),
                            opacity_mean_change=float((pc.get_opacity - torch.sigmoid(start['_opacity'])).mean()),
                            opacity_below_min_fraction=float((pc.get_opacity < .01).float().mean()),
                            albedo_mean_abs_change=float((pc._albedo_dc - start['_albedo_dc']).abs().mean()))
                ra, fgm = pk['rend_alpha'].detach(), alpha >= .5
                part = (alpha > 0) & (alpha < .99)
                disp.update(train_view_alpha_missing_fraction=float((fgm & (ra < .5)).sum() / fgm.sum().clamp_min(1)),
                            train_view_partial_pred_alpha_mean=float(ra[part].mean()) if part.any() else None,
                            train_view_partial_gt_alpha_mean=float(alpha[part].mean()) if part.any() else None)
                rec = dict(step=step, iteration=clock + step, camera=cam.image_name_train_light, route=args.route,
                           loss=float(loss), photo=float(photo), **{k: float(v) for k, v in geo.items()}, **pdiag,
                           **component_stats(pk, alpha), mean_visibility=float(pk['mean_visibility']),
                           direct_fraction=float(lighting.direct_fraction()), lrs_used=lrs, points=len(pc.get_xyz), **disp)
                metrics.write(json.dumps(rec, ensure_ascii=False) + '\n'); metrics.flush()
                print(json.dumps(rec, ensure_ascii=False), flush=True)
            if step in save_steps:
                save(step)
        cam.load2device('cpu')
    metrics.close()
    write_json(model / 'physical_finished.json', dict(
        status='TRAINING_FINISHED', stage=args.stage, route=args.route, smoke=args.smoke, clock_start=clock,
        final_iteration_relative_total=clock + args.steps, new_optimizer_updates=args.steps,
        lineage_updates=(dict(S0=34999, A0=A0_STEPS, G_parent=G_STEPS, G_continue=args.steps) if args.continue_from_g else
                         dict(S0=args.s0_parent_iteration - 1 if args.stage == 'A0' else 34999,
                              A0=A0_STEPS if args.stage == 'G' else args.steps,
                              G=args.steps if args.stage == 'G' else 0)),
        s0_parent_iteration=args.s0_parent_iteration if args.stage == 'A0' else None,
        continue_from_g=args.continue_from_g,
        points=len(pc.get_xyz), saves=saves, final_lighting=lighting.state(),
        photo_partial_alpha=args.photo_partial_alpha, opacity_lr_multiplier=args.opacity_lr_multiplier,
        geometry_lr_multiplier=args.geometry_lr_multiplier, rotation_lr_multiplier=args.rotation_lr_multiplier,
        photo_threshold=args.photo_threshold,
        normal_weight=args.normal_weight, composite=args.composite,
        evaluation='尚未评测；DEV/TEST由独立评测Agent使用对应帧GT方向'))


if __name__ == '__main__':
    main()
