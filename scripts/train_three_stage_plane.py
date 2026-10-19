"""按计划1019-40恢复八组Adam/RNG，增加冻结TRAIN可见平台法线先验。"""
import argparse
from copy import copy
import shutil
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scene.three_stage_native_pbr import load_three_stage_model
from scene.three_stage_pipeline import sha,write
from scene.three_stage_lighting import photometric_objective
from scripts.train_three_stage import install_guard,load_cameras,preview
from scripts.train_three_stage_refine import GEOMETRY,GEOMETRY_LR,tensor_sha,fingerprint,rng_state,restore_rng,expected_geometry_loss

def erode_fixed_mask(mask):
    """图外明确补0；固定5像素内缩，防止画面边界被max_pool忽略。"""
    value = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    return (-F.max_pool2d(-F.pad(value, (2, 2, 2, 2), value=0.), 5, stride=1)[0, 0]).bool().numpy()


def prepare_plane_assets(choice, model_dir):
    """仅在读取守卫和父RNG恢复前复制、核验冻结TRAIN研究资产。"""
    source = Path(choice['research_source'])
    reference = source/'frozen_reference.npz'
    assert sha(reference) == choice['research_reference_sha256']
    folder = model_dir/'training_plane'
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(reference, folder/'reference.npz')
    rows = {}
    assert len(choice['research_mask_sha256']) == 135
    for filename, digest in choice['research_mask_sha256'].items():
        path = Path(filename)
        frame = int(path.stem.split('_')[-1])
        assert 1 <= frame <= 150 and frame % 10 and sha(path) == digest
        assert path.parent == source/'frozen_masks'
        with np.load(path, allow_pickle=False) as data:
            raw = data['visible_band']
            assert raw.dtype == np.bool_ and raw.shape == (400, 400)
            mask = erode_fixed_mask(raw)
            assert mask.any() and np.all(~mask | raw)
        copied = folder/f'raw_{frame:04d}.npz'
        shutil.copyfile(path, copied)
        fixed = folder/f'mask_{frame:04d}.npz'
        np.savez_compressed(fixed, mask=mask)
        rows[f'image_{frame:04d}'] = dict(raw=copied.relative_to(model_dir).as_posix(),
            raw_sha256=sha(copied), source=str(path), mask=fixed.relative_to(model_dir).as_posix(),
            mask_sha256=sha(fixed), raw_pixels=int(raw.sum()), fixed_pixels=int(mask.sum()))
    assert set(rows) == {f'image_{n:04d}' for n in range(1, 151) if n % 10}
    manifest = folder/'manifest.json'
    write(manifest, dict(reference='training_plane/reference.npz', reference_sha256=sha(reference),
        frames=rows, mask_key='visible_band', erosion_kernel=5, outside_value=0,
        gt_read=False, test_read=False, training_only=True))
    return manifest.relative_to(model_dir).as_posix(), sha(manifest)


def load_plane_assets(model_dir, config, device='cuda'):
    """推理factory不调用本函数；只由训练与独立门禁读取loss资产。"""
    def asset(relative):
        path = (model_dir/relative).resolve()
        assert path.is_relative_to(model_dir.resolve())
        return path
    manifest_path = asset(config['plane_asset_manifest'])
    assert sha(manifest_path) == config['plane_asset_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    ref = asset(manifest['reference'])
    assert sha(ref) == manifest['reference_sha256']
    with np.load(ref, allow_pickle=False) as data:
        normal, center = data['normal'].copy(), data['center'].copy()
    assert normal.shape == center.shape == (3,) and np.isfinite(normal).all() and np.isfinite(center).all()
    assert abs(float(np.linalg.norm(normal))-1) < 1e-6
    masks = {}
    for name, row in manifest['frames'].items():
        path = asset(row['mask'])
        assert sha(path) == row['mask_sha256']
        with np.load(path, allow_pickle=False) as data:
            mask = data['mask'].copy()
        assert mask.dtype == np.bool_ and mask.shape == (400, 400) and int(mask.sum()) == row['fixed_pixels'] > 0
        masks[name] = torch.as_tensor(mask, device=device)[None]
    assert set(masks) == {f'image_{n:04d}' for n in range(1, 151) if n % 10}
    return dict(normal=F.normalize(torch.as_tensor(normal, device=device), dim=0),
                center=torch.as_tensor(center, device=device), masks=masks)


def plane_geometry_loss(model, camera, assets, protect, details=False):
    """固定分母；无当前支持的冻结mask像素计单位损失，不可通过缺失规避。"""
    from gaussian_renderer import render
    pipe = copy(model.pipe)
    pipe.depth_ratio = 0.
    pc = model.effective(camera)
    geometry = render(camera, pc, pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
        override_color=torch.zeros_like(pc.get_xyz), detach_opacity=protect)
    raw = geometry['rend_normal']
    finite = torch.isfinite(raw).all(0, keepdim=True)
    safe = torch.where(finite, raw, torch.zeros_like(raw))
    valid = finite & (safe.norm(dim=0, keepdim=True) > 1e-8) & (geometry['rend_alpha'].detach() > .5)
    target = assets['normal']
    if torch.dot(target, camera.camera_center-assets['center']) < 0:
        target = -target
    cosine = (F.normalize(safe, dim=0)*target[:, None, None]).sum(0, keepdim=True).clamp(-1, 1)
    pixel = torch.where(valid, 1-cosine, torch.ones_like(cosine))
    mask = assets['masks'][camera.image_name_train_light]
    count = mask.sum()
    loss = pixel[mask].sum()/count
    info = dict(fixed_pixels=int(count), valid_pixels=int((mask & valid).sum()),
        missing_pixels=int((mask & ~valid).sum()), missing_fraction=float((mask & ~valid).sum()/count))
    if details:
        return loss, info, dict(mask=mask, valid=valid, target=target, normal=F.normalize(safe, dim=0), pixel_loss=pixel)
    return loss, info


def checkpoint_resume(model, folder, step, optimizer, parent_cumulative):
    """继续训练仍存自足格式1，明确父累计数，禁止套用Stage2父公式。"""
    ply = folder/f'point_cloud/iteration_{step}/point_cloud.ply'
    if ply.exists():
        raise FileExistsError('禁止覆盖已有checkpoint。')
    model.pc.save_ply(str(ply))
    model.deform.save_weights(str(folder), step)
    directory = folder/f'three_stage/iteration_{step}'
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(format_version=1, completed_updates=step, stage=3, config=model.config,
                 stats=model.stats, binding=model.binding, lighting=model.lights.state_dict(),
                 env=model.env.state_dict(), env_initial=model.env_initial,
                 residual_raw=model.residual_raw.detach(), optimizer=optimizer.state_dict(),
                 parent_stage=3, parent_cumulative_actual_updates=parent_cumulative,
                 cumulative_actual_updates=parent_cumulative+step, **rng_state())
    torch.save(state, directory/'state.pt')
    write(directory/'receipt.json', dict(stage=3, completed_updates=step,
        cumulative_actual_updates=parent_cumulative+step, parent_stage=3,
        parent_cumulative_actual_updates=parent_cumulative,
        parent=str(model.parent), parent_iteration=model.parent_iteration, config=model.config,
        geometry_sha256={k: tensor_sha(getattr(model.pc, k)) for k in GEOMETRY.values()},
        checkpoint_sha256=sha(directory/'state.pt'), ply_sha256=sha(ply),
        deform_sha256=sha(folder/f'deform/iteration_{step}/deform.pth'),
        direction_max_deviation_deg=float(torch.rad2deg(torch.acos(
            (model.lights.directions()*model.lights.initial).sum(-1).clamp(-1, 1))).max()),
        gt_read=False, test_read=False, semantics='Stage3真实末端继续；父8组Adam及RNG完整恢复，末端常量lr继续，无moments重置'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent_iteration', type=int, choices=[4050], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get('THREE_STAGE_REPO_ROOT', code_root)).resolve()
    output, parent = args.output.resolve(), args.parent.resolve()
    assert output.is_relative_to(root/'output')
    assert args.smoke == ('smoke_test' in output.parts)
    if output.exists() and {p.name for p in output.iterdir()}-{'launch.json', 'train.log', 'job_state.json'}:
        raise FileExistsError('已有结果保留；重试使用独立目录。')
    output.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(args.preregistration.read_text())
    assert prereg['plan_sha256'] == sha(prereg['plan'])
    control = args.preregistration.resolve().parent
    choice_path = Path(prereg.get('parent_selection', control/'pbr_geo_refine_parent_train_only_v1.json'))
    assert sha(choice_path) == prereg['parent_selection_sha256']
    assert prereg['plan_sha256'] == '8bfdd69575986b22436e8d466ad4958cade9a3f359e0c073ba1ce2406aee917e'
    assert prereg['A46_offline_results_already_observed'] is True
    assert prereg['settings_derived_from_train_only'] is True
    assert prereg['gt_used_for_parameters_or_budget'] is False
    assert parent == root/'output/1019-53-LH-static-jumpingjacks-ELDL-V3-A45原生PBR末端联合衰减日程/model_mlp'
    parent_record = json.loads((parent/'three_stage_training.json').read_text())
    parent_receipt = json.loads((parent/f'three_stage/iteration_{args.parent_iteration}/receipt.json').read_text())
    assert parent_record['stage'] == 3 and not parent_record['smoke']
    assert parent_receipt['checkpoint_sha256'] == '5f026947a0c5643337a4df3488a2f3a303e933024fc6cf06f76d36423f87a85e'
    parent_cumulative = parent_receipt['cumulative_actual_updates']
    assert parent_cumulative == 55249 and args.parent_iteration % 135 == 0
    binding = parent_record['binding']
    for path, digest in binding['input_sha256'].items():
        assert sha(path) == digest, path
    choice = json.loads(choice_path.read_text())
    assert choice['parent'] == str(parent) and choice['plane_weight'] == .5 and choice['mask_erosion_kernel'] == 5
    model_dir = output/'model_mlp'
    model_dir.mkdir(exist_ok=False)
    manifest, manifest_sha = prepare_plane_assets(choice, model_dir)
    install_guard(binding['source'], output, parent, binding)
    torch.set_num_threads(4)
    model = load_three_stage_model(parent, args.parent_iteration)
    parent_state = torch.load(parent/f'three_stage/iteration_{args.parent_iteration}/state.pt', weights_only=False)
    assert model.config['normal_weight'] == .5 and model.config['lambda_residual'] == .1
    assert model.pipe.depth_ratio == 1. and model.config['pbr_samples'] == 256 and model.config['pbr_pixels'] == 2048
    config = dict(model.config)
    config.update(stage3_continuation=True, geometry_continuation='plane_floor_resume',
                  iterations=20 if args.smoke else 8100, schedule_horizon=20 if args.smoke else 8100,
                  refine_normal_weight=.5, refine_normal_depth_ratio=0.,
                  refine_protect_end=200, parent_geometry_updates=4050,
                  refine_geometry_lr=dict(GEOMETRY_LR),
                  refine_appearance_lr='inherit_parent_actual_constant', parent_cumulative_actual_updates=parent_cumulative)
    config['refine_geometry_lr_schedule'] = 'inherited_floor_constant'
    config['stage3_full_state_resume'] = True
    config.update(plane_weight=.5, plane_erosion_kernel=5, plane_asset_manifest=manifest,
                  plane_asset_manifest_sha256=manifest_sha, plane_training_only=True)
    plane_assets = load_plane_assets(model_dir, config)
    model.config, model.parent, model.parent_iteration = config, parent, args.parent_iteration
    model.anchor_geometry = {key: getattr(model.pc, key).detach().clone() for key in GEOMETRY.values()}
    model.lights.requires_grad_(False)
    model.deform.deform.requires_grad_(False)
    appearance = dict(albedo=model.pc._albedo_dc, roughness=model.pc._roughness,
                      environment=model.env.base, residual_amplitude=model.residual_raw)
    parameters = dict(appearance, **{name: getattr(model.pc,key) for name,key in GEOMETRY.items()})
    parent_groups = parent_state['optimizer']['param_groups']
    assert [g['name'] for g in parent_groups] == list(parameters)
    assert parent_state['completed_updates'] == 4050 and len(parent_state['optimizer']['state']) == 8
    for group in parent_groups:
        if group['name'] in GEOMETRY_LR:
            assert group['lr'] == GEOMETRY_LR[group['name']]
    groups = [dict(params=[parameter], lr=group['lr'], initial_lr=group['initial_lr'], name=name)
              for (name,parameter),group in zip(parameters.items(),parent_groups)]
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    optimizer.load_state_dict(parent_state['optimizer'])
    adam_parent = fingerprint(parent_state['optimizer'])
    assert fingerprint(optimizer.state_dict()) == adam_parent
    active = {id(p) for g in optimizer.param_groups for p in g['params']}
    for value in vars(model.pc).values():
        if isinstance(value, torch.nn.Parameter):
            value.requires_grad_(id(value) in active)
    for p in appearance.values():
        p.requires_grad_(True)
    cameras = load_cameras(binding['source'])
    assert len(cameras) == 135
    saves = [0, 4, 8, 20] if args.smoke else [0, 2700, 5400, 8100]
    import diff_surfel_rasterization as backend
    assert backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer'
    source_names = ['scripts/train_three_stage_plane.py', 'scripts/train_three_stage_refine.py', 'scripts/train_three_stage.py',
                    'scene/three_stage_pipeline.py', 'scene/three_stage_lighting.py',
                    'scene/three_stage_native_pbr.py', 'scene/three_stage_normal_origin.py',
                    'scene/three_stage_direct_base.py', 'gaussian_renderer/render_ir.py']
    marker = dict(parent_record, smoke=args.smoke, config=config, parent=str(parent),
                  parent_iteration=args.parent_iteration, parent_stage=3, parent_cumulative_actual_updates=parent_cumulative,
                  actual_entrypoint='scripts/train_three_stage_plane.py', stage3_continuation=True,
                  preregistration_sha256=sha(args.preregistration), training_plan_sha256=prereg['plan_sha256'],
                  parent_checkpoint_sha256=parent_receipt['checkpoint_sha256'],
                  sources={str(code_root/n): sha(code_root/n) for n in source_names}, rasterizer_provenance=backend.provenance)
    write(model_dir/'three_stage_training.json', marker)
    (model_dir/'cfg_args').write_text((parent/'cfg_args').read_text())
    write(output/'plan.json', dict(argv=sys.argv, config=config, saves=saves, parent=str(parent),
        parent_iteration=args.parent_iteration, parent_cumulative_actual_updates=parent_cumulative,
        source=binding['source'], model=str(model_dir), stage=3, command=shlex.join(sys.argv)))
    # 所有构造、读相机及初始化核验结束后才恢复父随机状态；父相机栈在完整周期边界为空。
    restore_rng(parent_state)
    assert fingerprint(rng_state()) == fingerprint({k: parent_state[k] for k in rng_state()})
    write(output/'initial_continuation_audit.json', dict(status='PASS_PARENT_STATE_ADAM_AND_RNG_RESTORED',
        parent_checkpoint_sha256=parent_receipt['checkpoint_sha256'], parent_optimizer=adam_parent,
        restored_rng=fingerprint(rng_state()), geometry_sha256={k: tensor_sha(getattr(model.pc, k)) for k in GEOMETRY.values()},
        lighting=fingerprint(model.lights.state_dict()), env=fingerprint(model.env.state_dict()),
        residual_raw=fingerprint(model.residual_raw), all_eight_optimizer_states_preserved=True,
        all_eight_actual_lrs={g['name']: g['lr'] for g in optimizer.param_groups},
        camera_stack_empty=True, stage=3, gt_read=False, test_read=False))
    checkpoint_resume(model, model_dir, 0, optimizer, parent_cumulative)
    stack, order, start = [], [], time.time()
    for step in range(1, config['iterations']+1):
        if not stack:
            stack = cameras.copy()
        camera = stack.pop(random.randrange(len(stack)))
        camera.load2device()
        order.append(camera.image_name_train_light)
        rates = dict(GEOMETRY_LR)
        for group in optimizer.param_groups:
            if group['name'] in GEOMETRY:
                group['lr'] = rates[group['name']]
        package = model.render(camera, step, training=True)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        photo, terms = photometric_objective(package, target, alpha, camera.observed_saturation_mask.cuda(), model.stats,
                                             support=package['loss_support'])
        alpha_loss = F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6, 1-1e-6), alpha)
        photo = photo+.1*alpha_loss
        protect = config['parent_geometry_updates']+step <= config['refine_protect_end']
        normal = expected_geometry_loss(model, camera, package, protect)
        plane, plane_info = plane_geometry_loss(model, camera, plane_assets, protect)
        env_anchor = (model.env.base-math.log(model.env_initial)).square().mean()
        amp_anchor = (.25*model.residual_raw.sigmoid()).square()
        loss = photo+.5*normal+.5*plane+.01*env_anchor+.01*amp_anchor
        assert bool(torch.isfinite(loss)), '非有限loss；保留失败产物。'
        audit = args.smoke or step in saves or step % 100 == 0
        geometry_gradients = {}
        if audit:
            params = [getattr(model.pc, k) for k in GEOMETRY.values()]
            normal_grads = torch.autograd.grad(.5*normal, params, retain_graph=True, allow_unused=True)
            plane_grads = torch.autograd.grad(.5*plane, params, retain_graph=True, allow_unused=True)
            photo_grads = torch.autograd.grad(photo, params, retain_graph=True, allow_unused=False)
            for name, ng, pg, plg in zip(GEOMETRY, normal_grads, photo_grads, plane_grads):
                assert torch.isfinite(pg).all() and (ng is None or torch.isfinite(ng).all())
                assert plg is None or torch.isfinite(plg).all()
                geometry_gradients[name] = dict(plane_gradient_connected=plg is not None,
                    plane_gradient_norm=0. if plg is None else float(plg.norm()), normal_gradient_connected=ng is not None,
                    normal_gradient_norm=0. if ng is None else float(ng.norm()), photo_alpha_gradient_norm=float(pg.norm()))
            assert not protect or geometry_gradients['opacity']['normal_gradient_norm'] == 0.
            assert protect or geometry_gradients['opacity']['normal_gradient_connected']
        loss.backward()
        gradients = {}
        if audit:
            before = {g['name']: g['params'][0].detach().clone() for g in optimizer.param_groups}
            for group in optimizer.param_groups:
                param = group['params'][0]
                assert param.grad is not None and torch.isfinite(param.grad).all(), group['name']
                gradients[group['name']] = dict(lr=group['lr'], gradient_norm=float(param.grad.norm()), gradient_finite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            from utils.sh_utils import RGB2SH, SH2RGB
            model.pc._albedo_dc.copy_(RGB2SH(SH2RGB(model.pc._albedo_dc).clamp(.030001, .969999)))
            model.env.base.clamp_(math.log(model.env_initial*.5), math.log(model.env_initial*2))
        if audit:
            for group in optimizer.param_groups:
                gradients[group['name']]['actual_max_change'] = float((group['params'][0]-before[group['name']]).abs().max())
            row = dict(step=step, camera=camera.image_name_train_light, loss=float(loss), alpha_loss=float(alpha_loss),
                plane=float(plane), plane_weight=.5, plane_support=plane_info,
                photo_alpha=float(photo), normal=float(normal), normal_weight=.5, normal_depth_ratio=0.,
                pbr_depth_ratio=model.pipe.depth_ratio, normal_opacity_protected=protect,
                env_anchor=float(env_anchor), amplitude_anchor=float(amp_anchor),
                geometry_branch_gradients=geometry_gradients, groups=gradients, seconds=time.time()-start,
                terms={k: float(v) for k, v in terms.items()})
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del before
        if step in saves:
            checkpoint_resume(model, model_dir, step, optimizer, parent_cumulative)
        camera.load2device('cpu')
        del package, loss, photo, normal, plane
    write(output/'camera_order.json', dict(frames=order, actual_updates=step, resumed_parent_rng=True))
    checks = []
    with torch.no_grad():
        for index in (0, len(cameras)//2, -1):
            camera = cameras[index]
            camera.load2device()
            torch.manual_seed(1000+index)
            torch.cuda.manual_seed_all(1000+index)
            package = model.render(camera, step, components=True)
            assert torch.isfinite(package['render']).all() and package['component_add_max_error'] < 2e-5
            preview(output/f'previews/{camera.image_name_train_light}', package)
            checks.append(dict(frame=camera.image_name_train_light, add_max_error=float(package['component_add_max_error']),
                               render=package['render'].cpu(), alpha=package['rend_alpha'].cpu()))
            camera.load2device('cpu')
        model.reload_endpoint(model_dir, step)
        for index, check in zip((0, len(cameras)//2, -1), checks):
            camera = cameras[index]
            camera.load2device()
            torch.manual_seed(1000+index)
            torch.cuda.manual_seed_all(1000+index)
            package = model.render(camera, step, components=True)
            check['reload_max_abs'] = float((check.pop('render')-package['render'].cpu()).abs().max())
            check['reload_alpha_max_abs'] = float((check.pop('alpha')-package['rend_alpha'].cpu()).abs().max())
            assert check['reload_max_abs'] < 2e-5 and check['reload_alpha_max_abs'] < 2e-5
            camera.load2device('cpu')
    drift = {k: float((getattr(model.pc, k)-v).abs().max()) for k, v in model.anchor_geometry.items()}
    assert all(v > 0 for v in drift.values())
    write(output/'engineering_receipt.json', dict(status='PASS_CONTINUATION_RELOAD_AND_LINEAR_COMPONENTS', checks=checks,
        smoke=args.smoke, actual_updates=step, cumulative_actual_updates=parent_cumulative+step,
        geometry_frozen=False, geometry_max_drift=drift,
        gt_read=False, test_read=False, inference_target_read=False, accuracy_claim=False))
    write(output/'state.json', dict(status='TRAINING_FINISHED_PENDING_VISUAL_AND_ACCURACY_REVIEW', exit_code=0, actual_updates=step))
    (output/'README.md').write_text('# 原生PBR可见平台法线先验继续\n\n'
        f'Stage3继续，source `{binding["source"]}`，model `{model_dir}`，父A45@4050累计55249；新增{step}，累计{parent_cumulative+step}。\n\n'
        f'完整外部训练/内置三TRAIN渲染命令见launch.json；内部参数`{shlex.join(sys.argv)}`。日志train.log、指标metrics.jsonl、渲染previews。\n\n'
        '四类与物理分量实际目检、完整几何离线评价、视频尚待执行；工程重载和加和PASS，最终精度待评。失败产物全部保留。\n')

if __name__ == '__main__':
    main()
