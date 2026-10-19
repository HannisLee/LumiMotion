"""保留原生PBR与父Adam状态的Stage3末端继续及预登记几何日程。"""
import argparse
import copy
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
from scene.three_stage_pipeline import sha, write
from scene.three_stage_lighting import photometric_objective
from scripts.train_three_stage import install_guard, load_cameras, preview


GEOMETRY = dict(xyz='_xyz', rotation='_rotation', scale='_scaling', opacity='_opacity')
GEOMETRY_LR = dict(xyz=1e-7, rotation=5e-6, scale=1e-6, opacity=1e-4)
JOINT_GEOMETRY_LR = dict(xyz=1e-6, rotation=5e-5, scale=1e-5, opacity=1e-3)


def geometry_lrs(mode, step, horizon):
    """保留原两种常量模式；新增既有联合初值到同一末端的线性日程。"""
    assert horizon > 0 and 0 <= step <= horizon
    if mode == 'frozen':
        return {name: 0. for name in GEOMETRY_LR}
    if mode == 'tiny' or (mode == 'joint_decay' and step == horizon):
        # 终点显式使用登记floor，避免1-.9的浮点舍入改变末端记录。
        return dict(GEOMETRY_LR)
    if mode == 'joint_decay':
        return {name: lr*(1-.9*step/horizon) for name, lr in JOINT_GEOMETRY_LR.items()}
    raise ValueError('未知预登记几何日程：'+mode)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fingerprint(value):
    """按值核验Adam及RNG，不以torch容器字节代替张量内容。"""
    if torch.is_tensor(value):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=tensor_sha(value))
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def rng_state():
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state())


def restore_rng(state):
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state(state['cuda_rng'].cpu())


def expected_geometry_loss(model, camera, photo, protect):
    """Expected副本独立于原生PBR的各半混合深度；仅normal支路保护opacity。"""
    from gaussian_renderer import render
    pipe = copy.copy(model.pipe)
    pipe.depth_ratio = 0.
    pc = model.effective(camera)
    package = render(camera, pc, pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
                     override_color=torch.zeros_like(pc.get_xyz), detach_opacity=protect)
    normal = F.normalize(package['rend_normal'], dim=0)
    expected = F.normalize(package['surf_normal'], dim=0)
    alpha = camera.gt_alpha_mask.cuda()
    interior = -F.max_pool2d(-(alpha > .99).float()[None], 5, stride=1, padding=2)[0]
    mask = interior*(photo['rend_alpha'].detach() > .5)
    return ((1-(normal*expected).sum(0, keepdim=True))*mask).sum()/mask.sum().clamp_min(1)


def checkpoint(model, folder, step, optimizer, parent_cumulative):
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
        gt_read=False, test_read=False, semantics='Stage3真实末端继续；父4组Adam及RNG恢复，新增几何空moments'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent_iteration', type=int, choices=[4050], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', choices=['frozen', 'tiny', 'joint_decay'], required=True)
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
    if args.geometry == 'joint_decay':
        assert prereg['plan_sha256'] == '0815e6ee5e66959b7fc90c7d850c29351b89721ac9328ae34175f693c04071a4'
        assert prereg['new_A43_A44_test_values_or_images_read_before_registration'] is False
        assert prereg['gt_used_for_parameters_or_budget'] is False
    # 计划明确固定A42完整终点；不从离线评价推断父路径或参数。
    assert parent == root/'output/1019-49-LH-static-jumpingjacks-ELDL-V3-A42追加预算父主要DL小残差/model_mlp'
    parent_record = json.loads((parent/'three_stage_training.json').read_text())
    parent_receipt = json.loads((parent/f'three_stage/iteration_{args.parent_iteration}/receipt.json').read_text())
    assert parent_record['stage'] == 3 and not parent_record['smoke']
    assert parent_receipt['checkpoint_sha256'] == '280c3904ac6416583eac5618e9dad6abc9110924d54f51e15bb29e32713b87f1'
    parent_cumulative = parent_receipt['cumulative_actual_updates']
    assert parent_cumulative == 51199 and args.parent_iteration % 135 == 0
    binding = parent_record['binding']
    for path, digest in binding['input_sha256'].items():
        assert sha(path) == digest, path
    install_guard(binding['source'], output, parent, binding)
    torch.set_num_threads(4)
    model = load_three_stage_model(parent, args.parent_iteration)
    parent_state = torch.load(parent/f'three_stage/iteration_{args.parent_iteration}/state.pt', weights_only=False)
    assert model.config['normal_weight'] == .5 and model.config['lambda_residual'] == .1
    assert model.pipe.depth_ratio == 1. and model.config['pbr_samples'] == 256 and model.config['pbr_pixels'] == 2048
    config = dict(model.config)
    config.update(stage3_continuation=True, geometry_continuation=args.geometry,
                  iterations=20 if args.smoke else 4050, schedule_horizon=20 if args.smoke else 4050,
                  refine_normal_weight=.5, refine_normal_depth_ratio=0.,
                  refine_protect_end=4 if args.smoke else 200,
                  refine_geometry_lr=geometry_lrs(args.geometry, 0, 20 if args.smoke else 4050),
                  refine_appearance_lr='inherit_parent_actual_constant', parent_cumulative_actual_updates=parent_cumulative)
    if args.geometry == 'joint_decay':
        config['refine_geometry_lr_schedule'] = 'stage2_joint_linear_to_floor'
    model.config, model.parent, model.parent_iteration = config, parent, args.parent_iteration
    model.anchor_geometry = {key: getattr(model.pc, key).detach().clone() for key in GEOMETRY.values()}
    model.lights.requires_grad_(False)
    model.deform.deform.requires_grad_(False)
    appearance = dict(albedo=model.pc._albedo_dc, roughness=model.pc._roughness,
                      environment=model.env.base, residual_amplitude=model.residual_raw)
    parent_groups = parent_state['optimizer']['param_groups']
    assert [g['name'] for g in parent_groups] == list(appearance)
    groups = [dict(params=[p], lr=group['lr'], initial_lr=group['initial_lr'], name=name)
              for (name, p), group in zip(appearance.items(), parent_groups)]
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    optimizer.load_state_dict(parent_state['optimizer'])
    adam_parent = fingerprint(parent_state['optimizer'])
    assert fingerprint(optimizer.state_dict()) == adam_parent
    for name, key in GEOMETRY.items():
        optimizer.add_param_group(dict(params=[getattr(model.pc, key)], name=name,
                                       lr=config['refine_geometry_lr'][name], initial_lr=config['refine_geometry_lr'][name]))
    active = {id(p) for g in optimizer.param_groups for p in g['params']}
    for value in vars(model.pc).values():
        if isinstance(value, torch.nn.Parameter):
            value.requires_grad_(id(value) in active)
    for p in appearance.values():
        p.requires_grad_(True)
    cameras = load_cameras(binding['source'])
    assert len(cameras) == 135
    saves = [0, 4, 8, 20] if args.smoke else [0, 1350, 2700, 4050]
    model_dir = output/'model_mlp'
    model_dir.mkdir(exist_ok=False)
    import diff_surfel_rasterization as backend
    assert backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer'
    source_names = ['scripts/train_three_stage_refine.py', 'scripts/train_three_stage.py',
                    'scene/three_stage_pipeline.py', 'scene/three_stage_lighting.py',
                    'scene/three_stage_native_pbr.py', 'scene/three_stage_normal_origin.py',
                    'scene/three_stage_direct_base.py', 'gaussian_renderer/render_ir.py']
    marker = dict(parent_record, smoke=args.smoke, config=config, parent=str(parent),
                  parent_iteration=args.parent_iteration, parent_stage=3, parent_cumulative_actual_updates=parent_cumulative,
                  actual_entrypoint='scripts/train_three_stage_refine.py', stage3_continuation=True,
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
        residual_raw=fingerprint(model.residual_raw), geometry_optimizer_state_empty=all(
            getattr(model.pc, k) not in optimizer.state for k in GEOMETRY.values()),
        appearance_actual_lrs={g['name']: g['lr'] for g in optimizer.param_groups[:4]},
        camera_stack_empty=True, stage=3, gt_read=False, test_read=False))
    checkpoint(model, model_dir, 0, optimizer, parent_cumulative)
    stack, order, start = [], [], time.time()
    for step in range(1, config['iterations']+1):
        if not stack:
            stack = cameras.copy()
        camera = stack.pop(random.randrange(len(stack)))
        camera.load2device()
        order.append(camera.image_name_train_light)
        rates = geometry_lrs(args.geometry, step, config['iterations'])
        for group in optimizer.param_groups:
            if group['name'] in GEOMETRY:
                group['lr'] = rates[group['name']]
        package = model.render(camera, step, training=True)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        photo, terms = photometric_objective(package, target, alpha, camera.observed_saturation_mask.cuda(), model.stats,
                                             support=package['loss_support'])
        alpha_loss = F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6, 1-1e-6), alpha)
        photo = photo+.1*alpha_loss
        protect = step <= config['refine_protect_end']
        normal = expected_geometry_loss(model, camera, package, protect)
        env_anchor = (model.env.base-math.log(model.env_initial)).square().mean()
        amp_anchor = (.25*model.residual_raw.sigmoid()).square()
        loss = photo+.5*normal+.01*env_anchor+.01*amp_anchor
        assert bool(torch.isfinite(loss)), '非有限loss；保留失败产物。'
        audit = args.smoke or step in saves or step % 100 == 0
        geometry_gradients = {}
        if audit:
            params = [getattr(model.pc, k) for k in GEOMETRY.values()]
            normal_grads = torch.autograd.grad(.5*normal, params, retain_graph=True, allow_unused=True)
            photo_grads = torch.autograd.grad(photo, params, retain_graph=True, allow_unused=False)
            for name, ng, pg in zip(GEOMETRY, normal_grads, photo_grads):
                assert torch.isfinite(pg).all() and (ng is None or torch.isfinite(ng).all())
                geometry_gradients[name] = dict(normal_gradient_connected=ng is not None,
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
                photo_alpha=float(photo), normal=float(normal), normal_weight=.5, normal_depth_ratio=0.,
                pbr_depth_ratio=model.pipe.depth_ratio, normal_opacity_protected=protect,
                env_anchor=float(env_anchor), amplitude_anchor=float(amp_anchor),
                geometry_branch_gradients=geometry_gradients, groups=gradients, seconds=time.time()-start,
                terms={k: float(v) for k, v in terms.items()})
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del before
        if args.geometry == 'frozen':
            assert all(torch.equal(getattr(model.pc, k), v) for k, v in model.anchor_geometry.items())
        if step in saves:
            checkpoint(model, model_dir, step, optimizer, parent_cumulative)
        camera.load2device('cpu')
        del package, loss, photo, normal
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
    assert (max(drift.values()) == 0) if args.geometry == 'frozen' else all(v > 0 for v in drift.values())
    write(output/'engineering_receipt.json', dict(status='PASS_CONTINUATION_RELOAD_AND_LINEAR_COMPONENTS', checks=checks,
        smoke=args.smoke, actual_updates=step, cumulative_actual_updates=parent_cumulative+step,
        geometry_frozen=args.geometry == 'frozen', geometry_max_drift=drift,
        gt_read=False, test_read=False, inference_target_read=False, accuracy_claim=False))
    write(output/'state.json', dict(status='TRAINING_FINISHED_PENDING_VISUAL_AND_ACCURACY_REVIEW', exit_code=0, actual_updates=step))
    (output/'README.md').write_text('# 原生PBR末端继续\n\n'
        f'Stage3继续，source `{binding["source"]}`，model `{model_dir}`，父A42@4050累计51199；新增{step}，累计{parent_cumulative+step}。\n\n'
        f'完整外部训练/内置三TRAIN渲染命令见launch.json；内部参数`{shlex.join(sys.argv)}`。日志train.log、指标metrics.jsonl、渲染previews。\n\n'
        '四类与物理分量实际目检、完整几何离线评价、视频尚待执行；工程重载和加和PASS，最终精度待评。失败产物全部保留。\n')


if __name__ == '__main__':
    main()
