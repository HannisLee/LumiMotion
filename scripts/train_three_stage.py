"""复用原生B00的共享法线SH几何优化和原生PBR残差外观优化。"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scene.three_stage_pipeline import ThreeStageModel, sha, write
from scene.three_stage_lighting import photometric_objective
from scene.stage2_photometric import linear_to_srgb


def load_cameras(source):
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    infos = readCamerasFromTransforms(str(source), 'transforms_train.json', 'images', False, no_bg=True)
    config = SimpleNamespace(resolution=2, data_device='cpu', load2gpu_on_the_fly=True)
    cameras = [loadCam(config, i, info, 1.) for i, info in enumerate(infos)]
    assert [int(c.image_name_train_light.split('_')[-1]) for c in cameras] == [n for n in range(1, 151) if n % 10]
    return cameras


def install_guard(source, output, parent, binding):
    source, output, parent = [Path(p).resolve() for p in (source, output, parent)]
    exact = {Path(binding[k]).resolve() for k in ('directions', 'stats')}
    restricted = {'.ply', '.pth', '.pt', '.exr', '.blend', '.png', '.jpg', '.jpeg', '.npz', '.hdr'}
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if any(path.is_relative_to(p) for p in (source, output, parent)) or path in exact:
            return
        if 'eval_only' in path.parts or path.name in {'lights.json', 'object_pose.json', 'bone_pose.json'} or path.suffix.lower() in restricted:
            raise PermissionError('三阶段TRAIN禁止读取非白名单模型、图像或GT：'+str(path))
    sys.addaudithook(guard)


def geometry_loss(model, camera, photo, protect_opacity):
    from gaussian_renderer import render
    pc = model.effective(camera)
    geometry = render(camera, pc, model.pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
                      override_color=torch.zeros_like(pc.get_xyz), detach_opacity=protect_opacity)
    normal = F.normalize(geometry['rend_normal'], dim=0)
    depth_normal = F.normalize(geometry['surf_normal'], dim=0)
    alpha = camera.gt_alpha_mask.cuda()
    interior = -F.max_pool2d(-(alpha > .99).float()[None], 5, stride=1, padding=2)[0]
    mask = interior*(photo['rend_alpha'].detach() > .5)
    loss = ((1-(normal*depth_normal).sum(0, keepdim=True))*mask).sum()/mask.sum().clamp_min(1)
    return loss


def preview(path, package):
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    views = dict(rgb=package['render'], alpha=package['rend_alpha'].repeat(3, 1, 1),
        normal=(F.normalize(package['rend_normal'], dim=0)+1)/2,
        base=linear_to_srgb(package['base_linear'].clamp_min(0)),
        residual=linear_to_srgb(package['residual_linear'].clamp_min(0)))
    for field in ('environment_linear', 'direct_base_linear'):
        if field in package:
            views[field] = linear_to_srgb(package[field].clamp_min(0))
    for name, tensor in views.items():
        image = (tensor.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1]*255).round().astype(np.uint8)
        assert cv2.imwrite(str(path/(name+'.png')), image)


def audit_initial(model, cameras, output):
    """三个TRAIN相机比较B00真实入口与保留逐帧几何后的同一光栅。"""
    from scene import GaussianModel
    from gaussian_renderer import render
    parent = GaussianModel(3, no_binary_separation=False, fea_dim=1)
    parent.load_ply(str(model.parent/f'point_cloud/iteration_{model.parent_iteration}/point_cloud.ply'))
    parent.active_sh_degree = 3
    keys = ('_xyz', '_rotation', '_scaling', '_opacity', 'feature', '_albedo_rest', '_albedo_dc_stage1')
    assert all(torch.equal(getattr(parent, k), getattr(model.pc, k)) for k in keys)
    rows = []
    with torch.no_grad():
        for i in (0, len(cameras)//2, -1):
            camera = cameras[i]; camera.load2device()
            effective = model.effective(camera)
            values = effective.values
            # 使用父模型相同原生sRGB颜色，单独验证逐帧几何包装无更改。
            a = render(camera, parent, model.pipe, torch.zeros(3, device='cuda'),
                **{k: values[k] for k in ('d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color')})
            from scene.three_stage_lighting import EffectiveGaussians
            b = render(camera, EffectiveGaussians(parent, values), model.pipe, torch.zeros(3, device='cuda'),
                       0., 0., 0., d_color=values['d_color'])
            differences = {k: float((a[k]-b[k]).abs().max()) for k in ('render', 'rend_alpha', 'rend_normal', 'surf_depth')}
            # 二次四元数单位化只允许float舍入误差。
            assert all(v < 2e-5 for v in differences.values()), differences
            assert torch.isfinite(effective.get_scaling).all() and (effective.get_scaling > 0).all()
            rows.append(dict(camera=camera.image_name_train_light, max_abs=differences,
                             deform_xyz_max=float(values['d_xyz'].abs().max()),
                             deform_rotation_max=float(values['d_rotation'].abs().max())))
            camera.load2device('cpu')
    write(output/'initial_forward_audit.json', dict(status='PASS_NATIVE_EFFECTIVE_GEOMETRY', rows=rows,
        inherited_keys=list(keys), original_geometry_bitexact=True, parent_optimizer_loaded=False,
        material_conversion='只有Stage2的物理albedo DC改变；原生DC副本保留', gt_read=False))
    del parent


def main(model_class=None, actual_entrypoint='scripts/train_three_stage.py', extra_sources=()):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', type=int, choices=[2, 3], required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parent_iteration', type=int)
    parser.add_argument('--direction_lr', type=float, choices=[0., 1e-6], default=0.)
    parser.add_argument('--lambda_mix', type=float, choices=[.25, .5], default=.25,
                        help='仅Stage2混合权重；Stage3自动继承父Stage2记录。')
    parser.add_argument('--normal_weight', type=float, choices=[.1, .5], default=.1,
                        help='仅Stage2的GS/Expected一致性损失权重；Stage3继承父记录。')
    parser.add_argument('--normal_protection', choices=['warmup', 'joint'], default='warmup',
                        help='仅Stage2：normal到opacity的恢复期保护或全联合阶段隔离。')
    parser.add_argument('--lambda_residual', type=float, choices=[0., .1], default=0.)
    parser.add_argument('--pbr_samples', type=int, choices=[16, 128, 256], default=16)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get('THREE_STAGE_REPO_ROOT', str(code_root))).resolve()
    args.output = args.output.resolve()
    assert args.output.is_relative_to(root/'output')
    assert args.smoke == ('smoke_test' in args.output.parts)
    if args.output.exists() and set(p.name for p in args.output.iterdir())-{'launch.json', 'train.log', 'job_state.json'}:
        raise FileExistsError('已有实验禁止覆盖，重试必须使用新目录。')
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output
    binding = json.loads(args.binding.read_text())
    for name, digest in binding['input_sha256'].items():
        assert sha(name) == digest, name
    parent = args.parent or Path(binding['parent'])
    parent_iteration = args.parent_iteration if args.parent_iteration is not None else binding['parent_iteration']
    if args.stage == 2:
        assert parent.resolve() == Path(binding['parent']) and parent_iteration == 35000
    else:
        record = json.loads((parent/'three_stage_training.json').read_text())
        assert record['stage'] == 2
        assert args.smoke or not record['smoke'], '工程冒烟父模型不能进入正式训练。'
        if args.lambda_mix != .25:
            raise ValueError('Stage3不新设Stage2混合权重，必须继承父记录。')
        if args.normal_weight != .1:
            raise ValueError('Stage3不新设几何一致性权重，必须继承父记录。')
        if args.normal_protection != 'warmup':
            raise ValueError('Stage3不新设几何梯度保护，必须继承父记录。')
    install_guard(binding['source'], output, parent, binding)
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    inherited_mix = args.lambda_mix if args.stage == 2 else record['config']['lambda_mix']
    config = dict(direction_lr=args.direction_lr, lambda_mix=inherited_mix, lambda_residual=args.lambda_residual,
        normal_weight=args.normal_weight if args.stage == 2 else record['config'].get('normal_weight', .1),
        normal_protection=args.normal_protection if args.stage == 2 else record['config'].get('normal_protection', 'warmup'),
        shadow_bias=.009978598594665527, pbr_samples=args.pbr_samples, pbr_pixels=2048,
        base_end=4 if args.smoke else 540, residual_end=8 if args.smoke else 1080,
        protect_end=12 if args.smoke else 1280, iterations=20 if args.smoke else (8100 if args.stage == 2 else 4050))
    if args.stage == 2 and config['normal_protection'] == 'joint':
        config['protect_end'] = config['iterations']
    saves = [0, 4, 8, 20] if args.smoke else ([0, 540, 1080, 4050, 8100] if args.stage == 2 else [0, 1350, 2700, 4050])
    model_dir = output/'model_mlp'; model_dir.mkdir()
    model = (model_class or ThreeStageModel)(parent, parent_iteration, binding, args.stage, config)
    cameras = load_cameras(binding['source'])
    import diff_surfel_rasterization as backend
    assert backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer'
    marker = dict(stage=args.stage, smoke=args.smoke, config=config, binding=binding,
        binding_sha256=sha(args.binding), parent=str(parent), parent_iteration=parent_iteration,
        actual_entrypoint=actual_entrypoint, gt_supervision=False, external_test_read=False,
        train_frames=[n for n in range(1, 151) if n % 10],
        rasterizer_provenance=backend.provenance,
        assets_sha256={str(Path(binding[k])): sha(binding[k]) for k in ('directions', 'stats')},
        sources={str(code_root/name): sha(code_root/name) for name in ['scripts/train_three_stage.py', 'scene/three_stage_pipeline.py',
            'scene/three_stage_lighting.py', 'gaussian_renderer/render_ir.py', *extra_sources]})
    write(model_dir/'three_stage_training.json', marker)
    # 原始cfg作为可查看的相机和模型结构记录；新光照语义只由three_stage记录决定。
    (model_dir/'cfg_args').write_text((parent/'cfg_args').read_text())
    write(output/'plan.json', dict(argv=sys.argv, config=config, saves=saves, parent=str(parent),
        parent_iteration=parent_iteration, source=binding['source'], model=str(model_dir), stage=args.stage,
        command=' '.join(shlex.quote(v) for v in sys.argv)))
    if args.stage == 2:
        audit_initial(model, cameras, output)
    geometry_names = {'xyz': '_xyz', 'rotation': '_rotation', 'scale': '_scaling', 'opacity': '_opacity'}
    parameters = [('albedo', model.pc._albedo_dc, 1e-3 if args.stage == 2 else 2e-4)]
    if args.stage == 2:
        parameters += [(k, getattr(model.pc, v), lr) for (k, v), lr in zip(geometry_names.items(), [1e-6, 5e-5, 1e-5, 1e-3])]
        parameters += [('sh', model.lights.sh, 1e-3), ('direction', model.lights.raw_direction, args.direction_lr)]
    else:
        parameters += [('roughness', model.pc._roughness, 2e-4), ('environment', model.env.base, 1e-3)]
        if args.lambda_residual > 0:
            parameters += [('residual_amplitude', model.residual_raw, 1e-3)]
    active = {id(p) for _, p, _ in parameters}
    for p in vars(model.pc).values():
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad_(id(p) in active)
    groups = [dict(params=[p], lr=lr, initial_lr=lr, name=name) for name, p, lr in parameters]
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    stack = []
    model.checkpoint(model_dir, 0, optimizer)
    start = time.time()
    for step in range(1, config['iterations']+1):
        if not stack:
            stack = cameras.copy()
        camera = stack.pop(random.randrange(len(stack))); camera.load2device()
        for group in optimizer.param_groups:
            name = group['name']; lr = group['initial_lr']
            if args.stage == 2:
                if name in geometry_names or name == 'direction':
                    progress = max(0, step-config['residual_end'])/max(1, config['iterations']-config['residual_end'])
                    lr = 0. if step <= config['residual_end'] else lr*(1-.9*progress)
                elif name == 'sh':
                    lr = 1e-3 if step <= config['base_end'] else (0. if step <= config['residual_end'] else 1e-5)
                elif name == 'albedo':
                    lr = 1e-3 if step <= config['base_end'] else 2e-4
            else:
                lr *= 1-.9*step/config['iterations']
            group['lr'] = lr
        package = model.render(camera, step, training=args.stage == 3)
        target, alpha = camera.original_image_train_light.cuda(), camera.gt_alpha_mask.cuda()
        saturated = camera.observed_saturation_mask.cuda()
        loss, terms = photometric_objective(package, target, alpha, saturated, model.stats,
            support=package['loss_support'] if args.stage == 3 else None)
        alpha_loss = F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6, 1-1e-6), alpha)
        loss = loss+.1*alpha_loss
        if args.stage == 2 and step > config['residual_end']:
            protect_normal_opacity = step <= config['protect_end']
            normal = geometry_loss(model, camera, package, protect_normal_opacity)
            anchor, smooth = model.lights.direction_regularization()
            loss = loss+config['normal_weight']*normal+anchor+10*smooth
            terms.update(normal=normal, direction_anchor=anchor, direction_smooth=smooth,
                         normal_opacity_protected=protect_normal_opacity)
        if args.stage == 3:
            anchor = (model.env.base-math.log(model.env_initial)).square().mean()
            loss = loss+.01*anchor+.01*(.25*model.residual_raw.sigmoid()).square()
            terms['env_anchor'] = anchor
        if not torch.isfinite(loss):
            raise FloatingPointError('非有限loss，所有产物保留。')
        loss.backward()
        audit = args.smoke or step in saves or step % 100 == 0
        if audit:
            gradients = {g['name']: dict(lr=g['lr'], gradient_norm=sum(float(p.grad.norm()) for p in g['params'] if p.grad is not None),
                gradient_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in g['params'])) for g in groups}
            assert all(row['gradient_finite'] for row in gradients.values()), gradients
            before = {g['name']: [p.detach().clone() for p in g['params']] for g in groups} if args.smoke else None
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            from utils.sh_utils import RGB2SH, SH2RGB
            model.pc._albedo_dc.copy_(RGB2SH(SH2RGB(model.pc._albedo_dc).clamp(.030001, .969999)))
            if args.direction_lr > 0 and step > config['residual_end']:
                model.lights.project()
            if model.env is not None:
                model.env.base.clamp_(math.log(model.env_initial*.5), math.log(model.env_initial*2))
        if audit:
            if before:
                for g in groups:
                    gradients[g['name']]['actual_max_change'] = max(float((p-v).abs().max()) for p, v in zip(g['params'], before[g['name']]))
            row = dict(step=step, camera=camera.image_name_train_light, loss=float(loss), alpha_loss=float(alpha_loss),
                seconds=time.time()-start, groups=gradients, visibility_mean=float(package['visibility_mean']),
                terms={k: float(v) for k, v in terms.items()},
                base_linear_mean=float(package['base_linear'].detach().mean()),
                residual_linear_mean=float(package['residual_linear'].detach().mean()))
            with (output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if step in saves:
            model.checkpoint(model_dir, step, optimizer)
        camera.load2device('cpu')
        del package, loss
    # 保存候选用重算BVH的完整TRAIN前向检查，不借用训练mask或缓存可见度。
    checks = []
    with torch.no_grad():
        for i in (0, len(cameras)//2, -1):
            camera = cameras[i]; camera.load2device()
            torch.manual_seed(1000+i); torch.cuda.manual_seed_all(1000+i)
            package = model.render(camera, step, components=True)
            assert torch.isfinite(package['render']).all()
            error = float(package['component_add_max_error'])
            assert error < 2e-5, error
            preview(output/f'previews/{camera.image_name_train_light}', package)
            checks.append(dict(frame=camera.image_name_train_light, add_max_error=error,
                render=package['render'].detach().cpu(), alpha=package['rend_alpha'].detach().cpu()))
            camera.load2device('cpu')
        model.reload_endpoint(model_dir, step)
        for i, check in zip((0, len(cameras)//2, -1), checks):
            camera = cameras[i]; camera.load2device()
            torch.manual_seed(1000+i); torch.cuda.manual_seed_all(1000+i)
            package = model.render(camera, step, components=True)
            check['reload_max_abs'] = float((check.pop('render')-package['render'].cpu()).abs().max())
            check['reload_alpha_max_abs'] = float((check.pop('alpha')-package['rend_alpha'].cpu()).abs().max())
            assert check['reload_max_abs'] < 2e-5 and check['reload_alpha_max_abs'] < 2e-5, check
            camera.load2device('cpu')
    if args.stage == 3:
        assert all(torch.equal(getattr(model.pc, k), v) for k, v in model.anchor_geometry.items()), 'Stage3几何发生变化'
    write(output/'engineering_receipt.json', dict(status='PASS_RELOAD_AND_LINEAR_COMPONENTS', checks=checks,
        smoke=args.smoke, geometry_frozen=args.stage == 3, gt_read=False, test_read=False,
        accuracy_claim=False, actual_updates=step, inference_target_read=False))
    write(output/'state.json', dict(status='TRAINING_FINISHED_PENDING_VISUAL_AND_ACCURACY_REVIEW', exit_code=0, actual_updates=step))
    (output/'README.md').write_text('# 三阶段训练实验\n\n'
        f'逻辑Stage{args.stage}，source：`{binding["source"]}`；model：`{model_dir}`；相对iteration：{step}。\n\n'
        f'完整训练参数：`{shlex.join(sys.argv)}`；完整外部启动命令见同目录launch.json。日志train.log，定量metrics.jsonl。\n\n'
        '渲染命令：同一训练入口在保存后自动重算完整BVH并渲染三个TRAIN相机；输出previews。离线完整渲染命令和全序列评价待补。\n\n'
        'RGB／Alpha／Normal目检：尚未执行。Separation：尚未导出。EL／DL或PBR／残差目检：尚未执行。\n\n'
        '代表图片：previews/image_0001、image_0075、image_0149。视频：尚未导出。完整几何定量：尚未执行。\n\n'
        '工程保存／重载与线性加和：PASS，见engineering_receipt.json。最终重建验收：待评，不能宣称PASS。失败结果完整保留。\n')


if __name__ == '__main__':
    main()
