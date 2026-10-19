"""仅用TRAIN分解冻结PBR误差；无遮挡只作机制探针，不作为候选模型。"""
import argparse
import importlib
import json
from pathlib import Path
import shlex
import sys

import cv2
import torch
import torch.nn.functional as F

from scene.three_stage_native_pbr import load_three_stage_model
from scene.three_stage_pipeline import sha, write
from scene.three_stage_lighting import censored_error
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from scripts.train_three_stage import install_guard, load_cameras


def correlation(a, b, mask):
    a, b = a[mask].double(), b[mask].double()
    if a.numel() < 2:
        return None
    a, b = a-a.mean(), b-b.mean()
    denominator = a.square().sum().sqrt()*b.square().sum().sqrt()
    return float((a*b).sum()/denominator) if denominator > 1e-12 else None


def png(path, tensor):
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    pixels = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]*255).round().astype('uint8')
    if not cv2.imwrite(str(path), pixels):
        raise IOError(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--normal-origin-probe', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.model, args.output = args.model.resolve(), args.output.resolve()
    if not args.output.is_relative_to(root/'output'):
        raise ValueError('诊断输出必须进入output。')
    marker = json.loads((args.model/'three_stage_training.json').read_text())
    if marker['smoke'] or marker['stage'] != 3 or marker['gt_supervision'] or marker['config']['lambda_residual'] != 0:
        raise ValueError('探针只接受正式Stage3原生PBR基础组。')
    if marker['config'].get('pbr_origin_mode', 'ray_only') != 'ray_only':
        raise ValueError('此探针对照旧ray_only起点，拒绝在新模型上重复叠加法线偏移。')
    for filename, digest in marker['sources'].items():
        if sha(filename) != digest:
            raise ValueError('冻结训练源码变化。')
        path = Path(filename)
        if not (path.parent.name == 'scripts' and path.name.startswith('train_three_stage')):
            if sha(root/path.parent.name/path.name) != digest:
                raise ValueError('当前实际前向与训练来源不同。')
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.model/f'three_stage/iteration_{args.iteration}/state.pt'
    before = sha(checkpoint)
    write(args.output/'launch.json', dict(command=shlex.join([sys.executable, '-m', 'scripts.diagnose_three_stage_pbr_error', *sys.argv[1:]]),
        script_sha256=sha(__file__), mode_note='固定状态机制探针；不做优化、不保存新模型、不读取TEST/GT'))
    binding = marker['binding']
    install_guard(binding['source'], args.output, args.model, binding)
    torch.set_num_threads(4)
    model = load_three_stage_model(args.model, args.iteration)
    if model.config.get('pbr_surface_depth_mode') != 'native_half':
        raise ValueError('只诊断已修复原生混合深度PBR。')
    cameras = load_cameras(binding['source'])
    rows = []
    with torch.no_grad():
        for camera in cameras:
            number = int(camera.image_name_train_light.split('_')[-1])
            if number not in [1, 75, 149]:
                continue
            camera.load2device()
            target = camera.original_image_train_light.cuda()
            observed = srgb_to_linear(target)
            original_alpha = camera.gt_alpha_mask.cuda()
            saturated = camera.observed_saturation_mask.cuda()
            # 原生diffuse/specular显示字段默认截断，不能逆变换恢复HDR。
            # 只旁观实际着色器返回值，不改变着色结果或冻结源码。
            ir = importlib.import_module('gaussian_renderer.render_ir')
            original_equation = ir.rendering_equation
            captured = []

            def record_equation(*values, **options):
                package = original_equation(*values, **options)
                captured.append({key: package[key].detach().clone() for key in ['diffuse', 'specular']})
                return package

            try:
                ir.rendering_equation = record_equation
                native = model.render(camera, args.iteration, components=True)
            finally:
                ir.rendering_equation = original_equation
            channels = {}
            for kind in ['diffuse', 'specular']:
                image = torch.zeros_like(native['render_linear']).permute(1, 2, 0)
                image[native['mask']] = torch.cat([item[kind] for item in captured], dim=0)
                channels[kind] = image.permute(2, 0, 1)*native['rend_alpha']
            normal = F.normalize(native['rend_normal'], dim=0)
            fg = original_alpha > .5
            upward = (original_alpha > .99) & (native['rend_alpha'] > .99) & (normal[2:3] > .9)
            visibility = native['visibility']/native['rend_alpha'].clamp_min(1e-6)
            original_trace = model.pc.trace
            packages = {'native': native}
            try:
                model.pipe.wo_indirect = True
                packages['no_indirect_probe'] = model.render(camera, args.iteration, components=True)
                model.pipe.wo_indirect = False

                def no_hit(origin, direction, **kwargs):
                    shape = direction.shape[:-1]
                    return dict(alpha=direction.new_zeros(shape), color=direction.new_zeros((*shape, 3)))

                model.pc.trace = no_hit
                packages['unoccluded_probe'] = model.render(camera, args.iteration, components=True)
            finally:
                model.pc.trace = original_trace
                model.pipe.wo_indirect = False
                model.visibility.invalidate()
            if args.normal_origin_probe:
                # 只检查一个已有尺度：沿预测着色法线偏移native light_t_min。
                # 深度、Gaussian、BRDF、环境、DL及原沿射线偏移均保持原值。
                def shifted_equation(*values, **options):
                    values = list(values)
                    values[3] = values[3]+values[2]*model.pipe.light_t_min
                    return original_equation(*values, **options)

                try:
                    ir.rendering_equation = shifted_equation
                    packages['normal_origin_probe'] = model.render(camera, args.iteration, components=True)
                finally:
                    ir.rendering_equation = original_equation
            frame = args.output/f'image_{number:04d}'
            frame.mkdir()
            png(frame/'target.png', target)
            png(frame/'alpha.png', native['rend_alpha'])
            png(frame/'normal.png', (normal*.5+.5)*native['rend_alpha'])
            from gaussian_renderer import render
            effective = model.effective(camera)
            feature = model.pc.get_binary_feature()
            colors = torch.cat((feature, 1-feature, torch.zeros_like(feature)), dim=1)
            separation = render(camera, effective, model.pipe, effective.get_xyz.new_zeros(3), 0., 0., 0., override_color=colors)['render']
            png(frame/'separation.png', separation)
            png(frame/'hemisphere_visibility.png', visibility)
            png(frame/'albedo.png', native['base_color'])
            png(frame/'roughness.png', native['roughness'])
            png(frame/'indirect_incident_display.png', native['light_indirect'])
            deficit = (observed-native['render_linear']).clamp_min(0).mean(0, keepdim=True)
            overshoot = ((native['render_linear']-observed).clamp_min(0)*(~saturated)).mean(0, keepdim=True)
            png(frame/'positive_deficit_x4.png', deficit*4)
            png(frame/'unsaturated_overshoot_x4.png', overshoot*4)
            row = dict(frame=number, modes={}, regions={})
            for name, package in packages.items():
                if not torch.equal(package['rend_alpha'], native['rend_alpha']):
                    raise AssertionError('只读探针改变Alpha。')
                if float(package['component_add_max_error']) >= 2e-5:
                    raise AssertionError('物理分量加和失败。')
                png(frame/(name+'.png'), package['render'])
                error = censored_error(package['render_linear'], observed, saturated).mean(0, keepdim=True)
                mode_visibility = package['visibility']/package['rend_alpha'].clamp_min(1e-6)
                highpass = mode_visibility-F.avg_pool2d(mode_visibility[None], 3, 1, 1)[0]
                png(frame/(name+'_visibility.png'), mode_visibility)
                row['modes'][name] = dict(censored_foreground_mean=float(error[fg].mean()),
                    censored_upward_mean=float(error[upward].mean()) if upward.any() else None,
                    alpha_max_abs=float((package['rend_alpha']-native['rend_alpha']).abs().max()),
                    normal_max_abs=float((package['rend_normal']-native['rend_normal']).abs().max()),
                    depth_max_abs=float((package['surf_depth']-native['surf_depth']).abs().max()),
                    upward_visibility_mean=float(mode_visibility[upward].mean()) if upward.any() else None,
                    upward_visibility_std=float(mode_visibility[upward].std()) if upward.any() else None,
                    upward_visibility_highpass_rms=float(highpass[upward].square().mean().sqrt()) if upward.any() else None)
            diffuse, specular = channels['diffuse'], channels['specular']
            row['display_inverse_diffuse_specular_error'] = float((
                (srgb_to_linear(native['diffuse'])+srgb_to_linear(native['specular']))*native['rend_alpha']-native['base_linear']).abs().max())
            row['diffuse_specular_add_max_abs'] = float((diffuse+specular-native['base_linear']).abs().max())
            if row['diffuse_specular_add_max_abs'] > 2e-5:
                raise AssertionError('PBR漫反射/镜面线性分解失败。')
            png(frame/'diffuse.png', linear_to_srgb(diffuse))
            png(frame/'specular.png', linear_to_srgb(specular))
            for name, mask in [('foreground', fg), ('predicted_upward', upward),
                               ('upward_visibility_below_half', upward & (visibility < .5)),
                               ('upward_visibility_half_to_eight', upward & (visibility >= .5) & (visibility < .8)),
                               ('upward_visibility_above_eight', upward & (visibility >= .8))]:
                count = int(mask.sum())
                if not count:
                    row['regions'][name] = dict(pixels=0)
                    continue
                row['regions'][name] = dict(pixels=count, visibility_mean=float(visibility[mask].mean()),
                    positive_deficit=float(deficit[mask].mean()), unsaturated_overshoot=float(overshoot[mask].mean()),
                    diffuse_mean=float(diffuse[:, mask[0]].mean()), specular_mean=float(specular[:, mask[0]].mean()),
                    correlation_deficit_to_occlusion=correlation(deficit, 1-visibility, mask))
            rows.append(row)
            write(args.output/f'row_{number:04d}.json', row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            camera.load2device('cpu')
        after = sha(checkpoint)
        if before != after:
            raise AssertionError('只读诊断改变checkpoint。')
    write(args.output/'receipt.json', dict(status='PASS_TRAIN_PBR_ERROR_DIAGNOSTIC_EXECUTION', model=str(args.model),
        iteration=args.iteration, source=binding['source'], frames=[1, 75, 149], rows=rows,
        checkpoint_sha256=after, checkpoint_unchanged=True, script_sha256=sha(__file__),
        optimizer_updates=0, gt_read=False, test_read=False, manual_review='PENDING',
        normal_origin_probe=args.normal_origin_probe, normal_origin_offset=model.pipe.light_t_min if args.normal_origin_probe else None,
        region_note='上向区域只由预测GS法线和原Alpha定义；不是GT平台语义；可见度分桶只作机制诊断。',
        probe_note='去间接/无遮挡是固定状态探针，不是新的训练候选、最终渲染或最优参数扫描。'))


if __name__ == '__main__':
    main()
