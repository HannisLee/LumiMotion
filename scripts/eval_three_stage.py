"""复用V3固定几何评测协议，显式加载共享SH／原生PBR残差checkpoint。"""
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from LIHAN.scripts import eldl_v3_eval as legacy


def model_record(model, iteration, split, smoke=False):
    model = Path(model)
    record = legacy.read(model/'three_stage_training.json')
    if record['gt_supervision'] is not False or record['external_test_read'] is not False:
        raise ValueError('三阶段训练隔离记录非法。')
    if record['train_frames'] != legacy.TRAIN:
        raise ValueError('三阶段必须使用135TRAIN。')
    if record['smoke'] and not smoke:
        raise ValueError('冒烟模型禁止进入正式精度表。')
    files = [model/'three_stage_training.json', model/'cfg_args',
        model/f'point_cloud/iteration_{iteration}/point_cloud.ply', model/f'deform/iteration_{iteration}/deform.pth',
        model/f'three_stage/iteration_{iteration}/state.pt', model/f'three_stage/iteration_{iteration}/receipt.json']
    for filename, digest in record['sources'].items():
        if legacy.sha(filename) != digest:
            raise ValueError('冻结训练源码被改动，需要显式新版本adapter：'+filename)
        relative = Path(filename).parts[-2:]
        current = Path(__file__).resolve().parents[1]/Path(*relative)
        # 评测不执行优化入口；原训练入口仍必须由上面的冻结SHA核验。
        # CLI扩展不应迫使旧Stage2重训；三个实际前向模块仍严格逐字匹配。
        optimization_entry = Path(filename).parent.name == 'scripts' and Path(filename).name.startswith('train_three_stage')
        if not optimization_entry and legacy.sha(current) != digest:
            raise ValueError('当前接口与冻结训练源码不同：'+str(current))
    recorded_ratio = 1. if record['config'].get('pbr_surface_depth_mode') == 'native_half' else 0.
    record = dict(record, baseline='three_stage', args=dict(source_path=record['binding']['source'], depth_ratio=recorded_ratio))
    return record, {str(path): legacy.sha(path) for path in files}


def adapter(model, iteration, source, split, smoke=False):
    import torch
    import diff_surfel_rasterization as backend
    from scene.three_stage_native_pbr import load_three_stage_model
    from scene.cameras import Camera
    from scene.dataset_readers import blender_c2w_to_camera_extrinsics
    from scripts.eval_geometry_gt import GeometryRun
    from utils.geometry_eval_utils import unit_normal
    from gaussian_renderer import render
    record, files = model_record(model, iteration, split, smoke)
    if backend.provenance['source_sha256'] != record['rasterizer_provenance']['source_sha256']:
        raise ValueError('训练评测原生CUDA来源不同。')

    class Run(GeometryRun):
        def __init__(self):
            self.pipeline = load_three_stage_model(model, iteration)
            self.pc, self.deform = self.pipeline.pc, self.pipeline.deform
            self.model, self.iteration, self.source = Path(model), iteration, Path(source)
            self.context = None
            self.record, self.files, self.backend = record, files, backend.provenance
            self.config = dict(vars(self.pipeline.pipe), train_light_folder='images')
            self.source_manifest = legacy.read(self.source/'manifest.json')
            self.files[str(self.source/'manifest.json')] = legacy.sha(self.source/'manifest.json')
            self.training_core_comparison = {name: True for name in record['sources'] if not (Path(name).parent.name == 'scripts' and Path(name).name.startswith('train_three_stage'))}
            self.light_prediction_provenance = dict(method='checkpoint TRAIN-only nlerp and nearest endpoint hold',
                gt_light_read=False, checkpoint_asset=True, inference_target_read=False)

        def camera(self, metadata, frame, divisor=2):
            path = self.source/'images'/frame['file_path']
            expected = self.source_manifest['files']['images/'+frame['file_path']]
            if legacy.sha(path) != expected:
                raise ValueError('评测RGB输入SHA变化。')
            self.files[str(path)] = expected
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw.dtype != np.uint16 or metadata['rgba16'] is not True:
                raise ValueError('必须使用固定16位评测源。')
            oh, ow = raw.shape[:2]; w, h = round(ow/divisor), round(oh/divisor)
            if abs(frame['cx']-ow/2) > 1e-5 or abs(frame['cy']-oh/2) > 1e-5:
                raise ValueError('原生光栅需要居中主点。')
            rgba = raw[..., [2, 1, 0, 3]].astype(np.float32)/65535.
            alpha = cv2.resize(rgba[..., 3], (w, h), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgba[..., :3]*rgba[..., 3:4], (w, h), interpolation=cv2.INTER_AREA)
            rotation, translation = blender_c2w_to_camera_extrinsics(frame['transform_matrix'])
            cam = Camera(colmap_id=frame['source_frame'], R=rotation, T=translation,
                FoVx=2*math.atan(ow/(2*frame['fl_x'])), FoVy=2*math.atan(oh/(2*frame['fl_y'])),
                image_train_light=torch.zeros(3, h, w), gt_alpha_mask=torch.from_numpy(alpha[None]),
                image_name_train_light=path.stem, image_path_train_light=str(path), uid=frame['source_frame'],
                data_device='cuda', fid=frame['time'])
            return cam, alpha, np.array([frame['fl_x']/divisor, frame['fl_y']/divisor,
                                         frame['cx']/divisor, frame['cy']/divisor]), rgb

        def deformation(self, camera):
            return self.pipeline.effective(camera).values

        def rgb(self, camera, colors=None):
            if colors is not None:
                effective = self.pipeline.effective(camera)
                return render(camera, effective, self.pipeline.pipe, torch.zeros(3, device='cuda'),
                              0., 0., 0., override_color=colors)
            return self.pipeline.render(camera, iteration, components=True)

        def prediction(self, camera, c2w, intrinsic):
            prediction = self.render(camera, c2w, intrinsic)
            package = self.rgb(camera)
            normal, valid = unit_normal(package['rend_normal'].permute(1, 2, 0).cpu().numpy())
            visible = prediction['alpha'] >= .5
            depth_mix = min(.5, self.pipeline.pipe.depth_ratio) if self.pipeline.stage == 3 else self.pipeline.pipe.depth_ratio
            shading_depth = (1-depth_mix)*prediction['expected']+depth_mix*prediction['median']
            parity = dict(normal_max_abs=float(np.max(np.abs(normal[valid]-prediction['normal'][valid]))) if valid.any() else 0.,
                alpha_max_abs=float(np.max(np.abs(package['rend_alpha'][0].cpu().numpy()-prediction['alpha']))),
                surf_depth_max_abs=float(np.max(np.abs(package['surf_depth'][0].cpu().numpy()[visible]-shading_depth[visible]))) if visible.any() else 0.,
                component_add_max_error=float(package['component_add_max_error']))
            if any(v > 2e-5 for v in parity.values()):
                raise AssertionError('完整RGB与独立几何／分量加和不同：'+str(parity))
            for field in ('render', 'base_linear', 'residual_linear', 'render_linear'):
                prediction['rgb' if field == 'render' else field] = package[field].permute(1, 2, 0).cpu().numpy()
            return prediction, parity
    return Run()


def main():
    old_plan, old_metrics = legacy.make_plan, legacy.metrics
    def make_plan(args):
        commands = old_plan(args)
        for item in commands:
            command = item['command']
            command = ['scripts.eval_three_stage' if v == 'LIHAN.scripts.eldl_v3_eval' else
                       str(Path(__file__).resolve()) if v == str(Path(legacy.__file__).resolve()) else v for v in command]
            if len(command) > 3 and Path(command[1]).name == 'bootstrap_1011.py':
                command[2] = str(Path(__file__).resolve().parents[1])
            import shlex
            item.update(command=command, shell=shlex.join(command))
        protocol = legacy.read(args.output/'protocol.json')
        protocol['three_stage_adapter_sha256'] = legacy.sha(__file__)
        legacy.write(args.output/'protocol.json', protocol)
        plan = legacy.read(args.output/'plan.json')
        plan.update(steps=commands, protocol_sha256=legacy.sha(args.output/'protocol.json'),
                    three_stage_adapter_sha256=legacy.sha(__file__))
        legacy.write(args.output/'plan.json', plan)
        lines = ['# 三阶段固定协议评测', '',
            f'source：`{args.source}`；model：`{args.model}`；iteration：{args.iteration}；split：{args.split}。', '',
            '本模型使用三阶段显式adapter；完整实际训练来源见training_record_snapshot.json。',
            '输出目录：当前实验目录；渲染predictions（冒烟为smoke_render）、Mesh为predicted_meshes、统计metrics、分量components、四类图与视频four_classes。',
            '日志logs；完整实际渲染／重建／聚合／可视化命令（GPU环境见plan.json）：', '']
        for item in commands:
            lines += ['```bash', item['shell'], '```', '']
        lines += ['定量指标：等待执行。代表图片／视频：等待各步骤生成。', '',
            'RGB／Alpha／Normal／Separation及EL／DL或PBR／残差目检：尚未执行。', '',
            '最终结论：PENDING；工程通过不代替重建精度。失败保留所有checkpoint、渲染、统计及日志。', '']
        if args.split == 'held':
            lines += ['本划分是已参与开发的TRAIN相机；不称为未见验证集，不读取受限GT。', '']
        (args.output/'README.md').write_text('\n'.join(lines))
        return commands

    def metrics(args):
        old_metrics(args)
        import torch
        state = torch.load(args.model/f'three_stage/iteration_{args.iteration}/state.pt', map_location='cpu', weights_only=False)
        stats = state['stats']; rows = []
        out = args.output/'components'; out.mkdir(exist_ok=False)
        manifest = legacy.read(args.output/'predictions/manifest.json')
        for number, entry in manifest['frames'].items():
            data = dict(np.load(args.output/'predictions'/entry['file'], allow_pickle=False))
            srgb = data['target_rgb']
            observed = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
            y = observed@np.array([.2126, .7152, .0722])
            foreground = data['source_alpha'] > .5
            error = np.abs(data['render_linear']-observed)
            row = dict(frame=int(number))
            for name, mask in [('full', foreground), ('body', foreground & (y <= stats['q_base_value'])),
                               ('bright', foreground & (y >= stats['q_bright_value']))]:
                row[name] = dict(pixels=int(mask.sum()), linear_rgb_observed_mae=float(error[mask].mean()) if mask.any() else None)
            for field in ('base_linear', 'residual_linear', 'render_linear'):
                linear = np.maximum(data[field], 0)
                display = np.where(linear <= .0031308, 12.92*linear, 1.055*np.maximum(linear, 1e-10)**(1/2.4)-.055)
                legacy.save_png(out/f'{field}_{int(number):04d}.png', display)
                row[field+'_foreground_mean'] = float(data[field][foreground].mean())
            row['add_max_error'] = float(np.abs(data['base_linear']+data['residual_linear']-data['render_linear']).max())
            assert row['add_max_error'] < 2e-5
            rows.append(row)
        legacy.write(out/'metrics.json', dict(status='PASS_SEGMENTED_OBSERVED_RGB_EXECUTION', rows=rows,
            q_base=stats['q_base_value'], q_bright=stats['q_bright_value'], denominator='固定原Alpha前景，TRAIN冻结阈值',
            saturation_note='本表报告截断观测误差，不把饱和亮度解释为真实辐射度；训练另用逐通道删失',
            component_semantics='Stage2加权EL／DL，或Stage3原生PBR基础／额外小残差；不是Separation图'))
    legacy.make_plan, legacy.model_record, legacy.adapter, legacy.metrics = make_plan, model_record, adapter, metrics
    # 逐步执行也验证新adapter，不让原生协议哈希掩盖新增接口的变更。
    if '--output' in sys.argv:
        output = Path(sys.argv[sys.argv.index('--output')+1])
        if (output/'plan.json').exists():
            plan = legacy.read(output/'plan.json')
            if plan.get('three_stage_adapter_sha256') != legacy.sha(__file__):
                raise ValueError('三阶段评测adapter已变化，旧计划不能继续。')
    legacy.main()


if __name__ == '__main__':
    main()
