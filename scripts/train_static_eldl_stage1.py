"""静态ELDL-V3 S0：随机初始化原生LumiMotion Stage1，严格共享静态几何与颜色（117 TRAIN）。

与历史B00（native、135 TRAIN、时变MLP全开）的差别：
1. 只加载 data_fit_v2 的117 TRAIN，DEV18/TEST15 训练全程不可见；
2. 训练开始前即把形变网络的三个几何输出头置零并 requires_grad=False（d_xyz/d_rotation/d_scaling 恒为0）；
3. pred_color=False，不构造时变颜色头，d_color=None（无逐帧阴影调制/动态颜色）；
其余（原生Trainer、loss、增密、reset、学习率、SH日程、seed）与B00命令一致。
原生Trainer在保存iteration k时尚未执行第k步optimizer更新：checkpoint k 含 k-1 次真实更新；本入口逐次计数并落档。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts.train_stage1 import Trainer
from scripts.train_static_dl import freeze_geometry_heads
from utils.general_utils import safe_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_source(source, model, split_path):
    manifest = json.loads((source / 'no_gt_manifest.json').read_text())
    split = json.loads(split_path.read_text())
    assert manifest['gt_read'] is False and manifest['init_weights_loaded'] is False
    assert manifest['dataset'] == split['dataset'] == 'LH-static-jumpingjacks-ELDL-V3'
    assert set(p.name for p in source.iterdir()) == {'images', 'points3d.ply', 'transforms_train.json', 'no_gt_manifest.json'}
    for relative, expected in manifest['files'].items():
        path = source / relative
        assert path.resolve().is_relative_to(source)
        assert sha(path) == expected, relative
    actual_images = {str(p.relative_to(source)) for p in (source / 'images').iterdir()}
    assert actual_images == {p for p in manifest['files'] if p.startswith('images/')}
    frames = [f['source_frame'] for f in json.loads((source / 'transforms_train.json').read_text())['frames']]
    assert frames == manifest['train_frames'] == split['train_frames'] and len(frames) == 117
    assert not set(frames) & (set(split['dev_frames']) | set(split['test_frames']))
    assert not model.exists(), '已有模型目录禁止覆盖'
    allowed = [source, model, split_path.parent]

    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if any(path.is_relative_to(a) for a in allowed):
            return
        if path.suffix.lower() in {'.ply', '.pth', '.pt', '.exr', '.blend', '.png', '.jpg', '.jpeg', '.npz'}:
            raise PermissionError('S0训练禁止读取外部模型/图像/GT：' + str(path))
        if path.name in {'lights.json', 'object_pose.json', 'camera.json', 'bone_pose.json'}:
            raise PermissionError('S0训练禁止读取受限真值：' + str(path))
    sys.addaudithook(guard)
    return manifest, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--save_iterations', type=int, nargs='+', default=[10000, 20000, 35000])
    parser.add_argument('--deform-type', dest='deform_type', default='mlp')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.pred_color = False  # 关闭时变颜色自由度（ModelParams默认True且store_true无法从命令行关闭）
    source, model = Path(args.source_path).resolve(), Path(args.model_path).resolve()
    if not args.eval or args.deform_type != 'mlp':
        parser.error('必须eval隔离划分，且使用mlp检查点格式（几何头置零冻结）')
    if 'output' not in model.parts:
        parser.error('模型必须在output内')
    if args.smoke:
        if 'smoke_test' not in model.parts or args.iterations > 200:
            parser.error('冒烟只能在smoke_test，最多200步')
    elif args.iterations != 35000:
        parser.error('S0正式预算固定为名义35000')
    manifest, split = validate_source(source, model, args.split.resolve())
    safe_state(args.quiet)
    import diff_surfel_rasterization as backend
    assert backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer'
    saves = sorted(set(args.save_iterations + [args.iterations]))
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
                      testing_iterations=[], saving_iterations=saves)
    net = trainer.deform.deform
    assert trainer.deform.name == 'mlp' and net.pred_color is False and not hasattr(net, 'gaussian_color')
    heads = freeze_geometry_heads(net)
    assert all(not p.requires_grad for n in heads for p in getattr(net, n).parameters())
    assert not trainer.scene.getTestCameras()
    cams = trainer.scene.getTrainCameras()
    assert len(cams) == 117 and sorted(int(c.image_name_train_light.split('_')[-1]) for c in cams) == split['train_frames']
    first = cams[0]
    assert (first.gt_alpha_mask == 0).any() and (first.gt_alpha_mask == 1).any()
    assert hasattr(first, 'observed_saturation_mask'), '缺少缩小前满码通道记录'
    init_xyz_sha = tensor_sha(trainer.gaussians.get_xyz)
    deform_init = {k: tensor_sha(v) for k, v in net.state_dict().items()}

    # 真实optimizer更新计数与每个保存点的更新数
    counter = dict(gaussian=0, deform=0)
    gstep, dstep = trainer.gaussians.optimizer.step, trainer.deform.optimizer.step

    def gaussian_step(*a, **k):
        counter['gaussian'] += 1
        return gstep(*a, **k)

    def deform_step(*a, **k):
        counter['deform'] += 1
        return dstep(*a, **k)
    trainer.gaussians.optimizer.step = gaussian_step
    trainer.deform.optimizer.step = deform_step
    save_log = []
    scene_save = trainer.scene.save

    def logged_save(iteration):
        scene_save(iteration)
        save_log.append(dict(iteration=iteration, optimizer_updates_applied=counter['gaussian'],
                             deform_updates_applied=counter['deform'], points=int(trainer.gaussians.get_xyz.shape[0]),
                             time=time.strftime('%Y-%m-%dT%H:%M:%S')))
        (model / 'save_log.json').write_text(json.dumps(save_log, ensure_ascii=False, indent=2))
    trainer.scene.save = logged_save

    # 抽查：形变输出恒为0、无时变颜色
    zero_checks = []
    deform_call = trainer.deform.step

    def checked_step(*a, **k):
        out = deform_call(*a, **k)
        if trainer.iteration % 1000 == 1 or args.smoke:
            vals = dict(d_xyz=float(out['d_xyz'].abs().max()), d_rotation=float(out['d_rotation'].abs().max()),
                        d_scaling=float(out['d_scaling'].abs().max()), d_color_is_none=out['d_color'] is None)
            assert vals['d_xyz'] == 0 and vals['d_rotation'] == 0 and vals['d_scaling'] == 0 and vals['d_color_is_none']
            zero_checks.append(dict(iteration=trainer.iteration, **vals))
        return out
    trainer.deform.step = checked_step

    record = dict(status='RUNNING', stage='S0静态原生Stage1', args={k: str(v) for k, v in vars(args).items()},
                  gt_supervision=False, init_weights_loaded=False, train_frames=split['train_frames'],
                  dev_frames_hidden=split['dev_frames'], test_frames_hidden=split['test_frames'],
                  input_manifest_sha256=sha(source / 'no_gt_manifest.json'), split_sha256=sha(args.split),
                  initial_point_cloud_sha256=sha(source / 'points3d.ply'), initial_xyz_tensor_sha256=init_xyz_sha,
                  rasterizer_provenance=backend.provenance, static_config=dict(
                      geometry_heads_zero_frozen=list(heads), pred_color=False, d_color='None',
                      note='训练前冻结；不是训练后截取某一帧'),
                  cameras_extent=float(trainer.scene.cameras_extent),
                  source_code_root=str(Path(__file__).resolve().parents[1]),
                  update_semantics='原生Trainer：iteration k先保存再optimizer.step，且iteration==iterations时不更新；checkpoint k含k-1次更新')
    (model / 'static_eldl_stage1.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    trainer.train(args.iterations)
    deform_final = {k: tensor_sha(v) for k, v in net.state_dict().items()}
    record.update(status='TRAINING_FINISHED_PENDING_EVALUATION', iterations=args.iterations,
                  actual_optimizer_updates=counter['gaussian'], actual_deform_optimizer_updates=counter['deform'],
                  saves=save_log, zero_output_checks=zero_checks,
                  geometry_heads_final_all_zero=all(float(getattr(net, n).weight.abs().max()) == 0 and
                                                     float(getattr(net, n).bias.abs().max()) == 0 for n in heads),
                  deform_state_unchanged=deform_init == deform_final,
                  final_points=int(trainer.gaussians.get_xyz.shape[0]),
                  checkpoint_sha256={str(p.relative_to(model)): sha(p) for p in sorted(model.rglob('*'))
                                     if p.is_file() and p.suffix in {'.ply', '.pth'}})
    (model / 'static_eldl_stage1.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (model / 'training_finished.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
