"""动态V3原生Stage1隔离入口；保持原生Trainer的损失、增密与优化日程。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts.train_stage1 import Trainer
from utils.general_utils import safe_state


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--save_iterations', type=int, nargs='+', default=[10000,20000,30000,35000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    dataset = lp.extract(args)
    args.model_path = dataset.model_path
    source, model = Path(dataset.source_path).resolve(), Path(dataset.model_path).resolve()
    assert not model.exists(), '禁止在已有模型目录重新启动'
    assert args.eval and args.is_blender and args.deform_type == 'mlp'
    assert args.depth_ratio == 1 and args.resolution == 2
    if args.smoke:
        assert 'smoke_test' in model.parts and 1 < args.iterations <= 50
    else:
        assert args.iterations == 35000 and 'smoke_test' not in model.parts
    manifest = json.loads((source/'no_gt_manifest.json').read_text())
    assert manifest['dataset'] == 'LH-dynamic-jumpingjacks-ELDL-V3'
    assert not manifest['gt_read'] and not manifest['init_weights_loaded']
    assert set(p.name for p in source.iterdir()) == {'images','transforms_train.json','points3d.ply','no_gt_manifest.json'}
    for relative, digest in manifest['files'].items():
        path = source/relative
        assert path.resolve().is_relative_to(source) and sha(path) == digest
    records = json.loads((source/'transforms_train.json').read_text())['frames']
    assert [f['source_frame'] for f in records] == manifest['train_frames']
    assert not set(manifest['train_frames']) & set(manifest['dev_frames']+manifest['test_frames'])
    # 常规模型依赖已在导入时装载；运行时禁止读取外部权重、图像与真值。
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([source, model])
    safe_state(args.quiet)
    import diff_surfel_rasterization as backend
    assert backend.provenance['method'] == 'repository_lumimotion_cuda_rasterizer'
    trainer = Trainer(args=args, dataset=dataset, opt=op.extract(args), pipe=pp.extract(args),
                      testing_iterations=[], saving_iterations=sorted(set(args.save_iterations+[args.iterations])))
    assert trainer.scene.loaded_iter is None and trainer.iteration == 1
    assert not trainer.scene.getTestCameras() and len(trainer.scene.getTrainCameras()) == 117
    positions = {Path(f['file_path']).stem: torch.tensor(f['transform_matrix'])[:3,3] for f in records}
    camera_error = max(float((c.camera_center.cpu()-positions[c.image_name_train_light]).abs().max()) for c in trainer.scene.getTrainCameras())
    assert camera_error < 1e-4
    marker = dict(status='RUNNING', dataset=manifest['dataset'], args=vars(args),
        source=str(source), model=str(model), input_manifest_sha256=sha(source/'no_gt_manifest.json'),
        native_trainer_sha256=sha(Path(__import__('scripts.train_stage1', fromlist=['__file__']).__file__)),
        train_frames=manifest['train_frames'], dev_frames=manifest['dev_frames'], test_frames=manifest['test_frames'],
        weights_loaded=False, gt_supervision=False, dev_test_images_loaded=False,
        camera_center_max_abs_error=camera_error, rasterizer=backend.provenance,
        actual_optimizer_updates=args.iterations-1,
        native_schedule_note='原生终步保存但不optimizer.step，保持此语义；无PS/Lambertian/提前法线正则')
    (model/'dynamic_v3_training.json').write_text(json.dumps(marker, ensure_ascii=False, indent=2))
    started = time.monotonic()
    audit_path = model/'train_audit.jsonl'
    for _ in range(args.iterations):
        iteration = trainer.iteration
        trainer.train_step()
        assert math.isfinite(trainer.ema_loss_for_log), '损失非有限'
        if iteration == 1 or iteration % 500 == 0 or iteration == args.iterations:
            row = dict(iteration=iteration, elapsed_seconds=time.monotonic()-started,
                       loss_ema=trainer.ema_loss_for_log, gaussians=len(trainer.gaussians.get_xyz),
                       xyz_finite=bool(torch.isfinite(trainer.gaussians.get_xyz).all()),
                       deform_finite=all(bool(torch.isfinite(v).all()) for v in trainer.deform.deform.state_dict().values()),
                       gpu_memory_peak_bytes=torch.cuda.max_memory_allocated(),
                       gaussian_lrs={g['name']:g['lr'] for g in trainer.gaussians.optimizer.param_groups},
                       deformation_lrs={g['name']:g['lr'] for g in trainer.deform.optimizer.param_groups})
            assert row['xyz_finite'] and row['deform_finite']
            with audit_path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # 原生PLY与形变权重保留；另外归档优化器和RNG以供明确的后续恢复实现使用。
    torch.save(dict(gaussian_optimizer=trainer.gaussians.optimizer.state_dict(),
                    deformation_optimizer=trainer.deform.optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                    next_iteration=trainer.iteration, actual_updates=args.iterations-1), model/'optimizer_rng_terminal.pt')
    marker.update(status='TRAINING_FINISHED_PENDING_RELOAD_AND_EVALUATION',
                  checkpoint_sha256={str(p.relative_to(model)):sha(p) for p in
                    [model/f'point_cloud/iteration_{args.iterations}/point_cloud.ply',
                     model/f'deform/iteration_{args.iterations}/deform.pth',model/'cfg_args']})
    (model/'dynamic_v3_training.json').write_text(json.dumps(marker, ensure_ascii=False, indent=2))
    (model/'training_finished.json').write_text(json.dumps(marker, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
