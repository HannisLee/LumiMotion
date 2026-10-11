#!/usr/bin/env python3
"""原生训练检查点必要重载：逐文件绑定、相机划分、零头及单相机几何map。"""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--load_iter', type=int, required=True)
    p.add_argument('--source_path', required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from argparse import Namespace
    import diff_surfel_rasterization as rasterizer
    from scripts.eval_geometry_gt import GeometryRun
    from gaussian_renderer import render
    from utils.geometry_eval_utils import unit_normal
    from goal_eval_protocol import CONTROL, read, write, digest, verify
    args.output_dir.mkdir(parents=True, exist_ok=False)
    verify(read(CONTROL / 'protocol.json'))
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    marker_path = run.model / 'static_dl_training.json'
    marker = read(marker_path)
    training = marker['rasterizer_provenance']
    effective = getattr(rasterizer, 'provenance', None)
    if not effective or effective['method'] != 'repository_lumimotion_cuda_rasterizer':
        raise AssertionError('必须通过bootstrap加载原生CUDA光栅。')
    for key in ('method', 'source_sha256', 'extension_sha256', 'source_files'):
        if training[key] != effective[key]:
            raise AssertionError('训练／评测实际光栅来源不一致：' + key)
    if digest(effective['extension_path']) != training['extension_sha256']:
        raise AssertionError('实际.so文件改变。')
    for relative, expected in effective['source_files'].items():
        if digest(ROOT / relative) != expected:
            raise AssertionError('原生CUDA源码改变：' + relative)
    train_source = Path(marker['args']['source_path'])
    camera_errors = []
    for split in ('train', 'test'):
        tr = read(train_source / f'transforms_{split}.json')['frames']
        ev = read(run.source / f'transforms_{split}.json')['frames']
        if [r['source_frame'] for r in tr] != [r['source_frame'] for r in ev]:
            raise AssertionError('训练／评测划分帧号不同。')
        for a, b in zip(tr, ev):
            np.testing.assert_allclose(a['transform_matrix'], b['transform_matrix'], atol=1e-5, rtol=0)
            if not np.isclose(a['time'], b['time'], atol=1e-7, rtol=0):
                raise AssertionError('训练／评测帧时间不同。')
            camera_errors.append(float(np.abs(np.asarray(a['transform_matrix'])-b['transform_matrix']).max()))
    heads = {}
    for name in ('gaussian_warp', 'gaussian_rotation', 'gaussian_scaling'):
        layer = getattr(run.deform.deform, name)
        heads[name] = dict(weight_nonzero=int(torch.count_nonzero(layer.weight)),
                           bias_nonzero=int(torch.count_nonzero(layer.bias)))
        if any(heads[name].values()):
            raise AssertionError('重载后几何头权重或偏置非零。')
    metadata = read(run.source / 'transforms_test.json')
    record = next(r for r in metadata['frames'] if r['source_frame'] == 10)
    camera, _, intrinsics, _ = run.camera(metadata, record, 2)
    cfg = dict(run.config, compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    with torch.no_grad():
        d = run.deformation(camera)
        maxima = {k: float(d[k].abs().max()) for k in ('d_xyz', 'd_rotation', 'd_scaling')}
        if any(maxima.values()):
            raise AssertionError('标准重载forward不是严格零形变。')
        prediction = run.render(camera, np.asarray(record['transform_matrix']), intrinsics)
        package = render(camera, run.pc, Namespace(**cfg), torch.zeros(3, device='cuda'),
            **{k: d[k] for k in ('d_xyz', 'd_rotation', 'd_scaling', 'd_opacity', 'd_color')})
    normal, valid = unit_normal(package['rend_normal'].permute(1, 2, 0).cpu().numpy())
    alpha = package['rend_alpha'].squeeze(0).cpu().numpy()
    surface = package['surf_depth'].squeeze(0).cpu().numpy()
    mixture = prediction['expected']*(1-cfg['depth_ratio'])+prediction['median']*cfg['depth_ratio']
    visible = prediction['alpha'] >= .5
    differences = dict(normal=float(np.max(np.abs(normal[valid]-prediction['normal'][valid]))),
                       alpha=float(np.max(np.abs(alpha-prediction['alpha']))),
                       depth=float(np.max(np.abs(surface[visible]-mixture[visible]))))
    if any(x > 2e-5 for x in differences.values()):
        raise AssertionError('真实SH渲染map与原GeometryRun不一致。')
    np.savez_compressed(args.output_dir / 'reload_arrays_0010.npz', **prediction,
                        native_normal=normal, native_alpha=alpha, native_surface_depth=surface)
    run.files[str(marker_path)] = digest(marker_path)
    report = dict(status='PASS', model=str(run.model), iteration=args.load_iter, source=str(run.source),
        scope='单相机接入；没有MAE或重建质量验收', training_source=str(train_source),
        camera_frames_checked=150, camera_max_abs=max(camera_errors), geometry_heads=heads,
        deformation_max_abs=maxima, geometry_map_max_abs=differences,
        training_rasterizer=training, evaluation_rasterizer=effective, sources=run.files)
    for path, expected in run.files.items():
        if digest(path) != expected:
            raise AssertionError('重载期间checkpoint来源改变。')
    write(args.output_dir / 'reload_audit.json', report)
    print('原生训练／评测重载PASS；三几何头严格0；同.so SHA256=' + effective['extension_sha256'], flush=True)


if __name__ == '__main__':
    main()
