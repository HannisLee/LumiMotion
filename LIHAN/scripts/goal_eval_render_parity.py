#!/usr/bin/env python3
"""同物理checkpoint和训练camera，核验原训练函数与评测适配器fresh/fresh输出。"""
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
    p.add_argument('--frame', type=int, default=1)
    p.add_argument('--output_dir', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    from argparse import Namespace
    from scripts.eval_geometry_gt import GeometryRun
    from scene.static_dl_photometric import DirectionOnlyLights, StaticVisibility, render_photometric
    from goal_eval_render import physical_renderer
    from goal_eval_protocol import CONTROL, read, write, digest
    from utils.geometry_eval_utils import unit_normal
    import diff_surfel_rasterization as effective_rasterizer
    effective_backend = getattr(effective_rasterizer, 'provenance', None)
    if not effective_backend or effective_backend.get('method') != 'repository_lumimotion_cuda_rasterizer':
        raise AssertionError('物理parity必须经bootstrap绑定原生CUDA光栅器。')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = GeometryRun(args.model_path, args.load_iter, args.source_path)
    metadata = read(run.source / 'transforms_train.json')
    record = next(r for r in metadata['frames'] if r['source_frame'] == args.frame)
    camera, _, intrinsics, _ = run.camera(metadata, record, 2)
    config = read(run.model / 'static_dl_training.json')['args']
    lighting = read(run.model / 'static_dl_lighting' / f'iteration_{args.load_iter}' / 'lighting.json')
    reference_lights = DirectionOnlyLights(config['static_dl_directions'])
    with torch.no_grad():
        reference_lights.log_intensity.copy_(torch.as_tensor(lighting['intensity_from_train_rgb'], device='cuda').log())
    reference_visibility = (StaticVisibility(config['static_dl_shadow_bias'], config['static_dl_bvh_interval'])
        if config['static_dl_mode'] == 'lambertian_visibility' and args.load_iter >= config['static_dl_visibility_start'] else None)
    wrapper, provenance = physical_renderer(run, args.load_iter, CONTROL / 'eval_directions_only_single.json')
    if wrapper is None:
        raise AssertionError('该检查点尚未进入物理模式。')
    with torch.no_grad():
        reference = render_photometric(camera, run.pc, Namespace(**config), torch.zeros(3, device='cuda'),
                                       reference_lights, reference_visibility, args.load_iter)
        actual = wrapper(camera)
        geometry = run.render(camera, np.asarray(record['transform_matrix']), intrinsics)
        deformation = run.deformation(camera)
    checks = {}
    arrays = {}
    for key in ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth', 'light_intensity', 'mean_visibility'):
        a, b = reference[key].cpu().numpy(), actual[key].cpu().numpy()
        arrays['reference_' + key] = a
        arrays['wrapper_' + key] = b
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('重载出现非有限值：' + key)
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        checks[key] = dict(max_abs=float(np.max(np.abs(a-b))), passed=True)
    normal, valid = unit_normal(actual['rend_normal'].permute(1, 2, 0).cpu().numpy())
    np.testing.assert_allclose(normal[valid], geometry['normal'][valid], atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(actual['rend_alpha'].squeeze(0).cpu().numpy(), geometry['alpha'], atol=2e-5, rtol=1e-5)
    mask = geometry['alpha'] >= .5
    mixture = geometry['expected']*(1-config['depth_ratio'])+geometry['median']*config['depth_ratio']
    np.testing.assert_allclose(actual['surf_depth'].squeeze(0).cpu().numpy()[mask], mixture[mask], atol=2e-5, rtol=1e-5)
    for key in ('d_xyz', 'd_rotation', 'd_scaling'):
        if torch.count_nonzero(deformation[key]):
            raise AssertionError('重载几何输出头不为零：' + key)
    arrays.update(geometry)
    np.savez_compressed(args.output_dir / 'render_parity_arrays.npz', **arrays)
    report = dict(status='PASS', scope='保存checkpoint重载后，原训练函数vs评测wrapper，同训练camera、fresh/fresh BVH；非旧缓存图。',
        frame=args.frame, model=str(run.model), iteration=args.load_iter, checks=checks,
        geometry_maps='PASS：原GeometryRun world normal/alpha/mixed depth一致；三几何头严格0',
        physical_provenance=provenance, effective_rasterizer=effective_backend, sources=run.files)
    write(args.output_dir / 'parity.json', report)
    print('物理重载一致性PASS：' + str(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
