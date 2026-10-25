"""统一浮点 PSNR/SSIM、前景 PSNR 和四类诊断图；禁止按 TEST 选择参数。"""
import argparse
import copy
import json
import math
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from gaussian_renderer import render
from scripts.render_stage2 import Stage2RenderRun, save_image, to_image
from scripts.stage2_runtime import FrozenScene, supervision, file_sha256
from scene.stage2_photometric import frame_key
from utils.loss_utils import ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'train', 'all'], default='test')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = Stage2RenderRun(args.model, args.iteration)
    original_context = run.context
    run.args.source_path = args.source
    run.context = FrozenScene(run.args, args.model, args.iteration)
    del original_context
    ctx = run.context
    cameras = ctx.test_cameras if args.split == 'test' else (ctx.train_cameras if args.split == 'train' else ctx.train_cameras + ctx.test_cameras)
    if args.split == 'train':
        cameras = [cameras[i] for i in [0, len(cameras)//2, len(cameras)-1]]
    cameras = sorted(cameras, key=lambda c: float(c.fid.item()))
    names = ['rgb', 'gt', 'alpha', 'normal', 'separation', 'albedo', 'roughness', 'error']
    for name in names + ['arrays']:
        (args.output / name).mkdir()
    writers = {name: imageio.get_writer(str(args.output / (name + '.mp4')), fps=10 if args.split == 'test' else 25, macro_block_size=1) for name in names}
    hashes_before = dict(ctx.source_files)
    rows = []
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    try:
        with torch.no_grad():
            for cam in cameras:
                package = run.render(cam)
                target, mask = supervision(cam, ctx.background)
                image = package['render'].clamp(0, 1)
                mse = (image-target).square()
                assert torch.isfinite(image).all()
                key = frame_key(cam.image_name_train_light)
                alpha = package['rend_alpha']
                normal = torch.nn.functional.normalize(package['rend_normal'], dim=0)
                deformation = ctx.deformation(cam)
                feature = ctx.pc.get_binary_feature().reshape(-1, 1)
                colors = torch.cat([1-feature, feature, torch.zeros_like(feature)], dim=1)
                separation = render(cam, ctx.pc, run.args, ctx.background, override_color=colors,
                                    **{k: deformation[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity']})['render']
                images = dict(rgb=image, gt=target, alpha=alpha.expand(3, -1, -1), normal=(normal*.5+.5)*alpha,
                              separation=separation, albedo=package['base_color'], roughness=package['roughness'].expand(3, -1, -1),
                              error=(image-target).abs().mean(0, keepdim=True).expand(3, -1, -1)*5)
                for name, tensor in images.items():
                    save_image(args.output / name / (key+'.png'), tensor)
                    writers[name].append_data(to_image(tensor))
                array = dict(rgb=image.cpu().numpy(), target=target.cpu().numpy(), mask=mask.cpu().numpy(),
                             normal=normal.cpu().numpy(), alpha=alpha.cpu().numpy())
                np.savez_compressed(args.output / 'arrays' / (key+'.npz'), **array)
                row = dict(frame=key, time=float(cam.fid.item()), psnr=-10*math.log10(max(float(mse.mean()), 1e-12)),
                           ssim=float(ssim(image, target)),
                           psnr_foreground=-10*math.log10(max(float((mse*mask).sum()/(3*mask.sum())), 1e-12)))
                # 以保存的浮点数组在 CPU 上独立复算，不以 PNG 近似替代。
                independent = -10*np.log10(max(float(np.mean((array['rgb'].astype(np.float64)-array['target'])**2)), 1e-12))
                assert abs(independent-row['psnr']) < 1e-4
                row['psnr_cpu_recomputed'] = independent
                rows.append(row)
                print(json.dumps(row), flush=True)
                ctx.unload_camera(cam)
    finally:
        for writer in writers.values():
            writer.close()
    assert all(file_sha256(Path(p)) == sha for p, sha in hashes_before.items())
    if args.split == 'test':
        assert sorted(int(r['frame']) for r in rows) == list(range(10, 151, 10))
    report = dict(model=str(args.model), iteration=args.iteration, source=args.source, split=args.split,
                  frames=rows, metrics={k: float(np.mean([r[k] for r in rows])) for k in ['psnr', 'ssim', 'psnr_foreground']},
                  protocol='400×400、黑底完整浮点 sRGB clamp[0,1]；逐帧 PSNR/SSIM 后均值；项目 11×11 sigma1.5 SSIM；前景另列；不使用 TEST 调参',
                  checkpoint_sha256=hashes_before, checkpoint_unchanged=True,
                  separation_semantics='红色静态、绿色动态；no_binary_separation=True 的全绿色是定义，不是语义分离成功',
                  manual_review='PENDING')
    (args.output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(report['metrics'], flush=True)


if __name__ == '__main__':
    main()
