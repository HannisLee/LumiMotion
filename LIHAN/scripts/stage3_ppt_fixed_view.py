"""固定TRAIN相机、改变真实模型时间；无GT目标缓冲，不为未观测视角编造分数。"""
import argparse
import copy
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from gaussian_renderer import render
from scripts.render_stage2 import Stage2RenderRun, save_image, to_image
from scene.stage2_photometric import frame_key
from scripts.stage2_runtime import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--view', default='0149')
    parser.add_argument('--frames', type=int, default=150)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = Stage2RenderRun(args.model, args.iteration)
    ctx = run.context
    anchor = ctx.camera_by_key[args.view]
    assert args.view in ctx.training_keys
    ctx.load_camera(anchor)
    camera = copy.copy(anchor)
    camera.original_image_train_light = torch.zeros_like(anchor.original_image_train_light)
    camera.gt_alpha_mask = torch.zeros_like(anchor.gt_alpha_mask)
    names = ['rgb', 'alpha', 'normal', 'separation', 'albedo', 'roughness']
    writers = {name: imageio.get_writer(str(args.output/(name+'.mp4')), fps=25, macro_block_size=1) for name in names}
    for name in names:
        (args.output/name).mkdir()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    frames = np.rint(np.linspace(1, 150, args.frames)).astype(int)
    checksums = dict(ctx.source_files)
    saved = []
    previous = None
    try:
        with torch.no_grad():
            for index in frames:
                camera.fid = torch.tensor([index/150], device='cuda', dtype=torch.float32)
                camera.image_name_train_light = f'frame_{index:04d}.png'
                package = run.render(camera)
                alpha = package['rend_alpha']
                normal = torch.nn.functional.normalize(package['rend_normal'], dim=0)
                deformation = ctx.deformation(camera)
                feature = ctx.pc.get_binary_feature().reshape(-1, 1)
                colors = torch.cat([1-feature, feature, torch.zeros_like(feature)], dim=1)
                separation = render(camera, ctx.pc, run.args, ctx.background, override_color=colors,
                                    **{k: deformation[k] for k in ['d_xyz', 'd_rotation', 'd_scaling', 'd_opacity']})['render']
                images = dict(rgb=package['render'], alpha=alpha.expand(3, -1, -1), normal=(normal*.5+.5)*alpha,
                              separation=separation, albedo=package['base_color'], roughness=package['roughness'].expand(3, -1, -1))
                for name, tensor in images.items():
                    assert torch.isfinite(tensor).all()
                    writers[name].append_data(to_image(tensor))
                    if int(index) in [1, 75, 150] or args.frames != 150:
                        save_image(args.output/name/f'{index:04d}.png', tensor)
                current = package['render'].detach().clamp(0, 1).cpu().numpy()
                saved.append(dict(frame=int(index), time=float(camera.fid.item()),
                                  rgb_mean_abs_change_from_previous=None if previous is None else float(np.abs(current-previous).mean())))
                previous = current
                print(f'固定相机{args.view}，时间{index}/150', flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    assert all(file_sha256(Path(p)) == value for p, value in checksums.items())
    decoded = {}
    for name in names:
        reader = imageio.get_reader(str(args.output/(name+'.mp4')))
        try:
            count = sum(1 for image in reader if image.shape == (400, 400, 3))
        finally:
            reader.close()
        assert count == len(frames)
        decoded[name] = count
    record = dict(status='PASS_ENGINEERING_ONLY', model=str(args.model), iteration=args.iteration, fixed_train_view=args.view,
                  original_camera_center=anchor.camera_center.cpu().tolist(), source=run.args.source_path,
                  world_view_transform=anchor.world_view_transform.cpu().tolist(), full_proj_transform=anchor.full_proj_transform.cpu().tolist(),
                  frames=saved, full_video_decoded_frames=decoded, target_buffers_zeroed=True, metrics=None,
                  interpretation='相机内外参固定，形变按frame/150变化；方向PBR使用对应时刻的估计光，环境PBR沿原模型环境光；无该固定视角逐时GT，不计算PSNR/SSIM',
                  checkpoint_sha256=checksums, checkpoint_unchanged=True, manual_review='PENDING')
    (args.output/'receipt.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
