"""固定TRAIN相机、遍历全部时间的只读稳定性渲染；不读取TEST/GT。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from scene.three_stage_native_pbr import load_three_stage_model
from scene.three_stage_pipeline import sha, write
from scripts.train_three_stage import install_guard, load_cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--view', type=int, choices=[1, 75, 149], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.model, args.output = args.model.resolve(), args.output.resolve()
    if not args.output.is_relative_to(root/'output'):
        raise ValueError('所有渲染必须进入output。')
    if args.smoke and not args.output.is_relative_to(root/'output/smoke_test'):
        raise ValueError('冒烟必须进入output/smoke_test。')
    marker = json.loads((args.model/'three_stage_training.json').read_text())
    if marker['smoke'] or marker['gt_supervision'] or marker['external_test_read']:
        raise ValueError('此诊断只接受正式无GT模型。')
    for filename, digest in marker['sources'].items():
        if sha(filename) != digest:
            raise ValueError('原冻结训练源码发生变化。')
        path = Path(filename)
        if not (path.parent.name == 'scripts' and path.name.startswith('train_three_stage')):
            if sha(root/path.parent.name/path.name) != digest:
                raise ValueError('当前实际前向与冻结训练来源不同。')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'launch.json', dict(argv=sys.argv, command=shlex.join([sys.executable, '-m', 'scripts.render_three_stage_fixed_view', *sys.argv[1:]]),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), code_sha256=sha(__file__), gt_read=False, test_read=False))
    binding = marker['binding']
    install_guard(binding['source'], args.output, args.model, binding)
    torch.set_num_threads(4)
    model = load_three_stage_model(args.model, args.iteration)
    camera = next(c for c in load_cameras(binding['source']) if int(c.image_name_train_light.split('_')[-1]) == args.view)
    camera.load2device()
    # 不读取相应时间的目标图像；唯一已有TRAIN相机RGB/Alpha也全部置零。
    camera.original_image_train_light.zero_()
    camera.gt_alpha_mask.zero_()
    frames = [1, 75, 150] if args.smoke else list(range(1, 151))
    writers = {}
    rows = []
    try:
        for kind in ['rgb', 'alpha', 'normal', 'separation']:
            (args.output/kind).mkdir()
            writers[kind] = imageio.get_writer(str(args.output/(kind+'.mp4')), fps=24, macro_block_size=1, codec='libx264')
        with torch.no_grad():
            from gaussian_renderer import render
            for number in frames:
                camera.fid.fill_((number-1)/149.)
                camera.image_name_train_light = f'image_{number:04d}'
                package = model.render(camera, args.iteration, components=True)
                feature = model.pc.get_binary_feature()
                colors = torch.cat((feature, 1-feature, torch.zeros_like(feature)), dim=1)
                effective = model.effective(camera)
                sep = render(camera, effective, model.pipe, effective.get_xyz.new_zeros(3), 0., 0., 0., override_color=colors)['render']
                alpha = package['rend_alpha']
                normal = torch.nn.functional.normalize(package['rend_normal'], dim=0)
                arrays = dict(rgb=package['render'], alpha=alpha.expand(3, -1, -1), normal=(normal*.5+.5)*alpha, separation=sep)
                for kind, value in arrays.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError('时序渲染含非有限数。')
                    pixels = (value.clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
                    if not cv2.imwrite(str(args.output/kind/f'{number:04d}.png'), pixels[..., ::-1]):
                        raise IOError('时序PNG写入失败。')
                    writers[kind].append_data(pixels)
                error = float(package['component_add_max_error'])
                if error >= 2e-5:
                    raise AssertionError('时序线性分量加和不一致。')
                rows.append(dict(frame=number, time=(number-1)/149., add_max_error=error,
                    alpha_pixels=int((alpha>.5).sum()), residual_linear_mean=float(package['residual_linear'].mean())))
                if number % 10 == 0:
                    print('固定视角时序', args.view, number, flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    videos = {}
    for kind in writers:
        path = args.output/(kind+'.mp4')
        cap = cv2.VideoCapture(str(path)); count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        if count != len(frames):
            raise ValueError('固定视角视频帧数不完整。')
        videos[kind] = dict(path=str(path), decoded_frames=count, sha256=sha(path))
    write(args.output/'receipt.json', dict(status='PASS_FIXED_VIEW_RENDER_EXECUTION', model=str(args.model), iteration=args.iteration,
        source=binding['source'], fixed_train_camera=args.view, frames=frames, videos=videos, rows=rows,
        checkpoint_sha256=sha(args.model/f'three_stage/iteration_{args.iteration}/state.pt'), code_sha256=sha(__file__),
        optimizer_updates=0, gt_read=False, test_read=False, inference_target_rgb_alpha_zero=True, manual_review='PENDING'))
    (args.output/'README.md').write_text('# 固定TRAIN视角时序只读诊断\n\n'
        f'source：`{binding["source"]}`；model：`{args.model}`；iteration={args.iteration}；固定TRAIN相机{args.view}，时间覆盖{frames}。\n\n'
        '完整命令与实际GPU见launch.json；日志在上级render.log，四类PNG和视频在当前目录；receipt.json验证解码帧数与checkpoint SHA。\n\n'
        '已有TRAIN目标RGB/Alpha置零，时间150只使用TRAIN方向插值/端点保持；不读取TEST目标、GT灯向或几何，不进行任何optimizer更新。\n\n'
        '**PASS（只读渲染执行）**；RGB/Alpha/Normal/Separation实际目检PENDING，不能仅凭执行通过认定时序稳定。\n')


if __name__ == '__main__':
    main()
