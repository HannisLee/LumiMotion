"""固定TRAIN状态核验原生PBR的Expected／Median追踪起点语义。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scene.three_stage_pipeline import ThreeStageModel, sha, write
from scripts.train_three_stage import load_cameras, install_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.parent = a.parent.resolve(); a.output = a.output.resolve()
    root = Path(__file__).resolve().parents[1]
    if not a.output.is_relative_to(root/'output'):
        raise ValueError('输出必须位于output。')
    a.output.mkdir(parents=True, exist_ok=False)
    record = json.loads((a.parent/'three_stage_training.json').read_text())
    binding = record['binding']; config = dict(record['config'], direction_lr=0., lambda_residual=0., pbr_samples=256)
    install_guard(binding['source'], a.output, a.parent, binding)
    torch.set_num_threads(4)
    model = ThreeStageModel(a.parent, a.iteration, binding, 3, config)
    cameras = load_cameras(binding['source']); rows = []
    with torch.no_grad():
        for i in (0, len(cameras)//2, -1):
            camera = cameras[i]; camera.load2device(); packages = {}
            for ratio in (0., 1.):
                model.pipe.depth_ratio = ratio
                package = model.render(camera, a.iteration, components=True, training=False)
                packages[ratio] = package
                rgb = package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                assert cv2.imwrite(str(a.output/f'{camera.image_name_train_light}_ratio{int(ratio)}.png'), (rgb[..., ::-1]*255).round().astype(np.uint8))
            expected, native = packages[0.], packages[1.]
            fg = (camera.gt_alpha_mask.cuda() > .99) & (native['rend_alpha'] > .99)
            normal = F.normalize(native['rend_normal'], dim=0)
            upward = fg & (normal[2:3] > .9)
            distance = (native['surf_depth']-expected['surf_depth']).abs()*camera.rays_d_hw_unnormalized.norm(dim=-1)[None]
            row = dict(frame=camera.image_name_train_light,
                alpha_max_abs=float((native['rend_alpha']-expected['rend_alpha']).abs().max()),
                world_point_shift_mean=float(distance[fg].mean()), world_point_shift_p95=float(torch.quantile(distance[fg], .95)),
                linear_rgb_difference=float((native['render_linear']-expected['render_linear']).abs()[fg.expand(3, -1, -1)].mean()))
            for name, mask in [('foreground', fg), ('predicted_upward_surface', upward)]:
                row[name] = dict(pixels=int(mask.sum()))
                for ratio, package in packages.items():
                    visibility = package['visibility']/package['rend_alpha'].clamp_min(1e-6)
                    row[name][f'visibility_ratio{int(ratio)}'] = float(visibility[mask].mean()) if mask.any() else None
            rows.append(row); camera.load2device('cpu')
    write(a.output/'receipt.json', dict(status='PASS_NATIVE_PBR_POINT_DEPTH_AUDIT', rows=rows,
        parent=str(a.parent), parent_iteration=a.iteration,
        checkpoint_sha256=sha(a.parent/f'three_stage/iteration_{a.iteration}/state.pt'),
        compared=dict(stage2_inherited=0., native_pipeline_default=1., native_effective_median_mix=.5),
        same_geometry_material_environment=True, no_optimizer_updates=True, gt_read=False, test_read=False,
        region_note='上向区域仅由预测GS法线及原Alpha选择，不是GT平台分区。'))
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
