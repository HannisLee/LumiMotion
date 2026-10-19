"""仅用TRAIN比较固定PBR状态的半球积分精度，不训练或读取GT。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

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
    if record['stage'] != 2 or record['gt_supervision'] is not False:
        raise ValueError('只允许本轮Stage2父资产。')
    binding = record['binding']; config = dict(record['config'], direction_lr=0., lambda_residual=0.)
    install_guard(binding['source'], a.output, a.parent, binding)
    torch.set_num_threads(4)
    model = ThreeStageModel(a.parent, a.iteration, binding, 3, config)
    cameras = load_cameras(binding['source'])
    rows = []
    with torch.no_grad():
        for i in (0, len(cameras)//2, -1):
            camera = cameras[i]; camera.load2device()
            arrays = {}; foreground = camera.gt_alpha_mask[0] > .5
            for samples in (16, 128, 256):
                model.pipe.diffuse_sample_num = samples
                package = model.render(camera, a.iteration, components=True, training=False)
                if not torch.isfinite(package['render_linear']).all():
                    raise FloatingPointError('PBR积分非有限。')
                arrays[samples] = package['render_linear'].clone()
                rgb = package['render'].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                cv2.imwrite(str(a.output/f'{camera.image_name_train_light}_{samples}.png'), (rgb[..., ::-1]*255).round().astype(np.uint8))
            for samples in (16, 128):
                difference = (arrays[samples]-arrays[256]).abs()[:, foreground]
                rows.append(dict(frame=camera.image_name_train_light, samples=samples, reference_samples=256,
                    foreground_pixels=int(foreground.sum()), linear_mae_to_reference=float(difference.mean()),
                    linear_p95_to_reference=float(torch.quantile(difference.flatten(), .95)),
                    reference_foreground_mean=float(arrays[256][:, foreground].mean())))
            camera.load2device('cpu')
    write(a.output/'receipt.json', dict(status='PASS_TRAIN_PBR_INTEGRATION_AUDIT', parent=str(a.parent),
        parent_iteration=a.iteration, parent_checkpoint_sha256=sha(a.parent/f'three_stage/iteration_{a.iteration}/state.pt'),
        samples=[16, 128, 256], source=binding['source'], rows=rows,
        no_optimizer_updates=True, same_material_geometry_environment=True, gt_read=False, external_test_read=False,
        purpose='仅衡量固定状态数值积分误差；256是数值参考，不是GT或最终精度结论。'))
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
