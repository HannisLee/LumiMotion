"""仅用Stage2优化视角的冻结Stage1预测，准备邻域图的时间稳定掩码。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.stage2_runtime import load_photometric_run, file_sha256
from scene.stage2_photometric import frame_key
from utils.general_utils import build_rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    context, _, _, config, _ = load_photometric_run(args.model_path, args.iteration)
    if context.pc.stage2_geometry is None:
        raise ValueError('需要具有已记录内部训练划分的几何Stage2检查点。')
    pc = context.pc
    extent = pc.stage2_geometry.scene_extent
    keys = []
    move = torch.zeros(len(pc.get_xyz), device='cuda')
    angle = torch.zeros_like(move)
    with torch.no_grad():
        for i, camera in enumerate(context.train_cameras):
            if i % config.geometry_validation_stride == 0:
                continue
            keys.append(frame_key(camera.image_name_train_light))
            deformation = context.deformation(camera)
            xyz = pc.get_xyz+deformation['d_xyz']
            normal = build_rotation(pc.get_rotation_bias(deformation['d_rotation']))[:, :, 2]
            if not torch.isfinite(xyz).all() or not torch.isfinite(normal).all():
                raise ValueError('冻结来源的预测不有限，不能建立时间稳定掩码。')
            if len(keys) == 1:
                reference_xyz, reference_normal = xyz.clone(), normal.clone()
            move = torch.maximum(move, (xyz-reference_xyz).norm(dim=-1)/extent)
            change = torch.atan2(torch.cross(normal, reference_normal, dim=-1).norm(dim=-1),
                                 (normal*reference_normal).sum(-1).clamp(-1, 1))*180/np.pi
            angle = torch.maximum(angle, change)
        if not keys:
            raise ValueError('训练划分为空。')
        stable = ((move <= .005) & (angle <= 5.)).cpu().numpy()
    context.verify_frozen()
    metadata = dict(format_version=1, method='frozen_stage1_training_time_variation',
        source=str(context.source), source_iteration=context.iteration,
        source_files=context.source_files, optimization_keys=keys,
        gt_normal_inputs=False, gt_geometry_inputs=False,
        thresholds=dict(move_max_scene_fraction=.005, normal_change_degrees=5.),
        source_points=len(stable), stable_points=int(stable.sum()),
        script_sha256=file_sha256(__file__))
    np.savez_compressed(out/'source_static_mask.npz', stable_mask=stable,
        move_max_scene_fraction=move.cpu().numpy(), normal_change_max_degrees=angle.cpu().numpy(),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)))
    (out/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(dict(argv=sys.argv, cwd=str(Path.cwd())), indent=2)+'\n')
    (out/'prepare_stage2_static_mask.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(source_points=len(stable), stable_points=int(stable.sum()),
                         optimization_views=len(keys), gt_normal_inputs=False)), flush=True)


if __name__ == '__main__':
    main()
