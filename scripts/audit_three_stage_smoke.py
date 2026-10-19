"""检查自足重载、无目标推理、normal透明度保护并导出独立Separation。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scene.three_stage_pipeline import ThreeStageModel, write, sha
from scripts.train_three_stage import load_cameras, geometry_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    model = ThreeStageModel.load(a.model, a.iteration)
    camera = load_cameras(model.binding['source'])[0]; camera.load2device()
    with torch.no_grad():
        before = model.render(camera, a.iteration, components=True)
        # 推理相机的RGB与Alpha都替换，不应改变推理输出或混合权重。
        camera.original_image_train_light.zero_(); camera.gt_alpha_mask.zero_()
        after = model.render(camera, a.iteration, components=True)
        target_difference = max(float((before[k]-after[k]).abs().max()) for k in ('render', 'base_linear', 'residual_linear', 'rend_alpha'))
        assert target_difference == 0, target_difference
        from gaussian_renderer import render
        feature = model.pc.get_binary_feature()
        colors = torch.cat((feature, 1-feature, torch.zeros_like(feature)), dim=-1)
        sep = render(camera, model.effective(camera), model.pipe, torch.zeros(3, device='cuda'),
                     0., 0., 0., override_color=colors)['render']
        pixels = (sep.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1]*255).round().astype(np.uint8)
        assert cv2.imwrite(str(a.output/'separation.png'), pixels)
        initial, current = model.lights.initial, model.lights.directions()
        cross = torch.linalg.cross(current.double(), initial.double(), dim=-1).norm(dim=-1)
        maximum_angle = float(torch.rad2deg(torch.atan2(cross, (current.double()*initial.double()).sum(-1))).max())
        if model.config['direction_lr'] == 0:
            assert torch.equal(current, torch.nn.functional.normalize(initial, dim=-1)) or float((current-initial).abs().max()) < 2e-7
        elif model.stage == 2:
            assert float((current-initial).abs().max()) > 1e-7
        assert maximum_angle <= .50001
    gradient = {}
    if model.stage == 2:
        camera = load_cameras(model.binding['source'])[0]; camera.load2device()
        package = model.render(camera, a.iteration)
        protected = geometry_loss(model, camera, package, True)
        normal_opacity = torch.autograd.grad(protected, model.pc._opacity, allow_unused=True)[0]
        assert normal_opacity is None or normal_opacity.abs().max() == 0
        rgb_alpha = torch.autograd.grad(package['render'].mean()+package['rend_alpha'].mean(), model.pc._opacity)[0]
        assert torch.isfinite(rgb_alpha).all() and rgb_alpha.abs().max() > 0
        gradient = dict(protected_normal_opacity_max=0 if normal_opacity is None else float(normal_opacity.abs().max()),
                        rgb_alpha_opacity_gradient_norm=float(rgb_alpha.norm()))
    receipt = dict(status='PASS_SELF_CONTAINED_RELOAD_TARGET_FREE_INFERENCE', stage=model.stage,
        model=str(a.model), iteration=a.iteration, checkpoint_sha256=sha(a.model/f'three_stage/iteration_{a.iteration}/state.pt'),
        target_rgb_and_alpha_replacement_max_error=target_difference, direction_max_deviation_deg=maximum_angle,
        normal_opacity=gradient, separation=str(a.output/'separation.png'),
        gt_read=False, test_read=False, scientific_accuracy_pass=False)
    write(a.output/'receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
