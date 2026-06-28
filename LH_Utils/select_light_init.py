"""Select Stage 1 V2 directional-light initialization by phase multistart."""

from __future__ import annotations

import json
import math
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import DeformModel, GaussianModel, Scene
from scene.photometric_lambertian import PhotometricLambertianRenderer
from utils.loss_utils import l1_loss
from utils.general_utils import safe_state


def freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def evaluate_candidate(
    renderer: PhotometricLambertianRenderer,
    scene: Scene,
    gaussians: GaussianModel,
    deform: DeformModel,
    pipeline,
    background: torch.Tensor,
    short_iters: int,
    lr: float,
    load2gpu_on_the_fly: bool,
) -> float:
    renderer.training_setup(type("Args", (), {"photometric_light_lr": lr})())
    renderer.set_light_lr(lr)
    cameras = scene.getTrainCameras()
    losses = []

    for _ in range(short_iters):
        view = cameras[random.randrange(len(cameras))]
        if load2gpu_on_the_fly:
            view.load2device()

        with torch.no_grad():
            xyz = gaussians.get_xyz
            time_input = view.fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_values = deform.step(
                xyz,
                time_input,
                feature=gaussians.get_binary_feature(eval=True),
                is_training=False,
                camera_center=view.camera_center,
            )
            d_values = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in d_values.items()
            }

        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            d_values["d_xyz"],
            d_values["d_rotation"],
            d_values["d_scaling"],
            d_opacity=d_values["d_opacity"],
            d_color=d_values["d_color"],
            photometric_renderer=renderer,
        )
        image = render_pkg["render"]
        gt = view.original_image_train_light.cuda()
        loss = l1_loss(image, gt)

        renderer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        renderer.optimizer.step()
        losses.append(float(loss.detach().item()))

        if load2gpu_on_the_fly:
            view.load2device("cpu")

    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    parser = ArgumentParser(description="Select a Stage 1 V2 photometric light initialization.")
    model = ModelParams(parser)
    opt_group = OptimizationParams(parser)
    pipeline_group = PipelineParams(parser)
    parser.add_argument("--load_iter", type=int, default=-1)
    parser.add_argument("--deform-type", dest="deform_type", type=str, default="mlp")
    parser.add_argument("--num_phases", type=int, default=8)
    parser.add_argument("--try_reverse_direction", action="store_true")
    parser.add_argument("--short_iters", type=int, default=1000)
    parser.add_argument("--candidate_lr", type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    dataset = model.extract(args)
    opt = opt_group.extract(args)
    pipeline = pipeline_group.extract(args)
    pipeline.render_mode = "photometric_lambertian"

    deform = DeformModel(
        deform_type=dataset.deform_type,
        is_blender=dataset.is_blender,
        hyper_dim=dataset.hyper_dim,
        pred_color=dataset.pred_color,
    )
    deform.load_weights(dataset.model_path, iteration=args.load_iter)
    freeze_module(deform.deform)

    gaussians = GaussianModel(
        dataset.sh_degree,
        no_binary_separation=dataset.no_binary_separation,
        fea_dim=dataset.hyper_dim,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.load_iter)
    gaussians.enable_photometric_albedo()
    for value in vars(gaussians).values():
        if torch.is_tensor(value) and getattr(value, "requires_grad", False):
            value.requires_grad_(False)
    for param in [
        gaussians._xyz,
        gaussians._albedo_dc,
        gaussians._albedo_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians._photometric_albedo,
    ]:
        if torch.is_tensor(param):
            param.requires_grad_(False)

    background_value = 1.0 if dataset.white_background else 0.0
    background = torch.full((3,), background_value, dtype=torch.float32, device="cuda")

    signs = [1, -1] if args.try_reverse_direction else [int(opt.photometric_init_direction_sign)]
    results = []
    best = None

    for sign in signs:
        for phase_index in range(args.num_phases):
            phase = 2.0 * math.pi * phase_index / max(args.num_phases, 1)
            renderer = PhotometricLambertianRenderer.from_args(scene.all_timesteps, opt, device="cuda")
            renderer.light_model.reset_circle_init(phase=phase, direction_sign=sign)
            loss = evaluate_candidate(
                renderer,
                scene,
                gaussians,
                deform,
                pipeline,
                background,
                args.short_iters,
                args.candidate_lr,
                dataset.load2gpu_on_the_fly,
            )
            record = {
                "phase_index": int(phase_index),
                "phase": float(phase),
                "direction_sign": int(sign),
                "loss": float(loss),
                "state_dict": {key: value.detach().cpu() for key, value in renderer.state_dict().items()},
                "config": renderer.config_dict(),
            }
            results.append(record)
            print(
                f"candidate phase={phase_index}/{args.num_phases} sign={sign:+d} "
                f"loss={loss:.6f}"
            )
            if best is None or loss < best["loss"]:
                best = record

    output_dir = Path(dataset.model_path) / "photometric_multistart" / f"iteration_{scene.loaded_iter}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_payload = {
        "best_phase": best["phase"],
        "best_phase_index": best["phase_index"],
        "best_direction_sign": best["direction_sign"],
        "best_candidate_loss": best["loss"],
        "light_model_state": best["state_dict"],
        "config": best["config"],
        "all_candidates": [
            {key: value for key, value in record.items() if key not in {"state_dict"}}
            for record in results
        ],
    }
    torch.save(best_payload, output_dir / "best_light_init.pth")
    with (output_dir / "best_light_init.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: value for key, value in best_payload.items() if key != "light_model_state"},
            handle,
            indent=2,
        )
    print(f"Best light init saved to {output_dir}")


if __name__ == "__main__":
    main()
