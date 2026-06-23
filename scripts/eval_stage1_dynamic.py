"""Evaluate a Stage 1 checkpoint on every held-out dynamic frame."""

import json
import os
from argparse import ArgumentParser
from os import makedirs

import torch
import torchvision
from piq import LPIPS
from pytorch_msssim import ms_ssim
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import DeformModel, GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import alex_lpips, psnr, ssim as ssim_metric


def evaluate(dataset: ModelParams, pipeline: PipelineParams, load_iter: int) -> None:
    dataset.eval = True
    if getattr(pipeline, "render_mode", "original") != "original":
        raise ValueError("eval_stage1_dynamic.py currently evaluates render_mode='original' only.")

    deform = DeformModel(
        deform_type=dataset.deform_type,
        is_blender=dataset.is_blender,
        hyper_dim=dataset.hyper_dim,
        pred_color=dataset.pred_color,
    )
    deform.load_weights(dataset.model_path, iteration=load_iter)

    gaussians = GaussianModel(
        dataset.sh_degree,
        no_binary_separation=dataset.no_binary_separation,
        fea_dim=dataset.hyper_dim,
    )
    scene = Scene(dataset, gaussians, load_iteration=load_iter)
    cameras = scene.getTestCameras()
    if not cameras:
        raise RuntimeError("No test cameras found. Run with --eval and provide transforms_test.json.")

    background_value = 1.0 if dataset.white_background else 0.0
    background = torch.full((3,), background_value, dtype=torch.float32, device="cuda")
    output_dir = os.path.join(
        dataset.model_path, "eval_stage1_dynamic", f"ours_{scene.loaded_iter}"
    )
    makedirs(output_dir, exist_ok=True)

    vgg_lpips = LPIPS().cuda().eval()
    per_frame = []
    metric_names = ["l1", "psnr", "ssim", "lpips_vgg", "ms_ssim", "lpips_alex"]

    with torch.no_grad():
        for index, view in enumerate(tqdm(cameras, desc="Eval Stage 1 dynamic")):
            if dataset.load2gpu_on_the_fly:
                view.load2device()

            xyz = gaussians.get_xyz
            time_input = view.fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_values = deform.step(
                xyz.detach(),
                time_input,
                feature=gaussians.get_binary_feature(),
                is_training=False,
                camera_center=view.camera_center,
            )

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
            )
            prediction = render_pkg["render"].clamp(0.0, 1.0)
            ground_truth = view.original_image_train_light.cuda().clamp(0.0, 1.0)
            if view.gt_alpha_mask is None:
                mask = torch.ones_like(ground_truth[:1])
            else:
                mask = view.gt_alpha_mask.cuda()
            prediction = prediction * mask + background[:, None, None] * (1.0 - mask)
            ground_truth = ground_truth * mask + background[:, None, None] * (1.0 - mask)
            error = (prediction - ground_truth).abs()

            albedo_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                d_values["d_xyz"],
                d_values["d_rotation"],
                d_values["d_scaling"],
                d_opacity=d_values["d_opacity"],
                d_color=None,
                override_color=gaussians.get_albedo,
            )
            normal = (render_pkg["rend_normal_view"] * 0.5 + 0.5).clamp(0.0, 1.0)

            prefix = f"{index:03d}_{view.image_name_train_light}"
            torchvision.utils.save_image(prediction, os.path.join(output_dir, f"{prefix}_render.png"))
            torchvision.utils.save_image(ground_truth, os.path.join(output_dir, f"{prefix}_gt.png"))
            torchvision.utils.save_image(error, os.path.join(output_dir, f"{prefix}_error.png"))
            torchvision.utils.save_image(mask, os.path.join(output_dir, f"{prefix}_mask.png"))
            torchvision.utils.save_image(
                albedo_pkg["render"].clamp(0.0, 1.0),
                os.path.join(output_dir, f"{prefix}_albedo.png"),
            )
            torchvision.utils.save_image(normal, os.path.join(output_dir, f"{prefix}_normal.png"))
            torchvision.utils.save_image(
                torch.cat((ground_truth, prediction, error), dim=2),
                os.path.join(output_dir, f"{prefix}_comparison.png"),
            )

            pred_batch = prediction.unsqueeze(0)
            gt_batch = ground_truth.unsqueeze(0)
            frame_metrics = {
                "index": index,
                "frame": view.image_name_train_light,
                "time": float(view.fid.item()),
                "l1": float(torch.nn.functional.l1_loss(prediction, ground_truth).item()),
                "psnr": float(psnr(pred_batch, gt_batch).mean().item()),
                "ssim": float(ssim_metric(pred_batch, gt_batch, data_range=1.0).mean().item()),
                "lpips_vgg": float(vgg_lpips(pred_batch, gt_batch).mean().item()),
                "ms_ssim": float(ms_ssim(pred_batch, gt_batch, data_range=1.0).mean().item()),
                "lpips_alex": float(alex_lpips(pred_batch, gt_batch).mean().item()),
            }
            per_frame.append(frame_metrics)

            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")

    averages = {
        name: sum(frame[name] for frame in per_frame) / len(per_frame)
        for name in metric_names
    }
    results = {
        "stage": 1,
        "iteration": scene.loaded_iter,
        "test_frame_count": len(per_frame),
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "metrics_average": averages,
        "metrics_per_frame": per_frame,
        "visualization_layout": "comparison images are [ground truth | render | absolute error]",
    }
    results_path = os.path.join(dataset.model_path, "results_stage1_dynamic.json")
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(
        "Stage 1 average: "
        f"PSNR {averages['psnr']:.4f}, SSIM {averages['ssim']:.4f}, "
        f"LPIPS(VGG) {averages['lpips_vgg']:.4f}, "
        f"MS-SSIM {averages['ms_ssim']:.4f}, LPIPS(Alex) {averages['lpips_alex']:.4f}"
    )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a LumiMotion Stage 1 checkpoint")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--load_iter", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    evaluate(model.extract(args), pipeline.extract(args), args.load_iter)
