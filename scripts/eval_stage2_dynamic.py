"""Evaluate a Stage 2 checkpoint on every held-out dynamic frame."""

import json
import os
import re
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from piq import LPIPS
from pytorch_msssim import ms_ssim
from tqdm import tqdm
from torchvision.io import read_image

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer.render_ir import render_ir
from scene import DeformModel, GaussianModel, Scene
from scene.light import EnvLight
from utils.general_utils import safe_state
from utils.graphics_utils import srgb_to_rgb
from utils.image_utils import alex_lpips, psnr, ssim as ssim_metric


def evaluate(
    dataset: ModelParams,
    pipeline: PipelineParams,
    optimization: OptimizationParams,
    load_iter: int,
) -> None:
    dataset.eval = True
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

    env_light = EnvLight(
        path=None,
        device="cuda",
        resolution=[optimization.envmap_resolution // 2, optimization.envmap_resolution],
        max_res=optimization.envmap_resolution,
        activation=optimization.envmap_activation,
    )
    env_light.load_weights(dataset.model_path, scene.loaded_iter)

    background_value = 1.0 if dataset.white_background else 0.0
    background = torch.full((3,), background_value, dtype=torch.float32, device="cuda")
    output_dir = os.path.join(
        dataset.model_path, "eval_stage2_dynamic", f"ours_{scene.loaded_iter}"
    )
    makedirs(output_dir, exist_ok=True)

    vgg_lpips = LPIPS().cuda().eval()
    per_frame = []
    metric_names = ["l1", "psnr", "ssim", "lpips_vgg", "ms_ssim", "lpips_alex"]
    bvh_built = False
    source_root = Path(dataset.source_path)
    albedo_root = source_root.parent / "albedo"
    manifest_path = source_root / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_albedo = manifest.get("source_metadata", {}).get("albedo_directory")
        if manifest_albedo:
            albedo_root = Path(manifest_albedo).expanduser()
    albedo_records = []

    with torch.no_grad():
        for index, view in enumerate(tqdm(cameras, desc="Eval Stage 2 dynamic")):
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

            bvh_args = {
                "d_rotation": d_values["d_rotation"],
                "d_xyz": d_values["d_xyz"],
                "d_scaling": d_values["d_scaling"],
            }
            if not bvh_built:
                gaussians.build_bvh(**bvh_args)
                bvh_built = True
            else:
                gaussians.update_bvh(**bvh_args)

            render_pkg = render_ir(
                viewpoint_camera=view,
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                d_xyz=d_values["d_xyz"],
                d_rotation=d_values["d_rotation"],
                d_scaling=d_values["d_scaling"],
                d_opacity=d_values["d_opacity"],
                d_color=d_values["d_color"],
                relight=False,
                env_light=env_light,
                training=False,
            )

            target_size = render_pkg["render"].shape[1:]
            ground_truth = F.interpolate(
                view.original_image_train_light.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)
            if view.gt_alpha_mask is None:
                mask = torch.ones((1, *target_size), dtype=ground_truth.dtype, device="cuda")
            else:
                mask = F.interpolate(
                    view.gt_alpha_mask.cuda().unsqueeze(0),
                    size=target_size,
                    mode="nearest",
                ).squeeze(0)

            prediction = render_pkg["render"] * mask + background[:, None, None] * (1.0 - mask)
            prediction = prediction.clamp(0.0, 1.0)
            ground_truth = ground_truth * mask + background[:, None, None] * (1.0 - mask)
            error = (prediction - ground_truth).abs()
            normal = (render_pkg["rend_normal_view"] * 0.5 + 0.5).clamp(0.0, 1.0)

            prefix = f"{index:03d}_{view.image_name_train_light}"
            torchvision.utils.save_image(prediction, os.path.join(output_dir, f"{prefix}_render.png"))
            torchvision.utils.save_image(ground_truth, os.path.join(output_dir, f"{prefix}_gt.png"))
            torchvision.utils.save_image(error, os.path.join(output_dir, f"{prefix}_error.png"))
            torchvision.utils.save_image(mask, os.path.join(output_dir, f"{prefix}_mask.png"))
            torchvision.utils.save_image(
                render_pkg["base_color"].clamp(0.0, 1.0),
                os.path.join(output_dir, f"{prefix}_albedo.png"),
            )
            torchvision.utils.save_image(
                render_pkg["roughness"].clamp(0.0, 1.0),
                os.path.join(output_dir, f"{prefix}_roughness.png"),
            )
            torchvision.utils.save_image(normal, os.path.join(output_dir, f"{prefix}_normal.png"))
            torchvision.utils.save_image(
                torch.cat((ground_truth, prediction, error), dim=2),
                os.path.join(output_dir, f"{prefix}_comparison.png"),
            )

            frame_match = re.search(r"(\d+)$", view.image_name_train_light)
            albedo_path = (
                albedo_root / f"albedo_{int(frame_match.group(1)):04d}.png"
                if frame_match is not None
                else None
            )
            if albedo_path is not None and albedo_path.is_file():
                albedo_gt = read_image(str(albedo_path)).float()[:3].cuda() / 255.0
                albedo_gt = F.interpolate(
                    albedo_gt.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                albedo_records.append(
                    {
                        "index": index,
                        "frame": view.image_name_train_light,
                        "prefix": prefix,
                        "prediction": render_pkg["base_color_linear"].clamp(0.0, 1.0),
                        "ground_truth": srgb_to_rgb(albedo_gt).clamp(0.0, 1.0),
                        "mask": mask,
                    }
                )

            pred_batch = prediction.unsqueeze(0)
            gt_batch = ground_truth.unsqueeze(0)
            per_frame.append(
                {
                    "index": index,
                    "frame": view.image_name_train_light,
                    "time": float(view.fid.item()),
                    "l1": float(F.l1_loss(prediction, ground_truth).item()),
                    "psnr": float(psnr(pred_batch, gt_batch).mean().item()),
                    "ssim": float(ssim_metric(pred_batch, gt_batch, data_range=1.0).mean().item()),
                    "lpips_vgg": float(vgg_lpips(pred_batch, gt_batch).mean().item()),
                    "ms_ssim": float(ms_ssim(pred_batch, gt_batch, data_range=1.0).mean().item()),
                    "lpips_alex": float(alex_lpips(pred_batch, gt_batch).mean().item()),
                }
            )

            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")

    averages = {
        name: sum(frame[name] for frame in per_frame) / len(per_frame)
        for name in metric_names
    }
    results = {
        "stage": 2,
        "iteration": scene.loaded_iter,
        "test_frame_count": len(per_frame),
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "metrics_average": averages,
        "metrics_per_frame": per_frame,
        "visualization_layout": "comparison images are [ground truth | render | absolute error]",
    }

    if albedo_records:
        foreground_predictions = torch.cat(
            [
                record["prediction"].permute(1, 2, 0)[record["mask"][0] > 0]
                for record in albedo_records
            ],
            dim=0,
        )
        foreground_ground_truths = torch.cat(
            [
                record["ground_truth"].permute(1, 2, 0)[record["mask"][0] > 0]
                for record in albedo_records
            ],
            dim=0,
        )
        albedo_scale = (
            foreground_ground_truths / foreground_predictions.clamp_min(1e-6)
        ).median(dim=0).values
        albedo_per_frame = []

        for record in albedo_records:
            albedo_prediction = (
                record["prediction"] * albedo_scale[:, None, None]
            ).clamp(0.0, 1.0)
            albedo_ground_truth = record["ground_truth"]
            albedo_mask = record["mask"]
            albedo_prediction = albedo_prediction * albedo_mask
            albedo_ground_truth = albedo_ground_truth * albedo_mask
            albedo_error = (albedo_prediction - albedo_ground_truth).abs()

            torchvision.utils.save_image(
                albedo_prediction,
                os.path.join(output_dir, f"{record['prefix']}_albedo_scaled.png"),
            )
            torchvision.utils.save_image(
                albedo_ground_truth,
                os.path.join(output_dir, f"{record['prefix']}_albedo_gt.png"),
            )
            torchvision.utils.save_image(
                torch.cat((albedo_ground_truth, albedo_prediction, albedo_error), dim=2),
                os.path.join(output_dir, f"{record['prefix']}_albedo_comparison.png"),
            )

            pred_batch = albedo_prediction.unsqueeze(0)
            gt_batch = albedo_ground_truth.unsqueeze(0)
            albedo_per_frame.append(
                {
                    "index": record["index"],
                    "frame": record["frame"],
                    "l1": float(F.l1_loss(albedo_prediction, albedo_ground_truth).item()),
                    "psnr": float(psnr(pred_batch, gt_batch).mean().item()),
                    "ssim": float(ssim_metric(pred_batch, gt_batch, data_range=1.0).mean().item()),
                    "lpips_vgg": float(vgg_lpips(pred_batch, gt_batch).mean().item()),
                    "ms_ssim": float(ms_ssim(pred_batch, gt_batch, data_range=1.0).mean().item()),
                    "lpips_alex": float(alex_lpips(pred_batch, gt_batch).mean().item()),
                }
            )

        results["albedo_scale_rgb_linear"] = [float(value) for value in albedo_scale]
        results["albedo_metrics_average"] = {
            name: sum(frame[name] for frame in albedo_per_frame) / len(albedo_per_frame)
            for name in metric_names
        }
        results["albedo_metrics_per_frame"] = albedo_per_frame
        results["albedo_ground_truth_root"] = str(albedo_root)
        results["albedo_visualization_layout"] = (
            "albedo comparison images are [linear GT | scaled linear prediction | absolute error]"
        )
    results_path = os.path.join(dataset.model_path, "results_stage2_dynamic.json")
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(
        "Stage 2 average: "
        f"PSNR {averages['psnr']:.4f}, SSIM {averages['ssim']:.4f}, "
        f"LPIPS(VGG) {averages['lpips_vgg']:.4f}, "
        f"MS-SSIM {averages['ms_ssim']:.4f}, LPIPS(Alex) {averages['lpips_alex']:.4f}"
    )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a LumiMotion Stage 2 checkpoint")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument("--load_iter", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    evaluate(
        model.extract(args),
        pipeline.extract(args),
        optimization.extract(args),
        args.load_iter,
    )
