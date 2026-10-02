#!/usr/bin/env python3
"""逐相机/逐时间评估 Stage 1 渲染 Alpha 与训练图 RGBA Alpha 的一致性。"""

from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

# 该独立诊断脚本位于 LIHAN/scripts，直接执行时 Python 不会自动加入仓库根目录。
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import DeformModel, GaussianModel, Scene
from scene.photometric_lambertian import PhotometricLambertianRenderer
from utils.general_utils import safe_state


def _save_representatives(items: list[tuple[str, torch.Tensor, torch.Tensor]], output: Path) -> None:
    if not items:
        return
    indices = sorted({0, len(items) // 2, len(items) - 1})
    images = []
    for index in indices:
        _, gt, predicted = items[index]
        difference = (gt - predicted).abs()
        images.extend((gt, predicted, difference))
    torchvision.utils.save_image(torchvision.utils.make_grid(images, nrow=3), output)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--load_iter", type=int, required=True)
    parser.add_argument("--alpha-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if not 0.0 <= args.alpha_threshold <= 1.0:
        parser.error("--alpha-threshold 必须在 [0, 1] 内")
    safe_state(args.quiet)
    dataset = model.extract(args)
    dataset.eval = True
    pipeline_args = pipeline.extract(args)
    if pipeline_args.render_mode == "original":
        pipeline_args.render_mode = "original_sh"
    if pipeline_args.render_mode not in {"original_sh", "photometric_lambertian"}:
        raise ValueError("仅支持 original_sh 或 photometric_lambertian checkpoint")

    with torch.no_grad():
        deform = DeformModel(
            deform_type=dataset.deform_type,
            is_blender=dataset.is_blender,
            hyper_dim=dataset.hyper_dim,
            pred_color=dataset.pred_color,
        )
        if not deform.load_weights(dataset.model_path, iteration=args.load_iter):
            raise FileNotFoundError(f"缺少 deform checkpoint：iteration {args.load_iter}")
        gaussians = GaussianModel(
            dataset.sh_degree,
            dataset.no_binary_separation,
            dataset.hyper_dim,
        )
        scene = Scene(dataset, gaussians, load_iteration=args.load_iter)
        renderer = None
        if pipeline_args.render_mode == "photometric_lambertian":
            renderer = PhotometricLambertianRenderer(scene.all_timesteps, device="cuda")
            renderer.load_weights(dataset.model_path, scene.loaded_iter)
            renderer.eval()
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        cameras = sorted(
            [*scene.getTrainCameras(), *scene.getTestCameras()],
            key=lambda camera: float(camera.fid.item()),
        )
        rows, representatives = [], []
        total_intersection = total_union = total_pred = total_gt = 0
        all_mae, all_bce, soft_mae = [], [], []
        for camera in cameras:
            if camera.gt_alpha_mask is None:
                raise RuntimeError(f"相机无显式 Alpha：{camera.image_name_train_light}")
            if dataset.load2gpu_on_the_fly:
                camera.load2device()
            count = gaussians.get_xyz.shape[0]
            d_values = deform.step(
                gaussians.get_xyz.detach(),
                camera.fid.unsqueeze(0).expand(count, -1),
                feature=gaussians.get_binary_feature(),
                camera_center=camera.camera_center,
            )
            rendered = render(
                camera,
                gaussians,
                pipeline_args,
                background,
                d_values["d_xyz"],
                d_values["d_rotation"],
                d_values["d_scaling"],
                d_opacity=d_values["d_opacity"],
                d_color=d_values["d_color"],
                photometric_renderer=renderer,
            )
            predicted = rendered["rend_alpha"].clamp(0.0, 1.0)[0]
            target = camera.gt_alpha_mask.cuda()[0].clamp(0.0, 1.0)
            pred_hard = predicted >= args.alpha_threshold
            gt_hard = target >= args.alpha_threshold
            intersection = int((pred_hard & gt_hard).sum().item())
            union = int((pred_hard | gt_hard).sum().item())
            predicted_pixels = int(pred_hard.sum().item())
            gt_pixels = int(gt_hard.sum().item())
            mae = float((predicted - target).abs().mean().item())
            bce = float(F.binary_cross_entropy(predicted.clamp(1e-6, 1 - 1e-6), target).item())
            soft = (target > 0.0) & (target < 1.0)
            soft_value = float((predicted[soft] - target[soft]).abs().mean().item()) if soft.any() else None
            total_intersection += intersection
            total_union += union
            total_pred += predicted_pixels
            total_gt += gt_pixels
            all_mae.append(mae)
            all_bce.append(bce)
            if soft_value is not None:
                soft_mae.append(soft_value)
            rows.append({
                "image_name": camera.image_name_train_light,
                "fid": float(camera.fid.item()),
                "predicted_coverage": float(predicted.mean().item()),
                "gt_coverage": float(target.mean().item()),
                "intersection": intersection,
                "union": union,
                "iou": float(intersection / union) if union else 1.0,
                "mae": mae,
                "bce": bce,
                "soft_edge_mae": soft_value,
            })
            representatives.append((camera.image_name_train_light, target[None], predicted[None]))
            if dataset.load2gpu_on_the_fly:
                camera.load2device("cpu")

        rows.sort(key=lambda row: row["fid"])
        with (output / "alpha_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        precision = total_intersection / total_pred if total_pred else 1.0
        recall = total_intersection / total_gt if total_gt else 1.0
        metrics = {
            "status": "PASS",
            "frames": len(rows),
            "alpha_threshold": args.alpha_threshold,
            "micro": {
                "iou": total_intersection / total_union if total_union else 1.0,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "predicted_hard_coverage": total_pred / (len(rows) * camera.image_height * camera.image_width),
                "gt_hard_coverage": total_gt / (len(rows) * camera.image_height * camera.image_width),
            },
            "mean_over_frames": {
                "iou": sum(row["iou"] for row in rows) / len(rows),
                "predicted_coverage": sum(row["predicted_coverage"] for row in rows) / len(rows),
                "gt_coverage": sum(row["gt_coverage"] for row in rows) / len(rows),
                "mae": sum(all_mae) / len(all_mae),
                "bce": sum(all_bce) / len(all_bce),
                "soft_edge_mae": sum(soft_mae) / len(soft_mae) if soft_mae else None,
            },
            "representative_contact_sheet": "alpha_gt_pred_absdiff_contact_sheet.png",
            "columns": "每组三列依次为 GT Alpha、渲染 Alpha、绝对误差；帧为首/中/末。",
        }
        (output / "alpha_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _save_representatives(representatives, output / metrics["representative_contact_sheet"])
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
