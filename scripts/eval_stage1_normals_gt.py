#!/usr/bin/env python3
"""Compare Stage-1 independent or GS normals with Blender world-space EXR normals."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import DeformModel, GaussianModel, Scene
from scene.photometric_lambertian import PhotometricLambertianRenderer
from utils.general_utils import safe_state
from utils.gt_normal_utils import load_gt_normal, normal_paths
from utils.normal_eval_utils import (
    alpha_normalized_normal_map,
    masked_normal_cosine_loss,
    normal_angular_error_degrees,
    resolve_normal_source,
)


def _frame_records(source_path: Path) -> tuple[list[dict], list[dict]]:
    records = []
    for filename in ("transforms_train.json", "transforms_test.json"):
        with (source_path / filename).open("r", encoding="utf-8") as handle:
            records.append(sorted(json.load(handle)["frames"], key=lambda item: item["file_path"]))
    return records[0], records[1]


def _make_contact_sheet(images: list[torch.Tensor], output_path: Path) -> None:
    if not images:
        return
    indices = sorted({0, len(images) // 2, len(images) - 1})
    torchvision.utils.save_image(torchvision.utils.make_grid([images[index] for index in indices], nrow=3), output_path)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--load_iter", type=int, required=True)
    parser.add_argument("--gt_normal_dir", type=Path, required=True)
    parser.add_argument("--alpha_threshold", type=float, default=0.5)
    parser.add_argument(
        "--normal_source",
        choices=("auto", "independent", "gs"),
        default="auto",
        help=(
            "Normal source to evaluate. 'auto' preserves the historical behavior: "
            "independent normal when present, otherwise GS raster normal."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional explicit output directory; preserves earlier evaluation results.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if not 0.0 <= args.alpha_threshold <= 1.0:
        parser.error("--alpha_threshold must lie in [0, 1]")
    safe_state(args.quiet)
    dataset = model.extract(args)
    dataset.eval = True
    gt_normals = normal_paths(args.gt_normal_dir)

    with torch.no_grad():
        deform = DeformModel(
            deform_type=dataset.deform_type,
            is_blender=dataset.is_blender,
            hyper_dim=dataset.hyper_dim,
            pred_color=dataset.pred_color,
        )
        if not deform.load_weights(dataset.model_path, iteration=args.load_iter):
            raise FileNotFoundError(f"Missing deformation iteration {args.load_iter}")
        gaussians = GaussianModel(dataset.sh_degree, dataset.no_binary_separation, dataset.hyper_dim)
        scene = Scene(dataset, gaussians, load_iteration=args.load_iter)
        mode = getattr(pipeline.extract(args), "render_mode", "photometric_lambertian")
        pipeline_args = pipeline.extract(args)
        pipeline_args.render_mode = "photometric_lambertian" if mode == "original" else mode
        if pipeline_args.render_mode not in {"original_sh", "photometric_lambertian"}:
            raise ValueError(
                "This evaluator supports original_sh or photometric_lambertian checkpoints."
            )
        renderer = None
        if pipeline_args.render_mode == "photometric_lambertian":
            renderer = PhotometricLambertianRenderer(
                scene.all_timesteps, device="cuda"
            )
            renderer.load_weights(dataset.model_path, scene.loaded_iter)
            renderer.eval()
        normal_source = resolve_normal_source(
            args.normal_source, gaussians.use_photometric_normal
        )
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        train_records, test_records = _frame_records(Path(dataset.source_path))
        camera_records = list(zip(scene.getTrainCameras(), train_records)) + list(
            zip(scene.getTestCameras(), test_records)
        )
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else Path(dataset.model_path) / "normal_gt_eval" / f"ours_{scene.loaded_iter}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, gs_images, gt_images, error_images = [], [], [], []
        for camera, record in camera_records:
            source_frame = int(record["source_frame"])
            gt_path = gt_normals.get(source_frame)
            if gt_path is None:
                raise FileNotFoundError(f"No GT normal EXR for source frame {source_frame}")
            if dataset.load2gpu_on_the_fly:
                camera.load2device()
            count = gaussians.get_xyz.shape[0]
            d_values = deform.step(gaussians.get_xyz.detach(), camera.fid.unsqueeze(0).expand(count, -1), feature=gaussians.get_binary_feature(), camera_center=camera.camera_center)
            rendered = render(camera, gaussians, pipeline_args, background, d_values["d_xyz"], d_values["d_rotation"], d_values["d_scaling"], d_opacity=d_values["d_opacity"], d_color=d_values["d_color"], photometric_renderer=renderer)
            # LH's Blender Normal pass is exported in world space. Compare it
            # directly with the renderer's world-space normal map; treating the
            # EXR channels as camera-local applies a spurious camera transform.
            if normal_source == "independent_photometric_normal":
                encoded_normal = rendered["photometric_normal"] * 0.5 + 0.5
                normal_rendered = render(
                    camera,
                    gaussians,
                    pipeline_args,
                    background,
                    d_values["d_xyz"],
                    d_values["d_rotation"],
                    d_values["d_scaling"],
                    d_opacity=d_values["d_opacity"],
                    d_color=d_values["d_color"],
                    override_color=encoded_normal,
                )
                gs_normal = alpha_normalized_normal_map(
                    normal_rendered["render"], normal_rendered["rend_alpha"]
                )
            else:
                gs_normal = F.normalize(rendered["rend_normal"], dim=0)
            gt_normal, gt_valid = load_gt_normal(
                gt_path, camera.image_height, camera.image_width
            )
            gt_normal = gt_normal.to(gs_normal.device)
            valid = gt_valid.to(gs_normal.device) & (
                rendered["rend_alpha"][0] >= args.alpha_threshold
            )
            error = normal_angular_error_degrees(gs_normal, gt_normal, valid)
            cosine_loss = masked_normal_cosine_loss(gs_normal, gt_normal, valid)
            values = error[valid]
            if values.numel() == 0:
                raise RuntimeError(f"No valid normal pixels for source frame {source_frame}")
            rows.append(
                {
                    "source_frame": source_frame,
                    "fid": float(camera.fid.item()),
                    "valid_pixels": int(values.numel()),
                    "cosine_loss": float(cosine_loss.item()),
                    "mean_deg": float(values.mean().item()),
                    "median_deg": float(values.median().item()),
                    "p95_deg": float(torch.quantile(values, 0.95).item()),
                }
            )
            gs_images.append((gs_normal * 0.5 + 0.5).cpu())
            gt_images.append((gt_normal * 0.5 + 0.5).cpu())
            error_images.append(torch.stack((error / 180.0,) * 3).cpu())
            if dataset.load2gpu_on_the_fly:
                camera.load2device("cpu")
        all_values = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("cosine_loss", "mean_deg", "median_deg", "p95_deg")
        }
        metrics = {
            "coordinate_convention": (
                "GT Blender world space compared directly with renderer world space"
            ),
            "normal_source": normal_source,
            "alpha_threshold": args.alpha_threshold,
            "summary_mean_over_frames": all_values,
            "frames": rows,
        }
        (output_dir / "normal_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        _make_contact_sheet(gs_images, output_dir / "gs_normal_contact_sheet.png")
        _make_contact_sheet(gt_images, output_dir / "gt_normal_contact_sheet.png")
        _make_contact_sheet(error_images, output_dir / "normal_error_contact_sheet.png")
        print(json.dumps({"output": str(output_dir), "summary_mean_over_frames": all_values}, indent=2))


if __name__ == "__main__":
    main()
