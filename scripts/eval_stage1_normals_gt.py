#!/usr/bin/env python3
"""Compare Stage-1 GS normals with per-frame Blender-camera-space EXR normals."""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import DeformModel, GaussianModel, Scene
from scene.photometric_lambertian import PhotometricLambertianRenderer
from utils.general_utils import safe_state
from utils.normal_eval_utils import (
    blender_camera_normal_to_runtime_view,
    normal_angular_error_degrees,
)


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Cannot find a frame number in {path}")
    return int(match.group(1))


def _normal_paths(directory: Path) -> dict[int, Path]:
    paths = {}
    for path in sorted(directory.glob("*.exr")):
        frame = _frame_id(path)
        if frame in paths:
            raise ValueError(f"Duplicate EXR normal frame {frame}: {path}")
        paths[frame] = path
    if not paths:
        raise ValueError(f"No EXR normals found in {directory}")
    return paths


def _frame_records(source_path: Path) -> tuple[list[dict], list[dict]]:
    records = []
    for filename in ("transforms_train.json", "transforms_test.json"):
        with (source_path / filename).open("r", encoding="utf-8") as handle:
            records.append(sorted(json.load(handle)["frames"], key=lambda item: item["file_path"]))
    return records[0], records[1]


def _load_gt_normal(path: Path, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load LH EXR normal channels with OpenEXR instead of OpenCV.

    The project OpenCV build cannot decode EXR. LH exports `Normal.X/Y/Z`,
    while RGB EXRs remain supported as a fallback for compatible datasets.
    """
    channels = OpenEXR.File(str(path)).parts[0].channels
    names = ("Normal.X", "Normal.Y", "Normal.Z")
    if not all(name in channels for name in names):
        names = ("R", "G", "B")
    if not all(name in channels for name in names):
        raise ValueError(f"Expected Normal.X/Y/Z or RGB channels in {path}, got {list(channels)}")
    image = np.stack([channels[name].pixels for name in names], axis=-1)
    normal = torch.from_numpy(image).permute(2, 0, 1).float()
    normal = F.interpolate(normal[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    valid = normal.norm(dim=0, keepdim=True) > 1e-6
    normal = F.normalize(normal, dim=0)
    normal = blender_camera_normal_to_runtime_view(normal.permute(1, 2, 0)).permute(2, 0, 1)
    return normal, valid[0]


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
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if not 0.0 <= args.alpha_threshold <= 1.0:
        parser.error("--alpha_threshold must lie in [0, 1]")
    safe_state(args.quiet)
    dataset = model.extract(args)
    dataset.eval = True
    gt_normals = _normal_paths(args.gt_normal_dir.expanduser().resolve())

    with torch.no_grad():
        deform = DeformModel(dataset.deform_type, dataset.is_blender, dataset.hyper_dim, dataset.pred_color)
        if not deform.load_weights(dataset.model_path, iteration=args.load_iter):
            raise FileNotFoundError(f"Missing deformation iteration {args.load_iter}")
        gaussians = GaussianModel(dataset.sh_degree, dataset.no_binary_separation, dataset.hyper_dim)
        scene = Scene(dataset, gaussians, load_iteration=args.load_iter)
        mode = getattr(pipeline.extract(args), "render_mode", "photometric_lambertian")
        pipeline_args = pipeline.extract(args)
        pipeline_args.render_mode = "photometric_lambertian" if mode == "original" else mode
        if pipeline_args.render_mode != "photometric_lambertian":
            raise ValueError("This evaluator targets photometric_lambertian checkpoints only.")
        renderer = PhotometricLambertianRenderer(scene.all_timesteps, device="cuda")
        renderer.load_weights(dataset.model_path, scene.loaded_iter)
        renderer.eval()
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        train_records, test_records = _frame_records(Path(dataset.source_path))
        camera_records = list(zip(scene.getTrainCameras(), train_records)) + list(zip(scene.getTestCameras(), test_records))
        output_dir = Path(dataset.model_path) / "normal_gt_eval" / f"ours_{scene.loaded_iter}"
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
            gs_normal = F.normalize(rendered["rend_normal_view"], dim=0)
            gt_normal, gt_valid = _load_gt_normal(gt_path, camera.image_height, camera.image_width)
            gt_normal = gt_normal.to(gs_normal.device)
            valid = gt_valid.to(gs_normal.device) & (rendered["rend_alpha"][0] >= args.alpha_threshold)
            error = normal_angular_error_degrees(gs_normal, gt_normal, valid)
            values = error[valid]
            if values.numel() == 0:
                raise RuntimeError(f"No valid normal pixels for source frame {source_frame}")
            rows.append({"source_frame": source_frame, "fid": float(camera.fid.item()), "valid_pixels": int(values.numel()), "mean_deg": float(values.mean().item()), "median_deg": float(values.median().item()), "p95_deg": float(torch.quantile(values, 0.95).item())})
            gs_images.append((gs_normal * 0.5 + 0.5).cpu())
            gt_images.append((gt_normal * 0.5 + 0.5).cpu())
            error_images.append(torch.stack((error / 180.0,) * 3).cpu())
            if dataset.load2gpu_on_the_fly:
                camera.load2device("cpu")
        all_values = {key: float(np.mean([row[key] for row in rows])) for key in ("mean_deg", "median_deg", "p95_deg")}
        (output_dir / "normal_metrics.json").write_text(json.dumps({"coordinate_convention": "GT Blender camera (+X,+Y,-Z) -> runtime view (+X,-Y,+Z)", "alpha_threshold": args.alpha_threshold, "summary_mean_over_frames": all_values, "frames": rows}, indent=2) + "\n", encoding="utf-8")
        _make_contact_sheet(gs_images, output_dir / "gs_normal_contact_sheet.png")
        _make_contact_sheet(gt_images, output_dir / "gt_normal_contact_sheet.png")
        _make_contact_sheet(error_images, output_dir / "normal_error_contact_sheet.png")
        print(json.dumps({"output": str(output_dir), "summary_mean_over_frames": all_values}, indent=2))


if __name__ == "__main__":
    main()
