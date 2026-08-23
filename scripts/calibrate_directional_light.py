#!/usr/bin/env python3
"""Calibrate a fixed directional Lambertian irradiance from Blender GT passes."""

from __future__ import annotations

import json
import math
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0.0)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.maximum(value, 0.0031308) ** (1.0 / 2.4) - 0.055,
    )


def fit_nonnegative_scalar(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.square(prediction).sum())
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Cannot calibrate intensity from an all-zero prediction.")
    return max(float((prediction * target).sum()) / denominator, 0.0)


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Cannot find a frame number in {path}")
    return int(match.group(1))


def _load_rgba(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    return image


def _load_normal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    channels = OpenEXR.File(str(path)).parts[0].channels
    names = ("Normal.X", "Normal.Y", "Normal.Z")
    if not all(name in channels for name in names):
        names = ("R", "G", "B")
    if not all(name in channels for name in names):
        raise ValueError(f"Missing normal channels in {path}: {list(channels)}")
    normal = np.stack([channels[name].pixels for name in names], axis=-1).astype(np.float32)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = length[..., 0] > 1e-6
    normal = normal / np.maximum(length, 1e-6)
    return normal, valid


def _parse_center(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.replace(",", " ").split()])
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("--reference_center must contain three finite values.")
    return values


def _directions_world(
    frame_key: str,
    cameras: dict,
    lights: dict,
    reference_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    light_position = np.asarray(lights[frame_key]["light_pos_world"], dtype=np.float64)
    surface_to_light = light_position - reference_center
    surface_to_light /= np.linalg.norm(surface_to_light)
    camera_position = np.asarray(
        cameras["frames"][frame_key]["extrinsics"]["position_world"],
        dtype=np.float64,
    )
    surface_to_camera = camera_position - reference_center
    surface_to_camera /= np.linalg.norm(surface_to_camera)
    return surface_to_light, surface_to_camera


def _frame_data(
    root: Path,
    frame: int,
    cameras: dict,
    lights: dict,
    reference_center: np.ndarray,
    alpha_threshold: float,
    min_ndotl: float,
    orient_camera_facing: bool,
) -> dict[str, np.ndarray]:
    key = f"{frame:04d}"
    target = _load_rgba(root / "image" / f"image_{key}.png")
    albedo = _load_rgba(root / "albedo" / f"albedo_{key}.png")
    normal, normal_valid = _load_normal(root / "normal_exr" / f"normal_{key}.exr")
    if target.shape != albedo.shape or target.shape[:2] != normal.shape[:2]:
        raise ValueError(f"Frame {key} RGB/albedo/normal resolutions do not match.")
    # LH's EXR Normal pass is world-space, so both dot products must also use
    # world-space light/view directions. The previous camera-space conversion
    # mixed coordinate systems and biased the fitted intensity.
    direction, view_direction = _directions_world(
        key, cameras, lights, reference_center
    )
    if orient_camera_facing:
        facing = (normal * view_direction[None, None]).sum(axis=-1, keepdims=True) >= 0
        normal = normal * np.where(facing, 1.0, -1.0)
    ndotl = np.clip((normal * direction[None, None]).sum(axis=-1), 0.0, 1.0)
    target_linear = srgb_to_linear(target[..., :3])
    albedo_linear = srgb_to_linear(albedo[..., :3])
    base_linear = (albedo_linear / math.pi) * ndotl[..., None]
    valid = (
        (albedo[..., 3] >= alpha_threshold)
        & normal_valid
        & (ndotl >= min_ndotl)
        & (target[..., :3].max(axis=-1) < 0.995)
        & (albedo_linear.max(axis=-1) > 1e-4)
    )
    return {
        "target_srgb": target[..., :3],
        "target_linear": target_linear,
        "albedo_alpha": albedo[..., 3],
        "base_linear": base_linear,
        "valid": valid,
        "direction_world": direction,
        "view_direction_world": view_direction,
        "ndotl": ndotl,
    }


def _psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.square(prediction - target).mean())
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def _contact_sheet(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    rendered_rows = []
    for target, prediction, error in rows:
        images = []
        for image in (target, prediction, error):
            uint8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil = Image.fromarray(uint8).resize((640, 360), Image.Resampling.LANCZOS)
            images.append(np.asarray(pil))
        rendered_rows.append(np.concatenate(images, axis=1))
    Image.fromarray(np.concatenate(rendered_rows, axis=0)).save(path)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference_center", default="0,0,0.8")
    parser.add_argument("--alpha_threshold", type=float, default=0.5)
    parser.add_argument("--min_ndotl", type=float, default=0.05)
    parser.add_argument(
        "--preserve_signed_normals",
        action="store_true",
        help="Do not match the 2DGS camera-facing normal orientation before calibration.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.alpha_threshold <= 1.0:
        parser.error("--alpha_threshold must lie in [0,1].")
    if not 0.0 <= args.min_ndotl <= 1.0:
        parser.error("--min_ndotl must lie in [0,1].")

    root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_center = _parse_center(args.reference_center)
    cameras = json.loads((root / "camera.json").read_text(encoding="utf-8"))
    lights = json.loads((root / "lights.json").read_text(encoding="utf-8"))
    frames = sorted(_frame_id(path) for path in (root / "image").glob("*.png"))
    if not frames:
        raise ValueError(f"No RGB PNG frames found under {root / 'image'}")

    sum_xy = 0.0
    sum_x2 = 0.0
    sum_xy_rgb = np.zeros(3, dtype=np.float64)
    sum_x2_rgb = np.zeros(3, dtype=np.float64)
    per_frame = []
    for frame in frames:
        data = _frame_data(
            root,
            frame,
            cameras,
            lights,
            reference_center,
            args.alpha_threshold,
            args.min_ndotl,
            not args.preserve_signed_normals,
        )
        x = data["base_linear"][data["valid"]].astype(np.float64)
        y = data["target_linear"][data["valid"]].astype(np.float64)
        intensity = fit_nonnegative_scalar(x, y)
        sum_xy += float((x * y).sum())
        sum_x2 += float(np.square(x).sum())
        sum_xy_rgb += (x * y).sum(axis=0)
        sum_x2_rgb += np.square(x).sum(axis=0)
        per_frame.append(
            {
                "frame": frame,
                "valid_pixels": int(data["valid"].sum()),
                "intensity": intensity,
                "legacy_albedo_ndotl_multiplier": intensity / math.pi,
                "direction_world_surface_to_light": data["direction_world"].tolist(),
                "direction_world_surface_to_camera": data["view_direction_world"].tolist(),
                "ndotl_mean_valid": float(data["ndotl"][data["valid"]].mean()),
            }
        )

    intensity = max(sum_xy / sum_x2, 0.0)
    intensity_rgb = np.divide(
        sum_xy_rgb,
        sum_x2_rgb,
        out=np.zeros_like(sum_xy_rgb),
        where=sum_x2_rgb > 0,
    )
    selected = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
    contact_rows = []
    foreground_psnr = []
    for frame in frames:
        data = _frame_data(
            root,
            frame,
            cameras,
            lights,
            reference_center,
            args.alpha_threshold,
            args.min_ndotl,
            not args.preserve_signed_normals,
        )
        prediction_linear = data["base_linear"] * intensity
        prediction_srgb = np.clip(linear_to_srgb(prediction_linear), 0.0, 1.0)
        foreground = data["albedo_alpha"] >= args.alpha_threshold
        foreground_psnr.append(
            _psnr(prediction_srgb[foreground], data["target_srgb"][foreground])
        )
        if frame in selected:
            mask = data["albedo_alpha"][..., None]
            target = data["target_srgb"] * mask
            prediction = prediction_srgb * mask
            error = np.abs(prediction - target)
            contact_rows.append((target, prediction, error))

    frame_intensities = np.asarray([row["intensity"] for row in per_frame])
    result = {
        "model": "target_linear = albedo_linear / pi * intensity * max(N dot L, 0)",
        "source_root": str(root),
        "reference_center_world": reference_center.tolist(),
        "normal_coordinate_space": "Blender world",
        "alpha_threshold": args.alpha_threshold,
        "min_ndotl": args.min_ndotl,
        "camera_facing_normal_orientation": not args.preserve_signed_normals,
        "frames": len(frames),
        "recommended_global_intensity": intensity,
        "equivalent_legacy_albedo_ndotl_multiplier": intensity / math.pi,
        "per_channel_intensity_diagnostic": intensity_rgb.tolist(),
        "per_frame_intensity": {
            "min": float(frame_intensities.min()),
            "median": float(np.median(frame_intensities)),
            "mean": float(frame_intensities.mean()),
            "p95": float(np.quantile(frame_intensities, 0.95)),
            "max": float(frame_intensities.max()),
        },
        "foreground_psnr_srgb": {
            "mean_over_frames": float(np.mean(foreground_psnr)),
            "min": float(np.min(foreground_psnr)),
            "max": float(np.max(foreground_psnr)),
        },
        "contact_sheet_columns": ["target", "calibrated_directional", "absolute_error"],
        "contact_sheet_frames": selected,
        "per_frame": per_frame,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    _contact_sheet(contact_rows, output_dir / "eval_contact_sheet.png")
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
