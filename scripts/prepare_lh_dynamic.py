#!/usr/bin/env python3
"""Convert an LH dynamic scene with per-frame cameras to LumiMotion transforms."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--test-stride", type=int, default=8)
    parser.add_argument("--camera-extent", type=float, default=1.0)
    args = parser.parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.source.is_dir():
        parser.error(f"Source directory does not exist: {args.source}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output directory is not empty: {args.output}")
    if args.test_stride < 0:
        parser.error("--test-stride must be non-negative")
    if not math.isfinite(args.camera_extent) or args.camera_extent <= 0:
        parser.error("--camera-extent must be finite and positive")
    return args


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"File name has no frame suffix: {path}")
    return int(match.group(1))


def index_files(directory: Path, suffixes: set[str]) -> dict[int, Path]:
    if not directory.is_dir():
        raise ValueError(f"Required directory does not exist: {directory}")
    result: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        number = frame_number(path)
        if number in result:
            raise ValueError(f"Duplicate frame {number} in {directory}")
        result[number] = path
    if not result:
        raise ValueError(f"No supported files found in {directory}")
    return result


def validate_matrix(value: object, label: str) -> list[list[float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"{label} rotation is not orthonormal")
    if np.linalg.det(rotation) <= 0:
        raise ValueError(f"{label} rotation must be right-handed")
    return matrix.tolist()


def main() -> int:
    args = parse_args()
    source = args.source
    camera_path = source / "camera.json"
    lights_path = source / "lights.json"
    poses_path = source / "object_pose.json"
    for required in (camera_path, lights_path, poses_path):
        if not required.is_file():
            raise ValueError(f"Required metadata is missing: {required}")

    camera = read_json(camera_path)
    camera_frames = camera.get("frames", {})
    lights = read_json(lights_path)
    poses = read_json(poses_path).get("frames", {})
    images = index_files(source / "image", {".png", ".jpg", ".jpeg"})
    albedos = index_files(source / "albedo", {".png", ".jpg", ".jpeg"})
    normals = index_files(source / "normal_exr", {".exr"})
    frame_ids = sorted(images)
    expected = set(frame_ids)
    sources = {
        "albedo": set(albedos),
        "normal_exr": set(normals),
        "camera.json": {int(key) for key in camera_frames},
        "lights.json": {int(key) for key in lights},
        "object_pose.json": {int(key) for key in poses},
    }
    for label, actual in sources.items():
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Frame mismatch for {label}: missing={missing[:10]}, extra={extra[:10]}")

    resolution = camera.get("resolution", {})
    width = int(resolution.get("width", 0))
    height = int(resolution.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("camera.json has an invalid resolution")

    train_ids = []
    test_ids = []
    for ordinal, number in enumerate(frame_ids, start=1):
        (test_ids if args.test_stride and ordinal % args.test_stride == 0 else train_ids).append(number)
    if not train_ids:
        raise ValueError("The requested split leaves no training frames")

    args.output.mkdir(parents=True, exist_ok=True)
    images_root = args.output / "images"
    images_root.mkdir(exist_ok=True)
    nonzero_pixels: dict[str, int] = {}
    soft_pixels: dict[str, int] = {}
    camera_centers = []
    frame_records: dict[int, dict] = {}

    for number in frame_ids:
        key = f"{number:04d}"
        image = np.asarray(Image.open(images[number]).convert("RGBA"), dtype=np.uint8)
        albedo = np.asarray(Image.open(albedos[number]).convert("RGBA"), dtype=np.uint8)
        if image.shape[:2] != (height, width) or albedo.shape[:2] != (height, width):
            raise ValueError(f"Unexpected image size at frame {number}")
        alpha = albedo[..., 3]
        rgb = np.rint(image[..., :3].astype(np.float32) * (alpha[..., None] / 255.0)).astype(np.uint8)
        rgba = np.concatenate((rgb, alpha[..., None]), axis=-1)
        output_name = f"frame_{number:04d}.png"
        Image.fromarray(rgba, mode="RGBA").save(images_root / output_name)
        nonzero_pixels[key] = int(np.count_nonzero(alpha))
        soft_pixels[key] = int(np.count_nonzero((alpha > 0) & (alpha < 255)))

        camera_frame = camera_frames[key]
        intrinsics = camera_frame.get("intrinsics", {})
        extrinsics = camera_frame.get("extrinsics", {})
        fx = float(intrinsics.get("fx", 0.0))
        fy = float(intrinsics.get("fy", 0.0))
        fov_x = float(intrinsics.get("fov_x_rad", 0.0))
        fov_y = float(intrinsics.get("fov_y_rad", 0.0))
        if fx <= 0 or fy <= 0 or not 0 < fov_x < math.pi or not 0 < fov_y < math.pi:
            raise ValueError(f"Invalid intrinsics at frame {number}")
        camera_to_world = validate_matrix(extrinsics.get("camera_to_world"), f"camera frame {number}")
        camera_centers.append(np.asarray(camera_to_world, dtype=np.float64)[:3, 3])
        frame_records[number] = {
            "file_path": output_name,
            "time": (number - frame_ids[0]) / max(frame_ids[-1] - frame_ids[0], 1),
            "source_frame": number,
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "fl_x": fx,
            "fl_y": fy,
            "cx": float(intrinsics.get("cx", width / 2.0)),
            "cy": float(intrinsics.get("cy", height / 2.0)),
            "transform_matrix": camera_to_world,
        }

    first = frame_records[frame_ids[0]]
    common = {
        "camera_angle_x": first["camera_angle_x"],
        "camera_angle_y": first["camera_angle_y"],
        "fl_x": first["fl_x"],
        "fl_y": first["fl_y"],
        "cx": first["cx"],
        "cy": first["cy"],
        "w": width,
        "h": height,
        "camera_extent": args.camera_extent,
        "per_frame_intrinsics": True,
    }
    write_json(args.output / "transforms_train.json", {**common, "frames": [frame_records[n] for n in train_ids]})
    write_json(args.output / "transforms_test.json", {**common, "frames": [frame_records[n] for n in test_ids]})

    centers = np.asarray(camera_centers)
    center_mean = centers.mean(axis=0)
    computed_radius = float(np.linalg.norm(centers - center_mean, axis=1).max() * 1.1)
    manifest = {
        "format": "LumiMotion Blender transforms with per-frame cameras",
        "source_scene": str(source),
        "generated_root": str(args.output),
        "summary": {
            "frames": len(frame_ids),
            "train_frames": len(train_ids),
            "test_frames": len(test_ids),
            "resolution": [width, height],
            "per_frame_intrinsics": True,
            "unique_camera_matrices": len({np.asarray(frame_records[n]["transform_matrix"]).round(7).tobytes() for n in frame_ids}),
            "mask_source": "albedo-alpha",
            "background": "black",
            "computed_camera_radius": computed_radius,
            "camera_extent_fallback": args.camera_extent,
        },
        "split": {"train": train_ids, "test": test_ids, "test_stride": args.test_stride},
        "mask": {
            "source": "albedo-alpha",
            "nonzero_pixels": nonzero_pixels,
            "soft_pixels": soft_pixels,
        },
        "source_metadata": {
            "camera": str(camera_path),
            "lights": str(lights_path),
            "object_pose": str(poses_path),
            "albedo_directory": str(source / "albedo"),
            "normal_exr_directory": str(source / "normal_exr"),
        },
        "stage1_usage": {
            "used": ["generated RGB/RGBA images", "per-frame camera intrinsics", "per-frame camera_to_world", "albedo soft alpha", "normalized time"],
            "not_consumed_by_stage1": ["albedo RGB", "normal EXR", "lights.json", "object_pose.json"],
        },
    }
    write_json(args.output / "dataset_manifest.json", manifest)
    print(f"Converted {len(frame_ids)} frames: train={len(train_ids)}, test={len(test_ids)}")
    print(f"Output: {args.output}")
    print(f"Computed camera radius: {computed_radius:.6f}")
    print(f"Mask coverage: {min(nonzero_pixels.values()) / (width * height):.3%} - {max(nonzero_pixels.values()) / (width * height):.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
