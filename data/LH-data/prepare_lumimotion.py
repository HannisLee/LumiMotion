#!/usr/bin/env python3
"""Convert LH-data scenes into LumiMotion Blender-format datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    lh_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Convert LH-data/origin scenes to LumiMotion's Blender transforms format under "
            "LH-data/transfer. With no scene arguments, every valid origin scene is converted."
        )
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        type=Path,
        help=(
            "Scene names under --origin-root or explicit scene directories. "
            "With no values, all direct children of --origin-root are converted."
        ),
    )
    parser.add_argument(
        "--origin-root",
        type=Path,
        default=lh_root / "origin",
        help="Directory containing original scene folders (default: %(default)s).",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=lh_root / "transfer",
        help="Directory receiving converted scene folders (default: %(default)s).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override the target scene folder name; valid only when converting one scene.",
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=8,
        help="Hold out every Nth frame for transforms_test.json; 0 disables holdout (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-source",
        choices=("image-background", "albedo-alpha", "image-alpha"),
        default="image-background",
        help="Source used to build the object alpha mask (default: %(default)s).",
    )
    parser.add_argument(
        "--background-threshold",
        type=int,
        default=1,
        help="Maximum per-channel deviation treated as image background (default: %(default)s).",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white", "keep"),
        default="black",
        help="RGB background written to generated RGBA images (default: %(default)s).",
    )
    parser.add_argument(
        "--camera-extent",
        type=float,
        default=1.0,
        help="Non-zero normalization radius used for fixed-camera training (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite known generated files if the output directory already exists.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate source data and print a summary without writing converted files.",
    )
    args = parser.parse_args()

    args.origin_root = args.origin_root.expanduser().resolve()
    args.transfer_root = args.transfer_root.expanduser().resolve()
    if not args.origin_root.is_dir():
        parser.error(f"--origin-root does not exist: {args.origin_root}")

    if not args.scenes:
        args.scenes = sorted(
            path
            for path in args.origin_root.iterdir()
            if path.is_dir() and (path / "camera.json").is_file()
        )
    else:
        resolved_scenes = []
        for value in args.scenes:
            candidate = value.expanduser()
            if not candidate.is_absolute() and not candidate.exists():
                candidate = args.origin_root / candidate
            resolved_scenes.append(candidate.resolve())
        args.scenes = resolved_scenes
    if not args.scenes:
        parser.error(f"No scenes containing camera.json were found under {args.origin_root}.")
    if args.output_name is not None and len(args.scenes) != 1:
        parser.error("--output-name can only be used when converting exactly one scene.")
    if args.output_name is not None and Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a single directory name, not a path.")
    if args.test_stride < 0:
        parser.error("--test-stride must be non-negative.")
    if not 0 <= args.background_threshold <= 255:
        parser.error("--background-threshold must be in [0, 255].")
    if not math.isfinite(args.camera_extent) or args.camera_extent <= 0:
        parser.error("--camera-extent must be a finite positive number.")
    return args


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def frame_number(path: Path) -> int:
    match = FRAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"File name has no numeric frame suffix: {path}")
    return int(match.group(1))


def index_files(directory: Path, extensions: set[str]) -> dict[int, Path]:
    if not directory.is_dir():
        raise ValueError(f"Required directory does not exist: {directory}")
    indexed: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        number = frame_number(path)
        if number in indexed:
            raise ValueError(f"Duplicate frame {number} in {directory}: {indexed[number].name}, {path.name}")
        indexed[number] = path
    if not indexed:
        raise ValueError(f"No supported files found in {directory}")
    return indexed


def most_common_border_color(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return colors[int(np.argmax(counts))]


def image_background_mask(rgb: np.ndarray, threshold: int) -> tuple[np.ndarray, list[int]]:
    background = most_common_border_color(rgb)
    distance = np.max(np.abs(rgb.astype(np.int16) - background.astype(np.int16)), axis=-1)
    alpha = np.where(distance > threshold, 255, 0).astype(np.uint8)
    return alpha, background.astype(int).tolist()


def load_mask(
    image_rgba: np.ndarray,
    albedo_path: Path | None,
    mask_source: str,
    threshold: int,
) -> tuple[np.ndarray, list[int] | None]:
    if mask_source == "image-background":
        return image_background_mask(image_rgba[..., :3], threshold)
    if mask_source == "image-alpha":
        return image_rgba[..., 3].copy(), None
    if albedo_path is None:
        raise ValueError("--mask-source albedo-alpha requires matching files in the albedo directory.")
    albedo_rgba = np.asarray(Image.open(albedo_path).convert("RGBA"), dtype=np.uint8)
    if albedo_rgba.shape[:2] != image_rgba.shape[:2]:
        raise ValueError(
            f"Image/albedo size mismatch: image={image_rgba.shape[:2]}, "
            f"albedo={albedo_rgba.shape[:2]} ({albedo_path})"
        )
    return albedo_rgba[..., 3].copy(), None


def composite_rgba(image_rgba: np.ndarray, alpha: np.ndarray, background: str) -> np.ndarray:
    rgb = image_rgba[..., :3].astype(np.float32)
    alpha_float = alpha.astype(np.float32)[..., None] / 255.0
    if background == "black":
        rgb = rgb * alpha_float
    elif background == "white":
        rgb = rgb * alpha_float + 255.0 * (1.0 - alpha_float)
    output = np.empty((*alpha.shape, 4), dtype=np.uint8)
    output[..., :3] = np.rint(rgb).clip(0, 255).astype(np.uint8)
    output[..., 3] = alpha
    return output


def validate_matrix(matrix: object, label: str) -> list[list[float]]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix.")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"{label} rotation is not orthonormal.")
    if np.linalg.det(rotation) <= 0:
        raise ValueError(f"{label} rotation must be right-handed.")
    return array.tolist()


def validate_frame_sets(
    scene: Path,
    images: dict[int, Path],
    albedos: dict[int, Path],
    normals: dict[int, Path],
    lights: dict,
    poses: dict | None,
) -> list[int]:
    frame_ids = sorted(images)
    expected = set(frame_ids)
    comparisons = {
        "albedo": set(albedos),
        "normal_exr": set(normals),
        "lights.json": {int(key) for key in lights},
    }
    if poses is not None:
        comparisons["object_pose.json"] = {int(key) for key in poses.get("frames", {})}
    for label, actual in comparisons.items():
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"{scene}: frame mismatch for {label}; missing={missing[:10]}, extra={extra[:10]}"
            )
    return frame_ids


def convert_scene(scene_arg: Path, args: argparse.Namespace) -> dict:
    scene = scene_arg.expanduser().resolve()
    camera_path = scene / "camera.json"
    lights_path = scene / "lights.json"
    poses_path = scene / "object_pose.json"
    for required in (camera_path, lights_path):
        if not required.is_file():
            raise ValueError(f"Required metadata is missing: {required}")

    camera = read_json(camera_path)
    lights = read_json(lights_path)
    poses = read_json(poses_path) if poses_path.is_file() else None
    images = index_files(scene / "image", IMAGE_EXTENSIONS)
    albedos = index_files(scene / "albedo", IMAGE_EXTENSIONS)
    normals = index_files(scene / "normal_exr", {".exr"})
    frame_ids = validate_frame_sets(scene, images, albedos, normals, lights, poses)

    resolution = camera.get("resolution", {})
    width = int(resolution.get("width", 0))
    height = int(resolution.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera resolution in {camera_path}")

    intrinsics = camera.get("intrinsics", {})
    fov_x = float(intrinsics.get("fov_x_rad", 0.0))
    fov_y = float(intrinsics.get("fov_y_rad", 0.0))
    if not 0 < fov_x < math.pi or not 0 < fov_y < math.pi:
        raise ValueError(f"Invalid camera FOV in {camera_path}")
    camera_to_world = validate_matrix(
        camera.get("extrinsics", {}).get("camera_to_world"),
        f"{camera_path}: camera_to_world",
    )

    first_size = Image.open(images[frame_ids[0]]).size
    if first_size != (width, height):
        raise ValueError(
            f"Camera/image resolution mismatch in {scene}: camera={(width, height)}, image={first_size}"
        )

    train_ids = []
    test_ids = []
    for ordinal, number in enumerate(frame_ids, start=1):
        if args.test_stride and ordinal % args.test_stride == 0:
            test_ids.append(number)
        else:
            train_ids.append(number)
    if not train_ids:
        raise ValueError("The requested split leaves no training frames.")

    summary = {
        "scene": str(scene),
        "frames": len(frame_ids),
        "train_frames": len(train_ids),
        "test_frames": len(test_ids),
        "resolution": [width, height],
        "fixed_camera": bool(camera.get("fixed_camera", False)),
        "has_object_pose": poses is not None,
        "fov_x_rad": fov_x,
        "fov_y_rad": fov_y,
        "mask_source": args.mask_source,
        "background": args.background,
        "camera_extent": args.camera_extent,
    }
    if args.validate_only:
        return summary

    output_name = args.output_name or scene.name
    output_root = (args.transfer_root / output_name).resolve()
    if output_root == scene or scene in output_root.parents:
        raise ValueError(f"Transfer output must not be inside the origin scene: {output_root}")
    images_root = output_root / "images"
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise ValueError(f"Output is not empty: {output_root}. Pass --overwrite to regenerate it.")
    images_root.mkdir(parents=True, exist_ok=True)

    generated_paths: dict[int, str] = {}
    background_colors: dict[str, list[int]] = {}
    mask_pixels: dict[str, int] = {}
    for number in frame_ids:
        source_rgba = np.asarray(Image.open(images[number]).convert("RGBA"), dtype=np.uint8)
        if source_rgba.shape[:2] != (height, width):
            raise ValueError(
                f"Unexpected image size for {images[number]}: {source_rgba.shape[:2]}, expected {(height, width)}"
            )
        alpha, detected_background = load_mask(
            source_rgba,
            albedos.get(number),
            args.mask_source,
            args.background_threshold,
        )
        output_rgba = composite_rgba(source_rgba, alpha, args.background)
        output_name = f"frame_{number:04d}.png"
        Image.fromarray(output_rgba, mode="RGBA").save(images_root / output_name)
        generated_paths[number] = output_name
        mask_pixels[f"{number:04d}"] = int(np.count_nonzero(alpha))
        if detected_background is not None:
            background_colors[f"{number:04d}"] = detected_background

    frame_position = {number: index for index, number in enumerate(frame_ids)}

    def frame_entry(number: int) -> dict:
        denominator = max(len(frame_ids) - 1, 1)
        return {
            "file_path": generated_paths[number],
            "time": frame_position[number] / denominator,
            "source_frame": number,
            "transform_matrix": camera_to_world,
        }

    common = {
        "camera_angle_x": fov_x,
        "camera_angle_y": fov_y,
        "fl_x": float(intrinsics.get("fx", 0.0)),
        "fl_y": float(intrinsics.get("fy", 0.0)),
        "cx": float(intrinsics.get("cx", width / 2.0)),
        "cy": float(intrinsics.get("cy", height / 2.0)),
        "w": width,
        "h": height,
        "camera_extent": args.camera_extent,
    }
    write_json(output_root / "transforms_train.json", {**common, "frames": [frame_entry(n) for n in train_ids]})
    write_json(output_root / "transforms_test.json", {**common, "frames": [frame_entry(n) for n in test_ids]})

    manifest = {
        "format": "LumiMotion Blender transforms",
        "source_scene": str(scene),
        "generated_root": str(output_root),
        "summary": summary,
        "split": {"train": train_ids, "test": test_ids, "test_stride": args.test_stride},
        "mask": {
            "source": args.mask_source,
            "background_threshold": args.background_threshold,
            "detected_background_rgb": background_colors,
            "nonzero_pixels": mask_pixels,
        },
        "source_metadata": {
            "camera": str(camera_path),
            "lights": str(lights_path),
            "object_pose": str(poses_path) if poses is not None else None,
            "albedo_directory": str(scene / "albedo"),
            "normal_exr_directory": str(scene / "normal_exr"),
        },
        "stage1_usage": {
            "used": ["generated RGB/RGBA images", "camera intrinsics", "camera_to_world", "normalized time"],
            "not_consumed_by_original_stage1": ["albedo RGB", "normal EXR", "lights.json", "object_pose.json"],
        },
    }
    write_json(output_root / "dataset_manifest.json", manifest)
    summary["output"] = str(output_root)
    return summary


def main() -> int:
    args = parse_args()
    results = []
    try:
        for scene in args.scenes:
            results.append(convert_scene(scene, args))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    mode = "Validated" if args.validate_only else "Converted"
    for result in results:
        print(
            f"{mode}: {result['scene']} | frames={result['frames']} "
            f"train={result['train_frames']} test={result['test_frames']} "
            f"resolution={result['resolution'][0]}x{result['resolution'][1]} "
            f"mask={result['mask_source']}"
        )
        if "output" in result:
            print(f"  output: {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
