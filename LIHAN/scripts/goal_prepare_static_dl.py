#!/usr/bin/env python3
"""为静态 DL Goal 建立只依赖输入 RGBA、相机和训练光向量的原始数据视图。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"(\d+)(?=\.png$)")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shared-init-ply", type=Path, required=True,
                        help="由同一随机种子生成并复用的100k点初始点云")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    shared_init = args.shared_init_ply.expanduser().resolve()
    image_root = source / "image"
    image_paths = sorted(image_root.glob("*.png"))
    if len(image_paths) != 150:
        raise ValueError(f"预期150张原始RGBA输入图，实际为{len(image_paths)}张。")
    if output.exists():
        raise FileExistsError(f"输出已经存在，拒绝覆盖：{output}")
    if not shared_init.is_file():
        raise FileNotFoundError(f"共享随机点云不存在：{shared_init}")

    camera = read_json(source / "camera.json")
    lights = read_json(source / "lights.json").get("frames", {})
    camera_frames = camera.get("frames", {})
    resolution = camera.get("resolution", {})
    width, height = int(resolution.get("width", 0)), int(resolution.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("camera.json 分辨率无效。")

    frame_ids = []
    for path in image_paths:
        match = FRAME_RE.search(path.name)
        if match is None:
            raise ValueError(f"文件名缺少帧号：{path.name}")
        frame_ids.append(int(match.group(1)))
    if frame_ids != list(range(1, 151)):
        raise ValueError("静态 DL 输入必须是 frame 1..150，且不能缺帧或重复。")
    train_ids = [n for n in frame_ids if n % 10 != 0]
    test_ids = [n for n in frame_ids if n % 10 == 0]
    if len(train_ids) != 135 or len(test_ids) != 15:
        raise AssertionError("固定135/15数据划分校验失败。")

    records = {}
    train_directions = {}
    test_directions = {}
    input_hashes = {}
    for number, path in zip(frame_ids, image_paths):
        key = f"{number:04d}"
        if key not in camera_frames or key not in lights:
            raise ValueError(f"camera.json 或 lights.json 缺少帧 {key}。")
        with Image.open(path) as image:
            if image.mode != "RGBA" or image.size != (width, height):
                raise ValueError(f"输入图必须原样为{width}x{height} RGBA：{path}")
            # 仅使用原始image RGBA；清除全透明像素下不可见RGB以匹配既有读取口径。
            rgba = np.asarray(image, dtype=np.uint8).copy()
            alpha = rgba[..., 3]
            if not np.any(alpha):
                raise ValueError(f"输入Alpha为空：{path}")
            rgba[alpha == 0, :3] = 0
        frame = camera_frames[key]
        intrinsics = frame.get("intrinsics", {})
        extrinsics = frame.get("extrinsics", {})
        c2w = np.asarray(extrinsics.get("camera_to_world"), dtype=np.float64)
        w2c = np.asarray(extrinsics.get("world_to_camera"), dtype=np.float64)
        fx, fy = float(intrinsics.get("fx", 0)), float(intrinsics.get("fy", 0))
        fov_x, fov_y = float(intrinsics.get("fov_x_rad", 0)), float(intrinsics.get("fov_y_rad", 0))
        if c2w.shape != (4, 4) or w2c.shape != (4, 4) or not np.isfinite(c2w).all() or not np.isfinite(w2c).all():
            raise ValueError(f"帧{key}相机矩阵无效。")
        if not np.allclose(c2w @ w2c, np.eye(4), atol=1e-5):
            raise ValueError(f"帧{key} camera-to-world 与 world-to-camera 不互逆。")
        if fx <= 0 or fy <= 0 or not 0 < fov_x < math.pi or not 0 < fov_y < math.pi:
            raise ValueError(f"帧{key}相机内参无效。")
        records[number] = {
            "file_path": path.name,
            "time": float(frame["time_normalized"]),
            "time_seconds_from_first_frame": float(frame["time_seconds_from_first_frame"]),
            "source_frame": number,
            "camera_angle_x": fov_x,
            "camera_angle_y": fov_y,
            "fl_x": fx,
            "fl_y": fy,
            "cx": float(intrinsics.get("cx", width / 2)),
            "cy": float(intrinsics.get("cy", height / 2)),
            "transform_matrix": c2w.tolist(),
        }
        lamps = lights[key].get("lights", [])
        if not lamps:
            raise ValueError(f"帧{key}缺少方向光。")
        vectors = [np.asarray(lamp.get("light_dir_world"), dtype=np.float64) for lamp in lamps]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError(f"帧{key}方向光向量无效。")
        if not all(np.allclose(vectors[0], v, atol=1e-6) for v in vectors[1:]):
            raise ValueError(f"帧{key}多盏方向光方向不一致，不能简化为单一GT方向。")
        norm = float(np.linalg.norm(vectors[0]))
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError(f"帧{key}GT方向光不是单位向量：norm={norm}")
        if number in train_ids:
            train_directions[key] = {"light_dir_world": vectors[0].tolist()}
        else:
            test_directions[key] = {"light_dir_world": vectors[0].tolist()}
        input_hashes[key] = sha256(path)

    first = records[1]
    common = {
        "camera_angle_x": first["camera_angle_x"],
        "camera_angle_y": first["camera_angle_y"],
        "fl_x": first["fl_x"],
        "fl_y": first["fl_y"],
        "cx": first["cx"],
        "cy": first["cy"],
        "w": width,
        "h": height,
        "camera_extent": 1.0,
        "per_frame_intrinsics": True,
    }
    output.mkdir(parents=True)
    images_output = output / "images"
    images_output.mkdir()
    for number, path in zip(frame_ids, image_paths):
        with Image.open(path) as image:
            rgba = np.asarray(image, dtype=np.uint8).copy()
        rgba[rgba[..., 3] == 0, :3] = 0
        Image.fromarray(rgba, mode="RGBA").save(images_output / path.name)
    shutil.copy2(shared_init, output / "points3d.ply")
    write_json(output / "transforms_train.json", {**common, "frames": [records[n] for n in train_ids]})
    write_json(output / "transforms_test.json", {**common, "frames": [records[n] for n in test_ids]})
    write_json(output / "gt_directions_train.json", {
        "coordinate_space": "world",
        "direction_convention": "unit vector from surface toward light, for max(n dot l, 0)",
        "frames": train_directions,
    })
    evaluation_only = output.parent / "eval_only"
    if evaluation_only.exists():
        raise FileExistsError(f"评测专用目录已经存在，拒绝覆盖：{evaluation_only}")
    evaluation_only.mkdir()
    write_json(evaluation_only / "gt_directions_test.json", {
        "coordinate_space": "world",
        "direction_convention": "unit vector from surface toward light, for max(n dot l, 0)",
        "usage": "仅供测试渲染/评测；禁止训练读取",
        "frames": test_directions,
    })
    write_json(output / "input_manifest.json", {
        "status": "PASS",
        "dataset": "LH-static-jumpingjacks_DL_V2",
        "source": str(source),
        "image_source": "仅用image/*.png原始RGBA；全透明像素的不可见RGB置零以匹配历史读取口径；未读取材质图",
        "supervision_inputs": ["train RGB", "train image-alpha", "train frame/time", "train GT light direction"],
        "excluded_sources": ["albedo", "normal_exr", "depth", "roughness", "material", "bone_pose", "object_pose"],
        "train_frames": train_ids,
        "test_frames": test_ids,
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "resolution": [width, height],
        "camera_convention": "原始camera-to-world，未预补偿平移；逐帧内参原样保留",
        "gt_direction_train_count": len(train_directions),
        "initialization": {
            "method": "复用1011-01首次加载产生的唯一100000点随机初始点云",
            "random_seed": 0,
            "seed_source": "safe_state固定Python/NumPy种子；初始PLY已生成并作为各实验共享输入",
            "shared_source": str(shared_init),
            "sha256": sha256(shared_init),
            "vertex_count": 100000,
        },
        "image_sha256": input_hashes,
        "camera_json_sha256": sha256(source / "camera.json"),
        "lights_json_sha256": sha256(source / "lights.json"),
    })
    print(f"PASS: {len(train_ids)} train / {len(test_ids)} test; raw RGBA-derived inputs saved to {output / 'images'}")
    print(f"source={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
