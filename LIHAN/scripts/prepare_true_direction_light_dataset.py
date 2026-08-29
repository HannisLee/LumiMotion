#!/usr/bin/env python3
"""将真实方向光 LH 数据转换为可由既有 GT 方向光管线直接读取的数据集。

本脚本不改动训练、渲染或数据加载代码。它做三件事：

1. 调用 ``scripts/prepare_lh_dynamic.py`` 生成标准 Blender transforms/RGBA 数据集；
2. 将仅含 xyz 的 ``Point3D.ply`` 补成 loader 所需的 xyz/normal/RGB ``points3d.ply``；
3. 将 ``light_dir_world``（surface-to-light）写成旧 ``gt_directional`` 接口所需的
   虚拟 ``light_pos_world``，并验证两者在旧 Lambertian 管线中的方向严格一致。

虚拟灯位仅供 ``--photometric_light_mode gt_directional`` 使用；不能作为 ``gt_point``
的物理点光源，因为后者会施加距离衰减。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=_path, help="原始真实方向光数据集目录")
    parser.add_argument("output", type=_path, help="必须为空的转换数据集目录")
    parser.add_argument("--test-stride", type=int, default=8)
    parser.add_argument("--camera-extent", type=float, default=1.0)
    parser.add_argument(
        "--virtual-light-distance",
        type=float,
        default=10.0,
        help="虚拟灯位到初始化点云均值的距离；gt_directional 下只影响可视化，不影响着色方向",
    )
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"源数据集不存在：{args.source}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"输出目录非空，拒绝覆盖：{args.output}")
    if args.test_stride < 0:
        parser.error("--test-stride 必须非负")
    if not math.isfinite(args.camera_extent) or args.camera_extent <= 0:
        parser.error("--camera-extent 必须为有限正数")
    if not math.isfinite(args.virtual_light_distance) or args.virtual_light_distance <= 0:
        parser.error("--virtual-light-distance 必须为有限正数")
    return args


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_xyz(source_ply: Path) -> np.ndarray:
    ply = PlyData.read(source_ply)
    if "vertex" not in ply:
        raise ValueError(f"PLY 缺少 vertex 元素：{source_ply}")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"PLY 缺少 xyz 属性：{source_ply}")
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    if xyz.size == 0 or not np.isfinite(xyz).all():
        raise ValueError(f"PLY 的 xyz 必须非空且有限：{source_ply}")
    return xyz


def write_loader_ply(path: Path, xyz: np.ndarray) -> None:
    """写入当前 fetchPly() 所要求的完整属性，而不改变原 xyz。"""
    vertex = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["nx"], vertex["ny"], vertex["nz"] = 0.0, 0.0, 0.0
    vertex["red"], vertex["green"], vertex["blue"] = 128, 128, 128
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def normalize(values: np.ndarray, label: str) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(lengths <= 1e-12):
        raise ValueError(f"{label} 含非有限或零长度方向")
    return values / lengths


def make_legacy_lights(
    source_lights: dict,
    reference_center: np.ndarray,
    virtual_distance: float,
) -> tuple[dict, dict]:
    """以 P=C+D*r 构造虚拟点，使旧管线得到的 surface-to-light 正好为 D。

    旧 gt_directional 首先计算 R=normalize(C-P)，随后 shading 使用 -R。
    原始 light_dir_world 的语义是 surface-to-light D；因此 P=C+D*r。
    """
    try:
        keys = sorted(source_lights, key=lambda key: int(key))
    except ValueError as exc:
        raise ValueError("lights.json 的帧键必须可转为整数") from exc
    if not keys:
        raise ValueError("lights.json 为空")

    output: dict[str, dict] = {}
    gt_directions = []
    positions = []
    for key in keys:
        entry = source_lights[key]
        if "light_dir_world" not in entry:
            raise ValueError(f"帧 {key} 缺少 light_dir_world")
        direction = normalize(
            np.asarray(entry["light_dir_world"], dtype=np.float64).reshape(1, -1),
            f"帧 {key} 的 light_dir_world",
        )
        if direction.shape != (1, 3):
            raise ValueError(f"帧 {key} 的 light_dir_world 必须是长度 3")
        direction = direction[0]
        position = reference_center + virtual_distance * direction
        output[key] = {
            "type": "directional_compat_virtual_point",
            "source_type": entry.get("source_type", "SUN"),
            "light_pos_world": position.tolist(),
            "light_rgb": entry.get("light_rgb", [1.0, 1.0, 1.0]),
            "intensity": entry.get("intensity"),
            "light_dir_world": direction.tolist(),
            "compatibility": {
                "source_direction_convention": "surface_to_light",
                "legacy_mode_required": "gt_directional",
                "virtual_light_distance": virtual_distance,
                "distance_attenuation": "disabled_by_gt_directional",
            },
        }
        gt_directions.append(direction)
        positions.append(position)

    gt = np.asarray(gt_directions, dtype=np.float64)
    positions_array = np.asarray(positions, dtype=np.float64)
    # 与 scene.photometric_lambertian 的实际旧路径逐项对应：
    # raw_ray = normalize(reference_center - light_pos_world);
    # surface_to_light = -raw_ray.
    recovered_surface_to_light = -normalize(
        reference_center[None] - positions_array,
        "虚拟灯位恢复方向",
    )
    cosine = np.clip(np.sum(gt * recovered_surface_to_light, axis=1), -1.0, 1.0)
    angular_error_deg = np.degrees(np.arccos(cosine))
    # 模拟旧管线的 torch float32 读取与归一化，作为实际运行精度的验收值。
    center32 = reference_center.astype(np.float32)
    positions32 = positions_array.astype(np.float32)
    recovered32 = -(center32[None] - positions32)
    recovered32 /= np.linalg.norm(recovered32, axis=1, keepdims=True)
    gt32 = gt.astype(np.float32)
    gt32 /= np.linalg.norm(gt32, axis=1, keepdims=True)
    cosine32 = np.clip(np.sum(gt32 * recovered32, axis=1), -1.0, 1.0)
    angular_error32_deg = np.degrees(np.arccos(cosine32))
    report = {
        "status": "PASS" if float(angular_error32_deg.max()) <= 0.03 else "FAILED",
        "frames": len(keys),
        "reference_center_from_input_ply_mean": reference_center.tolist(),
        "virtual_light_distance": virtual_distance,
        "input_direction_convention": "surface_to_light",
        "legacy_gt_directional_raw_ray_convention": "light_to_surface",
        "legacy_shading_direction": "surface_to_light = -normalize(reference_center - light_pos_world)",
        "formula": "light_pos_world = reference_center + virtual_light_distance * normalize(light_dir_world)",
        "float64_angular_error_deg": {
            "mean": float(angular_error_deg.mean()),
            "max": float(angular_error_deg.max()),
        },
        "float32_legacy_path_angular_error_deg": {
            "mean": float(angular_error32_deg.mean()),
            "max": float(angular_error32_deg.max()),
            "threshold_max": 0.03,
        },
        "usage": "仅可与 --photometric_light_mode gt_directional 配合；不可使用 gt_point。",
    }
    return output, report


def validate_transforms(output: Path) -> dict:
    manifest = read_json(output / "dataset_manifest.json")
    train = read_json(output / "transforms_train.json")["frames"]
    test = read_json(output / "transforms_test.json")["frames"]
    files = [output / "images" / frame["file_path"] for frame in [*train, *test]]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"转换图片缺失：{missing[:3]}")
    if len(train) != manifest["summary"]["train_frames"] or len(test) != manifest["summary"]["test_frames"]:
        raise ValueError("transforms 帧数与 manifest 不一致")
    return {"train_frames": len(train), "test_frames": len(test), "image_files": len(files)}


def main() -> int:
    args = parse_args()
    source_ply = args.source / "Point3D.ply"
    source_lights_path = args.source / "lights.json"
    for path in (source_ply, source_lights_path):
        if not path.is_file():
            raise ValueError(f"必需输入不存在：{path}")

    xyz = read_xyz(source_ply)
    reference_center = xyz.mean(axis=0)
    source_lights = read_json(source_lights_path)
    legacy_lights, light_report = make_legacy_lights(
        source_lights,
        reference_center,
        args.virtual_light_distance,
    )
    if light_report["status"] != "PASS":
        raise RuntimeError(f"虚拟灯位方向验证失败：{light_report}")

    repository_root = Path(__file__).resolve().parents[2]
    converter = repository_root / "scripts" / "prepare_lh_dynamic.py"
    if not converter.is_file():
        raise FileNotFoundError(f"找不到既有转换器：{converter}")
    command = [
        sys.executable,
        str(converter),
        str(args.source),
        str(args.output),
        "--test-stride",
        str(args.test_stride),
        "--camera-extent",
        str(args.camera_extent),
    ]
    subprocess.run(command, check=True)

    raw_ply_backup = args.output / "points3d_xyz_only.ply.bak"
    shutil.copy2(source_ply, raw_ply_backup)
    write_loader_ply(args.output / "points3d.ply", xyz)
    write_json(args.output / "lights_compat_point_position.json", legacy_lights)
    transforms_report = validate_transforms(args.output)
    report = {
        "status": "PASS",
        "source": str(args.source),
        "output": str(args.output),
        "point_cloud": {
            "source": str(source_ply),
            "raw_xyz_backup": str(raw_ply_backup),
            "loader_ply": str(args.output / "points3d.ply"),
            "points": int(len(xyz)),
            "xyz_preserved": True,
            "added_properties": {"normal": [0.0, 0.0, 0.0], "rgb": [128, 128, 128]},
        },
        "transforms": transforms_report,
        "legacy_light_compatibility": light_report,
    }
    write_json(args.output / "conversion_validation.json", report)
    print(f"PASS: 输出数据集：{args.output}")
    print(f"PASS: 兼容灯光：{args.output / 'lights_compat_point_position.json'}")
    print(
        "PASS: float32 旧方向路径最大角误差 "
        f"{light_report['float32_legacy_path_angular_error_deg']['max']:.8f}°"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
