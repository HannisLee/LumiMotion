"""Blender world-space GT normal 的通用加载工具。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import OpenEXR
import torch
import torch.nn.functional as F


def frame_id_from_name(name: str | Path) -> int:
    """从 frame_0001(.png/.exr) 一类名称中取出帧号。"""
    text = Path(name).name
    match = re.search(r"(\d+)(?:\.[^.]+)?$", text)
    if match is None:
        raise ValueError(f"Cannot find a frame number in {name}")
    return int(match.group(1))


def normal_paths(directory: Path) -> dict[int, Path]:
    """建立 source_frame -> EXR 路径索引，并拒绝重复帧。"""
    directory = Path(directory).expanduser().resolve()
    paths: dict[int, Path] = {}
    for path in sorted(directory.glob("*.exr")):
        frame = frame_id_from_name(path)
        if frame in paths:
            raise ValueError(f"Duplicate EXR normal frame {frame}: {path}")
        paths[frame] = path
    if not paths:
        raise ValueError(f"No EXR normals found in {directory}")
    return paths


def source_frame_by_image_name(source_path: Path) -> dict[str, int]:
    """从 train/test transforms 建立 camera image stem -> source_frame 映射。"""
    source_path = Path(source_path)
    mapping: dict[str, int] = {}
    for filename in ("transforms_train.json", "transforms_test.json"):
        path = source_path / filename
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)["frames"]
        for record in records:
            image_name = Path(record["file_path"]).stem
            source_frame = int(
                record.get("source_frame", frame_id_from_name(record["file_path"]))
            )
            previous = mapping.get(image_name)
            if previous is not None and previous != source_frame:
                raise ValueError(
                    f"Conflicting source_frame for {image_name}: "
                    f"{previous} vs {source_frame}"
                )
            mapping[image_name] = source_frame
    return mapping


def load_gt_normal(path: Path, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load LH Blender Normal-pass EXR as normalized world-space [3,H,W]."""
    channels = OpenEXR.File(str(path)).parts[0].channels
    names = ("Normal.X", "Normal.Y", "Normal.Z")
    if not all(name in channels for name in names):
        names = ("R", "G", "B")
    if not all(name in channels for name in names):
        raise ValueError(
            f"Expected Normal.X/Y/Z or RGB channels in {path}, "
            f"got {list(channels)}"
        )
    image = np.stack([channels[name].pixels for name in names], axis=-1)
    normal = torch.from_numpy(image).permute(2, 0, 1).float()
    normal = F.interpolate(
        normal[None], size=(height, width), mode="bilinear", align_corners=False
    )[0]
    valid = normal.norm(dim=0) > 1e-6
    return F.normalize(normal, dim=0), valid
