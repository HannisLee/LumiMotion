#!/usr/bin/env python3
"""把训练得到的 photometric.pth 检查点导出为网页查看器可读的 JSON。

浏览器无法解析 PyTorch pickle 文件，因此先用本脚本抽取逐帧光线方向
（以及可选的 GT 点光源世界坐标），生成与 light_dirs.json 相同结构的
JSON，再拖入查看器即可。

用法示例：

    # 导出单个检查点（输出到同目录 photometric_perlight.json）
    python tools/export_photometric_pth.py /path/to/photometric.pth

    # 指定输出文件
    python tools/export_photometric_pth.py /path/to/photometric.pth -o CODE/data/perlight.json

    # 导出一个实验目录下所有 photometric.pth（递归搜索）
    python tools/export_photometric_pth.py /path/to/experiment_mlp --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _normalize(directions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("光线方向中存在零长度向量。")
    return directions / norms


def _find_state_dict(payload: dict) -> dict:
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError("检查点中找不到 state_dict。")
    return state_dict


def _get_tensor(state_dict: dict, key: str):
    for candidate_key, candidate_value in state_dict.items():
        if candidate_key == key or candidate_key.endswith("." + key):
            return candidate_value
    return None


def _orthonormal_tangents(direction):
    import torch.nn.functional as F
    direction = F.normalize(direction, dim=-1)
    x_axis = torch.zeros_like(direction)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(direction)
    y_axis[..., 1] = 1.0
    reference = torch.where((direction[..., :1].abs() < 0.9), x_axis, y_axis)
    tangent = F.normalize(torch.cross(direction, reference, dim=-1), dim=-1)
    bitangent = F.normalize(torch.cross(direction, tangent, dim=-1), dim=-1)
    return tangent, bitangent


def _reconstruct_structured_directions(state_dict: dict, config: dict) -> np.ndarray | None:
    """重建 StructuredDirectionalLightModel（傅里叶基 + 切向残差）的方向。"""
    import math
    import torch.nn.functional as F

    coefficients = _get_tensor(state_dict, "light_model.fourier_coefficients")
    basis = _get_tensor(state_dict, "light_model.fourier_basis")
    if coefficients is None or basis is None:
        return None
    coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
    basis = torch.as_tensor(basis, dtype=torch.float64)
    base = F.normalize(basis @ coefficients, dim=-1)

    residual = _get_tensor(state_dict, "light_model.raw_tangent_residual")
    if residual is not None:
        residual = torch.as_tensor(residual, dtype=torch.float64)
        max_angle_degrees = float(config.get("light_residual_angle_degrees", 10.0))
        tangent, bitangent = _orthonormal_tangents(base)
        coordinates = torch.tanh(residual)
        magnitude = torch.linalg.vector_norm(coordinates, dim=-1, keepdim=True)
        coordinates = coordinates / magnitude.clamp_min(1.0)
        displacement = math.tan(math.radians(max_angle_degrees)) * (
            coordinates[..., :1] * tangent + coordinates[..., 1:2] * bitangent
        )
        base = base + displacement
    return _normalize(_to_numpy(base))


def _extract_directions(state_dict: dict, config: dict) -> np.ndarray:
    for key in ("light_model._raw_light_dir_table", "raw_light_dir"):
        value = _get_tensor(state_dict, key)
        if value is not None:
            directions = _to_numpy(value)
            if directions.ndim != 2 or directions.shape[1] != 3:
                raise ValueError(f"光线方向形状异常: {directions.shape}")
            return _normalize(directions)
    structured = _reconstruct_structured_directions(state_dict, config)
    if structured is not None:
        return structured
    raise ValueError(
        "检查点中没有光线方向表（light_model._raw_light_dir_table / raw_light_dir / fourier_coefficients）。"
    )


def _extract_timesteps(payload: dict, state_dict: dict, expected: int) -> np.ndarray:
    for source in (payload.get("timesteps"), state_dict.get("light_model.timesteps")):
        if source is not None:
            times = _to_numpy(source).reshape(-1)
            if times.shape[0] == expected:
                return times
    return np.arange(expected, dtype=np.float64)


def export_checkpoint(checkpoint_path: Path, output_path: Path | None) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"检查点必须是字典: {checkpoint_path}")

    config = payload.get("config", {}) or {}
    initialization = payload.get("initialization", {}) or {}
    state_dict = _find_state_dict(payload)

    directions = _extract_directions(state_dict, config)
    timesteps = _extract_timesteps(payload, state_dict, directions.shape[0])

    gt_positions = None
    for candidate_key, candidate_value in state_dict.items():
        if candidate_key == "gt_light_positions" or candidate_key.endswith(".gt_light_positions"):
            gt_positions = _to_numpy(candidate_value)
            break
    if gt_positions is not None and gt_positions.shape != directions.shape:
        print(f"警告: GT 光源位置形状 {gt_positions.shape} 与方向不匹配，已忽略。", file=sys.stderr)
        gt_positions = None

    light_mode = config.get("light_mode") or initialization.get("light_mode") or "learned_directional"
    frames = []
    for index in range(directions.shape[0]):
        frame = {
            "index": index,
            "fid": float(timesteps[index]),
            "direction": directions[index].tolist(),
            "ray_direction_light_to_surface": directions[index].tolist(),
        }
        if gt_positions is not None:
            frame["light_position_world"] = gt_positions[index].tolist()
        frames.append(frame)

    export = {
        "photometric_version": payload.get("photometric_version", "unknown"),
        "light_mode": light_mode,
        "direction_convention": config.get(
            "direction_convention", initialization.get("direction_convention", "light_to_surface")
        ),
        "initialization": initialization,
        "source_checkpoint": str(checkpoint_path),
        "frames": frames,
    }

    if output_path is None:
        output_path = checkpoint_path.with_name("photometric_perlight.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="photometric.pth 文件，或包含若干 photometric.pth 的目录")
    parser.add_argument("-o", "--output", help="输出 JSON 路径（仅单文件模式有效）")
    parser.add_argument("--all", action="store_true", help="目录模式下导出全部检查点；默认只导出迭代号最大的一个")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"输入不存在: {input_path}")

    if input_path.is_file():
        output = export_checkpoint(input_path, Path(args.output).expanduser() if args.output else None)
        print(f"已导出: {output}")
        return

    checkpoints = sorted(input_path.rglob("photometric.pth"))
    if not checkpoints:
        raise SystemExit(f"目录中没有 photometric.pth: {input_path}")
    if not args.all:
        def iteration_key(path: Path) -> int:
            for part in reversed(path.parts):
                if part.startswith("iteration_"):
                    try:
                        return int(part.split("_", 1)[1])
                    except ValueError:
                        continue
            return -1
        checkpoints = [max(checkpoints, key=iteration_key)]
    for checkpoint in checkpoints:
        output = export_checkpoint(checkpoint, None)
        print(f"已导出: {output}")


if __name__ == "__main__":
    main()
