#!/usr/bin/env python3
"""Visualize learned and GT Stage 1 Lambertian directions on a unit sphere.

Every ``light_to_surface`` vector is normalized and plotted on the same unit
sphere. This deliberately removes scene scale and point-light distance so the
angular trajectory difference is directly visible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go


DIRECTION_CONVENTION = "light_to_surface"
DEFAULT_REPRESENTATIVE_FRAMES = (0, 59, 119)
# The first group preserves the original world-space source visualization.  The
# second group shows the revised unit-sphere direction visualization.
WORLD_SPACE_VIEWS = (
    ("01", "World-space perspective", 24, -56, "perspective"),
    ("02", "World-space top: X-Y", 90, -90, "top_xy"),
    ("03", "World-space side: X-Z", 0, -90, "side_xz"),
)
UNIT_SPHERE_VIEWS = (
    ("04", "Unit sphere perspective", 24, -56, "perspective"),
    ("05", "Unit sphere top: X-Y", 90, -90, "top_xy"),
    ("06", "Unit sphere side: X-Z", 0, -90, "side_xz"),
)


@dataclass(frozen=True)
class LightCheckpoint:
    """Light information restored directly from a photometric checkpoint."""

    path: Path
    directions: np.ndarray
    timesteps: np.ndarray
    gt_positions: np.ndarray | None
    reference_center: np.ndarray | None
    version: str
    light_mode: str


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return result


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only existed.
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"Photometric checkpoint must be a dictionary: {path}")
    return loaded


def load_light_checkpoint(path: str | Path) -> LightCheckpoint:
    """Load a Stage 1 Lambertian light table without constructing GPU modules."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Photometric checkpoint does not exist: {checkpoint_path}")
    payload = _torch_load(checkpoint_path)
    config = payload.get("config", {})
    initialization = payload.get("initialization", {})
    if not isinstance(config, dict) or not isinstance(initialization, dict):
        raise ValueError(f"Malformed config/initialization block: {checkpoint_path}")
    convention = config.get("direction_convention", initialization.get("direction_convention"))
    if convention != DIRECTION_CONVENTION:
        raise ValueError(
            f"Unsupported direction convention {convention!r} in {checkpoint_path}; "
            f"expected {DIRECTION_CONVENTION!r}."
        )

    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Malformed state_dict in {checkpoint_path}")
    raw_directions = state_dict.get("light_model._raw_light_dir_table")
    if raw_directions is None:
        raw_directions = state_dict.get("raw_light_dir")  # Legacy v1 checkpoints.
    if raw_directions is None:
        raise ValueError(f"Missing per-frame light directions in {checkpoint_path}")
    directions = _to_numpy(raw_directions, "light directions")
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] == 0:
        raise ValueError(f"Expected light directions shaped [T,3], got {directions.shape}")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"Light directions contain a zero-length vector: {checkpoint_path}")
    directions = directions / norms

    raw_timesteps = payload.get("timesteps", state_dict.get("light_model.timesteps"))
    if raw_timesteps is None:
        raise ValueError(f"Missing timestep table in {checkpoint_path}")
    timesteps = _to_numpy(raw_timesteps, "timesteps").reshape(-1)
    if timesteps.shape[0] != directions.shape[0]:
        raise ValueError(
            f"Timestep/light count mismatch in {checkpoint_path}: "
            f"{timesteps.shape[0]} vs {directions.shape[0]}"
        )

    gt_positions = state_dict.get("gt_light_positions")
    if gt_positions is not None:
        gt_positions = _to_numpy(gt_positions, "GT light positions")
        if gt_positions.shape != directions.shape:
            raise ValueError(
                f"GT position/light shape mismatch in {checkpoint_path}: "
                f"{gt_positions.shape} vs {directions.shape}"
            )
    reference_center = initialization.get("reference_center")
    if reference_center is not None:
        reference_center = _to_numpy(reference_center, "reference center").reshape(-1)
        if reference_center.shape != (3,):
            raise ValueError(
                f"Reference center must be [3], got {reference_center.shape} in {checkpoint_path}"
            )

    return LightCheckpoint(
        path=checkpoint_path,
        directions=directions,
        timesteps=timesteps,
        gt_positions=gt_positions,
        reference_center=reference_center,
        version=str(payload.get("photometric_version", "unknown")),
        light_mode=str(config.get("light_mode", "learned_directional")),
    )


def load_gt_lights_directory(dataset_dir: str | Path) -> LightCheckpoint:
    """Load GT point-light positions directly from a transferred dataset.

    The transfer manifest keeps the original ``lights.json`` path.  This
    avoids requiring a learned photometric checkpoint for GT-only plots.
    """
    dataset_path = Path(dataset_dir).expanduser().resolve()
    manifest_path = dataset_path / "dataset_manifest.json"
    light_path = dataset_path / "lights.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest.get("source_metadata", {}).get("lights")
        if source:
            source_path = Path(source)
            candidates = [source_path, Path(str(source).replace("/mnt/workspace/users/han.li/lumimotion/", str(Path.cwd()) + "/"))]
            light_path = next((candidate for candidate in candidates if candidate.is_file()), light_path)
    if not light_path.is_file():
        raise FileNotFoundError(f"GT lights.json not found below {dataset_path}")
    payload = json.loads(light_path.read_text(encoding="utf-8"))
    entries = sorted(payload.items(), key=lambda item: int(item[0]))
    positions = _to_numpy([entry["light_pos_world"] for _, entry in entries], "GT light positions")
    center = np.zeros(3, dtype=np.float64)
    reference_center = manifest.get("reference_center") if manifest_path.is_file() else None
    if reference_center is not None:
        center = _to_numpy(reference_center, "reference center").reshape(3)
    vectors = center[None, :] - positions
    distances = np.linalg.norm(vectors, axis=1)
    directions = vectors / distances[:, None]
    return LightCheckpoint(dataset_path, directions, np.arange(len(entries)), positions, center, "lights.json", "gt_point")


def make_gt_only_comparison(gt: LightCheckpoint) -> dict[str, Any]:
    distances = np.linalg.norm(gt.reference_center[None, :] - gt.gt_positions, axis=1)
    return {
        "center": gt.reference_center,
        "timesteps": gt.timesteps,
        "learned_directions": gt.directions,
        "gt_directions": gt.directions,
        "gt_positions": gt.gt_positions,
        "learned_virtual_positions": gt.gt_positions,
        "gt_distances": distances,
        "virtual_radius": float(np.median(distances)),
        "angular_errors": np.zeros_like(distances),
        "angular_errors_deg": np.zeros_like(distances),
        "gt_only": True,
    }


def parse_representative_frames(value: str, num_frames: int) -> tuple[int, ...]:
    frames = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not frames:
        raise ValueError("At least one representative frame is required.")
    if len(set(frames)) != len(frames):
        raise ValueError("Representative frames must be unique.")
    if any(frame < 0 or frame >= num_frames for frame in frames):
        raise ValueError(f"Representative frames must be within [0, {num_frames - 1}].")
    return frames


def compare_light_trajectories(
    learned: LightCheckpoint,
    gt: LightCheckpoint,
) -> dict[str, Any]:
    """Build a common-center, world-space learned-versus-GT comparison."""

    if gt.gt_positions is None or gt.reference_center is None:
        raise ValueError("GT checkpoint must contain gt_light_positions and initialization.reference_center.")
    if learned.directions.shape != gt.directions.shape:
        raise ValueError(
            f"Learned/GT light shape mismatch: {learned.directions.shape} vs {gt.directions.shape}"
        )
    if not np.allclose(learned.timesteps, gt.timesteps, rtol=1e-6, atol=1e-6):
        raise ValueError("Learned and GT checkpoints do not have aligned timesteps.")

    center = gt.reference_center
    gt_vectors = center[None, :] - gt.gt_positions
    gt_distances = np.linalg.norm(gt_vectors, axis=1)
    if np.any(gt_distances <= np.finfo(np.float64).eps):
        raise ValueError("A GT light coincides with the reference center.")
    gt_directions = gt_vectors / gt_distances[:, None]
    virtual_radius = float(np.median(gt_distances))
    learned_virtual_positions = center[None, :] - learned.directions * virtual_radius
    dots = np.clip(np.sum(learned.directions * gt_directions, axis=1), -1.0, 1.0)
    angular_errors = np.degrees(np.arccos(dots))

    return {
        "center": center,
        "timesteps": learned.timesteps,
        "learned_directions": learned.directions,
        "gt_directions": gt_directions,
        "gt_positions": gt.gt_positions,
        "learned_virtual_positions": learned_virtual_positions,
        "gt_distances": gt_distances,
        "virtual_radius": virtual_radius,
        "angular_errors_deg": angular_errors,
    }


def _set_unit_sphere_limits(ax: Any) -> None:
    radius = 1.08
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))


def _set_world_space_limits(ax: Any, comparison: dict[str, Any]) -> None:
    all_positions = np.vstack((
        comparison["gt_positions"],
        comparison["learned_virtual_positions"],
        comparison["center"][None, :],
    ))
    lower = all_positions.min(axis=0)
    upper = all_positions.max(axis=0)
    radius = max(float(np.max(upper - lower)) * 0.58, 0.1)
    midpoint = (lower + upper) * 0.5
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    ax.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _add_unit_sphere(ax: Any) -> None:
    azimuth = np.linspace(0.0, 2.0 * np.pi, 44)
    polar = np.linspace(0.0, np.pi, 24)
    azimuth, polar = np.meshgrid(azimuth, polar)
    ax.plot_wireframe(
        np.cos(azimuth) * np.sin(polar),
        np.sin(azimuth) * np.sin(polar),
        np.cos(polar),
        rstride=2, cstride=3, color="#a9a9a9", linewidth=0.35, alpha=0.35,
    )


def _add_direction_trajectory(ax: Any, directions: np.ndarray, colors: np.ndarray, label: str, marker: str) -> None:
    segments = np.stack((directions[:-1], directions[1:]), axis=1)
    ax.add_collection3d(Line3DCollection(segments, colors=colors[:-1], linewidths=1.4, alpha=0.75))
    ax.scatter(
        directions[:, 0], directions[:, 1], directions[:, 2], c=colors, s=16,
        marker=marker, depthshade=False, label=label,
    )


def _add_representative_direction_arrows(
    ax: Any,
    directions: np.ndarray,
    frames: tuple[int, ...],
    color: str,
    label: str,
) -> None:
    for index, frame in enumerate(frames):
        direction = directions[frame]
        ax.quiver(
            0.0, 0.0, 0.0, direction[0] * 0.94, direction[1] * 0.94, direction[2] * 0.94,
            color=color, linewidth=2.0, arrow_length_ratio=0.08,
            label=label if index == 0 else None,
        )
        ax.text(direction[0], direction[1], direction[2], f" t={frame}", fontsize=8)


def _add_representative_world_arrows(
    axis: Any,
    positions: np.ndarray,
    center: np.ndarray,
    frames: tuple[int, ...],
    color: str,
    label: str,
) -> None:
    for index, frame in enumerate(frames):
        position = positions[frame]
        vector = center - position
        axis.quiver(
            position[0], position[1], position[2], vector[0], vector[1], vector[2],
            color=color, linewidth=1.8, arrow_length_ratio=0.08,
            label=label if index == 0 else None,
        )
        axis.text(position[0], position[1], position[2], f" t={frame}", fontsize=8)


def _draw_world_space_view(
    axis: Any,
    comparison: dict[str, Any],
    frames: tuple[int, ...],
    elevation: int,
    azimuth: int,
    title: str,
    *,
    show_legend: bool,
) -> None:
    """Draw the original world-coordinate source trajectory comparison."""

    num_frames = comparison["timesteps"].shape[0]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, num_frames))
    gt_positions = comparison["gt_positions"]
    _add_direction_trajectory(axis, gt_positions, colors, "GT light position", "^")
    if not comparison.get("gt_only"):
        learned_positions = comparison["learned_virtual_positions"]
        _add_direction_trajectory(axis, learned_positions, colors, "Learned virtual source", "o")
    center = comparison["center"]
    axis.scatter(
        center[0], center[1], center[2], c="#111111", marker="*", s=110,
        depthshade=False, label="Reference center",
    )
    _add_representative_world_arrows(
        axis, gt_positions, center, frames, "#e76f51", "GT light→center",
    )
    if not comparison.get("gt_only"):
        _add_representative_world_arrows(
            axis, learned_positions, center, frames, "#2878b5", "Learned virtual→center",
        )
    _set_world_space_limits(axis, comparison)
    axis.set_xlabel("World X")
    axis.set_ylabel("World Y")
    axis.set_zlabel("World Z")
    axis.set_title(title)
    axis.view_init(elev=elevation, azim=azimuth)
    if show_legend:
        axis.legend(loc="upper left", fontsize=8)


def _draw_unit_sphere_view(
    axis: Any,
    comparison: dict[str, Any],
    frames: tuple[int, ...],
    elevation: int,
    azimuth: int,
    title: str,
    *,
    show_legend: bool,
) -> None:
    """Draw one numbered projection with the same data and visual semantics."""

    num_frames = comparison["timesteps"].shape[0]
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, num_frames))
    _add_unit_sphere(axis)
    _add_direction_trajectory(axis, comparison["gt_directions"], colors, "GT direction", "^")
    if not comparison.get("gt_only"):
        _add_direction_trajectory(axis, comparison["learned_directions"], colors, "Learned direction", "o")
    axis.scatter(
        0.0, 0.0, 0.0, c="#111111", marker="*", s=110, depthshade=False,
        label="Unit-sphere origin",
    )
    _add_representative_direction_arrows(
        axis, comparison["gt_directions"], frames, "#e76f51", "GT light→surface",
    )
    if not comparison.get("gt_only"):
        _add_representative_direction_arrows(
            axis, comparison["learned_directions"], frames, "#2878b5", "Learned light→surface",
        )
    _set_unit_sphere_limits(axis)
    axis.set_xlabel("Direction X")
    axis.set_ylabel("Direction Y")
    axis.set_zlabel("Direction Z")
    axis.set_title(title)
    axis.view_init(elev=elevation, azim=azimuth)
    if show_legend:
        axis.legend(loc="upper left", fontsize=8)


def write_numbered_views(
    output_dir: Path,
    comparison: dict[str, Any],
    frames: tuple[int, ...],
) -> None:
    """Export original world-space and revised unit-sphere views by index."""

    plt.style.use("seaborn-v0_8-whitegrid")
    for index, view_name, elevation, azimuth, filename_suffix in WORLD_SPACE_VIEWS:
        figure = plt.figure(figsize=(8.5, 8.0), constrained_layout=True)
        axis = figure.add_subplot(1, 1, 1, projection="3d")
        _draw_world_space_view(
            axis,
            comparison,
            frames,
            elevation,
            azimuth,
            f"{index} · {view_name}",
            show_legend=True,
        )
        figure.suptitle(
            "V2 Stage 1 Lambertian: learned virtual and GT light sources in world space",
            fontsize=13,
        )
        figure.savefig(
            output_dir / f"light_position_world_{index}_{filename_suffix}.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(figure)

    for index, view_name, elevation, azimuth, filename_suffix in UNIT_SPHERE_VIEWS:
        figure = plt.figure(figsize=(8.5, 8.0), constrained_layout=True)
        axis = figure.add_subplot(1, 1, 1, projection="3d")
        _draw_unit_sphere_view(
            axis,
            comparison,
            frames,
            elevation,
            azimuth,
            f"{index} · {view_name}",
            show_legend=True,
        )
        figure.suptitle(
            "V2 Stage 1 Lambertian: learned vs GT directions on the unit sphere",
            fontsize=13,
        )
        figure.savefig(
            output_dir / f"light_direction_unit_sphere_{index}_{filename_suffix}.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(figure)


def write_contact_sheet(
    output_path: Path,
    comparison: dict[str, Any],
    frames: tuple[int, ...],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(19, 12.5), constrained_layout=True)
    all_views = tuple(("world", *view) for view in WORLD_SPACE_VIEWS) + tuple(
        ("unit_sphere", *view) for view in UNIT_SPHERE_VIEWS
    )
    for plot_index, (view_kind, index, view_name, elevation, azimuth, _) in enumerate(all_views, start=1):
        axis = figure.add_subplot(2, 3, plot_index, projection="3d")
        draw_view = _draw_world_space_view if view_kind == "world" else _draw_unit_sphere_view
        draw_view(axis, comparison, frames, elevation, azimuth, f"{index} · {view_name}", show_legend=plot_index == 1)
    figure.suptitle("V2 Stage 1 Lambertian: world-space sources (01–03) and unit-sphere directions (04–06)", fontsize=15)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_interactive_html(
    output_path: Path,
    comparison: dict[str, Any],
    frames: tuple[int, ...],
) -> None:
    num_frames = comparison["timesteps"].shape[0]
    time_colors = np.linspace(0.0, 1.0, num_frames)
    gt_directions = comparison["gt_directions"]
    learned_directions = comparison["learned_directions"]
    azimuth = np.linspace(0.0, 2.0 * np.pi, 44)
    polar = np.linspace(0.0, np.pi, 24)
    azimuth, polar = np.meshgrid(azimuth, polar)

    figure = go.Figure()
    figure.add_trace(go.Surface(
        x=np.cos(azimuth) * np.sin(polar), y=np.sin(azimuth) * np.sin(polar), z=np.cos(polar),
        showscale=False, colorscale=[[0.0, "#d3d3d3"], [1.0, "#d3d3d3"]], opacity=0.18,
        hoverinfo="skip", name="Unit sphere",
    ))
    for directions, name, symbol in (
        (gt_directions, "GT direction", "diamond"),
        (learned_directions, "Learned direction", "circle"),
    ):
        figure.add_trace(go.Scatter3d(
            x=directions[:, 0], y=directions[:, 1], z=directions[:, 2], mode="lines+markers",
            line={"color": "#555555", "width": 3},
            marker={"size": 3.6, "symbol": symbol, "color": time_colors, "colorscale": "Turbo", "showscale": name.startswith("GT"), "colorbar": {"title": "Normalized time"}},
            name=name,
        ))
    figure.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0], mode="markers",
        marker={"size": 7, "symbol": "diamond", "color": "#111111"}, name="Unit-sphere origin",
    ))
    for directions, color, name in (
        (gt_directions, "#e76f51", "GT selected arrows"),
        (learned_directions, "#2878b5", "Learned selected arrows"),
    ):
        figure.add_trace(go.Cone(
            x=np.zeros(len(frames)), y=np.zeros(len(frames)), z=np.zeros(len(frames)),
            u=directions[list(frames), 0] * 0.94, v=directions[list(frames), 1] * 0.94, w=directions[list(frames), 2] * 0.94,
            anchor="tail", showscale=False, colorscale=[[0.0, color], [1.0, color]],
            sizemode="absolute", sizeref=0.18, name=name,
        ))

    # Empty highlight traces are updated by the animation slider.
    figure.add_trace(go.Cone(name="GT frame highlight", showscale=False))
    figure.add_trace(go.Cone(name="Learned frame highlight", showscale=False))
    animation_frames = []
    for frame in range(num_frames):
        traces = []
        for directions, color in ((gt_directions, "#e76f51"), (learned_directions, "#2878b5")):
            traces.append(go.Cone(
                x=[0.0], y=[0.0], z=[0.0],
                u=[directions[frame, 0] * 0.96], v=[directions[frame, 1] * 0.96], w=[directions[frame, 2] * 0.96],
                anchor="tail", showscale=False, colorscale=[[0.0, color], [1.0, color]],
                sizemode="absolute", sizeref=0.2,
            ))
        animation_frames.append(go.Frame(name=str(frame), data=traces, traces=[5, 6]))
    figure.frames = animation_frames
    figure.update_layout(
        title="V2 learned and GT light-to-surface directions on the unit sphere",
        scene={"xaxis_title": "Direction X", "yaxis_title": "Direction Y", "zaxis_title": "Direction Z", "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Frame: "},
            "steps": [{"label": str(frame), "method": "animate", "args": [[str(frame)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]} for frame in range(num_frames)],
        }],
    )
    figure.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def write_metrics(
    output_path: Path,
    comparison: dict[str, Any],
    frames: tuple[int, ...],
    learned: LightCheckpoint,
    gt: LightCheckpoint,
) -> dict[str, Any]:
    errors = comparison["angular_errors_deg"]
    metrics = {
        "direction_convention": DIRECTION_CONVENTION,
        "reference_center_world": comparison["center"].tolist(),
        "virtual_light_radius": comparison["virtual_radius"],
        "frame_count": int(errors.shape[0]),
        "angle_error_degrees": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "min": float(errors.min()),
            "max": float(errors.max()),
        },
        "gt_distance": {
            "mean": float(comparison["gt_distances"].mean()),
            "min": float(comparison["gt_distances"].min()),
            "max": float(comparison["gt_distances"].max()),
        },
        "representative_frames": [
            {
                "index": frame,
                "fid": float(comparison["timesteps"][frame]),
                "angle_error_deg": float(errors[frame]),
                "gt_light_position_world": comparison["gt_positions"][frame].tolist(),
                "learned_virtual_source_world": comparison["learned_virtual_positions"][frame].tolist(),
                "learned_light_to_surface": comparison["learned_directions"][frame].tolist(),
                "gt_light_to_surface": comparison["gt_directions"][frame].tolist(),
            }
            for frame in frames
        ],
        "inputs": {
            "learned_photometric": str(learned.path),
            "gt_photometric": str(gt.path),
            "learned_photometric_version": learned.version,
            "gt_photometric_version": gt.version,
        },
    }
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def write_readme(output_path: Path, args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    summary = metrics["angle_error_degrees"]
    representative = ", ".join(str(frame["index"]) for frame in metrics["representative_frames"])
    output_path.write_text(
        "# only_clothV2 light direction 3D diagnostic\n\n"
        "此目录为只读 checkpoint 诊断产物；不修改训练、渲染或既有 checkpoint。\n\n"
        "## 输入\n\n"
        f"- Learned checkpoint: `{args.learned_photometric}`\n"
        f"- GT checkpoint: `{args.gt_photometric}`\n"
        f"- Direction convention: `{DIRECTION_CONVENTION}`\n"
        f"- Representative frames: `{representative}`\n\n"
        "## 生成命令\n\n"
        "```bash\n"
        "conda run --no-capture-output -n lumimotion-garuda \\\n"
        "  python -m scripts.visualize_stage1_light_trajectory \\\n"
        f"  --learned-photometric {args.learned_photometric} \\\n"
        f"  --gt-photometric {args.gt_photometric} \\\n"
        f"  --output-dir {args.output_dir}\n"
        "```\n\n"
        "## 读图方式\n\n"
        "- `light_position_world_01_perspective.png` 至 `light_position_world_03_side_xz.png`：恢复的世界坐标光源轨迹图。GT 三角为真实灯光位置；learned 圆点是以 GT 距离中位数构造的虚拟光源，只用于将纯方向权重放到相同空间中比较。\n"
        "- `light_direction_unit_sphere_04_perspective.png` 至 `light_direction_unit_sphere_06_side_xz.png`：新版单位球方向箭头图；每个点均是归一化的 `light_to_surface` 向量。\n"
        "- `light_direction_3d_contact_sheet.png`：两组图的 2×3 总览；`01`–`03` 为世界坐标，`04`–`06` 为单位球。可直接按编号指定保留或删除。\n"
        "- `light_direction_3d.html`：可旋转的单位球视图；时间滑块高亮单帧 GT 与 learned 箭头。\n"
        "- GT 三角与 learned 圆点分别是权重中的方向表；箭头从单位球原点指向球面，因此表示自由向量的 `light_to_surface` 方向，而非灯光位置。\n\n"
        "## 定量结论\n\n"
        f"- Learned relative to GT reference direction: mean `{summary['mean']:.2f}°`, median `{summary['median']:.2f}°`, P95 `{summary['p95']:.2f}°`.\n"
        "- 该图比较 reference center 的整体入射方向；不替代每个变形 Gaussian 的局部点光方向。\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-photometric", help="Path to learned-light photometric.pth")
    parser.add_argument("--gt-photometric", help="Path to GT-point-light photometric.pth")
    parser.add_argument("--gt-data-dir", help="Transferred dataset directory; plot GT light directions only")
    parser.add_argument("--point-cloud", help="Deprecated compatibility option; unit-sphere visualization does not use geometry")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG, HTML, JSON and README")
    parser.add_argument("--sample-points", type=int, default=5000, help="Deprecated compatibility option")
    parser.add_argument(
        "--representative-frames", default=",".join(map(str, DEFAULT_REPRESENTATIVE_FRAMES)),
        help="Comma-separated zero-based frame indices used for arrows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gt_data_dir:
        gt = load_gt_lights_directory(args.gt_data_dir)
        learned = gt
        comparison = make_gt_only_comparison(gt)
    else:
        if not args.learned_photometric or not args.gt_photometric:
            raise SystemExit("请提供 --gt-data-dir，或同时提供 --learned-photometric 和 --gt-photometric")
        learned = load_light_checkpoint(args.learned_photometric)
        gt = load_light_checkpoint(args.gt_photometric)
        comparison = compare_light_trajectories(learned, gt)
    frames = parse_representative_frames(args.representative_frames, comparison["timesteps"].shape[0])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    write_numbered_views(output_dir, comparison, frames)
    write_contact_sheet(output_dir / "light_direction_3d_contact_sheet.png", comparison, frames)
    write_interactive_html(output_dir / "light_direction_3d.html", comparison, frames)
    metrics = write_metrics(output_dir / "light_direction_metrics.json", comparison, frames, learned, gt)
    write_readme(output_dir / "README.md", args, metrics)
    print(f"Output: {output_dir}")
    print(f"Mean learned-vs-GT angle: {metrics['angle_error_degrees']['mean']:.3f} degrees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
