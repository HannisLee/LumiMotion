"""绘制 Stage1 前期法线误差与逐损失梯度审计图。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PARAMETERS = ("rotation", "position", "scale", "opacity", "albedo")
LOSSES = ("total", "rgb_l1", "rgb_dssim", "alpha", "normal", "distortion")


def read_gradients(path: Path):
    values = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["parameter_group"], row["loss_term"])
            values[key].append(
                (
                    int(row["iteration"]),
                    float(row["gradient_rms"]),
                    float(row["gradient_l2"]),
                    float(row["lr_times_gradient_l2"]),
                    float(row["learning_rate"]),
                )
            )
    return values


def read_normals(root: Path):
    rows = []
    for path in root.glob("ours_*/normal_metrics.json"):
        iteration = int(path.parent.name.split("_")[-1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary_mean_over_frames"]
        rows.append(
            (
                iteration,
                float(summary["mean_deg"]),
                float(summary["median_deg"]),
                float(summary["p95_deg"]),
            )
        )
    return sorted(rows)


def save_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_normal(rows, output_dir: Path):
    data = np.asarray(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(data[:, 0], data[:, 1], "o-", linewidth=2.2, label="Mean")
    ax.plot(data[:, 0], data[:, 2], "s-", linewidth=1.8, label="Median")
    ax.plot(data[:, 0], data[:, 3], "^-", linewidth=1.8, label="P95")
    ax.set(title="World-space GT normal error, first 500 steps", xlabel="Iteration", ylabel="Angular error (degree)")
    ax.grid(alpha=0.28)
    ax.legend()
    fig.savefig(output_dir / "normal_error_first500.png", dpi=220)
    plt.close(fig)


def plot_loss_gradients(values, output_dir: Path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, constrained_layout=True)
    colors = {"total": "black", "rgb_l1": "tab:blue", "rgb_dssim": "tab:orange", "alpha": "tab:green", "normal": "tab:red", "distortion": "tab:purple"}
    for ax, parameter in zip(axes.flat, PARAMETERS):
        for loss in LOSSES:
            rows = values.get((parameter, loss), ())
            if not rows:
                continue
            data = np.asarray(rows, dtype=np.float64)
            ax.plot(data[:, 0], np.maximum(data[:, 1], 1e-14), label=loss, color=colors[loss], linewidth=2 if loss == "total" else 1.2)
        ax.set(title=parameter, ylabel="Gradient RMS (log)", yscale="log")
        ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    axes[2, 0].set_xlabel("Iteration")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Weighted loss gradients by Gaussian parameter group")
    fig.savefig(output_dir / "loss_gradient_by_parameter.png", dpi=220)
    plt.close(fig)


def plot_effective_update(values, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for parameter in PARAMETERS:
        rows = values.get((parameter, "total"), ())
        if not rows:
            continue
        data = np.asarray(rows, dtype=np.float64)
        ax.plot(data[:, 0], np.maximum(data[:, 3], 1e-14), label=parameter, linewidth=1.8)
    ax.set(title="LR-scaled total gradient norm", xlabel="Iteration", ylabel="Learning rate x gradient L2 (log)", yscale="log")
    ax.grid(alpha=0.28)
    ax.legend(ncol=3)
    fig.savefig(output_dir / "lr_scaled_total_gradient.png", dpi=220)
    plt.close(fig)


def plot_combined(normal_rows, values, output_dir: Path):
    normal = np.asarray(normal_rows, dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    axes[0].plot(normal[:, 0], normal[:, 1], "o-", linewidth=2.2, label="Mean normal error")
    axes[0].plot(normal[:, 0], normal[:, 2], "s-", linewidth=1.6, label="Median")
    axes[0].set(ylabel="Angular error (degree)", title="Normal collapse and optimization gradients")
    axes[0].grid(alpha=0.28)
    axes[0].legend()
    for parameter in PARAMETERS:
        rows = values.get((parameter, "total"), ())
        data = np.asarray(rows, dtype=np.float64)
        axes[1].plot(data[:, 0], np.maximum(data[:, 1], 1e-14), label=parameter, linewidth=1.6)
    axes[1].set(xlabel="Iteration", ylabel="Total gradient RMS (log)", yscale="log")
    axes[1].grid(alpha=0.28)
    axes[1].legend(ncol=3)
    fig.savefig(output_dir / "normal_and_total_gradient_first500.png", dpi=220)
    plt.close(fig)


def read_parameter_drift(point_cloud_root: Path, iterations):
    from plyfile import PlyData

    def load(iteration):
        vertex = PlyData.read(
            point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
        )["vertex"].data
        xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1)
        rotation = np.stack([vertex[f"rot_{index}"] for index in range(4)], axis=1)
        rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)
        scale = np.stack([vertex[f"scale_{index}"] for index in range(2)], axis=1)
        opacity = np.asarray(vertex["opacity"])
        albedo = np.stack(
            [vertex[f"photometric_albedo_raw_{index}"] for index in range(3)], axis=1
        )
        return xyz, rotation, scale, opacity, albedo

    reference = load(min(iterations))
    rows = []
    for iteration in iterations:
        current = load(iteration)
        cosine = np.abs(np.sum(reference[1] * current[1], axis=1)).clip(-1.0, 1.0)
        rotation = np.degrees(2.0 * np.arccos(cosine))
        position = np.linalg.norm(current[0] - reference[0], axis=1)
        scale = np.abs(current[2] - reference[2]).mean(axis=1)
        opacity = np.abs(current[3] - reference[3])
        albedo = np.abs(current[4] - reference[4]).mean(axis=1)
        rows.append((iteration, rotation.mean(), np.quantile(rotation, 0.95), position.mean(), np.quantile(position, 0.95), scale.mean(), opacity.mean(), albedo.mean()))
    return rows


def plot_parameter_drift(rows, output_dir: Path):
    data = np.asarray(rows, dtype=np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].plot(data[:, 0], data[:, 1], "o-", label="Mean")
    axes[0].plot(data[:, 0], data[:, 2], "s-", label="P95")
    axes[0].set(title="Rotation drift", xlabel="Iteration", ylabel="Degree")
    axes[0].legend()
    axes[1].plot(data[:, 0], data[:, 3], "o-", label="Mean")
    axes[1].plot(data[:, 0], data[:, 4], "s-", label="P95")
    axes[1].set(title="Position drift", xlabel="Iteration", ylabel="L2")
    axes[1].legend()
    for column, label in ((5, "log-scale"), (6, "opacity logit"), (7, "albedo logit")):
        axes[2].plot(data[:, 0], np.maximum(data[:, column], 1e-12), "o-", label=label)
    axes[2].set(title="Mean raw-parameter drift", xlabel="Iteration", ylabel="Absolute drift (log)", yscale="log")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.28)
    fig.savefig(output_dir / "parameter_drift_first500.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient_csv", type=Path, required=True)
    parser.add_argument("--normal_root", type=Path, required=True)
    parser.add_argument("--point_cloud_root", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gradients = read_gradients(args.gradient_csv)
    normals = read_normals(args.normal_root)
    if not normals:
        raise RuntimeError(f"No normal_metrics.json found below {args.normal_root}")
    save_csv(args.output_dir / "normal_trajectory.csv", ("iteration", "mean_deg", "median_deg", "p95_deg"), normals)
    plot_normal(normals, args.output_dir)
    plot_loss_gradients(gradients, args.output_dir)
    plot_effective_update(gradients, args.output_dir)
    plot_combined(normals, gradients, args.output_dir)
    if args.point_cloud_root is not None:
        drift = read_parameter_drift(args.point_cloud_root, [row[0] for row in normals])
        save_csv(
            args.output_dir / "parameter_drift.csv",
            ("iteration", "rotation_mean_deg", "rotation_p95_deg", "position_mean", "position_p95", "log_scale_mean_abs", "opacity_logit_mean_abs", "albedo_logit_mean_abs"),
            drift,
        )
        plot_parameter_drift(drift, args.output_dir)


if __name__ == "__main__":
    main()
