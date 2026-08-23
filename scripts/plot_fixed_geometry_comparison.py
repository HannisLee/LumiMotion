"""绘制固定几何 albedo-only 与自由几何训练的前 500 步对照。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_psnr(model_path: Path):
    accumulator = EventAccumulator(str(model_path))
    accumulator.Reload()
    return {
        item.step: item.value
        for item in accumulator.Scalars("train/reconstruct - psnr")
    }


def read_free_normals(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            int(row["iteration"]): float(row["mean_deg"])
            for row in csv.DictReader(handle)
        }


def read_fixed_normal(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["summary_mean_over_frames"]["mean_deg"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--free_model", type=Path, required=True)
    parser.add_argument("--fixed_model", type=Path, required=True)
    parser.add_argument("--free_normal_csv", type=Path, required=True)
    parser.add_argument("--fixed_normal_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    free_psnr = read_psnr(args.free_model)
    fixed_psnr = read_psnr(args.fixed_model)
    free_normal = read_free_normals(args.free_normal_csv)
    fixed_normal = read_fixed_normal(args.fixed_normal_json)
    iterations = sorted(set(free_psnr) & set(fixed_psnr) & set(free_normal))

    rows = [
        (iteration, free_psnr[iteration], fixed_psnr[iteration], free_normal[iteration], fixed_normal)
        for iteration in iterations
    ]
    with (args.output_dir / "fixed_vs_free.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("iteration", "free_psnr", "fixed_psnr", "free_normal_mean_deg", "fixed_normal_mean_deg"))
        writer.writerows(rows)

    data = np.asarray(rows, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(data[:, 0], data[:, 1], "o-", linewidth=2, label="Free geometry")
    axes[0].plot(data[:, 0], data[:, 2], "s-", linewidth=2, label="Fixed geometry, albedo only")
    axes[0].set(title="RGB reconstruction", xlabel="Iteration", ylabel="PSNR (dB)")
    axes[1].plot(data[:, 0], data[:, 3], "o-", linewidth=2, label="Free geometry")
    axes[1].plot(data[:, 0], data[:, 4], "s-", linewidth=2, label="Fixed geometry, albedo only")
    axes[1].set(title="World-space GT normal error", xlabel="Iteration", ylabel="Mean angular error (degree)")
    for axis in axes:
        axis.grid(alpha=0.28)
        axis.legend()
    fig.savefig(args.output_dir / "fixed_vs_free_psnr_normal.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
