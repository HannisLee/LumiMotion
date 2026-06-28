"""Plot learned light direction time curves, optionally compared with GT lights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot x/y/z light direction curves from light_directions.csv. If --lights_json "
            "is provided, GT light_pos_world is converted to a direction and overlaid."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV from LH_Utils.export_light_directions.")
    parser.add_argument(
        "--lights_json",
        "--lights-json",
        dest="lights_json",
        type=Path,
        default=None,
        help="Optional original lights.json for GT comparison.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Target point used for point-light GT directions (default: 0 0 0).",
    )
    parser.add_argument("--flip_learned", "--flip-learned", action="store_true", help="Flip learned directions.")
    parser.add_argument("--flip_gt", "--flip-gt", action="store_true", help="Flip GT directions.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser.parse_args()


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def load_learned_dirs(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"dir_x", "dir_y", "dir_z"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing CSV columns in {csv_path}: {sorted(missing)}")
        rows = [[float(row["dir_x"]), float(row["dir_y"]), float(row["dir_z"])] for row in reader]
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")
    return normalize(np.asarray(rows, dtype=np.float64))


def sorted_light_items(payload: object) -> list[tuple[str, dict]]:
    if isinstance(payload, dict):
        return sorted(payload.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
    if isinstance(payload, list):
        return [(str(index), value) for index, value in enumerate(payload)]
    raise SystemExit(f"Unsupported lights.json payload type: {type(payload)!r}")


def extract_gt_vector(entry: dict, target: np.ndarray) -> np.ndarray:
    for key in ("light_dir_world", "light_direction_world", "direction", "light_dir"):
        if key in entry:
            return np.asarray(entry[key], dtype=np.float64)
    for key in ("light_pos_world", "light_position_world", "position", "light_pos"):
        if key in entry:
            return np.asarray(entry[key], dtype=np.float64) - target
    raise SystemExit(f"Cannot find light direction or position in lights.json entry keys: {sorted(entry)}")


def load_gt_dirs(lights_json: Path, target: np.ndarray, limit: int) -> np.ndarray:
    with lights_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    vectors = [extract_gt_vector(entry, target) for _, entry in sorted_light_items(payload)]
    if not vectors:
        raise SystemExit(f"No light entries found in {lights_json}")
    if len(vectors) != limit:
        print(f"Warning: GT light count {len(vectors)} != learned count {limit}; truncating to common length.")
    n = min(len(vectors), limit)
    return normalize(np.asarray(vectors[:n], dtype=np.float64))


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    dots = np.sum(a[:n] * b[:n], axis=-1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def step_angle_deg(directions: np.ndarray) -> np.ndarray:
    if len(directions) < 2:
        return np.zeros((len(directions),), dtype=np.float64)
    dots = np.sum(directions[1:] * directions[:-1], axis=-1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return np.concatenate([[0.0], angles])


def plot(args: argparse.Namespace) -> Path:
    csv_path = args.csv.expanduser().resolve()
    learned = load_learned_dirs(csv_path)
    if args.flip_learned:
        learned = -learned

    gt = None
    if args.lights_json is not None:
        gt = load_gt_dirs(args.lights_json.expanduser().resolve(), np.asarray(args.target), len(learned))
        if args.flip_gt:
            gt = -gt
        learned = learned[: len(gt)]

    output = args.output.expanduser().resolve() if args.output else csv_path.with_name("light_timeseries_compare.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = np.arange(len(learned))
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    labels = ("x", "y", "z")
    colors = ("#2563eb", "#16a34a", "#9333ea")

    for axis, label, color, component in zip(axes[:3], labels, colors, range(3)):
        axis.plot(frames, learned[:, component], color=color, linewidth=1.6, label=f"learned {label}")
        if gt is not None:
            axis.plot(frames, gt[:, component], color="#dc2626", linewidth=1.2, linestyle="--", label=f"gt {label}")
        axis.set_ylabel(label)
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=2)

    if gt is not None:
        err = angular_error_deg(learned, gt)
        axes[3].plot(frames, err, color="#ea580c", linewidth=1.6, label="learned vs gt angular error")
        axes[3].set_ylabel("deg")
        axes[3].set_title(f"Mean angular error: {err.mean():.2f} deg")
    else:
        angles = step_angle_deg(learned)
        axes[3].plot(frames, angles, color="#ea580c", linewidth=1.6, label="learned adjacent-frame angle")
        axes[3].set_ylabel("deg")
        axes[3].set_title(f"Mean adjacent angle: {angles[1:].mean() if len(angles) > 1 else 0.0:.2f} deg")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("frame index")

    fig.suptitle(args.title or "Light direction time series", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    output = plot(parse_args())
    print(f"Wrote time-series plot to {output}")


if __name__ == "__main__":
    main()
