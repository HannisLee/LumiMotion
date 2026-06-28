"""Plot learned light directions as an XY polar trajectory, optionally with GT lights."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a polar XY projection from light_directions.csv. If --lights_json is "
            "provided, GT light_pos_world is converted to a direction and plotted for comparison."
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


def polar_xy(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arctan2(directions[:, 1], directions[:, 0])
    radius = np.linalg.norm(directions[:, :2], axis=-1)
    return theta, np.clip(radius, 0.0, 1.0)


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

    output = args.output.expanduser().resolve() if args.output else csv_path.with_name("light_polar_compare.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    frames = np.arange(len(learned))
    theta, radius = polar_xy(learned)
    ax.plot(theta, radius, color="#2563eb", linewidth=1.5, label="learned")
    sc = ax.scatter(theta, radius, c=frames, cmap="viridis", s=18, zorder=3)

    if gt is not None:
        gt_theta, gt_radius = polar_xy(gt)
        ax.plot(gt_theta, gt_radius, color="#dc2626", linewidth=1.5, linestyle="--", label="gt lights.json")
        ax.scatter(gt_theta, gt_radius, color="#dc2626", s=10, alpha=0.45)
        err = angular_error_deg(learned, gt)
        title = args.title or f"Light direction polar XY, mean angular error {err.mean():.2f} deg"
    else:
        title = args.title or "Learned light direction polar XY"

    ax.set_title(title)
    ax.set_rlim(0.0, 1.0)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.08))
    fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.75, label="frame index")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    output = plot(parse_args())
    print(f"Wrote polar plot to {output}")


if __name__ == "__main__":
    main()
