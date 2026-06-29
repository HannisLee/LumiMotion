"""Compute light-direction loss between a predicted CSV and GT lights.json.

Compares learned photometric light directions (the six-column CSV produced by
``LH_Utils.export_light_directions``) against ground-truth light directions
parsed from ``lights.json``. The GT convention matches
``LH_Utils.plot_light_timeseries``: each entry's direction is
``(light_pos_world - target)`` (or ``light_dir_world`` if present), normalized,
sorted by integer key, then aligned row-by-row with the CSV.

Usage::

    python -m LH_Utils.light_direction_loss \
        --csv output/.../light_directions_it35000.csv \
        --lights_json data/LH-data/static/<scene>/lights.json \
        [--output result.txt] [--target 0 0 0] [--flip_learned] [--flip_gt] \
        [--per_frame]

The module also exposes :func:`compute_light_loss` for in-process batch use.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Loaders (GT convention shared with LH_Utils.plot_light_timeseries)
# --------------------------------------------------------------------------- #
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def load_learned_dirs(csv_path: Path) -> np.ndarray:
    """Load unit light directions from the six-column CSV (dir_x, dir_y, dir_z)."""
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
        return sorted(
            payload.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
        )
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


def load_gt_dirs(lights_json: Path, target: np.ndarray) -> np.ndarray:
    with lights_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    vectors = [extract_gt_vector(entry, target) for _, entry in sorted_light_items(payload)]
    if not vectors:
        raise SystemExit(f"No light entries found in {lights_json}")
    return normalize(np.asarray(vectors, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(learned: np.ndarray, gt: np.ndarray) -> dict:
    """Compare two [N, 3] unit-direction arrays (truncated to common length)."""
    n = min(len(learned), len(gt))
    L = learned[:n]
    G = gt[:n]
    dots = np.clip(np.sum(L * G, axis=-1), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(dots))
    diff = L - G
    return {
        "n_compared": int(n),
        "n_learned": int(len(learned)),
        "n_gt": int(len(gt)),
        "angular_mean_deg": float(ang_deg.mean()),
        "angular_median_deg": float(np.median(ang_deg)),
        "angular_std_deg": float(ang_deg.std()),
        "angular_max_deg": float(ang_deg.max()),
        "angular_min_deg": float(ang_deg.min()),
        "cosine_similarity_mean": float(dots.mean()),
        "cosine_distance_mean": float((1.0 - dots).mean()),
        "mse_xyz": float((diff ** 2).sum(axis=-1).mean()),
        "mae_xyz": float(np.abs(diff).mean()),
        "angular_per_frame_deg": ang_deg,
    }


def compute_light_loss(
    csv_path: Path | str,
    lights_json: Path | str,
    target: Iterable[float] = (0.0, 0.0, 0.0),
    flip_learned: bool = False,
    flip_gt: bool = False,
) -> dict:
    """Load both inputs, align, and return the metrics dict (no I/O side effects)."""
    learned = load_learned_dirs(Path(csv_path))
    gt = load_gt_dirs(Path(lights_json), np.asarray(list(target), dtype=np.float64))
    if flip_learned:
        learned = -learned
    if flip_gt:
        gt = -gt
    return compute_metrics(learned, gt)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(
    csv_path: Path,
    lights_json: Path,
    metrics: dict,
    flip_learned: bool,
    flip_gt: bool,
    per_frame: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("Light-direction loss report")
    lines.append("=" * 52)
    lines.append(f"csv (learned) : {csv_path}")
    lines.append(f"lights_json   : {lights_json}")
    flags = []
    if flip_learned:
        flags.append("flip_learned")
    if flip_gt:
        flags.append("flip_gt")
    lines.append(f"flags         : {', '.join(flags) if flags else '(none)'}")
    lines.append(
        f"n compared    : {metrics['n_compared']}  "
        f"(learned={metrics['n_learned']}, gt={metrics['n_gt']})"
    )
    lines.append("-" * 52)
    lines.append("Direction error metrics (lower is better):")
    lines.append(f"  cosine distance (1-cos) : {metrics['cosine_distance_mean']:11.6f}   <- primary loss")
    lines.append(f"  mean angular error      : {metrics['angular_mean_deg']:11.4f} deg")
    lines.append(f"  median angular error    : {metrics['angular_median_deg']:11.4f} deg")
    lines.append(f"  std angular error       : {metrics['angular_std_deg']:11.4f} deg")
    lines.append(f"  max angular error       : {metrics['angular_max_deg']:11.4f} deg")
    lines.append(f"  min angular error       : {metrics['angular_min_deg']:11.4f} deg")
    lines.append(f"  cosine similarity       : {metrics['cosine_similarity_mean']:11.6f}")
    lines.append(f"  MSE  (xyz, unit vec)    : {metrics['mse_xyz']:11.6f}")
    lines.append(f"  MAE  (xyz, unit vec)    : {metrics['mae_xyz']:11.6f}")
    if per_frame:
        ang = metrics["angular_per_frame_deg"]
        lines.append("-" * 52)
        lines.append("per-frame angular error (deg):")
        lines.append(f"{'idx':>4}  {'ang_deg':>9}")
        for i, a in enumerate(ang):
            lines.append(f"{i:4d}  {a:9.3f}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute light-direction loss between a predicted light_directions CSV "
            "and a GT lights.json. Accepts two file paths; writes a txt report "
            "(or prints to stdout when --output is omitted)."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Predicted light_directions_it*.csv.")
    parser.add_argument(
        "--lights_json",
        "--lights-json",
        dest="lights_json",
        type=Path,
        required=True,
        help="GT lights.json (light_pos_world is converted to a direction).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional txt report path. If omitted, the report is printed to stdout.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Target point for point-light GT directions (default: 0 0 0).",
    )
    parser.add_argument("--flip_learned", "--flip-learned", action="store_true", help="Negate learned directions.")
    parser.add_argument("--flip_gt", "--flip-gt", action="store_true", help="Negate GT directions.")
    parser.add_argument("--per_frame", "--per-frame", action="store_true", help="Include per-frame angular error table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    lights_json = args.lights_json.expanduser().resolve()
    metrics = compute_light_loss(
        csv_path,
        lights_json,
        target=args.target,
        flip_learned=args.flip_learned,
        flip_gt=args.flip_gt,
    )
    report = format_report(
        csv_path,
        lights_json,
        metrics,
        flip_learned=args.flip_learned,
        flip_gt=args.flip_gt,
        per_frame=args.per_frame,
    )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Wrote light loss report to {output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
