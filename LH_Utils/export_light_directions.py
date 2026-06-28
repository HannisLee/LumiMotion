"""Export Stage 1 photometric light directions to a six-column CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from scene.photometric_lambertian import DirectionalLightModel


CSV_COLUMNS = ("raw_x", "raw_y", "raw_z", "dir_x", "dir_y", "dir_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw_light_dir and normalized light_dir from a photometric "
            "Stage 1 checkpoint. The CSV has exactly six columns: "
            "raw_x, raw_y, raw_z, dir_x, dir_y, dir_z."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to photometric.pth. If omitted, --model_path and --iteration are used.",
    )
    parser.add_argument(
        "--model_path",
        "--model-path",
        dest="model_path",
        type=Path,
        default=None,
        help="Experiment model directory containing photometric/iteration_<N>/photometric.pth.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=35000,
        help="Checkpoint iteration used with --model_path (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <model_path>/light_directions.csv.",
    )
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    if args.model_path is None:
        raise SystemExit("Either --checkpoint or --model_path is required.")
    return (
        args.model_path.expanduser().resolve()
        / "photometric"
        / f"iteration_{args.iteration}"
        / "photometric.pth"
    )


def default_output_path(checkpoint: Path) -> Path:
    # checkpoint = <model_path>/photometric/iteration_<N>/photometric.pth
    return checkpoint.parents[2] / "light_directions.csv"


def load_payload(checkpoint: Path) -> dict:
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def load_raw_light_dir(checkpoint: Path) -> torch.Tensor:
    payload = load_payload(checkpoint)
    state = payload["state_dict"] if "state_dict" in payload else payload

    if "raw_light_dir" in state:
        raw = state["raw_light_dir"]
    elif "light_model._raw_light_dir_table" in state:
        raw = state["light_model._raw_light_dir_table"]
    elif "light_model._light_ctrl" in state:
        config = dict(payload.get("config", {}))
        timesteps = payload.get("timesteps")
        if timesteps is None:
            ctrl = state["light_model._light_ctrl"]
            timesteps = torch.arange(int(ctrl.shape[0]), dtype=torch.float32)
        light_model = DirectionalLightModel(
            timesteps,
            light_param=config.get("light_param", "bspline"),
            num_ctrl_points=config.get("num_ctrl_points", state["light_model._light_ctrl"].shape[0]),
            init_r_xy=config.get("init_r_xy", 0.8),
            init_z=config.get("init_z", 0.6),
            init_phase=config.get("init_phase", 0.0),
            init_direction_sign=config.get("init_direction_sign", 1),
            device="cpu",
        )
        light_model.load_state_dict(
            {
                key.replace("light_model.", ""): value
                for key, value in state.items()
                if key.startswith("light_model.")
            },
            strict=False,
        )
        raw = light_model.get_all_raw_light_dirs()
    else:
        raise SystemExit(f"No V1/V2 light direction tensor found in checkpoint: {checkpoint}")

    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    raw = raw.detach().cpu().float()
    if raw.ndim != 2 or raw.shape[-1] != 3:
        raise SystemExit(f"raw_light_dir must have shape [T, 3], got {tuple(raw.shape)}")
    return raw


def write_csv(output: Path, raw: torch.Tensor) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    light_dir = F.normalize(raw, dim=-1)
    rows = torch.cat([raw, light_dir], dim=-1).numpy()

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([f"{float(value):.10f}" for value in row])


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args)
    output = args.output.expanduser().resolve() if args.output else default_output_path(checkpoint)
    raw = load_raw_light_dir(checkpoint)
    write_csv(output, raw)
    print(f"Wrote {raw.shape[0]} light directions to {output}")


if __name__ == "__main__":
    main()
