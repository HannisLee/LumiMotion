"""Write reproducible metrics for a Stage 1 photometric continuation."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections.abc import Iterable

import numpy as np
import torch
from plyfile import PlyData


CHECKPOINT_ITERATIONS = (35001, 36000, 40000, 45000, 50000, 55000)
METRIC_TAGS = {
    # training_report writes every scheduled evaluation under the train tags;
    # except for L1, the values are the test-camera aggregates returned by the
    # existing evaluator.
    "l1": "train/reconstruct - l1_loss",
    "psnr": "train/reconstruct - psnr",
    "ssim": "train/reconstruct - ssim",
    "lpips": "train/reconstruct - lpips",
    "ms_ssim": "train/reconstruct - ms-ssim",
    "alex_lpips": "train/reconstruct - alex-lpips",
    "points": "total_points",
}
PLY_GROUPS = {
    "xyz": ("x", "y", "z"),
    "rotation": ("rot_",),
    "scale": ("scale_",),
    "opacity": ("opacity",),
    "roughness": ("roughness",),
    "feature": ("fea_",),
    "albedo_sh": (
        "albedo_dc_",
        "albedo_dc_stage1_",
        "albedo_rest_",
        "f_dc_",
    ),
}


def _write_json(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _property_names(ply):
    return [prop.name for prop in ply.elements[0].properties]


def _select_properties(names: Iterable[str], selectors: tuple[str, ...]):
    selected = []
    for name in names:
        if any(name == selector or name.startswith(selector) for selector in selectors):
            selected.append(name)
    return selected


def _ply_matrix(ply, names):
    vertex = ply.elements[0].data
    return np.stack([np.asarray(vertex[name], dtype=np.float64) for name in names], axis=1)


def _max_state_dict_drift(reference_path, result_path):
    reference = torch.load(reference_path, map_location="cpu")
    result = torch.load(result_path, map_location="cpu")
    if reference.keys() != result.keys():
        raise ValueError("Deformation checkpoint keys differ.")
    maximum = 0.0
    for key in reference:
        maximum = max(
            maximum,
            float((reference[key].double() - result[key].double()).abs().max().item()),
        )
    return maximum


def parameter_drift(baseline_model, model_path, iteration):
    reference_path = os.path.join(
        baseline_model,
        "point_cloud",
        "iteration_35000",
        "point_cloud.ply",
    )
    result_path = os.path.join(
        model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )
    reference = PlyData.read(reference_path)
    result = PlyData.read(result_path)
    if reference.elements[0].count != result.elements[0].count:
        raise ValueError(
            "Point count changed: "
            f"{reference.elements[0].count} -> {result.elements[0].count}."
        )
    reference_names = _property_names(reference)
    result_names = _property_names(result)

    max_abs_drift = {}
    for group, selectors in PLY_GROUPS.items():
        names = _select_properties(reference_names, selectors)
        if not names or not all(name in result_names for name in names):
            raise ValueError(f"Missing PLY properties for {group}: {names}.")
        max_abs_drift[group] = float(
            np.max(np.abs(_ply_matrix(reference, names) - _ply_matrix(result, names)))
        )

    rotation_names = _select_properties(reference_names, ("rot_",))
    rotation_a = _ply_matrix(reference, rotation_names)
    rotation_b = _ply_matrix(result, rotation_names)
    rotation_a /= np.linalg.norm(rotation_a, axis=1, keepdims=True).clip(1e-12)
    rotation_b /= np.linalg.norm(rotation_b, axis=1, keepdims=True).clip(1e-12)
    rotation_dot = np.abs(np.sum(rotation_a * rotation_b, axis=1)).clip(-1.0, 1.0)
    rotation_angle = np.degrees(2.0 * np.arccos(rotation_dot))

    raw_names = _select_properties(result_names, ("photometric_albedo_raw_",))
    if len(raw_names) != 3:
        raise ValueError("Final PLY is missing photometric albedo.")
    raw_albedo = _ply_matrix(result, raw_names)
    albedo = 1.0 / (1.0 + np.exp(-raw_albedo))

    reference_deform = os.path.join(
        baseline_model,
        "deform",
        "iteration_35000",
        "deform.pth",
    )
    result_deform = os.path.join(
        model_path,
        "deform",
        f"iteration_{iteration}",
        "deform.pth",
    )
    return {
        "point_count": int(result.elements[0].count),
        "max_abs_drift": max_abs_drift,
        "rotation_angle_deg": {
            "mean": float(rotation_angle.mean()),
            "median": float(np.median(rotation_angle)),
            "p95": float(np.percentile(rotation_angle, 95)),
            "max": float(rotation_angle.max()),
        },
        "deform_max_abs_drift": _max_state_dict_drift(
            reference_deform,
            result_deform,
        ),
        "photometric_albedo": {
            "mean": float(albedo.mean()),
            "min": float(albedo.min()),
            "max": float(albedo.max()),
        },
    }


def training_metrics(model_path):
    # TensorBoard 2.16 still references NumPy aliases removed in NumPy 2.
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values = {name: {} for name in METRIC_TAGS}
    event_paths = sorted(glob.glob(os.path.join(model_path, "events.out.tfevents.*")))
    if not event_paths:
        raise FileNotFoundError(f"No TensorBoard event file in {model_path}.")
    for event_path in event_paths:
        accumulator = EventAccumulator(event_path)
        accumulator.Reload()
        available = set(accumulator.Tags()["scalars"])
        for name, tag in METRIC_TAGS.items():
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                if event.step in CHECKPOINT_ITERATIONS:
                    values[name][str(event.step)] = float(event.value)
    return values


def _load_photometric(model_path, iteration):
    path = os.path.join(
        model_path,
        "photometric",
        f"iteration_{iteration}",
        "photometric.pth",
    )
    return torch.load(path, map_location="cpu")


def _directions(checkpoint):
    state = checkpoint["state_dict"]
    raw = state["light_model._raw_light_dir_table"].double()
    return torch.nn.functional.normalize(raw, dim=-1).numpy()


def _gt_reference(checkpoint, gt_lights_path):
    initialization = checkpoint.get("initialization", {})
    center = initialization.get("object_center")
    if center is None:
        center = initialization.get("reference_center")
    if center is None:
        raise ValueError("Photometric checkpoint does not record its object center.")
    with open(gt_lights_path, "r", encoding="utf-8") as handle:
        lights = json.load(handle)
    ordered = [lights[key] for key in sorted(lights, key=lambda value: int(value))]
    positions = np.asarray(
        [entry["light_pos_world"] for entry in ordered],
        dtype=np.float64,
    )
    rays = np.asarray(center, dtype=np.float64)[None] - positions
    return rays / np.linalg.norm(rays, axis=1, keepdims=True).clip(1e-12)


def _angle_stats(directions, reference):
    dot = np.sum(directions * reference, axis=1).clip(-1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    return {
        "mean": float(angles.mean()),
        "median": float(np.median(angles)),
        "max": float(angles.max()),
    }


def light_metrics(model_path, gt_lights_path, iteration):
    initial = _load_photometric(model_path, 35001)
    final = _load_photometric(model_path, iteration)
    initial_directions = _directions(initial)
    final_directions = _directions(final)
    reference = _gt_reference(initial, gt_lights_path)
    smoothness = (
        np.square(final_directions[1:] - final_directions[:-1]).mean()
        if final_directions.shape[0] > 1
        else 0.0
    )
    return {
        "light_mode": final.get("config", {}).get(
            "light_mode",
            "learned_directional",
        ),
        "reference_gt": "normalize(scene_center - light_pos_world)",
        "initial_angle_deg": _angle_stats(initial_directions, reference),
        "final_angle_deg": _angle_stats(final_directions, reference),
        "first_order_smoothness": float(smoothness),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--experiment_path", required=True)
    parser.add_argument("--gt_lights_path", required=True)
    parser.add_argument("--iteration", type=int, default=55000)
    args = parser.parse_args()

    os.makedirs(args.experiment_path, exist_ok=True)
    _write_json(
        training_metrics(args.model_path),
        os.path.join(args.experiment_path, "training_metrics.json"),
    )
    _write_json(
        parameter_drift(args.baseline_model, args.model_path, args.iteration),
        os.path.join(args.experiment_path, "parameter_drift.json"),
    )
    _write_json(
        light_metrics(args.model_path, args.gt_lights_path, args.iteration),
        os.path.join(args.experiment_path, "light_direction_metrics.json"),
    )


if __name__ == "__main__":
    main()
