import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.visualize_stage1_light_trajectory import (
    DIRECTION_CONVENTION,
    UNIT_SPHERE_VIEWS,
    WORLD_SPACE_VIEWS,
    compare_light_trajectories,
    load_light_checkpoint,
    parse_representative_frames,
)


def write_checkpoint(path: Path, directions, *, gt_positions=None, reference_center=None, timesteps=None):
    directions = torch.tensor(directions, dtype=torch.float32)
    if timesteps is None:
        timesteps = torch.arange(directions.shape[0], dtype=torch.float32)
    state_dict = {
        "light_model._raw_light_dir_table": directions,
        "light_model.timesteps": timesteps,
    }
    if gt_positions is not None:
        state_dict["gt_light_positions"] = torch.tensor(gt_positions, dtype=torch.float32)
    torch.save(
        {
            "state_dict": state_dict,
            "timesteps": timesteps,
            "photometric_version": "test",
            "config": {"direction_convention": DIRECTION_CONVENTION},
            "initialization": {
                "direction_convention": DIRECTION_CONVENTION,
                **({"reference_center": reference_center} if reference_center is not None else {}),
            },
        },
        path,
    )


class VisualizeStage1LightTrajectoryTest(unittest.TestCase):
    def test_six_numbered_views_have_stable_unique_indices_and_filenames(self):
        all_views = WORLD_SPACE_VIEWS + UNIT_SPHERE_VIEWS
        self.assertEqual(tuple(view[0] for view in all_views), ("01", "02", "03", "04", "05", "06"))
        self.assertEqual(len({(view[0], view[4]) for view in all_views}), 6)

    def test_load_normalizes_directions_and_keeps_gt_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.pth"
            write_checkpoint(
                path,
                [[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
                gt_positions=[[0.0, 0.0, -2.0], [0.0, -3.0, 0.0]],
                reference_center=[0.0, 0.0, 0.0],
            )
            checkpoint = load_light_checkpoint(path)
        np.testing.assert_allclose(checkpoint.directions, [[0, 0, 1], [0, 1, 0]])
        np.testing.assert_allclose(checkpoint.gt_positions, [[0, 0, -2], [0, -3, 0]])

    def test_comparison_builds_virtual_sources_and_angles(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            learned_path = directory / "learned.pth"
            gt_path = directory / "gt.pth"
            write_checkpoint(learned_path, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
            write_checkpoint(
                gt_path,
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                gt_positions=[[0.0, 0.0, -4.0], [0.0, -4.0, 0.0]],
                reference_center=[0.0, 0.0, 0.0],
            )
            comparison = compare_light_trajectories(load_light_checkpoint(learned_path), load_light_checkpoint(gt_path))
        self.assertAlmostEqual(comparison["virtual_radius"], 4.0)
        np.testing.assert_allclose(comparison["learned_virtual_positions"], [[0, 0, -4], [-4, 0, 0]])
        np.testing.assert_allclose(comparison["angular_errors_deg"], [0, 90], atol=1e-5)

    def test_representative_frames_are_validated(self):
        self.assertEqual(parse_representative_frames("0,2,4", 5), (0, 2, 4))
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_representative_frames("0,0", 5)
        with self.assertRaisesRegex(ValueError, "within"):
            parse_representative_frames("5", 5)


if __name__ == "__main__":
    unittest.main()
