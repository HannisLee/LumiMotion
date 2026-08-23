import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils.gt_normal_utils import frame_id_from_name, source_frame_by_image_name
from utils.normal_eval_utils import (
    alpha_normalized_normal_map,
    masked_normal_cosine_loss,
)


class GTNormalOracleTest(unittest.TestCase):
    def test_frame_mapping_preserves_exported_source_frame(self):
        train = {
            "frames": [
                {"file_path": "frame_0001.png", "source_frame": 11},
            ]
        }
        test = {
            "frames": [
                {"file_path": "frame_0008.png", "source_frame": 18},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "transforms_train.json").write_text(json.dumps(train))
            (root / "transforms_test.json").write_text(json.dumps(test))
            self.assertEqual(
                source_frame_by_image_name(root),
                {"frame_0001": 11, "frame_0008": 18},
            )
        self.assertEqual(frame_id_from_name("normal_0042.exr"), 42)
        self.assertEqual(frame_id_from_name("frame_0007"), 7)

    def test_alpha_normalized_normal_map_decodes_alpha_weighted_color(self):
        normal = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        alpha = torch.tensor([[[0.4]]])
        encoded_sum = (normal * 0.5 + 0.5) * alpha
        decoded = alpha_normalized_normal_map(encoded_sum, alpha)
        torch.testing.assert_close(decoded, normal)

    def test_masked_normal_cosine_loss_has_expected_value_and_gradient(self):
        alpha = torch.ones(1, 1, 1)
        encoded_sum = torch.tensor([0.5, 0.5, 1.0]).view(3, 1, 1)
        encoded_sum.requires_grad_(True)
        predicted = alpha_normalized_normal_map(encoded_sum, alpha)
        gt = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
        valid = torch.ones(1, 1, dtype=torch.bool)
        loss = masked_normal_cosine_loss(predicted, gt, valid)
        loss.backward()
        torch.testing.assert_close(loss.detach(), torch.tensor(1.0))
        self.assertIsNotNone(encoded_sum.grad)
        self.assertGreater(float(encoded_sum.grad.abs().sum()), 0.0)

    def test_masked_normal_cosine_loss_ignores_invalid_pixels(self):
        predicted = torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]]
        )
        target = torch.tensor(
            [[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
        )
        valid = torch.tensor([[True, False]])
        torch.testing.assert_close(
            masked_normal_cosine_loss(predicted, target, valid), torch.tensor(0.0)
        )


if __name__ == "__main__":
    unittest.main()
