import unittest

from scripts.train_stage1 import photometric_material_learning_rates


class PhotometricMaterialScheduleTest(unittest.TestCase):
    def test_historical_defaults_train_both_material_groups(self):
        self.assertEqual(
            photometric_material_learning_rates(
                10001, 10001, -1, -1, 1e-3, 1e-3
            ),
            (1e-3, 1e-3),
        )

    def test_albedo_then_normal_are_mutually_exclusive(self):
        for iteration in (10001, 10200, 10500):
            self.assertEqual(
                photometric_material_learning_rates(
                    iteration, 10001, 10501, 10501, 1e-3, 1e-4
                ),
                (1e-3, 0.0),
            )
        for iteration in (10501, 10510, 11000):
            self.assertEqual(
                photometric_material_learning_rates(
                    iteration, 10001, 10501, 10501, 1e-3, 1e-4
                ),
                (0.0, 1e-4),
            )

    def test_material_groups_are_frozen_before_photometric_switch(self):
        self.assertEqual(
            photometric_material_learning_rates(
                10000, 10001, 10501, 10501, 1e-3, 1e-4
            ),
            (0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
