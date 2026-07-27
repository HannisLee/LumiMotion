import unittest
from types import SimpleNamespace

from scripts.train_stage1 import Trainer


class _Optimizer:
    def __init__(self, names):
        self.param_groups = [{"name": name, "lr": 1.0} for name in names]


class PhotometricTrainingScheduleTest(unittest.TestCase):
    def make_trainer(self, iteration):
        trainer = Trainer.__new__(Trainer)
        trainer.iteration = iteration
        trainer.requested_render_mode = "photometric_lambertian"
        trainer.photometric_staged_training = True
        trainer.photometric_deform_unfreeze_iter = 22_000
        trainer.photometric_rotation_unfreeze_iter = 30_000
        trainer.photometric_deform_lr_scale_after_unfreeze = 0.1
        trainer.photometric_rotation_lr_scale_after_unfreeze = 0.1
        trainer._last_photometric_training_stage = None
        trainer.opt = SimpleNamespace(
            photometric_start_iter=20_000,
            rotation_lr=0.001,
        )
        trainer.gaussians = SimpleNamespace(
            optimizer=_Optimizer(
                ["xyz", "rotation", "feature", "photometric_albedo"]
            )
        )
        trainer.deform = SimpleNamespace(
            optimizer=_Optimizer(["mlp", "mlp_color"]),
            deform_scheduler_args=lambda _: 0.002,
        )
        return trainer

    def group_lrs(self, optimizer):
        return {group["name"]: group["lr"] for group in optimizer.param_groups}

    def test_albedo_only_stage(self):
        trainer = self.make_trainer(20_000)
        trainer.apply_photometric_training_schedule()
        self.assertEqual(
            self.group_lrs(trainer.gaussians.optimizer),
            {
                "xyz": 0.0,
                "rotation": 0.0,
                "feature": 0.0,
                "photometric_albedo": 1.0,
            },
        )
        self.assertTrue(
            all(group["lr"] == 0.0 for group in trainer.deform.optimizer.param_groups)
        )

    def test_deformation_unfreeze_stage(self):
        trainer = self.make_trainer(22_000)
        trainer.apply_photometric_training_schedule()
        self.assertEqual(
            self.group_lrs(trainer.gaussians.optimizer)["rotation"],
            0.0,
        )
        self.assertTrue(
            all(
                abs(group["lr"] - 0.0002) < 1e-12
                for group in trainer.deform.optimizer.param_groups
            )
        )

    def test_rotation_unfreeze_stage(self):
        trainer = self.make_trainer(30_000)
        trainer.apply_photometric_training_schedule()
        lrs = self.group_lrs(trainer.gaussians.optimizer)
        self.assertAlmostEqual(lrs["rotation"], 0.0001)
        self.assertEqual(lrs["xyz"], 0.0)
        self.assertEqual(lrs["feature"], 0.0)

    def test_pbr_schedule_keeps_geometry_and_deformation_fixed(self):
        trainer = Trainer.__new__(Trainer)
        trainer.iteration = 35_001
        trainer._last_photometric_training_stage = None
        trainer.opt = SimpleNamespace(
            photometric_albedo_lr=0.001,
            photometric_pbr_roughness_lr=0.0005,
        )
        trainer.gaussians = SimpleNamespace(
            optimizer=_Optimizer(
                [
                    "xyz",
                    "rotation",
                    "scaling",
                    "opacity",
                    "roughness",
                    "photometric_albedo",
                ]
            )
        )
        trainer.deform = SimpleNamespace(
            optimizer=_Optimizer(["mlp", "mlp_color"])
        )

        trainer.apply_pbr_training_schedule()

        self.assertEqual(
            self.group_lrs(trainer.gaussians.optimizer),
            {
                "xyz": 0.0,
                "rotation": 0.0,
                "scaling": 0.0,
                "opacity": 0.0,
                "roughness": 0.0005,
                "photometric_albedo": 0.001,
            },
        )
        self.assertTrue(
            all(group["lr"] == 0.0 for group in trainer.deform.optimizer.param_groups)
        )


if __name__ == "__main__":
    unittest.main()
