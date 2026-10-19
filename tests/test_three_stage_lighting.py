"""方向极小更新、时间正则和环境幅度的机制回归。"""
import math
from types import SimpleNamespace

import torch

from scene.three_stage_lighting import SharedLighting, photometric_objective


def lights():
    asset = dict(gt_read=False, source_split='TRAIN', frames={f'{n:04d}': dict(light_dir_world=[0., 0., 1.])
                 for n in range(1, 151) if n % 10})
    return SharedLighting(asset, 1e-6, device='cpu')


def test_sub_acos_precision_direction_update_survives_projection():
    model = lights()
    with torch.no_grad():
        model.raw_direction[0, 0] += 1e-6
    model.project()
    assert model.directions()[0, 0] > 9e-7
    assert torch.allclose(model.directions().norm(dim=-1), torch.ones(135))


def test_hard_cone_and_time_anchor_penalize_drift():
    model = lights()
    with torch.no_grad():
        model.raw_direction[10, 0] = .5
    model.project(.5)
    direction = model.directions()[10].detach()
    degrees = math.degrees(math.atan2(float(direction[:2].norm()), float(direction[2])))
    assert abs(degrees-.5) < 1e-5
    anchor, smooth = model.direction_regularization()
    assert anchor > 0 and smooth > 0
    (anchor+10*smooth).backward()
    assert model.raw_direction.grad.norm() > 0


def test_environment_positive_bounded_and_shared():
    model = lights()
    with torch.no_grad():
        model.sh.fill_(100)
    normals = torch.nn.functional.normalize(torch.randn(1000, 3), dim=-1)
    environment = model.environment(normals)
    assert (environment >= .75*2*math.pi-1e-6).all()
    assert (environment <= 1.25*2*math.pi+1e-6).all()
    assert torch.equal(environment[:, 0], environment[:, 1])


def test_unsaturated_dark_foreground_and_censored_highlights_keep_loss():
    stats = dict(q_base_value=.7, q_bright_value=.95, transition_width=.05, q50=.5)
    target = torch.tensor([[[.2, 1.]]]).expand(3, 1, 2)
    prediction = torch.tensor([[[.5, 1.5]]]).expand(3, 1, 2).clone().requires_grad_()
    saturated = torch.tensor([[[False, True]]]).expand_as(target)
    package = dict(render_linear=prediction, base_linear=prediction, residual_linear=prediction*0)
    loss, terms = photometric_objective(package, target, torch.ones(1, 1, 2), saturated, stats,
                                       base_weight=0, residual_weight=0)
    loss.backward()
    assert prediction.grad[:, :, 0].abs().sum() > 0
    assert prediction.grad[:, :, 1].abs().sum() == 0
    assert terms['dark_linear_mae'] > 0


def test_test_direction_uses_only_train_neighbors():
    model = lights()
    with torch.no_grad():
        index = model.ids.index(11)
        model.raw_direction[index] = torch.tensor([1., 0., 1.])
    expected = torch.nn.functional.normalize((model.directions()[model.ids.index(9)]+model.directions()[index])[None], dim=-1)
    actual = model.for_camera(SimpleNamespace(image_name_train_light='image_0010'))
    assert torch.allclose(actual, expected)
