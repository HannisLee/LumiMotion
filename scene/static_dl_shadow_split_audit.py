"""选定真实 checkpoint 后的原训练相机梯度审计，不更新任何参数。"""
import torch

from scene.static_dl_shadow_split import render_split_photometric, geometry_regularization
from scene.static_dl_photometric import robust_photometric_loss


def audit_checkpoint_routes(camera, pc, pipe, background, lights, settings):
    directions, _ = lights.for_camera(camera)
    directions = directions.detach().clone().requires_grad_(True)
    receiver = pc.get_xyz.detach().clone().requires_grad_(True)
    parameters = dict(xyz=pc._xyz, rotation=pc._rotation, scale=pc._scaling,
                      opacity=pc._opacity, material=pc._albedo_dc, light=lights.log_intensity,
                      direction=directions, receiver_origin=receiver)
    original = {k:v.requires_grad for k,v in parameters.items()}
    for value in parameters.values():
        value.requires_grad_(True)
    target = camera.original_image_train_light.cuda()
    alpha = camera.gt_alpha_mask.cuda()
    if pipe.white_background:
        target = alpha*target+(1-alpha)*background[:, None, None]

    def summary(gradients):
        return {k:dict(is_none=v is None, norm=0. if v is None else float(v.norm()),
                       max_abs=0. if v is None else float(v.abs().max()),
                       finite=v is None or bool(torch.isfinite(v).all())) for k,v in gradients.items()}

    def grads(loss):
        return {k:None if g is None else g.detach().cpu() for k,g in zip(parameters,
            torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True))}

    def render(route, detach_camera=True):
        return render_split_photometric(camera, pc, pipe, background, directions,
            lights.log_intensity.exp(), settings.shadow_bias, route=route,
            receiver_positions=receiver, detach_camera=detach_camera)

    def photo(package):
        return robust_photometric_loss(package['render'], target, alpha, settings.shadow_weight,
            settings.highlight_weight, settings.residual_threshold)[0]

    def difference(left, right, key):
        a, b = left[key], right[key]
        if a is None and b is None:
            return 0., 0.
        a = torch.zeros_like(b) if a is None else a
        b = torch.zeros_like(a) if b is None else b
        return float((a-b).abs().max()), float((a-b).norm()/a.norm().clamp_min(1e-20))

    try:
        all_grads, forward, initial_photo = {}, {}, {}
        visibility_stats = None
        for route in ('control', 'split', 'occluder'):
            package = render(route)
            if route in ('control', 'split'):
                forward[route] = {k:package[k].detach().cpu() for k in
                    ('render', 'render_linear', 'rend_alpha', 'rend_normal', 'surf_depth')}
            value = photo(package)
            initial_photo[route] = float(value.detach())
            visibility = package['light_visibility']
            visibility_stats = dict(mean=float(visibility.mean()),
                zero_fraction=float((visibility==0).float().mean()),
                below_point_one_fraction=float((visibility<.1).float().mean()),
                above_point_nine_fraction=float((visibility>.9).float().mean()))
            all_grads[route] = grads(value)
        summaries = {k:summary(v) for k,v in all_grads.items()}
        forward_diff = {k:float((forward['control'][k]-forward['split'][k]).abs().max())
                        for k in forward['control']}
        common = {k:difference(all_grads['control'], all_grads['split'], k)
                  for k in ('rotation', 'material', 'light')}
        xyz = all_grads['split']['xyz']
        expected_xyz = all_grads['occluder']['xyz']
        xyz_error = (float((xyz-expected_xyz).norm()/expected_xyz.norm().clamp_min(1e-20))
                     if xyz is not None and expected_xyz is not None else 1e30)
        package = render('split')
        map_grads = summary(grads(package['rend_alpha'].mean()+package['surf_depth'].mean()+package['rend_normal'].square().mean()))
        negative_arrays = grads(photo(render('control', detach_camera=False)))
        negative = summary(negative_arrays)
        negative_q_difference = difference(all_grads['control'], negative_arrays, 'rotation')[0]
        terms, _ = geometry_regularization(camera, pc, pipe, background, alpha)
        geometry = summary(grads(settings.alpha_weight*terms['alpha']+
            settings.normal_weight*terms['normal']+settings.dist_weight*terms['distortion']))
        def zero(group, names):
            return all(group[k]['finite'] and group[k]['norm']==0 for k in names)
        def nonzero(group, names):
            return all(group[k]['finite'] and group[k]['norm']>0 for k in names)
        checks = dict(forward_equal=all(v==0 for v in forward_diff.values()),
            receiver_allowed=nonzero(summaries['control'], ('rotation','material','light')),
            receiver_forbidden=zero(summaries['control'], ('xyz','scale','opacity','direction','receiver_origin')),
            shadow_allowed=nonzero(summaries['occluder'], ('xyz',)),
            shadow_forbidden=zero(summaries['occluder'], ('rotation','material','light','scale','opacity','direction','receiver_origin')),
            common_gradients_equal=all(a<=2e-6 and r<=1e-5 for a,r in common.values()),
            xyz_matches_shadow=xyz_error<1e-4,
            camera_maps_forbidden=zero(map_grads, parameters),
            negative_control_nonzero=nonzero(negative, ('xyz','rotation')) and negative_q_difference>1e-8,
            real_geometry_allowed=nonzero(geometry, ('xyz','rotation')),
            real_geometry_forbidden=zero(geometry, ('material','light','scale','opacity','direction','receiver_origin')))
        return dict(status='PASS' if all(checks.values()) else 'FAILED', camera=camera.image_name_train_light,
            checks=checks, forward_max_abs_difference=forward_diff, gradients=summaries,
            initial_photometric_loss=initial_photo, visibility=visibility_stats,
            common_gradient_difference_abs_relative=common, xyz_shadow_relative_error=xyz_error,
            camera_maps=map_grads, camera_negative_control=negative,
            camera_rotation_bypass_negative_difference=negative_q_difference, geometry=geometry,
            parameter_updates=0, note='只用原训练RGB/自身Alpha计算导数；未更新模型，不参与测试MAE。')
    finally:
        for name, value in parameters.items():
            value.requires_grad_(original[name])
