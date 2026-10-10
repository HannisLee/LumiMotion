"""显式SUN训练按实际光线数计算像素预算；默认沿用历史MC预算。"""


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + '必须是正整数。')
    return value


def validate_actual_ray_budget(args):
    if not getattr(args, 'pbr_directional_actual_ray_budget', False):
        return
    if (getattr(args, 'render_mode', 'original_ir') != 'original_ir' or
            getattr(args, 'pbr_light_mode', 'envmap') != 'gt_directional'):
        raise ValueError('实际SUN光线预算仅适用于GT方向光original_ir。')
    if hasattr(args, 'trace_num_rays'):
        _positive_integer(args.trace_num_rays, 'trace_num_rays')


def rays_per_photo_pixel(pipe, directional_lights=None, relight=False):
    """SUN为每灯一条delta射线，SH间接光为diffuse_sample_num条。"""
    if (not getattr(pipe, 'pbr_directional_actual_ray_budget', False) or
            directional_lights is None):
        return pipe.diffuse_sample_num + pipe.light_sample_num
    validate_actual_ray_budget(pipe)
    directions, irradiances = directional_lights
    if (len(directions.shape) != 2 or directions.shape[1] != 3 or
            directions.shape != irradiances.shape):
        raise ValueError('SUN预算需要匹配的L×3方向与RGB辐照度。')
    count = _positive_integer(directions.shape[0], 'SUN数量')
    indirect_disabled = pipe.wo_indirect or (relight and pipe.wo_indirect_relight)
    if not indirect_disabled:
        count += _positive_integer(pipe.diffuse_sample_num, '间接光采样数')
    return count


def photo_pixel_budget(pipe, opt, directional_lights=None, relight=False):
    cost = rays_per_photo_pixel(pipe, directional_lights, relight)
    if (getattr(pipe, 'pbr_directional_actual_ray_budget', False) and
            directional_lights is not None):
        budget = _positive_integer(opt.trace_num_rays, 'trace_num_rays')
        if budget < cost:
            raise ValueError('trace_num_rays不足以追踪一个完整SUN着色像素。')
    return opt.trace_num_rays // cost, cost
