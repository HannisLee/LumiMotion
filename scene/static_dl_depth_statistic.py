"""只改变训练法线一致性使用的深度统计量，不改变物理RGB前向。"""
from copy import copy

from scene.static_dl_shadow_split import geometry_regularization as original_geometry_regularization


def geometry_regularization(camera, pc, pipe, background, original_rgba_alpha, statistic):
    if statistic not in ('median', 'expected'):
        raise ValueError('未知法线一致性深度统计量：'+str(statistic))
    geometry_pipe = copy(pipe)
    geometry_pipe.depth_ratio = 1. if statistic == 'median' else 0.
    # 所有mask、归一化、双向梯度、Alpha和distortion定义沿用已冻结函数。
    return original_geometry_regularization(camera, pc, geometry_pipe, background, original_rgba_alpha)
