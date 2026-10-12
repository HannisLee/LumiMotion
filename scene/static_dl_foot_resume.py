"""仅恢复足部续训已保存状态；不改变损失、拓扑或渲染。"""
import random

import numpy as np
import torch


def restore_training_state(path, pc, light_optimizer, camera_rng, cameras, iteration):
    """载入可信本地训练产物，逐项核验Adam与随机状态；BVH由原interval=1重建。"""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    checks = {}

    def require(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise AssertionError('状态恢复核验失败：' + name)

    require('iteration', saved['iteration'] == iteration == 16000)
    require('updates', saved['optimizer_updates'] == 2000)
    require('source_reset_policy', saved['reset_policy'] == 'skip_15000')
    require('real_clock', not saved['clock_shift_diagnostic'])
    for label, optimizer in [('gaussian', pc.optimizer), ('light', light_optimizer)]:
        state = saved[label]
        require(label + '_group_names', [g['name'] for g in state['param_groups']] ==
                [g['name'] for g in optimizer.param_groups])
        require(label + '_group_count', len(state['param_groups']) == len(optimizer.param_groups))
        for group, target in zip(state['param_groups'], optimizer.param_groups):
            name = group['name']
            require(label + '_' + name + '_param_count', len(group['params']) == len(target['params']))
            require(label + '_' + name + '_lr', group['lr'] == target['lr'])
            for index, parameter in zip(group['params'], target['params']):
                if index not in state['state']:
                    # 未参与前向的属性没有Adam状态，保持这一事实。
                    continue
                item = state['state'][index]
                for key in ('exp_avg', 'exp_avg_sq'):
                    require(f'{label}_{name}_{key}_shape', item[key].shape == parameter.shape)
                    require(f'{label}_{name}_{key}_finite', bool(torch.isfinite(item[key]).all()))
        optimizer.load_state_dict(state)
        actual = optimizer.state_dict()
        require(label + '_all_groups_equal', actual['param_groups'] == state['param_groups'])
        require(label + '_state_keys_equal', actual['state'].keys() == state['state'].keys())
        for index, item in state['state'].items():
            require(f'{label}_{index}_keys_equal', actual['state'][index].keys() == item.keys())
            for key, value in item.items():
                found = actual['state'][index][key]
                require(f'{label}_{index}_{key}_equal', torch.equal(found.cpu(), value) if torch.is_tensor(value) else found == value)
    for name, shape in [('xyz_gradient_accum', (len(pc.get_xyz), 1)),
                        ('denom', (len(pc.get_xyz), 1)),
                        ('max_radii2D', (len(pc.get_xyz),))]:
        value = saved[name]
        require(name + '_shape', tuple(value.shape) == shape)
        require(name + '_finite', bool(torch.isfinite(value).all()))
        setattr(pc, name, value.to(pc.get_xyz.device).clone())
        require(name + '_equal', torch.equal(getattr(pc, name).cpu(), value))
    stack = list(saved['remaining_camera_indices'])
    require('camera_stack_unique_valid', len(set(stack)) == len(stack) and
            all(isinstance(i, int) and 0 <= i < len(cameras) for i in stack))
    random.setstate(saved['python_rng'])
    camera_rng.setstate(saved['camera_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    require('python_rng_equal', random.getstate() == saved['python_rng'])
    require('camera_rng_equal', camera_rng.getstate() == saved['camera_rng'])
    actual_numpy = np.random.get_state()
    require('numpy_rng_equal', actual_numpy[0] == saved['numpy_rng'][0] and
            np.array_equal(actual_numpy[1], saved['numpy_rng'][1]) and
            actual_numpy[2:] == saved['numpy_rng'][2:])
    require('torch_rng_equal', torch.equal(torch.get_rng_state(), saved['torch_rng']))
    require('cuda_rng_equal', torch.equal(torch.cuda.get_rng_state().cpu(), saved['cuda_rng']))
    preview = random.Random()
    preview.setstate(saved['camera_rng'])
    preview_stack = stack.copy() or list(range(len(cameras)))
    first_index = preview_stack.pop(preview.randrange(len(preview_stack)))
    return stack, dict(status='PASS', checks=checks, source=str(path),
                       restored_iteration=iteration, remaining_camera_indices=stack.copy(),
                       expected_first_camera=cameras[first_index].image_name_train_light,
                       expected_first_camera_index=first_index,
                       restored_xyz_lr=pc.optimizer.param_groups[0]['lr'],
                       bvh='冻结interval=1，每一步均按当前模型重建，不依赖跨步缓存',
                       limits='PLY与光强exp/JSON/log重载不声称逐位历史重放；不补做16000边界拓扑/reset',
                       optimizer_updates=0)
