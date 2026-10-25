"""本轮专用原生PBR混合照明接入；环境项和SUN项在线性辐亮度相加。"""
import copy
import importlib
import json
from pathlib import Path
import torch


def install():
    module = importlib.import_module('gaussian_renderer.render_ir')
    if getattr(module, '_el_eldl_mixed_installed', False):
        return
    original_equation = module.rendering_equation
    original_render = module.render_ir

    def equation(*args, **kwargs):
        if kwargs.get('directional_lights') is None or kwargs.get('envlight') is None:
            return original_equation(*args, **kwargs)
        environment = dict(kwargs)
        environment['directional_lights'] = None
        env = original_equation(*args, **environment)
        direct = dict(kwargs)
        direct['envlight'] = None
        # 环境分支已包含一次原生SH间接光，SUN分支只计算直接项，避免重复。
        positional = list(args)
        pipe = copy.copy(positional[6] if len(positional) > 6 else direct['pipe'])
        pipe.wo_indirect = True
        if len(positional) > 6:
            positional[6] = pipe
        else:
            direct['pipe'] = pipe
        sun = original_equation(*positional, **direct)
        return {k: (env[k] + sun[k] if k in {'diffuse', 'specular', 'light_direct', 'light', 'light_blocked'} else env[k]) for k in env}

    def render(*args, **kwargs):
        pipe = kwargs.get('pipe', args[2] if len(args) > 2 else None)
        path = getattr(pipe, 'pbr_mixed_lights_path', '')
        if path:
            from scene.pbr_directional import load_gt_directional_lights
            cam = kwargs.get('viewpoint_camera', args[0] if args else None)
            env = kwargs.get('env_light')
            if env is None:
                raise ValueError('ELDL必须保留训练得到的环境光')
            scale = env.mixed_log_intensity.exp() * float(getattr(pipe, "pbr_mixed_direct_multiplier", 1.))
            directions, energy = load_gt_directional_lights(path).for_camera(cam, 1.)
            kwargs['directional_lights'] = (directions, energy * scale)
        return original_render(*args, **kwargs)

    module.rendering_equation = equation
    module.render_ir = render
    module._el_eldl_mixed_installed = True


def attach(trainer):
    """在原生优化器和保存规则之外只添加一个正方向光强度。"""
    path = getattr(trainer.args, 'pbr_mixed_lights_path', '')
    if not path:
        return
    trainer.pipe.pbr_mixed_lights_path = path
    trainer.env_light.mixed_log_intensity = torch.nn.Parameter(torch.tensor(0., device='cuda'))
    trainer.env_light.optimizer.add_param_group({'params': [trainer.env_light.mixed_log_intensity], 'lr': .001})
    original_save = trainer.env_light.save_weights

    def save(model, iteration):
        original_save(model, iteration)
        record = {'iteration': iteration, 'effective_intensity': float(trainer.env_light.mixed_log_intensity.exp()),
                  'direction_source': path, 'environment_and_directional': True, 'gt_intensity_used': False}
        (Path(model) / f'envmap/iteration_{iteration}/mixed.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    trainer.env_light.save_weights = save


def install_evaluation():
    install()
    from scene.light import EnvLight
    original_load = EnvLight.load_weights
    def load(self, output, iteration):
        if (Path(output)/f"envmap/iteration_{iteration}/mixed.json").exists():
            self.mixed_log_intensity = torch.nn.Parameter(torch.tensor(0., device="cuda"), requires_grad=False)
        original_load(self, output, iteration)
    EnvLight.load_weights = load
    from scripts.render_stage2 import Stage2RenderRun
    original = Stage2RenderRun.__init__

    def init(self, output, iteration):
        original(self, output, iteration)
        if getattr(self.args, 'pbr_mixed_lights_path', ''):
            record = json.loads((Path(output)/f'envmap/iteration_{iteration}/mixed.json').read_text())
            import math
            assert abs(float(self.env.mixed_log_intensity.exp()) - record['effective_intensity']) < 1e-6
    Stage2RenderRun.__init__ = init
