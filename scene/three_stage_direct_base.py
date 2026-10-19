"""原生环境PBR与一次主要DL组成基础，另保留受限的小幅DL残差。"""
from types import SimpleNamespace

import torch

from scene.three_stage_normal_origin import NormalOriginPBRModel
from scene.three_stage_lighting import surface_components
from scene.stage2_photometric import linear_to_srgb


class DirectBasePBRModel(NormalOriginPBRModel):
    def __init__(self, parent, iteration, binding, stage, config):
        config['pbr_base_mode'] = 'environment_plus_inherited_direct'
        super().__init__(parent, iteration, binding, stage, config)
        inherited = self.parent_record['config']['lambda_mix']
        if config['lambda_mix'] != inherited or not 0 < inherited < 1:
            raise ValueError('主要DL只能继承父Stage2的共享混合权重。')

    @classmethod
    def load(cls, folder, step):
        obj = super().load(folder, step)
        if obj.config.get('pbr_base_mode') != 'environment_plus_inherited_direct':
            raise ValueError('拒绝将旧环境PBR模型重解释为包含主要DL的新基础。')
        if not 0 < obj.config['lambda_mix'] < 1:
            raise ValueError('无效的继承混合权重。')
        return obj

    def render(self, camera, step=0, components=False, training=False):
        from gaussian_renderer import render
        from gaussian_renderer.render_ir import render_ir

        pc = self.effective(camera)
        _, direct, vis = surface_components(camera, pc, self.lights, self.visibility, step)
        package = render_ir(camera, pc, self.pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
            d_color=pc.values['d_color'], env_light=self.env, training=training,
            opt=SimpleNamespace(train_ray=training,
                trace_num_rays=self.config['pbr_pixels']*self.config['pbr_samples'],
                pbr_sampling_alpha_threshold=0., pbr_sample_original_foreground=True))
        dl = render(camera, pc, self.pipe, pc.get_xyz.new_zeros(3), 0., 0., 0.,
                    override_color=direct)
        if not torch.equal(package['rend_alpha'], dl['rend_alpha']):
            raise AssertionError('原生环境PBR与主要DL的Alpha不一致。')
        direct_linear = dl['render']
        if training:
            direct_linear = direct_linear*package['mask'][None]
        mix = self.config['lambda_mix']
        environment = (1-mix)*package['render_linear']
        main_direct = mix*direct_linear
        base = environment+main_direct
        amplitude = .25*self.residual_raw.sigmoid()
        residual = direct_linear*amplitude*self.config['lambda_residual']
        total = base+residual
        package.update(environment_linear=environment, direct_base_linear=main_direct,
            direct_unit_linear=direct_linear, base_linear=base, residual_linear=residual,
            render_linear=total, render=linear_to_srgb(total.clamp_min(0)),
            visibility_mean=vis.mean().detach(), residual_amplitude=amplitude,
            base_component_add_max_error=(base-environment-main_direct).abs().max(),
            component_add_max_error=(total-base-residual).abs().max())
        if training:
            package['loss_support'] = package['mask'][None] | (package['rend_alpha'] <= 0)
        return package
