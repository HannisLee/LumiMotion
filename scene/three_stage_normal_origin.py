"""仅为Stage3的原生PBR半球追踪增加显式法线起点保护。"""
from scene.three_stage_native_pbr import NativePBRModel


class NormalOriginPBRModel(NativePBRModel):
    def __init__(self, parent, iteration, binding, stage, config):
        config['pbr_origin_mode'] = 'normal_existing_tmin'
        config['pbr_origin_normal_offset'] = config['shadow_bias']
        super().__init__(parent, iteration, binding, stage, config)
        self.pipe.pbr_origin_normal_offset = config['pbr_origin_normal_offset']

    @classmethod
    def load(cls, folder, step):
        obj = super().load(folder, step)
        if obj.config.get('pbr_origin_mode') != 'normal_existing_tmin' or obj.config.get('pbr_origin_normal_offset') != obj.config['shadow_bias']:
            raise ValueError('法线起点模式或既有t_min绑定不匹配，禁止重解释旧模型。')
        obj.pipe.pbr_origin_normal_offset = obj.config['pbr_origin_normal_offset']
        return obj
