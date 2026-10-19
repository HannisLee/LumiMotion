"""Stage3恢复原生PBR混合深度起点；旧Expected模型按原语义重载。"""
import json
from pathlib import Path

from scene.three_stage_pipeline import ThreeStageModel


class NativePBRModel(ThreeStageModel):
    def __init__(self, parent, iteration, binding, stage, config):
        if stage != 3:
            raise ValueError('原生混合深度接口只用于逻辑Stage3。')
        config['pbr_surface_depth_mode'] = 'native_half'
        config['pbr_depth_ratio'] = 1.
        super().__init__(parent, iteration, binding, stage, config)
        # 原生render_ir内部min(.5, depth_ratio)，原生默认1对应各50%。
        self.pipe.depth_ratio = 1.

    @classmethod
    def load(cls, folder, step):
        obj = super().load(folder, step)
        if obj.stage != 3 or obj.config.get('pbr_surface_depth_mode') != 'native_half' or obj.config.get('pbr_depth_ratio') != 1.:
            raise ValueError('拒绝把旧Expected模型重解释为原生混合深度模型。')
        obj.pipe.depth_ratio = 1.
        return obj


def load_three_stage_model(folder, step):
    """从显式配置分派；不猜测、不修改旧模型、不使用GT回退。"""
    record = json.loads((Path(folder)/'three_stage_training.json').read_text())
    mode = record['config'].get('pbr_surface_depth_mode', 'legacy_expected')
    if mode == 'native_half':
        return NativePBRModel.load(folder, step)
    if mode == 'legacy_expected':
        return ThreeStageModel.load(folder, step)
    raise ValueError('未知三阶段PBR追踪起点格式：'+str(mode))
