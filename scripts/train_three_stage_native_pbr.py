"""逻辑Stage3原生PBR点深度接口：原生混合深度＋受限Lambertian残差。"""
from scene.three_stage_native_pbr import NativePBRModel
from scripts.train_three_stage import main


if __name__ == '__main__':
    main(model_class=NativePBRModel, actual_entrypoint='scripts/train_three_stage_native_pbr.py',
         extra_sources=('scene/three_stage_native_pbr.py', 'scripts/train_three_stage_native_pbr.py'))
