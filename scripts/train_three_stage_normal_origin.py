"""同父原生PBR法线起点保护和受限Lambertian残差对照。"""
from scene.three_stage_normal_origin import NormalOriginPBRModel
from scripts.train_three_stage import main


if __name__ == '__main__':
    main(model_class=NormalOriginPBRModel, actual_entrypoint='scripts/train_three_stage_normal_origin.py',
         extra_sources=('scene/three_stage_native_pbr.py', 'scene/three_stage_normal_origin.py',
                        'scripts/train_three_stage_normal_origin.py'))
