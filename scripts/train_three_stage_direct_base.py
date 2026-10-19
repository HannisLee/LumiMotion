"""主要DL显式进入PBR基础的同父小残差对照。"""
from scene.three_stage_direct_base import DirectBasePBRModel
from scripts.train_three_stage import main


if __name__ == '__main__':
    main(model_class=DirectBasePBRModel,
         actual_entrypoint='scripts/train_three_stage_direct_base.py',
         extra_sources=('scene/three_stage_native_pbr.py', 'scene/three_stage_normal_origin.py',
                        'scene/three_stage_direct_base.py', 'scripts/train_three_stage_direct_base.py'))
