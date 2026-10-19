"""动态V3独立GT准备入口：逐时刻求值网格与CPU射线，不进入训练。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from LIHAN.scripts.eldl_v3_prepare_eval_gt import main


if __name__ == '__main__':
    main(source=ROOT/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3',
         allowed=ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控/eval_only',
         script=__file__, blend_relative='scene_snapshot.blend', allow_dynamic=True)
