"""隔离运行本轮Stage3，继承父模型架构并登记混合光参数。"""
import json
import runpy
import sys
from pathlib import Path
import arguments
import arguments.stage2 as parameters
from LIHAN.scripts.stage3_ppt_native_entry import architecture
from LIHAN.scripts.stage3_el_eldl_mixed import install

install()
parent = sys.argv[sys.argv.index('--stage1_model_path')+1]
defaults = architecture(parent)
original = arguments.ModelParams
class Model(original):
    def __init__(self, parser, *args, **kwargs):
        super().__init__(parser, *args, **kwargs)
        parser.set_defaults(**defaults)
arguments.ModelParams = Model
original_pipe = parameters.Stage2PipelineParams
class Pipeline(original_pipe):
    def __init__(self, parser):
        self.pbr_mixed_lights_path = ''
        self.pbr_static_parent = False
        super().__init__(parser)
parameters.Stage2PipelineParams = Pipeline
runpy.run_module('scripts.train_stage2', run_name='__main__')
