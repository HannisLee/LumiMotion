"""继承父模型架构后执行原生外观入口，不修改训练损失或更新规则。"""
import ast
import json
import runpy
import sys
from pathlib import Path
import arguments


def architecture(path):
    node = ast.parse((Path(path) / 'cfg_args').read_text().strip(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace':
        raise ValueError('父模型配置必须为 Namespace 字面量')
    values = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return {key: values[key] for key in ['sh_degree', 'deform_type', 'hyper_dim', 'pred_color', 'no_binary_separation']}


if __name__ == '__main__':
    parent = sys.argv[sys.argv.index('--stage1_model_path') + 1]
    defaults = architecture(parent)
    original = arguments.ModelParams

    class InheritedModelParams(original):
        def __init__(self, parser, *args, **kwargs):
            super().__init__(parser, *args, **kwargs)
            parser.set_defaults(**defaults)

    arguments.ModelParams = InheritedModelParams
    print('父模型架构：' + json.dumps(defaults), flush=True)
    runpy.run_module('scripts.train_stage2', run_name='__main__')
