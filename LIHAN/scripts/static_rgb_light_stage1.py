"""本轮纯RGB静态Stage1；复用原生训练器，不加载历史模型。"""
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts.train_stage1 import Trainer
from scripts.train_static_dl import freeze_geometry_heads
from utils.general_utils import safe_state
from LIHAN.scripts.static_rgb_light_prepare import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    lp,op,pp = ModelParams(p),OptimizationParams(p),PipelineParams(p)
    p.add_argument('--save_iterations',nargs='+',type=int,default=[10000,20000,35000])
    p.add_argument('--quiet',action='store_true')
    a=p.parse_args(); a.pred_color=False
    dataset=lp.extract(a); source=Path(dataset.source_path); model=Path(dataset.model_path)
    assert not model.exists(), '禁止覆盖模型'
    manifest=json.loads((source/'rgb_only_manifest.json').read_text())
    assert manifest['gt_read'] is False and manifest['original_alpha_used'] is False
    for name,h in manifest['files'].items(): assert sha(source/name)==h,name
    # 原生Scene会读取transforms_test；只给它空测试集，训练不可见DEV/TEST。
    assert json.loads((source/'transforms_test.json').read_text())['frames']==[]
    def guard(event, values):
        if event!='open' or not isinstance(values[0],(str,bytes)): return
        path=Path(os.fsdecode(values[0])).resolve()
        if any(x in path.parts for x in ('albedo','normal_exr','gt_depth')) or path.name in {'lights.json','object_pose.json','bone_pose.json'}:
            raise PermissionError('纯RGB训练禁止GT文件：'+str(path))
        if path.suffix in {'.ply','.pth','.pt','.png','.exr'} and 'output' in path.parts and not (path.is_relative_to(source) or path.is_relative_to(model)):
            raise PermissionError('禁止外部实验资产：'+str(path))
    sys.addaudithook(guard)
    safe_state(a.quiet)
    trainer=Trainer(a,dataset,op.extract(a),pp.extract(a),[],a.save_iterations+[a.iterations])
    heads=freeze_geometry_heads(trainer.deform.deform)
    assert not trainer.deform.deform.pred_color
    counter={'updates':0}; original=trainer.gaussians.optimizer.step
    def step(*args,**kwargs):
        counter['updates']+=1
        return original(*args,**kwargs)
    trainer.gaussians.optimizer.step=step
    record=dict(status='RUNNING',source=str(source),model=str(model),args=vars(a),
                input_manifest_sha256=sha(source/'rgb_only_manifest.json'),gt_used=False,
                static_geometry_heads=list(heads),pred_color=False,train_frames=manifest['train_frames'])
    write(model/'rgb_only_stage1.json',record)
    trainer.train(a.iterations)
    assert all(float(getattr(trainer.deform.deform,n).weight.abs().max())==0 for n in heads)
    record.update(status='FINISHED',actual_updates=counter['updates'],points=len(trainer.gaussians.get_xyz))
    write(model/'rgb_only_stage1.json',record)


if __name__=='__main__': main()
