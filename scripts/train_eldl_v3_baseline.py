"""V3 原生/静态SH基线：TRAIN白名单输入，无外部测试与历史模型。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from scripts.train_stage1 import Trainer
from scripts.train_static_dl import freeze_geometry_heads
from utils.general_utils import safe_state


def validate_source(source,model):
    manifest = json.loads((source/'no_gt_manifest.json').read_text())
    assert manifest['gt_read'] is False and manifest['init_weights_loaded'] is False
    assert manifest['dataset'] == 'LH-static-jumpingjacks-ELDL-V3'
    assert set(p.name for p in source.iterdir()) == {'images','points3d.ply','transforms_train.json','no_gt_manifest.json'}
    for relative,expected in manifest['files'].items():
        path = source/relative
        assert path.resolve().is_relative_to(source)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected,relative
    actual_images = {str(p.relative_to(source)) for p in (source/'images').iterdir()}
    assert actual_images == {p for p in manifest['files'] if p.startswith('images/')}
    records = json.loads((source/'transforms_train.json').read_text())['frames']
    assert [f['source_frame'] for f in records] == manifest['train_frames']
    assert all(i%10 for i in manifest['train_frames'])
    assert not model.exists(), '已有模型目录禁止覆盖'
    def guard(event,values):
        if event != 'open' or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to(source) or path.is_relative_to(model):
            return
        if path.suffix.lower() in {'.ply','.pth','.pt','.exr','.blend','.png','.jpg','.jpeg','.npz'}:
            raise PermissionError('无GT训练禁止读取外部模型/图像/GT：'+str(path))
        if path.name in {'lights.json','object_pose.json','bone_pose.json'}:
            raise PermissionError('无GT训练禁止读取受限真值：'+str(path))
    sys.addaudithook(guard)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lp,op,pp = ModelParams(parser),OptimizationParams(parser),PipelineParams(parser)
    parser.add_argument('--baseline',choices=['native','static_sh'],required=True)
    parser.add_argument('--save_iterations',type=int,nargs='+',default=[8000,12000,16000])
    parser.add_argument('--deform-type',dest='deform_type',default='mlp')
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    source,model = Path(args.source_path).resolve(),Path(args.model_path).resolve()
    if not args.eval or args.deform_type != 'mlp':parser.error('必须eval隔离划分，且mlp格式')
    if not model.is_relative_to(Path(__file__).resolve().parents[1]/'output'):
        # 快照位于output内，输出可在仓库output的任意实验目录。
        if 'output' not in model.parts:parser.error('模型必须在output内')
    if args.smoke:
        if 'smoke_test' not in model.parts or args.iterations>50:parser.error('冒烟只能smoke_test，最多50步')
    elif args.iterations<16000:parser.error('正式基线至少16000步')
    manifest = validate_source(source,model)
    safe_state(args.quiet)
    import diff_surfel_rasterization as backend
    assert backend.provenance['method']=='repository_lumimotion_cuda_rasterizer'
    trainer = Trainer(args=args,dataset=lp.extract(args),opt=op.extract(args),pipe=pp.extract(args),
                      testing_iterations=[],saving_iterations=sorted(set(args.save_iterations+[args.iterations])))
    if args.baseline == 'static_sh':freeze_geometry_heads(trainer.deform.deform)
    assert not trainer.scene.getTestCameras()
    transforms=json.loads((source/'transforms_train.json').read_text())['frames']
    positions={Path(f['file_path']).stem:torch.tensor(f['transform_matrix'],dtype=torch.float32)[:3,3] for f in transforms}
    centers=[float((camera.camera_center.cpu()-positions[camera.image_name_train_light]).abs().max())
             for camera in trainer.scene.getTrainCameras()]
    assert max(centers)<1e-4,'相机中心与原始c2w不符'
    first=trainer.scene.getTrainCameras()[0]
    assert (first.gt_alpha_mask==0).any() and (first.gt_alpha_mask==1).any()
    assert hasattr(first,'observed_saturation_mask'),'缺少缩小前满码通道记录'
    record = dict(status='RUNNING',args=vars(args),gt_supervision=False,init_weights_loaded=False,
                  train_frames=manifest['train_frames'],internal_held_frames=manifest['internal_held_frames'],
                  external_test_read=False,input_manifest_sha256=hashlib.sha256((source/'no_gt_manifest.json').read_bytes()).hexdigest(),
                  rasterizer_provenance=backend.provenance,baseline=args.baseline,
                  source_code_root=str(Path(__file__).resolve().parents[1]),
                  input_audit=dict(max_camera_center_abs_error=max(centers),
                                   alpha_foreground_and_background=True,rgba16=True,
                                   saturation_mask_retained=True,resize='AREA'),
                  note='原生Trainer终步保存优化器更新前状态，原生实现实际optimizer更新为iterations-1；对比必须声明')
    (model/'eldl_v3_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    trainer.train(args.iterations)
    record.update(status='TRAINING_FINISHED_PENDING_EVALUATION',iterations=args.iterations,
                  actual_optimizer_updates=args.iterations-1)
    (model/'eldl_v3_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    (model/'training_finished.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
