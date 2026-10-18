"""135TRAIN无GT静态SH预热，真实1000/16000更新后导出纯几何。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene.eldl_v3_sh_warmup import PostStepSHTrainer as Trainer
from scripts.train_eldl_v3_baseline import validate_source
from scripts.train_static_dl import freeze_geometry_heads
from utils.general_utils import safe_state


def main():
    import socket
    assert socket.gethostname()=='ushas' and Path(sys.executable).resolve()==Path('/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python').resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    lp,op,pp = ModelParams(parser),OptimizationParams(parser),PipelineParams(parser)
    parser.add_argument('--baseline',choices=['static_sh'],default='static_sh')
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
    elif args.iterations!=16000:parser.error('本轮正式预热固定16000次optimizer调用')
    if not args.smoke and not {1000,12000,13000,14000,14999,15000,15100,16000}.issubset(set(args.save_iterations)):
        parser.error('正式预热必须保存预登记的实际1k/16k及末五点和reset边界。')
    manifest = validate_source(source,model)
    if manifest['train_frames'] != [i for i in range(1,151) if i%10]:parser.error('只允许共同135 TRAIN，不允许117 FIT模型')
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
                  actual_entrypoint='scripts/train_eldl_v3_sh_warmup.py',checkpoint_semantics='post_optimizer_post_native_topology',
                  train_frames=manifest['train_frames'],internal_held_frames=manifest['internal_held_frames'],
                  external_test_read=False,input_manifest_sha256=hashlib.sha256((source/'no_gt_manifest.json').read_bytes()).hexdigest(),
                  rasterizer_provenance=backend.provenance,baseline=args.baseline,
                  source_code_root=str(Path(__file__).resolve().parents[1]),
                  input_audit=dict(max_camera_center_abs_error=max(centers),
                                   alpha_foreground_and_background=True,rgba16=True,
                                   saturation_mask_retained=True,resize='AREA'),
                  note='只修保存与终点调用边界；保留native拓扑在optimizer之前。实际调用数不等于每组有效更新数')
    (model/'eldl_v3_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    trainer.train(args.iterations)
    record.update(status='TRAINING_FINISHED_PENDING_EVALUATION',iterations=args.iterations,
                  completed_optimizer_calls=trainer.completed_optimizer_calls)
    assert trainer.completed_optimizer_calls==args.iterations
    root=Path(__file__).resolve().parents[1]
    code_files=[root/p for p in ('scripts/train_eldl_v3_sh_warmup.py','scene/eldl_v3_sh_warmup.py','scripts/train_eldl_v3_baseline.py','scripts/train_stage1.py','scripts/train_static_dl.py','scene/gaussian_model.py','scene/dataset_readers.py','utils/rgba16.py')]
    checksum=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    endpoints={}
    for folder in sorted((model/'geometry_init').glob('actual_*')):
        receipt=json.loads((folder/'receipt.json').read_text());k=receipt['requested_step']
        assert folder.name==f'actual_{k}' and receipt['completed_optimizer_calls']==k
        assert receipt['checkpoint_semantics']=='post_optimizer_post_native_topology'
        assert receipt['optimizer_audit']['iteration']==k and receipt['gt_supervision'] is False
        assert checksum(folder/'geometry.pt')==receipt['geometry_sha256']
        assert checksum(model/f'point_cloud/iteration_{k}/point_cloud.ply')==receipt['ply_sha256']
        assert checksum(model/f'deform/iteration_{k}/deform.pth')==receipt['deform_sha256']
        endpoints[str(k)]=dict(geometry=str(folder/'geometry.pt'),geometry_sha256=checksum(folder/'geometry.pt'),
            receipt=str(folder/'receipt.json'),receipt_sha256=checksum(folder/'receipt.json'),
            ply=str(model/f'point_cloud/iteration_{k}/point_cloud.ply'),ply_sha256=receipt['ply_sha256'],
            deform=str(model/f'deform/iteration_{k}/deform.pth'),deform_sha256=receipt['deform_sha256'])
    if not args.smoke:assert all(str(k) in endpoints for k in (1000,12000,13000,14000,14999,15000,15100,16000))
    assert checksum(source/'no_gt_manifest.json')==record['input_manifest_sha256']
    assert all(checksum(source/p)==value for p,value in manifest['files'].items())
    (model/'eldl_v3_training.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    parent=dict(status='COMPLETED_SH_WARMUP_PENDING_ENGINEERING_REVIEW',gt_supervision=False,
        external_test_read=False,source=str(source),source_manifest_sha256=checksum(source/'no_gt_manifest.json'),
        source_files=manifest['files'],train_frames=manifest['train_frames'],seed=0,
        checkpoint_semantics=record['checkpoint_semantics'],completed_optimizer_calls=args.iterations,
        model_record_sha256=checksum(model/'eldl_v3_training.json'),endpoints=endpoints,
        code_sha256={str(p):checksum(p) for p in code_files},rasterizer_provenance=backend.provenance,
        smoke=args.smoke,pure_geometry_only=True,physics_initialization_permitted=False)
    (model/'geometry_parent_manifest.json').write_text(json.dumps(parent,ensure_ascii=False,indent=2)+'\n')
    (model/'training_finished.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
