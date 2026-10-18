"""只读验证新SH预热父实验，生成纯几何初始化工程收据。"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控'
FORMAL=ROOT/'output/1019-11-LH-static-jumpingjacks-ELDL-V3-P00共同静态SH几何预热'
SMOKE=ROOT/'output/smoke_test/1019-17-ELDL-V3-SH真实更新保存与纯几何包'
FRAMES=[i for i in range(1,151) if i%10]
KEYS={'_xyz':3,'_scaling':2,'_rotation':4,'_opacity':1}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)

def audit(experiment):
    experiment=experiment.resolve();require(experiment in (FORMAL,SMOKE),'不接纳任意历史模型父路径。')
    target=experiment/'parent_geometry_engineering_review.json';require(not target.exists(),'禁止覆盖父验收。')
    model=experiment/'model_mlp';source=CONTROL/'data_train_v2';state=read(experiment/'state.json');plan=read(experiment/'plan.json')
    require(state['exit_code']==0 and state.get('captured_while_alive'),'父训练未正常完成或缺真实native记录。')
    parent_path=model/'geometry_parent_manifest.json';parent=read(parent_path);record=read(model/'eldl_v3_training.json')
    require(parent['smoke']==(experiment==SMOKE) and parent['status']=='COMPLETED_SH_WARMUP_PENDING_ENGINEERING_REVIEW','父模型身份错误。')
    budget=20 if parent['smoke'] else 16000
    require(parent['completed_optimizer_calls']==budget and record['completed_optimizer_calls']==budget and record['args']['iterations']==budget,'真实预算不符。')
    require(parent['gt_supervision'] is False and parent['external_test_read'] is False and parent['pure_geometry_only'],'不是无GT纯几何来源。')
    require(parent['physics_initialization_permitted'] is False and record['init_weights_loaded'] is False,'父模型来源声明错误。')
    require(record['actual_entrypoint']=='scripts/train_eldl_v3_sh_warmup.py' and record['baseline']=='static_sh','不是新独立SH入口。')
    require(record['status']=='TRAINING_FINISHED_PENDING_EVALUATION','父记录未完成。')
    require(parent['checkpoint_semantics']==record['checkpoint_semantics']=='post_optimizer_post_native_topology','拒绝旧N-1/更新前保存。')
    require(parent['train_frames']==record['train_frames']==FRAMES and parent['source']==str(source),'必须原完整135 TRAIN。')
    require(sha(model/'eldl_v3_training.json')==parent['model_record_sha256'],'父模型记录变化。')
    manifest=read(source/'no_gt_manifest.json');require(manifest['gt_read'] is False and manifest['init_weights_loaded'] is False,'源不合法。')
    require(parent['source_files']==manifest['files'] and len(parent['source_files'])==137,'源清单不完整。')
    require({'points3d.ply','transforms_train.json'}.issubset(parent['source_files']),'源缺必需几何/相机。')
    require(sha(source/'no_gt_manifest.json')==parent['source_manifest_sha256'],'源manifest变化。')
    inputs={str(parent_path):sha(parent_path),str(experiment/'state.json'):sha(experiment/'state.json'),str(experiment/'plan.json'):sha(experiment/'plan.json'),str(model/'eldl_v3_training.json'):sha(model/'eldl_v3_training.json'),str(source/'no_gt_manifest.json'):sha(source/'no_gt_manifest.json'),str(Path(__file__).resolve()):sha(__file__)}
    for name,value in parent['source_files'].items():
        path=(source/name).resolve();require(path.is_relative_to(source) and sha(path)==value,'源文件变化或路径逃逸。');inputs[str(path)]=value
    code_root=ROOT if parent['smoke'] else CONTROL/'snapshots/sh_warmup_v1'
    require(record['source_code_root']==str(code_root),'父源码根目录错误。')
    for name,value in parent['code_sha256'].items():
        path=Path(name).resolve();require(path.is_relative_to(code_root) and path.suffix=='.py' and sha(path)==value,'父源码变化或逃逸。');inputs[str(path)]=value
    for name,value in state['runtime_modules'].items():require(sha(name)==value,'实际native二进制变化。');inputs[name]=value
    if not parent['smoke']:
        require(sha(code_root/'snapshot_manifest.json')==plan['snapshot_manifest_sha256'],'冻结源码快照变化。')
        inputs[str(code_root/'snapshot_manifest.json')]=sha(code_root/'snapshot_manifest.json')
    counts={}
    for step,endpoint in parent['endpoints'].items():
        k=int(step);folder=model/f'geometry_init/actual_{k}'
        expected={'geometry':folder/'geometry.pt','receipt':folder/'receipt.json','ply':model/f'point_cloud/iteration_{k}/point_cloud.ply','deform':model/f'deform/iteration_{k}/deform.pth'}
        for key,path in expected.items():
            require(endpoint[key]==str(path) and path.resolve()==path and sha(path)==endpoint[key+'_sha256'],'父端点路径逃逸或SHA错误。');inputs[str(path)]=sha(path)
        receipt=read(expected['receipt']);require(receipt['requested_step']==receipt['completed_optimizer_calls']==k and receipt['optimizer_audit']['iteration']==k,'端点实际步数错误。')
        require(receipt['checkpoint_semantics']==parent['checkpoint_semantics'] and receipt['gt_supervision'] is False,'端点语义错误。')
        values=torch.load(expected['geometry'],map_location='cpu',weights_only=True)
        require(set(values)==set(KEYS),'纯几何包混入其他权重。');n=len(values['_xyz']);require(n>0,'空几何。')
        require(all(v.shape==(n,KEYS[key]) and v.dtype==torch.float32 and torch.isfinite(v).all() for key,v in values.items()),'几何schema无效。')
        scale=values['_scaling'].exp();require(torch.isfinite(scale).all() and (scale>0).all() and (values['_rotation'].norm(dim=1)>0).all(),'尺度或旋转无效。')
        vertex=PlyData.read(expected['ply'])['vertex'];fields={'_xyz':['x','y','z'],'_scaling':['scale_0','scale_1'],'_rotation':['rot_'+str(i) for i in range(4)],'_opacity':['opacity']}
        for key,names in fields.items():require(torch.equal(values[key],torch.from_numpy(np.stack([vertex[x] for x in names],axis=1))),'纯几何与保存PLY不一致。')
        weights=torch.load(expected['deform'],map_location='cpu',weights_only=True)
        require(all(torch.count_nonzero(weights[h+'.'+s])==0 for h in ('gaussian_warp','gaussian_rotation','gaussian_scaling') for s in ('weight','bias')),'静态几何头非零。')
        counts[step]=n
    selected=['3','20'] if parent['smoke'] else ['1000','16000'];require(all(k in counts for k in selected),'短长端点缺失。')
    result=dict(status='PASS_SH_PARENT_GEOMETRY_ENGINEERING',parent_manifest=str(parent_path),parent_manifest_sha256=sha(parent_path),smoke=parent['smoke'],allowed_steps=[int(k) for k in selected],checkpoint_point_counts=counts,gt_read=False,external_test_read=False,accuracy_claim=False,initialization_scope='仅同smoke类型的预登记纯几何入口；不授权转移颜色/MLP/Adam/RNG',input_sha256=inputs)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,required=True);audit(parser.parse_args().experiment)
