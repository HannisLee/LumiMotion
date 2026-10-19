"""仅CPU审计固定8100终点的真实参数变化/冻结范围，不读TEST或GT。"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import sys

import numpy as np
from plyfile import PlyData
import torch

from LIHAN.scripts.dynamic_v3_pair_audit import tensor_difference
from utils.no_gt_guard import install_no_gt_guard


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def ply_fields(path):
    data=PlyData.read(path)['vertex'].data
    return {name:torch.from_numpy(np.asarray(data[name]).copy().astype(np.float64)) for name in data.dtype.names}


def summarize_angle(first,last):
    first=torch.nn.functional.normalize(first.double(),dim=-1)
    last=torch.nn.functional.normalize(last.double(),dim=-1)
    angle=torch.atan2(torch.linalg.cross(first,last,dim=-1).norm(dim=-1),(first*last).sum(-1))*180/torch.pi
    return dict(min=float(angle.min()),median=float(angle.median()),max=float(angle.max()),
        mean=float(angle.mean()),per_train_time=angle.tolist())


def audit(folder,out):
    torch.set_num_threads(4)
    marker_path=folder/'dynamic_v3_stage2.json';marker=read(marker_path)
    assert marker['actual_optimizer_updates']==8100 and marker['parent_iteration']==35000
    assert not marker['gt_supervision'] and not marker['external_test_read'] and not marker['parent_optimizer_loaded']
    manifest_path=Path(marker['source'])/'no_gt_manifest.json';manifest=read(manifest_path)
    assert marker['train_frames']==manifest['train_frames'] and len(marker['train_frames'])==117
    parent=Path(marker['parent']).resolve()
    assert parent.name=='model_mlp' and parent.parent.name=='1019-59-LH-dynamic-jumpingjacks-ELDL-V3-N00原生Stage1'
    allowed=[folder,parent,out,Path(marker['source']),Path(marker['stats']),Path(marker['lighting'])]
    install_no_gt_guard(allowed)
    sources={str(marker_path):sha(marker_path),str(manifest_path):sha(manifest_path)};states={};deforms={};plies={}
    for path,digest in marker['source_sha256'].items():
        assert sha(path)==digest,path
        sources[path]=digest
    for step in (0,1350,4050,8100):
        receipt_path=folder/f'dynamic_v3/iteration_{step}/receipt.json';receipt=read(receipt_path)
        assert receipt['completed_updates']==step and receipt['cumulative_actual_updates']==34999+step
        sources[str(receipt_path)]=sha(receipt_path)
        for relative,digest in receipt['checkpoint_sha256'].items():
            path=(folder/relative).resolve();assert path.is_relative_to(folder) and sha(path)==digest
            sources[str(path)]=digest
        states[step]=torch.load(folder/f'dynamic_v3/iteration_{step}/state.pt',map_location='cpu',weights_only=False)
        deforms[step]=torch.load(folder/f'deform/iteration_{step}/deform.pth',map_location='cpu',weights_only=False)
        plies[step]=ply_fields(folder/f'point_cloud/iteration_{step}/point_cloud.ply')
    initial,final=states[0],states[8100];config=final['config']
    assert config==marker['config'] and config['geometry_start']==1351 and config['deform_start']==4051
    assert config['iterations']==8100 and config['reset_steps']==[3000,6000] and config['protect_steps']==200
    assert config['base_end']==540 and config['direction_anchor']==.1 and config['direction_smooth']==.01
    parent_ply=parent/'point_cloud/iteration_35000/point_cloud.ply'
    parent_deform=parent/'deform/iteration_35000/deform.pth'
    sources[str(parent_ply)]=sha(parent_ply);sources[str(parent_deform)]=sha(parent_deform)
    parent_geometry=tensor_difference(ply_fields(parent_ply),plies[0])
    parent_mlp=tensor_difference(torch.load(parent_deform,map_location='cpu',weights_only=False),deforms[0])
    assert all(row['bitwise_equal'] for row in parent_geometry.values()),'初始化PLY未完整继承共同原生父资产'
    assert all(row['bitwise_equal'] for row in parent_mlp.values()),'初始化MLP未完整继承共同原生父资产'
    prefix_geometry=tensor_difference(plies[0],plies[1350])
    prefix_mlp=tensor_difference(deforms[0],deforms[4050])
    assert all(row['bitwise_equal'] for row in prefix_geometry.values()),'1350前几何发生变化'
    assert all(row['bitwise_equal'] for row in prefix_mlp.values()),'4050前形变发生变化'
    geometry=tensor_difference(plies[0],plies[8100]);mlp=tensor_difference(deforms[0],deforms[8100])
    physical=tensor_difference(initial['radiance'],final['radiance'])
    assert all(row['finite'] for data in (geometry,mlp,physical) for row in data.values())
    geometry_names=[name for name in geometry if name in ('x','y','z','opacity') or name.startswith(('rot_','scale_'))]
    for names in (('x','y','z'),tuple(k for k in geometry if k.startswith('rot_')),
                  tuple(k for k in geometry if k.startswith('scale_')),('opacity',)):
        assert any(not geometry[k]['bitwise_equal'] for k in names),'预期几何组没有实际变化: '+str(names)
    frozen_fields={name:row for name,row in geometry.items() if name not in geometry_names}
    assert all(row['bitwise_equal'] for row in frozen_fields.values()),'冻结PLY字段发生变化'
    color_mlp={name:row for name,row in mlp.items() if name.startswith('gaussian_color')}
    assert all(row['bitwise_equal'] for row in color_mlp.values()),'原生颜色头意外更新'
    assert any(not row['bitwise_equal'] for name,row in mlp.items() if not name.startswith('gaussian_color')),'形变未实际更新'
    for key in ('normal_sign','normal_sign_valid'):
        assert torch.equal(initial[key],final[key]),'固定法线朝向发生变化'
    for key in ('raw_rho','raw_intensity'):
        assert not physical[key]['bitwise_equal'],key+'未实际更新'
    assert torch.equal(initial['radiance']['gauge_support'],final['radiance']['gauge_support'])
    if marker['path']=='A':
        assert not physical['environment_coefficients']['bitwise_equal']
        assert torch.equal(states[1350]['radiance']['environment_coefficients'],final['radiance']['environment_coefficients'])
    else:assert physical['environment_coefficients']['bitwise_equal']
    first=initial['radiance']['lights.raw_directions'];last=final['radiance']['lights.raw_directions']
    assert torch.equal(initial['radiance']['lights.times'],final['radiance']['lights.times'])
    direction_lr=config['direction_lr'];assert direction_lr in (0.,1e-5)
    assert (not torch.equal(first,last)) if direction_lr else torch.equal(first,last),'灯向更新范围不符'
    if direction_lr:assert float((last.norm(dim=-1)-1).abs().max())<1e-5
    groups={group['name']:group for group in final['optimizer']['param_groups']}
    assert ('light' in groups)==bool(direction_lr)
    if direction_lr:assert groups['light']['lr']==1e-5
    assert groups['deformation']['lr']>0 and all(groups[name]['lr']>0 for name in ('xyz','rotation','scale','opacity','rho','D'))
    if marker['path']=='A':assert groups['SH']['lr']==0.
    counters={name:[float(final['optimizer']['state'][pid]['step']) if pid in final['optimizer']['state'] else None
                    for pid in group['params']] for name,group in groups.items()}
    log_path=folder.parent/'train_audit.jsonl';log=[json.loads(line) for line in log_path.read_text().splitlines()]
    assert log[-1]['step']==8100
    assert all(row['lrs']['deformation']==0 for row in log if row['step']<4051)
    assert all(row['lrs']['deformation']>0 for row in log if row['step']>=4051)
    if marker['path']=='A':
        assert all(row['lrs']['SH']==(.001 if row['step']<=540 else 0.) for row in log)
    else:assert all('SH' not in row['lrs'] for row in log)
    if direction_lr:
        assert all(row['lrs']['light']==(0. if row['step']<1351 else 1e-5) for row in log)
    else:assert all('light' not in row['lrs'] for row in log)
    reset_path=folder.parent/'opacity_resets.json';resets=read(reset_path)
    assert [row['step'] for row in resets]==[3000,6000]
    sources[str(log_path)]=sha(log_path);sources[str(reset_path)]=sha(reset_path)
    for path,digest in sources.items():assert sha(path)==digest
    return dict(status='PASS_FIXED_TERMINAL_OPTIMIZATION_CONTRACT_ONLY',model=str(folder),iteration=8100,
        gt_read=False,test_read=False,model_accuracy_evaluated=False,parent_iteration=35000,
        added_optimizer_calls=8100,cumulative_actual_updates=43099,optimizer_reinitialized=True,
        parent_geometry_bitwise=True,parent_deformation_bitwise=True,prefix_geometry_1350_bitwise=True,
        prefix_deformation_4050_bitwise=True,geometry_changes=geometry,deformation_changes=mlp,
        physical_changes=physical,final_optimizer_lrs={k:v['lr'] for k,v in groups.items()},
        optimizer_step_counters=counters,counter_limitation='Adam step包含lr为0但有梯度的调用，不能等同实际非零lr更新次数',
        nominal_nonzero_lr_steps=dict(geometry=6750,deformation=4050,rho=8100,D=8100,
            SH=540 if marker['path']=='A' else 0,light=6750 if direction_lr else 0),
        schedule_evidence='起止配置、绑定训练源码SHA及全部已记录审计行学习率相符；日志非逐步采样，不冒称独立核验每一步',
        sh_frozen_checkpoint_evidence='1350与8100环境系数逐位相同；没有540保存点，541起冻结另由绑定源码/配置及600起审计学习率支持' if marker['path']=='A' else 'B环境系数0与8100逐位相同且无SH优化器组',
        direction_lr=direction_lr,direction_angle_degrees=summarize_angle(first,last),
        final_direction_norm_max_error=float((last.norm(dim=-1)-1).abs().max()),
        direction_gradient='训练仅固定V近似导数，每次前向完整当前G_t BVH重算，不声称完整可见度梯度',
        opacity_reset_events=resets,source_sha256=sources,
        limitation='证明保存资产与更新/冻结契约；不证明实际求导每步非零、动态PS覆盖或质量提升')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    assert socket.gethostname()=='ushas' and out.is_relative_to(root/'output') and 'eval_only' not in out.parts
    out.mkdir(parents=True,exist_ok=True)
    assert all(p.name in ('run.log','launch.json') for p in out.iterdir()),'禁止覆盖审计结果'
    argv=list(getattr(sys,'orig_argv',[sys.executable]+sys.argv));command=shlex.join(argv)
    try:result=audit(args.model.resolve(),out)
    except Exception as error:
        result=dict(status='FAILED_OPTIMIZATION_CONTRACT',error=repr(error),model=str(args.model.resolve()),gt_read=False,test_read=False)
    result.update(command=command,code_sha256=sha(__file__))
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    passed=result['status'].startswith('PASS_')
    (out/'README.md').write_text('# 固定终点优化契约只读审计\n\n完整命令：\n\n```bash\n'+command+
        '\n```\n\nsource：模型marker所列117 TRAIN及共同原生父资产；model：'+str(args.model.resolve())+
        '；iteration：0/1350/4050/8100。日志run.log，定量audit.json。本入口不渲染，代表图/视频及四类目检不适用。'
        '\n\n最终结论：'+('PASS' if passed else 'FAILED')+'（仅参数更新/冻结契约，非科学质量结论）。\n')
    print(json.dumps({k:result[k] for k in ('status','error','direction_lr','direction_angle_degrees') if k in result},ensure_ascii=False),flush=True)
    if not passed:raise RuntimeError(result['error'])


if __name__=='__main__':main()
