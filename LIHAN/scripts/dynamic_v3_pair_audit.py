"""只读审计四组同父/同输入/同配置及PS前缀，不读取TEST或GT。"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

import torch


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_difference(first,second):
    assert set(first)==set(second)
    result={}
    for name,value in first.items():
        other=second[name]
        assert isinstance(value,torch.Tensor) and value.shape==other.shape and value.dtype==other.dtype,name
        delta=value.double()-other.double()
        result[name]=dict(bitwise_equal=bool(torch.equal(value,other)),
            max_abs=float(delta.abs().max()) if value.numel() else 0.,
            rms=float(delta.square().mean().sqrt()) if value.numel() else 0.,
            finite=bool(torch.isfinite(value).all() and torch.isfinite(other).all()))
    return result


def pair(first,second,paths):
    output={}
    for step in [0,1350]:
        receipts=[]
        for folder in [first,second]:
            receipt_path=folder/f'dynamic_v3/iteration_{step}/receipt.json'
            receipt=read(receipt_path);paths[str(receipt_path)]=sha(receipt_path)
            assert receipt['completed_updates']==step and receipt['parent_iteration']==35000
            for relative,digest in receipt['checkpoint_sha256'].items():
                file=(folder/relative).resolve();assert file.is_relative_to(folder) and sha(file)==digest
                paths[str(file)]=digest
            receipts.append(receipt)
        state=[torch.load(folder/f'dynamic_v3/iteration_{step}/state.pt',map_location='cpu',weights_only=False)
               for folder in [first,second]]
        deform=[torch.load(folder/f'deform/iteration_{step}/deform.pth',map_location='cpu',weights_only=False)
                for folder in [first,second]]
        ds=tensor_difference(*deform);physical=tensor_difference(state[0]['radiance'],state[1]['radiance'])
        axis=tensor_difference({key:state[0][key] for key in ['normal_sign','normal_sign_valid']},
                               {key:state[1][key] for key in ['normal_sign','normal_sign_valid']})
        ply=[folder/f'point_cloud/iteration_{step}/point_cloud.ply' for folder in [first,second]]
        geometry_exact=sha(ply[0])==sha(ply[1])
        deform_exact=all(item['bitwise_equal'] for item in ds.values())
        axis_exact=all(item['bitwise_equal'] for item in axis.values())
        maximum=max(item['max_abs'] for item in physical.values())
        finite=all(item['finite'] for item in physical.values())
        output[str(step)]=dict(geometry_ply_bitwise_equal=geometry_exact,deformation_bitwise_equal=deform_exact,
            normal_orientation_bitwise_equal=axis_exact,physical_state=physical,
            physical_max_abs=maximum,physical_bitwise_equal=maximum==0.,
            numeric_tolerance=1e-6,pass_geometry_and_numeric=geometry_exact and deform_exact and axis_exact and finite and maximum<=1e-6,
            note='独立CUDA原子累加允许产生浮点差；超过预登记阈值明确失败，不更改预算或回溯挑checkpoint')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['a0','a1','b0','b1']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[2];out=args.output.resolve()
    assert out.is_relative_to(root/'output') and 'eval_only' not in out.parts
    out.mkdir(parents=True,exist_ok=False)
    models={key:getattr(args,key.lower()).resolve() for key in ['A0','A1','B0','B1']}
    paths={};records={};configs={}
    for variant,folder in models.items():
        marker_path=folder/'dynamic_v3_stage2.json';record=read(marker_path)
        assert record['actual_optimizer_updates']==8100 and record['parent_iteration']==35000
        assert record['path']==variant[0] and record['ps_enabled']==(variant[1]=='1')
        assert not record['gt_supervision'] and not record['external_test_read']
        paths[str(marker_path)]=sha(marker_path);records[variant]=record
        configs[variant]={k:v for k,v in record['config'].items() if k!='ps_weight'}
        assert configs[variant]['direction_lr']==0.
    assert all(config==configs['A0'] for config in configs.values()),'PS开关以外共用配置不一致'
    for key in ['source','parent','parent_iteration','lighting','stats','train_frames']:
        assert all(record[key]==records['A0'][key] for record in records.values()),key
    for key in ['lighting','stats']:
        file=Path(records['A0'][key]);paths[str(file)]=sha(file)
        for record in records.values():assert record['source_sha256'][str(file)]==sha(file)
    lighting=read(records['A0']['lighting'])
    for name,digest in lighting['source_sha256'].items():assert sha(name)==digest;paths[name]=digest
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard(list(models.values())+[Path(p) for p in paths]+[out])
    pairs={name:pair(models[name+'0'],models[name+'1'],paths) for name in ['A','B']}
    passed=all(step['pass_geometry_and_numeric'] for item in pairs.values() for step in item.values())
    launcher_argv=list(getattr(sys,'orig_argv',[sys.executable]+sys.argv))
    result=dict(status='PASS_CONTROLLED_PREFIX_AUDIT' if passed else 'FAILED_NUMERICAL_PREFIX_EQUIVALENCE',
        gt_read=False,test_read=False,model_accuracy_evaluated=False,all_four_same_parent_inputs_config=True,
        pairs=pairs,source_sha256=paths,argv=sys.argv,launcher_argv=launcher_argv,
        limitation='只证明相同父资产/输入/日程与PS生效前前缀；不证明后续几何准确率、PS收益或GPU逐位确定性')
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text('# 四组同父配对前缀只读审计\n\n完整命令：\n\n```bash\n'+
        shlex.join(launcher_argv)+'\n```\n\nsource/model：audit.json所列四组完整8100终点及0/1350前缀。\n'
        'iteration：0/1350；最终终点身份固定8100。渲染输出、图片、视频：本入口不生成。\n'
        '日志与定量：audit.json，stdout仅同一摘要；四类可视化均不适用，未冒称目检。\n'
        '最终结论：'+('PASS' if passed else 'FAILED')+'（仅同父前缀契约，不是质量结论）。\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
