#!/usr/bin/env python3
"""从已完成的135帧原生贡献诊断固化w；只读NPZ和来源SHA，不调用模型。"""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic_dir',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    a = p.parse_args()
    source,output = a.diagnostic_dir.resolve(),a.output_dir.resolve()
    root = Path(__file__).resolve().parents[2]
    if not output.is_relative_to(root/'output') or output.exists():
        p.error('只允许output内全新资产目录。')
    contract = json.loads((source/'contract.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if (summary['status']!='PASS' or summary['frame_count']!=135 or summary['parameter_updates']!=0
        or not all(summary[k] for k in ('parameters_exactly_unchanged','RNG_exactly_unchanged','inputs_and_sources_sha_unchanged'))):
        raise AssertionError('源诊断未完整通过只读门槛。')
    frames = {str(i).zfill(4) for i in range(1,151) if i%10}
    if set(contract['frames']) != frames:
        raise AssertionError('源不是原135训练帧。')
    model = Path(contract['model'])
    checkpoint = {path:digest for path,digest in contract['input_and_code_sha256'].items()
                  if Path(path).is_relative_to(model) or path.endswith('transforms_train.json')
                  or 'directions' in Path(path).name}
    if not checkpoint or any(sha(path)!=value for path,value in checkpoint.items()):
        raise AssertionError('源检查点/方向/相机SHA变化。')
    output.mkdir(parents=True)
    records = {}
    try:
        for key in sorted(frames):
            file = source/'frames'/f'{key}.npz'
            d = np.load(file,allow_pickle=False)
            m,mu,v = d['foreground_mass'].astype(np.float64),d['mu'].astype(np.float64),d['visibility'].astype(np.float64)
            if any(x.shape!=(summary['point_count'],) or not np.isfinite(x).all() for x in (m,mu,v)):
                raise AssertionError('逐点资产非有限或长度错误。')
            active = m>.01
            x = np.clip((mu-.05)/.20,0,1)
            g = x*x*(3-2*x)*4*v*(1-v)
            mean = float(np.sum(m[active]*g[active])/m[active].sum()) if active.any() else 0.
            weight = np.ones_like(m)
            weight[active] = (1+g[active])/(1+mean)
            weight = weight.astype(np.float32)
            actual_mean = float(np.sum(m[active]*weight[active])/m[active].sum()) if active.any() else 1.
            if weight.min()<.5 or weight.max()>2 or abs(actual_mean-1)>1e-6:
                raise AssertionError('固定w范围或贡献均值1检查失败。')
            dest = output/f'{key}.npy'
            np.save(dest,weight,allow_pickle=False)
            records[key] = {'file':dest.name,'sha256':sha(dest),'diagnostic_npz_sha256':sha(file),
                'active_points':int(active.sum()),'g_contribution_mean':mean,'w_contribution_mean':actual_mean,
                'w_min':float(weight.min()),'w_max':float(weight.max()),'w_point_mean':float(weight.mean())}
        result = {'status':'PASS','model':str(model),'iteration':contract['iteration'],'point_count':summary['point_count'],
                  'source_checkpoint_sha256':checkpoint,'diagnostic_dir':str(source),'frames':records,
                  'diagnostic_contract_sha256':sha(source/'contract.json'),'diagnostic_summary_sha256':sha(source/'summary.json'),
                  'preparer_sha256':sha(__file__),'parameter_updates':0,
                  'formula':'A=(m>.01); g=smoothstep((mu-.05)/.2)*4V(1-V); gbar=sum_A(m*g)/sum_A(m); w_A=(1+g)/(1+gbar), else1',
                  'normalization':'原生前景合成质量加权均值1，不是点数均值1','empty_A':'all weights1',
                  'source_classes_fixed':True,'H_weight_enabled':False}
        (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('135帧固定方向权重资产PASS：'+str(output/'manifest.json'))
    except Exception as error:
        (output/'FAILED.json').write_text(json.dumps({'status':'FAILED','error':str(error)},ensure_ascii=False)+'\n')
        raise


if __name__ == '__main__':
    main()
