"""独立CPU评价冻结的V3投影估光；不选择或回写预测。"""
import argparse
import hashlib
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import torch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(values):
    return dict(count=len(values),mean_deg=float(np.mean(values)),median_deg=float(np.median(values)),
                p95_deg=float(np.percentile(values,95)),max_deg=float(np.max(values)),min_deg=float(np.min(values)))


def main(args):
    root=Path.cwd();out=args.experiment.resolve();freeze_path=out/'freeze.json'
    freeze=json.loads(freeze_path.read_text());assert freeze['status']=='FROZEN_BEFORE_GT' and not freeze['gt_read']
    for relative,digest in freeze['artifact_sha256'].items():assert sha(out/relative)==digest,relative
    evaluation=out/'eval_only';evaluation.mkdir(exist_ok=True)
    assert not (evaluation/'audit.json').exists() and not (evaluation/'direction_comparison.png').exists()
    raw=root/'data/LH-data/LH-dynamic-jumpingjacks-ELDL-V3';gtpath=raw/'lights.json'
    records=json.loads(gtpath.read_text());gt=[]
    for n in range(1,151):
        item=records[f'{n:04d}'];assert item['source_type']=='SUN' and item['primary_light']=='Orbit_Directional_Sun'
        value=np.asarray(item['light_dir_world'],dtype=np.float64)
        matrix=np.asarray(item['lights'][item['primary_light']]['matrix_world'])[:3,2]
        assert np.allclose(value/np.linalg.norm(value),matrix/np.linalg.norm(matrix),atol=1e-7,rtol=0)
        gt.append(value/np.linalg.norm(value))
    gt=np.asarray(gt)
    spec=importlib.util.spec_from_file_location('only_light_table',root/'scene/stage2_photometric.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames=np.arange(1,151);times=(frames-1)/149
    test=frames%10==0;dev=(frames>=61)&(frames<=79)&~test;train=~(test|dev)
    splits=dict(TRAIN=train,DEV=dev,TEST=test,ALL=np.ones(150,dtype=bool))
    summaries={};predictions={};errors={};per_frame={}
    for name,relative in [('original','original/result.json'),('ambient_coarse','ambient_contrast/coarse_result.json'),
                          ('ambient_refined','ambient_contrast/result.json')]:
        path=out/relative
        if not path.exists():continue
        result=json.loads(path.read_text())
        if 'directions' not in result:
            summaries[name]=dict(status=result['status']);continue
        assert not result['gt_read'] and result['train_frames']==frames[train].tolist()
        table=module.DirectionalLightTable(torch.tensor(result['times']),torch.tensor(result['directions']),False)
        with torch.no_grad():pred=torch.stack([table(float(t)) for t in times]).double().numpy()
        pred/=np.linalg.norm(pred,axis=-1,keepdims=True)
        angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(pred,gt),axis=-1),np.sum(pred*gt,axis=-1)))
        summaries[name]={split:stats(angle[mask]) for split,mask in splits.items()}
        predictions[name]=pred;errors[name]=angle
        per_frame[name]=[dict(frame=int(n),pred=pred[i].tolist(),gt=gt[i].tolist(),error_deg=float(angle[i])) for i,n in enumerate(frames)]
    previous=root/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控/eval_only/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-方向光真值角误差/audit.json'
    old=json.loads(previous.read_text())
    for new,key in [('previous_fixed','A0_8100'),('previous_DIR','DIR_8100')]:
        record=old['records'][key];summaries[new]=record['summary']
        errors[new]=np.asarray([x['angular_error_deg'] for x in record['per_frame']])
    passed='ambient_refined' in predictions and summaries['ambient_refined']['TEST']['mean_deg']<summaries['previous_fixed']['TEST']['mean_deg']
    audit=dict(status='PASS_DIRECTION_MEAN_IMPROVED' if passed else 'FAILED_DIRECTION_MEAN_NOT_IMPROVED',
        primary_preselected='ambient_refined',criterion='预先固定主方案TEST均值低于上一轮固定灯向；不代表完整渲染/几何收益',
        freeze_sha256=sha(freeze_path),gt_sha256=sha(gtpath),previous_audit_sha256=sha(previous),
        predictions_unchanged=True,optimizer_steps=0,gpu_initialized=torch.cuda.is_initialized(),
        angle='世界坐标表面指向SUN；atan2(norm(cross),dot)，不翻转、不GT对齐',
        interpolation='原DirectionalLightTable；未支持时刻与DEV/TEST仅归一化线性插值，端点夹取',
        summary=summaries,per_frame=per_frame,command=shlex.join([sys.executable]+sys.argv))
    assert not audit['gpu_initialized']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,8))
    for name,angle in errors.items():axes[0].plot(frames,angle,label=name,linewidth=1)
    axes[0].scatter(frames[test],np.zeros(test.sum()),marker='|',c='black',label='TEST frames')
    axes[0].set_ylabel('SUN angular error (deg)');axes[0].legend(ncol=3,fontsize=8);axes[0].grid(alpha=.2)
    if 'ambient_refined' in predictions:
        for dim,label in enumerate('XYZ'):
            axes[1].plot(frames,gt[:,dim],label=f'GT {label}')
            axes[1].plot(frames,predictions['ambient_refined'][:,dim],'--',label=f'projection {label}')
    axes[1].set_xlabel('Frame');axes[1].set_ylabel('World unit SUN direction');axes[1].legend(ncol=3,fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(evaluation/'direction_comparison.png',dpi=160);plt.close(fig)
    (evaluation/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    for relative,digest in freeze['artifact_sha256'].items():assert sha(out/relative)==digest,relative
    print(json.dumps(dict(status=audit['status'],summary=summaries),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    main(p.parse_args())
