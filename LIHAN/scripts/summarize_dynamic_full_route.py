"""完整路线结果的只读同分母汇总；不选择新检查点、不更改训练资产。"""
import argparse
import hashlib
import json
from pathlib import Path

from LIHAN.scripts.dynamic_v3_compare import verify_summary_arithmetic, denominator

ROOT=Path(__file__).resolve().parents[2]
METHODS=('gs_render','depth_normal','mesh')


def main(control):
    evidence={}
    def read(path):
        path=Path(path).resolve();evidence[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    state=read(control/'state.json');assert state['status'] in ('READY_FOR_REVIEW','COMPLETED_REVIEWED')
    assert all(j['exit_code']==0 for j in state['jobs'].values())
    lock=read(control/'evaluation_lock.json');assert lock['all_six_frozen_before_test'] and lock['primary_iteration']==16500
    baselines={
        'v3':{
            '原生S1':ROOT/'output/1019-58-LH-dynamic-jumpingjacks-ELDL-V3-两阶段PS总控/eval_only/1019-06-LH-dynamic-jumpingjacks-ELDL-V3-N00统一最终评价',
            '旧H0':ROOT/'output/1020-06-LH-dynamic-jumpingjacks-ELDL-V3-H0历史损失Stage2/eval_only/1020-01-LH-dynamic-jumpingjacks-ELDL-V3-H0统一评价'},
        'dlv2':{label:ROOT/f'output/1021-02-LH-dynamic-jumpingjacks_DL_V2-阴影投影H0Stage2复现重试/eval_only/1021-01-LH-dynamic-jumpingjacks_DL_V2-{tag}统一评价'
            for label,tag in [('原生S1','S1'),('旧H0','H0')]}}
    records={}
    for plan in read(control/'experiments.json'):
        dataset=plan['dataset'];out=Path(plan['output']);finished=read(out/'training_finished.json')
        assert finished['actual_optimizer_updates']==18000 and not finished['gt_read']
        events=read(out/'topology_reset_events.json')
        assert [r['iteration'] for r in events if r['event']=='opacity reset']==[3000,6000,9000,12000,15000]
        assert [r['iteration'] for r in events if 'Adam' in r['event']]==[10001,12001,14001,16001]
        rows=[json.loads(s) for s in (out/'train_audit.jsonl').read_text().splitlines()]
        assert rows[-1]['actual_optimizer_updates']==18000
        max_residual=max(abs(r['loss']-(r['photo']+.1*r['alpha']+.1*r['normal']+1000*r['distortion']+r['motion'])) for r in rows)
        assert max_residual<1e-6
        import torch
        import torch.nn.functional as F
        lighting=read(out/'lighting.json');endpoint_audits={}
        for step in (16000,16500,18000):
            folder=out/'model_mlp';receipt=read(folder/f'dynamic_full_route/iteration_{step}/receipt.json')
            for relative,digest in receipt['checkpoint_sha256'].items():
                assert hashlib.sha256((folder/relative).read_bytes()).hexdigest()==digest
            checkpoint=torch.load(folder/f'dynamic_full_route/iteration_{step}/state.pt',map_location='cpu',weights_only=False)
            # 原训练在CUDA上归一化；CPU重算可差一个float32 ULP，不能据此判作灯向漂移。
            expected=F.normalize(torch.tensor(lighting['directions'],dtype=torch.float32,device='cuda'),dim=-1).cpu()
            assert torch.equal(expected,checkpoint['lighting']['table.raw_directions'])
            if dataset=='dlv2':assert torch.count_nonzero(checkpoint['lighting']['coefficients'])==0
            endpoint_audits[str(step)]=dict(directions_bitwise_unchanged=True,receipt_verified=True,
                light_intensity=float(checkpoint['lighting']['log_intensity'].exp()),
                environment_enabled=checkpoint['environment_enabled'],
                points=len(checkpoint['max_radii2D']),actual_optimizer_updates=checkpoint['actual_optimizer_updates'])
        current={**baselines[dataset],**{str(s):out/f'eval_only/iteration_{s}' for s in (16000,16500,18000)}}
        metrics={};support=None
        for label,path in current.items():
            m=read(path/'metrics.json');verify_summary_arithmetic(m)
            if support is None:support=denominator(m)
            assert denominator(m)==support
            normal={region:{method:v['missing180_frame_mean_deg'] for method,v in value['normal'].items()}
                for region,value in m['summary'].items()}
            composite={k:sum(v.values())/3 if all(x is not None for x in v.values()) else None for k,v in normal.items()}
            conditional={k:v['mae_pixel_mean_deg'] for k,v in m['summary']['foreground']['normal'].items()}
            conditional_by_region={region:{k:v['mae_pixel_mean_deg'] for k,v in value['normal'].items()}
                for region,value in m['summary'].items()}
            surface=read(path/'surface_completeness.json');temporal=read(path/'temporal_residuals.json')
            metrics[label]=dict(path=str(path),iteration=m['iteration'],conditional=conditional,conditional_by_region=conditional_by_region,normal_fixed=normal,
                composite_fixed=composite,coverage={k:v['coverage_frame_mean'] for k,v in m['summary']['foreground']['normal'].items()},
                rgb=m['rgb']['foreground'],expected_depth=m['summary']['foreground']['depth']['expected'],
                fscore=surface['fscore_frame_mean'],chamfer=surface['symmetric_capped_missing_chamfer_frame_mean_world'],
                temporal_status=temporal['status'])
        baseline=metrics['原生S1'];primary=metrics['16500']
        records[dataset]=dict(metrics=metrics,primary_iteration=16500,actual_updates=18000,elapsed_seconds=finished['elapsed_seconds'],
            topology_events=len(events),loss_reconstruction_max_abs=max_residual,fixed_denominators_identical=True,
            endpoint_audits=endpoint_audits,
            gs_improvement_deg=baseline['conditional']['gs_render']-primary['conditional']['gs_render'],
            gs_improvement_percent=100*(1-primary['conditional']['gs_render']/baseline['conditional']['gs_render']),
            gs_conditional_pass=primary['conditional']['gs_render']<baseline['conditional']['gs_render'],
            gs_fixed_pass=primary['normal_fixed']['foreground']['gs_render']<baseline['normal_fixed']['foreground']['gs_render'],
            rgb_pass=primary['rgb']['mae_frame_mean']<=baseline['rgb']['mae_frame_mean'])
    for key,digest in evidence.items():assert hashlib.sha256(Path(key).read_bytes()).hexdigest()==digest
    result=dict(status='PASS_READ_ONLY_ARITHMETIC',primary_selection='预先固定16500；未按新TEST选择',records=records,evidence=evidence)
    destination=control/'result_review_v1';destination.mkdir(exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True)
    main(p.parse_args().control.resolve())
