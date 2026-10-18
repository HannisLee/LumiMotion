"""只核对固定新旧方向成对校准的FIT终点与同起点；不读开发图/GT/TEST。"""
import argparse
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_eldl_v3 as v
from scripts.validate_eldl_v3_raster import same_tree, ANNEAL


def inspect_pair(pair):
    pair=Path(pair).resolve()
    assert pair==ROOT/'output/1019-04-LH-static-jumpingjacks-ELDL-V3-L00连续混合估光/raster_direction_material_pair_v1'
    plan=v.read(pair/'pair_plan.json')
    assert plan['cases']==['old','refined'] and plan['epochs']==30 and plan['held_scored'] is False
    assert plan['minimum_direction_benefit']==.01 and plan['terminal_budget_change']==.01
    original=pair.parent/'raster_calibration_v2_anneal/init_3/checkpoint_epoch_30.pt'
    refined_path=pair.parent/'raster_direction_refinement_v2/result/direction_state.pt'
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([pair, original, original.parent/'fixed_foreground.pt', refined_path,
        pair.parent/'cache_v3_geometry_bias', pair.parent/'fit_v4_geometry_bias',
        ROOT/'output/1019-03-LH-static-jumpingjacks-ELDL-V3-I00内部静态SH/model_mlp',
        ROOT/'output/1019-01-LH-static-jumpingjacks-ELDL-V3-总控/data_train_v2'])
    hashes={str(pair/'pair_plan.json'):v.sha(pair/'pair_plan.json'),str(original):v.sha(original),
            str(refined_path):v.sha(refined_path),str(Path(__file__).resolve()):v.sha(__file__)}
    parent=torch.load(original,map_location='cpu',weights_only=False)
    refined=torch.load(refined_path,map_location='cpu',weights_only=False)
    records={}; configs={}; zeroes={}; terminals={}
    for case in ('old','refined'):
        out=pair/case; d=out/'result'
        state=v.read(out/'state.json'); assert state['exit_code']==0
        cfg=v.read(d/'configuration.json');complete=v.read(d/'train_epoch_30.json')
        assert cfg['case']==case and cfg['gt_read'] is False and cfg['external_test_read'] is False
        assert cfg['held_scored'] is False and cfg['held_used_for_selection'] is False
        assert cfg['geometry_fit_frames']==v.FIT and cfg['held_frames']==v.HELD
        assert cfg['geometry_frozen'] and cfg['directions_frozen'] and not cfg['constant']
        assert cfg['epochs']==30 and cfg['updates']==3510 and cfg['cumulative_epochs']==120
        assert cfg['optimizer_schedule']==ANNEAL and cfg['parent_checkpoint']==str(original)
        assert cfg['parent_checkpoint_sha256']==hashes[str(original)]
        assert complete['geometry_bitexact_unchanged'] and complete['directions_bitexact_unchanged']
        assert complete['held_scored'] is False and complete['epochs']==30 and complete['cumulative_epochs']==120
        ck=torch.load(d/'checkpoint_epoch_30.pt',map_location='cpu',weights_only=False)
        z=torch.load(d/'checkpoint_epoch_0.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==30 and ck['cumulative_epoch']==120 and len(ck['history'])==30
        expected=parent['directions'] if case=='old' else refined['directions']
        assert torch.equal(ck['directions'],expected) and torch.equal(z['directions'],expected)
        for key in ('rho','log_e','optimizer','rng'):
            assert same_tree(z[key],parent[key]),(case,'initial',key)
        assert all(int(s['step'])==14040 for s in ck['optimizer']['state'].values())
        for epoch,row in enumerate(ck['history'],1):
            lr=.003 if epoch<=10 else .001 if epoch<=20 else .0001
            assert row['epoch']==epoch and row['updates']==117*epoch and row['lr']==lr
            assert row['param_group_lrs']==[lr,lr]
        scores={}
        for epoch in (0,10,20,25,30):
            score=v.read(d/f'fit_epoch_{epoch}.json')
            values=score['per_frame'];assert len(values)==117 and all(math.isfinite(x) and x>=0 for x in values)
            assert abs(sum(values)/117-score['frame_mean'])<1e-12
            scores[str(epoch)]=score['frame_mean']
        change=abs(scores['30']-scores['25'])/max(scores['25'],1e-12)
        records[case]=dict(fit=scores,terminal_absolute_relative_change=change,budget_sufficient=change<=.01)
        configs[case]=cfg;zeroes[case]=z;terminals[case]=ck
        v.check_hashes(cfg['input_sha256']);hashes.update(cfg['input_sha256'])
        for file in [out/'state.json',d/'configuration.json',d/'train_epoch_30.json',d/'checkpoint_epoch_0.pt',
                     d/'checkpoint_epoch_30.pt',d/'fixed_foreground.pt',*[d/f'fit_epoch_{n}.json' for n in (0,10,20,25,30)]]:
            hashes[str(file)]=v.sha(file)
    for key in ('rho','log_e','optimizer','rng'):
        assert same_tree(zeroes['old'][key],zeroes['refined'][key]),('pair_initial',key)
    assert terminals['old']['rng']==terminals['refined']['rng']
    for key in ('geometry_fit_frames','held_frames','loss','rho_linear_bounds','environment_over_pi_bounds',
                'optimizer','optimizer_schedule','backend_provenance','parent_checkpoint_sha256','foreground_sha256'):
        assert configs['old'][key]==configs['refined'][key],key
    benefit=1-records['refined']['fit']['30']/records['old']['fit']['30']
    passed=all(row['budget_sufficient'] for row in records.values()) and benefit>=.01
    return dict(status='PASS_EQUAL_BUDGET_FIT_PAIR' if passed else 'FAILED_EQUAL_BUDGET_FIT_PAIR',
        gt_read=False,external_test_read=False,held_scored=False,held_used_for_selection=False,
        pair=str(pair),same_initial_rho_loge_adam_rng=True,same_terminal_frame_rng=True,
        sole_pair_difference='fixed_directions_old_vs_refined',epochs_each=30,updates_each=3510,
        cumulative_epochs=120,records=records,relative_refined_benefit=benefit,
        eligible_for_development_validation=passed,training_asset_exported=False,
        direction_identifiability='unresolved',input_sha256=hashes)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',type=Path,required=True);a=p.parse_args()
    out=a.pair/'fit_pair_receipt.json'
    if out.exists():raise FileExistsError('已存在成对结论，禁止覆盖。')
    result=inspect_pair(a.pair);v.write(out,result)
    print({k:value for k,value in result.items() if k!='input_sha256'})


if __name__=='__main__':main()
