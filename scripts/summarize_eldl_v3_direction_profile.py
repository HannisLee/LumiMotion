"""冻结阈值汇总TRAIN方向反证，不选新方向，不写训练资产。"""
import argparse
import hashlib
import json
import math
import torch
from pathlib import Path
CASES=['center','u_minus','u_plus','v_minus','v_plus']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);a=p.parse_args()
 base=a.profile.resolve();cal=a.calibration.resolve();selection=read(cal/'endpoint_selection.json');assert selection['status']=='FIT_BUDGET_PASS' and not selection['held_used']
 selected=selection['selected'];expected=cal/selected/'checkpoint_epoch_30.pt';expected_hash=sha(expected)
 rows={};hashes={str(cal/'endpoint_selection.json'):sha(cal/'endpoint_selection.json'),str(expected):expected_hash}
 for case in CASES:
  out=base/case;cfg=read(out/'configuration.json');state=read(base/'runs'/case/'state.json');result=read(out/'result.json')
  assert state['exit_code']==0 and cfg['case']==case and cfg['checkpoint_sha256']==expected_hash
  assert cfg['epochs']==20 and cfg['updates']==2340 and cfg['gt_read'] is False and cfg['held_used'] is False
  ck=torch.load(out/'checkpoint_epoch_20.pt',map_location='cpu',weights_only=False)
  assert ck['epoch']==20 and all(abs(g['lr']-.0001)<1e-12 for g in ck['optimizer']['param_groups'])
  assert all(int(item['step'])==12870 for item in ck['optimizer']['state'].values())
  assert all(torch.isfinite(ck[key]).all() for key in ('rho','environment','directions'))
  assert result['geometry_bitexact_unchanged'] and result['status']=='PROFILE_COMPLETE_NOT_LIGHT_ACCEPTANCE'
  assert len(result['fit']['per_frame'])==117 and [r['frame'] for r in result['fit']['per_frame']]==[i for i in range(1,151) if i%10 and not 61<=i<=79]
  score=float(result['fit']['frame_mean']);assert math.isfinite(score)
  rows[case]=dict(fit=score,time_blocks=result['fit']['time_blocks'],ambient_rgb=result['ambient_rgb'],albedo_bound_fraction=result['albedo_bound_fraction'])
  for path in (out/'configuration.json',out/'result.json',out/'checkpoint_epoch_20.pt',base/'runs'/case/'plan.json',base/'runs'/case/'state.json'):hashes[str(path)]=sha(path)
 original=[float(read(cal/f'init_{i}'/'fit_epoch_30.json')['frame_mean']) for i in (1,2,3)]
 original_selected=float(read(cal/selected/'fit_epoch_30.json')['frame_mean']);control=rows['center']['fit']
 spread=max(original)-min(original);center_change=abs(control-original_selected);epsilon=max(spread,center_change)
 differences={case:rows[case]['fit']-control for case in CASES[1:]}
 refuted=any(value < -epsilon for value in differences.values())
 support=all(value>epsilon for value in differences.values())
 status='REFUTED_BY_FIXED_LOCAL_ALTERNATIVE' if refuted else ('FINITE_LOCAL_SUPPORT_NO_UNIQUENESS_CLAIM' if support else 'UNRESOLVED_WITHOUT_LOCAL_REFUTATION')
 summary=dict(status=status,gt_read=False,held_used=False,rows=rows,original_selected_fit=original_selected,three_start_fit_range=spread,
  center_additional_fit_change=center_change,epsilon=epsilon,alternative_minus_center=differences,
  eligible_for_exploratory_geometry=not refuted,direction_identifiability='unresolved',
  limitation='20epoch低学习率仅有限局部nuisance重标定，不保证各方向全局或材质最优，不把此结果称GT灯向精确恢复。',
  selected_checkpoint=str(expected),selected_checkpoint_sha256=expected_hash,input_sha256=hashes,internal_validation_passed=False)
 target=base/'summary.json';assert not target.exists();target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
