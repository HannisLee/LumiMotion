"""CPU核查Stage2真实checkpoint、固定GT光照、优化范围与来源保护。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--iteration',type=int,default=55000);a=p.parse_args()
pre=json.loads((ROOT/'inspection/preflight.json').read_text())
path=ROOT/'model_mlp/stage2_photometric'/f'iteration_{a.iteration}/checkpoint.pth';s=torch.load(path,map_location='cpu')
assert s['format_version']==1 and s['iteration']==a.iteration and s['source_iteration']==35000 and 'geometry' not in s
assert s['source_files']==pre['stage1_files']
for file,digest in s['source_files'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
initial=torch.load(ROOT/'model_mlp/stage2_photometric/iteration_35000/checkpoint.pth',map_location='cpu')
assert initial['frozen_fingerprint']==s['frozen_fingerprint']
assert all(torch.equal(initial['light'][k],s['light'][k]) for k in initial['light'])
assert len(s['optimizer']['param_groups'])==1 and s['optimizer']['param_groups'][0]['name']=='albedo'
assert s['config']['render_mode']=='photometric_lambertian' and s['config']['loss_preset']=='lambertian_gt'
assert s['config']['photometric_light_mode']=='gt_directional' and s['config']['photometric_light_intensity']>0
cal=json.loads((ROOT/'model_mlp/calibration.json').read_text())
assert len(cal['train_keys'])==135 and set(map(int,cal['train_keys'])).isdisjoint(set(range(10,151,10)))
assert cal['intensity']==s['config']['photometric_light_intensity']
gt=json.loads((ROOT/'gt_directions.json').read_text());expected=torch.tensor([gt[f'{i:04d}']['light_dir_world'] for i in range(1,151)])
expected=torch.nn.functional.normalize(expected,dim=-1)
assert torch.allclose(s['light']['raw_directions'],expected,atol=1e-7,rtol=1e-6)
tensors=[]
def collect(x):
 if torch.is_tensor(x):tensors.append(x)
 elif isinstance(x,dict):
  for v in x.values():collect(v)
 elif isinstance(x,(list,tuple)):
  for v in x:collect(v)
collect(s);assert all(torch.isfinite(x).all() for x in tensors)
result=dict(status='PASS',iteration=a.iteration,source_iteration=35000,gaussians=pre['gaussians'],finite_tensors=len(tensors),source_hashes_unchanged=True,frozen_fingerprint_unchanged=True,light_state_unchanged=True,optimizer_groups=['albedo'],intensity=cal['intensity'],calibration_train_frames=135,gt_direction_max_difference=float((s['light']['raw_directions']-expected).abs().max()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(ROOT/'inspection'/f'checkpoint_{a.iteration}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
