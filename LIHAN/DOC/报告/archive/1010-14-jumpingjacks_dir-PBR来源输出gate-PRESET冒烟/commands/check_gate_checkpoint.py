from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from plyfile import PlyData
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parents[2]
sys.path.insert(0,str(CWD))
from scripts.render_stage2 import Stage2RenderRun
run=Stage2RenderRun(ROOT/'model_mlp',35020);ctx=run.context
cfg=vars(run.args);assert cfg['pbr_optimize_feature_gate']
src=ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
trained=ROOT/'model_mlp/point_cloud/iteration_35020/point_cloud.ply'
a=PlyData.read(src)['vertex'].data;b=PlyData.read(trained)['vertex'].data
assert len(a)==len(b)==99827 and all(np.isfinite(b[k]).all() for k in b.dtype.names)
frozen=[k for k in a.dtype.names if k=='opacity' or k.startswith('scale_')]
assert all(np.array_equal(a[k],b[k]) for k in frozen)
delta=np.abs(b['fea_0']-a['fea_0']);changed=delta>0
assert changed.any() and delta.max()<=cfg['pbr_feature_gate_max_logit_delta']+1e-5
source=torch.load(ROOT/'model_mlp/deform/iteration_35000/deform.pth',map_location='cpu')
actual=ctx.deform.deform.state_dict()
assert len(source)==32 and set(actual)==set(source)
assert all(torch.equal(source[k],actual[k].cpu()) for k in source)
assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
record=json.loads((ROOT/'model_mlp/pbr_geometry_source.json').read_text())
assert 'feature' in record['optimizer_groups'] and 'opacity' not in record['optimizer_groups']
assert 'scale' not in record['optimizer_groups']
for camera in ctx.train_cameras[:3]:
 d=ctx.deformation(camera);assert not d['d_xyz'].requires_grad
assert ctx.pbr_feature_gate_provenance['source_MLP_frozen']
ctx.verify_frozen()
manifest=json.loads((ROOT/'code_manifest.json').read_text())
assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
old=json.loads((ROOT/'code_manifest_train_launch.json').read_text())
diff=[k for k in old if old[k]!=manifest[k]]
assert diff==['tests/test_geometry_eval_source_runtime.py']
for k,v in old.items():assert hashlib.sha256((ROOT/'commands/code_snapshot_train_launch'/k).read_bytes()).hexdigest()==v
report=dict(status='PASS_SAVED_NATIVE_GATE',iteration=35020,points=len(a),all_finite=True,
 opacity_scale_frozen=True,feature_optimizer_group=True,source_feature_initialization=True,
 feature_changed_points=int(changed.sum()),max_source_logit_delta=float(delta.max()),
 source_MLP_frozen=True,source_NN_tensors_equal=32,source_files_unchanged=True,
 core_manifest_sha=True,post_training_fixture_only_changes=diff,optimizer_steps_diagnostic=0,
 GT_material_normal_geometry_photometry_inputs=False,full_MAE=False,normal_7_goal='NOT_EVALUATED')
(ROOT/'inspection/gate_checkpoint.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
