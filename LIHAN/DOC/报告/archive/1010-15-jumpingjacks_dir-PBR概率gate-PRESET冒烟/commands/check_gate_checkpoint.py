"""保存35020真实概率gate与source32NN/原source资产/边界/共享入口核验。"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from plyfile import PlyData
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parents[2]
sys.path.insert(0,str(CWD))
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_feature_gate import native_probability
run=Stage2RenderRun(ROOT/'model_mlp',35020);ctx=run.context
cfg=vars(run.args);assert cfg['pbr_optimize_feature_gate'] and cfg['pbr_feature_gate_parameterization']=='probability'
assert json.loads((ROOT/'training_runtime.json').read_text())['exit_code']==0
src=ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
trained=ROOT/'model_mlp/point_cloud/iteration_35020/point_cloud.ply'
a=PlyData.read(src)['vertex'].data;b=PlyData.read(trained)['vertex'].data
assert len(a)==len(b)==99827 and all(np.isfinite(b[k]).all() for k in b.dtype.names)
frozen=[k for k in a.dtype.names if k=='opacity' or k.startswith('scale_')]
assert all(np.array_equal(a[k],b[k]) for k in frozen)
initial=native_probability(torch.tensor(a['fea_0'].copy()))
actual=native_probability(torch.tensor(b['fea_0'].copy()))
delta=(actual-initial).abs();raw_delta=np.abs(b['fea_0']-a['fea_0'])
assert (raw_delta>0).any() and (delta>0).any()
assert float(delta.max())<=cfg['pbr_feature_gate_max_probability_delta']+1e-6
source=torch.load(ROOT/'model_mlp/deform/iteration_35000/deform.pth',map_location='cpu')
current=ctx.deform.deform.state_dict()
assert len(source)==32 and set(current)==set(source)
assert all(torch.equal(source[k],current[k].cpu()) for k in source)
assert not any(p.requires_grad for p in ctx.deform.deform.parameters())
record=json.loads((ROOT/'model_mlp/pbr_geometry_source.json').read_text())
assert 'feature' in record['optimizer_groups'] and 'opacity' not in record['optimizer_groups']
assert ctx.pbr_feature_gate_provenance['format_version']==2
assert ctx.pbr_feature_gate_provenance['source_MLP_frozen']
ctx.verify_frozen()
manifest=json.loads((ROOT/'code_manifest.json').read_text())
assert manifest==json.loads((ROOT/'code_manifest_train_launch.json').read_text()) and len(manifest)==474
for k,sha in manifest.items():
    for p in [CWD/k,ROOT/'commands/code_snapshot'/k,ROOT/'commands/code_snapshot_train_launch'/k]:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
report=dict(status='PASS_SAVED_NATIVE_PROBABILITY_GATE',iteration=35020,points=len(a),all_finite=True,
    opacity_scale_frozen=True,source_MLP_frozen=True,source_NN_tensors_equal=32,
    feature_optimizer_probability_group=True,source_native_probability_initialization=True,
    raw_feature_changed_points=int((raw_delta>0).sum()),native_probability_changed_points=int((delta>0).sum()),
    max_source_probability_delta=float(delta.max()),max_source_raw_logit_delta=float(raw_delta.max()),
    core_manifest_sha=True,GT_material_normal_geometry_photometry_inputs=False,full_MAE=False,
    normal_7_goal='NOT_EVALUATED',source_files=ctx.source_files)
p=ROOT/'inspection/gate_checkpoint.json';assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
