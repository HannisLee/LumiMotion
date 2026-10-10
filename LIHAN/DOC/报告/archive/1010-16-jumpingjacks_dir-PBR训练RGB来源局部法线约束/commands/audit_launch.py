"""当前真实B10启动只读审计；只检查输入/配置/来源，不优化/读GT标签。"""
from pathlib import Path
import os,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
old=ROOT.parent/'1010-15-jumpingjacks_dir-PBR概率gate优化'
model=ROOT/'model_mlp'
manifest=json.loads((ROOT/'code_manifest.json').read_text());assert len(manifest)==476
for rel,sha in manifest.items():
 for p in [CWD/rel,ROOT/'commands/code_snapshot'/rel,ROOT/'commands/code_snapshot_train_launch'/rel]:
  assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
cfg=json.loads((model/'stage2_config.json').read_text());before=json.loads((old/'model_mlp/stage2_config.json').read_text())
for key,value in {'pbr_source_local_rotation':False,'pbr_rgb_normal_prior_path':'','lambda_pbr_rgb_normal_prior':0.0}.items():before.setdefault(key,value)
diff={k:dict(old=before.get(k),new=cfg.get(k)) for k in set(before)|set(cfg) if before.get(k)!=cfg.get(k)}
expected={'source_path','model_path','loss_preset','pbr_gt_lights_path','pbr_source_local_rotation','pbr_rgb_normal_prior_path','lambda_pbr_rgb_normal_prior'}
assert set(diff)==expected,diff
assert cfg['pbr_source_local_rotation'] and cfg['lambda_pbr_rgb_normal_prior']==2.0
assert cfg['pbr_optimize_feature_gate'] and cfg['pbr_geometry_warmup']==2000
gate=json.loads((model/'pbr_feature_gate.json').read_text());assert gate['source_iteration']==35000
assert gate['source_MLP_frozen'] and gate['max_logit_delta'] is None and gate['format_version']==2 and gate['max_probability_delta']==1 and cfg['pbr_feature_gate_parameterization']=='probability' and gate['learning_rate']==.001 and gate['anchor_weight']==.001
assert not any(gate[k] for k in ['gt_normal_inputs','gt_geometry_inputs','gt_material_inputs','gt_light_photometry_inputs'])
src=json.loads((model/'stage1_source.json').read_text());assert gate['source_files']==src
for p,sha in src.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
for rel,sha in gate['source_snapshots'].items():assert hashlib.sha256((model/rel).read_bytes()).hexdigest()==sha
assert (ROOT/'gt_directions_only.json').read_bytes()==(old/'gt_directions_only.json').read_bytes()
with np.load(model/'pbr_plane_projection.npz') as a,np.load(old/'model_mlp/pbr_plane_projection.npz') as b:
 assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
 assert int(a['support'].sum())==15187
prov=json.loads((model/'rgb_alpha_provenance.json').read_text())
assert prov['status']=='PASS' and prov['frames']==150 and prov['mask_source']=='original-image-alpha'
geo=json.loads((model/'pbr_geometry_source.json').read_text());assert 'feature' in geo['optimizer_groups']
assert 'opacity' not in geo['optimizer_groups'] and 'scaling' not in geo['optimizer_groups']
handles=json.loads((ROOT/'process_handles.json').read_text());train=handles['train'];proc=Path('/proc')/str(train['pid'])
assert (proc/'stat').read_text().split()[21]==train['start_ticks']
assert (proc/'cmdline').read_bytes().split(b'\x00')[:-1]==[x.encode() for x in train['argv']]
assert os.readlink(proc/'cwd')==str(CWD)
assert 'before=592,957 | after=0' in (ROOT/'logs/train.log').read_text() or 'after=0' in (ROOT/'logs/train.log').read_text()
rgb=json.loads((model/'pbr_rgb_normal.json').read_text())
assert rgb['confidence_points']==29621 and rgb['constraint_weight']==2.0 and rgb['source_MLP_frozen']
assert all(rgb[k] is False for k in ['GT_normal_inputs','GT_material_inputs','GT_geometry_inputs','GT_light_photometry_inputs'])
for key,filename in [('asset_sha256','pbr_rgb_normal.npz'),('input_asset_sha256','pbr_rgb_normal_input.npz'),('input_metadata_sha256','pbr_rgb_normal_input.json')]:assert hashlib.sha256((model/filename).read_bytes()).hexdigest()==rgb[key]
result=dict(rgb_normal=rgb,status='PASS_LIVE_LAUNCH_ONLY',train_pid=train['pid'],start_ticks=train['start_ticks'],core_SHA_files=len(manifest),
 config_differences=diff,source_plane_arrays_bit_identical_to_B09=True,source_plane_support=15187,
 source_MLP_optimizer_params=0,feature_optimizer_enabled=True,source_files=src,own_RGBA_alpha_150='PASS',
 direction_table_byte_identical=True,optimizer_steps_diagnostic=0,GT_labels_read=False,final_MAE=False)
(ROOT/'inspection/launch_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
