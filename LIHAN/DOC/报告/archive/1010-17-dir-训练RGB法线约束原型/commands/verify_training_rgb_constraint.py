"""独立CPU复算原TRAIN RGB候选数组/来源SHA；不读GT标签。"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from plyfile import PlyData
ROOT=Path(__file__).resolve().parent.parent
META=ROOT/'candidate_assets/training_rgb_normal_constraint.json'
meta=json.loads(META.read_text())
asset=ROOT/'candidate_assets/training_rgb_normal_constraint.npz'
assert hashlib.sha256(asset.read_bytes()).hexdigest()==meta['asset_sha256']
assert meta['TRAIN_frames']==135 and meta['TEST_frames_used']==0 and meta['points']==99827
assert meta['optimizer_steps']==0 and not meta['GT_labels_read'] and meta['native_feature_probability_exact']
assert not meta['production_pipeline_changed'] and not meta['learned_40000_geometry_material_feature_used']
assert len(meta['training_frame_keys'])==135 and all(int(k)%10!=0 for k in meta['training_frame_keys'])
for label,item in meta['source_manifest'].items():
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],label
assert set(meta['TRAIN_RGBA_manifest'])==set(meta['training_frame_keys'])
for key,item in meta['TRAIN_RGBA_manifest'].items():
    assert int(key)%10!=0
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],key
with np.load(asset) as data:
    a={k:data[k].copy() for k in data.files}
n=99827
assert a['canonical_axis'].shape==(n,3) and a['confidence'].shape==(n,)
assert a['gram'].shape==(n,3,3) and a['rhs'].shape==(n,3)
assert all(len(v)==n and np.isfinite(v).all() for v in a.values())
assert np.isin(a['confidence'],[0,1]).all() and (a['observations']>=0).all() and (a['observations']<=135).all()
assert np.max(np.abs(np.linalg.norm(a['canonical_axis'],axis=-1)-1))<2e-6
gram=a['gram'].astype(np.float64);rhs=a['rhs'].astype(np.float64)
eigen=np.linalg.eigvalsh(gram)
vector=np.linalg.solve(gram+np.eye(3)[None]*1e-5,rhs[...,None])[...,0]
norm=np.linalg.norm(vector,axis=-1)
residual=np.maximum(0,a['y_squared'].astype(np.float64)-2*np.sum(vector*rhs,axis=-1)+np.einsum('ni,nij,nj->n',vector,gram,vector))
relative=np.sqrt(residual/np.maximum(a['y_squared'],1e-8))
conditioned=(a['observations']>=12)&(eigen[:,0]>.01*np.maximum(eigen[:,2],1e-8))
reliable=conditioned&(norm>.01)&(relative<.3)
with np.load(meta['source_manifest']['plane_arrays']['path']) as data:support=data['support'].astype(bool)
expected=reliable&~support
disagreement=np.flatnonzero(expected!=(a['confidence']>0))
assert len(disagreement)==0,disagreement[:20]
vertex=PlyData.read(meta['source_manifest']['PLY']['path'])['vertex'].data
q=np.column_stack([vertex[f'rot_{i}'] for i in range(4)]).astype(np.float64)
q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=q.T
source_axis=np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],axis=-1)
source_error=float(np.max(np.abs(source_axis-a['source_canonical_axis'])))
assert source_error<2e-6
target=vector/np.maximum(norm[:,None],1e-12)
target*=np.where(np.sum(target*source_axis,axis=-1,keepdims=True)<0,-1.,1.)
target=np.where(expected[:,None],target,source_axis)
axis_error=float(np.max(np.abs(target-a['canonical_axis'])))
assert axis_error<2e-4
selected_angle=np.rad2deg(np.arccos(np.clip(np.sum(target[expected]*a['canonical_axis'][expected],axis=-1),-1,1)))
assert int(expected.sum())==meta['confidence_points'] and not (expected&support).any()
record=dict(status='PASS_SOURCE_TRAIN_RGB_ARRAYS_AND_PROVENANCE_CPU',points=n,confidence_points=int(expected.sum()),
    confidence_disagreement_points=len(disagreement),source_axis_max_abs_error=source_error,
    CPU_vs_GPU_candidate_axis_max_abs_error=axis_error,CPU_vs_GPU_angle_max_deg=float(selected_angle.max()),
    source_files_SHA_verified=len(meta['source_manifest']),TRAIN_RGBA_SHA_verified=135,TEST_frames_used=0,
    GT_labels_read=False,optimizer_steps=0,production_pipeline_changed=False,
    created_at=datetime.datetime.now().astimezone().isoformat(),
    limitations='只复算训练RGB派生候选和来源，不证明GGX/阴影模型合适、GT方向准确率或7度。全部点保留。')
p=ROOT/'inspection/constraint_independent_audit.json';assert not p.exists()
p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
