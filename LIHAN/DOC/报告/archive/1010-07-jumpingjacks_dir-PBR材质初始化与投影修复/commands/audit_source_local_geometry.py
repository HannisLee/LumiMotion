"""来源35k点云局部几何诊断；不创建训练先验、不使用GT、不改变模型。"""
from pathlib import Path
import json, hashlib, sys
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normals(v):
    q=np.column_stack([v['rot_'+str(i)] for i in range(4)]).astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    w,x,y,z=q.T
    return np.column_stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
def stats(a):
    if not len(a):return {'count':0}
    return dict(count=len(a),mean=float(a.mean()),quantiles=dict(zip(['min','p50','p90','p99','max'],np.quantile(a,[0,.5,.9,.99,1]).tolist())))

source=Path(json.loads((ROOT/'model_mlp/pbr_geometry_source.json').read_text())['source_model'])
ply=source/'point_cloud/iteration_35000/point_cloud.ply'
before=sha(ply)
v=PlyData.read(ply)['vertex'].data
x=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)
n=normals(v)
feature=1/(1+np.exp(-2*v['fea_0'].astype(np.float64)))
opacity=1/(1+np.exp(-v['opacity'].astype(np.float64)))
scale=np.exp(np.column_stack([v[k] for k in ['scale_0','scale_1']]).astype(np.float64)).max(1)
rows=[]
for label,mask in [('static_feature_lt_half',feature<.5),('dynamic_feature_ge_half',feature>=.5)]:
    ids=np.flatnonzero(mask)
    distance,index=cKDTree(x[ids]).query(x[ids],k=33,workers=2)
    index=ids[index[:,1:]];distance=distance[:,1:]
    delta=x[index]-x[ids,None,:]
    centered=delta-delta.mean(1,keepdims=True)
    cov=np.einsum('nki,nkj->nij',centered,centered)/32
    val,vec=np.linalg.eigh(cov)
    axis=vec[:,:,0]
    ratio=val[:,0]/np.maximum(val[:,1],1e-12)
    valid=(opacity[ids]>.2)&(ratio<.2)&(distance[:,-1]<=scale[ids]*3)
    deviation=np.degrees(np.arccos(np.clip(np.abs((n[ids]*axis).sum(1)),0,1)))
    row=dict(group=label,points=len(ids),valid_local_plane_points=int(valid.sum()),
             valid_fraction=float(valid.mean()),source_axis_to_local_PCA_deg=stats(deviation[valid]),
             local_eigenvalue_ratio=stats(ratio[valid]),radius_over_scale=stats(distance[valid,-1]/scale[ids][valid]))
    old=CWD/'output/1010-05-jumpingjacks_dir-PBR原始RGBA严格输入/model_mlp/point_cloud/iteration_55000/point_cloud.ply'
    ov=PlyData.read(old)['vertex'].data
    row['B02_axis_to_same_source_local_PCA_deg']=stats(np.degrees(np.arccos(np.clip(np.abs((normals(ov)[ids]*axis).sum(1)),0,1)))[valid])
    rows.append(row)
assert sha(ply)==before
report=dict(status='PASS_READ_ONLY_DIAGNOSTIC',method='source_local_32_neighbours_PCA_by_source_soft_feature_group',
            source=str(ply),source_sha256=before,rows=rows,gt_inputs=False,optimizer_steps=0,
            generated_training_target=False,current_training_modified=False,final_mae=False,
            limitations='PCA一致性仅为来源点云局部诊断，不能当真实法线正确性或7°指标；可能受体积点/边界/混合表面影响。')
(ROOT/'inspection/source_local_geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
