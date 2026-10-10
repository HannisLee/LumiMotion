"""来源全135 TRAIN姿态的10度旋转响应只读原型；不改生产、不存teacher。"""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch
import torch.nn.functional as F
import numpy as np
from plyfile import PlyData
from types import SimpleNamespace
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_geometry import quaternion_normal
from scene.pbr_plane_projection import align_quaternion_normal
from scene.pbr_feature_gate import native_probability
from utils.general_utils import build_rotation

def multiply(a,b):
    w=a[:,:1]*b[:,:1]-(a[:,1:]*b[:,1:]).sum(-1,keepdim=True)
    vector=a[:,:1]*b[:,1:]+b[:,:1]*a[:,1:]+torch.cross(a[:,1:],b[:,1:],dim=-1)
    return torch.cat([w,vector],-1)

def transport(source_world,source_canonical,candidate):
    source_unit=F.normalize(source_canonical,dim=-1)
    source_inverse=source_unit*source_unit.new_tensor([1.,-1.,-1.,-1.])
    relative=multiply(source_inverse,F.normalize(candidate,dim=-1))
    return multiply(F.normalize(source_world,dim=-1),relative)*source_world.norm(dim=-1,keepdim=True)

def degrees(a,b):
    a=F.normalize(a.double(),dim=-1);b=F.normalize(b.double(),dim=-1)
    return torch.rad2deg(torch.acos((a*b).sum(-1).abs().clamp(0,1)))

def stats(x):
    x=x.double()
    return dict(mean=float(x.mean()),median=float(x.median()),p90=float(torch.quantile(x,.9)),
        p99=float(torch.quantile(x,.99)),maximum=float(x.max())) if len(x) else None

@torch.no_grad()
def main():
    run=Stage2RenderRun(ROOT/'model_mlp',40000);ctx=run.context
    source_path=ROOT/'model_mlp/point_cloud/iteration_35000/point_cloud.ply'
    v=PlyData.read(source_path)['vertex'].data
    xyz=torch.tensor(np.column_stack([v[k] for k in ['x','y','z']]),device='cuda')
    q=torch.tensor(np.column_stack([v['rot_%d'%i] for i in range(4)]),device='cuda')
    raw=torch.tensor(v['fea_0'].copy(),device='cuda')
    gate=native_probability(raw)[:,None]
    opacity=torch.sigmoid(torch.tensor(v['opacity'].copy(),device='cuda'))
    assert len(q)==99827 and (q.norm(dim=-1)>1e-8).all()
    with np.load(ROOT/'model_mlp/pbr_plane_projection.npz') as data:
        arrays={k:torch.tensor(data[k].copy(),device='cuda') for k in data.files}
    plane,offset=arrays['plane'],arrays['offset']
    dynamic=raw>=0
    near=(dynamic&(opacity>.05)&((xyz@plane+offset).abs()<=.2)&
          (arrays['max_plane_distance']<=.2)&((quaternion_normal(q)@plane).abs()>=.5)&
          (arrays['min_axis_cosine']>=.5))
    groups=dict(source_projected=arrays['support'],source_dynamic_near_plane=near,
        source_other_dynamic=dynamic&~near,source_static_unprojected=~dynamic&~arrays['support'])
    assert torch.equal(sum(mask.int() for mask in groups.values()),torch.ones_like(raw,dtype=torch.int32))
    axis=quaternion_normal(q)
    basis=F.one_hot(axis.abs().argmin(-1),num_classes=3).to(axis)
    tangent=F.normalize(torch.cross(axis,basis,dim=-1),dim=-1)
    theta=axis.new_tensor(np.deg2rad(10.))
    desired=axis*theta.cos()+tangent*theta.sin()
    candidate=align_quaternion_normal(q,desired)
    canonical_angle=degrees(quaternion_normal(q.double()),quaternion_normal(candidate.double()))
    assert (canonical_angle-10).abs().max()<.001
    original=SimpleNamespace(get_xyz=xyz,_xyz=xyz,_rotation=q)
    proposed=SimpleNamespace(get_xyz=xyz,_xyz=xyz,_rotation=candidate)
    train_meta=json.loads((Path(run.args.source_path)/'transforms_train.json').read_text())
    keys=[str(int(frame['source_frame'])).zfill(4) for frame in train_meta['frames']]
    assert len(keys)==135 and all(int(k)%10!=0 for k in keys)
    values={name:torch.zeros(len(q),device='cuda',dtype=torch.float64)
        for name in ['additive_sum','additive_max','local_sum','local_max','local_transport_error_max']}
    per_frame=[];initial_exact=True
    for key in keys:
        cam=ctx.camera_by_key[key];ctx.load_camera(cam)
        raw_d=ctx.deform.step(xyz,cam.fid.unsqueeze(0).expand(len(q),-1),iteration=35000,
            feature=gate,camera_center=cam.camera_center)
        source_d=ctx.pbr_plane_projection.apply(original,raw_d)
        additive_d=ctx.pbr_plane_projection.apply(proposed,raw_d)
        world=q+raw_d['d_rotation']
        baseline_transport=transport(world,q,q)
        candidate_transport=transport(world,q,candidate)
        # 来源自身为初值时逐bit返回原生raw additive dr；仅RAM原型。
        initial_d=raw_d['d_rotation']+(q-q)+(baseline_transport-baseline_transport)
        initial_exact=initial_exact and torch.equal(initial_d,raw_d['d_rotation'])
        local_d=dict(raw_d)
        local_d['d_rotation']=raw_d['d_rotation']+(q-candidate)+(candidate_transport-baseline_transport)
        local_d=ctx.pbr_plane_projection.apply(proposed,local_d)
        source_axis=quaternion_normal((q+source_d['d_rotation']).double())
        additive_axis=quaternion_normal((candidate+additive_d['d_rotation']).double())
        local_axis=quaternion_normal((candidate+local_d['d_rotation']).double())
        additive_angle=degrees(source_axis,additive_axis)
        local_angle=degrees(source_axis,local_axis)
        matrix_prediction=torch.einsum('nij,nj->ni',build_rotation(world).double(),
            torch.einsum('nij,nj->ni',build_rotation(q).double().transpose(1,2),
                        quaternion_normal(candidate.double())))
        error=degrees(local_axis,matrix_prediction)
        assert torch.isfinite(additive_angle).all() and torch.isfinite(local_angle).all()
        values['additive_sum']+=additive_angle;values['local_sum']+=local_angle
        for name,item in [('additive_max',additive_angle),('local_max',local_angle),('local_transport_error_max',error)]:
            values[name]=torch.maximum(values[name],item)
        nonplane=~arrays['support']
        per_frame.append(dict(frame=key,nonplane_points=int(nonplane.sum()),
            additive_response_mean_deg=float(additive_angle[nonplane].mean()),
            local_response_mean_deg=float(local_angle[nonplane].mean()),
            local_vs_matrix_transport_max_deg=float(error[nonplane].max())))
        ctx.unload_camera(cam)
    assert initial_exact
    result_groups={}
    for label,mask in dict(all=torch.ones_like(dynamic),**groups).items():
        result_groups[label]=dict(points=int(mask.sum()),canonical_angle_deg=stats(canonical_angle[mask]),
            mean_time_additive_axis_response_deg=stats((values['additive_sum']/135)[mask]),
            worst_time_additive_axis_response_deg=stats(values['additive_max'][mask]),
            mean_time_local_axis_response_deg=stats((values['local_sum']/135)[mask]),
            worst_time_local_axis_response_deg=stats(values['local_max'][mask]),
            local_vs_matrix_transport_max_deg=stats(values['local_transport_error_max'][mask&~arrays['support']]))
    ctx.verify_frozen()
    manifest=json.loads((ROOT/'code_manifest.json').read_text());assert len(manifest)==474
    assert all(hashlib.sha256((CWD/k).read_bytes()).hexdigest()==sha for k,sha in manifest.items())
    result=dict(status='PASS_READ_ONLY_SOURCE_LOCAL_ROTATION_RESPONSE_PROTOTYPE',frames=135,points=99827,
        groups=result_groups,rows=per_frame,source_iteration=35000,camera_NN_container=40000,
        learned_40000_geometry_material_feature_used=False,source_MLP_32_tensors_frozen=True,
        initial_raw_deformation_bit_exact=initial_exact,canonical_probe_degrees=10,
        optimizer_steps=0,GT_labels_read=False,training_prior_saved=False,scoring_mask_saved=False,
        production_pipeline_changed=False,new_Preset_implemented=False,
        source_PLY_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        created_at=datetime.datetime.now().astimezone().isoformat(),
        limitations='全135 source预测姿态的人工10度扰动只读数学原型；source groups不是GT语义。投影支持的法线被既有plane固定，不期望它响应；未删任何点/评分像素。局部四元数运输与旧additive的参数响应比较不是GT normal误差，不是已支持训练或达7度。候选四元数/normal只在RAM，不导出监督/teacher/prior。仅NN/camera容器加载40k，源35k参数用于全部诊断，当前B09训练不变。')
    path=ROOT/'inspection/source_local_rotation_response.json';assert not path.exists()
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
