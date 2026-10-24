"""CPU审计父预测TRAIN静态几何平面可行性，不读取任何几何GT或heldout观测。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    return dict(count=int(len(x)),mean=float(np.mean(x)),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(np.max(x)))
def plane(x):
    center=x.mean(0);_,_,v=np.linalg.svd(x-center,full_matrices=False)
    return center,v[-1]
def angles(normals,reference):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(normals,reference),axis=-1),np.abs(normals@reference)))


def main(args):
    torch.set_num_threads(4);cache=args.cache.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    manifest=json.loads((cache.parent/'manifest.json').read_text())
    assert manifest['gt_read'] is False and manifest['source_split']=='TRAIN' and sha(cache)==manifest['observation_sha256']
    data=torch.load(cache,map_location='cpu',weights_only=False)
    v=PlyData.read(str(args.parent/'point_cloud/iteration_35000/point_cloud.ply'))['vertex']
    ids=data['ids'].numpy();xyz=data['canonical_xyz'].numpy().astype(np.float64)
    parent_xyz=np.stack([v[k] for k in ['x','y','z']],-1)
    assert np.array_equal(parent_xyz[ids],data['canonical_xyz'].numpy())
    gate=1/(1+np.exp(np.clip(-2*np.asarray(v['fea_0'])[ids],-80,80)))
    assert np.array_equal(gate>.5,data['dynamic'].numpy())
    position=data['positions'].numpy().astype(np.float64);valid=data['valid'].numpy();support=valid.sum(0)>=10
    extent=float(np.linalg.norm(xyz[support].max(0)-xyz[support].min(0)))
    motion=np.sqrt(((position-position.mean(0))**2).sum(-1).mean(0))
    static=gate<.1;stable=motion<extent*.01;eligible=static&stable&support
    cloud=xyz[eligible];rng=np.random.default_rng(1024)
    triples=cloud[rng.integers(len(cloud),size=(512,3))]
    normals=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=-1,keepdims=True),1e-12)
    distances=np.abs(cloud@normals.T-np.sum(triples[:,0]*normals,axis=-1))
    best=np.argmax((distances<extent*.005).sum(0));mask=distances[:,best]<extent*.005
    origin,normal=plane(cloud[mask]);inlier=eligible&(np.abs((xyz-origin)@normal)<extent*.005)
    origin,normal=plane(xyz[inlier]);bootstrap=[]
    for _ in range(100):
        _,n=plane(xyz[inlier][rng.integers(inlier.sum(),size=inlier.sum())]);bootstrap.append(n)
    temporal=[]
    for frame_positions in position:
        _,n=plane(frame_positions[inlier]);temporal.append(n)
    predicted_normal=data['world_normals'].numpy().astype(np.float64)[:,inlier]
    predicted_normal/=np.maximum(np.linalg.norm(predicted_normal,axis=-1,keepdims=True),1e-12)
    result=dict(status='PASS_PREDICTED_PLANE_FEASIBILITY_ONLY',gt_read=False,heldout_read=False,
        source_sha256={str(cache):sha(cache),str(args.parent/'point_cloud/iteration_35000/point_cloud.ply'):sha(args.parent/'point_cloud/iteration_35000/point_cloud.ply')},
        frames=[x['frame'] for x in data['frames']],sampled_points=len(xyz),extent=extent,
        thresholds=dict(static_gate_lt=.1,motion_lt=extent*.01,plane_inlier_distance_lt=extent*.005,min_train_observations=10),
        counts=dict(static=int(static.sum()),stable=int(stable.sum()),train_supported=int(support.sum()),intersection=int(eligible.sum()),plane_inliers=int(inlier.sum())),
        intersection_fraction_of_static=float(eligible.sum()/max(1,static.sum())),
        plane_inlier_fraction_of_intersection=float(inlier.sum()/max(1,eligible.sum())),
        plane_origin=origin.tolist(),plane_normal_unsigned=normal.tolist(),
        canonical_plane_residual=stats(np.abs((xyz[inlier]-origin)@normal)),
        all_train_plane_residual=stats(np.abs((position[:,inlier]-origin)@normal).ravel()),
        bootstrap_direction_degrees=stats(angles(np.asarray(bootstrap),normal)),
        train_time_direction_degrees=stats(angles(np.asarray(temporal),normal)),
        predicted_gs_to_plane_degrees=stats(angles(predicted_normal.reshape(-1,3),normal)),
        limitations=['仅6584个TRAIN观测采样点，不能等同全Gaussian支持率','平面来自父预测，不证明与真实平台准确对齐','静态gate与低运动不能完全排除鞋足误纳','只审计可行性，未施加先验或修改训练'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    report='# 父预测静态平面CPU可行性审计\n\n不读取GT几何、DEV或TEST；复用原投影估光已登记的TRAIN预测轨迹缓存，先验证SHA与父PLY点坐标、门控逐元素一致。无训练、无GPU。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n结论只支持父预测静态平台具有可拟合一致平面的工程假设，不代表真实几何正确；若以后采用需独立DEV验证人物/鞋足和平台，不改评测掩膜。\n'
    (out/'1024-01-父预测静态平面可行性审计.md').write_text(report)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
