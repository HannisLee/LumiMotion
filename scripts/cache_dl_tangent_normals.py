"""以预测表面轨迹的局部切平面估计法线，保持原观测集合不变。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'tracks.pt').exists():raise FileExistsError('禁止覆盖派生缓存。')
    data=torch.load(args.input,weights_only=False,map_location='cpu')
    canonical=data['canonical_xyz'].numpy()
    distance,indices=cKDTree(canonical).query(canonical,k=16)
    neighbors=torch.from_numpy(indices)
    extent=float(np.linalg.norm(canonical.max(0)-canonical.min(0)))
    normal_list=[];accepted=[];angles=[]
    for positions,original in zip(data['positions'],data['normals']):
        neighborhood=positions[neighbors]
        centered=neighborhood-neighborhood.mean(1,keepdim=True)
        covariance=centered.transpose(1,2)@centered/16
        values,vectors=torch.linalg.eigh(covariance)
        tangent=vectors[:,:,0]
        tangent*=torch.where((tangent*original).sum(-1,keepdim=True)>=0,1.,-1.)
        radius=(neighborhood-positions[:,None]).norm(dim=-1).amax(-1)
        reliable=(values[:,1]>1e-10)&(values[:,0]/values[:,1].clamp_min(1e-10)<.1)&(radius<extent*.03)
        normal_list.append(torch.where(reliable[:,None],tangent,original))
        accepted.append(reliable)
        angles.append(torch.rad2deg(torch.acos((tangent*original).sum(-1).clamp(-1,1))))
    data['normals']=torch.stack(normal_list)
    reliable=torch.stack(accepted);angles=torch.stack(angles);valid=data['valid']
    data['tangent_normal_reliable']=reliable
    data['parent_cache_sha256']=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    data['normal_protocol']='16个canonical近邻，时刻变形后PCA；lambda0/lambda1<.1，半径<包围盒对角线.03，否则保留原法线；只按原法线定符号'
    torch.save(data,out/'tracks.pt')
    record=dict(input=args.input,output=str(out/'tracks.pt'),parent_cache_sha256=data['parent_cache_sha256'],
                accepted_valid_fraction=float((valid&reliable).sum()/valid.sum()),
                accepted_normal_angle_median=float(angles[valid&reliable].median()),
                accepted_normal_angle_p90=float(angles[valid&reliable].quantile(.9)),
                original_observation_mask_unchanged=True,geometry_updated=False,
                protocol=data['normal_protocol'])
    (out/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    # 预览脚本需要原代表帧尺寸；链接原图仅用于输入四类图追溯，不伪称新法线渲染。
    for path in Path(args.input).parent.glob('*.png'):
        (out/path.name).symlink_to(path.resolve())
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
