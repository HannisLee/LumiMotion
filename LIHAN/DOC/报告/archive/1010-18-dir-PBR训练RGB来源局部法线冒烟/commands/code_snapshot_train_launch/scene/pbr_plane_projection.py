"""只用Stage1预测建立平面几何投影；训练及所有原生渲染共享。"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from torch.nn import functional as F
from plyfile import PlyData
from scene.pbr_deformation import _source_assets, _sha, validate_deformation_anchor
from scene.pbr_geometry import fit_source_plane, quaternion_normal

METHOD='source_prediction_temporal_plane_geometry_projection'
ASSET='pbr_plane_projection.npz'
METADATA='pbr_plane_projection.json'


def validate_plane_projection(args):
    if not getattr(args,'pbr_plane_projection',False):return
    validate_deformation_anchor(args)
    if not getattr(args,'pbr_deformation_anchor',False):
        raise ValueError('来源平面投影必须启用原生方向光几何与XYZ输入锚定。')
    if (not math.isfinite(args.pbr_plane_support_band) or args.pbr_plane_support_band<=0 or
        args.pbr_plane_support_band<args.pbr_plane_band or
        not math.isfinite(args.pbr_plane_motion_span) or args.pbr_plane_motion_span<=0):
        raise ValueError('来源平面支持band/时间稳定阈值无效。')


def camera_signature(cameras):
    rows=[dict(frame=str(c.image_name_train_light),time=float(c.fid.item()),
               center=c.camera_center.detach().cpu().tolist()) for c in cameras]
    rows.sort(key=lambda row:row['frame'])
    if (len(rows)<3 or len({r['frame'] for r in rows})!=len(rows) or
        len({r['time'] for r in rows})<3 or
        any(not 0<=r['time']<=1 or len(r['center'])!=3 or
            not all(math.isfinite(x) for x in r['center']) for r in rows)):
        raise ValueError('来源平面投影需要至少三个不同训练时刻与有效相机。')
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return rows,digest


def quaternion_multiply(a,b):
    aw,av=a[:,:1],a[:,1:];bw,bv=b[:,:1],b[:,1:]
    return torch.cat((aw*bw-(av*bv).sum(1,keepdim=True),aw*bv+bw*av+torch.cross(av,bv,dim=1)),1)


def align_quaternion_normal(raw,target):
    """最短弧将实际几何轴旋到目标轴，保留平面切向旋转和raw norm。"""
    if (raw.ndim!=2 or raw.shape[1]!=4 or target.shape!=(len(raw),3) or
        not torch.isfinite(raw).all() or not torch.isfinite(target).all() or
        (raw.norm(dim=1)<1e-12).any() or (target.norm(dim=1)<1e-12).any()):
        raise ValueError('平面几何投影收到无效四元数/轴。')
    q=F.normalize(raw,dim=1);normal=quaternion_normal(q);target=F.normalize(target,dim=1)
    dot=(normal*target).sum(1,keepdim=True).clamp(-1,1)
    delta=torch.cat((1+dot,torch.cross(normal,target,dim=1)),1)
    axis=F.one_hot(normal.abs().argmin(1),num_classes=3).to(normal)
    axis=F.normalize(torch.cross(normal,axis,dim=1),dim=1)
    opposite=torch.cat((torch.zeros_like(dot),axis),1)
    delta=torch.where(dot>-1+1e-6,F.normalize(delta,dim=1),opposite)
    return F.normalize(quaternion_multiply(delta,q),dim=1)*raw.norm(dim=1,keepdim=True)


class PlaneProjection:
    def __init__(self,support,plane,offset,polarity):
        if (support.ndim!=1 or support.dtype!=torch.bool or plane.shape!=(3,) or
            offset.numel()!=1 or polarity.shape!=support.shape or
            not torch.isfinite(plane).all() or not torch.isfinite(offset).all() or
            not torch.allclose(plane.norm(),plane.new_tensor(1.),atol=1e-5,rtol=0) or
            not ((polarity==1)|(polarity==-1)).all()):
            raise ValueError('来源平面几何投影参数无效。')
        self.support=support;self.plane=plane;self.offset=offset.reshape(())
        self.polarity=polarity

    def apply(self,pc,prediction):
        xyz=pc.get_xyz;q=pc._rotation;dx=prediction['d_xyz'];dr=prediction['d_rotation']
        if (xyz.shape!=(len(self.support),3) or q.shape!=(len(self.support),4) or
            dx.shape!=xyz.shape or dr.shape!=q.shape or
            xyz.device!=self.support.device or xyz.dtype!=self.plane.dtype):
            raise ValueError('平面几何投影与实际Gaussian模型不符。')
        ids=self.support
        world_xyz=xyz[ids]+dx[ids]
        world_xyz=world_xyz-(world_xyz@self.plane+self.offset)[:,None]*self.plane
        world_q=q[ids]+dr[ids]
        target=self.polarity[ids,None]*self.plane
        world_q=align_quaternion_normal(world_q,target)
        out=dict(prediction);out['d_xyz']=dx.clone();out['d_rotation']=dr.clone()
        out['d_xyz'][ids]=world_xyz-xyz[ids]
        out['d_rotation'][ids]=world_q-q[ids]
        return out


@torch.no_grad()
def _derive(model,iteration,deform,cameras,args,device):
    xyz_np,source,snapshots=_source_assets(model,iteration)
    v=PlyData.read(model/f'point_cloud/iteration_{iteration}/point_cloud.ply')['vertex'].data
    q_np=np.column_stack([v[f'rot_{i}'] for i in range(4)])
    if not np.isfinite(q_np).all() or not np.isfinite(v['fea_0']).all() or not np.isfinite(v['opacity']).all():
        raise ValueError('来源旋转/feature/opacity无效。')
    source_state=torch.load(model/f'deform/iteration_{iteration}/deform.pth',map_location='cpu')
    current=deform.deform.state_dict()
    if set(source_state)!=set(current) or any(not torch.equal(source_state[k],current[k].detach().cpu()) for k in source_state):
        raise ValueError('平面投影必须使用完全未修改的来源MLP。')
    xyz=torch.as_tensor(xyz_np,device=device);q=torch.as_tensor(q_np,device=device)
    feature=torch.sigmoid(torch.tensor(v['fea_0'].copy(),device=device)[:,None]*2)
    opacity=torch.sigmoid(torch.tensor(v['opacity'].copy(),device=device))
    static=feature[:,0]<.5
    plane,offset,_,fit=fit_source_plane(xyz,static,opacity,q,band=args.pbr_plane_band)
    rows,signature=camera_signature(cameras)
    lo=hi=None;max_dist=xyz.new_zeros(len(xyz));min_cos=xyz.new_ones(len(xyz))
    for row in rows:
        d=deform.step(xyz,xyz.new_full((len(xyz),1),row['time']),iteration=iteration,
                      feature=feature,camera_center=xyz.new_tensor(row['center']),is_training=False)
        value=d['d_xyz'];rotation=d['d_rotation']
        if value.shape!=xyz.shape or rotation.shape!=q.shape or not torch.isfinite(value).all() or not torch.isfinite(rotation).all():
            raise ValueError('来源MLP形变形状或有限性错误。')
        lo=value.clone() if lo is None else torch.minimum(lo,value)
        hi=value.clone() if hi is None else torch.maximum(hi,value)
        max_dist=torch.maximum(max_dist,((xyz+value)@plane+offset).abs())
        min_cos=torch.minimum(min_cos,(quaternion_normal(q+rotation)@plane).abs())
    span=(hi-lo).norm(dim=1);canonical=(xyz@plane+offset).abs();cos=(quaternion_normal(q)@plane).abs()
    support=(static & (opacity>.05) & (canonical<=args.pbr_plane_support_band) &
             (max_dist<=args.pbr_plane_support_band) & (span<=args.pbr_plane_motion_span) &
             (cos>=.5) & (min_cos>=.5))
    if int(support.sum())<32:raise ValueError('缺少足够时间稳定的来源平面支持。')
    polarity=torch.where((quaternion_normal(q)@plane)>=0,1.,-1.)
    arrays=dict(support=support.cpu().numpy(),plane=plane.cpu().numpy(),offset=offset.cpu().numpy(),
                polarity=polarity.cpu().numpy(),motion_span=span.cpu().numpy(),
                max_plane_distance=max_dist.cpu().numpy(),min_axis_cosine=min_cos.cpu().numpy())
    details=dict(source_files=source,source_snapshots=snapshots,training_camera_sha256=signature,
                 training_cameras=len(rows),source_plane_fit=fit,support_points=int(support.sum()))
    return PlaneProjection(support,plane,offset,polarity),arrays,details


@torch.no_grad()
def create_plane_projection(model,pc,deform,cameras,args,source_iteration):
    validate_plane_projection(args);model=Path(model)
    if (model/ASSET).exists() or (model/METADATA).exists():raise FileExistsError('平面投影资产禁止覆盖。')
    expected,_,_=_source_assets(model,source_iteration)
    if not np.array_equal(pc.get_xyz.detach().cpu().numpy(),expected):
        raise ValueError('来源平面投影只能从真实复制Stage1坐标创建。')
    projection,arrays,details=_derive(model,source_iteration,deform,cameras,args,pc.get_xyz.device)
    np.savez(model/ASSET,**arrays)
    record=dict(format_version=1,method=METHOD,source_iteration=source_iteration,points=len(expected),
                fit_band=args.pbr_plane_band,support_band=args.pbr_plane_support_band,
                max_motion_span=args.pbr_plane_motion_span,asset_sha256=_sha(model/ASSET),
                gt_geometry_inputs=False,gt_normal_inputs=False,gt_material_inputs=False,**details)
    (model/METADATA).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return projection,record


@torch.no_grad()
def load_plane_projection(model,pc,deform,cameras,args,iteration):
    validate_plane_projection(args);model=Path(model);record=json.loads((model/METADATA).read_text())
    if (record.get('format_version')!=1 or record.get('method')!=METHOD or
        any(record.get(k) is not False for k in ['gt_geometry_inputs','gt_normal_inputs','gt_material_inputs']) or
        record.get('points')!=len(pc.get_xyz) or
        record.get('fit_band')!=args.pbr_plane_band or record.get('support_band')!=args.pbr_plane_support_band or
        record.get('max_motion_span')!=args.pbr_plane_motion_span or _sha(model/ASSET)!=record.get('asset_sha256')):
        raise ValueError('平面投影资产/来源协议与当前模型不符。')
    src=record.get('source_iteration')
    if type(src) is not int or not 1<=src<=iteration:raise ValueError('平面投影来源iteration无效。')
    if iteration>src:
        marker=json.loads((model/f'pbr_directional/iteration_{iteration}/lighting.json').read_text())
        if marker.get('iteration')!=iteration or marker.get('plane_projection_sha256')!=record['asset_sha256']:
            raise ValueError('当前PBR checkpoint未绑定来源平面几何模型。')
    projection,expected,details=_derive(model,src,deform,cameras,args,pc.get_xyz.device)
    if any(record.get(k)!=v for k,v in details.items()):raise ValueError('平面投影来源或相机记录不符。')
    with np.load(model/ASSET,allow_pickle=False) as asset:
        if set(asset.files)!=set(expected):raise ValueError('平面投影资产字段不符。')
        for k,value in expected.items():
            if asset[k].dtype!=value.dtype or not np.array_equal(asset[k],value):
                raise ValueError('平面投影资产与真实来源重算不符：'+k)
    return projection
