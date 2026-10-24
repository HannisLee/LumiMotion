"""CPU映射父预测平面支持到全点，并仅用TRAIN时刻父MLP检查运动与动态邻近。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import ConvexHull,Delaunay,cKDTree
from utils.time_utils import DeformNetwork


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(args):
    torch.set_num_threads(4);out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    audit=json.loads(args.plane_audit.read_text());assert not audit['gt_read'] and not audit['heldout_read']
    n=np.array(audit['plane_normal_unsigned']);origin=np.array(audit['plane_origin']);extent=audit['extent'];distance_limit=extent*.005
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    sample=cache['canonical_xyz'].numpy();sample_gate=~cache['dynamic'].numpy()
    sample_mask=sample_gate&(np.abs((sample-origin)@n)<distance_limit)&(cache['valid'].sum(0).numpy()>=10)
    axis=np.cross(n,[1.,0,0]);axis/=np.linalg.norm(axis);basis=np.stack([axis,np.cross(n,axis)])
    uv=(sample[sample_mask]-origin)@basis.T;hull=ConvexHull(uv);tri=Delaunay(uv[hull.vertices]);tree=cKDTree(uv)
    ply=args.parent/'point_cloud/iteration_35000/point_cloud.ply';v=PlyData.read(str(ply))['vertex'];xyz=np.stack([v[k] for k in ['x','y','z']],-1)
    gate=1/(1+np.exp(np.clip(-2*np.asarray(v['fea_0']),-80,80)));opacity=1/(1+np.exp(np.clip(-np.asarray(v['opacity']),-80,80)))
    q=np.stack([v['rot_'+str(i)] for i in range(4)],-1);q/=np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=q.T
    normal=np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],-1)
    xy=(xyz-origin)@basis.T
    gates=dict(high_confidence_static=gate<.1,opaque=opacity>.1,near_plane=np.abs((xyz-origin)@n)<distance_limit,
        within_train_hull=tri.find_simplex(xy)>=0,near_train_receiver=tree.query(xy)[0]<extent*.03,normal_not_side=np.abs(normal@n)>.7)
    initial=np.logical_and.reduce(list(gates.values()));selected=np.flatnonzero(initial)
    model=DeformNetwork(is_blender=True,pred_color=True,hyper_dim=1)
    checkpoint=args.parent/'deform/iteration_35000/deform.pth';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False));model.pred_color=False;model.eval().requires_grad_(False)
    times=[1,31,59,81,111,149];positions=[];nearest=[]
    for frame in times:
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(xyz),8192):
                point=torch.from_numpy(xyz[start:start+8192].copy());feat=torch.from_numpy(gate[start:start+8192,None].copy())
                delta=model(point,torch.full((len(point),1),(frame-1)/149),feature=feat,iteration=35000)['d_xyz']
                chunks.append((point+delta).numpy())
        current=np.concatenate(chunks);positions.append(current[selected]);nearest.append(cKDTree(current[gate>.5]).query(current[selected])[0])
        print('TRAIN frame',frame,'CPU完成',flush=True)
    positions=np.asarray(positions);motion=np.linalg.norm(positions-positions.mean(0),axis=-1).max(0)
    dynamic_distance=np.asarray(nearest).min(0)
    keep=(motion<extent*.001)&(dynamic_distance>extent*.02)
    final=selected[keep];np.savez_compressed(out/'support.npz',indices=final,initial_indices=selected,normal=n,origin=origin,extent=extent)
    result=dict(gt_read=False,heldout_read=False,gpu_used=False,source_sha256={str(ply):sha(ply),str(checkpoint):sha(checkpoint),str(args.plane_audit):sha(args.plane_audit)},
        total_points=len(xyz),individual_gate_counts={k:int(m.sum()) for k,m in gates.items()},initial_intersection=len(selected),
        final_support=len(final),removed_motion=int((motion>=extent*.001).sum()),removed_dynamic_proximity=int((dynamic_distance<=extent*.02).sum()),
        train_motion_frames=times,thresholds=dict(plane_distance=distance_limit,static_gate_lt=.1,opacity_gt=.1,train_neighbor_radius=extent*.03,
        absolute_normal_dot_gt=.7,motion_max_lt=extent*.001,dynamic_clearance_gt=extent*.02),
        support_motion_max=float(motion[keep].max()),support_dynamic_distance_min=float(dynamic_distance[keep].min()),
        support_plane_residual_mean=float(np.abs((xyz[final]-origin)@n).mean()),support_sha256=sha(out/'support.npz'),
        limitations=['动态净空只核验6个TRAIN时刻，不证明所有时刻鞋误纳为零','支持为父预测固定索引；不使用GT分区或GT几何','法线侧面门禁来自父预测，用于避免平台侧壁被压平，不是评测删点'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/tmp/lumimotion-ppt-1017/NotoSansCJKsc-Regular.otf')
    fig,axes=plt.subplots(1,2,figsize=(13,5));every=np.arange(0,len(xyz),12)
    for ax,coord in zip(axes,[(0,1),(0,2)]):
        ax.scatter(xyz[every,coord[0]],xyz[every,coord[1]],c='lightgray',s=1,label='父模型预测点')
        ax.scatter(xyz[final[::5],coord[0]],xyz[final[::5],coord[1]],c='#008c70',s=2,label='平面先验候选支持')
        dyn=np.flatnonzero(gate>.5)[::10];ax.scatter(xyz[dyn,coord[0]],xyz[dyn,coord[1]],c='#d64b34',s=2,label='继承的动态门控')
        ax.set_aspect('equal');ax.set_xlabel('XYZ'[coord[0]]);ax.set_ylabel('XYZ'[coord[1]]);ax.legend(markerscale=3,prop=font)
    fig.tight_layout();fig.savefig(out/'support_top_front.png',dpi=160);plt.close(fig)
    (out/'1024-01-全点预测平面支持映射.md').write_text('# 全点预测平面支持映射\n\n仅CPU和TRAIN预测；不读取GT/heldout，不新增训练。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n图中绿点是未来可选先验支持，红点来自父预测动态门控，灰点为父预测几何；不是GT语义。\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','plane-audit','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
