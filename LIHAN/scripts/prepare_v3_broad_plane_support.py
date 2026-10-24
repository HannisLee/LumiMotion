"""从父预测与全部117 TRAIN形变生成宽平面支持，不读取GT几何或heldout。"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull,Delaunay,cKDTree
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from utils.general_utils import build_rotation
from utils.no_gt_guard import install_no_gt_guard


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.no_grad()
def main(args):
    for key in ['model','source','plane_audit','cache','output','parent']:setattr(args,key,getattr(args,key).resolve())
    out=args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    plane=json.loads(args.plane_audit.read_text());assert plane['gt_read'] is False and plane['heldout_read'] is False
    assert sha(args.cache) in plane['source_sha256'].values()
    install_no_gt_guard([args.model,args.source,args.parent,args.plane_audit,args.cache,out])
    torch.set_num_threads(4);model=DynamicSurfaceModel.load(args.model,0);cameras=load_train_cameras(args.source)
    manifest=json.loads((args.source/'no_gt_manifest.json').read_text());frames=[int(c.image_name_train_light.split('_')[-1]) for c in cameras];assert frames==manifest['train_frames'] and len(frames)==117
    normal=np.array(plane['plane_normal_unsigned']);origin=np.array(plane['plane_origin']);extent=float(plane['extent'])
    cache=torch.load(args.cache,map_location='cpu',weights_only=False);sample=cache['canonical_xyz'].numpy()
    sample_mask=(~cache['dynamic'].numpy())&(np.abs((sample-origin)@normal)<extent*.005)&(cache['valid'].sum(0).numpy()>=10)
    axis=np.cross(normal,[1.,0,0]);axis/=np.linalg.norm(axis);basis=np.stack([axis,np.cross(normal,axis)])
    uv=(sample[sample_mask]-origin)@basis.T;tri=Delaunay(uv[ConvexHull(uv).vertices])
    xyz=model.pc.get_xyz.cpu().numpy();gate=model.pc.get_binary_feature().cpu().numpy().reshape(-1)
    axes=build_rotation(model.pc.get_rotation)[:,:,2].cpu().numpy()
    masks=dict(high_confidence_static=gate<.1,near_plane=np.abs((xyz-origin)@normal)<extent*.03,
        within_train_hull=tri.find_simplex((xyz-origin)@basis.T)>=0,normal_not_side=np.abs(axes@normal)>.7)
    selected=np.flatnonzero(np.logical_and.reduce(list(masks.values())))
    low=np.full((len(selected),3),np.inf);high=-low.copy();nearest=np.full(len(selected),np.inf)
    for j,camera in enumerate(cameras):
        camera.load2device();current=model.effective(camera).get_xyz.cpu().numpy();points=current[selected]
        low=np.minimum(low,points);high=np.maximum(high,points)
        nearest=np.minimum(nearest,cKDTree(current[gate>.5]).query(points,workers=4)[0])
        camera.load2device('cpu')
        if j%20==0:print('TRAIN形变支持核验',j+1,'/117',flush=True)
    motion=np.linalg.norm(high-low,axis=-1);keep=(motion<extent*.001)&(nearest>extent*.02);indices=selected[keep]
    np.savez_compressed(out/'support.npz',indices=indices,initial_indices=selected,normal=normal,origin=origin,extent=extent)
    ply=args.parent/'point_cloud/iteration_35000/point_cloud.ply';deform=args.parent/'deform/iteration_35000/deform.pth'
    result=dict(gt_read=False,heldout_read=False,training_updates=0,source_sha256={str(ply):sha(ply),str(deform):sha(deform),str(args.plane_audit):sha(args.plane_audit),str(args.cache):sha(args.cache)},
        total_points=len(xyz),individual_gate_counts={k:int(v.sum()) for k,v in masks.items()},initial_intersection=len(selected),final_support=len(indices),
        removed_motion=int((motion>=extent*.001).sum()),removed_dynamic_proximity=int((nearest<=extent*.02).sum()),train_motion_frames=frames,
        thresholds=dict(plane_distance=extent*.03,static_gate_lt=.1,opacity_threshold=None,absolute_normal_dot_gt=.7,motion_range_diagonal_lt=extent*.001,dynamic_clearance_gt=extent*.02),
        support_motion_max=float(motion[keep].max()),support_dynamic_distance_min=float(nearest[keep].min()),support_plane_residual_mean=float(np.abs((xyz[indices]-origin)@normal).mean()),
        support_sha256=sha(out/'support.npz'),limitations=['只用父预测门控、稳定和动态净空，不能保证无语义鞋误纳','面位置项仍按原tau=.005extent归一化，以smooth-L1鲁棒惩罚宽带远点','不改变评测掩膜或删除点'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','train_motion_frames']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','source','parent','plane-audit','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
