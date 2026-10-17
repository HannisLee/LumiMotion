"""原135 TRAIN静态RGB与固定相机的一次CPU多视角三角化可行性。"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import itertools
import json
import math
import shlex
import socket
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / 'output/1017-01-静态DL未知光总控'
BASE = C / 'data_train_T04_temporal08'
FROZEN = C / 'snapshots/fixed_T08_skip_final_reset_v1'
OUT = C / '1017-05-TRAIN静态RGB多视角三角化'
CONFIG = dict(nfeatures=3000, contrastThreshold=.02, edgeThreshold=10,
              sigma=1.6, nOctaveLayers=3, descriptor='RootSIFT', ratio=.75,
              alpha_gt=252, erode_size=5, coordinate_add=.5, neighbors=8,
              axis_angle_max_deg=40., epipolar_max_px=1., fit_max_px=2.,
              min_views=4, min_parallax_deg=3., held_modulus=9, held_remainder=4,
              duplicate_round_decimals=1, image_grid=4, threads=4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def summary(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    return dict(count=int(a.size), nonfinite=int(a.size-finite.size),
                **({k: float(v) for k, v in zip(['min','p10','median','p90','p95','max','mean'],
                    [finite.min(), *np.quantile(finite,[.1,.5,.9,.95]), finite.max(), finite.mean()])}
                   if finite.size else {k: None for k in ['min','p10','median','p90','p95','max','mean']}))


def original_functions():
    ns = {'np':np, 'math':math}
    manifest_path=FROZEN/'snapshot_manifest.json'
    assert sha(manifest_path)=='093a6ac43a432b4b3154612751976946f74708201980978e880f1e3e30c3e3a3'
    manifest=json.loads(manifest_path.read_text());assert len(manifest)==96
    for relative, names in [('scene/dataset_readers.py',['blender_c2w_to_camera_extrinsics']),
                            ('utils/graphics_utils.py',['getWorld2View2','focal2fov','fov2focal'])]:
        assert sha(FROZEN/relative)==manifest[relative]
        module = ast.parse((FROZEN/relative).read_text())
        nodes = [n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),relative,'exec'), ns)
    return ns


def skew(t):
    x,y,z = t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def fundamental(a,b):
    relative=b['E'] @ np.linalg.inv(a['E'])
    R,t=relative[:3,:3],relative[:3,3]
    return np.linalg.inv(b['K']).T @ skew(t) @ R @ np.linalg.inv(a['K'])


def epipolar_errors(F, u, v):
    q = np.c_[u,np.ones(len(u))]; r = np.c_[v,np.ones(len(v))]
    l2=q@F.T; l1=r@F
    numerator=np.abs(np.sum(r*l2,axis=1))
    return numerator/np.maximum(np.linalg.norm(l1[:,:2],axis=1),1e-15), numerator/np.maximum(np.linalg.norm(l2[:,:2],axis=1),1e-15)


def project(X, cameras, views):
    E=np.stack([cameras[i]['E'] for i in views]); K=np.stack([cameras[i]['K'] for i in views])
    pc=np.einsum('nij,j->ni',E[:,:3,:3],X)+E[:,:3,3]
    h=np.einsum('nij,nj->ni',K,pc)
    z=pc[:,2]; safe=np.where(np.abs(h[:,2])>1e-12,h[:,2],np.where(h[:,2]<0,-1e-12,1e-12))
    return h[:,:2]/safe[:,None], z


def dlt(P, uv):
    A=np.stack([uv[:,0,None]*P[:,2]-P[:,0],uv[:,1,None]*P[:,2]-P[:,1]],axis=1).reshape(-1,4)
    _,_,vt=np.linalg.svd(A,full_matrices=False)
    if abs(vt[-1,3])<1e-12:return None
    X=vt[-1,:3]/vt[-1,3]
    return X if np.isfinite(X).all() else None


def refine(X, cameras, views, uv):
    fit=least_squares(lambda x:(project(x,cameras,views)[0]-uv).ravel(),X,
                      method='lm',max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    return fit.x


def max_parallax(X,cameras,views):
    rays=X-np.stack([cameras[i]['center'] for i in views]); rays/=np.linalg.norm(rays,axis=1)[:,None]
    return float(np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1,1))))


def neighbors(i, pool, cameras):
    axis=cameras[i]['axis']
    candidates=[(float(np.degrees(np.arccos(np.clip(axis@cameras[j]['axis'],-1,1)))),j) for j in pool if j!=i]
    return [j for angle,j in sorted(candidates) if angle<=CONFIG['axis_angle_max_deg']][:CONFIG['neighbors']]


def native_uv(index_coordinates):
    return np.asarray(index_coordinates,dtype=float).reshape(-1,2)+CONFIG['coordinate_add']


def mutual_matches(a,b,features):
    da,db=features[a]['descriptors'],features[b]['descriptors']
    if min(len(da),len(db))<2:return []
    bf=cv2.BFMatcher(cv2.NORM_L2)
    f=bf.knnMatch(da,db,k=2); r=bf.knnMatch(db,da,k=2)
    forward={m.queryIdx:m.trainIdx for m,n in f if m.distance<CONFIG['ratio']*n.distance}
    reverse={m.queryIdx:m.trainIdx for m,n in r if m.distance<CONFIG['ratio']*n.distance}
    return [(k,v) for k,v in sorted(forward.items()) if reverse.get(v)==k]


def fit_track(observations,cameras,features):
    views=[i for i,k in observations]; uv=np.stack([features[i]['uv'][k] for i,k in observations])
    P=np.stack([cameras[i]['P'] for i in views]); best=None
    # 顺序固定；相同内点数和误差保留先枚举的source编号对。
    for a,b in itertools.combinations(range(len(views)),2):
        X=dlt(P[[a,b]],uv[[a,b]])
        if X is None:continue
        predicted,z=project(X,cameras,views); err=np.linalg.norm(predicted-uv,axis=1)
        keep=(z>1e-6)&(err<=CONFIG['fit_max_px']); n=int(keep.sum())
        if n<CONFIG['min_views']:continue
        score=(-n,float(err[keep].mean()))
        if best is None or score<best[0]:best=(score,X,keep,(views[a],views[b]))
    if best is None:return {'accepted':False,'reason':'无四视角正深度一致集'}
    _,X,keep,pair=best; indices=np.flatnonzero(keep); inviews=[views[j] for j in indices]
    X=refine(X,cameras,inviews,uv[indices]); pred,z=project(X,cameras,views); err=np.linalg.norm(pred-uv,axis=1)
    if not np.isfinite(X).all() or not ((z[indices]>1e-6)&(err[indices]<=CONFIG['fit_max_px'])).all():
        return {'accepted':False,'reason':'固定内点优化后未保持阈值'}
    parallax=max_parallax(X,cameras,inviews)
    if parallax<CONFIG['min_parallax_deg']:return {'accepted':False,'reason':'有效视差不足','parallax_deg':parallax}
    loo=[]
    for j in indices:
        other=indices[indices!=j]; Xloo=dlt(P[other],uv[other])
        if Xloo is None:
            loo.append(dict(held_view=views[j],valid=False));continue
        Xloo=refine(Xloo,cameras,[views[t] for t in other],uv[other]); pp,zz=project(Xloo,cameras,[views[j]])
        distance=np.linalg.norm(X-cameras[views[j]]['center'])
        loo.append(dict(held_view=views[j],valid=bool(np.isfinite(Xloo).all()),
                        displacement_mm=float(np.linalg.norm(Xloo-X)*1000),
                        displacement_relative_distance=float(np.linalg.norm(Xloo-X)/distance),
                        withheld_error_px=float(np.linalg.norm(pp[0]-uv[j])),withheld_depth=float(zz[0])))
    return dict(accepted=True,xyz=X.tolist(),seed_pair=list(pair),parallax_deg=parallax,
                views=len(views),inlier_views=len(indices),fit_errors_px=err.tolist(),depths=z.tolist(),
                inlier_indices=indices.tolist(),loo=loo)


def geometry_selftest():
    K=np.array([[500.,0,400],[0,500,400],[0,0,1]])
    cameras=[]
    for x,y in [(0.,0.),(.5,0.),(0.,.4),(-.3,-.1)]:
        E=np.eye(4); E[:3,3]=[-x,-y,0]
        cameras.append(dict(E=E,K=K,P=K@E[:3],center=-E[:3,3],axis=np.array([0.,0.,1.])))
    X=np.array([.2,.1,3.]); uv,z=project(X,cameras,list(range(4)))
    reconstructed=dlt(np.stack([c['P'] for c in cameras]),uv)
    assert np.max(np.abs(X-reconstructed))<1e-10 and (z>0).all()
    d1,d2=epipolar_errors(fundamental(cameras[0],cameras[1]),uv[:1],uv[1:2]); assert max(d1.max(),d2.max())<1e-8
    raw=np.array([[399.5,399.5],[470.,420.]])
    converted=native_uv(raw)
    assert np.array_equal(converted[0],[400.,400.])
    Kindex=K.copy();Kindex[:2,2]-=.5
    rays_native=np.c_[converted,np.ones(len(raw))]@np.linalg.inv(K).T
    rays_index=np.c_[raw,np.ones(len(raw))]@np.linalg.inv(Kindex).T
    assert np.allclose(rays_native,rays_index,atol=1e-15,rtol=0)
    # 实际原转换函数的已知Blender轴测试，避免只用自身投影做恒等核验。
    ns=original_functions();R,t=ns['blender_c2w_to_camera_extrinsics'](np.eye(4).tolist())
    E=ns['getWorld2View2'](R,t).astype(float)
    assert np.allclose(E,np.diag([1.,-1.,-1.,1.]))
    p,z=project(np.array([0.,0.,-2.]),[dict(E=E,K=K)],[0])
    assert z[0]>0 and np.allclose(p[0],native_uv([[399.5,399.5]])[0])
    fake=[dict(uv=u[None]) for u in uv]
    result=fit_track([(i,0) for i in range(4)],cameras,fake)
    assert result['accepted'] and max(r['displacement_mm'] for r in result['loo'])<1e-6
    bad=uv[1:2]+[0,10]; _,e=epipolar_errors(fundamental(cameras[0],cameras[1]),uv[:1],bad); assert e[0]>9
    return dict(status='PASS',projection_DLT_max_abs=float(np.max(np.abs(X-reconstructed))),
                known_F_exact=True,known_F_vertical_mismatch_rejected=True,LOO_exact=True,
                actual_native_uv_center=True,original_Blender_identity_axes=True)


def run(out):
    assert socket.gethostname().split('.')[0]=='ushas'
    cv2.setNumThreads(CONFIG['threads']);cv2.setRNGSeed(0)
    ns=original_functions(); transforms=json.loads((BASE/'transforms_train.json').read_text())
    frames=sorted(transforms['frames'],key=lambda f:int(Path(f['file_path']).stem.split('_')[-1]))
    assert len(frames)==135 and {int(Path(f['file_path']).stem.split('_')[-1]) for f in frames}=={i for i in range(1,151) if i%10}
    paths=[BASE/'transforms_train.json']+[BASE/'images'/(Path(f['file_path']).name+('' if Path(f['file_path']).suffix else '.png')) for f in frames]
    inputs={str(p):sha(p) for p in paths}; assert len(inputs)==136
    # 只读历史来源清单，不加载其点云/灯表；本项实际数据输入严格136项。
    manifest_path=BASE/'no_gt_manifest.json';manifest=json.loads(manifest_path.read_text())
    assert not manifest['gt_read'] and not manifest['init_weights_loaded'] and len(manifest['files'])==138
    assert all(manifest['files'][str(Path(p).relative_to(BASE))]==digest for p,digest in inputs.items())
    provenance=dict(config=CONFIG,inputs=inputs,source_sha256=sha(__file__),opencv=cv2.__version__,
                    historical_source_manifest_sha256=sha(manifest_path),
                    original_functions={r:sha(FROZEN/r) for r in ['scene/dataset_readers.py','utils/graphics_utils.py']},
                    command=shlex.join([sys.executable,*sys.argv]),GT_read=False,model_read=False,optimizer_updates=0)
    write(out/'frozen_config_and_inputs.json',provenance);write(out/'selftest.json',geometry_selftest())
    fit=[i for i in range(135) if i%9!=4];held=[i for i in range(135) if i%9==4]
    assert len(fit)==120 and len(held)==15
    cameras=[];features=[None]*135;feature_rows=[None]*135
    sift=cv2.SIFT_create(**{k:CONFIG[k] for k in ['nfeatures','contrastThreshold','edgeThreshold','sigma','nOctaveLayers']})
    (out/'features').mkdir();(out/'overlays').mkdir()
    for i,(frame,path) in enumerate(zip(frames,paths[1:])):
        def value(k):return frame.get(k,transforms.get(k))
        fx,fy=value('fl_x'),value('fl_y');fovx=value('camera_angle_x');fovy=value('camera_angle_y')
        fovx=ns['focal2fov'](fx,800) if fx is not None and fx>0 else fovx
        fovy=ns['focal2fov'](fy,800) if fy is not None and fy>0 else (fovy if fovy is not None else fovx)
        f=np.array([ns['fov2focal'](fovx,800),ns['fov2focal'](fovy,800)])
        R,t=ns['blender_c2w_to_camera_extrinsics'](frame['transform_matrix']);E=ns['getWorld2View2'](R,t).astype(np.float64)
        K=np.array([[f[0],0,400],[0,f[1],400],[0,0,1]])
        inverse=np.linalg.inv(E);axis=inverse[:3,2];axis/=np.linalg.norm(axis)
        camera=dict(E=E,K=K,P=K@E[:3],center=inverse[:3,3],axis=axis,name=Path(path).stem)
        probe=np.array([0.,0.,2.,1.])@inverse.T; uv,z=project(probe[:3], [camera], [0]);assert np.max(np.abs(uv-[400,400]))<1e-7
        cameras.append(camera)
    def extract(i):
        path=paths[i+1];camera=cameras[i]
        rgba=np.asarray(Image.open(path).convert('RGBA'));assert rgba.shape==(800,800,4)
        alpha=rgba[:,:,3]>CONFIG['alpha_gt']; mask=cv2.erode(alpha.astype(np.uint8)*255,np.ones((5,5),np.uint8))
        gray=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2GRAY);keys,desc=sift.detectAndCompute(gray,mask)
        unique={}
        for j in sorted(range(len(keys)),key=lambda j:(-keys[j].response,j)):
            unique.setdefault(tuple(np.round(keys[j].pt,1)),j)
        ids=sorted(unique.values()); xy=np.array([keys[j].pt for j in ids],float).reshape(-1,2)
        descriptors=np.empty((0,128),np.float32) if not ids else np.sqrt(desc[ids]/np.maximum(desc[ids].sum(1,keepdims=True),1e-12)).astype(np.float32)
        dist=cv2.distanceTransform(alpha.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        radii=np.array([keys[j].size for j in ids]); angles=np.array([keys[j].angle for j in ids]); responses=np.array([keys[j].response for j in ids])
        rounded=np.rint(xy).astype(int);boundary=dist[rounded[:,1],rounded[:,0]] if len(ids) else np.empty(0)
        feature=dict(uv=native_uv(xy),descriptors=descriptors,scale=radii,boundary_distance=boundary)
        features[i]=feature
        np.savez_compressed(out/'features'/f'{camera["name"]}.npz',**feature,orientation=angles,response=responses,
                            kept_raw_indices=np.array(ids,dtype=int),raw_index_uv=np.array([k.pt for k in keys]).reshape(-1,2),
                            raw_scale=np.array([k.size for k in keys]),raw_orientation=np.array([k.angle for k in keys]),raw_response=np.array([k.response for k in keys]))
        feature_rows[i]=dict(index=i,name=camera['name'],split='held' if i in held else 'fit',raw=len(keys),unique=len(ids),scale=summary(radii),boundary_distance_px=summary(boundary))
        print(json.dumps(dict(stage='features',split=feature_rows[i]['split'],index=i,unique=len(ids))),flush=True)
    for i in fit:extract(i)
    assert all(features[i] is None for i in held)
    write(out/'fit_features_summary_before_held.json',[feature_rows[i] for i in fit])
    camera_serial=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in cameras]
    write(out/'cameras.json',camera_serial)
    pairs=sorted({tuple(sorted([i,j])) for i in fit for j in neighbors(i,fit,cameras)})
    held_pairs=[(i,j) for i in held for j in neighbors(i,fit,cameras)]
    write(out/'pair_plan.json',dict(fit=fit,held=held,fit_pairs=pairs,held_pairs=held_pairs))
    parent={};rank={}
    def find(x):
        if x not in parent:parent[x]=x;rank[x]=0
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    pair_rows=[]
    with (out/'fit_match_edges.jsonl').open('w') as log:
        for p,(a,b) in enumerate(pairs):
            matches=mutual_matches(a,b,features)
            if matches:
                ka,kb=np.array(matches).T;u=features[a]['uv'][ka];v=features[b]['uv'][kb]
                e1,e2=epipolar_errors(fundamental(cameras[a],cameras[b]),u,v)
                good=(e1<=1)&(e2<=1)
                for (x,y),err1,err2 in zip(np.array(matches)[good],e1[good],e2[good]):
                    union((a,int(x)),(b,int(y)))
                    log.write(json.dumps(dict(a=a,b=b,ka=int(x),kb=int(y),epipolar_px=[float(err1),float(err2)]))+'\n')
                n=int(good.sum())
            else:n=0
            pair_rows.append(dict(a=a,b=b,mutual=len(matches),known_F_pass=n))
            if p%50==0:print(json.dumps(dict(stage='fit_matching',pair=p,total=len(pairs),**pair_rows[-1])),flush=True)
    write(out/'fit_pairs_summary.json',pair_rows)
    groups={}
    for obs in sorted(parent):groups.setdefault(find(obs),[]).append(obs)
    components=sorted(groups.values(),key=lambda g:g[0]);tracks=[];lookup={}
    with (out/'tracks.jsonl').open('w') as log:
        for tid,obs in enumerate(components):
            row=dict(id=tid,observations=obs)
            if len({i for i,k in obs})!=len(obs):row.update(accepted=False,reason='同帧多关键点冲突')
            elif len(obs)<4:row.update(accepted=False,reason='少于四拟合视角')
            else:row.update(fit_track(obs,cameras,features))
            if row['accepted']:
                row['observations']=[list(x) for x in obs];tracks.append(row)
                for j in row['inlier_indices']:lookup[obs[j]]=tid
            log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if tid%100==0:print(json.dumps(dict(stage='triangulation',component=tid,total=len(components),accepted=len(tracks))),flush=True)
    write(out/'frozen_tracks_before_held.json',tracks)
    frozen_tracks_sha=sha(out/'frozen_tracks_before_held.json');byid={t['id']:t for t in tracks}
    write(out/'held_isolation_boundary.json',dict(frozen_tracks_sha256=frozen_tracks_sha,
          fit_features_extracted=len(fit),held_features_not_extracted=all(features[i] is None for i in held),
          UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    for i in held:extract(i)
    write(out/'features_summary.json',feature_rows)
    associations={};held_pair_rows=[]
    with (out/'held_descriptor_edges.jsonl').open('w') as log:
        for a,b in held_pairs:
            matches=mutual_matches(a,b,features);linked=0
            for ka,kb in matches:
                tid=lookup.get((b,kb))
                if tid is None:continue
                associations.setdefault((a,ka),{}).setdefault(tid,set()).add(b);linked+=1
                log.write(json.dumps(dict(held=a,held_key=ka,fit=b,fit_key=kb,track=tid))+'\n')
            held_pair_rows.append(dict(held=a,fit=b,mutual=len(matches),linked_fit_inlier=linked))
    write(out/'held_pairs_summary.json',held_pair_rows)
    candidates=[];ambiguities=[]
    for (view,key),group in sorted(associations.items()):
        qualified={tid:support for tid,support in group.items() if len(support)>=2}
        if len(group)>1:
            ambiguities.append(dict(view=view,key=key,reason='同保留关键点关联多个轨迹',tracks={str(t):sorted(v) for t,v in group.items()}));continue
        if len(qualified)==1:
            tid,support=next(iter(qualified.items()));candidates.append(dict(view=view,key=key,track=tid,support=sorted(support)))
    multiple={}
    for r in candidates:multiple.setdefault((r['view'],r['track']),[]).append(r['key'])
    held_rows=[]
    for row in candidates:
        view,key,tid=row['view'],row['key'],row['track']
        if len(multiple[(view,tid)])>1:
            ambiguities.append(dict(**row,reason='同轨同保留帧关联多个位置'));continue
        predicted,z=project(np.array(byid[tid]['xyz']),cameras,[view]);observed=features[view]['uv'][key]
        row.update(predicted_uv=predicted[0].tolist(),observed_uv=observed.tolist(),error_px=float(np.linalg.norm(predicted[0]-observed)),depth=float(z[0]),negative_depth=bool(z[0]<=0),outside_image=bool(((predicted[0]<0)|(predicted[0]>=800)).any()))
        held_rows.append(row)
    write(out/'held_ambiguities.json',ambiguities);write(out/'held_predictions_all.json',held_rows)
    assert sha(out/'frozen_tracks_before_held.json')==frozen_tracks_sha
    xyz=np.array([t['xyz'] for t in tracks],float).reshape(-1,3);np.save(out/'triangulated_xyz.npy',xyz)
    coverage=[]
    for i,path in enumerate(paths[1:]):
        rgba=np.array(Image.open(path).convert('RGBA'));rgb=rgba[:,:,:3].copy();alpha=rgba[:,:,3]>252
        obs=[]
        if i in fit:
            for t in tracks:
                for j in t['inlier_indices']:
                    vi,key=t['observations'][j]
                    if vi==i:obs.append(features[i]['uv'][key])
        else:obs=[r['observed_uv'] for r in held_rows if r['view']==i]
        bins=np.zeros((4,4),int)
        for u,v in obs:
            bins[min(3,int(v//200)),min(3,int(u//200))]+=1
            cv2.circle(rgb,(int(round(u-.5)),int(round(v-.5))),3,(255,70,40),1)
        # 全轨迹投影不进行可见性/Alpha筛选，仅黄色十字展示正深度图内预测。
        for X in xyz:
            uv,z=project(X,cameras,[i]);u,v=uv[0]
            if z[0]>0 and 0<=u<800 and 0<=v<800:cv2.drawMarker(rgb,(int(round(u-.5)),int(round(v-.5))),(240,220,30),cv2.MARKER_CROSS,3,1)
        for line in [200,400,600]:cv2.line(rgb,(line,0),(line,799),(60,180,240),1);cv2.line(rgb,(0,line),(799,line),(60,180,240),1)
        Image.fromarray(rgb).save(out/'overlays'/f'{cameras[i]["name"]}.png')
        areas=alpha.reshape(4,200,4,200).sum(axis=(1,3))
        coverage.append(dict(view=i,name=cameras[i]['name'],split='held' if i in held else 'fit',observations=len(obs),grid4x4_observation_count=bins.tolist(),grid4x4_alpha_pixels=areas.tolist()))
    write(out/'coverage.json',coverage)
    assert all(sha(p)==digest for p,digest in inputs.items())
    assert sha(__file__)==provenance['source_sha256']
    assert sha(manifest_path)==provenance['historical_source_manifest_sha256']
    assert all(sha(FROZEN/r)==digest for r,digest in provenance['original_functions'].items())
    reject={}
    for line in (out/'tracks.jsonl').read_text().splitlines():
        row=json.loads(line)
        if not row['accepted']:reject[row['reason']]=reject.get(row['reason'],0)+1
    loo=[r for t in tracks for r in t['loo'] if r['valid']]
    result=dict(status='COMPLETED_CPU_DATA_FEASIBILITY_AWAITING_REVIEW',GT_read=False,model_read=False,camera_updates=0,optimizer_updates=0,
                input136_unchanged=True,source_unchanged=True,config=CONFIG,fit_views=fit,held_views=held,
                keypoints=sum(r['unique'] for r in feature_rows),fit_pairs=len(pairs),mutual_matches=sum(r['mutual'] for r in pair_rows),known_F_edges=sum(r['known_F_pass'] for r in pair_rows),
                components=len(components),rejected=reject,accepted_tracks=len(tracks),
                inlier_views=summary([t['inlier_views'] for t in tracks]),parallax_deg=summary([t['parallax_deg'] for t in tracks]),
                fit_inlier_errors_px=summary([t['fit_errors_px'][j] for t in tracks for j in t['inlier_indices']]),
                fit_all_track_errors_px=summary([e for t in tracks for e in t['fit_errors_px']]),
                loo_displacement_mm=summary([r['displacement_mm'] for r in loo]),loo_displacement_relative_distance=summary([r['displacement_relative_distance'] for r in loo]),
                loo_withheld_errors_px=summary([r['withheld_error_px'] for r in loo]),
                held_pairs=len(held_pairs),held_ambiguous_observations=len(ambiguities),held_predictions=len(held_rows),held_errors_px=summary([r['error_px'] for r in held_rows]),
                held_negative_depth=sum(r['negative_depth'] for r in held_rows),held_outside_image=sum(r['outside_image'] for r in held_rows),
                held_errors_le2_report_only=sum(r['error_px']<=2 for r in held_rows),held_errors_gt10_report_only=sum(r['error_px']>10 for r in held_rows),
                xyz_quantiles={axis:summary(xyz[:,j]) for j,axis in enumerate(['x','y','z'])},
                frozen_tracks_sha256=frozen_tracks_sha,provenance_sha256=sha(out/'frozen_config_and_inputs.json'),
                limitations=['保留关联不使用F/预测误差，不按保留误差删点','描述子可匹配子集，不是全表面无偏验证','拟合LOO属于全拟合资格筛选后的敏感性诊断','中心Alpha侵蚀不保证整个描述子支持域在前景','投影不表示可见；没有语义GT部位筛选','本结果不自动启动训练'])
    write(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:print(json.dumps(geometry_selftest(),ensure_ascii=False));return
    OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir()
    (OUT/'frozen_source.py').write_bytes(Path(__file__).read_bytes())
    class Tee:
        def __init__(self,stream,log):self.stream,self.log=stream,log
        def write(self,text):self.stream.write(text);self.log.write(text);return len(text)
        def flush(self):self.stream.flush();self.log.flush()
    with (OUT/'logs/construction.log').open('w') as log:
        streams=sys.stdout,sys.stderr
        sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
        write(OUT/'state.json',dict(status='RUNNING',pid=__import__('os').getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        try:
            run(OUT)
            write(OUT/'state.json',dict(status='COMPLETED_CPU_AWAITING_REVIEW',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        except BaseException as error:
            import traceback
            failure=dict(status='FAILED',type=type(error).__name__,message=str(error),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(OUT/'FAILED.json',failure);write(OUT/'state.json',failure)
            traceback.print_exc();raise
        finally:
            sys.stdout,sys.stderr=streams


if __name__=='__main__':main()
