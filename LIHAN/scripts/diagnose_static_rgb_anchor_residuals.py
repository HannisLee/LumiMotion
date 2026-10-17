"""冻结RGB位置对T08全图native缓存的CPU残差；不筛点、不训练。"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'output/1017-01-静态DL未知光总控'
PARENT=C/'1017-05-TRAIN静态RGB多视角三角化'
CACHE=C/'T08_plane_counterfactual_v1'
OUT=C/'1017-06-TRAIN静态RGB位置残差'
LOCK_SHA='79ebde6317c446e231c790b57b12546e0e3569bbfac7ce7c90ad101e29aa630e'
QUANT_SHA='58ae4dbce2bf1829bf659e54b9f4f4e6125c8146792c25588bd8921f994931df'
TRACK_SHA='1f73f399bc0fce74762a0ced282d8c4283b0efaed9172122f1389a27901936ca'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stats(values):
    a=np.array([x for x in values if x is not None],float);assert np.isfinite(a).all()
    out=dict(fixed_count=len(values),valid_count=len(a),missing_count=len(values)-len(a))
    out.update({k:None for k in ['mean','min','p10','median','p90','p95','max']})
    if len(a):out.update({k:float(v) for k,v in zip(['mean','min','p10','median','p90','p95','max'],[a.mean(),a.min(),*np.quantile(a,[.1,.5,.9,.95]),a.max()])})
    return out


def bilinear(depths,alpha,uv800):
    xy=np.asarray(uv800,float)/2-.5
    x,y=xy;ix,iy=int(np.floor(x)),int(np.floor(y))
    result=dict(array_xy=xy.tolist(),outside_four_neighbors=bool(ix<0 or iy<0 or ix+1>=400 or iy+1>=400))
    if result['outside_four_neighbors']:return result
    dx,dy=x-ix,y-iy
    weights=np.array([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy])
    positions=[(iy,ix),(iy,ix+1),(iy+1,ix),(iy+1,ix+1)]
    result['weights']=weights.tolist();aa=np.array([alpha[i,j] for i,j in positions],float)
    result['alpha_neighbors']=[float(v) if np.isfinite(v) else None for v in aa]
    result['alpha']=float(weights@aa) if np.isfinite(aa).all() else None
    for name,d in depths.items():
        v=np.array([d[i,j] for i,j in positions],float);valid=np.isfinite(v)&(v>0)
        result[name]=dict(neighbors=[float(z) if np.isfinite(z) else None for z in v],
            four_finite_positive=bool(valid.all()),valid_neighbors=int(valid.sum()),
            interpolated=float(weights@v) if valid.all() else None,
            minimum=float(v.min()) if valid.all() else None,maximum=float(v.max()) if valid.all() else None,
            range_mm=float(np.ptp(v)*1000) if valid.all() else None)
    e,m=result['expected']['interpolated'],result['median']['interpolated']
    result['EM_abs_difference_mm']=abs(e-m)*1000 if e is not None and m is not None else None
    return result


def equal_weight(rows,coordinate,method,key):
    values=[];groups={}
    for r in rows:
        sample=r['samples'][coordinate];value=None
        if not sample['outside_four_neighbors']:
            if method is None:value=sample.get(key)
            else:value=sample[method].get(key)
        values.append(value);groups.setdefault(r['track'],[]).append(value)
    per_track=[]
    for tid,v in sorted(groups.items()):
        good=[x for x in v if x is not None];per_track.append(float(np.mean(good)) if good else None)
    return dict(observation_equal=stats(values),track_equal_mean_of_valid_observations=stats(per_track))


def summarize(rows):
    result=dict(fixed_observations=len(rows),fixed_tracks=len({r['track'] for r in rows}),
                target_Z_m=stats([r['target_Z_m'] for r in rows]),
                trackmax_LOO_world_mm=stats([r['trackmax_LOO_world_mm'] for r in rows]),
                joint16_envelope_gap_mm=stats([r['joint16_envelope_gap_mm'] for r in rows]),
                joint16_gap_exceeds_LOO_reference_report_only=sum(r['joint16_gap_exceeds_LOO_reference'] is True for r in rows),coordinates={})
    for coordinate in ['observed','projected']:
        item=dict(outside_count=sum(r['samples'][coordinate]['outside_four_neighbors'] for r in rows))
        for method in ['expected','median']:
            item[method]={key:equal_weight(rows,coordinate,method,key) for key in ['signed_residual_mm','absolute_residual_mm','range_mm','target_gap_to_four_neighbor_range_mm']}
        item['alpha']=equal_weight(rows,coordinate,None,'alpha')
        item['EM_abs_difference_mm']=equal_weight(rows,coordinate,None,'EM_abs_difference_mm')
        result['coordinates'][coordinate]=item
    return result


def selftest():
    yy,xx=np.mgrid[:400,:400];plane=1+xx*.001+yy*.002
    uv=(np.array([100.25,120.75])+.5)*2
    result=bilinear(dict(expected=plane,median=plane),np.ones_like(plane),uv)
    assert np.allclose(result['array_xy'],[100.25,120.75])
    assert abs(result['expected']['interpolated']-(1+100.25*.001+120.75*.002))<1e-12
    jump=np.full((400,400),2.);jump[:,101:]=4.
    r=bilinear(dict(expected=jump,median=jump),np.ones_like(jump),uv)
    assert r['expected']['range_mm']==2000 and 2<r['expected']['interpolated']<4
    missing=plane.copy();missing[120,100]=0
    r=bilinear(dict(expected=missing,median=plane),np.ones_like(plane),uv)
    assert r['expected']['interpolated'] is None and r['expected']['valid_neighbors']==3
    assert stats([1,None,3])['fixed_count']==3 and stats([1,None,3])['missing_count']==1
    assert bilinear(dict(expected=plane,median=plane),np.ones_like(plane),[.5,.5])['outside_four_neighbors']
    camera_point=np.array([1.,2.,3.]);assert camera_point[2]!=np.linalg.norm(camera_point)
    return dict(status='PASS',native800_to_array400_half_pixel=True,bilinear_plane_exact=True,
                discontinuity_recorded=True,no_missing_neighbor_renormalization=True,fixed_missing_denominator=True,cameraZ_not_ray_distance=True)


def run():
    assert socket.gethostname().split('.')[0]=='ushas'
    lock_path=CACHE/'membership_lock.json';assert sha(lock_path)==LOCK_SHA
    lock=read(lock_path);assert lock['GT_read'] is False and lock['optimizer_created'] is False
    quant_path=PARENT/'independent_CPU_review_v1/quantitative_integrity.json';assert sha(quant_path)==QUANT_SHA
    quant=read(quant_path);assert quant['accepted_tracks']==115 and quant['fit_inlier_observations']==600
    tracks_path=PARENT/'frozen_tracks_before_held.json';assert sha(tracks_path)==TRACK_SHA
    tracks=read(tracks_path);held=read(PARENT/'held_predictions_all.json');cameras=read(PARENT/'cameras.json')
    assert len(tracks)==115 and len(held)==13 and len(cameras)==135
    inputs={str(p):sha(p) for p in [lock_path,quant_path,tracks_path,PARENT/'held_predictions_all.json',PARENT/'cameras.json',PARENT/'held_isolation_boundary.json']}
    for p,digest in inputs.items():
        if p in quant['input_SHA']:assert digest==quant['input_SHA'][p]
    feature_cache={}
    for camera in cameras:
        path=PARENT/'features'/f'{camera["name"]}.npz';assert sha(path)==quant['input_SHA'][str(path)]
        inputs[str(path)]=sha(path)
        with np.load(path) as data:feature_cache[camera['name']]=data['uv'].copy()
    # 只读原全图缓存，不读会员数组、plane、Stage1身份或N/H候选。
    assert len(lock['baseline_geometry_sha256'])==135
    for p,digest in lock['baseline_geometry_sha256'].items():assert sha(p)==digest;inputs[p]=digest
    model=ROOT/'output/1017-09-LH-static-jumpingjacks_DL_V2-T08-仅跳过15000透明度重置16k/model_mlp'
    for relative in ['unknown_light_training.json','static_dl_training.json','point_cloud/iteration_16000/point_cloud.ply','deform/iteration_16000/deform.pth']:
        p=model/relative;digest=sha(p);assert digest==lock['input_sha256'][str(p)];inputs[str(p)]=digest
    assert read(model/'unknown_light_training.json')['gt_supervision'] is False
    config=read(model/'static_dl_training.json')['args']
    assert config['resolution']==2 and config['depth_ratio']==1 and not config['white_background']
    provenance_path=PARENT/'frozen_config_and_inputs.json'
    assert sha(provenance_path)=='72c4a59319235b0627dc1016d8ae83a97ac02ab442293ba11774ff427ed27b3c'
    provenance=read(provenance_path);inputs[str(provenance_path)]=sha(provenance_path)
    transforms=Path(config['source_path'])/'transforms_train.json'
    assert sha(transforms)==lock['input_sha256'][str(transforms)]==provenance['inputs'][str(transforms)]
    inputs[str(transforms)]=sha(transforms)
    for p,digest in lock['native_runtime'].items():assert sha(p)==digest;inputs[p]=digest
    for p,digest in lock['input_sha256'].items():
        if p.endswith('/scripts/T08_plane_counterfactual.py') or p.endswith('/gaussian_renderer/__init__.py'):
            assert sha(p)==digest;inputs[p]=digest
    inputs[str(Path(__file__).resolve())]=sha(__file__)
    write(OUT/'frozen_inputs_and_source.json',dict(inputs=inputs,command=shlex.join([sys.executable,*sys.argv]),
          model=str(model),iteration=16000,cache_lock_sha256=LOCK_SHA,parent_quant_sha256=QUANT_SHA,
          GT_read=False,model_tensor_loaded=False,source='原115静态RGB三角化位置，对T08原全图缓存'))
    write(OUT/'selftest.json',selftest())
    raw=[];byid={t['id']:t for t in tracks}
    for t in tracks:
        for j in t['inlier_indices']:
            view,key=t['observations'][j];raw.append(dict(split='fit',track=t['id'],view=view,key=key,uv=feature_cache[cameras[view]['name']][key]))
    for r in held:raw.append(dict(split='held',track=r['track'],view=r['view'],key=r['key'],uv=np.array(r['observed_uv'])))
    assert sum(r['split']=='fit' for r in raw)==600 and sum(r['split']=='held' for r in raw)==13
    rows=[]
    for view in sorted({r['view'] for r in raw}):
        camera=cameras[view];E=np.array(camera['E']);K=np.array(camera['K'])
        path=CACHE/'baseline_geometry'/f'{camera["name"]}.npz'
        with np.load(path) as cache:
            alpha=cache['alpha'];depths={k:cache[k] for k in ['expected','median']}
        assert alpha.shape==(400,400) and all(a.shape==(400,400) for a in depths.values())
        for r in [r for r in raw if r['view']==view]:
            t=byid[r['track']];X=np.array(t['xyz']);pc=E[:3,:3]@X+E[:3,3];assert pc[2]>0
            projected=K@pc;projected=projected[:2]/projected[2]
            maximum=max(x['displacement_mm'] for x in t['loo'] if x['valid'])
            own=[x['displacement_mm'] for x in t['loo'] if x['valid'] and x['held_view']==view]
            row=dict(split=r['split'],track=r['track'],view=view,frame=camera['name'],key=r['key'],target_Z_m=float(pc[2]),
                observed_uv800=np.asarray(r['uv']).tolist(),projected_uv800=projected.tolist(),
                original_reprojection_error_px=float(np.linalg.norm(np.asarray(r['uv'])-projected)),
                own_LOO_world_mm=own[0] if own and r['split']=='fit' else None,
                trackmax_LOO_world_mm=maximum,trackmax_LOO_cameraZ_amplitude_reference_mm=float(np.linalg.norm(E[2,:3])*maximum),samples={})
            all_neighbors=[]
            for name,uv in [('observed',r['uv']),('projected',projected)]:
                sample=bilinear(depths,alpha,uv)
                if not sample['outside_four_neighbors']:
                    for method in ['expected','median']:
                        d=sample[method];value=d['interpolated']
                        d['signed_residual_mm']=(value-pc[2])*1000 if value is not None else None
                        d['absolute_residual_mm']=abs(d['signed_residual_mm']) if value is not None else None
                        d['target_gap_to_four_neighbor_range_mm']=max(d['minimum']-pc[2],pc[2]-d['maximum'],0)*1000 if value is not None else None
                        if d['four_finite_positive']:all_neighbors.extend(d['neighbors'])
                row['samples'][name]=sample
            row['joint16_envelope_gap_mm']=float(max(min(all_neighbors)-pc[2],pc[2]-max(all_neighbors),0)*1000) if len(all_neighbors)==16 else None
            row['joint16_gap_exceeds_LOO_reference']=bool(row['joint16_envelope_gap_mm']>row['trackmax_LOO_cameraZ_amplitude_reference_mm']) if row['joint16_envelope_gap_mm'] is not None else None
            rows.append(row)
        print(json.dumps(dict(stage='cached_depth_sampling',frame=camera['name'],rows_so_far=len(rows))),flush=True)
    assert len(rows)==613
    write(OUT/'all613_observations.json',rows)
    pertrack=[]
    for tid in sorted(byid):
        group=[r for r in rows if r['track']==tid]
        pertrack.append(dict(track=tid,fit=summarize([r for r in group if r['split']=='fit']),held=summarize([r for r in group if r['split']=='held'])))
    write(OUT/'all115_track_summaries.json',pertrack)
    result=dict(status='COMPLETED_CPU_RESIDUAL_DIAGNOSTIC_AWAITING_REVIEW',fit=summarize([r for r in rows if r['split']=='fit']),
        held=summarize([r for r in rows if r['split']=='held']),cache_maps_verified=135,sampled_views=len({r['view'] for r in raw}),
        GT_read=False,model_tensor_loaded=False,GPU_used=False,new_render=False,new_matching=False,new_triangulation=False,
        optimizer_updates=0,training_admission=False,no_observation_or_track_reselection=True,
        limitations=['缺失保留原分母，未重归一化四邻点','LOO仅条件敏感性幅值参考，不是标准差或置信区间','Alpha和两深度差只报告，不作为筛点','13held是成功描述子子集，不能外推全身','包络/LOO比较是诊断，不是训练会员资格'])
    assert all(sha(p)==digest for p,digest in inputs.items())
    result['all_inputs_and_source_before_after_SHA_exact']=True
    write(OUT/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:print(json.dumps(selftest(),ensure_ascii=False));return
    OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir();(OUT/'frozen_source.py').write_bytes(Path(__file__).read_bytes())
    with (OUT/'logs/diagnostic.log').open('w') as log:
        streams=sys.stdout,sys.stderr
        class Tee:
            def __init__(self,s):self.s=s
            def write(self,t):self.s.write(t);log.write(t);return len(t)
            def flush(self):self.s.flush();log.flush()
        sys.stdout,sys.stderr=Tee(streams[0]),Tee(streams[1])
        try:
            write(OUT/'state.json',dict(status='RUNNING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            run();write(OUT/'state.json',dict(status='COMPLETED_CPU_AWAITING_REVIEW',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        except BaseException as e:
            import traceback
            failure=dict(status='FAILED',type=type(e).__name__,message=str(e),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(OUT/'FAILED.json',failure);write(OUT/'state.json',failure);traceback.print_exc();raise
        finally:sys.stdout,sys.stderr=streams


if __name__=='__main__':main()
