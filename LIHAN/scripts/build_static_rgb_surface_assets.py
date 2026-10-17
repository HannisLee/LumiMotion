"""原115静态RGB轨迹600fit观测转固定Median像素/ray资产，无模型读入。"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import datetime
import hashlib
import json
import math
import shlex
import socket
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'output/1017-01-静态DL未知光总控'
PARENT=C/'1017-05-TRAIN静态RGB多视角三角化'
SOURCE=C/'data_train_T04_temporal08'
FROZEN=C/'snapshots/fixed_T08_skip_final_reset_v1'
OUT=C/'1017-07-TRAIN固定Median邻域资产'
TRACK_SHA='1f73f399bc0fce74762a0ced282d8c4283b0efaed9172122f1389a27901936ca'
QUANT_SHA='58ae4dbce2bf1829bf659e54b9f4f4e6125c8146792c25588bd8921f994931df'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def pixels(observed,projected):
    result=set()
    for uv in [observed,projected]:
        xy=np.asarray(uv)/2-.5;x,y=np.floor(xy).astype(int)
        assert 0<=x<399 and 0<=y<399
        result.update((int(y+dy),int(x+dx)) for dy in (0,1) for dx in (0,1))
    result=sorted(result);assert 4<=len(result)<=8
    return result


def world_rays(E,K400,pixel_yx):
    inverse=np.linalg.inv(E)
    q=np.array([[x+.5,y+.5,1.] for y,x in pixel_yx])
    camera=q@np.linalg.inv(K400).T
    assert np.allclose(camera[:,2],1,rtol=0,atol=1e-15)
    return inverse[:3,3],camera@inverse[:3,:3].T


def original_functions():
    p=FROZEN/'snapshot_manifest.json'
    assert sha(p)=='093a6ac43a432b4b3154612751976946f74708201980978e880f1e3e30c3e3a3'
    manifest=read(p);assert len(manifest)==96
    ns={'np':np,'math':math}
    for relative,names in [('utils/graphics_utils.py',['getWorld2View2']),
                           ('scene/dataset_readers.py',['blender_c2w_to_camera_extrinsics','getNerfppNorm'])]:
        assert sha(FROZEN/relative)==manifest[relative]
        tree=ast.parse((FROZEN/relative).read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes}==set(names)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),relative,'exec'),ns)
    return ns


def selftest():
    assert pixels([201.5,241.5],[201.5,241.5])==[(120,100),(120,101),(121,100),(121,101)]
    assert len(pixels([201.5,241.5],[205.5,245.5]))==8
    E=np.eye(4);E[:3,3]=[.2,-.4,.1];K=np.array([[500.,0,200.],[0,510.,200.],[0,0,1.]])
    pp=pixels([401.5,401.5],[401.5,401.5]);origin,rays=world_rays(E,K,pp)
    world=origin+rays*3.;camera=world@E[:3,:3].T+E[:3,3]
    uv=camera@K.T;uv=uv[:,:2]/uv[:,2:]
    assert np.max(np.abs(uv-np.array([[x+.5,y+.5] for y,x in pp])))<1e-10
    assert np.max(np.abs(camera[:,2]-3))<1e-12
    # 一轮135个相机的平均等于原轨迹等权，不按当帧观测数再平均。
    records=[(0,0,2.),(0,1,4.),(1,0,8.)];counts={0:2,1:1}
    per_camera=[135/2*sum(d/counts[j] for cam,j,d in records if cam==i) for i in range(135)]
    assert abs(np.mean(per_camera)-((2+8)/2+4)/2)<1e-12
    return dict(status='PASS',fixed_union4_to8=True,original_pixel_center_and_cameraZ_backprojection=True,
                uniform135_camera_correct_track_equal=True)


def run():
    assert socket.gethostname().split('.')[0]=='ushas'
    quant_path=PARENT/'independent_CPU_review_v1/quantitative_integrity.json'
    assert sha(quant_path)==QUANT_SHA;quant=read(quant_path)
    tracks_path=PARENT/'frozen_tracks_before_held.json';assert sha(tracks_path)==TRACK_SHA
    cameras_path=PARENT/'cameras.json';assert sha(cameras_path)==quant['input_SHA'][str(cameras_path)]
    tracks=read(tracks_path);cameras=read(cameras_path);assert len(tracks)==115 and len(cameras)==135
    transforms_path=SOURCE/'transforms_train.json';source_manifest_path=SOURCE/'no_gt_manifest.json'
    source_manifest=read(source_manifest_path);assert not source_manifest['gt_read'] and not source_manifest['init_weights_loaded']
    assert sha(transforms_path)==source_manifest['files']['transforms_train.json']==quant['input_SHA'][str(transforms_path)]
    transforms=read(transforms_path)
    frames=sorted(transforms['frames'],key=lambda f:int(Path(f['file_path']).stem.split('_')[-1]))
    assert len(frames)==135 and all(Path(f['file_path']).stem==c['name'] for f,c in zip(frames,cameras))
    ns=original_functions();infos=[]
    for frame,camera in zip(frames,cameras):
        R,t=ns['blender_c2w_to_camera_extrinsics'](frame['transform_matrix'])
        E=ns['getWorld2View2'](R,t)
        assert np.array_equal(E,np.array(camera['E'],dtype=np.float32));infos.append(SimpleNamespace(R=R,T=t))
    extent=float(ns['getNerfppNorm'](infos)['radius'])
    assert extent==4.290023326873779
    inputs={str(p):sha(p) for p in [quant_path,tracks_path,cameras_path,transforms_path,source_manifest_path,
           FROZEN/'snapshot_manifest.json',FROZEN/'scene/dataset_readers.py',FROZEN/'utils/graphics_utils.py',Path(__file__).resolve()]}
    features={};needed=sorted({t['observations'][j][0] for t in tracks for j in t['inlier_indices']})
    assert len(needed)==92 and all(i%9!=4 for i in needed)
    for i in needed:
        p=PARENT/'features'/f'{cameras[i]["name"]}.npz';assert sha(p)==quant['input_SHA'][str(p)];inputs[str(p)]=sha(p)
        with np.load(p) as a:features[i]=a['uv'].copy()
    targets=[];packed={}
    for i,camera in enumerate(cameras):
        E=np.array(camera['E']);K=np.diag([.5,.5,1.])@np.array(camera['K']);origin=np.linalg.inv(E)[:3,3]
        packed[camera['name']]=dict(source_index=i,fit_view=i%9!=4,E=E.tolist(),K400=K.tolist(),
                                   origin_world=origin.tolist(),observations=[])
    for target_index,t in enumerate(tracks):
        X=np.array(t['xyz']);n=len(t['inlier_indices']);assert n>=4
        targets.append(dict(id=t['id'],xyz=X.tolist(),fit_observations=n))
        for j in t['inlier_indices']:
            i,key=t['observations'][j];assert i%9!=4
            frame=packed[cameras[i]['name']];E=np.array(frame['E']);K=np.array(cameras[i]['K'])
            pc=E[:3,:3]@X+E[:3,3];assert pc[2]>0
            uv=K@pc;projected=uv[:2]/uv[2];observed=features[i][key]
            pixel_yx=pixels(observed,projected);origin,rays=world_rays(E,np.array(frame['K400']),pixel_yx)
            assert np.array_equal(origin,np.array(frame['origin_world']))
            frame['observations'].append(dict(target_index=target_index,track_id=t['id'],feature_index=key,
                 observed_uv800=observed.tolist(),projected_uv800=projected.tolist(),pixels_yx=pixel_yx,rays_world=rays.tolist()))
    assert sum(len(f['observations']) for f in packed.values())==600
    assert all(not f['observations'] for f in packed.values() if not f['fit_view'])
    pixel_counts={}
    for f in packed.values():
        for obs in f['observations']:
            n=len(obs['pixels_yx']);pixel_counts[n]=pixel_counts.get(n,0)+1
    asset=dict(version=1,method='固定最多8原生Median像素三维表面距离',gt_read=False,init_weights_loaded=False,
               parent_tracks_sha256=TRACK_SHA,parent_independent_quant_sha256=QUANT_SHA,
               held_used_as_targets=False,track_count=115,fit_observation_count=600,camera_count=135,
               camera_extent=extent,targets=targets,frames=packed,pixel_count_distribution=pixel_counts,
               weighting='135/115 * sum_current_camera distance_clipped_by_extent/extent/n_track_fit_obs',
               coefficient=.1,start='原normal激活阶段',invalid_distance='scene_camera_extent',
               inputs=inputs,generator_sha256=sha(__file__))
    assert all(sha(p)==digest for p,digest in inputs.items())
    write(OUT/'anchors.json',asset)
    result=dict(status='PASS_CPU_ASSET_CONSTRUCTION_AWAITING_INDEPENDENT',asset_sha256=sha(OUT/'anchors.json'),
                tracks=115,fit_observations=600,cameras=135,cameras_with_observations=92,held_targets=0,
                camera_extent=extent,pixel_count_distribution=pixel_counts,all_input_and_source_SHA_unchanged=True,
                GT_read=False,model_loaded=False,GPU_used=False,optimizer_updates=0)
    write(OUT/'result.json',result);write(OUT/'selftest.json',selftest());print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:print(json.dumps(selftest(),ensure_ascii=False));return
    OUT.mkdir(exist_ok=False);(OUT/'logs').mkdir();(OUT/'generator.py').write_bytes(Path(__file__).read_bytes())
    with (OUT/'logs/construction.log').open('w') as log:
        streams=sys.stdout,sys.stderr
        class Tee:
            def __init__(self,s):self.s=s
            def write(self,t):self.s.write(t);log.write(t);return len(t)
            def flush(self):self.s.flush();log.flush()
        sys.stdout,sys.stderr=Tee(streams[0]),Tee(streams[1])
        try:
            write(OUT/'launch.json',dict(command=shlex.join([sys.executable,*sys.argv]),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            write(OUT/'state.json',dict(status='RUNNING'));run()
            write(OUT/'state.json',dict(status='COMPLETED_CPU_AWAITING_INDEPENDENT',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        except BaseException as error:
            import traceback
            failure=dict(status='FAILED',type=type(error).__name__,message=str(error))
            write(OUT/'FAILED.json',failure);write(OUT/'state.json',failure);traceback.print_exc();raise
        finally:sys.stdout,sys.stderr=streams


if __name__=='__main__':main()
