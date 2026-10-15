"""严格加载更新形变checkpoint的隔离GT几何评测；不调用优化器。"""
import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from utils.geometry_eval_utils import (evaluate_region,aggregate_regions,load_normal_exr,depth_normals,unit_normal,erode,file_sha256,normal_errors,summarize)
from scripts.probe_dynamic_dl_selfcal import read_config


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model','source','gt-root','gt-depth','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--checkpoints',nargs='+',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    cfg=read_config(Path(args.model)/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    ply=Path(args.model)/'point_cloud/iteration_35000/point_cloud.ply';pc.load_ply(str(ply))
    deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
    assert deform.load_weights(args.model,35000);deform.deform.eval().requires_grad_(False)
    initial={k:getattr(pc,k).detach().clone() for k in ['_xyz','_scaling','_rotation','_opacity']}
    initial_deform={k:v.detach().clone() for k,v in deform.deform.state_dict().items()}
    checkpoint_states={};sources={str(ply):file_sha256(ply)}
    for filename in args.checkpoints:
        path=Path(filename);name=path.parent.name
        if name in checkpoint_states:raise ValueError('组名重复')
        runtime=json.loads((path.parent/'runtime.json').read_text())
        for source in [ply,Path(args.model)/'deform/iteration_35000/deform.pth',Path(args.model)/'cfg_args']:
            assert file_sha256(source)==runtime['inputs'][str(source.resolve())]
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert Path(state['args']['model']).resolve()==Path(args.model).resolve()
        assert state['material_updates']==360 and state['args']['mode'] in ['joint','frozen']
        assert runtime['native']['source_sha256']==sys.modules['diff_surfel_rasterization'].provenance['source_sha256']
        assert 'deform_state_dict' in state
        checkpoint_states[name]=state;sources[str(path.resolve())]=file_sha256(path)
    metadata=json.loads((Path(args.source)/'transforms_test.json').read_text())
    infos=readCamerasFromTransforms(args.source,'transforms_test.json',cfg.train_light_folder,False)
    manifest=json.loads((Path(args.gt_depth)/'manifest.json').read_text())
    assert manifest['width']==400 and manifest['height']==400
    source_records={Path(r['file_path']).stem:r for r in metadata['frames']}
    cached=[]
    for i,info in enumerate(infos):
        camera=loadCam(cfg,i,info,1.);camera.load2device();name=camera.image_name_train_light
        record=source_records[name];number=int(name.rsplit('_',1)[1]);entry=manifest['frames'][str(number)]
        assert np.isclose(entry['time'],float(camera.fid),atol=1e-6)
        gt_path=Path(args.gt_depth)/entry['file'];assert file_sha256(gt_path)==entry['sha256']
        sources[str(gt_path.resolve())]=entry['sha256']
        with np.load(gt_path,allow_pickle=False) as data:gt={k:data[k] for k in data.files}
        c2w=np.asarray(record['transform_matrix']);assert np.allclose(c2w,gt['camera_to_world'],atol=1e-5)
        h,w=camera.image_height,camera.image_width
        intrinsics=np.array([w/(2*math.tan(camera.FoVx/2)),h/(2*math.tan(camera.FoVy/2)),w/2,h/2])
        assert (h,w)==gt['depth_z'].shape and np.allclose(intrinsics,gt['intrinsics'],atol=1e-5)
        exr=Path(args.gt_root)/'normal_exr'/f'normal_{number:04d}.exr'
        normal,valid=load_normal_exr(exr,h,w);sources[str(exr.resolve())]=file_sha256(exr)
        alpha=camera.gt_alpha_mask.detach().cpu().numpy()[0]
        common=(alpha>=.5)&gt['valid'];names=gt['object_names'].tolist()
        subject_ids=[j for j,n in enumerate(names) if n.startswith('Character_')]
        foot_ids=[j for j,n in enumerate(names) if 'Sneakers' in n or 'Socks' in n]
        assert subject_ids and foot_ids
        masks={'foreground':common,'interior':erode((alpha>=.99)&gt['valid'],2),
               'subject':common&np.isin(gt['object_id'],subject_ids),
               'platform':common&~np.isin(gt['object_id'],subject_ids),
               'feet':common&np.isin(gt['object_id'],foot_ids)}
        # 与训练一致：初始位置上的默认形变输出，iteration默认0，固定身份位移。
        ds=deform.step(initial['_xyz'],camera.fid.reshape(1,1).expand(len(initial['_xyz']),1),feature=pc.get_binary_feature(),camera_center=camera.camera_center)
        ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
        cached.append((camera,ds,c2w,intrinsics,normal,valid,gt,masks,number))
    bg=torch.zeros(3,device='cuda');pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    summary={}
    for group,state in [('default',dict(geometry=initial,deform_state_dict=initial_deform)),*checkpoint_states.items()]:
        deform.deform.load_state_dict(state['deform_state_dict'],strict=True)
        # 旧checkpoint不含opacity时恢复默认值，禁止跨组继承候选透明度。
        for key,value in initial.items():getattr(pc,key).copy_(value)
        for key,value in state['geometry'].items():getattr(pc,key).copy_(value.to('cuda'))
        folder=out/group;folder.mkdir();rows=[]
        for camera,ds,c2w,intrinsics,normal,valid,gt,masks,number in cached:
            query=deform.step(pc.get_xyz,camera.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=camera.camera_center)
            ds={k:query[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
            pipe.depth_ratio=0.;package=render(camera,pc,pipe,bg,**ds,override_color=torch.zeros_like(pc.get_xyz))
            expected=package['surf_depth'][0].cpu().numpy();alpha=package['rend_alpha'][0].cpu().numpy()
            gsnormal=unit_normal(package['rend_normal'].permute(1,2,0).cpu().numpy())[0]
            pipe.depth_ratio=1.;median=render(camera,pc,pipe,bg,**ds,override_color=torch.zeros_like(pc.get_xyz))['surf_depth'][0].cpu().numpy()
            dn=depth_normals(expected,c2w,intrinsics)[0]
            prediction=dict(alpha=alpha,normal=gsnormal,expected=expected,median=median,depth_normal=dn)
            regions={name:evaluate_region(prediction,normal,valid,gt['depth_z'],mask,.5) for name,mask in masks.items()}
            penalties={}
            for name,mask in masks.items():
                penalties[name]={}
                for kind in ['expected','median']:
                    present=(alpha>=.5)&np.isfinite(prediction[kind])&(prediction[kind]>0)
                    errors=np.abs(prediction[kind]-gt['depth_z'])
                    penalties[name][kind]=summarize(np.where(present,errors,20.),mask)
            rows.append(dict(source_frame=number,regions=regions,depth_missing_penalty=penalties))
            np.savez_compressed(folder/f'prediction_{number:04d}.npz',**prediction,gt_depth=gt['depth_z'],gt_normal=normal,gt_valid=valid,object_id=gt['object_id'],**{'mask_'+k:v for k,v in masks.items()})
        aggregate=aggregate_regions(rows)
        for region in aggregate:
            aggregate[region]['depth_missing_penalty']={}
            for kind in ['expected','median']:
                values=[row['depth_missing_penalty'][region][kind]['mean'] for row in rows if row['depth_missing_penalty'][region][kind]['count']]
                aggregate[region]['depth_missing_penalty'][kind]=float(np.mean(values)) if values else None
            for kind in ['gs_render','depth_normal']:
                values=[row['regions'][region]['normal_metrics'][kind]['angular_with_missing_deg']['mean'] for row in rows if row['regions'][region]['normal_metrics'][kind]['angular_with_missing_deg']['count']]
                aggregate[region]['normal_metrics'][kind]['missing_penalty_frame_mean']=float(np.mean(values)) if values else None
        summary[group]=aggregate
        (folder/'metrics.json').write_text(json.dumps(dict(frames=rows,aggregate=aggregate),ensure_ascii=False,indent=2))
        print(json.dumps(dict(group=group,subject=aggregate['subject']),ensure_ascii=False),flush=True)
    assert summary['default']==summary['frozen'], '冻结基线几何未与默认精确重合'
    sources[str(Path(args.gt_depth)/'manifest.json')]=file_sha256(Path(args.gt_depth)/'manifest.json')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(args=vars(args),sources=sources,source_hash=file_sha256(Path(__file__)),
        native=sys.modules['diff_surfel_rasterization'].provenance,protocol='正确加载各组完整形变；固定15个历史测试帧，400x400；仅训练后离线评测；缺失深度固定惩罚20世界单位，缺失法线180度；不含预测Mesh'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
