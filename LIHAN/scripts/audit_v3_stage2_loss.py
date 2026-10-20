"""审计真实Stage2损失、PS拒绝链与TRAIN加权梯度；不执行优化器更新。"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex
import socket
import sys

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from gaussian_renderer import render
from LIHAN.scripts.dynamic_v3_surface import DynamicSurfaceModel,load_train_cameras
from LIHAN.scripts.dynamic_v3_train import photo_objective,common_geometry
from LIHAN.scripts.dynamic_v3_tracks import ps_geometry_loss
from utils.no_gt_guard import install_no_gt_guard


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def quantile(x):
    x=x[torch.isfinite(x)].float()
    return None if not x.numel() else dict(min=float(x.min()),median=float(x.median()),p90=float(torch.quantile(x,.9)),max=float(x.max()))


def support_chain(obs,target):
    valid=obs['valid'];jac=obs['jacobian'];singular=torch.linalg.svdvals(jac)
    checks=dict(finite=torch.isfinite(jac).all((-1,-2)),positive_determinant=obs['determinant']>0,
        condition_below20=obs['condition']<20,stretch_bounds=(singular[...,-1]>.1)&(singular[...,0]<10),
        rotation_matches_xyz=obs['head_tangent_cosine']>math.cos(math.radians(20)),
        finite_chord_error_below02=obs['chord_relative_error']<.2)
    joined=valid&obs['transport_valid'];assert torch.equal(joined,target['valid'])
    result={}
    for label,gate in [('dynamic',obs['dynamic']),('static',~obs['dynamic'])]:
        base=valid&gate[None];current=base.clone();sequential=[]
        for key,check in checks.items():
            before=int(current.sum());current &= check
            sequential.append(dict(check=key,before=before,after=int(current.sum()),removed=before-int(current.sum())))
        assert torch.equal(current,joined&gate[None])
        channel=target['valid'][...,None]&(~obs['saturated'])&(target['direct_observed']>=0)&(target['visibility'][...,None]>.2)
        confid=target['confidence']*gate
        result[label]=dict(tracks=int(gate.sum()),camera_observations=int(base.sum()),
            camera_tracks_six=int(((base.sum(0)>=6)&gate).sum()),transport_observations=int(current.sum()),
            transport_tracks_six=int(((current.sum(0)>=6)&gate).sum()),
            reliable_tracks=int((target['reliable']&gate).sum()),confidence_sum=float(confid.sum()),
            sequential_transport_rejection=sequential,
            independent_failure_counts={key:int((base&~check).sum()) for key,check in checks.items()},
            condition=quantile(obs['condition'][base]),rotation_cosine=quantile(obs['head_tangent_cosine'][base]),
            chord_error=quantile(obs['chord_relative_error'][base]),
            camera_channel_saturated_fraction=float(obs['saturated'][base].float().mean()) if base.any() else None,
            negative_direct_residual_fraction=float((target['direct_observed'][base]<0).float().mean()) if base.any() else None,
            usable_channels_after_transport=int((channel&gate[None,:,None]).sum()),
            reliable_ids=obs['ids'][target['reliable']&gate].tolist())
    return result


def main(args):
    assert socket.gethostname()=='ushas'
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[2];experiment=args.experiment.resolve();out=args.output.resolve()
    assert out.is_relative_to(root/'output/smoke_test')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'audit.json').exists()
    marker=json.loads((experiment/'training_finished.json').read_text())
    assert marker['actual_optimizer_updates']==8100
    source=Path(marker['source']);folder=experiment/'model_mlp';parent=Path(marker['parent']);stats_path=Path(marker['stats'])
    inputs={}
    def bind(p):p=Path(p).resolve();inputs[str(p)]=sha(p);return p
    def read(p):return json.loads(bind(p).read_text())
    for path,digest in marker['source_sha256'].items():assert sha(bind(path))==digest,path
    for name in ['dynamic_v3_train.py','dynamic_v3_surface.py','dynamic_v3_tracks.py','dynamic_v3_ps_math.py']:
        recorded=[p for p in marker['source_sha256'] if Path(p).name==name]
        assert len(recorded)==1 and sha(Path(__file__).with_name(name))==marker['source_sha256'][recorded[0]]
    install_no_gt_guard([source,folder,parent,stats_path,out]+[experiment/f'ps_refresh_{s}' for s in [1350,4050,5400,6750]])
    config=marker['config'];rows=[json.loads(line) for line in bind(experiment/'train_audit.jsonl').read_text().splitlines()]
    support={};last_observation=None;last_target=None
    for step in [1350,4050,5400,6750]:
        op=bind(experiment/f'ps_refresh_{step}/observations/observations.pt');tp=bind(experiment/f'ps_refresh_{step}/targets/targets.pt')
        obs=torch.load(op,map_location='cpu',weights_only=False);target=torch.load(tp,map_location='cpu',weights_only=False)
        support[str(step)]=support_chain(obs,target)
        last_observation,last_target=obs,target
        print(json.dumps(dict(support_step=step,dynamic=support[str(step)]['dynamic']['reliable_tracks'],static=support[str(step)]['static']['reliable_tracks'])),flush=True)
    obs,targets=last_observation,last_target
    dynamic=obs['dynamic'];supported=dynamic&targets['reliable']&(targets['confidence']>0)
    valid_dynamic_frames=[n for i,n in enumerate(targets['frame_numbers']) if (targets['valid'][i]&supported).any()]
    frames={1,81,149}
    if valid_dynamic_frames:frames.add(valid_dynamic_frames[len(valid_dynamic_frames)//2])
    frames=sorted(frames);cameras=load_train_cameras(source,frame_ids=frames)
    for step in [0,8100]:
        receipt=read(folder/f'dynamic_v3/iteration_{step}/receipt.json')
        for rel,digest in receipt['checkpoint_sha256'].items():assert sha(bind(folder/rel))==digest
    initial=torch.load(folder/'dynamic_v3/iteration_0/state.pt',map_location='cpu',weights_only=False)
    saved=torch.load(folder/'dynamic_v3/iteration_8100/state.pt',map_location='cpu',weights_only=False)
    model=DynamicSurfaceModel.load(folder,8100);stats=read(stats_path);ids=targets['ids'].cuda()
    ply=PlyData.read(bind(parent/'point_cloud/iteration_35000/point_cloud.ply'))['vertex'].data
    anchor=torch.from_numpy(np.stack([ply[k] for k in ['x','y','z']],-1).copy()).cuda()
    neighbors=cKDTree(anchor.cpu().numpy()).query(anchor[ids].cpu().numpy(),k=5)[1][:,1:]
    neighbors=torch.as_tensor(neighbors,device='cuda');reference=(anchor[neighbors]-anchor[ids,None]).norm(dim=-1).clamp_min(1e-5)
    extent=float((torch.quantile(anchor,.99,dim=0)-torch.quantile(anchor,.01,dim=0)).norm())
    parent_deform=copy.deepcopy(model.deform.deform)
    parent_deform.load_state_dict(torch.load(bind(parent/'deform/iteration_35000/deform.pth'),map_location='cuda',weights_only=False))
    parent_deform.eval().requires_grad_(False)
    named=[('xyz',model.pc._xyz),('rotation',model.pc._rotation),('scale',model.pc._scaling),('opacity',model.pc._opacity),
           ('rho',model.radiance.raw_rho),('D',model.radiance.raw_intensity),('SH',model.radiance.environment_coefficients),('light',model.radiance.lights.raw_directions)]
    named += [('deform.'+n,p) for n,p in model.deform.deform.named_parameters() if not n.startswith('gaussian_color')]
    for _,p in named:p.requires_grad_(True)
    params=[p for _,p in named];output_rows=[]
    for camera in cameras:
        camera.load2device();package=model.render(camera,components=True)
        assert not package['surface_components']['visibility'].requires_grad
        pred=package['render_linear'];target=camera.observed_linear.cuda();alpha=camera.gt_alpha_mask.cuda();sat=camera.observed_saturation_mask.cuda()
        fg=(alpha>.5).float();bg=1-fg;residual=pred-target
        error=(torch.where(sat,residual.clamp_max(0),residual).square()+.02**2).sqrt()-.02
        lum=(target*target.new_tensor([.2126,.7152,.0722])[:,None,None]).sum(0,keepdim=True)
        priority=torch.sigmoid((lum-stats['q_bright'])/.02)
        terms=dict(fg_rgb=(error*fg).sum()/(3*fg.sum()).clamp_min(1),
            bg_rgb=.1*(error*bg).sum()/(3*bg.sum()).clamp_min(1),
            bright_rgb=.1*(error*fg*priority).sum()/(3*fg.sum()).clamp_min(1),
            alpha=.1*F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6,1-1e-6),alpha))
        photo=sum(terms.values());assert float((photo-photo_objective(package,camera,stats)).abs())<1e-7
        normal,distortion=common_geometry(package,camera);ps,ps_info=ps_geometry_loss(model,camera,package,targets)
        ps_groups={};gate_infos={}
        for label,gate in [('dynamic',dynamic),('static',~dynamic)]:
            masked=dict(targets,confidence=targets['confidence']*gate)
            value,info=ps_geometry_loss(model,camera,package,masked)
            ps_groups[label]=value*max(info['weight_sum'],1)/max(ps_info['weight_sum'],1);gate_infos[label]=info
        assert float((sum(ps_groups.values())-ps).abs())<1e-6
        edge=(model.pc.get_xyz[neighbors]-model.pc.get_xyz[ids,None]).norm(dim=-1).clamp_min(1e-5)
        local=torch.log(edge/reference).clamp(-2,2).square().mean()+((model.pc.get_xyz-anchor)/extent).square().mean()
        with torch.no_grad():
            original=parent_deform(model.pc.get_xyz.detach(),camera.fid.reshape(1,1).expand(len(anchor),1),feature=model.pc.get_binary_feature(),camera_center=camera.camera_center)
        local=local+(package['surface_components']['geometry'].values['d_xyz']-original['d_xyz']).square().mean()/extent**2
        directions=F.normalize(model.radiance.lights.raw_directions,dim=-1)
        delta=directions-F.normalize(initial['radiance']['lights.raw_directions'].cuda(),dim=-1)
        light=.1*delta.square().sum(-1).mean()+.01*(delta[1:]-delta[:-1]).square().sum(-1).mean()
        objectives=dict(photo=photo,normal=.02*normal,distortion=1000*distortion,local=.001*local,
                        ps=.02*ps,light=light,ps_dynamic=.02*ps_groups['dynamic'],ps_static=.02*ps_groups['static'])
        total=sum(objectives[k] for k in ['photo','normal','distortion','local','ps','light']);objectives['total']=total
        gradient_reports={};photo_grads=None
        for name,value in objectives.items():
            grads=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True) if value.requires_grad else [None]*len(params)
            if name=='photo':photo_grads=[None if g is None else g.detach().clone() for g in grads]
            report={}
            for group in ['xyz','rotation','scale','opacity','rho','D','SH','light','deformation']:
                selected=[i for i,(n,_) in enumerate(named) if (n.startswith('deform.') if group=='deformation' else n==group)]
                norm2=0.;reference2=0.;dot=0.;exists=False
                for i in selected:
                    if grads[i] is None:continue
                    g=grads[i];assert torch.isfinite(g).all();exists=True;norm2+=float(g.square().sum())
                    if photo_grads[i] is not None:
                        reference2+=float(photo_grads[i].square().sum());dot+=float((g*photo_grads[i]).sum())
                report[group]=None if not exists else dict(norm=math.sqrt(norm2),cosine_to_photo=dot/math.sqrt(norm2*reference2) if norm2*reference2>0 else None)
            gradient_reports[name]=report
        with torch.no_grad():
            gate=model.pc.get_binary_feature();geometry=package['surface_components']['geometry']
            gate_image=render(camera,geometry,model.pipe,gate.new_zeros(3),0.,0.,0.,override_color=torch.cat([gate,1-gate,torch.zeros_like(gate)],-1))['render']
            pixel_dynamic=(gate_image[:1]>.5*package['rend_alpha'])&(fg.bool())
            component={}
            for label,mask in [('foreground',fg.bool()),('model_dynamic',pixel_dynamic),('model_static',fg.bool()&~pixel_dynamic)]:
                mask=mask.expand_as(pred);denom=float(pred[mask].sum());valid_channel=int(mask.sum())
                component[label]=dict(channels=valid_channel,
                    environment_radiance_fraction=float(package['environment_linear'][mask].sum())/max(denom,1e-8),
                    weighted_direct_radiance_fraction=float((.5*package['direct_linear'])[mask].sum())/max(denom,1e-8),
                    target_saturated_fraction=float(sat[mask].float().mean()) if valid_channel else None,
                    photo_censored_zero_fraction=float((sat[mask]&(residual[mask]>=0)).float().mean()) if valid_channel else None)
        row=dict(frame=int(camera.image_name_train_light.split('_')[-1]),loss={k:float(v.detach()) for k,v in objectives.items()},
                 photo_terms={k:float(v.detach()) for k,v in terms.items()},weighted_gradients=gradient_reports,
                 ps=ps_info,ps_by_gate=gate_infos,render_components=component,visibility_has_gradient=False,
                 component_sum_max_error=float(package['component_sum_max_error']))
        output_rows.append(row);print(json.dumps(dict(frame=row['frame'],loss=row['loss'],ps_by_gate=gate_infos,components=component)),flush=True)
        camera.load2device('cpu');del package,objectives,total,gradient_reports,grads,photo_grads,gate_image
    log_terms=[]
    for row in rows:
        weighted=dict(photo=row['photo'],normal=.02*row['normal'],distortion=1000*row['distortion'],local=.001*row['local_stability'],ps=.02*row['ps'])
        log_terms.append(dict(step=row['step'],weighted=weighted,unlogged_light_regular_residual=row['loss']-sum(weighted.values()),lrs=row['lrs'],coverage=row['foreground_alpha_coverage']))
    for path,digest in inputs.items():assert sha(path)==digest,path
    write(out/'audit.json',dict(status='PASS_READ_ONLY_LOSS_AUDIT',model=str(folder),iteration=8100,frames=frames,
        gt_read=False,test_read=False,optimizer_steps=0,weights_unchanged=True,config=config,
        saved_lrs={g['name']:g['lr'] for g in saved['optimizer']['param_groups']},support_chain=support,
        probe=output_rows,logged_weighted_terms=log_terms,input_sha256=inputs,code_sha256=sha(__file__),
        command=shlex.join([sys.executable]+sys.argv),limitations=['只读终点少量TRAIN帧梯度，不等同完整训练轨迹、Adam更新量或因果消融。','模型gate非GT人体标签。','环境占比按线性渲染辐亮度求和，不是GT能量分解。','不改变阈值、不恢复被拒绝轨迹、不执行新训练。']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
