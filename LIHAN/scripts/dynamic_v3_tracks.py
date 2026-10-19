"""真实TRAIN跨时刻观测、局部切平面核验与独立PS几何约束。"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.general_utils import build_rotation
from LIHAN.scripts.dynamic_v3_surface import native_normal_operator
from LIHAN.scripts.dynamic_v3_ps_math import estimate_normals


def pick_surface_ids(model,maximum=2048,seed=1019):
    """固定canonical体素，每体素优先高opacity；动态/静态各半，不读取图像颜色。"""
    xyz=model.pc.get_xyz.detach().cpu().numpy()
    lower,upper=np.quantile(xyz,[.001,.999],axis=0)
    cells=np.floor((xyz-lower)/np.maximum(upper-lower,1e-6)*63).clip(0,63).astype(np.int64)
    keys=cells[:,0]+64*cells[:,1]+4096*cells[:,2]
    opacity=model.pc.get_opacity[:,0].detach().cpu().numpy()
    supported=model.normal_sign_valid.detach().cpu().numpy()
    order=np.argsort(-opacity,kind='stable');order=order[supported[order]]
    _,first=np.unique(keys[order],return_index=True);pool=order[first]
    gate=model.pc.get_binary_feature()[:,0].detach().cpu().numpy()
    rng=np.random.default_rng(seed);chosen=[]
    for dynamic in (True,False):
        part=pool[(gate[pool]>.5)==dynamic]
        if len(part)>maximum//2:part=rng.choice(part,maximum//2,replace=False)
        chosen.extend(part.tolist())
    if len(chosen)<min(maximum,len(pool)):
        remaining=np.setdiff1d(pool,np.array(chosen,dtype=np.int64))
        take=min(maximum-len(chosen),len(remaining))
        chosen.extend(rng.choice(remaining,take,replace=False).tolist())
    assert chosen,'TRAIN可见canonical表面为空'
    return torch.tensor(sorted(chosen),device=model.pc.get_xyz.device,dtype=torch.long)


def sample_correspondences(model,camera,geometry,package,ids):
    """相机可见度仅用于对应；与光源BVH可见度独立。"""
    with torch.no_grad():
        xyz=geometry.get_xyz[ids]
        homogeneous=torch.cat([xyz,torch.ones_like(xyz[:,:1])],-1)
        clip=homogeneous@camera.full_proj_transform
        h,w=camera.image_height,camera.image_width
        ndc=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
        u=((ndc[:,0]+1)*w/2-.5).round().long()
        v=((ndc[:,1]+1)*h/2-.5).round().long()
        inside=(clip[:,3]>0)&(u>=1)&(u<w-1)&(v>=1)&(v<h-1)
        u,v=u.clamp(0,w-1),v.clamp(0,h-1)
        depth=(homogeneous@camera.world_view_transform)[:,2]
        error=(depth-package['surf_depth'][0,v,u]).abs()/depth.clamp_min(1e-6)
        alpha=camera.gt_alpha_mask.cuda()[0]
        interior=-F.max_pool2d(-alpha[None,None],3,stride=1,padding=1)[0,0]
        normal=build_rotation(geometry.get_rotation)[ids,:,2]*model.normal_sign[ids]
        rendered=F.normalize(package['rend_normal'][:,v,u].T,dim=-1)
        agreement=(normal*rendered).sum(-1)
        facing=(normal*F.normalize(camera.camera_center-xyz,dim=-1)).sum(-1)
        valid=inside&(error<.01)&(interior[v,u]>.99)&(package['rend_alpha'][0,v,u]>.95)
        valid &= (agreement>.8)&(facing>.1)
        before=int(valid.sum())
        good=torch.where(valid)[0].cpu().numpy()
        ranking=good[np.argsort(error[good].cpu().numpy(),kind='stable')]
        pixels=(v*w+u).cpu().numpy()
        _,unique=np.unique(pixels[ranking],return_index=True)
        valid.zero_();valid[torch.as_tensor(ranking[unique],device=ids.device)]=True
        observed=camera.observed_linear.to(ids.device)[:,v,u].T
        saturated=camera.observed_saturation_mask.to(ids.device)[:,v,u].T
        return dict(valid=valid,pixels=torch.stack([u,v],-1),depth_error=error,
            normal_agreement=agreement,facing=facing,world_normals=normal,
            observed=observed,saturated=saturated,before_unique=before)


def transport_audit(model,camera,ids,geometry):
    """固定feature延拓的XYZ Jacobian与有限surfel弦；不把独立旋转头冒充它。"""
    with torch.enable_grad():
        xyz=model.pc.get_xyz[ids].detach().clone().requires_grad_(True)
        feature=model.pc.get_binary_feature()[ids].detach()
        time=camera.fid.reshape(1,1).expand(len(ids),1)
        values=model.deform.step(xyz,time,iteration=35000,feature=feature,camera_center=camera.camera_center)
        world=xyz+values['d_xyz']
        jacobian=torch.stack([torch.autograd.grad(world[:,k].sum(),xyz,retain_graph=k<2)[0].detach() for k in range(3)],1)
    with torch.no_grad():
        basis=build_rotation(model.pc.get_rotation)[ids]
        u=(jacobian@basis[:,:,0,None]).squeeze(-1)
        v=(jacobian@basis[:,:,1,None]).squeeze(-1)
        transported=F.normalize(torch.cross(u,v,dim=-1),dim=-1)*model.normal_sign[ids]
        actual=build_rotation(geometry.get_rotation)[ids,:,2]*model.normal_sign[ids]
        cosine=(transported*actual).sum(-1)
        singular=torch.linalg.svdvals(jacobian)
        condition=singular[:,0]/singular[:,-1].clamp_min(1e-8)
        determinant=torch.linalg.det(jacobian)
        # 实际surfel尺度上的有限弦，避免只在无穷小处吻合。
        scales=model.pc.get_scaling[ids].clamp_min(1e-4)
        probe=torch.cat([xyz.detach()+basis[:,:,0]*scales[:,:1],xyz.detach()-basis[:,:,0]*scales[:,:1],
                         xyz.detach()+basis[:,:,1]*scales[:,1:2],xyz.detach()-basis[:,:,1]*scales[:,1:2]],0)
        ds=model.deform.step(probe,camera.fid.reshape(1,1).expand(len(probe),1),iteration=35000,
            feature=feature.repeat(4,1),camera_center=camera.camera_center)
        displaced=probe+ds['d_xyz'];a,b,c,d=displaced.chunk(4)
        exact_u=(a-b)/2;exact_v=(c-d)/2
        linear_u=u*scales[:,:1];linear_v=v*scales[:,1:2]
        chord_error=torch.maximum((exact_u-linear_u).norm(dim=-1)/exact_u.norm(dim=-1).clamp_min(1e-8),
                                  (exact_v-linear_v).norm(dim=-1)/exact_v.norm(dim=-1).clamp_min(1e-8))
        valid=torch.isfinite(jacobian).all((-1,-2))&(determinant>0)&(condition<20)
        valid &= (singular[:,-1]>.1)&(singular[:,0]<10)&(cosine>math.cos(math.radians(20)))&(chord_error<.2)
        world_basis=build_rotation(geometry.get_rotation)[ids]
        initialization_map=world_basis@basis.transpose(-1,-2)
        return dict(valid=valid,jacobian=jacobian,condition=condition,determinant=determinant,
            head_tangent_cosine=cosine,chord_relative_error=chord_error,
            initialization_normal_map=initialization_map,
            rotation_delta=geometry.values['d_rotation'][ids].detach())


def build_observations(model,cameras,output,ids=None,maximum=2048,transport=True):
    from gaussian_renderer import render
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    ids=pick_surface_ids(model,maximum) if ids is None else ids.to(model.pc.get_xyz.device)
    assert ids.ndim==1 and len(ids)>0 and len(torch.unique(ids))==len(ids)
    arrays={k:[] for k in ('observed','saturated','valid','pixels','depth_error','world_normals','positions')}
    extra={k:[] for k in ('transport_valid','jacobian','condition','determinant','head_tangent_cosine',
                         'chord_relative_error','initialization_normal_map','rotation_delta')}
    frames=[]
    for camera in cameras:
        camera.load2device()
        with torch.no_grad():
            geometry=model.effective(camera)
            package=render(camera,geometry,model.pipe,ids.new_zeros(3,dtype=torch.float32),0.,0.,0.,override_color=torch.zeros_like(geometry.get_xyz))
            sample=sample_correspondences(model,camera,geometry,package,ids)
            sample['positions']=geometry.get_xyz[ids]
            for key in arrays:arrays[key].append(sample[key].detach().cpu())
        if transport:
            check=transport_audit(model,camera,ids,geometry)
            for key in extra:
                extra[key].append(check['valid' if key=='transport_valid' else key].detach().cpu())
        frames.append(dict(frame=int(camera.image_name_train_light.split('_')[-1]),time=float(camera.fid),
                           camera_valid=int(sample['valid'].sum()),before_unique=sample['before_unique'],
                           transport_valid=int((sample['valid']&check['valid']).sum()) if transport else None))
        print(json.dumps(frames[-1]),flush=True)
        camera.load2device('cpu')
        del package,geometry,sample
    data={key:torch.stack(value) for key,value in arrays.items()}
    if transport:data.update({key:torch.stack(value) for key,value in extra.items()})
    data.update(ids=ids.cpu(),frames=frames,canonical_xyz=model.pc.get_xyz[ids].detach().cpu(),
        canonical_raw_quaternion=model.pc._rotation[ids].detach().cpu(),sign=model.normal_sign[ids].detach().cpu(),
        dynamic=(model.pc.get_binary_feature()[ids,0]>.5).detach().cpu())
    torch.save(data,out/'observations.pt')
    summary=dict(status='PASS_OBSERVATION_CACHE',gt_read=False,source_split='TRAIN',surfaces=len(ids),frames=len(cameras),
        camera_valid_observations=int(data['valid'].sum()),
        transport_valid_observations=int((data['valid']&data['transport_valid']).sum()) if transport else None,
        observation_sha256=hashlib.sha256((out/'observations.pt').read_bytes()).hexdigest(),
        scope='真实TRAIN图像采样与当前几何对应；尚非PS目标或几何准确率')
    summary['by_model_gate']={}
    for name,dynamic in [('dynamic',True),('static',False)]:
        chosen=data['dynamic']==dynamic
        camera_valid=data['valid'][:,chosen]
        item=dict(surfaces=int(chosen.sum()),camera_valid_observations=int(camera_valid.sum()),
                  tracks_with_six_camera_observations=int((camera_valid.sum(0)>=6).sum()))
        if transport:
            joined=camera_valid&data['transport_valid'][:,chosen]
            item.update(transport_valid_observations=int(joined.sum()),
                        tracks_with_six_transport_observations=int((joined.sum(0)>=6).sum()))
            for key in ['condition','head_tangent_cosine','chord_relative_error']:
                values=data[key][:,chosen][camera_valid]
                item[key]=None if not values.numel() else dict(min=float(values.min()),median=float(values.median()),max=float(values.max()))
        summary['by_model_gate'][name]=item
    (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return data


def build_ps_targets(model,cameras,observations,output,steps=200):
    """A使用detach的EL渲染残差；B使用原始观测。V始终来自各时刻完整几何。"""
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    device=model.pc.get_xyz.device;ids=observations['ids'].to(device)
    original=observations['observed'].to(device)
    valid=(observations['valid']&observations['transport_valid']).to(device)
    visibility=[];environment=[];directions=[]
    with torch.no_grad():
        for index,camera in enumerate(cameras):
            assert int(camera.image_name_train_light.split('_')[-1])==observations['frames'][index]['frame']
            camera.load2device()
            package=model.render(camera,components=model.path=='A')
            visibility.append(package['surface_components']['visibility'][ids,0].cpu())
            directions.append(model.radiance.lights(camera.fid).cpu())
            if model.path=='A':
                u,v=observations['pixels'][index].T.to(device)
                environment.append(package['environment_linear'][:,v,u].T.cpu())
            camera.load2device('cpu')
    observed=original-torch.stack(environment).to(device) if model.path=='A' else original
    vis=torch.stack(visibility).to(device);light=torch.stack(directions).to(device)
    delta=observations['rotation_delta'].to(device)
    operator=native_normal_operator(observations['canonical_raw_quaternion'].to(device),delta,observations['sign'].to(device))
    result=estimate_normals(observed,light,observations['initialization_normal_map'].to(device),vis,
        valid,observations['saturated'].to(device),float(model.radiance.intensity.detach())*(.5 if model.path=='A' else 1.),
        steps=steps,normal_operator=operator)
    # 反向180度的最小旋转不唯一，且当前局部近似不可用，拒绝而非任意选轴。
    canonical=build_rotation(observations['canonical_raw_quaternion'].to(device))[:,:,2]*observations['sign'].to(device)
    antipodal=(canonical*result['normal']).sum(-1)<-.95
    result['reliable'] &= ~antipodal;result['confidence'] *= ~antipodal
    result.update(ids=ids.detach(),raw_observed=original.detach(),direct_observed=observed.detach(),
        visibility=vis.detach(),directions=light.detach(),valid=valid.detach(),
        frame_numbers=[r['frame'] for r in observations['frames']],
        reference_raw_quaternion=observations['canonical_raw_quaternion'].to(device),
        sign=observations['sign'].to(device),source_path=model.path,
        environment_subtracted=model.path=='A',environment_error_acknowledged=model.path=='A')
    stored={key:value.detach().cpu() if isinstance(value,torch.Tensor) else value for key,value in result.items()}
    torch.save(stored,out/'targets.pt')
    dynamic=observations['dynamic'].to(device)
    summary=dict(status='PASS_TARGET_COMPUTATION_PENDING_GRADIENT' if result['reliable'].any() else 'FAILED_NO_IDENTIFIABLE_TRACKS',
        path=model.path,gt_read=False,target_sha256=hashlib.sha256((out/'targets.pt').read_bytes()).hexdigest(),
        total_tracks=len(ids),reliable_tracks=int(result['reliable'].sum()),
        dynamic_reliable_tracks=int((result['reliable']&dynamic).sum()),
        static_reliable_tracks=int((result['reliable']&~dynamic).sum()),
        camera_observation_fraction=float(observations['valid'].float().mean()),
        transport_observation_fraction=float(valid.float().mean()),
        reliable_track_fraction=float(result['reliable'].float().mean()),
        confidence_mean=float(result['confidence'].mean()),targets_detached=True,
        confidence_design=result['confidence_design'],
        channel_support_max=int(result['channel_valid_observations'].max()),
        scope='观测驱动多时刻目标；未证明PS几何梯度/收益，A残差包含环境估计误差')
    (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return stored


def ps_geometry_loss(model,camera,package,targets):
    """同一canonical伪法线同时约束实际GS轴与ExpectedDepth表面，目标无反向。"""
    device=model.pc.get_xyz.device;ids=targets['ids'].to(device)
    normal_target=targets['normal'].to(device).detach()
    confidence=targets['confidence'].to(device).detach()
    geometry=package['surface_components']['geometry']
    protected=package.get('geometry_for_normal',package)
    sample=sample_correspondences(model,camera,geometry,protected,ids)
    frame=int(camera.image_name_train_light.split('_')[-1]);index=targets['frame_numbers'].index(frame)
    weight=confidence*sample['valid']*targets['valid'][index].to(device)
    current=build_rotation(model.pc.get_rotation)[ids,:,2]*model.normal_sign[ids]
    canonical=(1-(current*normal_target).sum(-1))*weight
    # 当前真实deformation头运输detach目标；目标不使用当前预测Depth法线。
    operator=native_normal_operator(targets['reference_raw_quaternion'].to(device),
        geometry.values['d_rotation'][ids].detach()[None],targets['sign'].to(device))
    world_target=operator(normal_target)[0].detach()
    u,v=sample['pixels'].T
    depth_normal=F.normalize(protected['surf_normal'][:,v,u].T,dim=-1)
    depth=(1-(depth_normal*world_target).sum(-1))*weight
    denominator=weight.sum().clamp_min(1)
    loss=(canonical.sum()+depth.sum())/(2*denominator)
    return loss,dict(active_tracks=int((weight>0).sum()),weight_sum=float(weight.sum()),
        canonical=float(canonical.sum().detach()/denominator),depth=float(depth.sum().detach()/denominator))
