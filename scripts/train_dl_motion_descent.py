"""共享材质与受限位移头真实下降短对照；未知光默认禁止GT，参考组显式标记。"""
import argparse
import copy
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts.probe_dynamic_dl_selfcal import read_config, digest_tensor, save_image
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.pbr_rasterizer import native_module
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from utils.camera_utils import loadCam
from utils.general_utils import build_rotation


def run(args, out):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = read_config(Path(args.model)/'cfg_args')
    assert not cfg.white_background
    cfg.resolution = args.resolution; cfg.load2gpu_on_the_fly = True
    light = json.loads(Path(args.lights).read_text())
    if args.gt_direction_reference:
        assert light.get('gt_direction_reference') is True and light.get('gt_read') is True and light.get('control')=='oracle'
    else:
        assert not light.get('gt_direction_reference', False) and not light.get('gt_read', False)
    (out/'used_light.json').write_text(json.dumps(light,ensure_ascii=False,indent=2))
    infos = readCamerasFromTransforms(args.source, 'transforms_train.json', cfg.train_light_folder, False)
    cameras = [loadCam(cfg, i, info, 1.) for i, info in enumerate(infos)]
    assert [c.image_name_train_light for c in cameras] == light['frames']
    times = torch.tensor([float(c.fid) for c in cameras], device='cuda')
    assert torch.allclose(times, torch.tensor(light['times'], device='cuda'), atol=1e-6)
    assert bool((times[1:]>times[:-1]).all())
    train = light['train_indices']; held = light['held_indices']
    assert train == [i for i in range(len(times)) if i%9!=4]
    assert held == [i for i in range(len(times)) if i%9==4]
    selected = train if args.frames == 0 else [train[i] for i in np.linspace(0, len(train)-1, args.frames, dtype=int)]
    validation = held
    directions = torch.tensor(light['directions'], device='cuda')
    assert directions.shape == (len(times), 3) and torch.isfinite(directions).all()
    assert torch.allclose(directions.norm(dim=-1), torch.ones(len(times), device='cuda'), atol=1e-4)
    pc = GaussianModel(cfg.sh_degree, no_binary_separation=cfg.no_binary_separation, fea_dim=cfg.hyper_dim)
    ply = Path(args.model)/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'
    deform_path = Path(args.model)/f'deform/iteration_{args.iteration}/deform.pth'
    pc.load_ply(str(ply))
    deform = DeformModel(deform_type=cfg.deform_type, is_blender=cfg.is_blender, hyper_dim=cfg.hyper_dim, pred_color=cfg.pred_color)
    assert deform.load_weights(args.model, args.iteration)
    deform.deform.eval().requires_grad_(False)
    tensors = {k:v for k,v in vars(pc).items() if isinstance(v, torch.Tensor)}
    for v in tensors.values():v.requires_grad_(False)
    geometry = {k:tensors[k] for k in ['_xyz','_scaling','_rotation','_opacity']}
    initial_geometry = {k:v.detach().clone() for k,v in geometry.items()}
    head=deform.deform.gaussian_warp
    initial_head={k:v.detach().clone() for k,v in head.state_dict().items()}
    frozen = dict(tensors)
    frozen.update({k:geometry[k] for k in ['_scaling','_rotation']})
    frozen.update({'deform.'+k:v for k,v in deform.deform.state_dict().items() if not k.startswith('gaussian_warp.')})
    if args.geometry_groups in ['expanded','canonical']:
        for k in ['_rotation','_scaling','_opacity']:frozen.pop(k,None)
    if args.geometry_groups=='canonical' and not args.freeze_canonical_xyz:frozen.pop('_xyz',None)
    before = {k:digest_tensor(v) for k,v in frozen.items()}
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    assert cache['config']['resolution'] == args.resolution
    assert [f['name'] for f in cache['frames']] == light['frames']
    assert torch.allclose(cache['times'], times.cpu(), atol=1e-6)
    assert torch.equal(cache['canonical_xyz'], pc.get_xyz[cache['ids'].cuda()].detach().cpu())
    support = cache['valid'][train].sum(0)>=10
    lum = cache['rgb'].mean(-1)
    reference = torch.nanquantile(torch.where(cache['valid'][train],lum[train],torch.nan),.9,dim=0)
    bright = (lum/reference.clamp_min(.05)[None] >= .65) & torch.isfinite(reference)[None] & (reference[None]>.05)
    inputs = [Path(args.cache),ply, deform_path, Path(args.model)/'cfg_args', Path(args.source)/'transforms_train.json', Path(args.lights), Path(__file__)]
    if args.xyz_preconditioner=='jacobian':inputs.extend([Path(__file__).with_name('dl_xyz_metric.py'),Path('utils/time_utils.py').resolve()])
    if args.shadow_state:inputs.extend([Path(args.shadow_state),Path(__file__).with_name('dl_projection_guard.py')])
    runtime = dict(args=vars(args), inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                   native=sys.modules['diff_surfel_rasterization'].provenance, selected=selected, validation=validation,
                   gt_read=args.gt_direction_reference, scope=('已知训练光向隔离参考；实际下降接受' if args.gt_direction_reference else '未知光共享材质与位移头短对照；实际下降接受，无GT') )
    (out/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2))
    bg = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.)
    cached = {}; coverage = []; pixel_signs = {}; hidden_bound=0.; hidden_cache={}; gate_masks={}
    with torch.no_grad():
        for i in selected+validation:
            camera = cameras[i];camera.load2device()
            ds = deform.step(pc.get_xyz, camera.fid.reshape(1,1).expand(len(pc.get_xyz),1), feature=pc.get_binary_feature(), camera_center=camera.camera_center)
            if args.trust_metric=='actual' and args.geometry_groups!='canonical':hidden_cache[i]=ds['hidden'].detach()
            hidden_bound=max(hidden_bound,float((torch.sqrt(ds['hidden'].square().sum(-1)+1)*pc.get_binary_feature().flatten()).max()))
            ds = {k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
            normals = build_rotation(pc.get_rotation_bias(ds['d_rotation']))[:,:,2]
            sign = torch.where((normals*(camera.camera_center-pc.get_xyz-ds['d_xyz'])).sum(-1,keepdim=True)>=0,1.,-1.)
            mask = camera.gt_alpha_mask.to('cuda') if camera.gt_alpha_mask is not None else torch.ones_like(camera.original_image_train_light[:1])
            target = srgb_to_linear(camera.original_image_train_light*mask)
            original = np.array(Image.open(camera.image_path_train_light).convert('RGBA'))
            saturated = torch.from_numpy((original[...,:3]>=254).astype(np.float32)).permute(2,0,1)[None].cuda()
            saturated = F.adaptive_max_pool2d(saturated, target.shape[-2:])[0].bool() & (mask>.99)
            height,width=target.shape[-2:]
            all_mask=torch.zeros((1,height,width),device='cuda')
            lit_mask=torch.zeros_like(all_mask)
            good=cache['valid'][i]&support
            uv=cache['pixels'][i][good].cuda()
            all_mask[0,uv[:,1],uv[:,0]]=1
            uv=cache['pixels'][i][good&bright[i]].cuda()
            lit_mask[0,uv[:,1],uv[:,0]]=1
            chosen=lit_mask if args.observations=='lit' else all_mask
            base=render(camera,pc,pipe,bg,**ds,override_color=torch.full_like(pc.get_xyz,.5))
            surface_normal=F.normalize(base['surf_normal'],dim=0)
            facing=(surface_normal*(camera.camera_center[:,None,None]-base['surf_point'])).sum(0,keepdim=True)
            pixel_signs[i]=torch.where(facing>=0,1.,-1.)
            valid=(mask>.99)&(base['rend_alpha']>.95)&torch.isfinite(base['surf_depth'])&(base['surf_depth']>0)
            valid=-F.max_pool2d(-valid.float()[None],3,stride=1,padding=1)[0]>.5
            chosen=chosen*valid
            assert chosen.sum()>0
            if args.balance_head_groups:
                gp=render(camera,pc,pipe,bg,**ds,override_color=pc.get_binary_feature().expand(-1,3))
                gate_masks[i]=gp['render'][:1]/gp['rend_alpha'].clamp_min(1e-6)>.5
                assert (chosen*gate_masks[i]).sum()>0 and (chosen*(~gate_masks[i])).sum()>0
            interior=-F.max_pool2d(-(mask>.99).float()[None],5,stride=1,padding=2)[0]
            coverage.append(dict(index=i,all_pixels=int(all_mask.sum()),lit_pixels=int(lit_mask.sum()),foreground_pixels=int((mask>.5).sum())))
            if args.balance_head_groups:
                coverage[-1].update(dynamic_pixels=int((chosen*gate_masks[i]).sum()),static_pixels=int((chosen*(~gate_masks[i])).sum()))
            cached[i]=(ds,sign,target,saturated,chosen,all_mask,interior,mask)
            if i%20==0: print(json.dumps({'cached_frame':i,'mask_pixels':int(chosen.sum())}),flush=True)
    (out/'support.json').write_text(json.dumps(coverage,indent=2))
    ambient=torch.nn.Parameter(torch.tensor(0.,device='cuda'),requires_grad=args.ambient_max>0)
    albedo = torch.nn.Parameter(torch.full_like(pc.get_xyz, math.log(math.expm1(.5))))
    energy = torch.nn.Parameter(torch.full((len(train),), math.log(math.expm1(1.)), device='cuda'))
    train_times = times[train]
    right = torch.searchsorted(train_times,times).clamp(1,len(train)-1);left=right-1
    weight = ((times-train_times[left])/(train_times[right]-train_times[left])).clamp(0,1)
    def energy_values():
        values=F.softplus(energy);values=values/values.mean()
        return torch.lerp(values[left],values[right],weight)
    light_nodes=torch.nn.Parameter(torch.zeros((9,3),device='cuda'),requires_grad=args.refine_light)
    node_times=torch.linspace(float(train_times[0]),float(train_times[-1]),9,device='cuda')
    node_right=torch.searchsorted(node_times,times).clamp(1,8);node_left=node_right-1
    node_weight=((times-node_times[node_left])/(node_times[node_right]-node_times[node_left])).clamp(0,1)
    def current_directions():
        correction=torch.lerp(light_nodes[node_left],light_nodes[node_right],node_weight[:,None])
        tangent=correction-(correction*directions).sum(-1,keepdim=True)*directions
        return F.normalize(directions+tangent,dim=-1) if args.refine_light else directions
    def current_ds(i):
        if args.geometry_groups=='canonical':
            result=deform.step(pc.get_xyz,cameras[i].fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=cameras[i].camera_center)
            return {k:result[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
        with torch.no_grad():
            hidden=hidden_cache[i] if i in hidden_cache else deform.step(pc.get_xyz,cameras[i].fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=cameras[i].camera_center)['hidden']
        ds=dict(cached[i][0]);ds['d_xyz']=head(hidden)*pc.get_binary_feature()
        return ds
    @torch.no_grad()
    def displacement_bound(delta):
        if args.trust_metric=='parameter':
            return hidden_bound*float(torch.sqrt(sum(v.square().sum() for v in delta.values())))
        return max(float((F.linear(h,delta['weight'],delta['bias'])*pc.get_binary_feature()).norm(dim=-1).max()) for h in hidden_cache.values())
    def forward(i, audit_full=False, head_objective=False):
        ds,sign,target,saturated,chosen,all_mask,interior,mask = cached[i]
        ds=current_ds(i)
        # albedo图已预乘Alpha；法线Lambert项不可再次乘Alpha。
        colors=F.softplus(albedo)*energy_values()[i]
        if args.shading_normal=='gaussian':
            gaussian_normal=build_rotation(pc.get_rotation_bias(ds['d_rotation']))[:,:,2]*sign
            gaussian_cosine=(gaussian_normal*current_directions()[i]).sum(-1,keepdim=True)
            colors=colors*(gaussian_cosine.clamp_min(0)+ambient)
        package=render(cameras[i],pc,pipe,bg,**ds,override_color=colors)
        surface_normal=F.normalize(package['surf_normal'],dim=0)*pixel_signs[i]
        cosine=(surface_normal*current_directions()[i,:,None,None]).sum(0,keepdim=True)
        shading=cosine.clamp_min(0)+ambient
        if args.normal_gradient=='stop' and not audit_full:shading=shading.detach()
        if args.shading_normal=='depth':package['render']=package['render']*shading
        residual=package['render']-target
        residual=torch.where(saturated,residual.clamp_max(0),residual)
        photo=(residual.double().square()*chosen).sum()/(3*chosen.sum()).clamp_min(1)
        normal=F.normalize(package['rend_normal'],dim=0)
        depth_normal=F.normalize(package['surf_normal'],dim=0)
        consistency=((1-(normal.double()*depth_normal.double()).sum(0,keepdim=True))*interior).sum()/interior.sum().clamp_min(1)
        alpha_loss=F.binary_cross_entropy(package['rend_alpha'].double().clamp(1e-6,1-1e-6),mask.double())
        package['diagnostics']=dict(photo=float(photo.detach()),normal=float(consistency.detach()),alpha=float(alpha_loss.detach()),
            full_mse=float(residual.square().mean().detach()),all_support_mse=float(((residual.square()*all_mask).sum()/(3*all_mask.sum()).clamp_min(1)).detach()))
        distortion=package['rend_dist'].mean()
        regularization=args.normal_weight*consistency+args.alpha_weight*alpha_loss+args.distortion_weight*distortion
        package['diagnostics']['distortion']=float(distortion.detach())
        positive_support=chosen*(target.mean(0,keepdim=True)>.05)
        positivity=((args.positive_margin-cosine.double()).clamp_min(0).square()*positive_support).sum()/chosen.sum().clamp_min(1)
        package['diagnostics']['positivity']=float(positivity.detach())
        package['loss_terms']=dict(photo=photo,regularization=regularization,positivity=positivity)
        balanced_photo=photo
        if args.balance_head_groups:
            dynamic=chosen*gate_masks[i];static=chosen*(~gate_masks[i])
            dynamic_photo=(residual.double().square()*dynamic).sum()/(3*dynamic.sum())
            static_photo=(residual.double().square()*static).sum()/(3*static.sum())
            balanced_photo=.5*(dynamic_photo+static_photo)
            package['loss_terms']['balanced_photo']=balanced_photo
            package['diagnostics'].update(dynamic_photo=float(dynamic_photo.detach()),static_photo=float(static_photo.detach()))
        objective=photo+regularization
        if head_objective:
            if args.head_objective=='photo':objective=photo
            elif args.head_objective=='regularization':objective=regularization
            if args.balance_head_groups and args.head_objective!='regularization':objective=objective-photo+balanced_photo
            objective=objective+args.positive_weight*positivity
        return objective,package
    @torch.no_grad()
    def evaluate(label):
        losses={}; diagnostics={};folder=out/label;folder.mkdir()
        for i in selected+validation:
            loss,package=forward(i);losses[str(i)]=float(loss);diagnostics[str(i)]=package['diagnostics']
            # 固定早中晚保留时间保存完整四类图与深度。
            if i in validation:
                name=cameras[i].image_name_train_light
                save_image(folder/f'{name}_rgb.png',linear_to_srgb(package['render'].clamp_min(0)))
                save_image(folder/f'{name}_alpha.png',package['rend_alpha'])
                save_image(folder/f'{name}_normal.png',(F.normalize(package['rend_normal'],dim=0)*.5+.5)*package['rend_alpha'])
                gate=pc.get_binary_feature();colors=torch.cat([gate,1-gate,torch.zeros_like(gate)],dim=1)
                sep=render(cameras[i],pc,pipe,bg,**current_ds(i),override_color=colors)
                save_image(folder/f'{name}_separation.png',sep['render'])
                torch.save({k:package[k].cpu() for k in ['surf_depth','rend_alpha','rend_normal','surf_normal']},folder/f'{name}_depth.pt')
        return dict(diagnostics=diagnostics,per_frame=losses,train=float(np.mean([losses[str(i)] for i in selected])),held=float(np.mean([losses[str(i)] for i in validation])))
    groups=[{'params':[albedo],'lr':.02},{'params':[energy],'lr':.01}]
    if args.ambient_max>0:groups.append({'params':[ambient],'lr':.001})
    if args.refine_light:groups.append({'params':[light_nodes],'lr':.001})
    optimizer=torch.optim.Adam(groups)
    @torch.no_grad()
    def clamp_ambient():
        ambient.clamp_(0,args.ambient_max)
        if args.refine_light:light_nodes.mul_((.1/light_nodes.norm(dim=-1,keepdim=True).clamp_min(1e-30)).clamp_max(1))
    guard=None;guard_cost=None;light_history=[];light_guard_renders=0
    if args.shadow_state:
        from scripts.dl_projection_guard import ProjectionGuard
        assert runtime['inputs'][str(Path(args.cache))]==light['cache_sha256']
        guard=ProjectionGuard(cache,torch.load(args.shadow_state,map_location='cpu',weights_only=False),light)
        verification=guard.verify(current_directions())
        guard_cost=verification['cost']
        (out/'shadow_guard_verification.json').write_text(json.dumps(verification,indent=2))
    def material_step(i):
        nonlocal guard_cost,light_guard_renders
        old_nodes=light_nodes.detach().clone() if guard else None
        old_state=copy.deepcopy(optimizer.state.get(light_nodes)) if guard else None
        optimizer.zero_grad(set_to_none=True);loss,_=forward(i);loss.backward();optimizer.step();clamp_ambient()
        if guard:
            entry=dict(step=len(light_history)+1,frame=i,shadow_before=guard_cost,accepted=False,attempts=[])
            with torch.no_grad():
                proposal=light_nodes.detach().clone();old_rgb=None
                for fraction in [1.,.5,.25,.125,.0625,.03125]:
                    candidate=torch.lerp(old_nodes,proposal,fraction)
                    light_nodes.copy_(candidate)
                    proxy=guard.cost(current_directions())
                    attempt=dict(fraction=fraction,shadow_cost=proxy,accepted=False)
                    if proxy<=guard_cost+1e-9:
                        if old_rgb is None:
                            light_nodes.copy_(old_nodes);old_rgb=float(forward(i)[0]);light_guard_renders+=1
                        light_nodes.copy_(candidate);new_rgb=float(forward(i)[0]);light_guard_renders+=1
                        attempt.update(old_rgb=old_rgb,new_rgb=new_rgb)
                        if np.isfinite(new_rgb) and new_rgb<old_rgb-1e-10:
                            attempt['accepted']=True;entry['accepted']=True;guard_cost=proxy
                    entry['attempts'].append(attempt)
                    if entry['accepted']:break
                if not entry['accepted']:
                    light_nodes.copy_(old_nodes)
                    if old_state is None:optimizer.state.pop(light_nodes,None)
                    else:optimizer.state[light_nodes]=old_state
            light_history.append(entry)
        return loss
    history=[];metrics={'initial':evaluate('initial')}
    for step in range(args.warmup):
        loss=material_step(selected[step%len(selected)])
        if step%30==0:print(json.dumps(dict(material_step=step+1,loss=float(loss))),flush=True)
    metrics['warmup']=evaluate('warmup')
    head.requires_grad_(args.mode=='joint');accepted=0;proposal_renders=0
    optimized=dict(head.named_parameters())
    if args.geometry_groups in ['expanded','canonical']:
        for k in ['_rotation','_scaling','_opacity']:
            geometry[k].requires_grad_(True);optimized[k]=geometry[k]
    if args.geometry_groups=='canonical' and not args.freeze_canonical_xyz:
        geometry['_xyz'].requires_grad_(True);optimized['_xyz']=geometry['_xyz']
    initial_optimized={k:v.detach().clone() for k,v in optimized.items()}
    group_step={'_rotation':.001,'_scaling':.001,'_opacity':.005}
    group_total={'_rotation':.02,'_scaling':.02,'_opacity':.1}
    # canonical改变后隐藏特征和其余形变输出都会变化；实测全部离散时刻的位置。
    world_indices=selected+validation
    initial_world={i:(initial_geometry['_xyz']+cached[i][0]['d_xyz']).detach().clone() for i in world_indices} if args.geometry_groups=='canonical' else {}
    current_world=dict(initial_world)
    candidate_audit={}
    audit_lookup={int(v):j for j,v in enumerate(cache['ids'])} if args.audit_world_candidates else {}
    def point_support(point_id):
        j=audit_lookup.get(point_id)
        if j is None:return dict(sampled=False,scope='未抽样不等于没有渲染梯度')
        chosen=[]
        for i in selected:
            if not bool(cache['valid'][i,j]):continue
            u,v=map(int,cache['pixels'][i,j].tolist())
            if bool(cached[i][4][0,v,u]>0):chosen.append(i)
        return dict(sampled=True,valid_train_count=int(cache['valid'][selected,j].sum()),chosen_train_indices=chosen,scope='默认轨迹像素位于固定最终chosen掩膜；不等于该点独占像素贡献')
    @torch.no_grad()
    def measure_world(previous):
        positions={};step_bound=0.;total_bound=0.
        candidate_audit.clear()
        for i in world_indices:
            positions[i]=(pc.get_xyz+current_ds(i)['d_xyz']).detach()
            step_norm=(positions[i]-previous[i]).norm(dim=-1)
            if args.audit_world_candidates and float(step_norm.max())>step_bound:
                point_id=int(step_norm.argmax())
                candidate_audit.update(index=i,point_id=point_id,held=i in validation,world_step=float(step_norm[point_id]),dynamic_gate=float(pc.get_binary_feature()[point_id]),canonical_total=float((pc.get_xyz[point_id]-initial_geometry['_xyz'][point_id]).norm()))
            step_bound=max(step_bound,float(step_norm.max()))
            total_bound=max(total_bound,float((positions[i]-initial_world[i]).norm(dim=-1).max()))
        return positions,step_bound,total_bound
    if args.geometry_groups=='canonical':
        measured,step_bound,total_bound=measure_world(initial_world)
        assert total_bound<1e-6
        (out/'world_initial_verification.json').write_text(json.dumps(dict(frames=len(measured),max_difference=total_bound,full_deformation_query=True,canonical_input_jacobian=True),indent=2))
        initial_world=measured;current_world=dict(measured)
    xyz_metric=None
    if args.xyz_preconditioner=='jacobian':
        from scripts.dl_xyz_metric import diagonal_metric,affine_check
        check=affine_check('cuda')
        metric_indices=selected if args.xyz_metric_times=='train' else [0,1,2,64,65,66,132,133,134]
        assert set(metric_indices)<=set(selected)
        xyz_metric=torch.zeros_like(pc.get_xyz)
        for i in metric_indices:
            metric=diagonal_metric(pc.get_xyz,pc.get_xyz+current_ds(i)['d_xyz'])
            xyz_metric=torch.maximum(xyz_metric,metric)
        torch.save(dict(diagonal=xyz_metric.cpu(),indices=metric_indices,definition='max_t diag(J_t^T J_t), J=d(x+deform(x,t))/dx',damping=1.),out/'xyz_metric.pt')
        quantiles=torch.quantile(xyz_metric.flatten(),torch.tensor([0.,.5,.9,.99,1.],device='cuda')).tolist()
        (out/'xyz_metric_verification.json').write_text(json.dumps(dict(affine=check,indices=metric_indices,quantiles=quantiles,gt_read=False,parameters_changed=False),indent=2))
    for block in range(args.blocks):
        batch=[selected[(block*3+j)%len(selected)] for j in range(3)]
        for i in batch:
            loss=material_step(i)
        entry=dict(block=block+1,batch=batch,accepted=False,attempts=[],ambient=float(ambient.detach()))
        if args.mode=='joint' and args.geometry_groups=='canonical':
            outputs=[forward(i,head_objective=True) for i in batch]
            objective=sum(v[0] for v in outputs)/len(batch);proposal_renders+=len(batch)
            gradients=torch.autograd.grad(objective,tuple(optimized.values()))
            direction={k:-g.detach() for k,g in zip(optimized,gradients)}
            assert all(torch.isfinite(g).all() for g in direction.values())
            start={k:v.detach().clone() for k,v in optimized.items()}
            baseline=float(objective);entry['baseline']=baseline
            entry['gradient_norms']={k:float(v.norm()) for k,v in direction.items()}
            if xyz_metric is not None:
                direction['_xyz']=direction['_xyz']/(1.+xyz_metric)
                entry['preconditioned_xyz_norm']=float(direction['_xyz'].norm())
            with torch.no_grad():
                # 此处仅为提议预条件，绝不作为世界位移保证；所有接受须全时刻实测。
                hn=hidden_bound*float(torch.sqrt(sum(direction[k].square().sum() for k in initial_head)))
                proposal={k:start[k]+direction[k]*(1e-4/max(hn,1e-30)) for k in initial_head}
                for k in ['_rotation','_scaling','_opacity','_xyz']:
                    if k not in optimized:continue
                    maxnorm=float(direction[k].reshape(len(direction[k]),-1).norm(dim=-1).max())
                    step_size=1e-5 if k=='_xyz' else group_step[k]
                    cap=.001 if k=='_xyz' else group_total[k]
                    candidate=start[k]+direction[k]*(step_size/max(maxnorm,1e-30))
                    difference=candidate-initial_optimized[k]
                    difference=difference*(cap/difference.reshape(len(difference),-1).norm(dim=-1,keepdim=True).clamp_min(1e-30)).clamp_max(1)
                    proposal[k]=initial_optimized[k]+difference
                fraction=1.
                for attempt in range(12):
                    for k,v in optimized.items():v.copy_(torch.lerp(start[k],proposal[k],fraction))
                    positions,step_bound,total_bound=measure_world(current_world)
                    item=dict(fraction=fraction,world_step=step_bound,world_total=total_bound,accepted=False)
                    if args.audit_world_candidates:
                        detail=dict(candidate_audit);pid=detail['point_id']
                        detail['support']=point_support(pid)
                        detail['xyz_direction_norm']=float(direction['_xyz'][pid].norm()) if '_xyz' in direction else 0.
                        detail['canonical_step']=float((pc.get_xyz[pid]-start['_xyz'][pid]).norm()) if '_xyz' in start else 0.
                        detail['batch_chosen_indices']=[i for i in batch if i in detail['support'].get('chosen_train_indices',[])]
                        item['candidate_audit']=detail
                    safe=np.isfinite(step_bound) and np.isfinite(total_bound) and step_bound<=1.0002e-4 and total_bound<=.00100002
                    if safe:
                        trial=float(sum(forward(i,head_objective=True)[0] for i in batch)/len(batch));proposal_renders+=len(batch)
                        item['loss']=trial
                        if np.isfinite(trial) and trial<baseline-1e-10:
                            item['accepted']=True;entry['accepted']=True;accepted+=1;current_world=positions
                    entry['attempts'].append(item)
                    if entry['accepted']:break
                    fraction*=min(.5,.9*1e-4/max(step_bound,1e-30),.9*.001/max(total_bound,1e-30)) if not safe else .5
                if not entry['accepted']:
                    for k,v in optimized.items():v.copy_(start[k])
                entry['world_frames_checked']=len(world_indices)
        elif args.mode=='joint':
            outputs=[forward(i,head_objective=True) for i in batch]
            objective=sum(v[0] for v in outputs)/len(batch);proposal_renders+=len(batch)
            if block in [0,19,39]:
                components={name:sum(v[1]['loss_terms'][name] for v in outputs)/len(batch) for name in ['photo','regularization']}
                component_gradients={name:torch.cat([g.detach().flatten() for g in torch.autograd.grad(value,tuple(head.parameters()),retain_graph=True)]) for name,value in components.items()}
                gp=component_gradients['photo'];gr=component_gradients['regularization']
                entry['gradient_components']=dict(photo_norm=float(gp.norm()),regularization_norm=float(gr.norm()),cosine=float(F.cosine_similarity(gp,gr,dim=0)))
                if args.positive_weight>0:
                    positive=sum(v[1]['loss_terms']['positivity'] for v in outputs)/len(batch)
                    gpos=torch.cat([g.detach().flatten() for g in torch.autograd.grad(args.positive_weight*positive,tuple(head.parameters()),retain_graph=True)])
                    entry['gradient_components'].update(positive_norm=float(gpos.norm()),positive_photo_cosine=float(F.cosine_similarity(gpos,gp,dim=0)))
            gradients=torch.autograd.grad(objective,tuple(optimized.values()))
            all_direction={k:-g.detach() for k,g in zip(optimized,gradients)}
            direction={k:all_direction[k] for k in initial_head}
            direction_norm=torch.sqrt(sum(v.square().sum() for v in direction.values()))
            assert torch.isfinite(direction_norm) and direction_norm>0
            start={k:v.detach().clone() for k,v in head.state_dict().items()}
            baseline=float(objective);entry['baseline']=baseline
            with torch.no_grad():
                # 两种度量均限制同一离散时间世界位移上限；actual显式遍历固定隐藏特征。
                direction_bound=displacement_bound(direction)
                scale=1e-4/max(direction_bound,1e-30)
                entry['direction_bound']=direction_bound
                entry['parameter_bound']=hidden_bound*float(direction_norm)
                proposal={k:v+scale*direction[k] for k,v in start.items()}
                norm=float(torch.sqrt(sum((v-initial_head[k]).square().sum() for k,v in proposal.items())))
                cumulative_bound=displacement_bound({k:v-initial_head[k] for k,v in proposal.items()})
                fraction=min(1.,.001/max(cumulative_bound,1e-30))
                entry['cumulative_bound_before_projection']=cumulative_bound
                proposal={k:torch.lerp(initial_head[k],v,fraction) for k,v in proposal.items()}
                entry['bound_fraction']=fraction
                if args.trust_metric=='actual':
                    step_bound=displacement_bound({k:v-start[k] for k,v in proposal.items()})
                    step_fraction=min(1.,1e-4/max(step_bound,1e-30))
                    proposal={k:torch.lerp(start[k],v,step_fraction) for k,v in proposal.items()}
                    entry['step_fraction_after_projection']=step_fraction
                    entry['proposal_step_bound']=displacement_bound({k:v-start[k] for k,v in proposal.items()})
                    entry['proposal_cumulative_bound']=displacement_bound({k:v-initial_head[k] for k,v in proposal.items()})
                    entry['bound_before_numerical_backoff']=[entry['proposal_step_bound'],entry['proposal_cumulative_bound']]
                    entry['numerical_backoffs']=0
                    for _ in range(8):
                        if entry['proposal_step_bound']<=1.0002e-4 and entry['proposal_cumulative_bound']<=.00100002:break
                        proposal={k:torch.lerp(start[k],v,.99) for k,v in proposal.items()}
                        entry['numerical_backoffs']+=1
                        entry['proposal_step_bound']=displacement_bound({k:v-start[k] for k,v in proposal.items()})
                        entry['proposal_cumulative_bound']=displacement_bound({k:v-initial_head[k] for k,v in proposal.items()})
                    if entry['proposal_step_bound']>1.0002e-4 or entry['proposal_cumulative_bound']>.00100002:
                        proposal={k:v.clone() for k,v in start.items()}
                        entry['head_proposal_cancelled']=True
                        entry['proposal_step_bound']=0.
                        entry['proposal_cumulative_bound']=displacement_bound({k:v-initial_head[k] for k,v in proposal.items()})
                    assert entry['proposal_step_bound']<=1.0002e-4 and entry['proposal_cumulative_bound']<=.00100002
                # 额外几何组按各自参数单位限幅，同一回溯比例联合验收。
                for k in group_step:
                    if k not in optimized:continue
                    start[k]=optimized[k].detach().clone()
                    g=all_direction[k]
                    maxnorm=float(g.reshape(len(g),-1).norm(dim=-1).max())
                    step=g*(group_step[k]/max(maxnorm,1e-30))
                    delta=start[k]+step-initial_optimized[k]
                    dn=delta.reshape(len(delta),-1).norm(dim=-1,keepdim=True)
                    delta=delta*(group_total[k]/dn.clamp_min(1e-30)).clamp_max(1)
                    proposal[k]=initial_optimized[k]+delta
                for fraction_step in [1.,.5,.25,.125,.0625,.03125]:
                    for k,v in optimized.items():v.copy_(torch.lerp(start[k],proposal[k],fraction_step))
                    trial=float(sum(forward(i,head_objective=True)[0] for i in batch)/len(batch));proposal_renders+=len(batch)
                    ok=np.isfinite(trial) and trial<baseline-1e-10
                    entry['attempts'].append(dict(fraction=fraction_step,loss=trial,accepted=bool(ok)))
                    if ok:entry['accepted']=True;accepted+=1;break
                if not entry['accepted']:
                    for k,v in optimized.items():v.copy_(start[k])
        history.append(entry);print(json.dumps(entry),flush=True)
    metrics['final']=evaluate('final')
    with torch.no_grad():
        motion={str(i):float(((pc.get_xyz+current_ds(i)['d_xyz']-initial_world[i]) if args.geometry_groups=='canonical' else (current_ds(i)['d_xyz']-cached[i][0]['d_xyz'])).norm(dim=-1).max()) for i in selected+validation}
    unchanged=before=={k:digest_tensor(v) for k,v in frozen.items()}
    geometry_deltas={k:float((v.detach()-initial_geometry[k]).reshape(len(v),-1).norm(dim=-1).max()) for k,v in geometry.items()}
    assert args.geometry_groups not in ['expanded','canonical'] or all(geometry_deltas[k]<=group_total[k]+1e-6 for k in group_total)
    delta=max(float((v-initial_head[k]).abs().max()) for k,v in head.state_dict().items())
    passed=unchanged and max(motion.values())<=.00101 and all(np.isfinite(v['held']) for v in metrics.values()) and (accepted>0 and delta>0 if args.mode=='joint' else delta==0)
    torch.save(dict(geometry={k:v.detach().cpu() for k,v in geometry.items()},deform_state_dict={k:v.detach().cpu() for k,v in deform.deform.state_dict().items()},albedo=albedo.detach().cpu(),energy=energy.detach().cpu(),ambient=ambient.detach().cpu(),light_nodes=light_nodes.detach().cpu(),directions=current_directions().detach().cpu(),optimizer=optimizer.state_dict(),args=vars(args),material_updates=args.warmup+args.blocks*3,geometry_updates=accepted if args.geometry_groups in ['expanded','canonical'] else 0,geometry_deltas=geometry_deltas,deformation_updates=accepted),out/'checkpoint.pt')
    with torch.no_grad():
        inferred=current_directions()
        drift=torch.rad2deg(torch.atan2(torch.cross(directions,inferred,dim=-1).norm(dim=-1),(directions*inferred).sum(-1)))
        assert not args.refine_light or float(drift.max())<=math.degrees(math.atan(.1))+1e-4
        inferred_light={k:light[k] for k in ['frames','times','train_indices','held_indices','cache_sha256']}
        inferred_light.update(directions=inferred.cpu().tolist(),gt_read=args.gt_direction_reference,protocol='9个均匀时间控制点的共享残差；节点范数<=0.1，逐时刻投影到初始灯向切平面后归一化；无GT细化' if args.refine_light else '固定输入灯向',source_runtime=str(out/'runtime.json'))
        (out/'inferred_light.json').write_text(json.dumps(inferred_light,ensure_ascii=False,indent=2))
    result=dict(status='PASS' if passed else 'FAILED',scope='真实下降和冻结边界工程验收；梯度审计失败未改判，几何质量需独立评测',gt_read=args.gt_direction_reference,ambient=float(ambient.detach()),light_drift_deg=drift.cpu().tolist(),light_nodes=light_nodes.detach().cpu().tolist(),light_updates=sum(x['accepted'] for x in light_history) if guard else (args.warmup+args.blocks*3 if args.refine_light else 0),light_proposals=len(light_history),light_history=light_history,light_guard_renders=light_guard_renders,shadow_guard_final_cost=guard_cost,metrics=metrics,history=history,material_updates=args.warmup+args.blocks*3,geometry_updates=accepted if args.geometry_groups in ['expanded','canonical'] else 0,geometry_deltas=geometry_deltas,deformation_updates=accepted,proposal_renders=proposal_renders,frozen_unchanged=unchanged,head_delta=delta,hidden_bound=hidden_bound,motion=motion,max_memory_bytes=torch.cuda.max_memory_allocated())
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','history','motion','light_history']},ensure_ascii=False),flush=True)
    if not passed:raise RuntimeError('工程验收FAILED，保留checkpoint及所有输出')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','lights','output','cache']:p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['joint','frozen'],required=True)
    p.add_argument('--geometry-groups',choices=['head','expanded','canonical'],default='head')
    p.add_argument('--freeze-canonical-xyz',action='store_true')
    p.add_argument('--audit-world-candidates',action='store_true')
    p.add_argument('--xyz-metric-times',choices=['nine','train'],default='nine')
    p.add_argument('--xyz-preconditioner',choices=['none','jacobian'],default='none')
    p.add_argument('--historical-loss',action='store_true')
    p.add_argument('--shading-normal',choices=['depth','gaussian'],default='depth')
    p.add_argument('--gt-direction-reference',action='store_true')
    p.add_argument('--shadow-state',default=None)
    p.add_argument('--refine-light',action='store_true')
    p.add_argument('--ambient-max',type=float,default=0.)
    p.add_argument('--balance-head-groups',action='store_true')
    p.add_argument('--trust-metric',choices=['parameter','actual'],default='parameter')
    p.add_argument('--positive-weight',type=float,default=0.)
    p.add_argument('--positive-margin',type=float,default=.05)
    p.add_argument('--head-objective',choices=['full','photo','regularization'],default='full')
    p.add_argument('--iteration',type=int,default=35000);p.add_argument('--seed',type=int,default=1015)
    p.add_argument('--warmup',type=int,default=240);p.add_argument('--blocks',type=int,default=40)
    args=p.parse_args();args.resolution=4;args.frames=0;args.observations='lit';args.normal_gradient='full';args.normal_weight=.05
    assert not args.freeze_canonical_xyz or args.geometry_groups=='canonical'
    assert args.xyz_metric_times=='nine' or args.xyz_preconditioner=='jacobian'
    assert args.xyz_preconditioner=='none' or (args.geometry_groups=='canonical' and not args.freeze_canonical_xyz)
    args.alpha_weight=.1 if args.historical_loss else .01
    args.distortion_weight=1000. if args.historical_loss else 0.
    if args.historical_loss:args.normal_weight=.1
    assert args.geometry_groups not in ['expanded','canonical'] or (args.mode=='joint' and not args.refine_light)
    assert not args.shadow_state or args.refine_light, '投影约束仅用于未知灯向细化'
    assert not args.refine_light or (args.mode=='frozen' and not args.gt_direction_reference), '首次灯向细化只允许未知光固定几何'
    assert 0<=args.ambient_max<=.1, '亮度底项上界限制为平均DL归一尺度的0.1'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n');run(args,out)

if __name__=='__main__':main()
