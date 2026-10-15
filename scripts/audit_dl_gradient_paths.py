"""固定未知光终点投影与DL着色梯度路径诊断；零训练。"""
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
    geometry = {k:tensors[k] for k in ['_xyz','_scaling','_rotation']}
    initial_geometry = {k:v.detach().clone() for k,v in geometry.items()}
    head=deform.deform.gaussian_warp
    initial_head={k:v.detach().clone() for k,v in head.state_dict().items()}
    frozen = dict(tensors)
    frozen.update({k:geometry[k] for k in ['_scaling','_rotation']})
    frozen.update({'deform.'+k:v for k,v in deform.deform.state_dict().items() if not k.startswith('gaussian_warp.')})
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
    inputs = [Path(args.checkpoint),Path(args.cache),ply, deform_path, Path(args.model)/'cfg_args', Path(args.source)/'transforms_train.json', Path(args.lights), Path(__file__)]
    runtime = dict(args=vars(args), inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                   native=sys.modules['diff_surfel_rasterization'].provenance, selected=selected, validation=validation,
                   gt_read=args.gt_direction_reference, scope='投影与DL着色梯度路径诊断，无GT无训练' )
    (out/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2))
    bg = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.)
    cached = {}; coverage = []; pixel_signs = {}; hidden_bound=0.; hidden_cache={}
    with torch.no_grad():
        for i in selected+validation:
            camera = cameras[i];camera.load2device()
            ds = deform.step(pc.get_xyz, camera.fid.reshape(1,1).expand(len(pc.get_xyz),1), feature=pc.get_binary_feature(), camera_center=camera.camera_center)
            if args.trust_metric=='actual':hidden_cache[i]=ds['hidden'].detach()
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
            interior=-F.max_pool2d(-(mask>.99).float()[None],5,stride=1,padding=2)[0]
            coverage.append(dict(index=i,all_pixels=int(all_mask.sum()),lit_pixels=int(lit_mask.sum()),foreground_pixels=int((mask>.5).sum())))
            cached[i]=(ds,sign,target,saturated,chosen,all_mask,interior,mask)
            if i%20==0: print(json.dumps({'cached_frame':i,'mask_pixels':int(chosen.sum())}),flush=True)
    (out/'support.json').write_text(json.dumps(coverage,indent=2))
    albedo = torch.nn.Parameter(torch.full_like(pc.get_xyz, math.log(math.expm1(.5))))
    energy = torch.nn.Parameter(torch.full((len(train),), math.log(math.expm1(1.)), device='cuda'))
    train_times = times[train]
    right = torch.searchsorted(train_times,times).clamp(1,len(train)-1);left=right-1
    weight = ((times-train_times[left])/(train_times[right]-train_times[left])).clamp(0,1)
    def energy_values():
        values=F.softplus(energy);values=values/values.mean()
        return torch.lerp(values[left],values[right],weight)
    def current_ds(i):
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
        package=render(cameras[i],pc,pipe,bg,**ds,override_color=colors)
        surface_normal=F.normalize(package['surf_normal'],dim=0)*pixel_signs[i]
        cosine=(surface_normal*directions[i,:,None,None]).sum(0,keepdim=True)
        shading=cosine.clamp_min(0)
        if args.normal_gradient=='stop' and not audit_full:shading=shading.detach()
        base_render=package['render']
        package['render']=base_render*shading
        residual=package['render']-target
        residual=torch.where(saturated,residual.clamp_max(0),residual)
        photo=(residual.double().square()*chosen).sum()/(3*chosen.sum()).clamp_min(1)
        normal=F.normalize(package['rend_normal'],dim=0)
        depth_normal=F.normalize(package['surf_normal'],dim=0)
        consistency=((1-(normal.double()*depth_normal.double()).sum(0,keepdim=True))*interior).sum()/interior.sum().clamp_min(1)
        alpha_loss=F.binary_cross_entropy(package['rend_alpha'].double().clamp(1e-6,1-1e-6),mask.double())
        package['diagnostics']=dict(photo=float(photo.detach()),normal=float(consistency.detach()),alpha=float(alpha_loss.detach()),
            full_mse=float(residual.square().mean().detach()),all_support_mse=float(((residual.square()*all_mask).sum()/(3*all_mask.sum()).clamp_min(1)).detach()))
        regularization=args.normal_weight*consistency+.01*alpha_loss
        positive_support=chosen*(target.mean(0,keepdim=True)>.05)
        positivity=((args.positive_margin-cosine.double()).clamp_min(0).square()*positive_support).sum()/chosen.sum().clamp_min(1)
        package['diagnostics']['positivity']=float(positivity.detach())
        package['loss_terms']=dict(photo=photo,regularization=regularization,positivity=positivity)
        dynamic=chosen*gate_masks[i]
        static=chosen*(~gate_masks[i])
        dynamic_loss=(residual.double().square()*dynamic).sum()/(3*dynamic.sum()).clamp_min(1)
        static_loss=(residual.double().square()*static).sum()/(3*static.sum()).clamp_min(1)
        share=dynamic.sum()/chosen.sum()
        package['loss_terms'].update(dynamic=dynamic_loss,static=static_loss,dynamic_weighted=dynamic_loss*share,static_weighted=static_loss*(1-share),balanced=.5*(dynamic_loss+static_loss))
        for name,image in [('projection',base_render*shading.detach()),('shading',base_render.detach()*shading)]:
            rr=image-target;rr=torch.where(saturated,rr.clamp_max(0),rr)
            sq=rr.double().square()
            package['loss_terms'][name]=(sq*chosen).sum()/(3*chosen.sum())
            package['loss_terms']['balanced_'+name]=.5*((sq*dynamic).sum()/(3*dynamic.sum())+(sq*static).sum()/(3*static.sum()))
        objective=photo+regularization
        if head_objective:
            if args.head_objective=='photo':objective=photo
            elif args.head_objective=='regularization':objective=regularization
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
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert not ckpt['args'].get('gt_direction_reference',False)
    assert Path(ckpt['args']['lights']).resolve()==Path(args.lights).resolve()
    assert all(torch.equal(v.cpu(),ckpt['geometry'][k]) for k,v in geometry.items())
    deform.deform.load_state_dict(ckpt['deform_state_dict'],strict=True)
    with torch.no_grad():albedo.copy_(ckpt['albedo']);energy.copy_(ckpt['energy'])
    albedo.requires_grad_(False);energy.requires_grad_(False);head.requires_grad_(True)
    original={k:v.detach().clone() for k,v in head.state_dict().items()}
    all_before={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
    batches=[[selected[(block*3+j)%len(selected)] for j in range(3)] for block in [0,19,39]]
    gate_masks={};counts={}
    with torch.no_grad():
        for i in sum(batches,[]):
            gp=render(cameras[i],pc,pipe,bg,**current_ds(i),override_color=pc.get_binary_feature().expand(-1,3))
            gate_masks[i]=gp['render'][:1]/gp['rend_alpha'].clamp_min(1e-6)>.5
            chosen=cached[i][4]
            counts[i]=dict(dynamic=int((chosen*gate_masks[i]).sum()),static=int((chosen*(~gate_masks[i])).sum()))
            assert min(counts[i].values())>0
    def batch_values(batch):
        outputs=[forward(i)[1]['loss_terms'] for i in batch]
        terms={name:sum(o[name] for o in outputs)/len(outputs) for name in outputs[0]}
        terms['full']=terms['photo']+terms['regularization']+terms['positivity']
        terms['balanced_full']=terms['balanced']+terms['regularization']+terms['positivity']
        prefix='balanced_' if ckpt['args'].get('balance_head_groups',False) else ''
        terms['objective']=terms['balanced_full'] if prefix else terms['full']
        for name in ['projection','shading']:
            terms[name+'_objective']=terms[prefix+name]+terms['regularization']+terms['positivity']
        return terms
    rows=[]
    for batch in batches:
        terms=batch_values(batch)
        names=['photo','projection','shading','balanced','balanced_projection','balanced_shading','objective','projection_objective','shading_objective']
        gg={name:torch.autograd.grad(terms[name],tuple(head.parameters()),retain_graph=True) for name in names}
        flat={name:torch.cat([v.flatten() for v in g]) for name,g in gg.items()}
        row=dict(batch=batch,base={k:float(v) for k,v in terms.items()},gradient_norm={k:float(v.norm()) for k,v in flat.items()},projection_shading_cosine=float(F.cosine_similarity(flat['projection'],flat['shading'],dim=0)),balanced_projection_shading_cosine=float(F.cosine_similarity(flat['balanced_projection'],flat['balanced_shading'],dim=0)),sum_relative_error=float((flat['photo']-flat['projection']-flat['shading']).norm()/flat['photo'].norm()),balanced_sum_relative_error=float((flat['balanced']-flat['balanced_projection']-flat['balanced_shading']).norm()/flat['balanced'].norm()),objective_cosines={name:float(F.cosine_similarity(flat['objective'],flat[name],dim=0)) for name in ['projection_objective','shading_objective']},trials=[])
        with torch.no_grad():
            for name in ['objective','projection_objective','shading_objective']:
                direction={k:-v for (k,_),v in zip(head.named_parameters(),gg[name])}
                bound=displacement_bound(direction)
                for eps in [1e-4,3e-5]:
                    for k,v in head.state_dict().items():v.copy_(original[k]+direction[k]*(eps/bound))
                    vals=batch_values(batch)
                    actual=displacement_bound({k:v-original[k] for k,v in head.state_dict().items()})
                    row['trials'].append(dict(direction=name,requested_max_displacement=eps,actual_max_displacement=actual,values={k:float(v) for k,v in vals.items()}))
                    for k,v in head.state_dict().items():v.copy_(original[k])
        rows.append(row);print(json.dumps(row),flush=True)
    unchanged=all_before=={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
    result=dict(status='PASS' if unchanged else 'FAILED',gt_read=False,updates=0,unchanged=unchanged,counts=counts,rows=rows,scope='三批次投影/着色路径诊断，不是训练或科学几何验收')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));assert unchanged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','lights','output','cache','checkpoint']:p.add_argument('--'+name,required=True)
    args=p.parse_args();args.iteration=35000;args.seed=1015;args.resolution=4;args.frames=0;args.observations='lit';args.normal_gradient='full';args.normal_weight=.05;args.head_objective='full';args.gt_direction_reference=False;args.positive_weight=1.;args.positive_margin=.05;args.trust_metric='actual'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'command.txt').write_text(shlex.join(sys.argv)+'\n');run(args,out)

if __name__=='__main__':main()
