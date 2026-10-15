"""可靠受光筛选与深度法线一致性的受控短训练；未知光入口不读取GT。"""
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
    assert not light.get('gt_direction_reference', False) and not light.get('gt_read', False)
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
    frozen = {k:v for k,v in tensors.items() if k not in geometry}
    frozen.update({'deform.'+k:v for k,v in deform.deform.state_dict().items()})
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
    runtime = dict(args=vars(args), inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                   native=native_module().provenance, selected=selected, validation=validation,
                   gt_read=False, scope='受光筛选短训练；无显式阴影／环境／镜面，固定默认形变输出')
    (out/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2))
    bg = torch.zeros(3, device='cuda')
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=0.)
    cached = {}; coverage = []
    with torch.no_grad():
        for i in selected+validation:
            camera = cameras[i];camera.load2device()
            ds = deform.step(pc.get_xyz, camera.fid.reshape(1,1).expand(len(pc.get_xyz),1), feature=pc.get_binary_feature(), camera_center=camera.camera_center)
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
    def forward(i):
        ds,sign,target,saturated,chosen,all_mask,interior,mask = cached[i]
        normals = build_rotation(pc.get_rotation_bias(ds['d_rotation']))[:,:,2]*sign
        colors=F.softplus(albedo)*(normals@directions[i]).clamp_min(0)[:,None]*energy_values()[i]
        package=render(cameras[i],pc,pipe,bg,**ds,override_color=colors)
        residual=package['render']-target
        residual=torch.where(saturated,residual.clamp_max(0),residual)
        photo=(residual.square()*chosen).sum()/(3*chosen.sum()).clamp_min(1)
        normal=F.normalize(package['rend_normal'],dim=0)
        depth_normal=F.normalize(package['surf_normal'],dim=0)
        consistency=((1-(normal*depth_normal).sum(0,keepdim=True))*interior).sum()/interior.sum().clamp_min(1)
        alpha_loss=F.binary_cross_entropy(package['rend_alpha'].clamp(1e-6,1-1e-6),mask)
        package['diagnostics']=dict(photo=float(photo.detach()),normal=float(consistency.detach()),alpha=float(alpha_loss.detach()),
            full_mse=float(residual.square().mean().detach()),all_support_mse=float(((residual.square()*all_mask).sum()/(3*all_mask.sum()).clamp_min(1)).detach()))
        return photo+args.normal_weight*consistency+.01*alpha_loss,package
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
                sep=render(cameras[i],pc,pipe,bg,**cached[i][0],override_color=colors)
                save_image(folder/f'{name}_separation.png',sep['render'])
                torch.save({k:package[k].cpu() for k in ['surf_depth','rend_alpha','rend_normal','surf_normal']},folder/f'{name}_depth.pt')
        return dict(diagnostics=diagnostics,per_frame=losses,train=float(np.mean([losses[str(i)] for i in selected])),held=float(np.mean([losses[str(i)] for i in validation])))
    optimizer=torch.optim.Adam([{'params':[albedo],'lr':.02},{'params':[energy],'lr':.01}])
    geom_optimizer=torch.optim.Adam([{'params':[geometry['_xyz']],'lr':1e-5},{'params':[geometry['_scaling'],geometry['_rotation']],'lr':1e-4}])
    metrics={'initial':evaluate('initial')};history=[]
    for step in range(args.warmup):
        optimizer.zero_grad(set_to_none=True);loss,_=forward(selected[step%len(selected)]);loss.backward();optimizer.step()
        history.append(dict(phase='material',step=step+1,loss=float(loss)))
        print(json.dumps(history[-1]),flush=True)
    metrics['warmup']=evaluate('warmup')
    assert all(torch.equal(v,initial_geometry[k]) for k,v in geometry.items())
    for v in geometry.values():v.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True);loss,_=forward(selected[0]);loss.backward()
    gradient=geometry['_xyz'].grad;coordinate=int(gradient.abs().argmax());analytic=float(gradient.flatten()[coordinate])
    differences=[]
    with torch.no_grad():
        parameter=geometry['_xyz'].flatten();original=float(parameter[coordinate])
        for epsilon in [1e-3,3e-4,1e-4]:
            parameter[coordinate]=original+epsilon;plus=float(forward(selected[0])[0])
            parameter[coordinate]=original-epsilon;minus=float(forward(selected[0])[0])
            parameter[coordinate]=original
            numerical=(plus-minus)/(2*epsilon)
            differences.append(dict(epsilon=epsilon,numerical=numerical,relative_error=abs(numerical-analytic)/max(abs(numerical),abs(analytic),1e-8)))
    gradient_ok=abs(analytic)>1e-8 and all(x['relative_error']<=.2 for x in differences[1:])
    (out/'gradient.json').write_text(json.dumps(dict(coordinate=coordinate,analytic=analytic,differences=differences,passed=gradient_ok),indent=2))
    updates=0
    if args.geometry_mode=='frozen':
        for v in geometry.values():v.requires_grad_(False)
    if gradient_ok:
        for step in range(args.geometry_steps):
            optimizer.zero_grad(set_to_none=True);geom_optimizer.zero_grad(set_to_none=True)
            loss,_=forward(selected[step%len(selected)])
            if not torch.isfinite(loss):raise RuntimeError('非有限损失')
            loss.backward();optimizer.step()
            if args.geometry_mode=='train':geom_optimizer.step()
            updates+=1
            history.append(dict(phase='geometry',step=step+1,loss=float(loss)));print(json.dumps(history[-1]),flush=True)
    metrics['final']=evaluate('final')
    unchanged=before=={k:digest_tensor(v) for k,v in frozen.items()}
    delta={k:float((v-initial_geometry[k]).abs().max()) for k,v in geometry.items()}
    torch.save(dict(geometry={k:v.detach().cpu() for k,v in geometry.items()},albedo=albedo.detach().cpu(),energy=energy.detach().cpu(),
                    optimizer=optimizer.state_dict(),geometry_optimizer=geom_optimizer.state_dict(),args=vars(args),
                    material_updates=args.warmup+updates,geometry_updates=updates if args.geometry_mode=='train' else 0,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'checkpoint.pt')
    passed=gradient_ok and unchanged and (all(x>0 for x in delta.values()) if args.geometry_mode=='train' else all(x==0 for x in delta.values())) and all(np.isfinite(v['held']) for v in metrics.values())
    result=dict(status='PASS' if passed else 'FAILED',scope='受控短训练，GT几何离线评测另行执行',metrics=metrics,gradient_ok=gradient_ok,
                frozen_unchanged=unchanged,geometry_delta=delta,material_updates=args.warmup+updates,geometry_updates=updates if args.geometry_mode=='train' else 0,
                energy_mean=float(energy_values()[train].mean()),history=history,max_memory_bytes=torch.cuda.max_memory_allocated())
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
    if not passed:raise RuntimeError('工程验收失败，已保存结果和checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','lights','output','cache']:p.add_argument('--'+name,required=True)
    p.add_argument('--iteration',type=int,default=35000);p.add_argument('--seed',type=int,default=1015)
    p.add_argument('--resolution',type=int,default=4);p.add_argument('--frames',type=int,default=0)
    p.add_argument('--warmup',type=int,default=600);p.add_argument('--geometry-steps',type=int,default=600)
    p.add_argument('--observations',choices=['lit','all'],default='lit')
    p.add_argument('--geometry-mode',choices=['frozen','train'],default='train')
    p.add_argument('--normal-weight',type=float,default=.05)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'runtime.json').exists() or (out/'failure.json').exists():raise FileExistsError('禁止覆盖运行')
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    try:run(args,out)
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error)),ensure_ascii=False,indent=2));raise


if __name__=='__main__':main()
