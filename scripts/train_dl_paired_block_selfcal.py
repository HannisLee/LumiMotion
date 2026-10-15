"""保留原未知光轨迹的零初始化低维修正；GT不进入训练。"""
import argparse, hashlib, json, shlex, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import loadCam
from scripts.probe_dynamic_dl_selfcal import read_config, save_image


def basis(times, knots=8):
    centers=torch.linspace(0,1,knots,device=times.device)
    return (1-(times[:,None]-centers[None]).abs()*(knots-1)).clamp_min(0)


def lights(B, direction_controls, log_energy_controls, train, base_direction, base_energy):
    raw=base_direction+B@direction_controls
    assert torch.isfinite(raw).all() and raw.norm(dim=-1).min()>1e-6
    direction=F.normalize(raw,dim=-1)
    energy=base_energy*(B@log_energy_controls).exp()
    energy=energy/energy[train].mean()
    assert torch.isfinite(energy).all() and energy.min()>0
    return direction,energy


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--group',choices=['baseline','paired'],default='paired')
    parser.add_argument('--verify',help='独立进程重载并重算55帧')
    a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n');torch.set_num_threads(4);start=time.monotonic()
    fit=Path('output/smoke_test/1015-78-LH-dynamic-DL-T86材质约束配对/bounded')
    prior=torch.load(fit/'checkpoint.pt',map_location='cpu',weights_only=False)
    previous=json.loads((fit/'result.json').read_text())
    for f,h in previous['inputs'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    args=prior['args'];assert not args['gt_direction_reference']
    cfg=read_config(Path(args['model'])/'cfg_args');cfg.resolution=2;cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    pc.load_ply(str(Path(args['model'])/'point_cloud/iteration_35000/point_cloud.ply'))
    deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
    assert deform.load_weights(args['model'],35000);deform.deform.eval().requires_grad_(False)
    for v in vars(pc).values():
        if isinstance(v,torch.Tensor):v.requires_grad_(False)
    geometry={k:v.detach().cpu().clone() for k,v in vars(pc).items() if isinstance(v,torch.Tensor)}
    deformation={k:v.detach().cpu().clone() for k,v in deform.deform.state_dict().items()}
    cameras=[loadCam(cfg,i,c,1.) for i,c in enumerate(readCamerasFromTransforms(args['source'],'transforms_train.json',cfg.train_light_folder,False))]
    cache=torch.load(prior['cache_source'],map_location='cpu',weights_only=False)
    train=prior['train_frames'];held=prior['held_frames'];assert not set(train)&set(held)
    times=torch.tensor([float(c.fid) for c in cameras],device='cuda');B=basis(times)
    old_d=prior['directions'].cuda();old_e=prior['energy'].cuda()
    dc=torch.zeros((8,3),device='cuda',requires_grad=True)
    ec=torch.zeros(8,device='cuda',requires_grad=True)
    gauge=float(old_e[train].mean());material=(prior['albedo'].cuda()*gauge).requires_grad_(True)
    upper=gauge;initial_dc=dc.detach().clone();initial_ec=ec.detach().clone()
    if a.verify:
        ck=torch.load(Path(a.verify)/'checkpoint.pt',map_location='cpu',weights_only=False)
        record=json.loads((Path(a.verify)/'result.json').read_text())
        assert ck['protocol']=='block_selfcal_paired_v1' and ck['geometry_updates']==0
        for k,v in ck['geometry'].items():assert torch.equal(v,geometry[k])
        for k,v in ck['deform_state_dict'].items():assert torch.equal(v,deformation[k])
        dc.copy_(ck['direction_controls'].cuda());ec.copy_(ck['log_energy_controls'].cuda());material.copy_(ck['albedo'].cuda())
        assert torch.equal(ck['time_basis'],B.cpu())
        assert torch.equal(ck['base_direction'],old_d.cpu()) and torch.equal(ck['base_energy'],old_e.cpu())
    from gaussian_renderer import render
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    bg=torch.zeros(3,device='cuda');frames={}
    # 重算原生形变及深度法线，不把旧shading当新的法线。
    for i,q in cache.items():
        c=cameras[i];c.load2device()
        ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center)
        ds={k:(ds[k].detach() if isinstance(ds[k],torch.Tensor) else ds[k]) for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
        for k,v in q['ds'].items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,ds[k].cpu())
            else:assert v==ds[k]
        pack=render(c,pc,pipe,bg,**ds,override_color=material.detach())
        n=F.normalize(pack['surf_normal'],dim=0)*q['sign'].cuda()
        frames[i]=dict(ds=ds,normal=n.detach(),mask=q['mask'].cuda(),target=q['target'].cuda(),alpha=pack['rend_alpha'].detach())
        c.load2device('cpu')
    from scripts.dl_pair_photometric import collect,pair_loss
    pairs,support,pair_sources=collect(frames,train)
    torch.save({k:v.cpu() for k,v in pairs.items()},out/'pairs.pt')
    (out/'support.json').write_text(json.dumps(support,indent=2))
    D0,E0=lights(B,dc,ec,train,old_d,old_e)
    initial_pairs={split:float(pair_loss(pairs,D0,E0,split=='train')) for split in ['train','held']}
    weight=None
    def frame(i,D,E):
        c=cameras[i];c.load2device();q=frames[i]
        pack=render(c,pc,pipe,bg,**q['ds'],override_color=material)
        shading=(q['normal']*D[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1
        pred=pack['render']*E[i]*shading
        loss=((pred-q['target']).double().square()*q['mask']).sum()/(3*q['mask'].sum())
        return loss,pred,pack
    def evaluate():
        D,E=lights(B,dc,ec,train,old_d,old_e);values={}
        for i in frames:values[i]=float(frame(i,D,E)[0]);cameras[i].load2device('cpu')
        return dict(train=float(np.mean([values[i] for i in train])),held=float(np.mean([values[i] for i in held])),per_frame=values)
    initial=evaluate()
    initial_fidelity=None
    if not a.verify:
        D0,E0=lights(B,dc,ec,train,old_d,old_e)
        initial_fidelity=dict(direction_max_abs=float((D0-old_d).abs().max()),energy_max_abs=float((E0-old_e/gauge).abs().max()),photo_max_abs=max(abs(initial['per_frame'][i]-previous['final']['per_frame'][str(i)]) for i in frames))
        assert initial_fidelity['direction_max_abs']<2e-7 and initial_fidelity['energy_max_abs']<2e-7 and initial_fidelity['photo_max_abs']<1e-8
    history=[];om=torch.optim.Adam([material],lr=.003);ol=torch.optim.Adam([dc,ec],lr=.01)
    if not a.verify:
        for epoch in range(20):
            om.zero_grad(set_to_none=True);D,E=lights(B,dc.detach(),ec.detach(),train,old_d,old_e);before=0.
            for i in train:
                with torch.enable_grad():loss,_,_=frame(i,D,E);(loss/len(train)).backward()
                before+=float(loss)/len(train);cameras[i].load2device('cpu')
            assert torch.isfinite(material.grad).all();om.step();material.clamp_(0,upper)
            ol.zero_grad(set_to_none=True);light_before=0.
            # 两组相同的材质梯度预算；material组仍计算灯光梯度但不更新。
            for i in train:
                c=cameras[i];c.load2device();q=frames[i]
                base=render(c,pc,pipe,bg,**q['ds'],override_color=material.detach())['render'].detach()
                with torch.enable_grad():
                    D,E=lights(B,dc,ec,train,old_d,old_e)
                    pred=base*E[i]*((q['normal']*D[i,:,None,None]).sum(0,keepdim=True).clamp_min(0)+.1)
                    loss=((pred-q['target']).double().square()*q['mask']).sum()/(3*q['mask'].sum())
                    (loss/len(train)).backward()
                light_before+=float(loss)/len(train);c.load2device('cpu')
            assert torch.isfinite(dc.grad).all() and torch.isfinite(ec.grad).all()
            gp=torch.cat([dc.grad.flatten(),ec.grad.flatten()]).clone()
            with torch.enable_grad():
                Dp,Ep=lights(B,dc,ec,train,old_d,old_e);pl=pair_loss(pairs,Dp,Ep,True);gd,ge=torch.autograd.grad(pl,[dc,ec])
            gc=torch.cat([gd.flatten(),ge.flatten()])
            assert torch.isfinite(gc).all() and gc.norm()>0
            if weight is None:weight=float(gp.norm()/gc.norm())
            if a.group=='paired':dc.grad.add_(gd,alpha=weight);ec.grad.add_(ge,alpha=weight)
            ol.step()
            row=dict(epoch=epoch,before_material=before,before_light=light_before,direction_gradient_norm=float(dc.grad.norm()),energy_gradient_norm=float(ec.grad.norm()),pair_loss=float(pl),pair_weight=weight,gradient_dot=float(gp.dot(gc)))
            history.append(row);print(json.dumps(row),flush=True)
    final=evaluate();D,E=lights(B,dc,ec,train,old_d,old_e)
    final_pairs={split:float(pair_loss(pairs,D,E,split=='train')) for split in ['train','held']}
    if a.verify:
        assert all(abs(final_pairs[k]-record['final_pairs'][k])<1e-9 for k in final_pairs)
        errors={i:abs(v-record['final']['per_frame'][str(i)]) for i,v in final['per_frame'].items()}
        assert max(errors.values())<1e-9
        assert torch.allclose(D.cpu(),ck['directions'],atol=1e-7) and torch.allclose(E.cpu(),ck['energy'],atol=1e-7)
        assert abs(float(E[train].mean())-1)<1e-6
        (out/'result.json').write_text(json.dumps(dict(status='PASS',frames=len(frames),max_loss_error=max(errors.values()),geometry_exact=True,energy_train_mean=float(E[train].mean()),checkpoint_sha256=hashlib.sha256((Path(a.verify)/'checkpoint.pt').read_bytes()).hexdigest()),indent=2));return
    viz=[train[0],train[len(train)//2],train[-1],held[len(held)//2]]
    for i in viz:
        _,pred,pack=frame(i,D,E);save_image(out/f'{i:03d}_rgb.png',pred);save_image(out/f'{i:03d}_alpha.png',pack['rend_alpha'].expand(3,-1,-1));save_image(out/f'{i:03d}_normal.png',(frames[i]['normal']+1)/2)
        gate=render(cameras[i],pc,pipe,bg,**frames[i]['ds'],override_color=pc.get_binary_feature().expand(-1,3))['render'];save_image(out/f'{i:03d}_dynamic_gate.png',gate);cameras[i].load2device('cpu')
    inputs={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),fit/'checkpoint.pt',fit/'result.json',Path(prior['cache_source'])]+pair_sources}
    torch.save(dict(protocol='block_selfcal_paired_v1',group=a.group,pair_weight=weight,pair_support=support,base_direction=old_d.cpu(),base_energy=old_e.cpu(),geometry=geometry,deform_state_dict=deformation,config=vars(cfg),args=args,albedo=material.cpu(),direction_controls=dc.cpu(),log_energy_controls=ec.cpu(),directions=D.cpu(),energy=E.cpu(),time_basis=B.cpu(),times=times.cpu(),train_frames=train,held_frames=held,geometry_updates=0,material_updates=20,light_updates=20,energy_gauge='mean_on_training_frames_one',material_upper=upper,ambient=.1,material_optimizer=om.state_dict(),light_optimizer=ol.state_dict()),out/'checkpoint.pt')
    shifts=torch.rad2deg(torch.acos((D*old_d).sum(-1).clamp(-1,1)))
    result=dict(protocol='block_selfcal_paired_v1',group=a.group,gt_read=False,pair_weight=weight,initial_pairs=initial_pairs,final_pairs=final_pairs,pair_support=support,initial_fidelity=initial_fidelity,initial=initial,final=final,history=history,inputs=inputs,visualized_frames=viz,train_frames=train,held_frames=held,geometry_updates=0,material_updates=20,light_updates=20,control_change=float((dc-initial_dc).norm()),energy_control_change=float((ec-initial_ec).norm()),energy_train_mean=float(E[train].mean()),direction_shift_from_T05_mean=float(shifts.mean()),elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),scope='冻结几何的分块自标定接口；T05历史初值见过部分保留图像，非严格未见光照泛化；比较显式对应两端共同反射率消元约束；尚未证明独立增量')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
