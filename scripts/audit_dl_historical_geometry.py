"""历史几何损失的动态参数梯度适配审计；不更新模型，不读取GT。"""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.stage2_photometric import srgb_to_linear, linear_to_srgb
from scripts.probe_dynamic_dl_selfcal import read_config, digest_tensor, save_image
from utils.camera_utils import loadCam


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','model','lights','checkpoint','output']:p.add_argument('--'+k,required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    torch.manual_seed(1015)
    cfg=read_config(Path(args.model)/'cfg_args');cfg.resolution=4;cfg.load2gpu_on_the_fly=True
    light=json.loads(Path(args.lights).read_text());assert not light.get('gt_read',False)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert not ck['args'].get('gt_direction_reference',False)
    assert Path(ck['args']['lights']).resolve()==Path(args.lights).resolve()
    assert ck['args']['mode']=='frozen' and not ck['args'].get('refine_light',False)
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    ply=Path(args.model)/'point_cloud/iteration_35000/point_cloud.ply';pc.load_ply(str(ply))
    deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
    assert deform.load_weights(args.model,35000)
    assert all(torch.equal(v.cpu(),ck['geometry'][k]) for k,v in vars(pc).items() if k in ck['geometry'])
    assert all(torch.equal(v.cpu(),ck['deform_state_dict'][k]) for k,v in deform.deform.state_dict().items())
    deform.deform.eval().requires_grad_(False)
    for v in vars(pc).values():
        if isinstance(v,torch.Tensor):v.requires_grad_(False)
    params={k:getattr(pc,k) for k in ['_xyz','_rotation','_scaling','_opacity']}
    for v in params.values():v.requires_grad_(True)
    params.update({'head.'+k:v for k,v in deform.deform.gaussian_warp.named_parameters()})
    for v in params.values():v.requires_grad_(True)
    tensors={k:v for k,v in vars(pc).items() if isinstance(v,torch.Tensor)}
    def hashes():return {**{k:digest_tensor(v) for k,v in tensors.items()},**{'deform.'+k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}}
    before=hashes()
    infos=readCamerasFromTransforms(args.source,'transforms_train.json',cfg.train_light_folder,False)
    cameras=[loadCam(cfg,i,info,1.) for i,info in enumerate(infos)]
    assert [c.image_name_train_light for c in cameras]==light['frames']
    times=torch.tensor([float(c.fid) for c in cameras],device='cuda')
    assert torch.allclose(times,torch.tensor(light['times'],device='cuda'),atol=1e-6)
    train=light['train_indices'];indices=[0,1,2,64,65,66,132,133,134];assert set(indices)<=set(train)
    energy=F.softplus(ck['energy'].cuda());energy=energy/energy.mean()
    albedo=F.softplus(ck['albedo'].cuda());ambient=float(ck.get('ambient',0.))
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    bg=torch.zeros(3,device='cuda');rows=[]
    paths=[Path(__file__),Path(args.checkpoint),Path(args.lights),ply,Path(args.model)/'cfg_args',Path(args.model)/'deform/iteration_35000/deform.pth',Path(args.source)/'transforms_train.json',Path('gaussian_renderer/__init__.py')]
    (out/'runtime.json').write_text(json.dumps(dict(args=vars(args),inputs={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},native=sys.modules['diff_surfel_rasterization'].provenance,indices=indices,gt_read=False,ambient=ambient),ensure_ascii=False,indent=2))
    for i in indices:
        c=cameras[i];c.load2device();mask=c.gt_alpha_mask.cuda();target=srgb_to_linear(c.original_image_train_light*mask)
        # 完整重查询，保留canonical XYZ到所有形变输出的Jacobian；不复用旧缓存形变。
        ds=deform.step(pc.get_xyz,c.fid.reshape(1,1).expand(len(pc.get_xyz),1),feature=pc.get_binary_feature(),camera_center=c.camera_center)
        ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
        colors=albedo*energy[train.index(i)]
        full=render(c,pc,pipe,bg,**ds,override_color=colors)
        stop=render(c,pc,pipe,bg,**ds,override_color=colors,detach_opacity=True)
        interior=-F.max_pool2d(-(mask>.99).float()[None],5,stride=1,padding=2)[0]
        interior=interior*(full['rend_alpha'].detach()>.5)
        def normal_loss(pkg):
            n=F.normalize(pkg['rend_normal'],dim=0);d=F.normalize(pkg['surf_normal'],dim=0)
            return ((1-(n*d).sum(0,keepdim=True))*interior).sum()/interior.sum().clamp_min(1)
        n=F.normalize(full['surf_normal'],dim=0)
        sign=torch.where((n*(c.camera_center[:,None,None]-full['surf_point'])).sum(0,keepdim=True).detach()>=0,1.,-1.)
        l=torch.tensor(light['directions'][i],device='cuda')
        rgb=full['render']*((n*sign*l[:,None,None]).sum(0,keepdim=True).clamp_min(0)+ambient)
        raw=np.array(Image.open(c.image_path_train_light).convert('RGB'))
        saturated=F.adaptive_max_pool2d(torch.from_numpy((raw>=254).astype(np.float32)).permute(2,0,1)[None].cuda(),target.shape[-2:])[0].bool()
        residual=rgb-target;residual=torch.where(saturated,residual.clamp_max(0),residual)
        # 本审计用全部原图前景，不把它冒充旧亮面轨迹photo的逐值复现。
        terms={'photo':(residual.square()*mask).sum()/(3*mask.sum()).clamp_min(1),'normal':normal_loss(full),'normal_detach':normal_loss(stop),'alpha':F.binary_cross_entropy(full['rend_alpha'].clamp(1e-6,1-1e-6),mask),'distortion':full['rend_dist'].mean()}
        grads={name:torch.autograd.grad(loss,tuple(params.values()),retain_graph=True,allow_unused=True) for name,loss in terms.items()}
        norms={name:{k:0. if g is None else float(g.norm()) for k,g in zip(params,gg)} for name,gg in grads.items()}
        diffs={};finite=True
        for name,gg in grads.items():finite=finite and all(g is None or bool(torch.isfinite(g).all()) for g in gg)
        for k,a,b in zip(params,grads['normal'],grads['normal_detach']):
            aa=torch.zeros_like(params[k]) if a is None else a;bb=torch.zeros_like(params[k]) if b is None else b
            diffs[k]=float((aa-bb).abs().max())
        parity={k:float((full[k]-stop[k]).abs().max()) for k in ['render','rend_alpha','rend_normal','surf_normal','surf_depth','rend_dist']}
        opacity_idx=list(params).index('_opacity');g=grads['normal_detach'][opacity_idx]
        removed=g is None or not bool(torch.count_nonzero(g))
        preserved=all(v<=1e-6 for k,v in diffs.items() if k!='_opacity')
        row=dict(index=i,frame=c.image_name_train_light,values={k:float(v.detach()) for k,v in terms.items()},gradient_norms=norms,normal_route_max_difference=diffs,forward_max_difference=parity,finite=finite,normal_opacity_removed=removed,other_normal_gradients_preserved=preserved)
        rows.append(row);print(json.dumps(row),flush=True)
        if i in [0,65,134]:
            name=str(i)
            save_image(out/(name+'_rgb.png'),linear_to_srgb(rgb.detach().clamp_min(0)))
            save_image(out/(name+'_alpha.png'),full['rend_alpha'].detach())
            save_image(out/(name+'_normal.png'),(F.normalize(full['rend_normal'].detach(),dim=0)*.5+.5)*full['rend_alpha'].detach())
            with torch.no_grad():
                gate=pc.get_binary_feature();sep=render(c,pc,pipe,bg,**ds,override_color=torch.cat([gate,1-gate,torch.zeros_like(gate)],1))
                save_image(out/(name+'_separation.png'),sep['render'])
        del grads,terms,full,stop,ds
        c.load2device('cpu')
    unchanged=before==hashes()
    passed=unchanged and all(r['finite'] and r['normal_opacity_removed'] and r['other_normal_gradients_preserved'] and max(r['forward_max_difference'].values())==0 for r in rows)
    result=dict(status='PASS' if passed else 'FAILED',parameter_updates=0,gt_read=False,unchanged=unchanged,rows=rows,scope='九训练帧真实动态梯度支路审计；非训练、非有限差分通过、非几何成功')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert passed

if __name__=='__main__':main()
