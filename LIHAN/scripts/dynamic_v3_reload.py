"""在TRAIN固定帧验证Stage1权重重载、动态前向与真实几何/形变梯度。"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from gaussian_renderer import render
from scene import GaussianModel, DeformModel
from scene.dataset_readers import readCamerasFromTransforms
from scripts.probe_dynamic_dl_selfcal import read_config, save_image
from utils.camera_utils import loadCam


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, model, out = [p.resolve() for p in (args.source,args.model,args.output)]
    out.mkdir(parents=True,exist_ok=False)
    from utils.no_gt_guard import install_no_gt_guard
    install_no_gt_guard([source,model,out])
    record = json.loads((model/'training_finished.json').read_text())
    assert record['args']['iterations'] == args.iteration
    for relative, expected in record['checkpoint_sha256'].items():
        assert hashlib.sha256((model/relative).read_bytes()).hexdigest() == expected
    cfg = read_config(model/'cfg_args')
    cfg.load2gpu_on_the_fly = True
    cloud = GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    cloud.load_ply(str(model/f'point_cloud/iteration_{args.iteration}/point_cloud.ply'))
    cloud.active_sh_degree=min(cfg.sh_degree,args.iteration//1000)
    deform = DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
    assert deform.load_weights(str(model),args.iteration)
    saved = torch.load(model/f'deform/iteration_{args.iteration}/deform.pth',map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),saved[k]) for k,v in deform.deform.state_dict().items())
    original = {k:digest(v) for k,v in deform.deform.state_dict().items()}
    original_xyz=digest(cloud.get_xyz)
    infos=readCamerasFromTransforms(str(source),'transforms_train.json','images',False)
    wanted={1,81,149}
    infos=[info for info in infos if int(info.image_name_train_light.split('_')[-1]) in wanted]
    assert len(infos)==3
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=1.)
    background=torch.zeros(3,device='cuda')
    rows=[]
    for i,info in enumerate(infos):
        camera=loadCam(cfg,i,info,1.);camera.load2device()
        ds=deform.step(cloud.get_xyz,camera.fid.reshape(1,1).expand(len(cloud.get_xyz),1),
            iteration=max(1001,args.iteration),feature=cloud.get_binary_feature(),camera_center=camera.camera_center)
        ds={k:ds[k] for k in ('d_xyz','d_rotation','d_scaling','d_opacity','d_color')}
        # 真正动态前向单独测梯度；20步资产仍属warmup，不能称已恢复动作。
        package=render(camera,cloud,pipe,background,**ds)
        target=camera.original_image_train_light.cuda()
        loss=(package['render']-target).abs().mean()
        loss=loss+((package['rend_alpha']-camera.gt_alpha_mask.cuda())**2).mean()*.1
        loss.backward()
        def grad(tensor):
            return None if tensor.grad is None else dict(norm=float(tensor.grad.norm()),finite=bool(torch.isfinite(tensor.grad).all()))
        gradients={k:grad(v) for k,v in [('xyz',cloud._xyz),('rotation',cloud._rotation),('scale',cloud._scaling),('opacity',cloud._opacity)]}
        gradients.update({'deform.'+k:grad(v) for k,v in deform.deform.named_parameters() if k in ('gaussian_warp.weight','gaussian_rotation.weight','gaussian_scaling.weight','gaussian_color.weight')})
        assert all(v is None or v['finite'] for v in gradients.values())
        assert gradients['xyz'] and gradients['xyz']['norm']>0
        assert any(v and v['norm']>0 for k,v in gradients.items() if k.startswith('deform.'))
        frame=int(info.image_name_train_light.split('_')[-1])
        with torch.no_grad():
            gate=cloud.get_binary_feature()
            sep=render(camera,cloud,pipe,background,**ds,
                override_color=torch.cat([gate,1-gate,torch.zeros_like(gate)],dim=1))
            for suffix,tensor in [('rgb',package['render']),('alpha',package['rend_alpha']),
                                  ('gs_normal',(F.normalize(package['rend_normal'],dim=0)*.5+.5)*package['rend_alpha']),
                                  ('depth_normal',(F.normalize(package['surf_normal'],dim=0)*.5+.5)*package['rend_alpha']),
                                  ('separation',sep['render'])]:
                assert torch.isfinite(tensor).all()
                save_image(out/f'{frame:04d}_{suffix}.png',tensor)
            again=render(camera,cloud,pipe,background,**ds)
            repeat_error=float((package['render']-again['render']).abs().max())
            assert repeat_error==0
        rows.append(dict(frame=frame,time=float(camera.fid),loss=float(loss),gradients=gradients,
                         repeated_render_max_abs_error=repeat_error,
                         native_checkpoint_warmup=args.iteration<1000,dynamic_probe_iteration=max(1001,args.iteration)))
        for v in vars(cloud).values():
            if isinstance(v,torch.Tensor) and v.is_leaf:v.grad=None
        deform.deform.zero_grad(set_to_none=True)
    assert digest(cloud.get_xyz)==original_xyz
    assert {k:digest(v) for k,v in deform.deform.state_dict().items()}==original
    result=dict(status='PASS_RELOAD_AND_DYNAMIC_GRADIENT_ENGINEERING',iteration=args.iteration,
                model=str(model),source=str(source),gt_read=False,weights_unchanged=True,
                rows=rows,scope='TRAIN三帧重载/动态前向/有限非零梯度；不是完整遮挡、PS或法线精度验收')
    (out/'reload_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
