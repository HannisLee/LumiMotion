"""零更新：使用训练图像Alpha区分覆盖退化的分辨率与视图因素。"""
import argparse
import hashlib
import json
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
from scripts.probe_dynamic_dl_selfcal import read_config, digest_tensor, save_image


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','model','checkpoint','output']:p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join(sys.argv)+'\n')
    ck=Path(args.checkpoint);state=torch.load(ck,map_location='cpu',weights_only=False)
    runtime=json.loads((ck.parent/'runtime.json').read_text())
    cfg=read_config(Path(args.model)/'cfg_args');cfg.load2gpu_on_the_fly=True
    pc=GaussianModel(cfg.sh_degree,no_binary_separation=cfg.no_binary_separation,fea_dim=cfg.hyper_dim)
    ply=Path(args.model)/'point_cloud/iteration_35000/point_cloud.ply'
    paths=[ply,Path(args.model)/'deform/iteration_35000/deform.pth',Path(args.model)/'cfg_args']
    for path in paths:assert hashlib.sha256(path.read_bytes()).hexdigest()==runtime['inputs'][str(path.resolve())]
    assert Path(state['args']['model']).resolve()==Path(args.model).resolve()
    pc.load_ply(str(ply));deform=DeformModel(deform_type=cfg.deform_type,is_blender=cfg.is_blender,hyper_dim=cfg.hyper_dim,pred_color=cfg.pred_color)
    assert deform.load_weights(args.model,35000);deform.deform.eval().requires_grad_(False)
    initial={k:getattr(pc,k).detach().clone() for k in state['geometry']}
    assert all(torch.equal(initial[k].cpu(),state['geometry'][k]) for k in ['_rotation','_scaling'])
    frozen={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
    infos=readCamerasFromTransforms(args.source,'transforms_train.json',cfg.train_light_folder,False)
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,depth_ratio=0.)
    bg=torch.zeros(3,device='cuda');rows=[]
    for resolution in [4,2]:
        cfg.resolution=resolution
        for i,info in enumerate(infos):
            camera=loadCam(cfg,i,info,1.);camera.load2device()
            assert camera.gt_alpha_mask is not None
            observed=camera.gt_alpha_mask.cuda()>.5
            for k,v in initial.items():getattr(pc,k).copy_(v)
            ds=deform.step(initial['_xyz'],camera.fid.reshape(1,1).expand(len(initial['_xyz']),1),feature=pc.get_binary_feature(),camera_center=camera.camera_center)
            ds={k:ds[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']}
            a=render(camera,pc,pipe,bg,**ds,override_color=torch.zeros_like(pc.get_xyz))['rend_alpha']
            for k,v in state['geometry'].items():getattr(pc,k).copy_(v.cuda())
            b=render(camera,pc,pipe,bg,**ds,override_color=torch.zeros_like(pc.get_xyz))['rend_alpha']
            protected=observed&(a>=.5);lost=protected&(b<.5);gained=observed&(a<.5)&(b>=.5)
            row=dict(resolution=resolution,index=i,name=camera.image_name_train_light,split='held' if i%9==4 else 'train',
                     foreground=int(observed.sum()),initial_covered=int(protected.sum()),final_covered=int((observed&(b>=.5)).sum()),lost=int(lost.sum()),gained=int(gained.sum()),
                     limit=int(int(observed.sum())*.001),alpha_change_mean=float((a-b).abs()[observed].mean()))
            row['failed']=row['lost']>row['limit'];rows.append(row)
            if i in [0,4,67,130,134]:
                prefix=out/f'r{resolution}_{i:03d}'
                save_image(str(prefix)+'_initial.png',a);save_image(str(prefix)+'_final.png',b)
                rgb=torch.cat([observed.float(),observed.float(),observed.float()],dim=0)
                rgb[:,lost[0]]=torch.tensor([1.,0.,0.],device='cuda')[:,None]
                save_image(str(prefix)+'_lost.png',rgb)
            if i%30==0:print(json.dumps(row),flush=True)
            camera.load2device('cpu')
    assert frozen=={k:digest_tensor(v) for k,v in deform.deform.state_dict().items()}
    for k,v in state['geometry'].items():assert torch.equal(getattr(pc,k).cpu(),v)
    aggregates={}
    for resolution in [4,2]:
        for split in ['train','held']:
            r=[x for x in rows if x['resolution']==resolution and x['split']==split];n=sum(x['foreground'] for x in r)
            aggregates[f'r{resolution}_{split}']=dict(frames=len(r),failed_frames=sum(x['failed'] for x in r),
                 initial_coverage=sum(x['initial_covered'] for x in r)/n,final_coverage=sum(x['final_covered'] for x in r)/n,
                 lost_fraction=sum(x['lost'] for x in r)/n,worst_lost_fraction=max(x['lost']/x['foreground'] for x in r))
    inputs=paths+[ck,Path(__file__),Path(args.source)/'transforms_train.json']
    result=dict(status='PASS',scope='零优化诊断，不代表几何成功',gt_read=False,optimizer_updates=0,
                args=vars(args),inputs={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},summary=aggregates,rows=rows)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(aggregates),flush=True)

if __name__=='__main__':main()
