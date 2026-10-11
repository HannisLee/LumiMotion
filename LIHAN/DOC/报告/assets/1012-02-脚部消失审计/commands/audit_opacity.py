"""只读反事实诊断：在9000步副本模拟透明度重置，不执行任何优化器更新。"""
from pathlib import Path
from argparse import Namespace
from copy import copy
import hashlib,json,sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
from scripts.eval_geometry_gt import GeometryRun
from scene.static_dl_photometric import DirectionOnlyLights,StaticVisibility,render_photometric
from scene.static_dl_censored import censored_photometric_loss
from gaussian_renderer import render as render_native
from utils.general_utils import PILtoTorch,inverse_sigmoid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

results=[]
for prefix,tag in [('1011-27','T13'),('1011-28','T14')]:
    model=next((ROOT/'output').glob(prefix+'*/model_mlp'))
    run=GeometryRun(model,9000)
    config=run.config
    pipe=Namespace(**config)
    metadata=json.loads((run.source/'transforms_train.json').read_text())
    assert len(metadata['frames'])==135
    lights=DirectionOnlyLights(config['static_dl_directions'])
    lighting=json.loads((model/'static_dl_lighting/iteration_9000/lighting.json').read_text())
    with torch.no_grad():lights.log_intensity.copy_(torch.tensor(lighting['intensity_from_train_rgb'],device='cuda').log())
    lights.requires_grad_(False)
    pc=run.pc
    original_opacity=pc._opacity.detach().clone()
    z=pc.get_xyz[:,2].detach()
    groups={'feet_height':(z>.05)&(z<.35),'platform_height':abs(z)<.02,'upper_body':z>.35}
    source_hashes=dict(run.files)
    for state in ('saved','reset'):
        with torch.no_grad():
            pc._opacity.copy_(original_opacity if state=='saved' else inverse_sigmoid(torch.sigmoid(original_opacity).clamp_max(.01)))
        vis=StaticVisibility(config['static_dl_shadow_bias'],1)
        totals={k:torch.zeros_like(pc._opacity) for k in ('photo','alpha','normal','distortion')}
        loss_sums={k:0. for k in totals}
        per_frame=[]
        for i,record in enumerate(metadata['frames']):
            camera,_,_,image_path=run.camera(metadata,record,2)
            with Image.open(image_path) as im:input_tensor=PILtoTorch(im,(400,400)).cuda()
            target=input_tensor[:3];alpha_gt=input_tensor[3:4]
            package=render_photometric(camera,pc,pipe,torch.zeros(3,device='cuda'),lights,vis,9001)
            photo,_=censored_photometric_loss(package['render'],target,alpha_gt,config['static_dl_shadow_weight'],config['static_dl_highlight_weight'],config['static_dl_residual_threshold'])
            alpha=package['rend_alpha'].clamp(1e-6,1-1e-6)
            expected_pipe=copy(pipe);expected_pipe.depth_ratio=0.
            geo=render_native(camera,pc,expected_pipe,torch.zeros(3,device='cuda'),0.,0.,0.,override_color=torch.zeros_like(pc.get_xyz))
            mask=(alpha_gt>.99).float();mask=-F.max_pool2d(-mask[None],5,stride=1,padding=2)[0];mask=mask*(alpha.detach()>.5)
            normal=((1-(F.normalize(geo['rend_normal'],dim=0)*F.normalize(geo['surf_normal'],dim=0)).sum(0,keepdim=True))*mask).sum()/mask.sum().clamp_min(1)
            losses=dict(photo=photo,alpha=config['lambda_alpha_loss']*F.binary_cross_entropy(alpha,alpha_gt),normal=config['static_dl_normal_weight']*normal,distortion=config['lambda_dist']*package['rend_dist'].mean())
            row={'frame':record['source_frame'],'groups':{}}
            for name,loss in losses.items():
                grad=torch.autograd.grad(loss,pc._opacity,retain_graph=True)[0].detach()
                assert torch.isfinite(grad).all()
                totals[name]+=grad;loss_sums[name]+=float(loss.detach())
                row[name]=float(loss.detach())
                for group,m in groups.items():
                    row['groups'].setdefault(group,{})[name+'_signed_gradient_sum']=float(grad[m].sum())
            per_frame.append(row)
            del package,geo,losses,photo,normal,alpha,loss,grad,camera
            if (i+1)%45==0:print(tag,state,i+1,flush=True)
        totals={k:v/135 for k,v in totals.items()};totals['total']=sum(totals.values())
        result=dict(experiment=tag,state=state,checkpoint=9000,parameter_updates=0,frames=135,weighted_losses={k:v/135 for k,v in loss_sums.items()},groups={})
        for group,mask in groups.items():
            result['groups'][group]={'points':int(mask.sum())}
            for k,g in totals.items():
                a=g[mask].cpu().numpy().ravel(); result['groups'][group][k]=dict(signed_sum=float(a.sum()),absolute_sum=float(abs(a).sum()),positive_fraction=float((a>0).mean()),negative_fraction=float((a<0).mean()),zero_fraction=float((a==0).mean()))
        np.savez_compressed(OUT/f'opacity_gradient_{tag}_{state}.npz',z=z.cpu().numpy(),**{k:v.cpu().numpy().ravel() for k,v in totals.items()})
        (OUT/f'opacity_frames_{tag}_{state}.json').write_text(json.dumps(per_frame,ensure_ascii=False,indent=2)+'\n')
        results.append(result)
        (OUT/'opacity_gradient_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    for p,h in source_hashes.items():assert sha(p)==h,p
    with torch.no_grad():pc._opacity.copy_(original_opacity)
    print(tag,'完成；检查点SHA未变；优化器更新0次',flush=True)
    del run,pc,totals
    torch.cuda.empty_cache()
