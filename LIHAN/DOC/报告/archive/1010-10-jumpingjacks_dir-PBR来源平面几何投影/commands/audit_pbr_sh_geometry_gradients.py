"""实际B05 40k检查点三训练帧PBR/SH/几何梯度分解；不做优化、不读GT材质/法线/几何。"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
CWD=ROOT.parent.parent
sys.path.insert(0,str(CWD))
import torch
from torch.nn import functional as F
from types import SimpleNamespace
from scripts.render_stage2 import Stage2RenderRun
from scripts.loss_stage2 import build_loss_preset
from utils.loss_utils import l1_loss,ssim
from scene.pbr_deformation import deformation_input
from gaussian_renderer.render_ir import render_ir

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.manual_seed(1010)
run=Stage2RenderRun(ROOT/'model_mlp',40000)
ctx=run.context;pc=ctx.pc
pc._xyz.requires_grad_(True);pc._rotation.requires_grad_(True)
parameters=[pc._xyz,pc._rotation]
preset=build_loss_preset(run.args,'original_ir')
run.args.train_ray=True
rows=[]
for frame in ['0001','0075','0149']:
    camera=ctx.camera_by_key[frame]
    assert camera in ctx.train_cameras
    ctx.load_camera(camera)
    # 和真实Trainer相同：sourceMLP无梯度，actual Plane projection在其外求导。
    with torch.no_grad():
        deformation=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),
          camera.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),iteration=40000,
          feature=pc.get_binary_feature(),camera_center=camera.camera_center)
    deformation=ctx.pbr_plane_projection.apply(pc,deformation)
    if not run.bvh:
        pc.build_bvh(**{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling']})
        run.bvh=True
    else:pc.update_bvh(**{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling']})
    pkg=render_ir(camera,pc,run.args,ctx.background,
                  **{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                  opt=run.args,training=True)
    target=camera.original_image_train_light.cuda()
    photo=preset.compute(SimpleNamespace(env_light=None),pkg,target,pkg['render']).total
    physical=F.l1_loss(pkg['render'].permute(1,2,0)[pkg['mask']],target.permute(1,2,0)[pkg['mask']])
    sh_mask=(pkg['rend_alpha']>.9).float().expand_as(pkg['render_sh'])
    sh_image=pkg['render_sh']*sh_mask;sh_target=target*sh_mask
    sh_loss=.8*l1_loss(sh_image,sh_target)+.2*(1-ssim(sh_image,sh_target))
    assert torch.allclose(photo,physical+sh_loss,atol=1e-7,rtol=0)
    groups={'all':torch.ones(len(pc.get_xyz),dtype=torch.bool,device='cuda'),
            'projected_plane':ctx.pbr_plane_projection.support,
            'dynamic':pc.get_binary_feature()[:,0]>=.5,
            'static_unprojected':(pc.get_binary_feature()[:,0]<.5)&~ctx.pbr_plane_projection.support}
    normal=F.normalize(pkg['rend_normal'],dim=0)
    depth=F.normalize(pkg['surf_normal'],dim=0)
    valid=(pkg['rend_alpha'].detach()>.5)&(pkg['rend_normal'].detach().norm(dim=0,keepdim=True)>1e-6)
    valid &= pkg['surf_normal'].detach().norm(dim=0,keepdim=True)>1e-6
    valid=(F.avg_pool2d(valid.float()[None],3,stride=1,padding=1)[0]>=1).detach()
    def loss(n,d):
        return ((1-(n*d).sum(0,keepdim=True).clamp(-1,1))*valid).sum()/valid.sum().clamp_min(1)
    terms={'native_rgb_and_SH':photo,'physical_PBR_rgb':physical,'SH_rgb':sh_loss,'normal_depth_full':loss(normal,depth)*run.args.lambda_pbr_normal_depth,
           'normal_depth_GS_side':loss(normal,depth.detach())*run.args.lambda_pbr_normal_depth,
           'normal_depth_depth_side':loss(normal.detach(),depth)*run.args.lambda_pbr_normal_depth}
    gradients={}
    values={}
    for name,value in terms.items():
        assert torch.isfinite(value)
        grads=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
        grads=[torch.zeros_like(p) if g is None else g for p,g in zip(parameters,grads)]
        assert all(torch.isfinite(g).all() for g in grads)
        gradients[name]=grads;values[name]=float(value)
    row=dict(frame=frame,valid_pixels=int(valid.sum()),losses=values,parameters={})
    for i,name in enumerate(['xyz','rotation']):
        full=gradients['normal_depth_full'][i]
        summed=gradients['normal_depth_GS_side'][i]+gradients['normal_depth_depth_side'][i]
        delta=float((full-summed).abs().max())
        assert torch.allclose(full,summed,rtol=1e-4,atol=1e-6)
        photo_grad=gradients['native_rgb_and_SH'][i]
        physical_g=gradients['physical_PBR_rgb'][i];sh_g=gradients['SH_rgb'][i]
        assert torch.allclose(photo_grad,physical_g+sh_g,rtol=1e-4,atol=1e-6)
        row['parameters'][name]=dict(gradient_L2={k:float(g[i].norm()) for k,g in gradients.items()},
            full_branch_sum_max_difference=delta,groups={})
        for group,mask in groups.items():
            a=physical_g[mask].flatten();b=sh_g[mask].flatten();c=full[mask].flatten()
            row['parameters'][name]['groups'][group]=dict(points=int(mask.sum()),
                gradient_L2={k:float(g[i][mask].norm()) for k,g in gradients.items()},
                PBR_SH_gradient_cosine=float(F.cosine_similarity(a,b,dim=0)),
                PBR_ND_gradient_cosine=float(F.cosine_similarity(a,c,dim=0)))
    rows.append(row)
    ctx.unload_camera(camera)
    del pkg,gradients,terms,photo,normal,depth
    torch.cuda.empty_cache()
ctx.verify_frozen()
record=dict(status='PASS_READ_ONLY_GRADIENT_DIAGNOSTIC',iteration=40000,frames=[1,75,149],
    rows=rows,model_files=ctx.source_files,gt_material_normal_geometry_inputs=False,
    original_test_RGB_used=False,optimizer_steps=0,current_training_modified=False,final_mae=False,
    limitations='三训练帧PBR/SH梯度量级与冲突仅证明优化机制，不证明GT准确性；不替代全15帧MAE或完整训练。')
(ROOT/'inspection/pbr_sh_geometry_gradients.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
