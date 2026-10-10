"""实际40k检查点三训练帧梯度分解；不做优化、不读GT材质/法线/几何。"""
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
    deformation=ctx.deformation(camera)
    if not run.bvh:
        pc.build_bvh(**{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling']})
        run.bvh=True
    else:pc.update_bvh(**{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling']})
    pkg=render_ir(camera,pc,run.args,ctx.background,
                  **{k:deformation[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                  opt=run.args,training=True)
    target=camera.original_image_train_light.cuda()
    photo=preset.compute(SimpleNamespace(env_light=None),pkg,target,pkg['render']).total
    normal=F.normalize(pkg['rend_normal'],dim=0)
    depth=F.normalize(pkg['surf_normal'],dim=0)
    valid=(pkg['rend_alpha'].detach()>.5)&(pkg['rend_normal'].detach().norm(dim=0,keepdim=True)>1e-6)
    valid &= pkg['surf_normal'].detach().norm(dim=0,keepdim=True)>1e-6
    valid=(F.avg_pool2d(valid.float()[None],3,stride=1,padding=1)[0]>=1).detach()
    def loss(n,d):
        return ((1-(n*d).sum(0,keepdim=True).clamp(-1,1))*valid).sum()/valid.sum().clamp_min(1)
    terms={'native_rgb_and_SH':photo,'normal_depth_full':loss(normal,depth)*run.args.lambda_pbr_normal_depth,
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
        row['parameters'][name]=dict(gradient_L2={k:float(g[i].norm()) for k,g in gradients.items()},
            full_branch_sum_max_difference=delta,
            RGB_SH_and_full_consistency_gradient_cosine=float(F.cosine_similarity(photo_grad.flatten(),full.flatten(),dim=0)),
            RGB_SH_and_GS_side_gradient_cosine=float(F.cosine_similarity(photo_grad.flatten(),gradients['normal_depth_GS_side'][i].flatten(),dim=0)))
    rows.append(row)
    ctx.unload_camera(camera)
    del pkg,gradients,terms,photo,normal,depth
    torch.cuda.empty_cache()
ctx.verify_unchanged()
record=dict(status='PASS_READ_ONLY_GRADIENT_DIAGNOSTIC',iteration=40000,frames=[1,75,149],
    rows=rows,model_files=ctx.source_files,gt_material_normal_geometry_inputs=False,
    original_test_RGB_used=False,optimizer_steps=0,current_training_modified=False,final_mae=False,
    limitations='三训练帧梯度量级与冲突仅证明优化机制，不证明GT准确性；不替代全15帧MAE或完整训练。')
(ROOT/'inspection/normal_depth_gradients.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
