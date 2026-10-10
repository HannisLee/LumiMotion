"""真实Native Preset checkpoint GGX/BVH与几何梯度、替换灯和HDR检查；0步/noGT标签。"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parents[2]
sys.path.insert(0,str(CWD))
import torch
from scripts.render_stage2 import Stage2RenderRun,save_image
from scene.pbr_deformation import deformation_input
from gaussian_renderer.render_ir import render_ir
from scene.light import EnvLight

run=Stage2RenderRun(ROOT/'model_mlp',35020);ctx=run.context;pc=ctx.pc
assert run.args.pbr_rasterizer_backend=='lumimotion' and not run.args.wo_specular
assert run.args.loss_preset=='pbr_gt_directional_source_rgb_normal'
assert run.args.wo_indirect and run.args.wo_indirect_relight and run.args.pbr_directional_actual_ray_budget
probability=torch.nn.Parameter(pc.get_binary_feature().detach().clone())
parameters=[pc._albedo_dc,pc._roughness,pc._xyz,pc._rotation,probability]
assert run.args.pbr_feature_gate_parameterization=='probability'
def native_gate():
    output=pc.get_binary_feature().detach()+(probability-probability.detach())
    assert torch.equal(output.detach(),pc.get_binary_feature().detach())
    return output
trace_calls=[];original_trace=pc.trace
def recorded_trace(origins,directions,**kwargs):
    trace_calls.append(dict(shape=list(directions.shape),rays=int(directions.shape[0]*directions.shape[1])))
    return original_trace(origins,directions,**kwargs)
pc.trace=recorded_trace
parameter_sha_before=[hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for p in parameters]
for p in parameters:p.requires_grad_(True)
run.args.train_ray=True;rows=[]
for key in ['0001','0075','0149']:
    cam=ctx.camera_by_key[key];ctx.load_camera(cam)
    with torch.enable_grad():
        d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),
             cam.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),iteration=35020,
             feature=native_gate(),camera_center=cam.camera_center)
    d=ctx.pbr_rgb_normal.apply(pc,d)
    d=ctx.pbr_plane_projection.apply(pc,d)
    if run.bvh:pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
    else:pc.build_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']});run.bvh=True
    trace_calls.clear()
    pkg=render_ir(cam,pc,run.args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                  opt=run.args,training=True)
    eligible=int((pkg['rend_alpha'][0]>.9).sum());selected=int(pkg['mask'].sum())
    assert selected==eligible and pkg['photo_rays_per_pixel']==1 and pkg['photo_pixel_budget']==run.args.trace_num_rays
    assert len(trace_calls)==1 and trace_calls[0]['rays']==selected
    loss=(pkg['render'].permute(1,2,0)[pkg['mask']]-cam.original_image_train_light.cuda().permute(1,2,0)[pkg['mask']]).abs().mean()
    gradients=torch.autograd.grad(loss,parameters,allow_unused=False)
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
    assert all(g.abs().sum()>0 for g in gradients)
    rows.append(dict(frame=key,eligible_pixels=eligible,photo_pixels=selected,actual_trace_calls=list(trace_calls),loss=float(loss),
                     gradients={name:float(g.norm()) for name,g in zip(['albedo','roughness','xyz','rotation','probability_gate'],gradients)}))
    prior=ctx.pbr_rgb_normal.loss(pc)
    prior_grad=torch.autograd.grad(prior,pc._rotation)[0]
    assert torch.isfinite(prior) and torch.isfinite(prior_grad).all() and prior_grad.abs().sum()>0
    rows[-1]['TRAIN_rgb_axis_loss']=float(prior)
    rows[-1]['TRAIN_rgb_axis_gradient_L2']=float(prior_grad.norm())
    rows[-1]['TRAIN_rgb_axis_confidence_points']=int(ctx.pbr_rgb_normal.confidence.sum())
    ctx.unload_camera(cam)
    del pkg,d,gradients;torch.cuda.empty_cache()
# 同一个真实checkpoint的独立参数诊断；0次optimizer，不宣称其它Preset已训练。
from copy import deepcopy
from scene.pbr_directional import directional_lights_for_camera
cam=ctx.camera_by_key['0075'];ctx.load_camera(cam)
with torch.no_grad():
    d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),cam.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),iteration=35020,feature=native_gate(),camera_center=cam.camera_center)
d=ctx.pbr_rgb_normal.apply(pc,d)
d=ctx.pbr_plane_projection.apply(pc,d)
pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
base_lights=directional_lights_for_camera(run.args,cam);variants=[]
for name,actual,indirect,L,cap,cost in [('legacy_direct',False,False,1,262144,256),('actual_three_sun',True,False,3,512,3),('actual_with_indirect',True,True,1,262144,257)]:
    # 重建每个独立诊断的projection autograd图，避免重复反向已释放图。
    with torch.enable_grad():
        d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),cam.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),iteration=35020,feature=native_gate(),camera_center=cam.camera_center)
    d=ctx.pbr_rgb_normal.apply(pc,d)
    d=ctx.pbr_plane_projection.apply(pc,d)
    args=deepcopy(run.args);args.pbr_directional_actual_ray_budget=actual;args.wo_indirect=not indirect;args.trace_num_rays=cap
    lights=(base_lights[0].repeat(L,1),base_lights[1].repeat(L,1)/L);trace_calls.clear()
    pkg=render_ir(cam,pc,args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},opt=args,training=True,directional_lights=lights)
    selected=int(pkg['mask'].sum());expected=min(cap//cost,int((pkg['rend_alpha'][0]>.9).sum()))
    assert selected==expected and pkg['photo_rays_per_pixel']==cost
    assert trace_calls[0]['rays']==selected*L
    assert len(trace_calls)==(2 if indirect else 1)
    if indirect:assert trace_calls[1]['rays']==selected*args.diffuse_sample_num
    assert sum(r['rays'] for r in trace_calls)<=(cap if actual else selected*L)
    loss=(pkg['render'].permute(1,2,0)[pkg['mask']]-cam.original_image_train_light.cuda().permute(1,2,0)[pkg['mask']]).abs().mean()
    grads=torch.autograd.grad(loss,parameters,allow_unused=False)
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    variants.append(dict(name=name,selected_pixels=selected,budget=cap,cost=cost,actual_trace_calls=list(trace_calls),gradients_finite_nonzero=True))
    del pkg,grads;torch.cuda.empty_cache()
ctx.unload_camera(cam);del d
pc.trace=original_trace
assert parameter_sha_before==[hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for p in parameters]
dest=ROOT/'inspection/relight_images';dest.mkdir(exist_ok=False)
hdr=EnvLight(path=str(CWD/'example_envmaps/golden_bay_4k_32x16_rot330.hdr'),device='cuda',activation='none')
hdr.requires_grad_(False);hdr.set_transform(torch.tensor([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]],device='cuda'))
hdr.build_mips();hdr.update_pdf();hdr.pbr_samples=16
with torch.no_grad():
    for key in ['0001','0075','0150']:
        cam=ctx.camera_by_key[key]
        for name,kwargs in [('directional',dict(direction=torch.tensor([.5,-.5,1.],device='cuda'))),('hdr',dict(hdr=hdr))]:
            pkg=run.render(cam,**kwargs)
            assert torch.isfinite(pkg['render']).all() and torch.isfinite(pkg['rend_normal']).all()
            save_image(dest/f'{key}_{name}.png',pkg['render'])
        ctx.unload_camera(cam)
ctx.verify_frozen()
record=dict(status='PASS_NATIVE_PRESET_PBR_GGX_BVH_GRADIENTS_AND_RELIGHT',iteration=35020,
    rows=rows,budget_variants=variants,parameter_sha_unchanged=True,source_files=ctx.source_files,native_source=ctx.pbr_rasterizer_provenance,
    probability_coordinate_gradient=True,native_saved_PLY_forward_exact=True,optimizer_steps=0,gt_material_normal_geometry_inputs=False,hdr_samples=16,
    full_training_mae=False,limitations='仅20步smoke checkpoint的功能/非零有限梯度；HDR16仅smoke，最终仍150帧2048，不作为质量或7°结论。')
(ROOT/'inspection/pbr_gradients.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='native_source'},ensure_ascii=False,indent=2),flush=True)
