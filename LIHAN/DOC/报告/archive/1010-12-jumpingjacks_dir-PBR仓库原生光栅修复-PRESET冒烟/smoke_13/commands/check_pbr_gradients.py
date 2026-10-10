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
parameters=[pc._albedo_dc,pc._roughness,pc._xyz,pc._rotation]
for p in parameters:p.requires_grad_(True)
run.args.train_ray=True;rows=[]
for key in ['0001','0075','0149']:
    cam=ctx.camera_by_key[key];ctx.load_camera(cam)
    with torch.no_grad():
        d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),
             cam.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),iteration=35020,
             feature=pc.get_binary_feature(),camera_center=cam.camera_center)
    d=ctx.pbr_plane_projection.apply(pc,d)
    if run.bvh:pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
    else:pc.build_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']});run.bvh=True
    pkg=render_ir(cam,pc,run.args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},
                  opt=run.args,training=True)
    loss=(pkg['render'].permute(1,2,0)[pkg['mask']]-cam.original_image_train_light.cuda().permute(1,2,0)[pkg['mask']]).abs().mean()
    gradients=torch.autograd.grad(loss,parameters,allow_unused=False)
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
    assert all(g.abs().sum()>0 for g in gradients)
    rows.append(dict(frame=key,photo_pixels=int(pkg['mask'].sum()),loss=float(loss),
                     gradients={name:float(g.norm()) for name,g in zip(['albedo','roughness','xyz','rotation'],gradients)}))
    ctx.unload_camera(cam)
    del pkg,d,gradients;torch.cuda.empty_cache()
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
    rows=rows,source_files=ctx.source_files,native_source=ctx.pbr_rasterizer_provenance,
    optimizer_steps=0,gt_material_normal_geometry_inputs=False,hdr_samples=16,
    full_training_mae=False,limitations='仅20步smoke checkpoint的功能/非零有限梯度；HDR16仅smoke，最终仍150帧2048，不作为质量或7°结论。')
(ROOT/'inspection/pbr_gradients.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='native_source'},ensure_ascii=False,indent=2),flush=True)
