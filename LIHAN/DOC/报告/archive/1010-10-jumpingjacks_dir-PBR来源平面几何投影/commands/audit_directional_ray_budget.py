"""Native SUN像素预算/关闭SH间接光的只读GPU机制审计；不代表新训练结果。"""
from pathlib import Path
import sys,json,copy,hashlib,time
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root.parent.parent))
import torch
from scripts.render_stage2 import Stage2RenderRun
from scene.pbr_deformation import deformation_input
from gaussian_renderer.render_ir import render_ir
run=Stage2RenderRun(root/'model_mlp',40000);ctx=run.context;pc=ctx.pc
pc._xyz.requires_grad_(True);pc._rotation.requires_grad_(True)
base=copy.copy(run.args);dynamic=pc.get_binary_feature()[:,0]>=.5;rows=[];bvh=False
for frame in ['0001','0075','0149']:
 camera=ctx.camera_by_key[frame];assert camera in ctx.train_cameras;ctx.load_camera(camera)
 for mode,disabled,cap in [('native_sampled',False,base.trace_num_rays),('direct_sampled',True,base.trace_num_rays),('direct_dense',True,400*400*(base.diffuse_sample_num+base.light_sample_num))]:
  args=copy.copy(base);args.wo_indirect=disabled;args.train_ray=True;args.trace_num_rays=cap
  torch.manual_seed(1010)
  with torch.no_grad():
   d=ctx.deform.step(deformation_input(pc.get_xyz,ctx.pbr_deformation_xyz),camera.fid.unsqueeze(0).expand(len(pc.get_xyz),-1),
       iteration=40000,feature=pc.get_binary_feature(),camera_center=camera.camera_center)
  d=ctx.pbr_plane_projection.apply(pc,d)
  if not bvh:pc.build_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']});bvh=True
  else:pc.update_bvh(**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling']})
  torch.cuda.synchronize();started=time.perf_counter()
  pkg=render_ir(camera,pc,args,ctx.background,**{k:d[k] for k in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},opt=args,training=True)
  gt=camera.original_image_train_light.cuda();mask=pkg['mask'];loss=(pkg['render'].permute(1,2,0)[mask]-gt.permute(1,2,0)[mask]).abs().mean()
  gradients=torch.autograd.grad(loss,[pc._xyz,pc._rotation]);assert all(torch.isfinite(g).all() for g in gradients) and torch.isfinite(loss)
  torch.cuda.synchronize();seconds=time.perf_counter()-started
  suns=len(run.pbr_lights.for_camera(camera,args.pbr_light_intensity_scale)[0]);pixels=int(mask.sum())
  row=dict(frame=frame,mode=mode,wo_indirect=disabled,configured_trace_num_rays=cap,suns=suns,
      selected_photo_pixels=pixels,eligible_alpha09_pixels=int((pkg['rend_alpha'][0]>.9).sum()),
      expected_actual_trace_rays=pixels*(suns+(0 if disabled else args.diffuse_sample_num)),
      physical_rgb_L1_sampled=float(loss),forward_and_gradient_seconds=seconds,
      parameters={name:dict(gradient_L2=float(g.norm()),nonzero_points=int((g.abs().sum(1)>0).sum()),
        dynamic_nonzero_points=int(((g.abs().sum(1)>0)&dynamic).sum()),dynamic_total_points=int(dynamic.sum())) for name,g in zip(['xyz','rotation'],gradients)})
  rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  del gradients,pkg,d,loss;torch.cuda.empty_cache()
 ctx.unload_camera(camera)
ctx.verify_frozen();assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in ctx.source_files.items())
report=dict(status='PASS_READ_ONLY_DIRECTIONAL_RAY_BUDGET_DIAGNOSIS',iteration=40000,training_frames=[1,75,149],optimizer_steps=0,GT_material_normal_geometry_read=False,test_RGB_read=False,training_configuration_changed=False,rows=rows,limitation='已训练with-indirect材质/光强在no-indirect下未重新训练；sampled/dense像素集不同，L1不用于比较最终质量。只证明工作量/梯度覆盖，必须完整新训练及原15帧七MAE。')
(root/'inspection/directional_ray_budget.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
