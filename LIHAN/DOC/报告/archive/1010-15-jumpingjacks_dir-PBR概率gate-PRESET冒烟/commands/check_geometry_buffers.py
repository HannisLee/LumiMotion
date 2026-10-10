"""真实GeometryRun入口与原生PBR几何buffer逐像素比较；0优化步，无GT标签。"""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()))
import torch,numpy as np
from argparse import Namespace
from scripts.eval_geometry_gt import GeometryRun
from gaussian_renderer.render_ir import render_ir
from utils.geometry_eval_utils import unit_normal
from scene.pbr_geometry import quaternion_normal
root=Path.cwd();out=Path(__file__).resolve().parent.parent
models=[(out/'model_mlp',35020)]
results=[]
for model,iteration in models:
 run=GeometryRun(model,iteration);assert run.context is not None
 pipe=Namespace(**run.config);pipe.depth_ratio=0
 meta=json.loads((run.source/'transforms_test.json').read_text());rows=[]
 for row in [meta['frames'][i] for i in [0,7,14]]:
  camera,alpha,k,image=run.camera(meta,row,2);c2w=np.asarray(row['transform_matrix'],dtype=np.float64)
  with torch.no_grad():
   d=run.deformation(camera);expected=run.context.deformation(camera)
   for key in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']:
    if torch.is_tensor(d[key]):assert torch.equal(d[key],expected[key]),key
   pkg=render_ir(camera,run.pc,pipe,torch.zeros(3,device='cuda'),
     **{key:d[key] for key in ['d_xyz','d_rotation','d_scaling','d_opacity','d_color']},material_only=True)
   actual=run.render(camera,c2w,k)
   native_normal,_=unit_normal(pkg['rend_normal'].permute(1,2,0).cpu().numpy())
   native_alpha=pkg['rend_alpha'][0].cpu().numpy();native_depth=pkg['surf_depth'][0].cpu().numpy()
   np.testing.assert_allclose(actual['normal'],native_normal,atol=1e-6,rtol=0)
   np.testing.assert_allclose(actual['alpha'],native_alpha,atol=1e-6,rtol=0)
   np.testing.assert_allclose(actual['expected'],native_depth,atol=1e-6,rtol=0)
   record=dict(frame=row['source_frame'],normal_max_abs_error=float(np.abs(actual['normal']-native_normal).max()),
     alpha_max_abs_error=float(np.abs(actual['alpha']-native_alpha).max()),expected_Z_max_abs_error=float(np.abs(actual['expected']-native_depth).max()))
   if run.context.pbr_plane_projection is not None:
    p=run.context.pbr_plane_projection;ids=p.support
    record['supported_world_plane_max_distance']=float(((run.pc.get_xyz+d['d_xyz'])[ids]@p.plane+p.offset).abs().max())
    record['supported_world_axis_cosine_min']=float((quaternion_normal(run.pc._rotation+d['d_rotation'])[ids]@p.plane).abs().min())
    assert record['supported_world_plane_max_distance']<5e-7 and record['supported_world_axis_cosine_min']>1-1e-5
   camera.fid=torch.tensor([.375],device='cuda');overridden=run.deformation(camera)
   expected_override=run.context.deformation(camera)
   assert torch.equal(overridden['d_xyz'],expected_override['d_xyz']) and torch.equal(overridden['d_rotation'],expected_override['d_rotation'])
   record['mesh_target_time_override_consistent']=True;rows.append(record)
 run.context.verify_frozen()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in run.files.items())
 results.append(dict(model=str(model),iteration=iteration,frames=rows,source_sha_unchanged=True,hashed_sources=len(run.files)))
 del run;torch.cuda.empty_cache()
report=dict(status='PASS',kind='actual_GeometryRun_PBR_buffers_and_mesh_time_consistency',optimizer_steps=0,GT_labels_read=False,models=results)
(out/'inspection/geometry_buffers.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
