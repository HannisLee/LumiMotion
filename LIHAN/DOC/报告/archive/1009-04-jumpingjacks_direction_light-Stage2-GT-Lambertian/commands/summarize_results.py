"""汇总完整15帧MAE，保存首中尾法线总览和真值审核；质量目检需单独回填。"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=json.loads((ROOT/'normal_depth/metrics.json').read_text());fg=g['summary']['foreground']
a=json.loads((ROOT/'albedo/iteration_55000/metrics.json').read_text());raw=a['summaries']['raw']['foreground']
metrics=[dict(metric='Albedo MAE',mae=raw['mae_linear'],unit='linear RGB',coverage_pct=raw['coverage_pct'])]
for name in ['gs_render','depth_normal','mesh']:
 m=fg['normal_metrics'][name];metrics.append(dict(metric=m['label'],mae=m['mae_pixel_mean_deg'],unit='degree',coverage_pct=100*m['coverage']))
for name in ['expected','median']:
 m=fg['depth'][name];metrics.append(dict(metric=name.capitalize()+' Z Depth MAE',mae=m['mae_pixel_mean'],unit='m',coverage_pct=100*m['coverage']))
metrics.append(dict(metric='GS Normal XYZ component MAE (supplementary)',mae=fg['normal']['component_l1']['pixel_mean'],unit='unit normal XYZ',coverage_pct=100*fg['normal_coverage']))
metrics.append(dict(metric='Roughness MAE',mae=None,unit='r',coverage_pct=None,status='NOT_APPLICABLE',reason='Lambertian has no roughness output'))
result=dict(status='CALCULATED_REVIEW_PENDING',model_path=str(ROOT/'model_mlp'),iteration=55000,split='test',frames=list(range(10,151,10)),resolution=[400,400],metrics=metrics,normal_common_pixels=fg['common_normal_metrics'],gs_normal_missing_180_penalty_deg=fg['normal']['angular_with_missing_deg']['pixel_mean'],albedo_raw=raw,rgb=json.loads((ROOT/'model_mlp/renders_stage2/ours_55000/eval/metrics.json').read_text())['metrics'],intensity=json.loads((ROOT/'model_mlp/calibration.json').read_text())['intensity'],source_model_unchanged=True,independent_audit=json.loads((ROOT/'inspection/independent_audit.json').read_text()))
(ROOT/'metrics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with open(ROOT/'metrics_summary.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=['metric','mae','unit','coverage_pct','status','reason']);w.writeheader();w.writerows(metrics)
d=json.loads((ROOT/'gt_depth/manifest.json').read_text());mesh=json.loads((ROOT/'gt_meshes/manifest.json').read_text());assert len(d['frames'])==len(mesh['frames'])==15
assert d['meters_per_world_unit']==1 and d['unit_system']=='METRIC'
assert sha(mesh['source_blend'])==mesh['source_blend_sha256']
assert sha(d['mesh_manifest'])==d['mesh_manifest_sha256']
for num,entry in d['frames'].items():
 assert sha(ROOT/'gt_depth'/entry['file'])==entry['sha256']
 assert entry['ply']['exact_mesh_match']
 assert sha(entry['ply']['file'])==entry['ply']['sha256']
 assert mesh['frames'][num]['camera_max_error']<=1e-5
report=dict(status='PASS',frames=15,meters_per_world_unit=1,gt_depth='pixel center ray intersections of per-frame dir Blender evaluated PLY triangles; positive camera Z',normal_reference='Blender signed world-space shading Normal EXR may include material normal maps',source_blend=mesh['source_blend'],source_blend_sha256=mesh['source_blend_sha256'],alpha_hit_iou_min=min(e['audit']['alpha_hit_iou'] for e in d['frames'].values()),geometry_shading_normal_gap_deg=[e['audit']['exr_mesh_normal_deg']['mean'] for e in d['frames'].values()],same_gt_reference_for_all_three_normals=True)
(ROOT/'inspection/gt_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
canvas=Image.new('RGB',(400*4,424*3),'white');draw=ImageDraw.Draw(canvas)
for row,n in enumerate([10,80,150]):
 with np.load(ROOT/'normal_depth'/f'prediction_{n:04d}.npz') as arr:
  for col,(label,key) in enumerate([('GT shading Normal','gt_normal'),('GS Render Normal','normal'),('Depth Normal','depth_normal'),('Mesh Normal','mesh_normal')]):
   x=arr[key].astype(float);norm=np.linalg.norm(x,axis=-1);valid=(norm>1e-6)&np.isfinite(x).all(-1)&arr['gt_foreground'].astype(bool)
   if key=='mesh_normal':valid&=arr['mesh_valid']
   x=x/np.maximum(norm[...,None],1e-6);rgb=np.where(valid[...,None],np.clip(x*.5+.5,0,1),0)
   image=Image.fromarray((rgb*255).round().astype(np.uint8))
   canvas.paste(image,(400*col,424*row+24));draw.text((400*col+5,424*row+6),f'{n:04d} {label}',fill='black')
canvas.save(ROOT/'inspection/three_normals_first_middle_last.png')
print(json.dumps(result,ensure_ascii=False,indent=2))
