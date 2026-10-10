from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parents[2]
sys.path.insert(0,str(CWD))
from scripts.eval_albedo_gt import AlbedoRun
from scripts.eval_roughness_gt import RoughnessRun
albedo=AlbedoRun(ROOT/'model_mlp',35020);rough=RoughnessRun(ROOT/'model_mlp',35020,None)
rows=[]
for f in ['0001','0075','0150']:
 camera=albedo.context.camera_by_key[f];a=albedo.verify_renderer(camera)
 camera=rough.context.camera_by_key[f];_,r=rough.render(camera,verify=True)
 rows.append(dict(frame=f,albedo=a,roughness=r))
albedo.verify_unchanged();rough.context.verify_frozen()
result=dict(status='PASS_NATIVE_MATERIAL_BUFFERS',iteration=35020,rows=rows,gt_material_inputs=False,optimizer_steps=0)
(ROOT/'inspection/material_buffers.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
