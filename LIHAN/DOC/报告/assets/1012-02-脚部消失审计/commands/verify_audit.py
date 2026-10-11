"""交叉检查区域复算、来源哈希、渲染命令及图像解码。"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parents[1]
from utils.geometry_eval_utils import camera_rays,evaluate_region
summary=json.loads((OUT/'regional_summary.json').read_text())
checks=[]
for prefix in ('1011-01','1011-27','1011-28'):
    folder=next((ROOT/'output').glob(prefix+'*/evaluation_*/normal_depth'))
    metadata=json.loads((folder/'metrics.json').read_text())
    cams=json.loads((Path(metadata['command']['source_path'])/'transforms_test.json').read_text())
    expected={r['region']:r for r in summary if r['experiment']==folder.parent.parent.name and r['iteration']==metadata['iteration']}
    totals={'sum':0.,'count':0}
    for record in cams['frames']:
        a=dict(np.load(folder/f"prediction_{record['source_frame']:04d}.npz"))
        rays=camera_rays(record['transform_matrix'],[record['fl_x']/2,record['fl_y']/2,200,200],400,400)
        z=rays[...,2]+rays[...,5]*a['gt_depth'];feet=a['gt_foreground']&(z>.05)&(z<.35)
        result=evaluate_region(a,a['gt_normal'],np.linalg.norm(a['gt_normal'],axis=-1)>1e-6,a['gt_depth'],feet)
        v=result['normal_metrics']['gs_render']['angular_deg'];totals['sum']+=v['sum'];totals['count']+=v['count']
    assert totals['count']==expected['feet']['gs_render_count']
    assert abs(totals['sum']/totals['count']-expected['feet']['gs_render_mae'])<1e-10
    for key,alias in [('gs_render','gs_render'),('depth_normal','depth_normal'),('mesh','mesh')]:
        official=metadata['summary']['foreground']['normal_metrics'][key]['mae_pixel_mean_deg']
        assert abs(expected['foreground'][alias+'_mae']-official)<1e-9
    checks.append(dict(model=prefix,regional_original_function='PASS',foreground_original_metrics='PASS'))
for file,h in json.loads((OUT/'array_sources_sha256.json').read_text()).items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h,file
executed=json.loads((OUT/'commands/checkpoint_render_executed.json').read_text())
assert len(executed)==8 and all(p['exit_code']==0 for p in executed)
assert all(len(list(Path(p['output']).glob('prediction_*.npz')))==15 for p in executed)
for file in (OUT/'figures').glob('*.png'):
    with Image.open(file) as image:image.verify()
assert len(summary)==41*9
assert all(v['frames']==15 for v in summary)
gradient=json.loads((OUT/'opacity_gradient_summary.json').read_text())
assert len(gradient)==4 and all(v['frames']==135 and v['parameter_updates']==0 for v in gradient)
result=dict(status='PASS',checks=checks,experiment_count=18,checkpoint_count=41,test_frames_per_checkpoint=15,
            new_checkpoint_renders=8,new_render_frames=120,gradient_states=4,gradient_frames_per_state=135,
            source_hash_check='PASS',image_decode='PASS',parameter_optimizer_updates=0,
            note='PASS仅表示审计数值与来源核验；T13/T14脚部完整性结论为FAILED。')
(OUT/'audit_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
