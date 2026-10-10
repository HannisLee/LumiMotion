from pathlib import Path
import json,hashlib,numpy as np,shutil,sys
root=Path.cwd();sys.path.insert(0,str(root));run=root/'output/1010-10-jumpingjacks_dir-PBR来源平面几何投影';model=run/'model_mlp'
a=json.loads((root/'output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp/stage2_config.json').read_text());b=json.loads((model/'stage2_config.json').read_text())
changed={k:{'B04':a.get(k),'B05':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
assert set(changed)=={'loss_preset','pbr_plane_projection','pbr_plane_motion_span','pbr_plane_support_band','source_path','model_path','pbr_gt_lights_path'},changed
assert 'pbr_plane_motion_span' not in a and 'pbr_plane_support_band' not in a
assert (model/'pbr_gt_lights.json').read_bytes()==(root/'output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/model_mlp/pbr_gt_lights.json').read_bytes()
from scene.pbr_directional import validate_directional_training_inputs
provenance=json.loads((model/'rgb_alpha_provenance.json').read_text());assert provenance==validate_directional_training_inputs(b['source_path'])
plane=json.loads((model/'pbr_plane_projection.json').read_text());anchor=json.loads((model/'pbr_deformation_anchor.json').read_text())
assert plane['source_files']==anchor['source_files'] and plane['support_points']==15187 and plane['training_cameras']==135
assert hashlib.sha256((model/'pbr_plane_projection.npz').read_bytes()).hexdigest()==plane['asset_sha256']
assert hashlib.sha256((model/'pbr_deformation_anchor.npz').read_bytes()).hexdigest()==anchor['asset_sha256']
with np.load(model/'pbr_plane_projection.npz') as d:
 ids=d['support'];assert len(ids)==99827 and int(ids.sum())==15187
 stats={k:{'min':float(d[k][ids].min()),'max':float(d[k][ids].max()),'median':float(np.median(d[k][ids]))} for k in ['motion_span','max_plane_distance','min_axis_cosine']}
assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==h for k,h in json.loads((run/'code_manifest.json').read_text()).items())
record=dict(status='PASS_LAUNCH_ONLY',changed_config=changed,source_provenance=provenance,plane=plane,plane_supported_stats=stats,anchor=anchor,lighting_table_identical_to_B04=True,code_manifest_matches=True,final_mae=False)
(run/'inspection/launch_configuration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
runtime=json.loads((run/'training_runtime.json').read_text())
text='\n## 正式启动记录\n\n'+json.dumps(runtime,ensure_ascii=False)+'\n\n真实train session42059/PID245234/start ticks8927314；后处理session86603/PID245271绑定真实train；40k预览session2961同PID绑定，禁止退出后盲目重启。实际配置相比B04改变Plane Preset/projection、新增阈值.2/.01字段及本轮路径；其它既有参数相同，dir表字节相同。首次launch audit只列5个允许字段，漏列新阈值2项，断言失败；未修改训练，修正审计字段后PASS。150原RGBA来源、99827点全保留、source资产/cameraSHA/15187支持/代码manifest检查PASS，证据inspection/launch_configuration.json。最终MAE/质量待完成，未commit/push。\n'
for p in [run/'README.md',run/'README_training_progress.md',root/'LIHAN/DOC/训练/1010-10-jumpingjacks_dir-PBR来源平面几何投影训练报告.md',root/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md']:p.write_text(p.read_text()+text)
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 for rel in ['LIHAN/DOC/训练/1010-10-jumpingjacks_dir-PBR来源平面几何投影训练报告.md','LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md']:
  shutil.copy2(root/rel,checkout/rel)
print(json.dumps({'status':'PASS_LAUNCH_ONLY','changed_config_fields':list(changed),'support':15187,'support_statistics':stats,'runtime':runtime},ensure_ascii=False))
