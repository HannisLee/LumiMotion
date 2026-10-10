from pathlib import Path
import json,hashlib,shutil,datetime,cv2,shlex
r=Path(__file__).resolve().parent.parent;cwd=r.parents[2]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
review=json.loads((r/'inspection/visual_review.json').read_text())
assert review['actual_images_viewed'] and review['functional_status']=='PASS'
assert all(review['actual_sessions_exit'][k]==0 for k in ['cpu_retry','train','pbr','gate_checkpoint','shared_buffers','insights'])
assert 'Ran 202 tests' in (r/'logs/cpu_tests_retry.log').read_text() and '\nOK\n' in (r/'logs/cpu_tests_retry.log').read_text()
assert 'Training complete.' in (r/'logs/train.log').read_text()
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients']:
 assert json.loads((r/f'inspection/{name}.json').read_text())['status'].startswith('PASS')
manifest=json.loads((r/'code_manifest.json').read_text())
for f,h in manifest.items():
 assert hashlib.sha256((cwd/f).read_bytes()).hexdigest()==h
 assert hashlib.sha256((r/'commands/code_snapshot'/f).read_bytes()).hexdigest()==h
for c in ['rgb','alpha','normal','albedo']:assert len(list((r/'insights/insights'/c).glob('*.png')))==150
videos={}
for p in (r/'insights').rglob('*.mp4'):
 cap=cv2.VideoCapture(str(p));assert cap.isOpened();n=0
 while True:
  ok,img=cap.read()
  if not ok:break
  assert img.shape[:2]==(400,400);n+=1
 cap.release();assert n==150;videos[str(p)]=n
assert len(videos)==4
review['videos_decoded']=videos
(r/'inspection/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
addition='\n## 实际冒烟验收（'+now+'）\n\n202 CPU discovery实际exit0；真实20步Trainer实际exit0并关闭，35k→35020保存全部99827点。新增gate参数组真实训练68101点raw feature发生变化，max delta1.85966492e−5，低步ramp仅验证更新；opacity/scale不变，source32 MLP tensor完全原值，初始feature原值，保存source metadata/lighting marker并共享加载核验。无额外诊断优化、无GT材质/法线/几何/能量/颜色训练输入。\n\n真实native三TRAIN0001/75/149 SUN18843/15904/19137条，all-eligible alpha>.9；albedo/roughness/XYZ/q/feature梯度非零finite。gate photo L2 .00085692690/.00269499072/.00030556315。3SUN/with-indirect/历史预算variant同checkpoint零优化全部finite，参数SHA保持，不宣称不同variant已独立训练。GeometryRun alpha/worldnormal/Expected Z全buffer一致、目标timeoverride一致；albedo/roughness helper一致。\n\n150×四类及4视频全解码PASS，实际查看0001/0075/0150 RGB/Alpha/Normal/Albedo及三HDR16/SUN。功能PASS、实际质量FAILED_SMOKE_ONLY：人物破碎/背面暗、材质烘光、平台脚周伪起伏仍明显；不作为正式全15七MAE/7°，未commit/push/新分支。正式候选须从35k独立完整20k。\n\n首次CPU201项有2个旧GeometryRun mock缺少新增FrozenScene字段的错误，失败日志和原source snapshot保留；真实20步Trainer生产源码未改，之后仅修mock/增加metadata用例，202 retry全PASS。code_manifest_train_launch与现manifest唯一差异是tests/test_geometry_eval_source_runtime.py，生产核心完全相同。\n\n完整训练命令：\n```bash\n'+shlex.join(json.loads((r/'train_argv.json').read_text()))+'\n```\n\n诊断命令commands/{check_gate_checkpoint,check_pbr_gradients,check_shared_buffers,check_geometry_buffers,check_material_buffers,make_comparison,finish_smoke}.py；对应日志logs/{gate_checkpoint,pbr_gradients,shared_buffers,geometry_buffers,material_buffers}.log；渲染命令python -m scripts.render_stage2 --model_path '+str(r/'model_mlp')+' --load_iter35020 --task insights --output_path '+str(r/'insights')+'，日志logs/insights.log。图inspection/four_categories.png、relight.png；JSON和视频完整路径见inspection/visual_review.json。\n'
with (r/'README.md').open('a') as f:f.write(addition)
doc=cwd/'LIHAN/DOC/修改/1010-11-原生PBR来源输出gate微调与Preset.md'
body='''# 1010-11 原生PBR来源输出gate微调与Preset

## 修改依据和范围

已先读GOAL，B07完整GS16.81871120°/覆盖95.00328573%，质量与7°失败，保留全部失败产物。新增pbr_gt_directional_plane_native_direct_gate，只开放PLY单维feature的原生输出乘法gate，源35k MLP全部冻结/sourceXYZ输入锚定不变，全部99827点保留。B06真实只读gate梯度存在、B07source joint贡献图有动态near-plane平台贡献；该source组不是GT语义，不做强制静态/删点/评分域筛选。

Stage2OptimizationParams默认flagFalse，新增gate_lr.001、max_source_logit_delta4、anchor.001。Trainer只对gate候选保留feature optimizer组并冻结MLP全部组，允许Deform输出乘法对feature可导；source输入仍detach，MLP hidden不可导。warmup2000/ramp1000/decay同geometry，只在几何启用后加source logit归一化平方anchor；每4步Adam后将raw feature投影到source±4。初始feature来自真实原35k且逐bit校验，不重置/二值硬化/筛点。原生sigmoid(2*feature)、NN结构、GGX/BVH/SH辅助loss、Direct预算/材质/原geometry/loss与评分均保持。固定source NN32tensor不得变化。

scene/pbr_feature_gate.py保存本地source PLY/MLP SHA、metadata配置与无GT协议；checkpoint lighting marker绑定metadata SHA，加载核验source权重所有tensor、gate shape/finite/trust及配置。FrozenScene统一校验，GeometryRun/材质helper记录同metadata SHA，渲染/评测直接用保存PLY feature，禁止退回source gate。其它Preset旧no_grad/optimizer范围保持，Stage1无改动。原生Adam未完整保存，不声明无损resume。

训练只用原RGB/自身RGBA Alpha、相机time、原35k预测和GT方向白色单位灯；不用GT albedo/normal/roughness/depth/mesh/blend/光强/颜色初始化/loss/prior。GT仍仅离线评分。

修改文件：scene/pbr_feature_gate.py、arguments/stage2.py、scripts/{train_stage2,loss_stage2,stage2_runtime,eval_geometry_gt,eval_albedo_gt,eval_roughness_gt}.py；tests/test_pbr_feature_gate.py和共享Runtime测试。Trainer保留CRLF，diff --check用cr-at-eol设置通过。

'''+addition
doc.write_text(body)
goal=cwd/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## B08来源输出gate实现与功能冒烟\n'+addition)
index=cwd/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [原生PBR来源输出gate与Preset](DOC/修改/'+doc.name+')：202 CPU/真实20步/五参数梯度/共享buffer/4视频PASS，实际smoke质量FAILED，完整B08未开训。\n')
archive=cwd/'LIHAN/DOC/报告/archive/1010-14-jumpingjacks_dir-PBR来源输出gate-PRESET冒烟'
archive.mkdir(parents=True,exist_ok=False)
for folder in ['commands','logs','inspection']:shutil.copytree(r/folder,archive/folder)
for p in [r/'README.md',r/'train_argv.json',r/'code_manifest.json',r/'code_manifest_train_launch.json',r/'model_mlp/stage2_config.json',r/'model_mlp/pbr_feature_gate.json',r/'model_mlp/pbr_geometry_source.json']:
 shutil.copy2(p,archive/p.name)
sha={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(sha,ensure_ascii=False,indent=2)+'\n')
for checkout in [cwd,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 if checkout!=cwd:
  for p in [doc,goal,index]:shutil.copy2(p,checkout/p.relative_to(cwd))
  shutil.copytree(archive,checkout/archive.relative_to(cwd))
 for k,h in sha.items():assert hashlib.sha256((checkout/archive.relative_to(cwd)/k).read_bytes()).hexdigest()==h
print(json.dumps(dict(functional_status='PASS',visual_quality='FAILED_SMOKE_ONLY',archive_files=len(sha),verified_checkouts=3),ensure_ascii=False),flush=True)
