"""真实会话退出、实际看图与视频解码后落档B10；不自动通过画质或7度。"""
from pathlib import Path
import json,hashlib,shutil,datetime,cv2
r=Path(__file__).resolve().parent.parent;c=r.parents[2]
assert not (r/'inspection/visual_review.json').exists()
review=dict(actual_images_viewed=True,functional_status='PASS',visual_quality='FAILED_SMOKE_ONLY',
 actual_sessions_exit={'trainer_26981':0,'saved_load_70944':0,'GPU_78840':0,'shared_10852':0,'insights_73487':0,'CPU_81654':0},
 actual_sessions_closed=True,viewed_images=['inspection/four_categories.png','inspection/relight.png'],
 rgb_background='RGB黑色背景正常，首尾人物和平台保留；中帧人物接近黑色但Alpha主体完整，不能解释为缺失。首尾人物/裤腿有明显黑斑，平台足周黑灰杂点，20步画质不通过。',
 alpha='首中尾人物和完整椭圆平台均有Alpha；手臂、身体及脚周有少量模糊边缘和孤立点。没有删去难点或平台。',
 albedo_static_dynamic='人物首尾和中帧均可见，但明暗及纹理斑块混入反照率，手臂/裤腿碎斑，平台颜色受阴影污染；不认定已经完成材质分解。',
 normal_depth='实际渲染GS法线可见人物和平台，首尾平台主要紫蓝，中帧人物黄绿且表面杂色，足周法线噪点明显。此拼图只检查GS法线，未额外生成Depth/Mesh，不称其已验收。',
 relighting='同首中尾的方向光与HDR16均通过共享入口有实际亮度/色彩响应；HDR有橙色轮廓与高光、平台蓝色响应。方向光中帧人物近黑，不能仅据此宣称代码异常，也不能据重光照有响应认定几何正确。',
 GT_normal_material_geometry_photometry_inputs=False,source_RGB_derived_constraint=True,full_MAE=False,normal_7_goal='NOT_EVALUATED')
assert 'Ran 226 tests' in (r/'logs/cpu_tests.log').read_text() and '\nOK\n' in (r/'logs/cpu_tests.log').read_text()
runtime=json.loads((r/'training_runtime.json').read_text())
assert runtime['exit_code']==0 and runtime['verified_session_closed']
assert not (Path('/proc')/str(runtime['pid'])).exists() and 'Training complete.' in (r/'logs/train.log').read_text()
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients']:
 assert json.loads((r/f'inspection/{name}.json').read_text())['status'].startswith('PASS')
manifest=json.loads((r/'code_manifest.json').read_text());assert len(manifest)==476
assert manifest==json.loads((r/'code_manifest_train_launch.json').read_text())
for k,v in manifest.items():
 for p in [c/k,r/'commands/code_snapshot'/k,r/'commands/code_snapshot_train_launch'/k]:
  assert hashlib.sha256(p.read_bytes()).hexdigest()==v
for key in ['rgb','alpha','normal','albedo']:assert len(list((r/'insights/insights'/key).glob('*.png')))==150
videos={}
for p in (r/'insights').rglob('*.mp4'):
 cap=cv2.VideoCapture(str(p));assert cap.isOpened();n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape[:2]==(400,400);n+=1
 cap.release();assert n==150;videos[str(p)]=n
assert len(videos)==4;review['videos_decoded']=videos
(r/'inspection/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
saved=json.loads((r/'inspection/gate_checkpoint.json').read_text())
grad=json.loads((r/'inspection/pbr_gradients.json').read_text())
addition='\n## B10实际冒烟完成（'+datetime.datetime.now().astimezone().isoformat()+'）\n\n'
addition+='226 CPU、真实20步Trainer及保存加载/GPU/共享buffer/insights均实际exit0且会话关闭；476核心和双snapshot SHA保持。35000→35020，全部99827点finite，opacity/scale及来源NN32tensor冻结，概率gate改变69981点、maxdelta1.85966491699e-5；raw变化70033/max74.534210205只是有限逆sigmoid编码。独立源RGB候选29621点，lambda2，canonical轴loss0.102662175894、其q梯度L2 0.00525612570345非零，三TRAIN photo q梯度L2 .022683673/.602086246/.024002507，概率 .015508413/.038564317/.013200786；albedo/roughness/XYZ/q/gate均finite且非零。不能从全局梯度L2推断完整训练质量。\n\n'
addition+='真实DirectSUN/HDR16梯度预算、三SUN/间接/legacy只读0 optimizer检查PASS；共享Geometry alpha/GSnormal/ExpectedZ误差0，material albedo0/roughness≤5.96e−8，plane真实世界位置约2.7e−8m/轴cos≥.99999964。source-local运输应用在真实Gaussian q，所有入口同顺序先运输再plane。4类150×4及4视频各150帧全部解码；Depth/Mesh及全15七MAE尚未在smoke生成或评分。\n\n'
for key in ['rgb_background','alpha','albedo_static_dynamic','normal_depth','relighting']:addition+='- '+review[key]+'\n'
addition+='\n最终功能PASS，画质FAILED_SMOKE_ONLY，原GS Normal<7度NOT_EVALUATED。GT方向白色单位；新轴来自原135TRAIN RGB/自身Alpha和原35k预测，不是GT normal；GT材质/法线/粗糙度/深度/mesh/blend/灯能量颜色全部未参与训练/初始化/约束。全部点/帧/原photo/评分保留。正式B10需独立原35k→55k完整20k，恢复warmup2000/ramp1000；不用本smoke或B09学得参数继续训练。不commit/push/新目标分支。\n'
body=(r/'README.md').read_text()+addition;(r/'README.md').write_text(body)
doc=c/'LIHAN/DOC/修改/1010-14-原生PBR训练RGB来源局部法线与Preset.md'
with doc.open('a') as f:f.write(addition)
train=c/'LIHAN/DOC/训练/1010-18-dir-PBR训练RGB来源局部法线冒烟训练文档.md';train.write_text(body)
goal=c/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## B10真实Preset功能冒烟与实际人工目检\n'+addition)
index=c/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [B10来源RGB局部法线Preset实际冒烟](DOC/训练/'+train.name+')：226 CPU/20步/GPU/保存/共享buffer/4视频功能PASS，实际画质FAILED_SMOKE_ONLY，完整训练待执行。\n')
archive=c/'LIHAN/DOC/报告/archive/1010-18-dir-PBR训练RGB来源局部法线冒烟';archive.mkdir(exist_ok=False)
for folder in ['commands','logs','inspection']:shutil.copytree(r/folder,archive/folder)
for p in [r/'README.md',r/'train_argv.json',r/'training_runtime.json',r/'code_manifest.json',r/'code_manifest_train_launch.json']:
 shutil.copy2(p,archive/p.name)
for name in ['stage2_config.json','pbr_feature_gate.json','pbr_geometry_source.json','pbr_rgb_normal.npz','pbr_rgb_normal.json','pbr_rgb_normal_input.npz','pbr_rgb_normal_input.json']:
 shutil.copy2(r/'model_mlp'/name,archive/name)
m={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for checkout in [c,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 if checkout!=c:
  for p in [doc,train,goal,index]:shutil.copy2(p,checkout/p.relative_to(c))
  shutil.copytree(archive,checkout/archive.relative_to(c))
 for k,v in m.items():assert hashlib.sha256((checkout/archive.relative_to(c)/k).read_bytes()).hexdigest()==v
print(json.dumps(dict(functional='PASS',visual='FAILED_SMOKE_ONLY',archive_files=len(m),SHA_verified_checkouts=3),ensure_ascii=False),flush=True)
