"""真实exit0/实际看图后落档功能冒烟，不自动给画质/7°PASS。"""
from pathlib import Path
import json,hashlib,shutil,datetime,cv2
r=Path(__file__).resolve().parent.parent;c=r.parents[2]
review=json.loads((r/'inspection/visual_review.json').read_text())
assert review['actual_images_viewed'] and all(x==0 for x in review['actual_sessions_exit'].values())
assert 'Ran 211 tests' in (r/'logs/cpu_tests.log').read_text() and '\nOK\n' in (r/'logs/cpu_tests.log').read_text()
runtime=json.loads((r/'training_runtime.json').read_text())
assert runtime['exit_code']==0 and runtime['verified_session_closed']
assert not (Path('/proc')/str(runtime['pid'])).exists() and 'Training complete.' in (r/'logs/train.log').read_text()
assert review['visual_quality']=='FAILED_SMOKE_ONLY'
for name in ['gate_checkpoint','geometry_buffers','material_buffers','pbr_gradients']:
    assert json.loads((r/f'inspection/{name}.json').read_text())['status'].startswith('PASS')
manifest=json.loads((r/'code_manifest.json').read_text());assert len(manifest)==474
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
assert len(videos)==4
review['videos_decoded']=videos
(r/'inspection/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
saved=json.loads((r/'inspection/gate_checkpoint.json').read_text())
grad=json.loads((r/'inspection/pbr_gradients.json').read_text())
addition='\n## 实际冒烟完成（'+datetime.datetime.now().astimezone().isoformat()+'）\n\n'
addition+='211 CPU、20步Trainer/GPU五参数梯度/保存/共享geometry-material/insights六会话均实际exit0且关闭；474核心和双snapshot逐SHA一致，未在训练后改生产源码。35000→35020保留99827点finite，opacity/scale和32来源NN tensor原值。raw feature变化{}点、实际native概率变化{}点，max概率delta {:.10g}；raw最大偏移{:.8f}是逆sigmoid/有限端点编码，不等于世界几何的大改变，不沿用raw±4的旧信赖域。\n\n'.format(saved['raw_feature_changed_points'],saved['native_probability_changed_points'],saved['max_source_probability_delta'],saved['max_source_raw_logit_delta'])
addition+='TRAIN0001/0075/0149 photo概率梯度L2 '+','.join('{:.9f}'.format(row['gradients']['probability_gate']) for row in grad['rows'])+'，albedo/roughness/XYZ/q/概率全部finite非零；前向逐bit仍实际PLY native sigmoid，源码NN全冻结。单SUN实际18843/15904/19137光线，3SUN/间接/旧预算诊断都通过且0 optimizer，不称已独立训练。GeometryRun alpha/worldnormal/ExpectedZ最大误差0与目标time一致；albedo helper误差0，roughness≤5.96e−8。\n\n'
addition+='概率全域[0,1]允许源gate修正，未强制任何source组静态、不删点/帧/难像素；canonicalXYZ .08m/q60°仍只限制canonical参数，gate改变原NN输出的world形变另需完整RGB/Alpha/覆盖与MAE验收。输入没有GT材质/法线/几何/灯强度颜色，GT方向白色单位，RGB拟合强度非GT。原生完整Adam未保存，不声明无损resume。\n\n'
addition+='四类150×4、4视频全部解码150帧，实际看0001/0075/0150四类与HDR16/SUN代表图：\n\n'
for key in ['rgb_background','alpha','albedo_static_dynamic','normal_depth','relighting']:addition+='- '+review[key]+'\n'
addition+='\n最终功能PASS、画质FAILED_SMOKE_ONLY、正式全15七MAE及7°未评；下一完整B09只从原35k独立训练，warmup恢复2000，所有评测口径/覆盖/common/missing180原值，不以20步smoke或B08训练后checkpoint继续。未commit/push/新目标分支。\n'
body=(r/'README.md').read_text()+addition;(r/'README.md').write_text(body)
doc=c/'LIHAN/DOC/修改/1010-12-原生PBR概率gate优化与Preset.md';doc.write_text(body)
train=c/'LIHAN/DOC/训练/1010-16-dir-PBR概率gate冒烟训练文档.md';train.write_text(body)
goal=c/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## B09概率gate实现与真实功能冒烟\n'+addition)
index=c/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [原生PBR概率gate与Preset](DOC/修改/'+doc.name+')：211 CPU/真实20步/五参数GPU/共享buffer/4视频PASS，smoke质量FAILED，完整B09尚未启动。\n')
archive=c/'LIHAN/DOC/报告/archive/1010-15-jumpingjacks_dir-PBR概率gate-PRESET冒烟';archive.mkdir(exist_ok=False)
for folder in ['commands','logs','inspection']:shutil.copytree(r/folder,archive/folder)
for p in [r/'README.md',r/'train_argv.json',r/'training_runtime.json',r/'code_manifest.json',r/'code_manifest_train_launch.json',r/'model_mlp/stage2_config.json',r/'model_mlp/pbr_feature_gate.json',r/'model_mlp/pbr_geometry_source.json']:
    shutil.copy2(p,archive/p.name)
m={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for checkout in [c,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
    if checkout!=c:
        for p in [doc,train,goal,index]:shutil.copy2(p,checkout/p.relative_to(c))
        shutil.copytree(archive,checkout/archive.relative_to(c))
    for k,v in m.items():assert hashlib.sha256((checkout/archive.relative_to(c)/k).read_bytes()).hexdigest()==v
print(json.dumps(dict(functional='PASS',visual='FAILED_SMOKE_ONLY',archive_files=len(m),SHA_verified_checkouts=3),ensure_ascii=False),flush=True)
