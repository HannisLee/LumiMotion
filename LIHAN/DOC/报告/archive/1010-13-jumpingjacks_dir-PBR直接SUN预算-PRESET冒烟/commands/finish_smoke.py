from pathlib import Path
import json,hashlib,shutil,datetime,cv2
r=Path(__file__).resolve().parent.parent;cwd=r.parents[2]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
config=json.loads((r/'model_mlp/stage2_config.json').read_text())
assert config['loss_preset']=='pbr_gt_directional_plane_native_direct' and config['pbr_rasterizer_backend']=='lumimotion'
assert config['wo_indirect'] and config['wo_indirect_relight'] and config['pbr_directional_actual_ray_budget']
manifest=json.loads((r/'code_manifest.json').read_text())
for f,h in manifest.items():assert hashlib.sha256((cwd/f).read_bytes()).hexdigest()==h
for category in ['rgb','alpha','normal','albedo']:assert len(list((r/('insights/insights/'+category)).glob('*.png')))==150
videos={}
for p in (r/'insights').rglob('*.mp4'):
 cap=cv2.VideoCapture(str(p));assert cap.isOpened();n=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  assert frame.shape[:2]==(400,400);n+=1
 cap.release();assert n==150;videos[str(p)]=n
assert len(videos)==4
native=json.loads((r/'model_mlp/pbr_rasterizer.json').read_text())
assert native['source_sha256']=='82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0'
review=dict(functional_status='PASS',visual_quality='FAILED_SMOKE_ONLY',reviewed_at=now,frames=[1,75,150],
 rgb='黑背景正常、人物和平台存在；0075背面非常暗，头手和衣服破碎斑驳、脚周平台伪起伏，20步不足最终质量。',
 alpha='主体/平台大部覆盖，腰部边缘和脚周零碎薄雾残影，饱和Alpha不等于正确几何。',
 albedo='来源照明/正背面色差与平台棋盘格烘入，脚周污染仍有；20步仅验证功能。',
 normal='平台整体向上，0075不再全平台反向；人体/脚周混合轴斑驳仍明显。',
 hdr_directional='三帧替换SUN与HDR16实际看，响应/背景正常但伪影及材质烘光仍有；无重光照GT。',
 images=['inspection/four_categories.png','inspection/relight.png'],video_full_decode=videos,
 optimizer_steps_extra_diagnostics=0,full_mae=False,normal_7_goal='NOT_EVALUATED',commit_allowed=False,
 actual_sessions_exit={'cpu':0,'train':0,'pbr_first_failed':1,'pbr_retry':0,'shared_buffers':0,'insights':0},code_files=len(manifest))
(r/'inspection/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
addition='\n## 实际冒烟结论（'+now+'）\n\n194 CPU/20步真实Trainer/三帧all-eligible SUN及L3/with-indirect真实光线预算、材质与XYZ/q梯度、GeometryRun与材质helper逐buffer一致均PASS。参数SHA不变、0额外诊断优化，源码471 SHA核验。首次诊断重复backward图失败、保存后修正仅诊断图重建，retry实际exit0。35020 checkpoint flags/native来源绑定通过。\n\n四类150×4和4视频逐帧解码PASS，已实际看首中尾RGB/Alpha/normal/albedo及SUN/HDR16。实际质量FAILED_SMOKE_ONLY：人物破碎/背面暗、albedo烘光、脚周伪几何仍有；不是完整MAE/7°，不commit/push。功能允许下一步独立完整20k训练，不沿用20步权重。\n\n命令：check_pbr_gradients.py（失败和retry日志）、check_shared_buffers.py依次调用check_geometry_buffers.py/check_material_buffers.py；render_stage2 --model_path '+str(r/'model_mlp')+' --load_iter35020 --task insights --output_path '+str(r/'insights')+'；日志logs/{pbr_gradients_retry,geometry_buffers,material_buffers,shared_buffers,insights}.log。代表图inspection/four_categories.png、relight.png；各检查JSON同inspection。\n'
with (r/'README.md').open('a') as f:f.write(addition)
doc=cwd/'LIHAN/DOC/修改/1010-10-原生PBR直接SUN像素预算与DirectPreset.md'
with doc.open('a') as f:f.write(addition)
goal=cwd/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## B07功能冒烟实际完成\n'+addition)
archive=cwd/'LIHAN/DOC/报告/archive/1010-13-jumpingjacks_dir-PBR直接SUN预算-PRESET冒烟';archive.mkdir(parents=True,exist_ok=False)
for folder in ['commands','logs','inspection']:shutil.copytree(r/folder,archive/folder)
for p in [r/'README.md',r/'train_argv.json',r/'code_manifest.json',r/'model_mlp/stage2_config.json',r/'model_mlp/pbr_rasterizer.json']:shutil.copy2(p,archive/p.name)
sha={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(sha,ensure_ascii=False,indent=2)+'\n')
for checkout in [cwd,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 if checkout!=cwd:
  for p in [doc,goal]:shutil.copy2(p,checkout/p.relative_to(cwd))
  shutil.copytree(archive,checkout/archive.relative_to(cwd))
 for k,h in sha.items():assert hashlib.sha256((checkout/archive.relative_to(cwd)/k).read_bytes()).hexdigest()==h
print(json.dumps(dict(functional_status='PASS',visual_quality='FAILED_SMOKE_ONLY',archive_files=len(sha),verified_checkouts=3),ensure_ascii=False),flush=True)
