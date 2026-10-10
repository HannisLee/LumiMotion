"""B06完整七MAE后，记录实际全类目检FAILED，保留全量分数与失败产物。"""
from pathlib import Path
import json,hashlib,shutil,datetime
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
result=json.loads((ROOT/'metrics_summary.json').read_text());state=json.loads((ROOT/'postprocess_state.json').read_text())
assert result['status']=='COMPLETE_MAE_MANUAL_REVIEW_PENDING' and len(result['metrics'])==7 and result['independent_audit']['status']=='PASS'
assert state['status']=='PIPELINE_COMPLETED_REVIEW_PENDING' and not (Path('/proc')/str(state['pid'])).exists()
steps=[json.loads(x) for x in (ROOT/'runtime_history.jsonl').read_text().splitlines()];assert len(steps)==9 and all(x['exit_code']==0 for x in steps)
runtime=json.loads((ROOT/'training_runtime.json').read_text());assert runtime['training_exit_code']==0 and runtime['verified_training_session_closed'] and runtime['steps_completed']==20000
manifest=json.loads((ROOT/'code_manifest.json').read_text())
for rel,digest in manifest.items():
 assert hashlib.sha256((CWD/rel).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/'commands/code_snapshot'/rel).read_bytes()).hexdigest()==digest
native=json.loads((ROOT/'model_mlp/pbr_rasterizer.json').read_text());assert native['source_sha256']=='82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0'
review=dict(status='FAILED',reviewed_at=now,iteration=55000,
 rgb_background='黑背景正常，人物/平台存在；头手/衣服仍斑驳稀疏，脚周平台伪凸起/灰黑污染，B05/B06整体接近。RGB PSNR略改善但不足验收；GT0080本身背光暗，不只凭暗失败。',
 alpha='主体和平台基本覆盖，腰手边缘/脚周细碎薄雾残影；数值覆盖率全部报告，不从alpha饱和推断正确几何。',
 albedo_static_dynamic='正背面照明烘入材质、平台棋盘格和脚周污染；静动态feature冻结，来源误分未修复。Albedo与Roughness MAE较B05上升。',
 normal_depth='GS平台整体向上，0075不再旧installed整个plane反向，实际原生核修复有效；身体/脚周混合轴斑驳及伪凸起仍有。Expected/Median Z脚周明显凸起，Depth Normal高频噪声，Mesh0080下腿等缺面、0010/0150身体和平台仍不平整。',
 relighting='HDR0001/0076/0150与转灯0000/0036/0071已实际看：黑背景/主体/光照响应正常但人物和脚周伪影仍明显；无重光照GT，不声明精度。',
 reviewed_test_frames=[10,80,150],reviewed_insight_frames=[1,75,150],reviewed_hdr_frames=[1,76,150],reviewed_directional_frames=[0,36,71],
 images=['inspection/rgb_final_55000_comparison.png','inspection/four_categories_final_55000.png','inspection/relight_final_55000.png','normal_depth/comparison_0010.png','normal_depth/comparison_0080.png','normal_depth/comparison_0150.png'],
 normal_goal='FAILED',commit_allowed=False,kernel_fix_functional=True,final_training_is_native=True)
(ROOT/'inspection/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
result.update(status='FAILED_QUALITY',manual_review=review,normal_goal='FAILED',commit_allowed=False,pipeline_exit_code=0,verified_postprocess_session_closed=True,source_code_manifest_verified=True,kernel_runtime_provenance=str(ROOT/'model_mlp/pbr_rasterizer.json'))
(ROOT/'metrics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
artifact=json.loads((ROOT/'inspection/artifact_checks.json').read_text());artifact.update(manual_review='FAILED',visual_review=review);artifact['scope']=artifact['scope'].replace('人工审阅待完成','人工实际审阅FAILED；自动文件/数值PASS不等于质量PASS');(ROOT/'inspection/artifact_checks.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
state.update(status='FINISHED_QUALITY_FAILED',reviewed_at=now,pipeline_exit_code=0,verified_session_closed=True,final_gs_normal_mae_deg=result['metrics'][2]['mae'],commit_allowed=False);(ROOT/'postprocess_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
previous=json.loads((CWD/'output/1010-10-jumpingjacks_dir-PBR来源平面几何投影/metrics_summary.json').read_text())
table='| 指标 | B05同55k | B06原生CUDA | B06−B05 |\n| --- | ---: | ---: | ---: |\n'
for a,b in zip(previous['metrics'],result['metrics']):
 assert a['metric']==b['metric'];table+='| {} | {:.8f} | {:.8f} | {:+.8f} |\n'.format(b['metric'],a['mae'],b['mae'],b['mae']-a['mae'])
gain=previous['metrics'][2]['mae']-result['metrics'][2]['mae'];common=result['normal_common_pixels']
final='\n## B06最终完整验收与B05同55k比较（'+now+'）\n\n'+table
final+='\nGS法线改善{:.8f}°、覆盖率{:.8f}%→{:.8f}%；Depth/Mesh/Z略改善，Albedo和Roughness更差。RGB PSNR{:.8f}→{:.8f}dB，LPIPS{:.8f}→{:.8f}，略改善但人物/脚周伪影明显，最终质量FAILED、7°FAILED。全部99827点与15测试帧保留，评分/GT前景域不变。\n\n'.format(gain,previous['metrics'][2]['coverage_pct'],result['metrics'][2]['coverage_pct'],previous['rgb']['psnr'],result['rgb']['psnr'],previous['rgb']['lpips'],result['rgb']['lpips'])
final+='共同{}/{}（{:.8f}%），GS/Depth/Mesh共同MAE{:.8f}°/{:.8f}°/{:.8f}°；missing180={:.8f}°。\n\n'.format(common['valid_pixels'],common['gt_pixels'],common['coverage']*100,*[common['angular_deg'][k]['mae_pixel_mean_deg'] for k in ['gs_render','depth_normal','mesh']],result['gs_normal_missing_180_penalty_deg'])
final+='完整训练exec30746实际exit0，20k/55k于07:44:25完成，loop44:02；后处理exec47576实际exit0并关闭、9步骤全0，于07:58:11结束。187 CPU/真实20步GPU/共享逐buffer入口已验证，533来源SHA、15time×135TRAIN Mesh、481 roughness来源与独立15帧复算、6视频全解码PASS；这些不等于质量PASS。469核心/GLM及launch快照SHA再次一致，source32NN/冻结opacity-scale-feature保持。Native源SHA82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0绑定、原生实际编译.so SHA记录在pbr_rasterizer.json，不回退installed。B05旧kernel的实际结果保持，不覆盖为新kernel评分。\n'
body=(ROOT/'README.md').read_text();body=body.replace('状态：完整20000步训练、渲染、全部15测试帧MAE与独立复算完成；人工质量审閱待完成。不能据此标记图像质量或7°目标PASS。','状态：完整20000步训练、全量渲染、全部15测试帧七项MAE与独立复算完成；实际前后景/四类/HDR/Normal Depth Mesh质量FAILED、7°FAILED，保留完整失败产物，未commit/push。')
start=body.index('## 四类目检（待人工回填）');end=body.index('## 审计及结论',start)
section='## 四类实际目检：FAILED\n\n'+''.join('- '+review[k]+'\n' for k in ['rgb_background','alpha','albedo_static_dynamic','normal_depth'])
section+='\nHDR/转灯：'+review['relighting']+'\n\n已实际view上列代表图，6视频逐帧完整解码PASS；代表帧实际目检不宣称人工逐帧观看全部视频。GT只用于独立离线评测，没有生成任何训练GT geometry/normal/material prior。\n\n'
body=body[:start]+section+body[end:];body=body.replace('人工图像质量：PENDING','人工图像质量：FAILED');body=body.replace('先完成前后景人工审核，必要时修复重训，再进入目标分支。','已实际完成前后景/四类/HDR与Normal/Depth/Mesh审核且FAILED，需继续修复和新完整训练评测，质量验收后才提交并进入目标分支。');body+=final
(ROOT/'README.md').write_text(body)
paths=[CWD/'LIHAN/DOC/训练/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复训练报告.md',CWD/'LIHAN/DOC/报告/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复各项MAE评测报告.md']
for p in paths:p.write_text(body)
goal=CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md';goal.write_text(goal.read_text()+final)
index=CWD/'LIHAN/README.md';s=index.read_text();s=s.replace('完整20k/55k train exit0，RGB PSNR19.68220dB，四类实际目检仍FAILED_SO_FAR；全量七MAE/HDR最终审核待完成，未commit/push。','完整20k/55k train和9步后处理exit0，GS17.29780°/覆盖94.10134%，RGB PSNR19.68220dB；全部15帧七MAE/独立复算完成，前后景/四类/HDR与7°FAILED，未commit/push。');s+='\n- [B06完整七项MAE与失败验收报告](DOC/报告/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复各项MAE评测报告.md)：仓库原生kernel真实修复、GS下降.43302°，材质MAE上升，仍需源预测/RGB范围的优化。\n';index.write_text(s)
archive=CWD/'LIHAN/DOC/报告/archive/1010-12-jumpingjacks_dir-PBR仓库原生光栅修复'
for p in [ROOT/'README.md',ROOT/'metrics_summary.json',ROOT/'postprocess_state.json',ROOT/'runtime_history.jsonl',ROOT/'training_runtime.json']:
 shutil.copy2(p,archive/p.name)
for folder in ['commands','logs','inspection']:shutil.copytree(ROOT/folder,archive/folder,dirs_exist_ok=True)
# write_results自产生日志和post终态已完成，这次保留实际完整日志，manifest明确排除它自身。
sha={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
(archive/'archive_manifest.json').write_text(json.dumps(sha,ensure_ascii=False,indent=2)+'\n')
for checkout in [CWD,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 if checkout!=CWD:
  for p in paths+[goal,index]:shutil.copy2(p,checkout/p.relative_to(CWD))
  shutil.copytree(archive,checkout/archive.relative_to(CWD),dirs_exist_ok=True)
 for k,v in sha.items():assert hashlib.sha256((checkout/archive.relative_to(CWD)/k).read_bytes()).hexdigest()==v
print(json.dumps(dict(status='FAILED_QUALITY',GS_normal_MAE_deg=result['metrics'][2]['mae'],archive_files=len(sha),checkouts_verified=3,commit_allowed=False),ensure_ascii=False),flush=True)
