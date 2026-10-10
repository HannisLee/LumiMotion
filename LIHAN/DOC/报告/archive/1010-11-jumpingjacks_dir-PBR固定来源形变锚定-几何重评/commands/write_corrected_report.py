"""实际B04全15重评及人工误差图完成后保存更正报告，旧几何结果保留。"""
from pathlib import Path
import json,csv,hashlib,shlex,shutil
from datetime import datetime,timezone,timedelta
import numpy as np
root=Path(__file__).resolve().parent.parent;cwd=root.parent.parent;old=cwd/'output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定'
assert json.loads((root/'evaluation_state.json').read_text())['status']=='COMPLETED_REVIEW_PENDING'
G=json.loads((root/'normal_depth/metrics.json').read_text());fg=G['summary']['foreground']
A=json.loads((root/'albedo/iteration_55000/metrics.json').read_text())['summaries']['raw']['foreground']
R=json.loads((root/'roughness/metrics.json').read_text())['summaries']['foreground']
RGB=json.loads((old/'metrics_summary_legacy_unanchored.json').read_text())['rgb']
audit=json.loads((root/'inspection/independent_audit.json').read_text());assert audit['status']=='PASS'
assert len(G['frames'])==15 and [x['source_frame'] for x in G['frames']]==list(range(10,151,10))
assert json.loads((root/'roughness/verification.json').read_text())['passed'] if (root/'roughness/verification.json').exists() else '"passed": true' in (root/'logs/verify_roughness.log').read_text()
# 独立补验缺失180惩罚，不滤任何前景/难帧。
penalty_sum=0.;gt_count=0
for row in G['frames']:
 with np.load(root/'normal_depth'/('prediction_%04d.npz'%row['source_frame'])) as d:
  gt=d['gt_normal'].astype(np.float64);pred=d['normal'].astype(np.float64)
  ng=np.linalg.norm(gt,axis=-1);npred=np.linalg.norm(pred,axis=-1)
  eligible=d['gt_foreground'].astype(bool)&np.isfinite(gt).all(-1)&(ng>1e-6)
  valid=eligible&np.isfinite(pred).all(-1)&(npred>1e-6)&np.isfinite(d['alpha'])&(d['alpha']>=.5)
  angles=np.degrees(np.arccos(np.clip((gt[valid]*pred[valid]).sum(-1)/(ng[valid]*npred[valid]),-1,1)))
  penalty_sum+=float(angles.sum())+180*int((eligible&~valid).sum());gt_count+=int(eligible.sum())
penalty=penalty_sum/gt_count
np.testing.assert_allclose(penalty,fg['normal']['angular_with_missing_deg']['pixel_mean'],atol=1e-9,rtol=1e-9)
(root/'inspection/missing180_independent.json').write_text(json.dumps(dict(status='PASS',GT_pixels=gt_count,penalty_deg=penalty),indent=2)+'\n')
entries=[dict(metric='Albedo MAE',mae=A['mae_linear'],unit='linear RGB',coverage_pct=A['coverage_pct']),dict(metric='Roughness MAE',mae=R['mae'],unit='r',coverage_pct=R['coverage_pct'])]
for name in ['gs_render','depth_normal','mesh']:
 m=fg['normal_metrics'][name];entries.append(dict(metric=m['label'],mae=m['mae_pixel_mean_deg'],unit='degree',coverage_pct=100*m['coverage']))
for name in ['expected','median']:
 m=fg['depth'][name];entries.append(dict(metric=name.capitalize()+' Z Depth MAE',mae=m['mae_pixel_mean'],unit='m',coverage_pct=100*m['coverage']))
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
result=dict(status='COMPLETE_CORRECTED_MAE_QUALITY_FAILED',iteration=55000,frames=15,updated_at=now,model_path=str(old/'model_mlp'),source_geometry_evaluation_runtime='FrozenScene_shared_anchor',corrected_evaluation_output=str(root),metrics=entries,rgb=RGB,normal_common_pixels=fg['common_normal_metrics'],gs_normal_missing_180_penalty_deg=penalty,independent_audit=audit,material_geometry_gt_used_for_training=False,training_gt_inputs=['light_direction_only'],light_GT_photometry_used=False,light_intensity=json.loads((old/'metrics_summary_legacy_unanchored.json').read_text())['light_intensity'],commit_allowed=False)
(root/'metrics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (root/'metrics_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['metric','mae','unit','coverage_pct']);w.writeheader();w.writerows(entries)
table='| 指标 | MAE | 单位 | 覆盖率 |\n| --- | ---: | --- | ---: |\n'+''.join('| {} | {:.8f} | {} | {:.6f}% |\n'.format(e['metric'],e['mae'],e['unit'],e['coverage_pct']) for e in entries)
common=fg['common_normal_metrics'];common_text='共同有效像素{}/{}，覆盖{:.6f}%；GS缺失180惩罚{:.8f}°。\n\n'.format(common['valid_pixels'],common['gt_pixels'],100*common['coverage'],penalty)
common_text+='| 法线 | 共同像素MAE（度） |\n| --- | ---: |\n'+''.join('| {} | {:.8f} |\n'.format(k,v['mae_pixel_mean_deg']) for k,v in common['angular_deg'].items())
manual=dict(status='FAILED',reviewed_geometry_frames=[10,80,150],scope='真实更正后的3帧Normal/Depth/Mesh误差图；非逐张150审阅',normal_depth='平台脚周伪起伏和异常方向、人物头手/衣服/四肢斑驳；Depth更噪、Mesh腿部/手/局部头缺失，尤其0080。',rgb_alpha_albedo_hdr='原B04原生FrozenScene图像不受GeometryRun遗漏影响，沿用其已实际审阅FAILED；背景正常而人体斑驳、脚周伪结构、Alpha边雾、Albedo残留照明。',commit_allowed=False,normal7_status='FAILED')
(root/'inspection/visual_review.json').write_text(json.dumps(manual,ensure_ascii=False,indent=2)+'\n')
body='''# 1010-11 B04原生PBR固定来源形变锚定：完整几何更正评测

更新时间：'''+now+'''
最终状态：FAILED（图像质量与原GS法线<7°）；运行时/全15离线重评/独立复算PASS。未commit/push/新建目标分支。

## 更正范围与真实训练

B04已经完整20000步35k→55k（2026-10-10 04:30:08.950→05:14:28 JST，loop44:01）；本次0训练步，不重跑Stage1或Stage2，不改权重/材质/光强。Stage1来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp。训练135帧/15测试帧、400×400黑背景，训练仅原RGB/自身RGBA、相机/time、来源预测与GT方向；GT材质/法线/深度/mesh/颜色/能量禁止训练，GT只离线评分。

旧GeometryRun漏读Anchor，旧GS18.91234950°、Depth27.51005704°、Mesh22.23676747°及depth/common/missing180标记SUPERSEDED。旧数组计算自洽不证明运行时正确，不能据此比较Anchor效果。旧权重/数组/Mesh/统计/日志全部保留，旧metrics_summary_legacy_unanchored.json保存；本次使用FrozenScene共享源XYZ查询真实渲染几何，再完整15time×135train重新TSDF Mesh及全部15几何评分。

RGB、albedo、roughness原已使用FrozenScene，不受该遗漏影响，本次原产物以symlink引用并再次独立复算/hash核验，不冒充重新训练或重新渲染。自包含报告归档仅引用同一55k来源，以哈希核对。

## 正式七项MAE：全15帧

'''+table+'\n'+common_text+'''

原评分口径保留：GS/Depth/Mesh为原32-bit世界着色Normal有向角度，GT alpha≥.5所有FG，没有按物体/困难像素过滤。Expected/Median Z单位米，无scale/shift/ICP校准。Albedo raw linear RGB、GT alpha≥.99内缩2，缺失预测0；Roughness归一化本征r，源EEVEE/Principled Alpha透射混合GT、同GT core且pred alpha≥.9，覆盖率单列。透明r为可见透射率×Alpha加权值，不声称唯一第一命中值。

Mesh每个测试time独立融合所有135 TRAIN相机的预测Expected Z，voxel.02/sdf_trunc.08/depth20/alpha.5；不跨time、不用test位姿/GT几何。原模型99827全保留、MLP32tensor不变、source_xyz Anchor与当前marker同SHA。无删点/删帧/改评分。完整原定义见1010-09训练文档及1010-08修改文档。

## RGB与目检

'''+json.dumps(RGB,ensure_ascii=False)+'''

原B04 150×RGB/Alpha/Normal/Albedo、72方向光、150 HDR2048及6完整视频已生成并解码，原生FrozenScene正确，不因几何重评重复改写。原四类/HDR审核结果沿用FAILED：背景正常，人物平台存在但衣服/头手斑驳、脚周假几何/残影、Alpha边雾、Albedo光照残留。代表图仍在原B04 inspection及model_mlp/renders_stage2/ours_55000各分类目录；HDR与转灯只功能/目检，无GT精度结论。

本次实际查看更正后的normal_depth/comparison_0010.png、0080、0150：平台脚周伪起伏、身体法线斑驳、Expected/Median脚周明显Z误差；Depth法线较噪，Mesh局部手/腿/头缺面，0080更明显。仅3代表帧实际人工检查，不声称所有15/150逐张。真实源运行时修正略改善分数，但18.77743450°远高于7°，不能提交。B03同55k GS18.89548239°，B04更正18.77743450°小幅改善；训练RGB不变，不能称已解决图像质量。

## 可复核证据与限制

179 CPU实际exit0；真实GPU B04 55k/B05 source Plane逐像素 normal/alpha/Expected Z与原生PBR首中尾max error0、timeoverride一致、源hash不改、0优化步。重评3步执行实际exit0，独立CPU三法线/common/2depth/albedo与90来源SHA PASS；roughness独立复算15帧41SHA再次PASS；missing180单独CPU全15复算PASS。runtime_history.jsonl、inspection/independent_audit.json、missing180_independent.json、gpu_audit.json和logs可复核。完整权重/浮点数组保留，归档保存报告/metrics/代码与命令快照/logs/代表误差图，未删除失败产物。

B05训练持续，训练/renderer/MLP/参数不改，只有未执行离线GeometryRun修复，已保存旧launch manifest/snapshot和SHA修订记录。最终B05全部七项MAE仍待55k完成。

## 实际重评时间

'''+(root/'runtime_history.jsonl').read_text()+'\n状态：'+(root/'evaluation_state.json').read_text()+'\n\n## 完整命令\n\n'
plans=json.loads((root/'planned_commands.json').read_text())
plans.append(dict(name='verify_roughness',argv=['/home/lihan/miniconda3/envs/lumimotion/bin/python','-u','-m','scripts.verify_roughness_eval',str(root/'roughness')],log=str(root/'logs/verify_roughness.log')))
(root/'completed_commands.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
for c in plans:body+='### '+c['name']+'\n\n```bash\n'+shlex.join(c['argv'])+'\n```\n\n日志：'+c['log']+'\n\n'
body+='### 原B04完整训练命令\n\n```bash\n'+shlex.join(json.loads((old/'train_argv.json').read_text()))+'\n```\n\n训练日志：'+str(old/'logs/train.log')+'\n\n原全render/HDR/七项命令和人工审核见1010-09报告，几何分数以上方更正为准。\n'
(root/'README.md').write_text(body)
doc=cwd/'LIHAN/DOC/报告/1010-11-jumpingjacks_dir-PBR固定来源形变锚定几何更正与七项MAE报告.md';doc.write_text(body)
# 更新B04正式summary，以legacy文件保留错误旧分数；不改旧浮点/渲染/权重。
(old/'metrics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
intro='\n> 几何运行时更正已完成：旧三法线/两depth已作废，以[完整更正报告](../报告/1010-11-jumpingjacks_dir-PBR固定来源形变锚定几何更正与七项MAE报告.md)及output/1010-11更正数组为准；实际GS18.77743450°，质量/7°仍FAILED。\n\n'
updated=[]
for p in [old/'README.md',cwd/'LIHAN/DOC/训练/1010-09-jumpingjacks_dir-PBR固定来源形变锚定训练报告.md',cwd/'LIHAN/DOC/报告/1010-09-jumpingjacks_dir-PBR固定来源形变锚定各项MAE评测报告.md']:
 s=p.read_text();i=s.index('\n')+1;p.write_text(s[:i]+intro+s[i:]+'\n## 正式更正七项结果\n\n'+table+'\n'+common_text+'\n详见1010-11完整更正报告。\n');updated.append(p)
goal=cwd/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
goal.write_text(goal.read_text()+'\n## B04完整几何更正最终结果\n\n'+table+'\n'+common_text+'''\n旧B04几何分数SUPERSEDED且保留，更正全15time×135train Mesh/全15 geometry/独立复算及90来源SHA/rough41来源/missing180实际PASS；GPU原生buffer逐像素一致PASS。实际看0010/0080/0150更正图：平台脚周伪起伏、身体法线斑驳、Depth噪声、Mesh局部手/腿/头缺面，0080更明显；质量/7°FAILED。B04 actual18.77743 vsB03 18.89548小幅改善，不能再以旧18.91235声称Anchor未改善。RGB/albedo/rough原运行时正确不改，本次0训练步；B05继续最终20k，未commit/push/新建分支。\n''')
index=cwd/'LIHAN/README.md';index.write_text(index.read_text()+'\n- [B04完整七项MAE几何更正报告](DOC/报告/1010-11-jumpingjacks_dir-PBR固定来源形变锚定几何更正与七项MAE报告.md)：全15重评/独立复算PASS，真实GS18.77743°，旧GeometryRun分数SUPERSEDED；图像质量与7°FAILED。\n')
archive=cwd/'LIHAN/DOC/报告/archive/1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评';archive.mkdir(parents=True,exist_ok=False)
for name in ['README.md','metrics_summary.json','metrics_summary.csv','planned_commands.json','completed_commands.json','code_manifest.json','runtime_history.jsonl','evaluation_state.json']:shutil.copy2(root/name,archive/name)
for name in ['commands','logs','inspection']:shutil.copytree(root/name,archive/name)
for rel in ['normal_depth/metrics.json','normal_depth/comparison_0010.png','normal_depth/comparison_0080.png','normal_depth/comparison_0150.png','predicted_meshes/manifest.json','albedo/iteration_55000/metrics.json','roughness/metrics.json']:
 dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dest)
manifest={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
oldarchive=cwd/'LIHAN/DOC/报告/archive/1010-09-jumpingjacks_dir-PBR固定来源形变锚定'
(oldarchive/'geometry_runtime_corrected_final.md').write_text(table+'\n'+common_text+'\n完整更正报告/90source audit见1010-11归档；原旧数组/表/日志保留并SUPERSEDED。\n')
m={str(p.relative_to(oldarchive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in oldarchive.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
(oldarchive/'archive_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 for p in [doc,goal,index]+[p for p in updated if cwd/'LIHAN' in p.parents]+[oldarchive/'geometry_runtime_corrected_final.md',oldarchive/'archive_manifest.json']:shutil.copy2(p,checkout/p.relative_to(cwd))
 shutil.copytree(archive,checkout/archive.relative_to(cwd))
print(json.dumps(result,ensure_ascii=False,indent=2))
