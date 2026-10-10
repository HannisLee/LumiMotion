"""全量B05实际四类/重光照/Normal Depth目检结论及完整失败归档；不改模型或统计。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
result=json.loads((ROOT/'metrics_summary.json').read_text())
assert len(result['metrics'])==7 and result['independent_audit']['status']=='PASS'
state=json.loads((ROOT/'postprocess_state.json').read_text())
assert state['status']=='PIPELINE_COMPLETED_REVIEW_PENDING'
steps=[json.loads(x) for x in (ROOT/'runtime_history.jsonl').read_text().splitlines()]
assert len(steps)==9 and all(x['exit_code']==0 for x in steps)
assert json.loads((ROOT/'training_runtime.json').read_text())['verified_training_session_closed']
manifest=json.loads((ROOT/'code_manifest.json').read_text())
for rel,digest in manifest.items():
    assert hashlib.sha256((CWD/rel).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'commands/code_snapshot'/rel).read_bytes()).hexdigest()==digest
review={'status':'FAILED','reviewed_at':now,'iteration':55000,
    'rgb_background':'黑背景正常且主体平台存在；人物头手/衣服仍明显斑驳、稀疏，平台脚周伪起伏/灰块；与B04相近，PSNR略低。GT背光本来暗，不仅凭暗失败。',
    'alpha':'主体平台基本覆盖，边缘薄雾化与稀疏残影；完整数值覆盖率已报告。',
    'albedo_static_dynamic':'正背面光照仍烘入衣服/身体，平台棋盘格和脚周污染；冻结feature未修复源静动态误分类。',
    'normal_depth':'GS身体斑驳、平台脚周伪起伏/异常轴；Depth更噪，Mesh局部头手腿缺面，0080明显。15187 plane组实际轴正确，但旧installed kernel在0075等三个训练帧错误反向；不是评分后翻转。',
    'relighting':'HDR与转灯背景黑、人物平台可见且有照明响应；人体/脚周斑驳仍明显。未提供重光照GT，不作精度声明。',
    'reviewed_test_frames':[10,80,150],'reviewed_insight_frames':[1,75,150],
    'reviewed_hdr_frames':[1,76,150],'reviewed_directional_frames':[0,36,71],
    'images':['inspection/rgb_final_55000_comparison.png','inspection/four_categories_final_55000.png','inspection/relight.png',
              'normal_depth/comparison_0010.png','normal_depth/comparison_0080.png','normal_depth/comparison_0150.png',
              'inspection/plane_raster_group_comparison.png'],
    'normal_goal':'FAILED','commit_allowed':False,'subsequent_native_kernel_diagnostic_is_final_training':False}
(ROOT/'inspection/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
result.update(status='FAILED_QUALITY',manual_review=review,normal_goal='FAILED',commit_allowed=False,
              pipeline_exit_code=0,verified_postprocess_session_closed=True,source_code_manifest_verified=True,
              kernel_runtime_provenance=str(ROOT/'inspection/raster_kernel_provenance.json'))
(ROOT/'metrics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
artifact=json.loads((ROOT/'inspection/artifact_checks.json').read_text());artifact.update(manual_review='FAILED',visual_review=review)
artifact['scope']=artifact['scope'].replace('人工审阅待完成','人工实际审阅FAILED；自动文件/数值PASS不等于质量PASS')
(ROOT/'inspection/artifact_checks.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
state.update(status='FINISHED_QUALITY_FAILED',reviewed_at=now,pipeline_exit_code=0,verified_session_closed=True,
             final_gs_normal_mae_deg=result['metrics'][2]['mae'],commit_allowed=False)
(ROOT/'postprocess_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
body=(ROOT/'README.md').read_text()
body=body.replace('人工质量审閱待完成','人工质量已实际审阅FAILED')
body=body.replace('完整20000步训练、渲染、全部15测试帧MAE与独立复算完成；人工质量已实际审阅FAILED。不能据此标记图像质量或7°目标PASS。',
                  '完整20000步训练、全量渲染、全部15测试帧七项MAE及独立复算均完成；最终质量FAILED、7°FAILED，保留完整失败产物，未commit/push。')
start=body.index('## 四类目检（待人工回填）');end=body.index('## 审计及结论',start)
section='## 四类实际目检：FAILED\n\n'+''.join('- '+review[k]+'\n' for k in ['rgb_background','alpha','albedo_static_dynamic','normal_depth'])
section+='\nHDR/转灯：'+review['relighting']+'\n\n实际看0010/0080/0150测试图、0001/0075/0150四类图、HDR0001/0076/0150及转灯0000/0036/0071；六视频完整解码通过，代表图目检不宣称人工逐视频逐帧观看。代表证据：inspection/visual_review.json、rgb_final_55000_comparison.png、four_categories_final_55000.png、relight.png及normal_depth/comparison_0010/0080/0150.png。全部浮点图、Mesh、权重与日志保留。\n\n'
body=body[:start]+section+body[end:]
body=body.replace('人工图像质量：PENDING','人工图像质量：FAILED')
body=body.replace('先完成前后景人工审核，必要时修复重训，再进入目标分支。','已完成前后景人工审核且FAILED；后续修复需新完整训练及全量评测后才允许提交和进入目标分支。')
previous=json.loads((CWD/'output/1010-09-jumpingjacks_dir-PBR固定来源形变锚定/metrics_summary.json').read_text())
delta='| 指标 | B04已更正 | B05 | 差值B05−B04 |\n| --- | ---: | ---: | ---: |\n'
for a,b in zip(previous['metrics'],result['metrics']):
    assert a['metric']==b['metric']
    delta+='| {} | {:.8f} | {:.8f} | {:+.8f} |\n'.format(b['metric'],a['mae'],b['mae'],b['mae']-a['mae'])
common=result['normal_common_pixels']
final='\n## 最终验收与B04同55k更正结果比较\n\n'+delta
final+='\nGS改善1.04661466°，Depth/Mesh和两个Z误差均下降；roughness略升、RGB PSNR19.37506314 vs19.47597408略低，仍存在明显伪影，质量FAILED与7°FAILED。全前景评分定义不变，全部99827点/15测试帧保留。共同369930/415433（89.04684991%），GS/Depth/Mesh共同MAE16.77003773°/26.15792626°/19.37026547°，missing180为27.65368647°。\n\n训练exec42059及后处理exec86603均实际exit0并关闭，9步骤均exit0；后处理结束06:47:49 JST。179CPU、逐buffer共享运行时、92来源SHA及15time×135train Mesh、roughness独立15帧/41来源、六视频完整解码PASS；不将数值执行PASS当质量PASS。核心manifest与当时源/快照逐SHA核验PASS。\n\n旧installed CUDA实际kernel来源和法线Z符号缺陷已记录inspection/raster_kernel_provenance.json。B05训练/渲染/评分共享该旧kernel，分数是本轮实际模型结果，不覆盖为新kernel分数。后续独立原生构建诊断0optimizer、无GT标签且不改变环境installed包；不作为已完成的新训练或质量提升。\n'
body+=final
(ROOT/'README.md').write_text(body)
paths=[CWD/'LIHAN/DOC/训练/1010-10-jumpingjacks_dir-PBR来源平面几何投影训练报告.md',
       CWD/'LIHAN/DOC/报告/1010-10-jumpingjacks_dir-PBR来源平面几何投影各项MAE评测报告.md']
for p in paths:p.write_text(body)
goal=CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
goal.write_text(goal.read_text()+'\n## B05最终完整验收（'+now+'）\n'+final)
index=CWD/'LIHAN/README.md';text=index.read_text()
text=text.replace('173 CPU及真实GPU零步PASS；最终训练/质量/七项MAE待完成。',
 '179 CPU及真实GPU零步PASS；完整20k及全15七MAE/独立复算完成，GS17.73082°，前后景/四类/HDR实际质量FAILED、7°FAILED。')
text=text.replace('B04旧几何MAE SUPERSEDED，完整15帧重评进行中；179 CPU及真实GPU缓冲逐像素一致PASS；B05训练继续，未改变评分。',
 'B04旧几何MAE SUPERSEDED，完整15帧重评已更正；179 CPU及真实GPU缓冲逐像素一致PASS；B05同入口全量评测已完成，未改变评分。')
text+='\n- [B05完整七项MAE与失败验收报告](DOC/报告/1010-10-jumpingjacks_dir-PBR来源平面几何投影各项MAE评测报告.md)：GS17.73082°，较B04更正18.77743°下降1.04661°；质量及7°FAILED。另发现旧installed CUDA法线Z符号缺陷，独立原生核验继续，未提交。\n'
index.write_text(text)
archive=CWD/'LIHAN/DOC/报告/archive/1010-10-jumpingjacks_dir-PBR来源平面几何投影'
for p in [ROOT/'README.md',ROOT/'metrics_summary.json',ROOT/'postprocess_state.json',ROOT/'runtime_history.jsonl']:
    shutil.copy2(p,archive/p.name)
for folder in ['commands','logs','inspection']:
    shutil.copytree(ROOT/folder,archive/folder,dirs_exist_ok=True)
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
    for p in paths+[goal,index]:shutil.copy2(p,checkout/p.relative_to(CWD))
    shutil.copytree(archive,checkout/archive.relative_to(CWD),dirs_exist_ok=True)
print(json.dumps({'status':'FAILED_QUALITY','gs_normal_mae_deg':result['metrics'][2]['mae'],'commit_allowed':False,'docs_synced':3},ensure_ascii=False),flush=True)
