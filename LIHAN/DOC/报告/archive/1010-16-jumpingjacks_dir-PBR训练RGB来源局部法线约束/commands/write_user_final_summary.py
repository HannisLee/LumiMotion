"""仅在B10完整评测和实际目检结束后生成用户要求的最终总结；不训练。"""
from pathlib import Path
import json, hashlib, shutil, datetime, subprocess

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parent.parent
ARCHIVES = ROOT / 'LIHAN/DOC/报告/archive'
NAME = '1010-17-jumpingjacks_dir-原生PBR方向光迭代总结报告'
DOC = ROOT / 'LIHAN/DOC/报告' / (NAME + '.md')
assert not DOC.exists(), '总结已生成，禁止覆盖'
result = json.loads((RUN/'metrics_summary.json').read_text())
runtime = json.loads((RUN/'training_runtime.json').read_text())
state = json.loads((RUN/'postprocess_state.json').read_text())
assert result['status'] in ['FAILED_QUALITY', 'PASSED_QUALITY']
assert runtime['verified_training_session_closed'] and runtime['training_exit_code'] == 0
assert state['verified_session_closed'] and state['pipeline_exit_code'] == 0
assert result['manual_review']['actual_images_viewed']
assert len(result['metrics']) == 7 and len(result['frames']) == 15
history = [json.loads(s) for s in (RUN/'runtime_history.jsonl').read_text().splitlines()]
assert len(history) == 9 and all(s['exit_code'] == 0 for s in history)
cases = [
 ('B00','1010-02-jumpingjacks_dir-PBR方向光基线','原生PBR方向光初始基线'),
 ('B02','1010-05-jumpingjacks_dir-PBR原始RGBA严格输入','原RGB自身Alpha、GT仅方向、RGB估计强度'),
 ('B03','1010-07-jumpingjacks_dir-PBR材质初始化与投影修复','材质梯度初始化与范围投影修复'),
 ('B04','1010-11-jumpingjacks_dir-PBR固定来源形变锚定-几何重评','固定来源形变；使用更正后几何评分'),
 ('B05','1010-10-jumpingjacks_dir-PBR来源平面几何投影','来源预测平面真实XYZ/q投影'),
 ('B06','1010-12-jumpingjacks_dir-PBR仓库原生光栅修复','绑定仓库原生CUDA光栅及正确法线朝向'),
 ('B07','1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算','直接SUN Lambert+GGX、真实BVH光线预算'),
 ('B08','1010-14-jumpingjacks_dir-PBR来源输出gate微调','来源输出raw gate微调'),
 ('B09','1010-15-jumpingjacks_dir-PBR概率gate优化','直接概率优化、原生sigmoid保存与有限编码'),
 ('B10', RUN.name,'TRAIN RGB来源局部轴约束与source-local旋转运输'),
]
sources = {}
rows = []
for tag, name, change in cases:
 p = RUN/'metrics_summary.json' if tag == 'B10' else ARCHIVES/name/'metrics_summary.json'
 d = json.loads(p.read_text())
 sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
 rows.append((tag,name,change,d))
old = rows[-2][3]
envp = ARCHIVES/'1009-03-jumpingjacks_environment_light-MAE/metrics_summary.json'
lamp = ARCHIVES/'1009-04-jumpingjacks_direction_light-Stage2-GT-Lambertian/metrics_summary.json'
env, lam = [json.loads(p.read_text()) for p in [envp,lamp]]
for p in [envp,lamp]: sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
g = result['metrics'][2]
text = '# 原生LumiMotion、GT Lambertian与方向光PBR训练完整总结\n\n'
text += '日期：'+now+'（Asia/Tokyo）。用户要求完成当前训练后停止迭代；本报告生成时B10全部20,000步及9步后处理完成，不启动后续实验。\n\n'
text += '## 1. 最终结论\n\n'
text += '- B10：原Stage1 35000→55000独立Stage2 20,000步完成，训练和9步后处理均实际exit0且会话关闭；完整15测试帧七项MAE、独立CPU复算、实际代表图检查完成。\n'
text += '- 原有向、全前景像素加权GS Normal MAE **{:.8f}°**，覆盖率 **{:.8f}%**；目标<7° **{}**。人工画质 **{}**。\n'.format(g['mae'],g['coverage_pct'],result['normal_goal'],result['manual_review']['status'])
text += '- 相比B09同55k，GS误差变化{:+.8f}°；相比B00历史基线变化{:+.8f}°。最新实验不能仅凭约束loss下降认定法线更准确。\n'.format(g['mae']-old['metrics'][2]['mae'],g['mae']-rows[0][3]['metrics'][2]['mae'])
text += '- 停止迭代，保留全部成功/失败checkpoint、渲染、逐帧数组、日志和来源审计。不执行新的训练、commit、push或建分支；原目标中质量验收及后续分支条件尚未实现。\n\n'
text += '## 2. 已完成的前置任务\n\n'
text += '两套jumpingjacks_environment_light/direction_light均转换完毕，各150帧800²、135 TRAIN/15 TEST，time=frame/150、原相机/尺度不变。原版训练resolution2为400²、黑背景。environment_light文件名沿用来源，实际为双POINT；direction_light是两盏重叠SUN。\n\n'
text += 'env已用原版LumiMotion完整Stage1 35k+Stage2新增20k，并完成材质/HDR/NVS与MAE；dir已完成原版Stage1 35k、99827点和32形变tensor，随后完成GT Lambertian Stage2及离线评测。早期GT Lambertian兼容灯表保留两SUN的energy总和6，实际标量按135 TRAIN RGB标定；该阶段不套用后续严格PBR方向白色单位表的来源说明。原版转换借albedo文件Alpha，后续严格PBR转为原RGB自身Alpha并验证全部150帧Alpha逐位一致；没有重新训练Stage1或宣称最初转换就仅使用原RGB文件。\n\n'
text += '| 前置实验 | GS Normal MAE° | GS覆盖% | Depth Normal° | Mesh Normal° | Albedo MAE | Roughness MAE |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
def bymetric(d,name): return next(x for x in d['metrics'] if x['metric']==name)
for label,d in [('env原版LumiMotion',env),('dir GT Lambertian',lam)]:
 values = [bymetric(d,n)['mae'] for n in ['GS Render Normal MAE','Depth Normal MAE','Mesh Normal MAE','Albedo MAE','Roughness MAE']]
 text += '| {} | {:.8f} | {:.8f} | {:.8f} | {:.8f} | {:.8f} | {} |\n'.format(label,values[0],bymetric(d,'GS Render Normal MAE')['coverage_pct'],values[1],values[2],values[3],'N/A（Lambertian无输出）' if values[4] is None else '{:.8f}'.format(values[4]))
text += '\n这些是独立前置实验，不将其与最新方向光PBR当作同一配置的消融；历史报告PASS表示对应流程/复算验收，并非满足本次<7°或PBR画质门槛。前置完整命令与图像见训练1009-01、1009-02、1009-04及报告1009-03。\n\n'
text += '## 3. B00–B10迭代总表\n\n'
text += '| 实验 | 主要变化 | GS MAE° | GS覆盖% | Depth° | Mesh° | PSNR dB | 质量结论 |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n'
for tag,name,change,d in rows:
 m=d['metrics'];text+='| {} | {} | {:.8f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | FAILED |\n'.format(tag,change,m[2]['mae'],m[2]['coverage_pct'],m[3]['mae'],m[4]['mae'],d['rgb']['psnr'])
text += '\nB01（1010-04预测几何与RGB光强）因Alpha来源审计不符合严格输入要求中止，未完成正式55k/七项评测，保留现场，不填作完整结果。B04表采用1010-11更正后的几何重评，不采用旧1010-09错误runtime评分；材质/RGB沿用原实际产物。B00–B05沿用当时安装光栅，B06起绑定仓库原生CUDA内核，因此跨该界线差异包含实现修复，不能全归因单一优化项。B07之后直接SUN/BVH预算一致；B09→B10是同源、同20k/同评分的最新对比。全部完整实验均未达质量/<7°条件。\n\n'
text += '### 历次完整实验七项误差\n\n| 实验 | Albedo | Roughness | GS° | Depth Normal° | Mesh° | Expected Z m | Median Z m |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
for tag,name,change,d in rows:
 text += '| '+tag+' | '+' | '.join('{:.8f}'.format(m['mae']) for m in d['metrics'])+' |\n'
text += '\n每次的逐项覆盖、共同像素和missing180保留在各自metrics_summary.json；本总结档案复制所有完整实验实际指标JSON和来源SHA，原始误差数组留在各实验output。\n\n'
text += '## 4. 当前B10完整七项MAE\n\n| 指标 | B09 MAE | B10 MAE | B10−B09 | 单位 | B10覆盖% |\n| --- | ---: | ---: | ---: | --- | ---: |\n'
for a,b in zip(old['metrics'],result['metrics']):
 assert a['metric']==b['metric']
 text+='| {} | {:.8f} | {:.8f} | {:+.8f} | {} | {:.8f} |\n'.format(b['metric'],a['mae'],b['mae'],b['mae']-a['mae'],b['unit'],b['coverage_pct'])
c=result['normal_common_pixels']
text+='\n共同有效GS/Depth/Mesh像素 **{}/{}（{:.8f}%）**，共同口径MAE依次 **{:.8f}° / {:.8f}° / {:.8f}°**。GS缺失像素按180°惩罚 **{:.8f}°**。共同口径和惩罚项是附加诊断；目标仍使用原GS完整前景pixel-weighted定义。\n\n'.format(c['valid_pixels'],c['gt_pixels'],c['coverage']*100,*[c['angular_deg'][k]['mae_pixel_mean_deg'] for k in ['gs_render','depth_normal','mesh']],result['gs_normal_missing_180_penalty_deg'])
text+='### RGB重建\n\n| 指标 | B09 | B10 |\n| --- | ---: | ---: |\n'
for key in ['psnr','ssim','lpips','psnr_foreground','ssim_foreground_bbox','lpips_foreground_bbox']:
 text+='| {} | {:.8f} | {:.8f} |\n'.format(key,old['rgb'][key],result['rgb'][key])
text+='\n黑色背景占比较大，整体PSNR/SSIM须同时看前景指标和实际人物/平台；不以全图RGB较好替代几何验收。\n\n'
text+='## 5. 当前严格PBR评分定义与数据边界\n\n'
text+='- 全15 TEST：0010至0150每10帧，共415433个GT Alpha≥.5前景像素；保留人物、平台、难像素、全部99827 Gaussian。法线为有向world shading normal角度、像素加权，未转为无向/仅人物/仅共同像素的目标。\n'
text+='- Albedo：linear RGB，GT Alpha≥.99+erode2，预测缺失补0，无颜色拟合。Roughness：独立原blend EEVEE透明Alpha合成GT来源，GT Alpha≥.99+erode2、预测Alpha≥.9后去预乘；不是首命中粗糙度。\n'
text+='- Depth：独立原blend三角形camera-positive-Z、单位米，未对齐或缩放拟合；Expected/Median Z及Depth Normal分别报告。Mesh：每个目标时间独立融合同时间135 TRAIN预测视图，voxel.02/sdf.08/depth20/Alpha.5，不融合其他时间、GT或TEST。\n'
text+='- 训练/初始化/约束只使用原RGB、自身Alpha、相机time、原预测35k/固定NN、GT光方向。白色单位灯表仅保存方向；全局正强度由RGB拟合，不读取GT灯能量/颜色。GT albedo/normal/roughness/depth/mesh/blend仅供独立离线评分。\n\n'
text+='## 6. 管线修改及Preset落地\n\n'
text+='原生PBR保留Lambert+GGX F0=.04、真实Gaussian/BVH可见性、SH辅助重建。Direct SUN使用已知方向及RGB学习的全局强度；trace origin沿incident方向偏移.1，实际L个直接光线及启用时N个间接光线与分配预算一致。Direct默认关闭间接PBR照明但仍保留SH辅助训练，HDR/SUN和训练共享几何/材质。光栅按独立namespace绑定仓库原生source与.so SHA，未替换系统pip供其他训练使用。\n\n'
text+='后续修复材质初始化梯度与范围投影、源35k固定形变、真实XYZ/q平面投影、概率gate优化与native sigmoid有限保存。冻结源NN32/opacity/scale；几何warmup2000/ramp1000，canonical XYZ trust .08m、q60°，不声称gate/plane改变后的world位移仍受该canonical界限。每个完整实验从原35k独立初始化。\n\n'
text+='当前Preset **pbr_gt_directional_source_rgb_normal** 继承B09参数，新增 **pbr_source_local_rotation=True**、**lambda_pbr_rgb_normal_prior=2** 与资产路径。source-local运输为R_source_world(p)·R_source_canonicalᵀ·R_current_canonical，作用于真实Gaussian quaternion；初始source==current保持原d_rotation逐bit一致，运输后应用既有source-only plane15187。FrozenScene/周期预览/GeometryRun/Mesh/材质/HDR/SUN使用同一入口；不覆写评测法线图。\n\n'
text+='原135 TRAIN RGB线性亮度+自身Alpha/源预测可见性/源qXYZ/单位方向形成独立LS轴候选；新增约束29621非plane点（static17389/dynamic12232）。loss=sum(confidence·(1−dot))/sum(confidence)，按29621归一化，全部99827点仍参与原训练/评分。资产与全部source/135逐图SHA和checkpoint绑定；输入confidence/Gram/解/残差由CPU独立复算。它是RGB估计轴，未引入GT normal，也未保证GGX/阴影情况下成为准确法线。新Preset不增加optimizer参数；legacy默认False/0。完整Adam状态未保存，不宣称无损续训。\n\n'
text+='具体实现：[stage2参数]('+str(ROOT/'arguments/stage2.py')+')、[Preset]('+str(ROOT/'scripts/loss_stage2.py')+')、[RGB约束与运输]('+str(ROOT/'scene/pbr_rgb_normal.py')+')、[训练入口]('+str(ROOT/'scripts/train_stage2.py')+')；完整修改历史在LIHAN/DOC/修改/1010系列。\n\n'
text+='## 7. B10实际目检\n\n'
review=result['manual_review']
for key,label in [('rgb_background','RGB与背景'),('alpha','Alpha'),('albedo_static_dynamic','Albedo与静动态'),('normal_depth','法线/深度')]: text+='- **'+label+'**：'+review[key]+'\n'
text+='- **HDR/SUN**：'+review['relighting']+'\n'
text+='\n实际查看TEST10/80/150、insights1/75/150、HDR1/76/150、SUN0/36/71及normal-depth代表图；六视频均全帧解码，但未宣称人工逐帧观看全部视频。重光照没有对应GT，响应存在只说明功能可用。\n\n'
for rel in ['inspection/rgb_final_55000_comparison.png','inspection/four_categories_final_55000.png','inspection/relight_final_55000.png']:
 if (RUN/rel).is_file():text+='!['+Path(rel).stem+']('+str(RUN/rel)+')\n\n'
text+='## 8. 已有诊断证据与限制\n\n'
text+='warmup前2000步B09/B10 PhotoL1最大差2.68e−7，说明初始外观保持。新增轴loss从.0992468下降至55k .00371028，确实优化；它不是GT MAE。40k中间GS15.71225125°/覆盖93.93837%、Depth25.68376882°，中间Mesh未生成，不当正式七项或三法线共同评分。source confident q raw norm中位1.029/P991.805，没有支持“普遍巨大的q norm导致失效”的证据。\n\n'
text+='源RGB候选29621点在TRAIN1/75/149只贡献前景Alpha合成质量38.47%/28.52%/39.15%，plane贡献12.07%/16.43%/11.85%；点数不等于像素约束覆盖。近半贡献来自二者之外，单纯增大loss权重没有充分覆盖证据。source-only放松静态/axis条件后稳定近平台点仅15187→16110且无新增动态，简单放宽条件未解决主要缺口。此分组只用于诊断，未删评分像素。\n\n'
text+='停止要求之前完成的六TRAIN来源Jacobian诊断：动态近平台约37–44%、其余动态约44–48%det≤0；可逆动态轴与native旋转轴差约35–52°。源位置MLP不保证物理可逆体积形变，直接以J逆转置替换旋转候选未接受、未修改生产。有限差分相对最大差约.9–3.4%，近零角有float32舍入局限；不将该轴相容性诊断当GT法线准确度。全部现有诊断归档，不启动新候选实验。\n\n'
text+='## 9. 执行、审计和复现入口\n\n'
text+='训练 '+runtime['started_at']+' → '+runtime['finished_at']+'，source35000/target55000，正式warmup2000/ramp1000/λ2。训练会话4419、checkpoint审计10940、后处理65390均真实终止；40k/50k/55k审计exit0。226 CPU、20步Preset、真实五参数梯度/Direct预算与共享buffer检查已实际完成；本次总结未启动额外测试或训练。\n\n'
text+='| 后处理步骤 | 实际退出码 | 秒 |\n| --- | ---: | ---: |\n'
for row in history:text+='| {} | {} | {:.2f} |\n'.format(row['step'],row['exit_code'],row['elapsed_seconds'])
text+='\n全量四类150×4、HDR150×2048采样、SUN72、六视频、15 albedo/roughness/normal/depth及15独立Mesh完成；CPU七MAE、roughness来源和artifact审计PASS。476核心+两source快照SHA冻结并核对，native kernel及来源资产绑定。自动数值PASS不等于人工质量PASS。完整train_argv、planned_commands、所有shell/封装命令和日志保留在本次输出，训练/评测文档均为最终状态。\n\n'
text+='### 完整实际训练命令\n\n```bash\n'+(RUN/'commands/train.sh').read_text()+'\n```\n\n'
text+='复现渲染/评测时按 [planned_commands.json]('+str(RUN/'planned_commands.json')+') 的九个完整argv顺序，逐步命令、环境、source/model/iteration、输出和日志见 [本次README]('+str(RUN/'README.md')+')；本报告不执行这些命令。\n\n'
text+='### 主要产物\n\n'
for label,p in [('最终模型与55000 PLY',RUN/'model_mlp'),('七项指标与全部覆盖',RUN/'metrics_summary.json'),('人工目检记录',RUN/'inspection/visual_review.json'),('独立MAE复算',RUN/'inspection'),('训练文档',ROOT/'LIHAN/DOC/训练'/(RUN.name+'训练报告.md')),('评测报告',ROOT/'LIHAN/DOC/报告'/(RUN.name+'各项MAE评测报告.md')),('逐次GOAL',ROOT/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'),('B10完整归档',ARCHIVES/RUN.name),('RGB原型/20步冒烟',ROOT/'output/smoke_test'),('已有Jacobian诊断',ARCHIVES/'1010-19-dir-来源形变Jacobian法线运输诊断')]: text+='- ['+label+']('+str(p)+')\n'
text+='\n完整模型、大规模渲染和原评分数组留在output；归档保存报告、核心源码双快照、脚本、日志、指标、来源SHA及代表图。没有删除失败产物；没有把大型权重描述为已复制进文档归档。\n\n'
checkpoint = json.loads((RUN/'inspection/checkpoint_55000_audit.json').read_text())
text+='### 最终checkpoint与来源核验\n\n'
text+='全部{}点finite；opacity/scale与32来源NN冻结；canonical最大位移{:.9f}m、最大q变化{:.8f}°、概率gate最大变化{:.8f}；RGB拟合全局强度{:.8f}，不是GT能量。材质投影界限通道比例{:.6%}，硬clamp比例{}；大量通道到界提示仍有材质/照明拟合限制，不能据此认定物理分解成功。\n\n'.format(checkpoint['gaussians'],checkpoint['max_displacement'],checkpoint['max_rotation_deg'],checkpoint['max_feature_probability_delta'],checkpoint['actual_light_scale'],checkpoint['channels_at_projected_bound_fraction'],checkpoint['clamped_albedo_channel_fraction'])
text+='原35k PLY SHA256：8ebd88b6f847a20178458777d51aa191d5f67ba4bbc490fd1e99ef62f66d5529；原35k MLP与最终冻结MLP：e9ddf76b59a1ef8a85caa16891473e3a60663d903a5ad3928ac3e6ca46892bf5。最终55k PLY与RGB来源资产SHA见下表。\n\n| 文件 | SHA256 |\n| --- | --- |\n'
for path,sha in checkpoint['rgb_normal']['source_files'].items(): text+='| '+Path(path).name+' | '+sha+' |\n'
text+='\n原生光栅source SHA256：82b1840f4039649e548239445e8c83a6e2018d57c4fc0f7819c835f31aa098d0；.so SHA256：35bb1fd0a6f891df6e1e52082413b06d24946d5762b9157ade415b50bab7a3c0。模型pbr_rasterizer.json保留namespace/实际路径。\n\n'
text+='## 10. 代码与停止状态\n\n'
text+='分支 **'+branch+'**，HEAD **'+head+'**；修改留在工作区，未进行本轮commit/push，未创建1010-goal-opt-pbr-jacks。最初“画质通过后提交，再建分支达<7°”的条件未达成。用户已明确停止后续迭代，持续目标暂停；只完成当前训练的评测、归档与本总结。报告/索引/GOAL与档案同步当前、LumiMotion-perlight、LumiMotion三个checkout，其他checkout生产源码没有同步修改。\n'
DOC.write_text(text)
(RUN/'inspection/final_summary_sources.json').write_text(json.dumps(dict(generated_at=now,source_metrics_sha256=sources,branch=branch,HEAD=head,iterations_stopped=True,new_training_started=False),ensure_ascii=False,indent=2)+'\n')
archive=ARCHIVES/NAME
assert not archive.exists()
archive.mkdir()
shutil.copy2(DOC,archive/DOC.name)
for p in [RUN/'inspection/final_summary_sources.json',RUN/'metrics_summary.json',RUN/'training_runtime.json',RUN/'postprocess_state.json',RUN/'process_handles.json',RUN/'runtime_history.jsonl',Path(__file__)]:shutil.copy2(p,archive/p.name)
for tag,name,change,d in rows:
 p=RUN/'metrics_summary.json' if tag=='B10' else ARCHIVES/name/'metrics_summary.json'
 shutil.copy2(p,archive/(tag+'_metrics_summary.json'))
for label,p in [('env',envp),('GT_Lambertian',lamp)]:shutil.copy2(p,archive/(label+'_metrics_summary.json'))
manifest={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
goal=ROOT/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
with goal.open('a') as f:f.write('\n## 用户停止迭代后的完整总结（'+now+'）\n\nB10全部20k和九步后处理实际exit0/会话关闭；原GS MAE{:.8f}°/覆盖{:.8f}%，人工质量{}、<7°{}。停止后续迭代，未commit/push/新分支，持续目标暂停。完整总结：报告/{}；现有全部失败产物保留。\n'.format(g['mae'],g['coverage_pct'],review['status'],result['normal_goal'],DOC.name))
index=ROOT/'LIHAN/README.md'
with index.open('a') as f:f.write('\n- [当前训练结束与完整迭代总结](DOC/报告/'+DOC.name+')：B00–B10/前置env与GT Lambertian、七MAE/真实画质、审计和产物路径；按用户要求停止迭代。\n')
for checkout in [ROOT,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 if checkout!=ROOT:
  for p in [DOC,goal,index]:shutil.copy2(p,checkout/p.relative_to(ROOT))
  shutil.copytree(archive,checkout/archive.relative_to(ROOT),dirs_exist_ok=True)
 for rel,sha in manifest.items():assert hashlib.sha256((checkout/archive.relative_to(ROOT)/rel).read_bytes()).hexdigest()==sha
print(json.dumps(dict(summary=str(DOC),archive_files=len(manifest),SHA_verified_checkouts=3,GS_normal_MAE_deg=g['mae'],quality=review['status'],iterations_stopped=True),ensure_ascii=False),flush=True)
