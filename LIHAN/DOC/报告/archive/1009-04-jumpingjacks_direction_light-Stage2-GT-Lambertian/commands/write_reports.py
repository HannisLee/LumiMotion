"""完成人工审核后，写入训练/评测正式文档、归档小型证据并同步本地文档。"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,shutil,shlex,re
ROOT=Path(__file__).resolve().parent.parent;REPO=ROOT.parent.parent
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
review=json.loads((ROOT/'inspection/visual_review.json').read_text());assert review['status']=='PASS'
s=json.loads((ROOT/'metrics_summary.json').read_text())
for name in ['independent_audit','checkpoint_55000','gt_audit']:
 assert json.loads((ROOT/'inspection'/f'{name}.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'inspection/artifact_checks.json').read_text())['automatic_status']=='PASS'
s.update(status='PASS',completed_at=stamp,acceptance_scope='训练/产物/数值/真值与代表图验收，非模型质量阈值')
(ROOT/'metrics_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
losses=[json.loads(line) for line in (ROOT/'model_mlp/losses.jsonl').read_text().splitlines()]
assert losses[-1]['iteration']==55000 and len(losses)==200
training_seconds=losses[-1]['elapsed_seconds']
tag=ROOT.name;report=REPO/'LIHAN/DOC/报告'/f'{tag}-各项MAE与三类Normal评测报告.md';train=REPO/'LIHAN/DOC/训练'/f'{tag}训练报告.md'
archive_rel=Path('archive')/tag;archive=report.parent/archive_rel
commands=json.loads((ROOT/'planned_commands.json').read_text())
common=s['normal_common_pixels'];rgb=s['rgb'];ck=json.loads((ROOT/'inspection/checkpoint_55000.json').read_text())
text=f'''# 1009-04 jumpingjacks_direction_light GT Lambertian Stage2 各项MAE与三类Normal评测报告

日期：2026-10-09（Asia/Tokyo），venus / lumimotion。归档时间：{stamp}。

**状态：COMPLETED；35k→55k训练、完整15测试帧评测、独立CPU复算、视频解码和首中尾人工检查完成，流程与产物验收PASS。** PASS不表示模型达到质量阈值。

## 来源与设置

Stage1来源1009-02原版SH/35k，99,827 Gaussian；本次Stage2使用当前仓库photometric_lambertian / lambertian_gt，累计55k、新增20,000步。source为1009-02/data_view，135训练帧/15测试帧，测试0010、0020至0150，time=frame/150，resolution2、400×400、黑背景。来源模型、相机世界坐标保持不变，不做尺度或姿态拟合。

- 模型：`{ROOT}/model_mlp`；checkpoint：`stage2_photometric/iteration_55000/checkpoint.pth`。
- 完整输出：`{ROOT}`。
- 原始GT：`/home/lihan/data/LH-data/danamic/jumpingjacks_direction_light`。
- [训练文档](../训练/{train.name})；Stage2代码提交a9e4d1061fec9986d82362f78b42c0c269e6146e，源码快照与工作区差异随证据保存。

正式流水线19:56:27启动，最终55000权重20:07:32保存；训练循环实际耗时{training_seconds:.3f}秒，新增20,000步。

几何、尺度、旋转、opacity、SH、法线及完整形变网络冻结，仍按真实time求动态形变，仅训练跨时间共享sigmoid albedo，LR0.001。0.8前景L1+0.2遮罩DSSIM+0.001初始材质先验，无GT材质/法线/深度训练监督。Adam实际新增20,000步，冻结指纹、方向表、来源哈希检查通过。

两盏逐帧重叠白色SUN各energy3，角大小0，GT方向直接使用原light_dir_world（表面指向光源），不用灯位中心近似；兼容表保留两灯及energy总和6。固定着色光强{s['intensity']:.12f}来自135训练RGB、SH初值及冻结法线标定，不用测试帧或GT材质拟合，不当作Blender能量的物理恢复。简化Lambertian没有投射阴影、镜面或间接光，albedo可能吸收这些误差。原版Stage1颜色形变不参与此材质着色。

## 各项MAE：完整15测试帧

| 指标 | MAE（有效像素加权） | 覆盖率 |
| --- | ---: | ---: |
'''
for m in s['metrics']:
 if m['mae'] is None:text+=f"| {m['metric']} | 不适用：Lambertian无粗糙度输出 | — |\n"
 else:
  suffix='°' if m['unit']=='degree' else (' m' if m['unit']=='m' else '')
  value=f"{m['mae']:.8f}{suffix}"
  if m['unit']=='m':value+=f"（{100*m['mae']:.4f} cm）"
  text+=f"| {m['metric']} | {value} | {m['coverage_pct']:.6f}% |\n"
text+='''
Albedo在线性RGB评分，GT alpha≥0.99、内缩2像素，全部内部前景，三个通道绝对误差均值；不做材质分区或尺度拟合，漏检预测记0，覆盖率按预测alpha≥0.9诊断。原16-bit sRGB PNG经既有PIL入口转8-bit，原GT不改写。

Normal为归一化世界坐标有向夹角，GT alpha≥0.5全前景且GT三角面命中；GS要求预测alpha≥0.5且法线有效，Depth Normal额外要求中心差分邻域；Mesh按自身第一射线交点有效，不另要求GS Alpha。Normal GT为原32-bit Blender着色Normal EXR，可能含材质法线贴图；不是纯几何法线真值。三类均使用同一参考，缺失计入覆盖率分母。

Depth为正camera Z与dir自己的Blender逐帧求值三角面射线GT比较，METRIC、meters_per_world_unit=1，无ICP或尺度/偏移拟合。主统计按有效像素汇总，逐帧等权结果另存。

Roughness明确NOT_APPLICABLE：本模型没有粗糙度参数/输出，不能将加载器默认粗糙度当作预测评分。
'''
text+=f'''\n## 三法线共同有效像素

共同有效像素{common['valid_pixels']} / {common['gt_pixels']}，覆盖率{100*common['coverage']:.6f}%。

| 方法 | 共同像素MAE |
| --- | ---: |
'''
for key,label in [('gs_render','GS Render'),('depth_normal','Depth'),('mesh','Mesh')]:text+=f"| {label} Normal | {common['angular_deg'][key]['mae_pixel_mean_deg']:.8f}° |\n"
text+=f'''\nGS缺失法线按180°计入的补充MAE为{s['gs_normal_missing_180_penalty_deg']:.8f}°。Mesh为15个独立时刻的模型预测TSDF，各135训练相机在目标time融合Expected Z；不使用测试相机位姿或GT几何。参数固定voxel0.02、sdf_trunc0.08、depth_trunc20、预测alpha0.5，与env一致，未按GT误差调参。Mesh分数受预测深度与固定提取参数共同影响；冻结几何的Stage2不能据这些分数宣称改善Stage1几何。

## 完整RGB评测

15帧各自相机/time/已知GT方向，独立render_stage2入口：PSNR {rgb['psnr']:.8f} dB、SSIM {rgb['ssim']:.8f}、LPIPS {rgb['lpips']:.8f}；前景PSNR {rgb['psnr_foreground']:.8f} dB、遮罩前景包围框SSIM {rgb['ssim_foreground_bbox']:.8f}、LPIPS {rgb['lpips_foreground_bbox']:.8f}。这些是全15帧指标，不能混同Stage1日志的少量相机子集。

RGB/Alpha/Normal/Albedo视频各150帧，转灯72帧，GoldenBay HDR150帧（2048样本），六视频完整解码；HDR与转灯为响应示例，无对应重光照GT。渲染目录model_mlp/renders_stage2/ours_55000。

## 实际目检与质量限制

{review['conclusion']}

- RGB：{review['rgb']}
- Alpha：{review['alpha']}
- Albedo / separation：{review['albedo_separation']}
- Normal与Depth：{review['normal_depth']}
- 重光照：{review['relight']}

首中尾代表图审核不等于全部150时刻人工逐帧查看；MAE及数值复算覆盖所有15测试帧。在共同有效像素上Mesh法线误差高于GS、低于Depth Normal，不宣称新增Mesh指标改善GS。

![三类法线首中尾]({archive_rel}/inspection/three_normals_first_middle_last.png)

- [RGB与GT首中尾]({archive_rel}/inspection/rgb_comparison.png)
- [四类首中尾]({archive_rel}/inspection/four_categories.png)
- [转灯与HDR]({archive_rel}/inspection/relight.png)
- [Albedo首中尾对照]({archive_rel}/inspection/albedo_first_middle_last.png)
- [Albedo真值对照0010]({archive_rel}/albedo/iteration_55000/0010_comparison.png)
- [法线/深度/误差0010]({archive_rel}/normal_depth/comparison_0010.png)

## 审核与归档证据

独立CPU从浮点数组复算Albedo全GT内部前景、三法线各自/共同有效像素和角度、Expected/Median Z与覆盖率；与全量及逐帧结果一致。GT从dir自己blend导出，源blend及模型哈希未改变；15帧相机与PLY/NPZ求交真值审计通过。检查实际checkpoint有限、Adam20,000步、仅albedo优化组、固定GT方向/光强、初末冻结指纹一致。

- [汇总JSON]({archive_rel}/metrics_summary.json)、[CSV]({archive_rel}/metrics_summary.csv)、[三法线逐帧]({archive_rel}/normal_depth/normal_metrics_per_frame.csv)。
- [独立复算]({archive_rel}/inspection/independent_audit.json)、[GT审核]({archive_rel}/inspection/gt_audit.json)、[最终checkpoint]({archive_rel}/inspection/checkpoint_55000.json)、[完整视频与产物审核]({archive_rel}/inspection/artifact_checks.json)、[人工目检]({archive_rel}/inspection/visual_review.json)。
- [文件SHA256清单]({archive_rel}/archive_manifest.json)：文档、指标、逐帧表、命令、训练/渲染/评测源码快照、日志与代表图已归档。完整训练权重、GT及预测Mesh、浮点图保留原输出路径，未放入小型文档附件。

## 完整训练、渲染与评测命令

工作目录`{REPO}`，CUDA_VISIBLE_DEVICES=0、OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=4，Python lumimotion；Blender5.2.2用于逐帧真值几何导出。

'''
for c in commands:text+=f"### {c['name']}\n\n```bash\n{shlex.join(c['argv'])}\n```\n\n日志：`{c['log']}`。\n\n"
report.write_text(text)
text_for_snapshot=text.replace(str(archive_rel)+'/',str(archive)+'/').replace('../训练/',str(REPO/'LIHAN/DOC/训练')+'/')
t=train.read_text().replace('状态：已授权，输入准备通过；训练与最终评测进行中，验收待完成。','状态：COMPLETED，训练、渲染、完整15帧MAE与最终审核完成，流程及产物PASS。',1)
t=t.replace('原7视频各150帧已解码，质量目检待回填。','原7视频各150帧已解码，首中尾四类及静动态分离目检已回填1009-02。',1)
t+=f'''\n## 最终结果与归档\n\n{stamp}完成最终审核。真实55000 checkpoint finite、仅albedo优化组、Adam20,000步、初末冻结指纹和光照表不变，来源35k PLY/形变哈希不变；固定光强{s['intensity']:.12f}。完整15测试帧MAE、独立CPU复算、GT审核与六视频完整解码通过，首中尾RGB/Alpha/Albedo-separation/Normal真实目检完成，流程与产物验收PASS，不代表达到模型质量阈值。\n\n'''
for m in s['metrics']:
 t+=f"- {m['metric']}："+("不适用，Lambertian没有粗糙度输出。\n" if m['mae'] is None else f"{m['mae']:.8f} {m['unit']}，覆盖率{m['coverage_pct']:.6f}%。\n")
t+=f"\nRGB PSNR {rgb['psnr']:.8f} dB，SSIM {rgb['ssim']:.8f}，LPIPS {rgb['lpips']:.8f}。\n\n质量结论：{review['conclusion']}\n\n完整[评测报告](../报告/{report.name})，命令/日志/代表图/视频、SHA256清单及训练文档副本随1009-04归档附件保存；原checkpoint、浮点数组和GT/预测Mesh保留。\n"
t=t.replace('../报告/',str(REPO/'LIHAN/DOC/报告')+'/')
train.write_text(t);(ROOT/'README.md').write_text(t+'\n---\n\n'+text.replace(str(archive_rel)+'/',str(archive)+'/'))
state=json.loads((ROOT/'state.json').read_text());state.update(status='COMPLETED',step='finished',updated_at=stamp,acceptance='PASS',acceptance_scope=s['acceptance_scope']);(ROOT/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');(ROOT/'PASS').write_text('Training, full 15-frame MAE, artifacts and representative visual review; no quality threshold\n')
changes=REPO/'LIHAN/DOC/修改/1009-04-dir多SUN兼容表与GT-Lambertian执行记录.md';cs=changes.read_text().replace('最终指标与人工目检待训练和完整评测结束后回填。',f'完整55k训练、全15帧MAE、独立复算、六视频解码及人工目检已完成，流程PASS。详情见[评测报告](../报告/{report.name})；Roughness明确不适用，固定几何仍有局部碎片与法线误差。');changes.write_text(cs)
files=[ROOT/n for n in ['metrics_summary.json','metrics_summary.csv','planned_commands.json','gt_directions.json','state.json','code_commit.txt','code_status.txt','code_diff.patch']]
for folder in ['commands','logs','inspection']:
 files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for folder in ['albedo/iteration_55000','normal_depth']:
 files.extend(p for p in (ROOT/folder).iterdir() if p.is_file() and p.suffix in {'.json','.csv','.png'})
records=[]
def copy(source,relative):
 dest=archive/relative;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and dest.read_bytes()!=source.read_bytes():raise RuntimeError('归档路径冲突: '+str(dest))
 shutil.copy2(source,dest);records.append(dict(path=str(relative),source=str(source),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
for p in sorted(set(files)):copy(p,p.relative_to(ROOT))
for p in [train,report,changes]:
 if p==report:
  dest=archive/'documents'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text_for_snapshot);records.append(dict(path=str(dest.relative_to(archive)),source=str(p),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),note='归档副本链接改为绝对路径，正文和数值与主报告一致'))
 else:copy(p,Path('documents')/p.name)
(archive/'archive_manifest.json').write_text(json.dumps(dict(archived_at=stamp,report=str(report),training_report=str(train),experiment=str(ROOT),model_path=s['model_path'],iteration=55000,files=records,not_embedded='完整权重、浮点数组、GT及预测Mesh保留在source输出目录'),ensure_ascii=False,indent=2)+'\n')
block=f'''## 1009 dir GT Lambertian Stage2：完成并归档

[训练文档](DOC/训练/{train.name})；[全15帧MAE报告](DOC/报告/{report.name})：Stage1/35k只读来源，99,827 Gaussian，冻结几何及形变，仅材质训练至55k；真实GT SUN方向，135训练RGB标定固定光强{s['intensity']:.8f}。Albedo {s['metrics'][0]['mae']:.8f}，GS/Depth/Mesh Normal {s['metrics'][1]['mae']:.4f}°/{s['metrics'][2]['mae']:.4f}°/{s['metrics'][3]['mae']:.4f}°；覆盖率、共同像素与深度完整联报。Roughness无输出，标为不适用。训练、GT审核、MAE独立复算、六视频解码及首中尾实际目检PASS，质量限制和证据SHA256清单已归档。[多灯适配记录](DOC/修改/{changes.name})。

'''
for target in [REPO,Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 idx=target/'LIHAN/README.md';content=idx.read_text();pattern=r'(?ms)^## 1009 dir GT Lambertian Stage2.*?(?=^## |\Z)';assert re.search(pattern,content);content=re.sub(pattern,lambda m:block,content,count=1);idx.write_text(content)
 if target!=REPO:
  for p in [train,report,changes]:dest=target/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
  for p in archive.rglob('*'):
   if p.is_file():dest=target/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for e in records:assert hashlib.sha256((archive/e['path']).read_bytes()).hexdigest()==e['sha256']
print(json.dumps(dict(report=str(report),training_report=str(train),archive=str(archive),files=len(records),bytes=sum(e['bytes'] for e in records)),ensure_ascii=False,indent=2))
