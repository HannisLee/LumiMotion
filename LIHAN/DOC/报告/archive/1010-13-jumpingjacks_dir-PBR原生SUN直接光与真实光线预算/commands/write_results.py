"""真实全量产物完成后保存训练/MAE文档；人工质量gate保持待审阅。"""
from pathlib import Path
import json,csv,shlex,hashlib
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent.parent;CWD=ROOT.parent.parent
G=json.loads((ROOT/'normal_depth/metrics.json').read_text());fg=G['summary']['foreground']
A=json.loads((ROOT/'albedo/iteration_55000/metrics.json').read_text());a=A['summaries']['raw']['foreground']
R=json.loads((ROOT/'roughness/metrics.json').read_text());r=R['summaries']['foreground']
RGB=json.loads((ROOT/'model_mlp/renders_stage2/ours_55000/eval/metrics.json').read_text())['metrics']
AUDIT=json.loads((ROOT/'inspection/independent_audit.json').read_text());ART=json.loads((ROOT/'inspection/artifact_checks.json').read_text())
entries=[dict(metric='Albedo MAE',mae=a['mae_linear'],unit='linear RGB',coverage_pct=a['coverage_pct']),dict(metric='Roughness MAE',mae=r['mae'],unit='r',coverage_pct=r['coverage_pct'])]
for name in ['gs_render','depth_normal','mesh']:
 m=fg['normal_metrics'][name];entries.append(dict(metric=m['label'],mae=m['mae_pixel_mean_deg'],unit='degree',coverage_pct=100*m['coverage']))
for name in ['expected','median']:
 m=fg['depth'][name];entries.append(dict(metric=name.capitalize()+' Z Depth MAE',mae=m['mae_pixel_mean'],unit='m',coverage_pct=100*m['coverage']))
result=dict(status='COMPLETE_MAE_MANUAL_REVIEW_PENDING',model_path=str(ROOT/'model_mlp'),iteration=55000,frames=list(range(10,151,10)),metrics=entries,rgb=RGB,normal_common_pixels=fg['common_normal_metrics'],gs_normal_missing_180_penalty_deg=fg['normal']['angular_with_missing_deg']['pixel_mean'],independent_audit=AUDIT,training_gt_inputs=['light_direction_only'],material_geometry_gt_used_for_training=False,light_GT_photometry_used=False,light_intensity=json.loads((ROOT/'model_mlp/pbr_directional/iteration_55000/lighting.json').read_text())['effective_intensity_scale'],source_plane_diagnostic_only=False)
(ROOT/'metrics_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'metrics_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['metric','mae','unit','coverage_pct']);w.writeheader();w.writerows(entries)
table='| 指标 | MAE | 单位 | 覆盖率 |\n| --- | ---: | --- | ---: |\n'+''.join('| {} | {:.8f} | {} | {:.6f}% |\n'.format(e['metric'],e['mae'],e['unit'],e['coverage_pct']) for e in entries)
commands=json.loads((ROOT/'planned_commands.json').read_text());train=json.loads((ROOT/'train_argv.json').read_text())
allcmd='\n## 完整命令\n\n### train\n\n```bash\n'+shlex.join(train)+'\n```\n\n日志：'+str(ROOT/'logs/train.log')+'\n'
for c in commands:allcmd+='\n### '+c['name']+'\n\n```bash\n'+shlex.join(c['argv'])+'\n```\n\n日志：'+c['log']+'\n'
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
body='''# 1010-13 jumpingjacks dir 原生PBR直接SUN与真实光线预算

更新时间：'''+now+'''
状态：完整20000步训练、渲染、全部15测试帧MAE与独立复算完成；人工质量审閱待完成。不能据此标记图像质量或7°目标PASS。

## 训练与输入边界

来源output/1009-02-jumpingjacks_direction_light-原版LumiMotion-Stage1/model_mlp，35k；本次从35000到55000、20000步，仅Stage2。135训练帧/15测试帧、resolution2（400×400）、黑背景、原相机/time。训练用原RGB/输入RGBA、Stage1预测和GT方向；GT albedo/normal/roughness/depth/mesh/blend、GT光颜色/强度均不输入训练。

新source_path是本实验data_view，只从原image RGBA及camera转换，原生开训前逐图核验SHA与像素，rgb_alpha_provenance.json保存150帧PASS；不读取GT材质/几何。旧转换曾读取albedo alpha，B01因该来源声明错误主动中止、无新checkpoint/MAE。150帧原RGB与albedo的Alpha及新旧转换RGBA逐像素相同；固定35k来源继承旧转换但没有原RGB之外的轮廮信息，已披露且未重跑Stage1。

GT灯方向按帧去重、每帧白色单位光，正光强初值π，仅训练RGB拟合一个全局正标量，marker保存实际值；原GT能量不读取为训练参数。原生2000步diffuse warmup、之后diffuse+GGX；关闭训练与重光照的Stage1 SH间接光近似，原生SH辅助RGB损失保留，每4步优化。优化原生albedo_dc、albedo_dc_stage1、albedo_rest、roughness及真实xyz/rotation和全局正光强；opacity、尺度和二值静动态参数冻结。几何2000步warmup、1000步ramp，Canonical PLY位移最大.08世界单位/旋转最短弧最大60°，预测GS-depth一致性、来源静态RANSAC/SVD平面、输入Alpha轮廓和源参数弱锚定均不使用GT法线/几何；原MLP几何固定。实际Adam没有完整状态保存，原生分支不宣称支持无损resume。预训练35k初始化不计成本次训练成果。

本次Plane Native Direct Preset继承B06全部材质/几何参数，包括既有来源平面几何投影。只用未改的35k PLY/MLP、全部135训练相机与time筛选支持：feature<.5、Alpha>.05、canonical与所有训练时刻距source RANSAC/SVD平面≤.2m、全时间形变坐标box-span范数≤.01m、canonical及全部时刻法线轴与plane绝对cos≥.5。支持集合固定，全部99827点保留；实际渲染world XYZ投影到plane、actual quaternion的Z轴旋到source polarity×plane且保留切向旋转与raw norm。训练RGB、PBR/BVH/GGX、周期测试、离线GS Normal/Depth/Mesh/HDR/方向灯共享同一实际几何。不是输出后改法线图、不是改变评分或删除难像素。支持资产来源SHA、全训练相机SHA和精确重算及checkpoint marker绑定；GT材质/几何从未参与支持。

来源MLP输入仍固定源XYZ、feature/time/camera不变，源32tensor保持。Canonical PLY仍限制.08m/60°，但有效渲染位置还经过plane几何模型投影，其总世界位移不能宣称受canonical .08m限制。该额外平面模型假设可能误平坦化来源误分类或真实非平面细节，必须以原始全15帧七项MAE及目检验收。PBR DC初始化和更新后[.031,.969]内部投影保留；本轮继承B06原生CUDA及Plane模型，只改关闭SH间接近似和实际SUN像素预算；材质/光强/loss/lr/几何参数保持。不是GT normal/plane/mesh约束。

## 本次Direct SUN与真实光线预算

新增pbr_gt_directional_native_direct及pbr_gt_directional_plane_native_direct，本轮后者。wo_indirect/wo_indirect_relight=True，只关闭PBR来源SH间接近似，native SH辅助RGB损失不改。pbr_directional_actual_ray_budget=True，SUN每像素L条BVH光线，启用间接光的显式覆盖则额外diffuse_sample_num条。单灯直接预算262144允许全部alpha>.9可用photo像素；该规则、全99827点、所有训练/测试帧与原GT评分域不改。旧Preset默认仍历史diffuse+light分母，显式HDR保留原预算，拒绝非法模式/非正整数/不足一个完整像素预算。

194 CPU discovery实际exit0；20步真实Trainer35000→35020实际exit0，保存Direct flags和native marker。三TRAIN帧实际BVH分别18843/15904/19137条单SUN，等于全部可用photo像素；真实3SUN预算512选170像素/510条，with-indirect成本257选1020像素/262140条，旧direct默认仍1024像素。材质/roughness/XYZ/q梯度非零finite；独立诊断0额外optimizer且参数SHA保持。首次诊断复用projection图导致重复backward失败，源码/日志保存；修正仅诊断每variant重建图，实际retry exit0。

GeometryRun与PBR alpha/worldnormal/ExpectedZ全buffer maxerror0、Mesh目标time一致，albedo/roughness材质helper一致。150×四类与4视频、三帧HDR16/替换SUN仅smoke功能；实际目检质量仍FAILED，不作为正式MAE或7°。正式完整150 HDR仍2048样本。

继承B06显式仓库LumiMotion native kernel，逐Gaussian按-dot(n_view,p_view)定向，forward/backward一致。pbr_rasterizer_backend=lumimotion，来源资产保存CUDA/Python/GLM逐文件SHA、综合源SHA和实际.so SHA，checkpoint marker与加载核验源，不静默回退installed。训练、全部RGB/Alpha/Normal、HDR/SUN、GeometryRun与逐time Mesh、材质评测共享入口。没有GT训练材质/法线/几何/能量；方向表白色单位×训练RGB拟合正scale，重光照metadata明确light-table而非GT photometry。

## 各项MAE：全部15测试帧

'''+table+'''
Albedo按GT alpha≥.99内缩2像素的全内部前景在线性RGB评分，raw不校准；缺失预测记0。Roughness r采用dir自己的blend、源BLENDER_EEVEE、原Principled Alpha透明混合输出；透明部分定义为透射率×Alpha加权可见r，不声称唯一第一命中粗糙度；按GT alpha≥.99内缩2像素、预测Alpha≥.9的有效像素评分并报告覆盖率。

GS/Depth/Mesh Normal统一对原始32-bit世界着色Normal有向角度评分，GT可能含材质normal mapping；GT alpha≥.5全前景，不按物体/材质过滤。Expected/Median Z为正camera Z、米，无尺度/偏移/ICP拟合。GT深度沿用1009-04已审计dir逐帧blend三角面射线交点，GT与源模型SHA可复核；GT只有离线评测读取。

Mesh每个测试目标time分别由所有135训练位姿下模型预测深度融合，固定voxel.02/sdf_trunc.08/depth_trunc20/alpha.5；不跨动作融合、不用测试相机重建、不用GT几何。三法线共同像素结果另存metrics_summary.json。

## RGB及渲染

'''+json.dumps(RGB,ensure_ascii=False)+'''

渲染路径：'''+str(ROOT/'model_mlp/renders_stage2/ours_55000')+'''

RGB/Alpha/Normal/Albedo各150帧，转灯72帧、HDR150帧（2048采样）；6个视频均完整解码。转灯Alpha应保持不变，HDR/转灯无重光照GT，只有功能和目检，不给GT精度结论。本轮关闭来源SH间接近似，HDR/SUN仍使用原生GGX/BVH可见性；无重光照GT准确率声明。

## 四类目检（待人工回填）

- RGB与背景：待审阅55k首中尾；本次40k预览若存在另行报告；对照B00最终PSNR18.38456、人物斑驳与平台伪几何，未通过质量gate，不能当本次结果。
- Alpha：待审阅边缘、缺失与残影；数值覆盖率见上表。
- Albedo与静动态分离：待审閱光照/棋盘格残留；本次冻结静动态参数。
- Normal/Depth：待审閱GS斑驳、平台/脚周伪几何、Depth/Mesh缺面；真实几何参与更新，是否改善以全部15帧三类法线和覆盖率为准。

代表图：inspection/four_categories.png、rgb_comparison.png、relight.png；15帧误差图和浮点数组保留normal_depth、albedo、roughness；权重与日志保留，不删除失败结果。

## 审计及结论

194项CPU单元测试与真实源权重PBR/BVH/GGX GPU梯度检查通过。GeometryRun已共享FrozenScene Anchor/Plane真实几何入口；本轮35020 native smoke首中尾GPU alpha/normal/Expected Z与原生PBR buffer逐像素max error0，Mesh timeoverride一致，材质helper一致；完整训练与离线使用同一native来源。完整launch源码/manifest和CUDA/GLM快照保存。checkpoint的有限性、canonical xyz/rotation实际更新并在信赖域内、opacity/scale/feature冻结、roughness真实变化、灯表快照SHA、Stage1来源未改、全量图片/视频与背景统计见inspection/artifact_checks.json。三法线、共同像素、depth和albedo独立CPU数组复算见inspection/independent_audit.json，roughness独立复算日志logs/verify_roughness.log。

执行/产物：PASS（独立复算及数值/文件完整性）。人工图像质量：PENDING。法线<7°目标：'''+('ACHIEVED_METRIC_ONLY_PENDING_AUDIT' if fg['normal_metrics']['gs_render']['mae_pixel_mean_deg']<7 else 'FAILED')+'''

未依据本报告自动提交或push；先完成前后景人工审核，必要时修复重训，再进入目标分支。
'''+allcmd
# 最终文档同时保留实际时刻、全部共同像素结果和中间只读诊断。
common=fg['common_normal_metrics']
common_table='| 法线来源 | 共同像素MAE（度） |\n| --- | ---: |\n'+''.join('| {} | {:.8f} |\n'.format(k,v['mae_pixel_mean_deg']) for k,v in common['angular_deg'].items())
common_text='共同有效像素：{}/{}，覆盖率{:.6f}%；GS缺失预测记180°后的惩罚均角{:.8f}°。\n\n'.format(common['valid_pixels'],common['gt_pixels'],100*common['coverage'],fg['normal']['angular_with_missing_deg']['pixel_mean'])+common_table
body=body.replace('三法线共同像素结果另存metrics_summary.json。',common_text+'完整JSON另存metrics_summary.json。')
runtime=json.loads((ROOT/'training_runtime.json').read_text())
body+='\n## 实际训练时间与过程证据\n\n'+json.dumps(runtime,ensure_ascii=False,indent=2)+'\n\n以下记录为对应时刻的中间状态，不替代上方最终55k全量MAE与后续人工质量验收。\n'+(ROOT/'README_training_progress.md').read_text()
(ROOT/'README.md').write_text(body)
paths=[CWD/'LIHAN/DOC/训练/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算训练报告.md',CWD/'LIHAN/DOC/报告/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算各项MAE评测报告.md']
for p in paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
# 自包含小型归档：保留完整浮点/权重在原输出，以SHA和路径引用。
import shutil
archive=CWD/'LIHAN/DOC/报告/archive/1010-13-jumpingjacks_dir-PBR原生SUN直接光与真实光线预算';archive.mkdir(parents=True,exist_ok=False)
for p in [ROOT/'README.md',ROOT/'metrics_summary.json',ROOT/'metrics_summary.csv',ROOT/'code_manifest.json',ROOT/'code_manifest_train_launch.json',ROOT/'planned_commands.json',ROOT/'model_mlp/rgb_alpha_provenance.json',ROOT/'model_mlp/pbr_albedo_initialization.json',ROOT/'model_mlp/pbr_deformation_anchor.json',ROOT/'model_mlp/pbr_deformation_anchor.npz',ROOT/'model_mlp/pbr_plane_projection.json',ROOT/'model_mlp/pbr_plane_projection.npz',ROOT/'model_mlp/pbr_rasterizer.json',ROOT/'training_runtime.json',ROOT/'README_training_progress.md',ROOT/'runtime_history.jsonl']:
 shutil.copy2(p,archive/p.name)
for folder in ['commands','logs','inspection']:
 shutil.copytree(ROOT/folder,archive/folder)
for rel in ['normal_depth/metrics.json','albedo/iteration_55000/metrics.json','roughness/metrics.json']:
 p=ROOT/rel;dest=archive/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
manifest={str(p.relative_to(archive)):hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.rglob('*') if p.is_file()}
(archive/'archive_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 for p in paths:
  dest=checkout/p.relative_to(CWD);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 dest=checkout/archive.relative_to(CWD)
 if not dest.exists():shutil.copytree(archive,dest)
goal=CWD/'LIHAN/DOC/GOAL/1010-goal-opt-pbr-jacks.md'
goal.write_text(goal.read_text()+'\n### B07全量结果（'+now+'，人工审核待完成）\n\n'+table+'\nRGB：'+json.dumps(RGB,ensure_ascii=False)+'\n\n完整训练、15帧各项MAE与独立复算已完成；人工前后景/四类审閱待完成，未commit/push，7°目标未自动宣称通过。完整记录见训练/报告1010-13文档。\n')
for checkout in [Path('/home/lihan/reproduce/LumiMotion-perlight'),Path('/home/lihan/reproduce/LumiMotion')]:
 dest=checkout/goal.relative_to(CWD);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(goal.read_bytes())
print(json.dumps(result,ensure_ascii=False,indent=2))
