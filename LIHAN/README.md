# LumiMotion 文档库与 Agent 工作索引

更新日期：2026-10-07
适用范围：本仓库中的 LumiMotion 复现实验、数据准备、Stage 1/2 训练、离线评估与后续代码维护。

本文件是进入项目后的**第一阅读入口**。它不重复各实验的原始命令或指标；需要执行、判断或修改时，请沿下方链接回到对应的一手文档、脚本和实验产物。当前文档库的唯一入口目录为 `LIHAN/DOC/`。


## 1008 通用前景 MAE 使用指导

[简明指导](DOC/指导/1008-01-通用前景MAE评测使用指导.md)：Albedo、Normal、Roughness 均使用全前景，不按材质或物体类别拆分。更换数据集只需替换模型/GT路径；roughness直接读取逐帧浮点真值，不绑定场景、帧数或分辨率。[修改与核验](DOC/修改/1008-42-通用前景MAE与逐帧粗糙度真值评测.md)，33项测试、三项各15帧实际评测、通用纹理场景核验通过。

## 1008 CV4 Roughness MAE（历史材质分区协议）

[实现与真实权重验收](DOC/修改/1008-41-CV4静态PBR粗糙度MAE评测与真值核验.md)：原版1006-01/55k，15测试帧，roughness条件MAE **0.18104681**、四材质bMAE **0.24102391**、覆盖率 **99.991658%**。Blender四材质常量核验、黑白背景渲染一致性、21项测试及独立CPU复算通过；木框误差最高。未修改原PBR训练/渲染。[完整产物](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1008-01-CV4-PBR-Roughness-MAE/README.md)。

## 1007 动态 Stage1 质量差异审计

[1006-11 与 1006-03 审计报告](DOC/报告/1007-01-1006-11与1006-03动态Stage1质量差异审计报告.md)：核对相同配置和随机初始化，重新渲染两组各15测试帧，区分canonical PLY与时间形变后的显示。only_cloth沿采集轨迹的RGB拟合较好，固定视角布料几何仍有碎边；报告量化相机采样、薄布可见性、布料GT误差及高斯分配差异，并列出待验证的单因素对照。只读审计，未启动新训练。

## 1007 三类法线 MAE 与训练视角图

[实现与核验](DOC/修改/1007-02-三类NormalMAE命名与模型Mesh评测.md)：历史Normal MAE明确为GS Render Normal MAE；并列Depth Normal MAE，新增模型重建Mesh Normal MAE。1002-02/55k仅训练预测提取静态TSDF Mesh，15测试帧三项全前景8.0163°/12.3835°/11.4455°，覆盖率联报；旧指标不变，15项测试通过。[新流程与训练相机视角](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/normal_metrics_with_train_views.png)，三项共用GT Normal EXR。未修改训练损失或来源模型。

## 1006 jumpingjacks 默认随机方块训练：全部完成

用户已选择`jumpingjacks150_v5_spec32`，使用默认10万随机方块初始化，原版SH Stage1 35k → IR Stage2累计55k。训练光照Chapel Day，测试重光照Golden Bay。已核实原始目录已有PLY及原版历史模型input.ply本身就是默认seed=0随机方块；本次独立data_view重新生成同样输入并保存证据。用户已回复“OK开始吧”，已于2026-10-06 18:03（Asia/Tokyo）启动GPU0正式后台流水线；初始化后自动衔接Stage1渲染、Stage2及材质/HDR/NVS。Stage1 35k、Stage2 55k及完整渲染/评测全部完成，流程和产物PASS。15帧原版动态NVS：PSNR 27.3426、SSIM 0.948958、LPIPS 0.0194175。10视频各150帧完整解码，代表帧RGB和Alpha完整；人体normal斑驳、albedo明暗残留及局部重光照误差是主要质量限制。

[1006-11 jumpingjacks默认两阶段训练计划](DOC/训练/1006-11-jumpingjacks150_v5_spec32-原版随机方块初始化默认两阶段训练计划.md)。此前0930训练中止记录保持不变。

[1006-13双阶段MAE评测报告](DOC/报告/1006-13-jumpingjacks-双阶段Albedo与NormalDepth-MAE评测报告.md)：全15测试帧albedo MAE（全前景线性RGB）Stage1 SH代理0.08579548 → Stage2 0.06169353；normal GT缺失，150张depth全白，normal/depth真值MAE不可评测。完整数值、审计和GT对照均已保存。

## 1006 随机方块来源的GT Lambertian Stage2：全部完成

四组固定共用光强6.030241533058244，独立200步冒烟后正式35k→55k；4/4训练、全量MAE评测、6视频逐组全解码及代表图真实目检完成，流程PASS。

[四组报告汇总](DOC/报告/1006-10-V4四组固定共用光强GT-Lambertian-Stage2评测汇总.md)；[完整训练结果](DOC/训练/1006-10-V4随机方块Stage1来源的Stage2-GT灯位Lambertian训练核对.md)；[全量MAE](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/all_mae.csv)。材质、RGB和冻结几何的限制均逐组报告。

静态四组1006-01/02/06/07的独立比较见[静态四组Stage2汇总报告](DOC/报告/1006-12-静态四组Stage2-IR与GT-Lambertian汇总报告.md)，含各材质/尺度诊断、RGB、冻结几何、对应Stage1参考及代表图。

## 1006 四版本随机方块初始化重训：全部完成

**最终状态：4/4 COMPLETED，流程及产物验收PASS；训练只有1006-01～04。** 原版SH Stage1 35k、IR Stage2累计55k，两阶段共8组结果、120行逐帧指标已完成。

评测完成：2026-10-06 02:44:28（Asia/Tokyo）；最终核验及文档回填：2026-10-06T11:16:08+09:00。

[完整双阶段报告](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/报告/1006-01-四组Stage1与Stage2真值评测报告.md)；[阶段CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/metrics.csv)；[最终核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/final_audit.json)。

四组的Stage2全图及前景包围框SSIM均下降，depth和normal MAE均增加；共同有效像素比较也显示几何误差上升。只有1006-04的原始albedo MAE下降（0.138899→0.122299，约11.95%），其bMAE也下降；01、02、03的原始albedo MAE增加。Stage1为含光照的SH基色代理，以上数值变化不能解释为Stage1物理材质恢复更准确。

[四组执行记录（辅助文档）](DOC/训练/1006-05-V4四版本原版随机方块初始化重训核对.md)、[初始化审计](DOC/修改/1006-06-随机方块重训准备与初始化审计.md)、[调度及最终回填](DOC/修改/1006-07-四版本随机方块正式调度与结果回填.md)。默认随机方块100000点，原版35k/55k；冒烟按用户要求跳过。四组正式流程和双阶段评测、代表图目检均完成；质量限制见报告，PASS仅表示流程与产物验收。

## 全分支共享文档与历史归档

`LIHAN/` 和根目录 `LIHAN.md` 为全分支共享资料。训练、参数与实验记录仍以文中标注的来源分支和提交为准；当前分支未必具备其他分支的模型功能。服务器环境应先运行 `hostname`，再按根目录规则选择，文中的历史环境和本地 `output/` 链接仅供追溯。

- [远端分支归档索引与同步约定](DOC/报告/archive/1002-远端分支文档/1002-03-远端分支文档归档与同步说明.md)：原始论文复现、LH 静态两阶段结果、Stage1 V1/V2/V3/V5、历史环境及配图。

## 2026-10-05 Stage2 GT灯位 Lambertian 四组执行

使用1003四组原版SH Stage1 **iter35000**作为各自只读来源，新增20,000步Stage2 Lambertian至55000。每组独立输出、独立训练帧光强标定，最多两组并行。GT灯位沿用现有场景中心方向近似；详细灯光语义、来源哈希、命令及验收在各计划中。**四组冒烟、正式训练、完整渲染、评测与真实目检均已完成，文档已回填；正式流水线耗时51分23秒，实际最大并行2组。流程验收PASS，材质与动态细节缺陷见[执行汇总](../output/1005-07-V4四版本Stage2-GT灯位执行汇总/README.md)。**

- [1005-07 四版本Stage2 GT灯位训练结果](DOC/训练/1005-07-V4四版本Stage2-GT灯位Lambertian训练结果.md)：4/4完成，独立光强标定、RGB/前景与GT albedo评分、同协议原版对照和逐组质量限制。raw albedo均衡误差下降约10%–33%，RGB拟合指标低于原版。
- [1005-07 调度与验收回填](DOC/修改/1005-07-V4四版本Stage2-GT灯位执行与验收回填.md)：两槽后台流水线、冻结/梯度核查、完整产物与真实目检。

- [1005-03 静态V4 Lambertian训练计划](DOC/训练/1005-03-only_clothV4_lambertian-Stage2-GT灯位Lambertian训练计划.md)：来源1003-01，独立输出`output/1005-03-only_clothV4_lambertian-Stage2-GT灯位Lambertian/`。
- [1005-04 动态V4原始材质训练计划](DOC/训练/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian训练计划.md)：来源1003-02，独立输出`output/1005-04-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/`。
- [1005-05 动态V4 Lambertian训练计划](DOC/训练/1005-05-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian训练计划.md)：来源1003-03，独立输出`output/1005-05-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/`。
- [1005-06 静态V4原始材质训练计划](DOC/训练/1005-06-only_clothV4-Stage2-GT灯位Lambertian训练计划.md)：来源1003-04，独立输出`output/1005-06-only_clothV4-Stage2-GT灯位Lambertian/`。

## 2026-10-02 OnlyCloth V4 实验入口

- [1005-01 V4四版本材质与动态几何真值评测报告](DOC/报告/1005-01-V4四版本Albedo与动态法线深度真值评测报告.md)：1003-01/02/03/04最终55k、每组15测试帧；统一raw与训练帧scalar材质尺度诊断，动态从Blender重新导出逐帧PLY，联报全前景/布料法线和深度，含RGB/NVS与前景Albedo SSIM、PLY法线对照与完整验证。
- [1005-01 原版配置与逐帧PLY评测支持](DOC/修改/1005-01-原版材质配置与逐帧BlenderPLY评测支持.md)：cfg_args恢复、PLY实际求交与动态frame/time保护。

- [1003-02 V4 法线与深度真值评测](DOC/修改/1003-02-onlyclothV4法线与深度真值评测.md)：直接比较世界法线 EXR 和原场景三角面 Z 深度，已评测1002-02的35k/40k/50k/55k及GT/learned Stage2；含仅布料指标、覆盖率、阈值敏感性与完整运行入口。
- [1006-13 几何评测流程图与训练相机视角](DOC/修改/1006-13-onlyclothV4几何评测流程图与训练视角.md)：圆角流程图、105个训练相机分布、真实TRAIN 0001/0041/0081及1002-02原版55k法线/深度GT对照，提供PNG/SVG/PDF与来源记录。
- [1003-01 V4 四版本原版训练核对](DOC/训练/1003-01-V4四版本原版LumiMotion训练核对.md)：CV4 静态原始材质、CV4L 静态 Lambertian、CV4D 动态原始材质、CV4DL 动态 Lambertian；四组原版两阶段从头训练，输出/README/命令全部位于当前项目，**四组冒烟与正式流程验收全部完成，最多两组并行，正式35k/55k与最终回填4/4 PASS**。
- [1003-02 四版本训练结果](DOC/训练/1003-02-V4四版本原版LumiMotion训练结果.md)：四组最终指标、输出、目检与质量限制已回填；正式流水线5小时34分51秒，40个视频完整解码，PASS表示流程与产物验收。
- [1003-01 四版本调度与验收回填](DOC/修改/1003-01-V4四版本并行调度与验收回填.md)：后台双任务队列、产物检查、Stage1提前目检、回填状态与原版进度条解释。
- [当前数据集总览](DOC/指导/当前数据集总览.md)：V4 四版本的绝对训练路径、静动态属性、15 万点 canonical 初始化与 Alpha/光照语义。

本轮用户指定输出与活动文档仅保存在当前项目，执行代码使用原版 LumiMotion main；本轮不复制文档到其他项目。

- [Stage2 Lambertian 使用指南](DOC/指导/1002-01-Stage2-Lambertian训练渲染与续训使用说明.md)：实现分支 `1001-stage2-perlight`，原始 SH Stage1 + 独立 Stage2 双方向光模式。
- [输出目录说明](DOC/指导/1002-02-输出目录与Stage对应说明.md)：以 `1002-03-onlyclothV4-stage2_GT` 为例，逐块说明命令、模型、渲染与验收产物。
- [Stage2 三管线对照](DOC/训练/1002-02-onlyclothV4-Stage2三管线对照.md)：同一SH来源的原始IR、GT与learned训练、渲染和验收。
- [Stage2 验收与限制](DOC/报告/1002-30-Stage2三管线对照验收与限制.md)：同协议整图/前景指标、固定几何和恢复证据，以及材质与光照可辨识性限制。
- [Stage2 实现与 preset](DOC/修改/1002-28-Stage2屏幕空间Lambertian与Preset适配.md) / [HDR与来源保护](DOC/修改/1002-29-Stage2-HDR轴向回归与来源写入保护.md) / [旧渲染入口识别](DOC/修改/1002-31-Stage2续训配置与旧渲染入口识别.md) / [原始HDR配置读取](DOC/修改/1002-32-原始HDR渲染配置合并修复.md)：算法边界、兼容入口、完整恢复和验证证据。

- [1001-01 原始项目训练记录归档](DOC/训练/1001-01-onlyclothV4-FOVxy修复后原始管线重训.md)：在原始 `/home/lihan/reproduce/LumiMotion` 的 main 分支训练，完整 [产物副本](../output/1001-01-onlyclothV4-FOVxy修复重训/README.md) 已校验归档，原 NVS PSNR 33.3726 dB。
- [1002-01 perlight 原始管线复现计划](DOC/训练/1002-01-onlyclothV4原始实验归档与perlight原始管线复现.md)：当前目录的原始 SH 两阶段复现；[正式实验](../output/1002-01-onlyclothV4-perlight原始管线复现/README.md) 与 [冒烟实验](../output/smoke_test/1002-01-onlyclothV4-perlight原始管线冒烟/README.md) 分开记录。
- [1002-02 归档与执行工具记录](DOC/修改/1002-02-onlyclothV4归档与原始复现执行记录.md)：来源标注、venus 环境映射、执行与预检查。

## 先读什么

| 任务 | 首先阅读 | 然后查看 |
| --- | --- | --- |
| 了解仓库规则、命名、环境与产物保留要求 | [仓库协作约定](../LIHAN.md) | 本文“Agent 工作流程” |
| 首次理解项目代码与标准训练阶段 | [项目原始说明](../readme.md) | [Blender 数据训练指导](DOC/指导/blender数据训练指导.md) |
| jumpingjacks 论文默认 Stage 1 训练与评测 | [0930-01 训练计划与执行记录](DOC/训练/0930-01-jumpingjacks官方原生管线训练.md) | [实验产物、启动器与日志](../output/0930-01-jumpingjacks150_v5_spec32-official_pipeline/README.md) |
| 新建/检查 Blender 数据集 | [数据集转换指导](DOC/指导/blender数据集转换指导.md) | [训练集加载要求](DOC/指导/训练集加载要求.md) |
| 延续当前真实方向光实验 | [0829 冒烟训练记录](DOC/训练/0829-01-CV3-真实方向光-LambertianStage1冒烟训练.md) | 该实验目录的 `README.md`、`run.sh`、日志和指标 |
| 改 Stage 1 损失、预设或法线评估 | [0828 损失/评估完善](DOC/修改/0828-01-Stage1损失预设与法线评估完善.md) | [0826 loss 对象化重构](DOC/修改/0826-01-loss对象与preset组合式重构.md)、相关单测 |
| 判断 SH、Lambertian 或逐 GS 光方向的取舍 | [SH 优于 GT 光的审计](DOC/报告/08.24-CV3s-SH优于GT光Lambertian原因审计报告.md) | [坐标终审](DOC/报告/08.24-CV3s逐GS_GT位置光方向渲染坐标终审报告.md)、对应实验记录 |

## Agent 工作流程

1. **先确认现场。** 在仓库根目录运行 `hostname` 和 `git status --short`；不得覆盖、删除或回滚自己未产生的改动。当前文档库位于 `LIHAN/DOC/`，不要假设历史路径仍然有效。
2. **选择正确环境。** 服务器名与 Conda 环境一一对应：`mahadevi` → `lumimotion-mahadevi`、`minakshi` → `lumimotion-minakshi`、`parvati` → `lumimotion-parvati`、`ushas` → `lumimotion-ushas`、`garuda` → `lumimotion-garuda`。`venus` → `lumimotion`。本次同步在 `venus` 完成；执行时以实际 `hostname` 为准。`venus` 命令使用 `conda run --no-capture-output -n lumimotion ...` 或对应环境的绝对 Python 路径。
3. **按任务读一手资料。** 训练前至少阅读“指导”中的训练与加载要求、目标训练记录，以及关联的“修改”文档；代码改动前至少阅读最新修改记录、目标脚本和对应单测。
4. **让产物成为证据。** 实验结论以 `output/<实验名>/README.md`、完整命令、日志、checkpoint、渲染和 JSON/CSV 指标为准；文档中的摘要不能替代产物核验。冒烟结果只能写在 `output/smoke_test/`，不能作为 35k 正式训练的验收结论。
5. **完成后补齐记录。** 代码修改写入 `LIHAN/DOC/修改/`；新训练或重新训练写入 `LIHAN/DOC/训练/`。除本入口及实验目录的 `README.md` 外，新增文档使用 `日期-当日序号-修改内容.md` 命名，并以中文记录。

训练验收必须保留完整渲染命令、source/model/iteration、渲染目录、日志、定量指标、代表图像/视频、RGB/alpha/albedo-separation/normal 四类目检和最终 `PASS`/`FAILED`。失败实验的 checkpoint、渲染、统计和日志必须保留，不能覆盖或删除。

## 文档库导航

### 1. 指导：执行前的规范与数据前置条件

| 文档 | 用途 |
| --- | --- |
| [Blender 数据训练指导](DOC/指导/blender数据训练指导.md) | CV3/CV3L 数据别名、Stage 1 调度、训练和验收的主操作规范；新训练优先读。 |
| [训练集加载要求](DOC/指导/训练集加载要求.md) | 开训前检查 `transforms`、alpha、帧号、相机与目录结构，避免静默数据错误。 |
| [Blender 数据集转换指导](DOC/指导/blender数据集转换指导.md) | 原始 Blender 导出到可训练数据集的转换命令与相机/alpha/PLY 校验。 |
| [Blender 训练错误报告](DOC/指导/blender训练错误报告.md) | only_cloth 早期失败的根因库；遇到相机、尺度、alpha 或固定单视角问题时查阅。 |

### 2. 训练：实验事实、命令和验收结论

#### 当前优先级

- [0829-01 CV3 真实方向光 Lambertian Stage 1 冒烟](DOC/训练/0829-01-CV3-真实方向光-LambertianStage1冒烟训练.md)：数据与管线通过；正式 35k 尚未启动，且短训 independent normal 不通过。后续工作必须先确认真实 SUN 光的方向/强度标定与 staged-training 调度，不能直接将该冒烟视为正式通过。
- [0823-01 CV3s 三法线一致性](DOC/训练/0823-01-CV3s静态布料从iter1起GT光照Lambertian三法线一致性训练方案与执行记录.md)：迭代 1 起 GT 方向光 Lambertian 的已通过基线。
- [0824-01 CV3s 二法线消融](DOC/训练/0824-01-CV3s静态布料从iter1起GT光照Lambertian二法线一致性消融训练方案与执行记录.md)：已通过；是去掉 GS 几何自一致项、保留 live/MV 项的对照。
- [0824-04 CV3s 逐 GS GT 位置方向三法线](DOC/训练/0824-04-CV3s逐GS_GT位置光方向三法线消融实验.md)：已通过的方向 oracle 消融；法线改善但 RGB 不优于 0823 基线，不应直接替代默认 RGB 重建方案。

#### 已完成/历史对照

| 文档 | 结论或查阅目的 |
| --- | --- |
| [0824-02 gs_wrapping 论文损失对照](DOC/训练/0824-02-CV3s静态布料从iter1起GT光照Lambertian_gs_wrapping论文损失对照训练方案与执行记录.md) | 正式实验 `FAILED`；保留为论文权重映射与失败证据。 |
| [0824-03 SH 默认 2DGS normal 基线](DOC/训练/0824-03-CV3s_SH默认2DGSnormal基线训练与对照方案.md) | 实验 A `PASS`；Lambertian 对照 B 未启动，不能把它当成成对结论。 |
| [0824-05 逐 GS 方向 + 默认损失](DOC/训练/0824-05-CV3s逐GS_GT位置光方向Lambert默认损失训练方案与执行记录.md) | `FAILED`：RGB 可通过而 independent normal 明显退化，是单灯 RGB 不足以约束法线的直接证据。 |
| [0809–0818 训练与可视化记录](DOC/训练) | CV3/CV3L 的历史训练、标定、阶段冻结与 PPT 素材；需要追溯旧路径、旧指标或失败原因时按日期阅读。标有“归档”的实验不可作为当前默认方案。 |

### 3. 修改：代码现状与兼容性边界

修改文档按时间顺序阅读；最新状态优先于旧文档，但旧文档用于解释参数和设计来源。

| 顺序 | 文档 | 需要知道的内容 |
| --- | --- | --- |
| 0823-01 | [loss 集中化与预设传参](DOC/修改/0823-01-loss集中化重构与预设传参.md) | `scripts/loss.py`、`--loss_preset` 的引入和历史兼容目标。 |
| 0824-01 / 03 | [GS normal 开关](DOC/修改/0824-01-lambda_gs_normal开关与lambertian_normal2预设.md) / [gs_wrapping 预设](DOC/修改/0824-03-gs_wrapping预设.md) | `lambda_gs_normal`、`lambertian_normal2` 与 `gs_wrapping` 的实验含义。 |
| 0824-04 / 06 | [SH 评估兼容](DOC/修改/0824-04-eval_stage1_normals_gt支持SH基线.md) / [双法线与余弦损失](DOC/修改/0824-06-eval_stage1_normals_gt双法线与余弦损失.md) | normal evaluator 的模式支持、诊断兼容与双来源评估。 |
| 0824-05 | [逐 GS GT 位置光方向](DOC/修改/0824-05-逐GS_GT位置光方向消融模式.md) | `gt_point_direction_only` 的语义、坐标与无距离衰减约束。 |
| 0826-01 | [loss 对象与 preset 组合重构](DOC/修改/0826-01-loss对象与preset组合式重构.md) | 当前 `scripts/loss.py` 架构、PBR Stage 1 停用、全量单测基线。 |
| 0828-01 | [Stage1 损失预设与法线评估完善](DOC/修改/0828-01-Stage1损失预设与法线评估完善.md) | 最新 loss preset 构造方式，以及 `--normal_source {auto,independent,gs}` 和掩码余弦指标。 |

[0824-02 损失级光强归一化方案 D](DOC/修改/0824-02-损失级光强归一化方案D设计（待实施）.md) 是**未实施设计**，不得误认为现有训练已经具备该能力。

### 4. 报告：解释实验结果，不替代复现记录

| 文档 | 应在何时阅读 |
| --- | --- |
| [CV3s 坐标与逐 GS 光方向终审](DOC/报告/08.24-CV3s逐GS_GT位置光方向渲染坐标终审报告.md) | 怀疑方向符号、坐标系、时间索引或“GT 灯位是否真等于 GT 成像”时。结论是坐标链路正确，但当前只是局部中心方向 oracle。 |
| [SH 优于 GT 光 Lambertian 原因审计](DOC/报告/08.24-CV3s-SH优于GT光Lambertian原因审计报告.md) | 比较 SH 与 Lambertian 时。核心限制是 GT 灯位被简化、SH 外观自由度更强、相机/时间/光照未解耦。 |
| [SH 主干与 Per-light 方向融合方案](DOC/报告/08.24-SH主干与Per-light方向融合方案报告.md) | 评估下一阶段方案时。它是设计提案，尚未改代码或启动训练。 |
| [报告归档](DOC/报告/archive) | 查早期 Stage 1/2、PBR、法线、光强、坐标审计的原始证据；仅用于追溯，不覆盖上述现行结论。 |

## 代码、数据与产物的定位图

```text
LIHAN/DOC/指导 ── 数据规范、训练前检查
LIHAN/DOC/训练 ── 每次实验的计划、命令、结果与验收
LIHAN/DOC/修改 ── 代码变更缘由、影响与验证
LIHAN/DOC/报告 ── 跨实验审计与后续方案

scripts/train_stage1.py ── Stage 1 训练入口与调度
scripts/loss.py ────────── 损失对象、预设和组装
scene/photometric_lambertian.py ── Lambertian / GT 光模式
gaussian_renderer/__init__.py ─── 渲染模式分派
scripts/eval_stage1_normals_gt.py ─ GT normal 离线评估
scripts/render_stage1_insights.py ── RGB/alpha/albedo/normal 等可视化
LIHAN/scripts/ ─────────── 本地数据转换与 alpha 专项评估工具
tests/ ─────────────────── 各模块的可执行回归测试
output/ ────────────────── 训练、渲染、日志、指标和实验 README
```

数据路径以目标训练文档及其 `run.sh` 为准。常用简称：`CV3` 为 clothV3 动态数据，`CV3L` 为其 Lambertian 数据对，`CV3s` 为 clothV3 静态布料/相机运动版本。不要混用各自的 source、`lights.json`、GT normal 或光强标定值。

## 变更后的最低验证

- 改损失、参数解析或训练调度：先运行 `tests/test_stage1_loss.py`，再按改动范围运行 `python -m unittest discover -s tests -v`。
- 改相机、数据或灯光解释：运行相应的 `test_blender_camera_convention.py`、`test_directional_light_calibration.py`、`test_photometric_lambertian.py` 和数据转换/加载检查。
- 改 normal evaluator：运行 `test_normal_eval_utils.py`，并在同一 checkpoint 上明确记录采用 `independent`、`gs` 或 `auto`。
- 启动正式训练前：完成目标配置的冒烟，确认 checkpoint、日志、四类可视化和定量指标齐全；未满足验收条件时记录 `FAILED` 并保留产物。

当文档、代码和产物存在冲突时，优先级为：**当前代码与可复现实验产物 > 最新修改/训练记录 > 审计报告 > 历史归档**。发现不一致时，不要静默修正文档；先定位差异，再在相应目录新增中文记录。

四组Stage1/Stage2的SSIM、albedo、depth及normal MAE评测已完成：[双阶段真值评测报告](DOC/报告/1006-01-四组Stage1与Stage2真值评测报告.md)。训练仍只有01～04，全部结果、质量限制和最终状态已回填。
