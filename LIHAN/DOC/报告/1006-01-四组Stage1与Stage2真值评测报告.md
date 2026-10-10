# 1006-01～04：Stage1 与 Stage2 真值评测报告
<!-- 1006-final-record -->

**最终状态：4/4 COMPLETED，流程及产物验收PASS；训练只有1006-01～04。** 原版SH Stage1 35k、IR Stage2累计55k，两阶段共8组结果、120行逐帧指标已完成。

评测完成：2026-10-06 02:44:28（Asia/Tokyo）；最终核验及文档回填：2026-10-06T11:16:08+09:00。

[阶段CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/metrics.csv)；[最终核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/final_audit.json)。

四组的Stage2全图及前景包围框SSIM均下降，depth和normal MAE均增加；共同有效像素比较也显示几何误差上升。只有1006-04的原始albedo MAE下降（0.138899→0.122299，约11.95%），其bMAE也下降；01、02、03的原始albedo MAE增加。Stage1为含光照的SH基色代理，以上数值变化不能解释为Stage1物理材质恢复更准确。

<!-- /1006-final-record -->


结果更新时间：2026-10-06T02:44:27+09:00。四组均跳过冒烟、默认随机方块初始化，Stage1=35,000步，Stage2=累计55,000步；每阶段15个相同测试帧。

Stage1的albedo列为SH DC基色代理，含第一阶段外观中的光照影响；Stage2列为IR材质albedo。两者采用相同DC读取及线性光栅化流程，但Stage1尚未进行物理材质分解，不能把它的误差解释为已恢复物理反照率的质量。

| 编号 | 数据集 | 阶段 | SSIM↑ | 前景包围框SSIM↑ | Albedo MAE（线性RGB） | Albedo bMAE | Depth MAE（世界单位） | Normal MAE（°） | 法线覆盖率 | 深度覆盖率 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1006-01 | only_clothV4 | stage1 | 0.999192 | 0.995041 | 0.192875 | 0.129177 | 0.026094 | 7.4027 | 99.814% | 99.814% |
| 1006-01 | only_clothV4 | stage2 | 0.985574 | 0.907423 | 0.228857 | 0.131870 | 0.030203 | 7.9667 | 99.820% | 99.820% |
| 1006-02 | only_clothV4_lambertian | stage1 | 0.999261 | 0.995261 | 0.177255 | 0.114688 | 0.027284 | 7.5799 | 99.817% | 99.817% |
| 1006-02 | only_clothV4_lambertian | stage2 | 0.985740 | 0.909815 | 0.198856 | 0.117014 | 0.031638 | 8.0912 | 99.797% | 99.797% |
| 1006-03 | only_clothV4_Dynamic | stage1 | 0.997447 | 0.989456 | 0.158172 | 0.118708 | 0.102184 | 11.5613 | 99.675% | 99.675% |
| 1006-03 | only_clothV4_Dynamic | stage2 | 0.975469 | 0.877510 | 0.161104 | 0.121246 | 0.108388 | 11.9397 | 99.674% | 99.674% |
| 1006-04 | only_clothV4_Dynamic_Lambertian | stage1 | 0.995352 | 0.982812 | 0.138899 | 0.099402 | 0.087812 | 12.4948 | 99.332% | 99.332% |
| 1006-04 | only_clothV4_Dynamic_Lambertian | stage2 | 0.975389 | 0.879943 | 0.122299 | 0.091330 | 0.093827 | 12.8886 | 99.266% | 99.266% |

## Stage2 相对 Stage1 的变化

MAE负值表示误差下降，SSIM正值表示结构相似性提高；覆盖率变化另存stage_deltas.csv，不把漏预测引起的条件误差下降当作质量改善。

| 编号 | Δ SSIM | Δ Albedo MAE | Δ Depth MAE | Δ Normal MAE（°） |
| --- | ---: | ---: | ---: | ---: |
| 1006-01 | -0.013618 | 0.035982 | 0.004109 | 0.5640 |
| 1006-02 | -0.013521 | 0.021602 | 0.004355 | 0.5114 |
| 1006-03 | -0.021979 | 0.002931 | 0.006204 | 0.3784 |
| 1006-04 | -0.019963 | -0.016600 | 0.006015 | 0.3938 |

![两阶段误差对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/stage_comparison.png)

## 共同有效像素的几何比较

同一帧中使用两阶段均有效的相同像素，法线和深度分别定义有效交集；逐帧等权。此表补充主表，覆盖率仍不能忽略。

| 编号 | S1 Normal MAE° | S2 Normal MAE° | 共同法线覆盖率 | S1 Depth MAE | S2 Depth MAE | 共同深度覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1006-01 | 7.3869 | 7.9375 | 99.748% | 0.026099 | 0.030215 | 99.748% |
| 1006-02 | 7.5636 | 8.0687 | 99.739% | 0.027294 | 0.031649 | 99.739% |
| 1006-03 | 11.5314 | 11.9025 | 99.562% | 0.102097 | 0.108251 | 99.562% |
| 1006-04 | 12.4311 | 12.8456 | 99.140% | 0.087033 | 0.093708 | 99.140% |

## 指标口径与限制

- SSIM：RGB渲染与GT全图SSIM，15帧等权平均，越高越好。另报GT alpha遮罩后的前景包围框SSIM，避免黑背景过多导致全图指标偏高。35k使用原版SH渲染，55k复用原版IR、实际time及逐时刻更新BVH的RGB结果；不使用仅4个测试视角的训练日志SSIM。
- Albedo：get_albedo读取DC基色（原算法夹到sRGB 0.03～0.97），解码线性RGB后以零背景光栅化并除预测alpha；对GT alpha≥0.99且收缩2像素的全部真值内部前景计算三通道绝对误差，包含漏预测像素。主MAE为全部像素汇总；bMAE为材质MAE等权。训练帧尺度拟合仅作诊断，主结果不缩放、不作ICP或额外后处理。
- Depth：正相机Z，主表为expected depth的逐帧等权MAE；median、仅布料MAE与像素加权统计见原始metrics.json及metrics.csv。使用原世界单位，没有假设米或厘米。
- Normal：归一化世界空间法线与Blender EXR真值的有向夹角acos(dot)，单位度；主表GT前景逐帧等权MAE。XYZ分量MAE、内部前景、仅布料及漏预测180°惩罚诊断另外保留。法线/深度的主误差对预测alpha≥0.5及有效值计分，并同时报告GT定义分母的覆盖率。
- 两阶段相机、真实time和真值一致，分辨率640×360；真值仅离线评分，不反馈训练。Stage2原版冻结位置/尺度/旋转和几何形变，但可改变opacity，因此渲染法线和深度误差仍可能变化。

## 可复核产物

- **1006-01**：Stage1原始结果为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage1/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage1/albedo/iteration_35000/metrics.json)；Stage2为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/albedo/iteration_55000/metrics.json)。
- **1006-02**：Stage1原始结果为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/evaluation/stage1/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/evaluation/stage1/albedo/iteration_35000/metrics.json)；Stage2为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/evaluation/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/evaluation/albedo/iteration_55000/metrics.json)。
- **1006-03**：Stage1原始结果为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/evaluation/stage1/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/evaluation/stage1/albedo/iteration_35000/metrics.json)；Stage2为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/evaluation/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/evaluation/albedo/iteration_55000/metrics.json)。
- **1006-04**：Stage1原始结果为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/evaluation/stage1/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/evaluation/stage1/albedo/iteration_35000/metrics.json)；Stage2为[geometry](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/evaluation/geometry/metrics.json)、[albedo](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/evaluation/albedo/iteration_55000/metrics.json)。

汇总：[8组阶段指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/metrics.csv)、[120行逐帧指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/frames.csv)、[共同有效像素统计](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/common_valid_summary.csv)。完整新增命令、日志与状态在本任务目录；核验结果verification.json保存实际来源哈希。自动评测PASS表示流程及数值一致性检查通过，不代替重建质量阈值或人工目检。

可视报告：[本地预览](http://127.0.0.1:4186/)；[离线HTML](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/report-app/.data-app-offline/exports/stage-comparison.html)。科学对比图由真实指标生成，报告网页的浏览器目检未执行（当前会话缺少可调用的原生浏览器工具）。

## 最终产物验收与代表帧目检

目检仅覆盖固定相机frame_0008的t0/t60/t119，以及NVS测试索引0/7/14代表图；没有逐帧人工审阅全部视频。PASS表示复现流程、产物与代表帧检查完成，未设材质/几何质量阈值，不表示真值重建准确。

| 编号 | 原版训练/自动产物 | 最终代表帧目检 | 主要质量限制 | 流水线完成时间（Tokyo） |
| --- | --- | --- | --- | --- |
| 1006-01 | PASS：35k/55k、120帧诊断、10视频全解码、15帧NVS | [PASS记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/inspection/manual_review.json) | 红布、灰蓝座面和棕腿可辨；白底盘有灰色云状斑驳及残余明暗。 | 2026-10-06T00:38:30+09:00 |
| 1006-02 | PASS：35k/55k、120帧诊断、10视频全解码、15帧NVS | [PASS记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/inspection/manual_review.json) | 主要物体颜色可辨；白底盘仍有云状灰斑和残余明暗。 | 2026-10-06T00:38:19+09:00 |
| 1006-03 | PASS：35k/55k、120帧诊断、10视频全解码、15帧NVS | [PASS记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/inspection/manual_review.json) | 红布有明显灰白斑块，初始平展及折叠阶段均可见；底盘及座面也有斑驳。 中间折叠时布料薄边、局部碎片状低alpha和不均匀轮廓可见。 | 2026-10-06T02:41:09+09:00 |
| 1006-04 | PASS：35k/55k、120帧诊断、10视频全解码、15帧NVS | [PASS记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/inspection/manual_review.json) | 红布比03均匀一些，仍有暗斑及折叠明暗残留，底盘有云状灰斑。 初始及中间时刻边缘毛糙、低alpha局部和薄褶皱锯齿较明显。 | 2026-10-06T02:40:38+09:00 |

目检包括RGB、alpha、world normal、separation_large、Stage2 albedo/roughness/HDR和NVS对照；代表图分别为各组inspection/four_categories.png、materials_hdr.png、nvs_comparison.png。动态03的灰白斑块和04的毛糙薄边仍存在，不能因流程PASS忽略。未逐帧人工检查全部视频；完整视频解码是已有自动检查。

SSIM原始结果：

- 1006-01：[Stage1 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage1/rgb/metrics.json)；[Stage2 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/rgb/metrics.json)。
- 1006-02：[Stage1 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/evaluation/stage1/rgb/metrics.json)；[Stage2 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/evaluation/rgb/metrics.json)。
- 1006-03：[Stage1 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/evaluation/stage1/rgb/metrics.json)；[Stage2 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/evaluation/rgb/metrics.json)。
- 1006-04：[Stage1 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/evaluation/stage1/rgb/metrics.json)；[Stage2 RGB](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/evaluation/rgb/metrics.json)。

最终来源哈希、浮点数组重算及表格核验：[final_audit.json](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/evaluation/stage_comparison_task/final_audit.json)。历史部分结果partial_metrics.json / partial_verification.json保留供追溯，正式完整结果以metrics.json / verification.json为准。
