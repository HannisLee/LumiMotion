# CV4DL 固定共用光强 GT Lambertian Stage2 评测报告

结果更新：2026-10-06T17:26:36.863495+09:00。正式2026-10-06T17:09:40.986267+09:00 → 2026-10-06T17:25:30.551612+09:00；固定光强6.030241533058244；35k→55k新增20,000次Adam更新。**COMPLETED，流程/产物/代表图验收PASS；质量限制见下文。**

raw Albedo MAE由原版IR的0.122299变为0.120561，下降1.42%；材质均衡bMAE由0.091330变为0.085033。前景SSIM 0.879943→0.828440。

## 同来源三阶段主指标

| 数据集 | 阶段 | Albedo MAE | bMAE | Normal MAE° | Depth MAE | SSIM↑ | 前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV4DL | SH Stage1 35k | 0.138899 | 0.099402 | 12.4948 | 0.087812 | 0.995352 | 0.982812 |
| CV4DL | 原版IR Stage2 55k | 0.122299 | 0.091330 | 12.8886 | 0.093827 | 0.975389 | 0.879943 |
| CV4DL | GT Lambertian 55k | 0.120561 | 0.085033 | 12.4948 | 0.087812 | 0.964946 | 0.828440 |

Stage1的albedo是含光照的SH DC外观代理，不是已恢复的物理反照率。原版IR与Lambertian优化范围/成像模型不同，不能作为单因素消融。

## Albedo及四材质MAE

| 口径/区域 | SH35k代理 | IR55k材质 | GT Lambertian55k |
| --- | --- | --- | --- |
| raw · 全前景 | 0.138899 | 0.122299 | 0.120561 |
| raw · 红布 | 0.044865 | 0.047336 | 0.044908 |
| raw · 椅面 | 0.045293 | 0.045000 | 0.040988 |
| raw · 木框 | 0.082825 | 0.084326 | 0.062493 |
| raw · 底盘 | 0.224624 | 0.188655 | 0.191745 |
| raw · 四材质bMAE | 0.099402 | 0.091330 | 0.085033 |
| train_scalar_diagnostic · 全前景 | 0.124914 | 0.117265 | 0.115611 |
| train_scalar_diagnostic · 红布 | 0.067593 | 0.053371 | 0.042559 |
| train_scalar_diagnostic · 椅面 | 0.060997 | 0.051718 | 0.037499 |
| train_scalar_diagnostic · 木框 | 0.092139 | 0.091365 | 0.059769 |
| train_scalar_diagnostic · 底盘 | 0.174801 | 0.172740 | 0.183553 |
| train_scalar_diagnostic · 四材质bMAE | 0.098883 | 0.092299 | 0.080845 |
| train_rgb_diagnostic · 全前景 | 0.124314 | 0.112883 | 0.115637 |
| train_rgb_diagnostic · 红布 | 0.061982 | 0.060453 | 0.042255 |
| train_rgb_diagnostic · 椅面 | 0.061533 | 0.050323 | 0.037718 |
| train_rgb_diagnostic · 木框 | 0.090850 | 0.094081 | 0.059323 |
| train_rgb_diagnostic · 底盘 | 0.177754 | 0.158192 | 0.183811 |
| train_rgb_diagnostic · 四材质bMAE | 0.098030 | 0.090762 | 0.080777 |

raw为未缩放主指标；scalar与RGB系数仅由各自105训练帧的GT albedo离线拟合，测试不参与、不反馈训练。前景/材质高Alpha MAE、相对亮度MAE及像素误差P95均完整保存在全量MAE明细，不能将总体改善解释为所有子指标改善。

## 法线、深度与覆盖率

| 区域/指标 | SH35k | IR55k | GT Lambertian55k |
| --- | --- | --- | --- |
| foreground · Normal角度MAE° | 12.494832 | 12.888626 | 12.494832 |
| foreground · Normal分量MAE | 0.101947 | 0.105255 | 0.101947 |
| foreground · Expected Depth MAE | 0.087812 | 0.093827 | 0.087812 |
| foreground · Median Depth MAE | 0.087939 | 0.094114 | 0.087939 |
| foreground · Normal覆盖率% | 99.331578 | 99.266249 | 99.331578 |
| interior · Normal角度MAE° | 11.778614 | 12.139750 | 11.778614 |
| interior · Normal分量MAE | 0.096597 | 0.099675 | 0.096597 |
| interior · Expected Depth MAE | 0.086228 | 0.093036 | 0.086228 |
| interior · Median Depth MAE | 0.086186 | 0.093214 | 0.086186 |
| interior · Normal覆盖率% | 99.904066 | 99.834091 | 99.904066 |
| cloth_foreground · Normal角度MAE° | 26.316891 | 25.767123 | 26.316891 |
| cloth_foreground · Normal分量MAE | 0.216573 | 0.212470 | 0.216573 |
| cloth_foreground · Expected Depth MAE | 0.239794 | 0.268518 | 0.239794 |
| cloth_foreground · Median Depth MAE | 0.251770 | 0.281782 | 0.251770 |
| cloth_foreground · Normal覆盖率% | 98.535208 | 98.342490 | 98.535208 |
| cloth_interior · Normal角度MAE° | 25.901775 | 24.635117 | 25.901775 |
| cloth_interior · Normal分量MAE | 0.212906 | 0.203975 | 0.212906 |
| cloth_interior · Expected Depth MAE | 0.211456 | 0.245418 | 0.211456 |
| cloth_interior · Median Depth MAE | 0.219094 | 0.262200 | 0.219094 |
| cloth_interior · Normal覆盖率% | 99.738994 | 99.548612 | 99.738994 |

Normal角度MAE单位度，XYZ分量为单位法线分量绝对误差。Depth是正相机Z、原世界单位，分别保存逐帧等权和像素加权，无尺度/位移/ICP对齐。条件MAE必须连同覆盖率解读，另保留漏预测180°惩罚及深度导出法线诊断。Stage1几何被冻结，实际重新评分并复核浮点数组。

## RGB与Alpha MAE

| 本轮浮点指标 | 逐帧等权 | 像素加权 |
| --- | --- | --- |
| rgb_srgb_full_mae | 0.01036520 | 0.01036520 |
| rgb_srgb_foreground_mae | 0.08856818 | 0.08568340 |
| rgb_linear_full_mae | 0.01055082 | 0.01055082 |
| rgb_linear_foreground_mae | 0.09310916 | 0.09040798 |
| alpha_full_mae | 0.00190107 | 0.00190107 |
| alpha_foreground_mae | 0.00917886 | 0.00752512 |

RGB与Alpha用浮点前向渲染评分，非PNG量化评分；sRGB夹到[0,1]后另解码线性RGB，前景mask为GT alpha>0.5，Alpha使用真实软RGBA。

## 真实代表图目检与质量限制

rgb：悬空布料、折叠与最终椅背布料均可辨；0008颜色更红但仍斑驳，0064褶皱内侧明显过暗，0120残留色斑，平台阴影与GT不同。

alpha：首帧布料下方有细碎浮点，主要布料/椅子/底盘轮廓完整；折叠和最终轮廓存在，冻结Stage1的软边/碎点仍保留。

normal：首中末布料、椅子及底盘方向可辨；中段褶皱方向有变化，首帧碎点区存在异常小法线斑。本轮保持Stage1法线。

albedo：红布保留红黑斑与深暗折痕，底盘灰黑阴影斑显著，布下碎点仍带红色；材质未完全脱离光照和几何缺陷。

转灯首尾布料亮、正中全黑，存在明显亮暗响应；HDR首/中/末红布和粉橙底盘可辨，但碎点、深折痕和底盘原阴影仍存在。HDR无独立GT，仅作功能检查。

![四类代表图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/four_categories.png)

![RGB与GT](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/rgb_comparison.png)

![重光照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/relight.png)

仅人工查看首/中/末与15测试帧中的0/7/14代表图；6视频完整解码属于自动检查，没有逐帧人工观看所有视频。HDR无独立GT，流程PASS不表示重建质量达标。

## 来源、完整命令与可复核结果

- Stage1：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/model_mlp`，35000步；source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-04-only_clothV4_Dynamic_Lambertian-LumiMotion原版随机方块两阶段/data_view`。
- 最终checkpoint：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp/stage2_photometric/iteration_55000/checkpoint.pth`。
- [完整命令及日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/README.md)；[本组训练记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian训练记录.md)；[共用光强标定](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/shared_light_calibration.json)。
- [Albedo JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/evaluation/albedo/iteration_55000/metrics.json)；[几何JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/evaluation/geometry/metrics.json)；[RGB JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/model_mlp/renders_stage2/ours_55000/eval/metrics.json)；[RGB/Alpha MAE](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/evaluation/rgb_mae/metrics.json)；[Albedo SSIM](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/evaluation/albedo/ssim.json)。
- [所有MAE明细](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/all_mae.csv)；[180行逐帧对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/frames.csv)；[浮点重算及来源核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)。
- [自动产物验收](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/artifact_checks.json)；[真实目检记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-09-only_clothV4_Dynamic_Lambertian-Stage2-GT灯位Lambertian/inspection/manual_review.json)。失败或历史日志保持原样，不覆盖删除。
