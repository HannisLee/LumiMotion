# CV4L 固定共用光强 GT Lambertian Stage2 评测报告

结果更新：2026-10-06T17:26:36.861248+09:00。正式2026-10-06T16:54:18.996170+09:00 → 2026-10-06T17:09:40.982827+09:00；固定光强6.030241533058244；35k→55k新增20,000次Adam更新。**COMPLETED，流程/产物/代表图验收PASS；质量限制见下文。**

raw Albedo MAE由原版IR的0.198856变为0.137338，下降30.94%；材质均衡bMAE由0.117014变为0.083037。前景SSIM 0.909815→0.850182。

## 同来源三阶段主指标

| 数据集 | 阶段 | Albedo MAE | bMAE | Normal MAE° | Depth MAE | SSIM↑ | 前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV4L | SH Stage1 35k | 0.177255 | 0.114688 | 7.5799 | 0.027284 | 0.999261 | 0.995261 |
| CV4L | 原版IR Stage2 55k | 0.198856 | 0.117014 | 8.0912 | 0.031638 | 0.985740 | 0.909815 |
| CV4L | GT Lambertian 55k | 0.137338 | 0.083037 | 7.5799 | 0.027284 | 0.976506 | 0.850182 |

Stage1的albedo是含光照的SH DC外观代理，不是已恢复的物理反照率。原版IR与Lambertian优化范围/成像模型不同，不能作为单因素消融。

## Albedo及四材质MAE

| 口径/区域 | SH35k代理 | IR55k材质 | GT Lambertian55k |
| --- | --- | --- | --- |
| raw · 全前景 | 0.177255 | 0.198856 | 0.137338 |
| raw · 红布 | 0.111351 | 0.070979 | 0.063583 |
| raw · 椅面 | 0.033089 | 0.044822 | 0.023841 |
| raw · 木框 | 0.087881 | 0.078068 | 0.062447 |
| raw · 底盘 | 0.226432 | 0.274187 | 0.182277 |
| raw · 四材质bMAE | 0.114688 | 0.117014 | 0.083037 |
| train_scalar_diagnostic · 全前景 | 0.152638 | 0.134029 | 0.137260 |
| train_scalar_diagnostic · 红布 | 0.134909 | 0.089918 | 0.063547 |
| train_scalar_diagnostic · 椅面 | 0.049217 | 0.046369 | 0.023777 |
| train_scalar_diagnostic · 木框 | 0.099153 | 0.097832 | 0.062366 |
| train_scalar_diagnostic · 底盘 | 0.174601 | 0.161461 | 0.182162 |
| train_scalar_diagnostic · 四材质bMAE | 0.114470 | 0.098895 | 0.082963 |
| train_rgb_diagnostic · 全前景 | 0.152660 | 0.133810 | 0.137259 |
| train_rgb_diagnostic · 红布 | 0.130577 | 0.090057 | 0.063544 |
| train_rgb_diagnostic · 椅面 | 0.049998 | 0.046382 | 0.023782 |
| train_rgb_diagnostic · 木框 | 0.097899 | 0.097596 | 0.062354 |
| train_rgb_diagnostic · 底盘 | 0.175992 | 0.161102 | 0.182163 |
| train_rgb_diagnostic · 四材质bMAE | 0.113617 | 0.098784 | 0.082961 |

raw为未缩放主指标；scalar与RGB系数仅由各自105训练帧的GT albedo离线拟合，测试不参与、不反馈训练。前景/材质高Alpha MAE、相对亮度MAE及像素误差P95均完整保存在全量MAE明细，不能将总体改善解释为所有子指标改善。

## 法线、深度与覆盖率

| 区域/指标 | SH35k | IR55k | GT Lambertian55k |
| --- | --- | --- | --- |
| foreground · Normal角度MAE° | 7.579862 | 8.091242 | 7.579862 |
| foreground · Normal分量MAE | 0.061483 | 0.065627 | 0.061483 |
| foreground · Expected Depth MAE | 0.027284 | 0.031638 | 0.027284 |
| foreground · Median Depth MAE | 0.025395 | 0.030161 | 0.025395 |
| foreground · Normal覆盖率% | 99.817011 | 99.797060 | 99.817011 |
| interior · Normal角度MAE° | 6.938596 | 7.338797 | 6.938596 |
| interior · Normal分量MAE | 0.056516 | 0.059824 | 0.056516 |
| interior · Expected Depth MAE | 0.026597 | 0.031373 | 0.026597 |
| interior · Median Depth MAE | 0.024566 | 0.029763 | 0.024566 |
| interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 |
| cloth_foreground · Normal角度MAE° | 13.616663 | 14.263895 | 13.616663 |
| cloth_foreground · Normal分量MAE | 0.113055 | 0.118566 | 0.113055 |
| cloth_foreground · Expected Depth MAE | 0.031342 | 0.032935 | 0.031342 |
| cloth_foreground · Median Depth MAE | 0.029031 | 0.030096 | 0.029031 |
| cloth_foreground · Normal覆盖率% | 99.769099 | 99.742018 | 99.769099 |
| cloth_interior · Normal角度MAE° | 12.506744 | 12.814267 | 12.506744 |
| cloth_interior · Normal分量MAE | 0.103798 | 0.106660 | 0.103798 |
| cloth_interior · Expected Depth MAE | 0.026848 | 0.029014 | 0.026848 |
| cloth_interior · Median Depth MAE | 0.025547 | 0.026570 | 0.025547 |
| cloth_interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 |

Normal角度MAE单位度，XYZ分量为单位法线分量绝对误差。Depth是正相机Z、原世界单位，分别保存逐帧等权和像素加权，无尺度/位移/ICP对齐。条件MAE必须连同覆盖率解读，另保留漏预测180°惩罚及深度导出法线诊断。Stage1几何被冻结，实际重新评分并复核浮点数组。

## RGB与Alpha MAE

| 本轮浮点指标 | 逐帧等权 | 像素加权 |
| --- | --- | --- |
| rgb_srgb_full_mae | 0.00677089 | 0.00677089 |
| rgb_srgb_foreground_mae | 0.07121743 | 0.07091532 |
| rgb_linear_full_mae | 0.00739697 | 0.00739697 |
| rgb_linear_foreground_mae | 0.07875140 | 0.07912819 |
| alpha_full_mae | 0.00027389 | 0.00027389 |
| alpha_foreground_mae | 0.00160918 | 0.00156084 |

RGB与Alpha用浮点前向渲染评分，非PNG量化评分；sRGB夹到[0,1]后另解码线性RGB，前景mask为GT alpha>0.5，Alpha使用真实软RGBA。

## 真实代表图目检与质量限制

rgb：正式55k首尾正面红布接近GT但颜色仍有差异，主体清楚；测试0064背面布料/椅腿明显偏暗，平台阴影与GT不一致。

alpha：首中末椅子与底盘轮廓和孔隙稳定；冻结轮廓保持。

normal：布料、椅面与底盘世界法线连续且可区分；本轮保持Stage1法线。

albedo：首尾红布仍带褶皱阴影；背面存在大面积黑区和红灰色块，木框/底盘有明显明暗残留，材质分离不完全。

转灯有明显亮暗变化，背向近黑符合无补光的直接Lambertian；HDR正面红布及粉橙底盘可辨，背面仍有斑块/明暗残留。HDR没有独立GT，仅检查功能。

![四类代表图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/inspection/four_categories.png)

![RGB与GT](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/inspection/rgb_comparison.png)

![重光照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/inspection/relight.png)

仅人工查看首/中/末与15测试帧中的0/7/14代表图；6视频完整解码属于自动检查，没有逐帧人工观看所有视频。HDR无独立GT，流程PASS不表示重建质量达标。

## 来源、完整命令与可复核结果

- Stage1：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/model_mlp`，35000步；source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/data_view`。
- 最终checkpoint：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/model_mlp/stage2_photometric/iteration_55000/checkpoint.pth`。
- [完整命令及日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/README.md)；[本组训练记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian训练记录.md)；[共用光强标定](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/inspection/shared_light_calibration.json)。
- [Albedo JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/evaluation/albedo/iteration_55000/metrics.json)；[几何JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/evaluation/geometry/metrics.json)；[RGB JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/model_mlp/renders_stage2/ours_55000/eval/metrics.json)；[RGB/Alpha MAE](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/evaluation/rgb_mae/metrics.json)；[Albedo SSIM](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/evaluation/albedo/ssim.json)。
- [所有MAE明细](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/all_mae.csv)；[180行逐帧对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/frames.csv)；[浮点重算及来源核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)。
- [自动产物验收](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/inspection/artifact_checks.json)；[真实目检记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/inspection/manual_review.json)。失败或历史日志保持原样，不覆盖删除。
