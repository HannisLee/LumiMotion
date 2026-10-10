# CV4D 固定共用光强 GT Lambertian Stage2 评测报告

结果更新：2026-10-06T17:26:36.862386+09:00。正式2026-10-06T17:09:40.825612+09:00 → 2026-10-06T17:25:30.760242+09:00；固定光强6.030241533058244；35k→55k新增20,000次Adam更新。**COMPLETED，流程/产物/代表图验收PASS；质量限制见下文。**

raw Albedo MAE由原版IR的0.161104变为0.136366，下降15.35%；材质均衡bMAE由0.121246变为0.101979。前景SSIM 0.877510→0.808305。

## 同来源三阶段主指标

| 数据集 | 阶段 | Albedo MAE | bMAE | Normal MAE° | Depth MAE | SSIM↑ | 前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV4D | SH Stage1 35k | 0.158172 | 0.118708 | 11.5613 | 0.102184 | 0.997447 | 0.989456 |
| CV4D | 原版IR Stage2 55k | 0.161104 | 0.121246 | 11.9397 | 0.108388 | 0.975469 | 0.877510 |
| CV4D | GT Lambertian 55k | 0.136366 | 0.101979 | 11.5613 | 0.102184 | 0.961797 | 0.808305 |

Stage1的albedo是含光照的SH DC外观代理，不是已恢复的物理反照率。原版IR与Lambertian优化范围/成像模型不同，不能作为单因素消融。

## Albedo及四材质MAE

| 口径/区域 | SH35k代理 | IR55k材质 | GT Lambertian55k |
| --- | --- | --- | --- |
| raw · 全前景 | 0.158172 | 0.161104 | 0.136366 |
| raw · 红布 | 0.078823 | 0.069938 | 0.085568 |
| raw · 椅面 | 0.077789 | 0.076256 | 0.067304 |
| raw · 木框 | 0.083967 | 0.092752 | 0.065674 |
| raw · 底盘 | 0.234252 | 0.246037 | 0.189371 |
| raw · 四材质bMAE | 0.118708 | 0.121246 | 0.101979 |
| train_scalar_diagnostic · 全前景 | 0.141655 | 0.127682 | 0.126885 |
| train_scalar_diagnostic · 红布 | 0.104389 | 0.079144 | 0.079144 |
| train_scalar_diagnostic · 椅面 | 0.103630 | 0.099448 | 0.060299 |
| train_scalar_diagnostic · 木框 | 0.093906 | 0.111137 | 0.061364 |
| train_scalar_diagnostic · 底盘 | 0.175673 | 0.165584 | 0.175461 |
| train_scalar_diagnostic · 四材质bMAE | 0.119400 | 0.113828 | 0.094067 |
| train_rgb_diagnostic · 全前景 | 0.141780 | 0.125317 | 0.126960 |
| train_rgb_diagnostic · 红布 | 0.101873 | 0.081724 | 0.079026 |
| train_rgb_diagnostic · 椅面 | 0.104123 | 0.098336 | 0.060486 |
| train_rgb_diagnostic · 木框 | 0.093044 | 0.113089 | 0.061012 |
| train_rgb_diagnostic · 底盘 | 0.177754 | 0.158767 | 0.175695 |
| train_rgb_diagnostic · 四材质bMAE | 0.119198 | 0.112979 | 0.094055 |

raw为未缩放主指标；scalar与RGB系数仅由各自105训练帧的GT albedo离线拟合，测试不参与、不反馈训练。前景/材质高Alpha MAE、相对亮度MAE及像素误差P95均完整保存在全量MAE明细，不能将总体改善解释为所有子指标改善。

## 法线、深度与覆盖率

| 区域/指标 | SH35k | IR55k | GT Lambertian55k |
| --- | --- | --- | --- |
| foreground · Normal角度MAE° | 11.561345 | 11.939720 | 11.561345 |
| foreground · Normal分量MAE | 0.094363 | 0.097532 | 0.094363 |
| foreground · Expected Depth MAE | 0.102184 | 0.108388 | 0.102184 |
| foreground · Median Depth MAE | 0.101882 | 0.109401 | 0.101882 |
| foreground · Normal覆盖率% | 99.675374 | 99.673608 | 99.675374 |
| interior · Normal角度MAE° | 10.879992 | 11.197626 | 10.879992 |
| interior · Normal分量MAE | 0.089262 | 0.091979 | 0.089262 |
| interior · Expected Depth MAE | 0.100115 | 0.106979 | 0.100115 |
| interior · Median Depth MAE | 0.099623 | 0.107892 | 0.099623 |
| interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 |
| cloth_foreground · Normal角度MAE° | 22.028515 | 21.894953 | 22.028515 |
| cloth_foreground · Normal分量MAE | 0.182606 | 0.181928 | 0.182606 |
| cloth_foreground · Expected Depth MAE | 0.241789 | 0.246578 | 0.241789 |
| cloth_foreground · Median Depth MAE | 0.249046 | 0.255342 | 0.249046 |
| cloth_foreground · Normal覆盖率% | 99.452627 | 99.484077 | 99.452627 |
| cloth_interior · Normal角度MAE° | 21.104064 | 20.811039 | 21.104064 |
| cloth_interior · Normal分量MAE | 0.175648 | 0.173732 | 0.175648 |
| cloth_interior · Expected Depth MAE | 0.247455 | 0.249474 | 0.247455 |
| cloth_interior · Median Depth MAE | 0.254491 | 0.257628 | 0.254491 |
| cloth_interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 |

Normal角度MAE单位度，XYZ分量为单位法线分量绝对误差。Depth是正相机Z、原世界单位，分别保存逐帧等权和像素加权，无尺度/位移/ICP对齐。条件MAE必须连同覆盖率解读，另保留漏预测180°惩罚及深度导出法线诊断。Stage1几何被冻结，实际重新评分并复核浮点数组。

## RGB与Alpha MAE

| 本轮浮点指标 | 逐帧等权 | 像素加权 |
| --- | --- | --- |
| rgb_srgb_full_mae | 0.01212781 | 0.01212781 |
| rgb_srgb_foreground_mae | 0.10262247 | 0.10188773 |
| rgb_linear_full_mae | 0.01169669 | 0.01169669 |
| rgb_linear_foreground_mae | 0.10275460 | 0.10085178 |
| alpha_full_mae | 0.00104858 | 0.00104858 |
| alpha_foreground_mae | 0.00422199 | 0.00387615 |

RGB与Alpha用浮点前向渲染评分，非PNG量化评分；sRGB夹到[0,1]后另解码线性RGB，前景mask为GT alpha>0.5，Alpha使用真实软RGBA。

## 真实代表图目检与质量限制

rgb：悬空布料、折叠及最终椅背布料均可辨；0008布料偏粉且斑驳，0064褶皱内侧明显过暗，0120有条纹，平台阴影与GT不同。

alpha：首帧布料下方有细碎浮点，主要布料/椅子/底盘轮廓完整；折叠和最终轮廓存在，冻结Stage1的软边/碎点仍保留。

normal：首中末布料、椅子及底盘方向可辨；中段褶皱方向有变化，首帧碎点区存在异常小法线斑。本轮保持Stage1法线。

albedo：红布明显保留粉色亮斑和黑暗折痕，底盘灰黑阴影斑显著，初始布下碎点带红色；材质未完全脱离光照和几何缺陷。

转灯首尾布料亮、正中全黑，存在明显亮暗响应；HDR首/中/末红布和粉橙底盘可辨，但碎点、深折痕和底盘原阴影仍存在。HDR无独立GT，仅作功能检查。

![四类代表图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/inspection/four_categories.png)

![RGB与GT](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/inspection/rgb_comparison.png)

![重光照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/inspection/relight.png)

仅人工查看首/中/末与15测试帧中的0/7/14代表图；6视频完整解码属于自动检查，没有逐帧人工观看所有视频。HDR无独立GT，流程PASS不表示重建质量达标。

## 来源、完整命令与可复核结果

- Stage1：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/model_mlp`，35000步；source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-03-only_clothV4_Dynamic-LumiMotion原版随机方块两阶段/data_view`。
- 最终checkpoint：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp/stage2_photometric/iteration_55000/checkpoint.pth`。
- [完整命令及日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/README.md)；[本组训练记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian训练记录.md)；[共用光强标定](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/inspection/shared_light_calibration.json)。
- [Albedo JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/evaluation/albedo/iteration_55000/metrics.json)；[几何JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/evaluation/geometry/metrics.json)；[RGB JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/model_mlp/renders_stage2/ours_55000/eval/metrics.json)；[RGB/Alpha MAE](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/evaluation/rgb_mae/metrics.json)；[Albedo SSIM](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/evaluation/albedo/ssim.json)。
- [所有MAE明细](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/all_mae.csv)；[180行逐帧对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/frames.csv)；[浮点重算及来源核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)。
- [自动产物验收](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/inspection/artifact_checks.json)；[真实目检记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-08-only_clothV4_Dynamic-Stage2-GT灯位Lambertian/inspection/manual_review.json)。失败或历史日志保持原样，不覆盖删除。
