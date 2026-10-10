# CV4 固定共用光强 GT Lambertian Stage2 评测报告

结果更新：2026-10-06T17:26:36.859666+09:00。正式2026-10-06T16:54:18.996126+09:00 → 2026-10-06T17:09:40.822214+09:00；固定光强6.030241533058244；35k→55k新增20,000次Adam更新。**COMPLETED，流程/产物/代表图验收PASS；质量限制见下文。**

raw Albedo MAE由原版IR的0.228857变为0.138418，下降39.52%；材质均衡bMAE由0.131870变为0.088483。前景SSIM 0.907423→0.830631。

## 同来源三阶段主指标

| 数据集 | 阶段 | Albedo MAE | bMAE | Normal MAE° | Depth MAE | SSIM↑ | 前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV4 | SH Stage1 35k | 0.192875 | 0.129177 | 7.4027 | 0.026094 | 0.999192 | 0.995041 |
| CV4 | 原版IR Stage2 55k | 0.228857 | 0.131870 | 7.9667 | 0.030203 | 0.985574 | 0.907423 |
| CV4 | GT Lambertian 55k | 0.138418 | 0.088483 | 7.4027 | 0.026094 | 0.973556 | 0.830631 |

Stage1的albedo是含光照的SH DC外观代理，不是已恢复的物理反照率。原版IR与Lambertian优化范围/成像模型不同，不能作为单因素消融。

## Albedo及四材质MAE

| 口径/区域 | SH35k代理 | IR55k材质 | GT Lambertian55k |
| --- | --- | --- | --- |
| raw · 全前景 | 0.192875 | 0.228857 | 0.138418 |
| raw · 红布 | 0.140918 | 0.084398 | 0.078872 |
| raw · 椅面 | 0.048891 | 0.051056 | 0.034063 |
| raw · 木框 | 0.086500 | 0.074645 | 0.064024 |
| raw · 底盘 | 0.240401 | 0.317382 | 0.176972 |
| raw · 四材质bMAE | 0.129177 | 0.131870 | 0.088483 |
| train_scalar_diagnostic · 全前景 | 0.163314 | 0.145136 | 0.137758 |
| train_scalar_diagnostic · 红布 | 0.173886 | 0.119551 | 0.078417 |
| train_scalar_diagnostic · 椅面 | 0.071377 | 0.056775 | 0.033413 |
| train_scalar_diagnostic · 木框 | 0.098896 | 0.099799 | 0.063424 |
| train_scalar_diagnostic · 底盘 | 0.176403 | 0.168081 | 0.176095 |
| train_scalar_diagnostic · 四材质bMAE | 0.130140 | 0.111051 | 0.087837 |
| train_rgb_diagnostic · 全前景 | 0.163733 | 0.145162 | 0.137757 |
| train_rgb_diagnostic · 红布 | 0.170240 | 0.119183 | 0.078419 |
| train_rgb_diagnostic · 椅面 | 0.072140 | 0.056788 | 0.033406 |
| train_rgb_diagnostic · 木框 | 0.097848 | 0.099518 | 0.063441 |
| train_rgb_diagnostic · 底盘 | 0.178194 | 0.168262 | 0.176095 |
| train_rgb_diagnostic · 四材质bMAE | 0.129606 | 0.110938 | 0.087840 |

raw为未缩放主指标；scalar与RGB系数仅由各自105训练帧的GT albedo离线拟合，测试不参与、不反馈训练。前景/材质高Alpha MAE、相对亮度MAE及像素误差P95均完整保存在全量MAE明细，不能将总体改善解释为所有子指标改善。

## 法线、深度与覆盖率

| 区域/指标 | SH35k | IR55k | GT Lambertian55k |
| --- | --- | --- | --- |
| foreground · Normal角度MAE° | 7.402724 | 7.966735 | 7.402724 |
| foreground · Normal分量MAE | 0.059988 | 0.064498 | 0.059988 |
| foreground · Expected Depth MAE | 0.026094 | 0.030203 | 0.026094 |
| foreground · Median Depth MAE | 0.024407 | 0.028321 | 0.024407 |
| foreground · Normal覆盖率% | 99.814206 | 99.820440 | 99.814206 |
| interior · Normal角度MAE° | 6.723256 | 7.179154 | 6.723256 |
| interior · Normal分量MAE | 0.054773 | 0.058441 | 0.054773 |
| interior · Expected Depth MAE | 0.025236 | 0.029714 | 0.025236 |
| interior · Median Depth MAE | 0.023460 | 0.027666 | 0.023460 |
| interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 |
| cloth_foreground · Normal角度MAE° | 13.554440 | 14.188066 | 13.554440 |
| cloth_foreground · Normal分量MAE | 0.112759 | 0.117823 | 0.112759 |
| cloth_foreground · Expected Depth MAE | 0.031597 | 0.032161 | 0.031597 |
| cloth_foreground · Median Depth MAE | 0.029367 | 0.030108 | 0.029367 |
| cloth_foreground · Normal覆盖率% | 99.763398 | 99.800456 | 99.763398 |
| cloth_interior · Normal角度MAE° | 12.459933 | 12.831099 | 12.459933 |
| cloth_interior · Normal分量MAE | 0.103575 | 0.106674 | 0.103575 |
| cloth_interior · Expected Depth MAE | 0.026764 | 0.027390 | 0.026764 |
| cloth_interior · Median Depth MAE | 0.025872 | 0.026408 | 0.025872 |
| cloth_interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 |

Normal角度MAE单位度，XYZ分量为单位法线分量绝对误差。Depth是正相机Z、原世界单位，分别保存逐帧等权和像素加权，无尺度/位移/ICP对齐。条件MAE必须连同覆盖率解读，另保留漏预测180°惩罚及深度导出法线诊断。Stage1几何被冻结，实际重新评分并复核浮点数组。

## RGB与Alpha MAE

| 本轮浮点指标 | 逐帧等权 | 像素加权 |
| --- | --- | --- |
| rgb_srgb_full_mae | 0.00723373 | 0.00723373 |
| rgb_srgb_foreground_mae | 0.07617484 | 0.07580093 |
| rgb_linear_full_mae | 0.00766713 | 0.00766713 |
| rgb_linear_foreground_mae | 0.08166497 | 0.08202923 |
| alpha_full_mae | 0.00027430 | 0.00027430 |
| alpha_foreground_mae | 0.00159026 | 0.00154036 |

RGB与Alpha用浮点前向渲染评分，非PNG量化评分；sRGB夹到[0,1]后另解码线性RGB，前景mask为GT alpha>0.5，Alpha使用真实软RGBA。

## 真实代表图目检与质量限制

rgb：正式55k首尾正面红布比冒烟更接近GT，椅子和底盘可辨；测试0064背面布料/椅腿近黑，GT仍有照明，平台投影明暗与GT不同。

alpha：首中末主体轮廓、椅腿孔隙连续；冻结轮廓保持稳定。

normal：首中末椅面、褶皱和底盘方向可辨且连续；法线保持Stage1，正式未优化法线。

albedo：首尾布料偏红，仍保留褶皱明暗；背面黑区及红绿斑、木框颜色不均，底盘灰斑和阴影残留，未完全恢复纯材质。

转灯有明显亮暗变化，背向近黑符合无补光的直接Lambertian；HDR正面红布及粉橙底盘可辨，背面仍有斑块/明暗残留。HDR没有独立GT，仅检查功能。

![四类代表图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/four_categories.png)

![RGB与GT](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/rgb_comparison.png)

![重光照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/relight.png)

仅人工查看首/中/末与15测试帧中的0/7/14代表图；6视频完整解码属于自动检查，没有逐帧人工观看所有视频。HDR无独立GT，流程PASS不表示重建质量达标。

## 来源、完整命令与可复核结果

- Stage1：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/model_mlp`，35000步；source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/data_view`。
- 最终checkpoint：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/model_mlp/stage2_photometric/iteration_55000/checkpoint.pth`。
- [完整命令及日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/README.md)；[本组训练记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/训练/1006-06-only_clothV4-Stage2-GT灯位Lambertian训练记录.md)；[共用光强标定](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/shared_light_calibration.json)。
- [Albedo JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/evaluation/albedo/iteration_55000/metrics.json)；[几何JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/evaluation/geometry/metrics.json)；[RGB JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/model_mlp/renders_stage2/ours_55000/eval/metrics.json)；[RGB/Alpha MAE](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/evaluation/rgb_mae/metrics.json)；[Albedo SSIM](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/evaluation/albedo/ssim.json)。
- [所有MAE明细](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/all_mae.csv)；[180行逐帧对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/frames.csv)；[浮点重算及来源核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-10-V4四版本GT灯位Lambertian执行汇总/final_verification.json)。
- [自动产物验收](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/artifact_checks.json)；[真实目检记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/inspection/manual_review.json)。失败或历史日志保持原样，不覆盖删除。
