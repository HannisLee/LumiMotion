# 静态四组Stage2：原版IR与固定共用光强GT Lambertian汇总

更新：2026-10-06T17:52:09.446912+09:00。范围经用户确认：**1006-01、1006-02、1006-06、1006-07**，四组正式Stage2均为55k；附两份Stage1 35k参考。本次只汇总已完成评测。

GT Lambertian相对同来源IR，CV4原始Albedo MAE下降39.52%，CV4L下降30.94%；四材质bMAE分别下降32.90%、29.04%。两份数据的四种材质raw MAE都降低，但RGB全图/前景PSNR、SSIM均降低，LPIPS均升高。

GT Lambertian的法线/深度结果与各自Stage1 35k完全相同，不能解释为Stage2修复了几何。相对IR的几何误差更低，是两条优化流程保留/改变Stage1输出的结果。IR与Lambertian的成像模型、参数优化范围不同，属于管线比较。

## 四组对应关系与统一口径

| 实验 | 静态数据 | 管线 | Stage1来源 | 累计迭代 | Gaussian数 |
| --- | --- | --- | --- | --- | --- |
| 1006-01 | CV4 | 原版IR Stage2 55k | 1006-01 · 35k | 55k（新增20k） | 43,140 |
| 1006-06 | CV4 | GT Lambertian 55k | 1006-01 · 35k | 55k（新增20k） | 43,140 |
| 1006-02 | CV4L | 原版IR Stage2 55k | 1006-02 · 35k | 55k（新增20k） | 42,654 |
| 1006-07 | CV4L | GT Lambertian 55k | 1006-02 · 35k | 55k（新增20k） | 42,654 |

CV4为静态原始材质场景，CV4L为静态Lambertian材质场景；数据名中的lambertian描述GT材质，训练管线另列。两份Stage1均从原版10万点随机方块初始化。每组105训练/15测试帧，测试8、16、…、120，640×360，黑背景。灯位转场景中心方向的GT Lambertian近似固定共用光强6.030241533058244，只优化albedo，LR0.001；几何、opacity、法线和形变冻结。现有近似没有面积光积分、距离衰减、投射阴影、镜面或间接光。

## 主指标

| 实验 | 数据 | Albedo MAE↓ | bMAE↓ | Normal MAE°↓ | Expected Depth MAE↓ | RGB SSIM↑ | RGB前景SSIM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1006-01 | CV4 | 0.228857 | 0.131870 | 7.966735 | 0.030203 | 0.985574 | 0.907423 |
| 1006-06 | CV4 | 0.138418 | 0.088483 | 7.402724 | 0.026094 | 0.973556 | 0.830631 |
| 1006-02 | CV4L | 0.198856 | 0.117014 | 8.091242 | 0.031638 | 0.985740 | 0.909815 |
| 1006-07 | CV4L | 0.137338 | 0.083037 | 7.579862 | 0.027284 | 0.976506 | 0.850182 |

Albedo主值为未缩放线性RGB，在GT alpha≥0.99且内缩2像素的前景上按像素汇总；bMAE为红布/椅面/木框/底盘四材质等权。Normal为世界法线有向角度MAE，Depth为正相机Z、原世界单位，主值逐帧等权，不作尺度或ICP对齐。RGB前景SSIM采用遮罩后包围框口径。黑背景使全图RGB指标高于前景指标，应同时解读。

![静态四组主指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/figures/static_stage2_comparison.png)

## 材质区域与尺度诊断

| 指标/区域 | 1006-01 CV4 | 1006-06 CV4 | 1006-02 CV4L | 1006-07 CV4L |
| --- | --- | --- | --- | --- |
| raw · 全前景 | 0.228857 | 0.138418 | 0.198856 | 0.137338 |
| raw · 红布 | 0.084398 | 0.078872 | 0.070979 | 0.063583 |
| raw · 椅面 | 0.051056 | 0.034063 | 0.044822 | 0.023841 |
| raw · 木框 | 0.074645 | 0.064024 | 0.078068 | 0.062447 |
| raw · 底盘 | 0.317382 | 0.176972 | 0.274187 | 0.182277 |
| raw · 四材质bMAE | 0.131870 | 0.088483 | 0.117014 | 0.083037 |
| train_scalar_diagnostic · 全前景 | 0.145136 | 0.137758 | 0.134029 | 0.137260 |
| train_scalar_diagnostic · 红布 | 0.119551 | 0.078417 | 0.089918 | 0.063547 |
| train_scalar_diagnostic · 椅面 | 0.056775 | 0.033413 | 0.046369 | 0.023777 |
| train_scalar_diagnostic · 木框 | 0.099799 | 0.063424 | 0.097832 | 0.062366 |
| train_scalar_diagnostic · 底盘 | 0.168081 | 0.176095 | 0.161461 | 0.182162 |
| train_scalar_diagnostic · 四材质bMAE | 0.111051 | 0.087837 | 0.098895 | 0.082963 |
| train_rgb_diagnostic · 全前景 | 0.145162 | 0.137757 | 0.133810 | 0.137259 |
| train_rgb_diagnostic · 红布 | 0.119183 | 0.078419 | 0.090057 | 0.063544 |
| train_rgb_diagnostic · 椅面 | 0.056788 | 0.033406 | 0.046382 | 0.023782 |
| train_rgb_diagnostic · 木框 | 0.099518 | 0.063441 | 0.097596 | 0.062354 |
| train_rgb_diagnostic · 底盘 | 0.168262 | 0.176095 | 0.161102 | 0.182163 |
| train_rgb_diagnostic · 四材质bMAE | 0.110938 | 0.087840 | 0.098784 | 0.082961 |

CV4L在训练帧scalar诊断下，全前景MAE由IR的0.134029变成0.137260，略有上升；两份数据在scalar/RGB尺度诊断下的底盘MAE均高于IR。因此“raw改善”不等于所有尺度口径和材质都改善。尺度系数只用105训练帧GT albedo离线拟合，不使用测试帧，也不反馈训练；共用光强标定则只用训练RGB，不用GT albedo。

## RGB拟合与实际测试帧

| 实验 | 全图PSNR↑ | 全图SSIM↑ | 全图LPIPS↓ | 前景PSNR↑ | 前景包围框SSIM↑ | 前景包围框LPIPS↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 1006-01 | 34.886119 | 0.985574 | 0.018809 | 24.621293 | 0.907423 | 0.108937 |
| 1006-06 | 29.933975 | 0.973556 | 0.025162 | 19.794762 | 0.830631 | 0.131331 |
| 1006-02 | 35.319972 | 0.985740 | 0.017554 | 25.074959 | 0.909815 | 0.101091 |
| 1006-07 | 30.329839 | 0.976506 | 0.023811 | 20.202511 | 0.850182 | 0.123809 |

![GT及三阶段RGB对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/figures/static_rgb_comparison.png)

实际代表帧为0008、0064、0120，分别显示GT、SH35k参考、IR55k与GT Lambertian55k。两条Stage2均有背面或材质残余光照问题；GT Lambertian背面过暗、底盘阴影/灰斑和布料褶皱明暗残留，在图中保留。

## 法线、深度与覆盖率

| 指标/区域 | 1006-01 CV4 | 1006-06 CV4 | 1006-02 CV4L | 1006-07 CV4L |
| --- | --- | --- | --- | --- |
| foreground · Normal角度MAE° | 7.966735 | 7.402724 | 8.091242 | 7.579862 |
| foreground · Normal XYZ分量MAE | 0.064498 | 0.059988 | 0.065627 | 0.061483 |
| foreground · Normal漏预测惩罚角度MAE° | 8.288087 | 7.736776 | 8.452430 | 7.909888 |
| foreground · Expected Depth MAE | 0.030203 | 0.026094 | 0.031638 | 0.027284 |
| foreground · Median Depth MAE | 0.028321 | 0.024407 | 0.030161 | 0.025395 |
| foreground · Normal覆盖率% | 99.820440 | 99.814206 | 99.797060 | 99.817011 |
| foreground · Expected Depth覆盖率% | 99.820440 | 99.814206 | 99.797060 | 99.817011 |
| interior · Normal角度MAE° | 7.179154 | 6.723256 | 7.338797 | 6.938596 |
| interior · Normal XYZ分量MAE | 0.058441 | 0.054773 | 0.059824 | 0.056516 |
| interior · Normal漏预测惩罚角度MAE° | 7.179154 | 6.723256 | 7.338797 | 6.938596 |
| interior · Expected Depth MAE | 0.029714 | 0.025236 | 0.031373 | 0.026597 |
| interior · Median Depth MAE | 0.027666 | 0.023460 | 0.029763 | 0.024566 |
| interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| interior · Expected Depth覆盖率% | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| cloth_foreground · Normal角度MAE° | 14.188066 | 13.554440 | 14.263895 | 13.616663 |
| cloth_foreground · Normal XYZ分量MAE | 0.117823 | 0.112759 | 0.118566 | 0.113055 |
| cloth_foreground · Normal漏预测惩罚角度MAE° | 14.573042 | 14.009765 | 14.744643 | 14.049191 |
| cloth_foreground · Expected Depth MAE | 0.032161 | 0.031597 | 0.032935 | 0.031342 |
| cloth_foreground · Median Depth MAE | 0.030108 | 0.029367 | 0.030096 | 0.029031 |
| cloth_foreground · Normal覆盖率% | 99.800456 | 99.763398 | 99.742018 | 99.769099 |
| cloth_foreground · Expected Depth覆盖率% | 99.800456 | 99.763398 | 99.742018 | 99.769099 |
| cloth_interior · Normal角度MAE° | 12.831099 | 12.459933 | 12.814267 | 12.506744 |
| cloth_interior · Normal XYZ分量MAE | 0.106674 | 0.103575 | 0.106660 | 0.103798 |
| cloth_interior · Normal漏预测惩罚角度MAE° | 12.831099 | 12.459933 | 12.814267 | 12.506744 |
| cloth_interior · Expected Depth MAE | 0.027390 | 0.026764 | 0.029014 | 0.026848 |
| cloth_interior · Median Depth MAE | 0.026408 | 0.025872 | 0.026570 | 0.025547 |
| cloth_interior · Normal覆盖率% | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| cloth_interior · Expected Depth覆盖率% | 100.000000 | 100.000000 | 100.000000 | 100.000000 |

覆盖率必须与条件MAE一起解读；漏预测法线使用180°惩罚。Alpha/normal/expected/median浮点数组与Stage1的逐像素差值均为0，复核记录见来源核验。Stage1基色为含光照的SH DC外观代理，不代表已恢复物理albedo。

| 只读Stage1参考 | Albedo代理MAE | bMAE | Normal MAE° | Expected Depth MAE | Median Depth MAE | 前景SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| 1006-01 CV4 | 0.192875 | 0.129177 | 7.402724 | 0.026094 | 0.024407 | 0.995041 |
| 1006-02 CV4L | 0.177255 | 0.114688 | 7.579862 | 0.027284 | 0.025395 | 0.995261 |

## 已保存的浮点RGB与Alpha MAE

| 实验 | 指标 | 逐帧等权 | 像素加权 |
| --- | --- | --- | --- |
| 1006-06 | rgb_srgb_full_mae | 0.00723373 | 0.00723373 |
| 1006-06 | rgb_srgb_foreground_mae | 0.07617484 | 0.07580093 |
| 1006-06 | rgb_linear_full_mae | 0.00766713 | 0.00766713 |
| 1006-06 | rgb_linear_foreground_mae | 0.08166497 | 0.08202923 |
| 1006-06 | alpha_full_mae | 0.00027430 | 0.00027430 |
| 1006-06 | alpha_foreground_mae | 0.00159026 | 0.00154036 |
| 1006-07 | rgb_srgb_full_mae | 0.00677089 | 0.00677089 |
| 1006-07 | rgb_srgb_foreground_mae | 0.07121743 | 0.07091532 |
| 1006-07 | rgb_linear_full_mae | 0.00739697 | 0.00739697 |
| 1006-07 | rgb_linear_foreground_mae | 0.07875140 | 0.07912819 |
| 1006-07 | alpha_full_mae | 0.00027389 | 0.00027389 |
| 1006-07 | alpha_foreground_mae | 0.00160918 | 0.00156084 |

1006-06/07使用浮点前向渲染评分：sRGB夹到[0,1]后另解码线性RGB，前景mask为GT alpha>0.5，Alpha为真实软RGBA。1006-01/02既有同协议RGB评测未保存这套浮点RGB/Alpha MAE，报告保留缺项，不用PNG量化结果补作直接对照；其Albedo、法线及深度MAE完整。

## 正式代表图目检记录

### 1006-06 CV4

rgb：正式55k首尾正面红布比冒烟更接近GT，椅子和底盘可辨；测试0064背面布料/椅腿近黑，GT仍有照明，平台投影明暗与GT不同。

alpha：首中末主体轮廓、椅腿孔隙连续；冻结轮廓保持稳定。

normal：首中末椅面、褶皱和底盘方向可辨且连续；法线保持Stage1，正式未优化法线。

albedo：首尾布料偏红，仍保留褶皱明暗；背面黑区及红绿斑、木框颜色不均，底盘灰斑和阴影残留，未完全恢复纯材质。

### 1006-07 CV4L

rgb：正式55k首尾正面红布接近GT但颜色仍有差异，主体清楚；测试0064背面布料/椅腿明显偏暗，平台阴影与GT不一致。

alpha：首中末椅子与底盘轮廓和孔隙稳定；冻结轮廓保持。

normal：布料、椅面与底盘世界法线连续且可区分；本轮保持Stage1法线。

albedo：首尾红布仍带褶皱阴影；背面存在大面积黑区和红灰色块，木框/底盘有明显明暗残留，材质分离不完全。

四组此前训练流程/产物已验收；本次重新查看汇总对照图。视频完整解码与人工代表帧目检分别记录，不声称逐帧观看全部视频。流程PASS不等于物理材质质量达标。

## 可复核产物

- [1006-01 CV4 完整训练/渲染记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-01-only_clothV4-LumiMotion原版随机方块两阶段/README.md)。
- [1006-06 CV4 完整训练/渲染记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-06-only_clothV4-Stage2-GT灯位Lambertian/README.md)。
- [1006-02 CV4L 完整训练/渲染记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-02-only_clothV4_lambertian-LumiMotion原版随机方块两阶段/README.md)。
- [1006-07 CV4L 完整训练/渲染记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-07-only_clothV4_lambertian-Stage2-GT灯位Lambertian/README.md)。

[四组主指标CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/metrics.csv)；[各种MAE及覆盖率明细](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/all_mae.csv)；[60行逐帧对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/frames.csv)；[来源与报告核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/verification.json)；[SVG对照图](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/figures/static_stage2_comparison.svg)。

## 交互报告与核验范围

[交互报告预览](http://127.0.0.1:4188/)；[可独立打开的离线HTML](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-12-静态四组Stage2-IR与GT-Lambertian汇总/report-app/.data-app-offline/exports/static-four-report.html)。四组数据、592项MAE/覆盖率明细及两份Stage1参考已绑定来源。构建、快照身份和来源SHA256复核通过；科学图与RGB接触图已实际查看。当前会话无可调用的浏览器检查工具，交互页面视觉与控件目检未执行。
