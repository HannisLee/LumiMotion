# 0816-02 CV3 固定几何仅 Albedo 消融要点（归档）

- 日期：2026-08-16 ~ 08-17
- 服务器/环境：garuda / `lumimotion-garuda`
- 原目录：`output/不重要/0816-02-CV3-i5p5-fixed-geometry-albedo-only`
- 状态：`ARCHIVED / 不重要`（2026-08-17 降级归档；该实验不能检验法线恢复能力）

## 一句话总结

0816-04 自由几何实验的对照消融：相同初始化、GT directional 5.5，但冻结 rotation/position/scale/opacity/deformation 并关闭 densification/opacity reset，仅训练 albedo 500 步，用于定位早期法线崩坏的来源。

## 审计更正（关键）

- 2DGS 法线由 rotation 与 deformation 决定，二者一并冻结后法线不变是**代码定义上的必然**：本实验只能确认冻结开关生效，不构成法线恢复性或因果性证据。

## 定量

- RGB：PSNR `14.68 → 15.59`（iter 1 → 500）
- 法线：iter 1 与 500 完全一致 —— mean `23.23775°`、median `4.57464°`、P95 `90.00106°`
- PLY 变化：position/rotation/scale/opacity 最大绝对变化均为 0；albedo raw 最大变化 `0.52183`

## 与自由几何组对照（iter 500）

| 项目 | 自由几何 | 固定几何仅 albedo | 差异 |
| --- | ---: | ---: | --- |
| PSNR | 19.77930 dB | 15.58754 dB | 固定组低 4.19 dB |
| 法线 mean | 50.25454° | 23.23775° | 固定组好 27.02° |

- 对照证实：早期法线崩坏来自几何参数更新；对比图 `FIG/不重要/0816-02-CV3-i5p5-fixed-geometry-albedo-only/fixed_vs_free_psnr_normal.png`

## 四类可视化（全部 FAILED）

- RGB：整体很暗、纹理颗粒状，仅训 albedo 无法补偿近场光照/覆盖/动态误差
- Alpha：渲染覆盖仅 `0.07889`（输入 `0.20338`）；120 帧时序差异 ~`7.6e-7`，几乎完全静止
- Normal：稳定但无恢复（仍 23.24°，P95 ~90°）
- Separation：冻结且训练不足 1000 步，近乎全黑

## 结论与后续

- 该配置不应扩展到 35000 步，也不作为主方案；下一轮须保留可学习的独立 normal/rotation 自由度，同时冻结 position/scale/opacity 等补偿通道
- 完整报告：`DOC/报告/08.16-固定几何仅Albedo训练对照报告.md`
- 产物：checkpoint `point_cloud/iteration_{1..500}`、渲染 `only_clothV3_i5p5_fixed_geometry_albedo_only_mlp/renders_stage1_insights/ours_500/`、日志 `train_stage1.log` / `normal_eval_{1,500}.log`
