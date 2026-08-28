# 数据来源、指标口径与图表映射

## 审计范围

本报告覆盖 2026-08-23（含）至 2026-08-24 的 14 份原始文档，并另记录本轮新增的评估器修改文档：

- 修改记录 6 份：`DOC/修改/0823-01-*`、`DOC/修改/0824-01-*` 至 `DOC/修改/0824-05-*`；
- 专题报告 2 份：`DOC/报告/08.24-CV3s-SH优于GT光Lambertian原因审计报告.md`、`DOC/报告/08.24-SH主干与Per-light方向融合方案报告.md`；
- 训练记录 6 份：`DOC/训练/0823-01-*`、`DOC/训练/0824-01-*` 至 `DOC/训练/0824-05-*`；
- 本轮新增修改记录：`DOC/修改/0824-06-eval_stage1_normals_gt双法线与余弦损失.md`。

其中 `0824-02-损失级光强归一化方案D设计（待实施）` 与 `08.24-SH主干与Per-light方向融合方案报告` 均为设计方案，尚无实现或训练结果，不能纳入实验效果排名。

## 法线与损失定义

- **independent normal**：Lambertian checkpoint 中单独学习的 `photometric_normal`，直接参与着色。
- **GS / 2DGS normal**：`rend_normal`，由 2DGS rotation、deformation 与 rasterization 决定，代表几何法线。
- **真实法线**：`data/LH-data/static/only_clothV3/normal_exr` 中 Blender world-space EXR normal。
- **GT cosine loss**：在 GT normal 有效且该 checkpoint 渲染 alpha ≥ 0.5 的像素上计算 `mean(1-cos(pred, GT))`；越低越好。
- **角误差**：同一有效像素集上的夹角（度）；报告 mean / median / P95，均越低越好。
- `summary_mean_over_frames` 是先在每帧内统计，再对 120 帧等权平均。报告中的 P95 是“逐帧 P95 的平均”，不是合并所有像素后的 pooled P95。
- train / test 为前 105 个训练相机与后 15 个测试相机各自逐帧 mean 的平均。
- 不同实验用各自 alpha mask，故跨实验有效像素数略有差异；同一 checkpoint 的 independent / GS 比较使用完全相同的 mask。

## 训练监督与离线评估边界

- 审计的 6 条正式训练命令均未传 `--photometric_gt_normal_dir` 或非零 `--lambda_photometric_gt_normal`；默认权重为 0。
- 现有 `photometric_gt_normal` 训练项只监督 independent normal；仓库没有“GS raster normal 对 GT normal”的训练损失。
- `lambda_gs_normal` 比较的是 `rend_normal` 与当前深度导出的 `surf_normal`，不是 GT normal。
- `photometric_normal_live` / `photometric_normal_mv` 是当前几何或跨视角自一致性，也不是 GT normal。
- 本轮补出的 cosine loss 与角误差仅是 iteration 35000 的离线审计指标，不进入反向传播，也未重训任何 checkpoint。

## 图表映射

| 图表 / 表格 | 数据字段 | 原始来源 | 变换 |
| --- | --- | --- | --- |
| 双法线 GT mean 角误差 | `independent_mean_deg`、`gs_mean_deg` | 各实验 `normal_gt_eval_dual_audit/<source>/ours_35000/normal_metrics.json` | 读取 `summary_mean_over_frames.mean_deg`；SH 无 independent，留空 |
| 法线明细 | cosine、mean、median、P95、train/test mean、有效像素 | 同上 | 直接读取；train/test 按 evaluator 的相机拼接顺序分组 |
| 实验总览 | RGB 五项、alpha、normal、结论 | 训练日志、实验 README、渲染统计、双法线 JSON | RGB 取训练日志记录的最佳 test iteration；normal 固定取 iteration 35000 |
| 损失配置 | GS/live/mv/GT 权重与启动步 | `run.sh`、`cfg_args`、训练文档 | 逐项核对，不把 preset 名称当作实际权重证据 |
| 文档与证据合规 | 命令、产物、状态、因果强度 | DOC 与实验 README | 依据仓库训练输出规定逐项审阅 |

未绘制训练趋势图：每条训练只有 7 个离散 test 节点，且本报告的核心问题是来源配对后的终点法线质量；把不等间隔、少于 8 点的序列画成趋势图容易造成连续轨迹错觉。精确节点仍保留在训练日志中。

## 主要可复核产物

- `output/0823-01-CV3s-GTlight_lambert_iter1_3ncons/normal_gt_eval_dual_audit/`
- `output/0824-01-CV3s-GTlight_lambert_iter1_2ncons/normal_gt_eval_dual_audit/`
- `output/0824-02-CV3s-GTlight_lambert_iter1_gswrapping/normal_gt_eval_dual_audit/`
- `output/0824-03-CV3s-SH_default2DGSnormal/normal_gt_eval_dual_audit/`
- `output/0824-04-CV3s-GTposition_perGSdir_3ncons/normal_gt_eval_dual_audit/`
- `output/0824-05-CV3s-GTposition_perGSdir_lambert_default/normal_gt_eval_dual_audit/`
- 各实验的 `train_stage1.log`、`README.md` 与全时序渲染目录；Lambertian 实验位于 `CV3s_stage1_mlp/renders_stage1_insights/ours_35000/`，SH 实验使用显式输出目录 `renders_stage1_insights/ours_35000/`。

## 审计限制

1. RGB 最佳值通常来自 iteration 30000，而法线、alpha 与目检固定使用 iteration 35000；同一行不代表同一 checkpoint 的联合最优。
2. 每种配置只有一个正式 seed，不能把一次分叉确定归因于“随机不稳定性”；`0824-02` 的高权重正反馈是有日志支持的机制解释，seed 敏感性仍待重复实验。
3. SH 与 Lambertian 同时改变外观模型、normal 表达、损失调度和时间颜色自由度，SH 的优势是观测事实，不是单变量因果结论。
4. CV3s 的 camera、time 与 light 一一绑定；当前 test 是同轨迹插值，不能证明 novel-light relighting 泛化。
5. “GT light”只使用 AREA 灯中心导出的方向；距离、面积采样、visibility、软阴影、间接光与完整材质未纳入 Lambertian 公式。
