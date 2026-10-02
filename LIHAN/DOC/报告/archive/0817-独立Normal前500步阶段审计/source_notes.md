# 数据来源与图表映射

## 图表映射

| 图表 | 数据字段 | 原始来源 | 变换 |
| --- | --- | --- | --- |
| 10001→10500 法线误差 | mean/median/P95 | 两个数据集各自的 `normal_metrics.json` | 对 120 帧逐帧指标取均值；10500 减 10001 |
| 冻结与漂移审计 | PLY、deform checkpoint、light_dirs、photometric checkpoint | iteration 10001 与 10500 checkpoint | 逐元素绝对差；normal 使用归一化后夹角 |
| 运行状态 | `train_stage1.log` 与 checkpoint 目录 | 两个实验目录 | 日志最后 iteration 与最新落盘 checkpoint |

## 口径

- 角误差是在有效 GT normal 且渲染 alpha ≥ 0.5 的像素上计算。
- GT Blender normal 与 renderer world-space normal 直接比较。
- `summary_mean_over_frames` 是先逐帧统计，再对 120 帧取算术平均，并非合并所有像素后统计。
- “改善帧”定义为 iteration 10500 的逐帧 mean 角误差严格小于 iteration 10001。
- SH baseline 的 PSNR/SSIM/LPIPS 是训练日志记录的最佳 test 指标，仅用于说明 RGB baseline，不等价于法线恢复质量。

## 可复核路径

- `output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10001/normal_metrics.json`
- `output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/normal_metrics.json`
- `output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10001/normal_metrics.json`
- `output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/normal_gt_eval_independent/ours_10500/normal_metrics.json`
- `output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal/train_stage1.log`
- `output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal/train_stage1.log`

## 再审新增来源

- `output/smoke_test/0817-06-CV3-directional-oracle-recheck/calibration.json`
- `output/smoke_test/0817-07-CV3L-directional-oracle-recheck/calibration.json`
- `output/smoke_test/0817-08-CV3-material-schedule/gradient_audit.jsonl`
- `scripts/train_stage1.py` 中的 `photometric_material_learning_rates` 与 `update_photometric_state`
