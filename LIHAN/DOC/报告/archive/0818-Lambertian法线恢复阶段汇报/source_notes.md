# 来源和指标口径

## 正式实验

- CV3：`output/0817-04-CV3-GTlight_i5p5_A500_Nonly_lr1e4`
- CV3L：`output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4`
- 训练日志：各目录下 `train_stage1*.log`
- 进程验收：各目录下 `process_monitor_0818.log`
- normal 指标：各目录下 `normal_gt_eval_independent/ours_<iteration>/normal_metrics.json`
- normal 图片：同目录下 `gs_normal_contact_sheet.png`、`gt_normal_contact_sheet.png`、`normal_error_contact_sheet.png`

normal 指标口径：Blender world-space GT normal 与 renderer world-space independent photometric normal 直接比较；使用 `alpha > 0.5` 的有效像素；PPT 中的 mean / median / P95 均为 `summary_mean_over_frames`。

| 数据集 | iteration | mean | median | P95 |
|---|---:|---:|---:|---:|
| CV3 | 10001 | 25.5981° | 15.8961° | 74.9210° |
| CV3 | 10500 | 25.5981° | 15.8961° | 74.9210° |
| CV3 | 35000 | 28.6903° | 22.6103° | 68.8086° |
| CV3L | 10001 | 24.8087° | 18.4963° | 65.8672° |
| CV3L | 10500 | 24.8087° | 18.4963° | 65.8672° |
| CV3L | 35000 | 31.0245° | 24.1541° | 81.1362° |

10001 和 10500 完全相同是调度预期行为：该 500 步 independent normal 的学习率为 0。

## 光强与 directional oracle

- CV3：`output/smoke_test/0817-06-CV3-directional-oracle-recheck/calibration.json`
- CV3L：`output/smoke_test/0817-07-CV3L-directional-oracle-recheck/calibration.json`
- CV3 推荐 irradiance：`5.5043498758`，foreground oracle PSNR：`16.318336 dB`
- CV3L 推荐 irradiance：`7.8434866773`，foreground oracle PSNR：`16.524032 dB`

directional 近似的空间方向误差来自坐标/光照审计报告：CV3 mean `6.19–7.58°`、P95 `10.46–12.89°`、max `14.39°`；CV3L mean `6.27–7.65°`、P95 `10.41–12.74°`、max `19.74°`。

## 曝光对比图

- I=1：`output/smoke_test/0818-01-CV3-gtpoint-I1-true-lambertian/full_t0_camframe_0008.png`
- I=5.5：`output/smoke_test/0818-02-CV3-gtdirectional-I5p5-true-lambertian/full_t0_camframe_0008.png`

该图不是只改光强的受控 A/B：同时包含 `gt_point → gt_directional` 和不同 checkpoint。它只用于证明“早期配置辐射量过低”与“修复配置达到可用曝光”，不用于将每一处像素差异归因于光强。

## 梯度与早期实验

- 梯度图：`FIG/0816-01-CV3-i5p5-gradient-audit`
- 报告：`DOC/报告/08.16-前500步梯度与法线崩坏审计报告.md`
- 独立 normal 早期审计：`DOC/报告/0817-独立Normal前500步阶段审计`
- 调度/损失复审：`DOC/报告/08.17.Stage1-Lambertian损失函数审计报告.md`

早期独立 normal 实验中，CV3 `23.719° → 27.245°`，CV3L `25.385° → 28.148°`，两者均为 `0/120` 帧改善。后续代码复审确认该阶段实际为 albedo + normal 联合更新，因此只作为发现调度 bug 的失败记录，不作为“分阶段训练失败”的证据。

## 历史 SH baseline

- CV3：PSNR `36.13052`，SSIM `0.98804`，LPIPS `0.02326`，MS-SSIM `0.99490`，Alex-LPIPS `0.01818`。
- CV3L：PSNR `36.43810`，SSIM `0.98963`，LPIPS `0.02492`，MS-SSIM `0.99579`，Alex-LPIPS `0.02290`。

SH baseline 用于说明数据加载和原 Stage 1 的 RGB 拟合基础，不能证明 Lambertian normal 可恢复。
