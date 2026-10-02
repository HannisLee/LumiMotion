# Stage 1 逐帧 Alpha 评估工具

日期：2026-08-29
服务器/环境：garuda / `lumimotion-garuda`

## 背景

`render_stage1_insights` 以一个固定测试相机遍历时间，适合观察时序外观，但静态 CV3 场景的相机随帧运动，不能把该固定相机输出直接与每帧训练 RGBA Alpha 做 IoU。为避免误判前景/背景分离，新增独立的逐相机、逐时间 Alpha 评估器。

## 新文件

[LIHAN/scripts/eval_stage1_alpha.py](../../LIHAN/scripts/eval_stage1_alpha.py)

该脚本不修改训练、渲染或数据加载管线。它加载指定 Stage 1 checkpoint，对 105 train + 15 test 相机按各自 `fid` 渲染 `rend_alpha`，并与 `Camera.gt_alpha_mask`（训练图片第 4 通道）逐像素比较，写入：

- `alpha_per_frame.csv`：逐帧 coverage、IoU、MAE、BCE、soft-edge MAE；
- `alpha_metrics.json`：微平均 IoU、precision、recall、F1 与覆盖率；
- `alpha_gt_pred_absdiff_contact_sheet.png`：首/中/末帧的 GT / 预测 / 绝对误差三列对照。

## 本轮验证

对 `0829-01-CV3-TDL-LambertianStage1` iteration 1200：

- Alpha micro IoU `0.98013`，precision `0.99761`，recall `0.98243`，F1 `0.98997`；
- 预测/GT 硬前景覆盖率 `19.716% / 20.020%`，仅轻微欠覆盖；
- 平均 Alpha MAE `0.005280`、BCE `0.010354`；
- 结论：Alpha 前景背景分离 **PASS**。

首次直接执行脚本时因其位于 `LIHAN/scripts`、仓库根目录未自动进入 `sys.path` 而退出；失败日志保留在实验目录 `eval_alpha_1200.log`。现已在脚本开头显式加入仓库根目录，重试日志为 `eval_alpha_1200_retry01.log`，结果通过。
