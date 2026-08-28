# 0811-01 CV3 Lambertian 二档 alpha 修正训练要点

- 日期：2026-08-11 ~ 08-13
- 服务器/环境：garuda / `lumimotion-garuda`
- 原目录：`output/0811-only_clothV3-lambertian-modify`
- 状态：`FAILED`（Lambertian 验收无效；保留为失败对照，未覆盖）

## 一句话总结

按指导文档“第二档稳定策略”训练 CV3：RGBA soft alpha 前景监督、densify 1500–5000、5000 后固定 Gaussian 结构并禁用后段 prune-only。alpha 验收通过，但 08-13 审计确认本轮 Lambertian 渲染验收无效。

## 08-13 审计更正（关键）

- 训练命令使用 `gt_point + intensity=1`，经 `1/r²` 与 `/pi` 后真实 photometric RGB 理论上近黑。
- `render_stage1_insights` 未显式传 `--render_mode photometric_lambertian`，此前报告的“亮 RGB”实际来自默认 `original_sh`，不能作为 Lambertian 验收。
- 修正符号后的 120 帧 GT normal：mean `53.1946°`、median `40.4863°`、P95 `111.5535°`（`normal_gt_eval_signfix/ours_35000/`）。

## 训练与指标

- 命令要点：`--photometric_start_iter 10001 --photometric_light_mode gt_point --photometric_gt_light_intensity 1.0`，灯光 `data/LH-data/danamic/only_clothV3/lights.json`；日志 `train_stage1.log`
- 完成 35000 iter；最佳测试指标在 iter 10000：PSNR `40.67397`、SSIM `0.99425`、LPIPS `0.01708`、MS-SSIM `0.99850`、ALEX-LPIPS `0.01030`；5000 iter 点数 16223
- 渲染 120/120 帧完成：`lambertian-modify_mlp/renders_stage1_insights/ours_35000/`（RGB/alpha/normal/separation 图与视频齐全）

## 目检

- Alpha：`PASS` —— 首/中/末帧均为黑底，布料/椅子/底座前景分离，无明显漂浮 speckle（本轮主要目标）
- RGB（Lambertian）：无效 —— 实为 SH 输出，见上审计更正

## 后续影响

- 本 checkpoint 保留为失败对照；后继重训练 → `output/0813-only_clothV3-lambertian-directional-linear/`（见 `0813-01` 文档）
- 旧的 73k Gaussian 结果在 `output/0811-only_clothV3-lambertian-modify-before-alpha-fix-v2/`
