# 0817-04 CV3 独立 Normal 训练（lr 1e-3）阶段验收要点（归档）

- 日期：2026-08-17
- 服务器/环境：garuda / `lumimotion-garuda`（GPU0）
- 原目录：`output/不重要/0817-04-CV3-GTlight_i5p5_explicit_normal`
- 状态：`FAILED`（阶段验收），后归档至“不重要”

## 一句话总结

独立法线首版方案：1–10000 走原 SH Stage-1；10001 由当前 GS rotation 初始化独立 canonical normal，之后冻结 GS 位置/旋转/缩放/不透明度、deformation 与光，只训练独立 normal + albedo（normal LR `1e-3`，无 GT-normal loss）。前 500 步法线误差即系统性恶化。

## 执行状态

- 0817 03:56 启动；进程随后无 traceback 退出，日志停在约 iter 13063，最新持久化 checkpoint 为 10500
- 模型目录：`CV3_explicit_normal_mlp`；日志 `train_stage1.log`

## 10001 → 10500 阶段评估（120 帧离线，GT 不参与训练）

| iter | mean | median | P95 |
| ---: | ---: | ---: | ---: |
| 10001 | 23.719° | 15.652° | 67.632° |
| 10500 | 27.245° | 20.130° | 69.469° |

- mean 增加 `3.526°`（+14.87%）；逐帧 0/120 改善、120/120 恶化
- 独立法线平均漂移 `15.653°`（P95 `35.182°`）；albedo raw mean abs change `0.1720`
- 冻结核验：position/rotation/scale/opacity/light 逐元素零变化，deformation checkpoint SHA 相同

## 目检与结论

- 独立法线跨视角系统性漂移；error contact sheet 无整体收缩；RGB 最佳指标仍是 iter 10000 的 SH baseline → 阶段结论 `FAILED`，全部失败产物保留
- 完整审计报告：`DOC/报告/0817-独立Normal前500步阶段审计/README.md`
- 被同编号正式实验取代（normal LR 降至 `1e-4`、先 albedo 后 normal 两阶段、辐照度用标定值）：见 `0817-04-CV3固定光先Albedo后Normal训练要点.md`
