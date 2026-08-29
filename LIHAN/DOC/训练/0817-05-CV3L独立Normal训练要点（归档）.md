# 0817-05 CV3L 独立 Normal 训练（lr 1e-3）阶段验收要点（归档）

- 日期：2026-08-17
- 服务器/环境：garuda / `lumimotion-garuda`（GPU2）
- 原目录：`output/不重要/0817-05-CV3L-GTlight_i5p5_explicit_normal`
- 状态：`FAILED`（阶段验收），后归档至“不重要”

## 一句话总结

0817-04 的 CV3L 版本：数据 `data/LH-data/transfer-dynamic/only_clothV3_lambertian`（120 帧、105 train / 15 test，`lights.json` 120 条）；10001 起冻结原 GS/deformation/光，只训独立 normal 与 albedo（normal LR `1e-3`）。辐照度当时仍错误复用 CV3 的 `5.5`，是后来重做的原因之一。

## 执行状态

- 0817 03:56 启动；进程无 traceback 退出，日志停在约 iter 12320，最新持久化 checkpoint 10500
- 模型目录：`CV3L_explicit_normal_mlp`；日志 `train_stage1.log`

## 10001 → 10500 阶段评估（120 帧离线，GT 不参与训练）

| iter | mean | median | P95 |
| ---: | ---: | ---: | ---: |
| 10001 | 25.385° | 19.018° | 68.901° |
| 10500 | 28.148° | 24.231° | 68.491° |

- mean 增加 `2.762°`（+10.88%）；逐帧 0/120 改善、120/120 恶化；P95 虽微降 0.410°，不构成恢复
- 独立法线平均漂移 `14.896°`（P95 `33.175°`）；albedo raw mean abs change `0.1636`
- 冻结核验：position/rotation/scale/opacity/light 逐元素零变化，deformation checkpoint SHA 相同

## 目检与结论

- 独立法线跨视角系统性漂移；error contact sheet 无整体收缩；RGB 最佳指标仍是 iter 10000 SH baseline → 阶段结论 `FAILED`，全部产物保留
- 完整审计报告（与 CV3 版共用）：`DOC/报告/0817-独立Normal前500步阶段审计/README.md`
- 后续：CV3L 辐照度独立标定为 `7.8434867`，正式重做为 `output/0817-05-CV3L-GTlight_i7p8435_A500_Nonly_lr1e4`，见 `0817-05-CV3L固定光先Albedo后Normal训练要点.md`
