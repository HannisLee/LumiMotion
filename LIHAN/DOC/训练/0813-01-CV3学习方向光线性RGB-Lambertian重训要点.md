# 0813-01 CV3 学习方向光线性 RGB Lambertian 重训要点

- 日期：2026-08-13
- 服务器/环境：garuda / `lumimotion-garuda`
- 原目录：`output/0813-only_clothV3-lambertian-directional-linear`
- 状态：`FAILED`（辐射度/管线已修正；严格 Stage 1 几何与法线验收未过）

## 一句话总结

0811 的修正后继：保留其 alpha/增密配置，把 `gt_point` 换成每帧学习光方向 + 标定方向辐照度 `2.9`（后发现标定错误），并一并修正线性 RGB 管线与法线符号问题。

## 管线修正（本实验落地）

- sRGB albedo 先转线性 RGB 再做 Lambertian 着色；线性颜色 splat 合成，最后统一一次 sRGB 转换
- 方向 Lambertian 公式：`albedo_linear / pi * 2.9 * max(N·L, 0)`
- iter 10001 起冻结 xyz / scale / opacity / rotation / deformation（10001→35000 逐比特不变已验证），仅训练 albedo 与每帧光方向
- 移除 view-space normal 输出的多余第二次符号翻转

## 指标

| iter | 模式 | PSNR | SSIM | LPIPS | 点数 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10000 | original SH | 39.6992 | 0.993433 | 0.018635 | 16397 |
| 10001 | 线性 Lambertian | 21.4634 | 0.923506 | 0.081046 | 16397 |
| 30000 | 线性 Lambertian | **29.3265** | **0.968846** | 0.053527 | 16397 |
| 35000 | 线性 Lambertian | 28.4705 | 0.966528 | **0.053085** | 16397 |

- 切换点比 0811 切换（约 11.09 dB）高约 `10.4 dB`，且曝光正常（0811 为黑图）
- 最终 albedo mean `0.52518`；学习方向 vs Blender 灯位方向：30k `18.10°`、35k `17.34°`（仅诊断，训练不消费灯位）

## GT normal 与目检

- 几何冻结 → 30k/35k 法线完全相同：mean `75.709°`、median `70.928°`、P95 `127.738°`（`normal_gt_eval/ours_*/`）
- RGB：曝光色彩正常、布料运动连续，30k 与 35k 目视几乎无差
- Alpha：主轮廓稳定但前段椅子/布料区有漂浮 speckle；渲染覆盖 `0.17145` vs 输入 `0.20338`
- Normal：时序连续但数量上远离 GT；Separation：布料红（动）/椅台绿（静）稳定
- 渲染产物：`renders_stage1_insights/ours_30000_linear_directional/`、`ours_35000_linear_directional/`（120 帧 × 5 类 + 7 视频 + contact sheets）
- 单元测试 19 项全部通过

## 结论与后续

- `FAILED`：管线修正有效，但该基线几何无法提供准确法线初始化，alpha speckle 也不满足严格验收
- 保留 30000 作为“固定几何 photometric checkpoint”用于下一轮受控 normal/rotation 消融
- `2.9` 辐照度后被证实为错误标定（正确值约 `5.5`），见 `0813-02`、`0816-03` 文档
