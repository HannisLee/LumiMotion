# 0816-03 CV3 坐标与 Lambertian 管线复审要点

- 日期：2026-08-16
- 服务器/环境：garuda / `lumimotion-garuda`
- 原目录：`output/0816-only_clothV3-coordinate-pipeline-audit`
- 类型：审计（非训练）
- 结论：坐标链路全部正确；推荐辐照度 `5.50435`

## 一句话总结

对 `Blender → transforms → LumiMotion loader → rasterizer` 的相机与世界坐标链路及 GT directional light 生成做全面复审：9 项数值闭环检查全部 PASS，未发现 world/camera 混用、左右手切换或全局轴翻转。

## 08.09 之后已修复的两个问题

1. Stage-1 insight 的 view-space normal 不再在 Python 端额外取反（CUDA rasterizer 已输出 camera-facing）
2. GT EXR normal 确认为 Blender 世界系；GT normal 评估器与光强标定不再误按相机系转换

## 数值闭环（120 帧，全部 PASS）

- `transforms.transform_matrix` vs 源 `camera_to_world`：误差 0
- 源 `camera_to_world @ world_to_camera - I`：`1.13e-6`（float32 精度）
- loader runtime W2C = `diag(1,-1,-1) @ Blender_W2C`：误差 0；相机中心重建误差 `3.55e-15`
- runtime right/down/forward 与导出基向量：`2.38e-7`；`fl_x/fl_y` 与 FOV：`1.11e-16` rad
- GT directional checkpoint 灯位与 `lights.json`：0；`light_to_surface` 光线重算：`9.35e-8`
- 相机约定：`S=diag(+1,-1,-1)`，`det(S)=+1`，纯 camera-local 变换，不改变任何世界坐标；初始点云质心与 GT checkpoint reference center 一致，4096 初始点 frame 1/60 全部在视锥内

## Renderer normal 链路

- CUDA：`tn = W_runtime @ R[:,2]`，按 `dot(-tn, p_view)` 翻到 camera-facing；Python Lambertian 端用 `dot(normal, camera_center - position)` 做同一符号选择，两式等价
- Lambertian `N·L` 全程世界系；GT 评估对比世界系 EXR；insight PNG 仅为可视化用的 runtime view normal

## GT pass 光强调标

| EXR normal 符号 | 推荐辐照度 | 前景平均 PSNR |
| --- | ---: | ---: |
| 匹配 2DGS camera-facing | `5.50435` | 16.318 dB |
| 保留 EXR signed | `5.50441` | 16.322 dB |

- 两种符号差异 < `0.004 dB` → 不存在“多翻一次 normal 导致 5.5 偶然拟合”；每帧拟合强度范围 `4.61–6.58`，即固定 directional 近似忽略 area-light 距离/入射角/阴影/曝光后的系统误差

## 仍存在的限制（非坐标问题）

- `gt_directional` 把移动 area light 压缩为对场景中心的单方向：无近距离方向差、距离衰减、area size、阴影
- normal/albedo/rotation/deformation 仍可共同吸收未建模成像误差；固定 GT light 不会自动使 rotation 收敛到 GT normal
- loader 未用 `cx/cy`（本数据主点恰好居中，无影响）；`camera_extent` fallback 未被 Blender reader 消费（本数据半径约 2.408，无退化）

## 产物

- `calibration_camera_facing/`、`calibration_signed/`（各含 `calibration.json` + `eval_contact_sheet.png`）及两个标定日志
- 复现命令：`scripts.calibrate_directional_light`（`--reference_center=-0.0003969318,0.1565740108,0.6233808994`）+ 4 个单元测试，见原 README
- 中文详细报告：`DOC/报告/08.16-only_clothV3坐标与Lambertian管线复审.md`
