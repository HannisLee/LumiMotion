# CV3 真实方向光 Lambertian Stage 1 冒烟训练

日期：2026-08-29
服务器/环境：garuda / `lumimotion-garuda`
状态：数据与管线 **PASS**；正式 Stage 1 验收 **FAILED / 未启动**

## 实验位置

- 实验目录：`output/smoke_test/0829-01-CV3-TDL-LambertianStage1`
- 数据：`data/LH-data/transfer-static/only_clothV3_true_direction_light`
- 灯光：`lights_compat_point_position.json`，只配合 `gt_directional` 使用；
- 模型：`CV3_TDL_stage1_mlp`，iteration 1200。

完整命令、日志、checkpoint、全时序渲染、Alpha/normal 指标和代表图见实验 [README](../../output/smoke_test/0829-01-CV3-TDL-LambertianStage1/README.md)。

## 结果

- 1200 iter 正常完成；最佳测试 PSNR `26.71736`（iteration 1000），SSIM `0.97241`，LPIPS `0.06148`；
- 真实方向通过虚拟 `light_pos_world` 接入旧 `gt_directional`，无训练/渲染核心代码改动；
- Alpha 专项：IoU `0.98013`、F1 `0.98997`，前景覆盖轻微欠 `0.304` 个百分点，**PASS**；
- independent normal：mean `52.34998°`、median `60.52157°`、P95 `77.39873°`，短训不通过；
- 真实 SUN 的 irradiance 暂以旧 CV3s `5.5043499` 试跑，尚未重新标定；固定相机时序 RGB 图偏白，不得作为正式光度质量结论。

## 下一步

在启动 35k 前，与用户核对：真实 SUN 数据的方向与量级标定、标准 staged-training 的 `photometric_start_iter=10001` / albedo-normal 冻结调度，以及正式验收阈值。当前所有失败/未验收产物均已保留，未覆盖或删除。
