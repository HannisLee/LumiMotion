# only_clothV4：LumiMotion 原始两阶段训练与评估

> 归档来源：本记录对应的训练或代码修改发生在原始项目 `/home/lihan/reproduce/LumiMotion`（`main` 分支），不是在 `/home/lihan/reproduce/LumiMotion-perlight` 中执行。2026-10-02 将记录复制到本仓库；1001-01 的完整产物已复制至本仓库 `output/1001-01-onlyclothV4-FOVxy修复重训/`。原始绝对路径和命令保留用于追溯。0930 输出未在本次复制范围内。

2026-10-01 更正：此次实验使用了互换的 Blender FOVx/FOVy。原 PASS 仅说明执行流程成功，不能作为正确投影下的训练验收；原指标和检查点保留供追溯，需要重训后重新评价。

## 目标与状态

在 `/home/lihan/data/LH-data/transfer-static/only_clothV4` 上执行原始 LumiMotion 两阶段流程：Stage 1 几何训练 35,000 iter、Stage 2 材质与环境光训练至 55,000 iter，以及阶段一诊断、材质渲染、HDR 重光照渲染和 NVS 定量评估。

当前状态：**已完成，PASS**。2026-09-30 12:43:36 至 14:09:02（Asia/Tokyo）完成训练、渲染和评估。启动时 GPU 0 的语音服务占用约 20.2GB 显存；服务停止并释放显存后，任务自动启动。

## 数据与兼容处理

- 原始数据：`/home/lihan/data/LH-data/static/only_clothV4`，含 120 张 RGB、albedo、normal 及相机数据。
- 训练数据：`/home/lihan/data/LH-data/transfer-static/only_clothV4`，含 105 个训练视角、15 个测试视角、RGBA 图像、`transforms_train.json`、`transforms_test.json` 和 `points3d.ply`。
- 运行数据视图：`output/0930-01-onlyclothV4-LumiMotion原始管线/data_view`。其中变换、点云和 RGB 均为软链接，训练光照目录 `chapel_day_4k_32x16_rot0` 与测试光照目录 `golden_bay_4k_32x16_rot330` 都链接到同一份采集 RGB。
- 数据不含原始 LumiMotion 需要的第二种 HDR 光照观测、roughness 真值或 HDR 文件。因此不运行依赖这些真值的材质/静态重光照量化脚本；NVS 量化仍会生成。HDR 重光照使用仓库示例 `example_envmaps/golden_bay_4k_32x16_rot330.hdr`，结果为定性可视化。

## 命令与输出

执行脚本：`output/0930-01-onlyclothV4-LumiMotion原始管线/run_pipeline.sh`。

模型路径：`output/0930-01-onlyclothV4-LumiMotion原始管线/model_mlp`。

完整命令保存在运行脚本中，并将分别执行：

1. `scripts.train_stage1`：35,000 iter，`resolution=2`、`depth_ratio=1.0`、`diffuse` 原始默认设置。
2. `scripts.render_stage1_insights`：载入 iter 35,000。
3. `scripts.train_stage2`：从 iter 35,000 续训到 iter 55,000，`diffuse_sample_num=512`、`depth_ratio=0.0`。
4. `scripts.render_materials`、`scripts.render_relight_with_hdr` 与 `scripts.eval_nvs_static`：载入 iter 55,000；NVS 使用 2,048 个漫反射采样。

每个命令的日志写入 `output/0930-01-onlyclothV4-LumiMotion原始管线/logs/`，检查点、渲染结果、指标 JSON 和 TensorBoard 文件均留在 `model_mlp/`。主日志为 `logs/pipeline.log`。

## 实际结果

| 项目 | 结果 |
| --- | --- |
| Stage 1 | iter 35,000 完成；测试集 PSNR 39.23、SSIM 0.99、LPIPS 0.02；训练集 PSNR 42.20、SSIM 1.00、LPIPS 0.02。 |
| Stage 2 | 从 iter 35,000 成功续训至 iter 55,000；iter 40,000、50,000、55,000 均保存了 point cloud、deform 和 envmap checkpoint。 |
| NVS | 15 个测试视角平均 PSNR **20.96**、SSIM **0.933**、LPIPS **0.066**；文件为 `model_mlp/results_nvs_static.json`。 |
| 输出 | Stage 1 七类诊断视频、120 帧材质渲染、120 帧 HDR 重光照及 MP4、15 组 NVS render/GT/mask 均已生成。 |

## 可视化目检

1. Stage 1 完整渲染：椅子和红色布料的主体轮廓稳定；底座有模糊和轻微浮纹。
2. 法线：主体法线连续，布料轮廓和底座边缘仍有局部噪声。
3. 二元分离：红绿点在整个场景混合，未出现清晰的语义静/动态分界。
4. 材质与 HDR 重光照：Albedo 保留物体主色，roughness 有噪声；Golden Bay HDR 渲染出现合理暖色照明，无崩溃和黑屏。

结论：**PASS**。原始两阶段管线和适用的 NVS 指标已完整执行。由于数据没有第二种 HDR 光照观测或 roughness 真值，材质与重光照结果仅作定性验收，不能视为 PBR 指标。
