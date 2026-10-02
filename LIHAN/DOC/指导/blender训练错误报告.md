# LumiMotion Blender only_cloth 训练错误报告

日期：2026-07-15  
仓库：`/home/han.li/reproduce/LumiMotion`

## 1. 问题摘要

对以下两组 Stage 1 实验进行了对比：

- 成功基线：`output/Baseline/0713-d-nerf-relight-spec32-stage1/hook150_v5_spec32_r2_mlp`
- 异常实验：`output/Baseline/0714-only_cloth-original-stage1/only_cloth_stage1_mlp`

综合结论：only_cloth 训练异常的主因不是初始化点云质量，也不只是 alpha mask，而是以下问题叠加：

1. Blender 非方形图像的 `FovX/FovY` 在数据读取时被交换，only_cloth 使用了错误相机投影。
2. only_cloth 所有帧使用同一个相机矩阵，基于相机中心计算的 `cameras_extent` 为 0，导致 densification 的尺度判定退化。
3. 原始转换数据的 alpha 全为 255，同时背景是灰色、renderer 背景是黑色，迫使模型生成覆盖整幅图像的高 opacity、大尺度 Gaussian。
4. only_cloth 的动态/静态二值特征几乎全部退化为静态，无法描述落布的大形变。
5. 点云初始化实际只使用 XYZ；点云颜色、法线以及动态信息没有参与 Gaussian 初始化。
6. 固定单视角数据缺乏三维约束，同视角图像指标不能反映几何是否已经训飞。

因此，继续单独调整 mask、学习率或初始化点数不能从根本上解决问题。必须优先修复相机内参和非零场景尺度。

## 2. 定量对比

| 项目 | hook | only_cloth |
| --- | ---: | ---: |
| 图像分辨率 | 800×800 | 1280×720 |
| 帧数 | 150 | 120 |
| 唯一相机矩阵数 | 150 | 1 |
| 相机中心跨度 | `[5.03, 5.56, 1.24]` | `[0, 0, 0]` |
| 初始点数 | 100,000 | 4,096 |
| 最终点数 | 126,970 | 112,827 |
| 点数增长倍率 | 1.27× | 27.55× |
| 最终坐标范围 | 大致位于 ±2 | x/y 达数十，z 达 143 |
| 最大 Gaussian scale | 0.335 | 3242.6 |
| 中位 opacity | 0.101 | 1.0 |
| 动态特征占比（sigmoid > 0.5） | 33.88% | 0.276% |
| 最终同视角 PSNR | 约 43–46 | 30.87 |

only_cloth 的最终绝对点数并没有明显超过 hook，但其点数从 4,096 增长到 112,827，并伴随坐标、尺度和 opacity 同时失控。因此“点很多”是训练退化的结果和信号，不是唯一根因。

## 3. 相机 FOV 读取错误

only_cloth 在 `camera.json` 中的正确内参为：

```text
resolution = 1280 × 720
fx = fy = 1564.4444
FovX = 0.776637
FovY = 0.452353
```

实际写入训练相机的数据为：

```text
fx = 2781.235
fy = 880.0
```

原因位于 `scene/dataset_readers.py` 的 Blender transforms 读取逻辑：

```python
fovy = focal2fov(fov2focal(fovx, image_width), image_height)
FovY = fovx
FovX = fovy
```

这里把 `FovX` 和 `FovY` 赋反了。对于 hook 的 800×800 方形图像，两个 FOV 相同，所以该错误被完全掩盖；对于 only_cloth 的 16:9 图像，错误非常明显：

- 水平焦距相对正确值放大到 1.778 倍。
- 垂直焦距缩小到正确值的 0.562 倍。

即使点云位置正确，投影到图像后也会产生显著的水平/垂直形变。优化器只能通过移动、拉伸、复制 Gaussian 来补偿错误投影。

正确行为应当是：

```python
FovX = fovx
FovY = fovy
```

更稳妥的实现是直接使用 transforms 中的 `fl_x`、`fl_y`、图像宽高分别计算两个 FOV，避免假设方形像素或只使用 `camera_angle_x`。

## 4. 固定相机导致 cameras_extent 为 0

hook 的 150 帧包含 150 个不同相机矩阵。only_cloth 的 120 帧全部使用相同的 `camera_to_world`。

`scene/dataset_readers.py::getNerfppNorm()` 使用相机中心到平均中心的最大距离作为场景 radius。only_cloth 的相机中心没有变化，因此：

```text
cameras_extent = 0
```

数据转换清单虽然记录了 `camera_extent=1.0`，但当前 Blender reader 没有读取该字段，而是无条件重新计算相机跨度。

训练阶段 densification 使用该 extent 区分 clone 和 split。当 extent 为 0 时，尺度阈值也退化为 0，正常的 clone/split 空间判定失去含义。这解释了为什么修正 alpha 后点数仍会快速增长：

- 完整 mask 实验：4,096 → 53,093（iteration 3,000）
- 另一轮 mask 修正实验：iteration 5,000 已达到 278,619 个点

固定相机数据必须显式提供正的 scene extent。建议以 canonical 点云包围盒或数据文件中的明确配置为准，而不是使用相机运动范围。

## 5. 原始 alpha 和背景监督错误

`0714-only_cloth-original-stage1` 使用 `image-alpha` 转换，但原始 `image/*.png` 的 alpha 恒为 255：

```text
透明像素占比 = 0%
完全不透明像素占比 = 100%
背景 RGB = [63, 63, 63]
```

训练使用 `white_background=False` 和 `gt_alpha_mask_as_scene_mask=True`。因此 renderer 使用黑色背景，但 RGB loss 要求输出灰色背景，alpha loss 又要求整幅图不透明。模型只能生成覆盖整幅画面的 Gaussian 来表示背景。

最终 only_cloth 点云出现：

```text
x: -46.46 到 58.41
y: -38.01 到 42.15
z: -70.19 到 143.30
最大 Gaussian scale: 3242.59
中位 opacity: 1.0
```

相比之下，hook 最终点云仍位于大约 ±2 的有效空间，最大 scale 约 0.335。

正确的 only_cloth mask 应使用 albedo PNG 自带的 soft alpha，或使用背景颜色分割；RGB 应按该 alpha 合成到与 renderer 一致的黑色背景。

但 mask 修正只能解决背景监督错误，不能修复错误 FOV、extent=0 和单视角不可辨识问题。

## 6. 动态/静态分离退化

最终 PLY 中 `fea_0` 的分布显示：

```text
hook:       33.88% 的点 sigmoid(feature) > 0.5
only_cloth:  0.276% 的点 sigmoid(feature) > 0.5
```

only_cloth 几乎退化成全静态模型。落布从高处移动到椅子并发生大形变时，全静态解会导致：

- 布料在部分时间消失。
- 多个时间状态叠加产生拖影。
- Gaussian 通过扩大尺度和漂移位置拟合同一相机的二维图像。
- 同视角 PSNR 仍可能较高，但三维结构错误。

配置差异会放大这一问题：

```text
hook lambda_separation       = 0.001
only_cloth lambda_separation = 0.005
```

only_cloth 中动态布料只占前景的一部分，而椅子和圆台长期静止。更强的 separation L1 正则容易推动模型选择“全部静态”的平凡解。

## 7. 初始化点云并没有被完整使用

only_cloth 的 `pointcloud.ply` 包含位置、颜色和法线，但 `GaussianModel.create_from_pcd()` 的实际行为是：

- 使用点的 XYZ。
- 读取颜色后，没有用它初始化 albedo；albedo 仍被初始化为全零。
- 不使用输入法线。
- 不使用速度、动态标签、canonical timestep 或物体对应关系。
- 初始 opacity 固定为 0.1。

因此“好的初始化”只表示某一个时刻的表面位置较准确，不代表网络获得了落布全过程的动态初始化。

hook 使用 100k 随机点覆盖 canonical 空间，并通过 150 个变化视角持续约束三维结构。only_cloth 虽然点位更贴近表面，但只有 4,096 点、单一视角，而且布料运动范围和形变很大，反而更容易陷入二维拟合解。

## 8. 为什么同视角指标仍然不低

only_cloth 最终同视角测试指标约为：

```text
PSNR = 30.87
SSIM = 0.973
```

但 train/test 都使用同一个相机，只是按时间帧切分。模型可以用错误的深度、巨大 Gaussian 或时间相关形变拟合该相机看到的二维图像。

因此当前 eval 只验证同视角的时间插值，不验证 novel-view 几何一致性、点云空间边界、Gaussian scale、动态/静态划分或 alpha 几何质量。

## 9. 完整 mask 重训的实际报错

最新 albedo soft-alpha 直训没有正常完成。在 iteration 3,100：

```text
Gaussian number BEFORE PRUNE 55463
Gaussian number AFTER PRUNE 0
```

随后 rasterizer 报错：

```text
RuntimeError: Function _RasterizeGaussiansBackward returned an invalid gradient
got [0, 0, 3] but expected shape compatible with [0, 16, 3]
```

这不是 alpha 文件再次错误，而是所有 Gaussian 在 opacity reset 后的下一次 pruning 中被删光。它进一步说明固定视角、extent=0、opacity/pruning 与 densification 的组合不稳定。

## 10. 修复优先级

建议严格按以下顺序处理，避免继续产生不可比较的实验：

1. 修正 Blender reader 中的 `FovX/FovY`，并优先使用 `fl_x/fl_y`。
2. 对固定相机数据显式设置正的 scene extent，并验证该值实际传入 Scene/GaussianModel。
3. 使用 albedo soft alpha，把 RGB 合成到黑色背景。
4. 训练前将初始点云投影到训练图像，保存 overlay，确认相机和点云坐标系一致。
5. 将 `lambda_separation` 恢复为 hook 的 0.001，并记录动态点比例。
6. 为 densification 增加最大点数、空间包围盒和最大 Gaussian scale 保护。
7. 修复 opacity reset 与 min-opacity pruning 的边界，禁止一次 pruning 删除全部点。
8. 增加点云统计：点数、XYZ 范围、scale 分位数、opacity 分位数、动态点比例。
9. 除同视角 PSNR/SSIM 外，增加 alpha、深度/法线或 novel-view 检查。

## 11. 建议验收标准

新的 only_cloth 实验至少应满足：

- 训练相机内参恢复为 `fx≈fy≈1564.44`（原分辨率下）。
- `cameras_extent` 为显式正值，而不是 0。
- 输入 alpha 前景覆盖率约 16.9%–25.1%，背景 alpha 为 0。
- 初始点云投影与图像主体轮廓对齐。
- densification 期间点数平稳增长，不出现单阶段数倍暴增。
- 点云 XYZ 始终处于合理场景包围盒内。
- Gaussian scale 不出现数十、数百或数千量级。
- pruning 后点数始终大于 0。
- 动态点比例不是接近 0 的退化解。
- alpha 视频具有稳定透明背景，布料运动不出现明显叠影或消失。

## 12. 最终判断

hook 能从较差随机初始化学好，是因为它具有正确的方形相机投影、150 个变化视角、非零场景尺度，以及更合理的动态分离约束。only_cloth 即使具有更好的表面点初始化，也同时遭遇错误内参、零场景尺度、固定单视角、错误背景 alpha 和全静态退化。

这些差异不是普通的超参数偏差，而是训练问题定义和几何约束已经发生变化。only_cloth 的点云爆炸正是这些问题共同作用后的结果。
