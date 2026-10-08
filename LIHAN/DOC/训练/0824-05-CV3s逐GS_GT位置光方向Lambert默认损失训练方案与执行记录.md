# 0824-05 CV3s 逐 GS GT 位置光方向 + Lambertian 默认损失训练方案与执行记录

日期：2026-08-24  
服务器 / 环境：`garuda` / `lumimotion-garuda`  
状态：完成，**FAILED**（训练、渲染与双法线 GT 评估均完成）

## 1. 目的

以已完成的 `0824-04-CV3s-GTposition_perGSdir_3ncons` 为直接对照，保持逐 Gaussian GT 光源位置方向、光强、优化器、调度和评估协议不变，只把损失组合从 `lambertian_normal3` 改为 `lambertian_default`。

本实验回答：在逐 GS 局部光方向下，去掉 independent normal 的 live / mv 自一致性后，RGB、independent normal、alpha 和几何稳定性如何变化。

## 2. 核心配置

| 项目 | 配置 |
|---|---|
| 数据集 | CV3s，`data/LH-data/transfer-static/only_clothV3` |
| 划分 | 105 train / 15 test |
| 渲染 | iteration 1 起 `photometric_lambertian` |
| 光源 | `gt_point_direction_only`，逐 GS 单位方向，无距离衰减 |
| 光强 | `5.5043499`，白光 |
| 损失 preset | `lambertian_default` |
| independent normal | 保留；iter 1 起，仅由 RGB 梯度优化 |
| independent normal 额外约束 | init=0、live=0、mv=0、GT normal=0 |
| GS normal / distortion | 0.02 / 1000，`start_normal_reg=500` |
| 训练长度 | 35000 iter |
| Gaussian 上限 | 20000 |

相对 `0824-04`，本实验只删除：

- `lambda_photometric_normal_live=0.01`；
- `lambda_photometric_normal_mv=0.02`。

`start_normal_reg=500` 按用户确认保留，因此 GS raster normal—depth normal 自一致性与 distortion 的启用时刻不变。

## 3. 输出

- 实验根目录：`output/0824-05-CV3s-GTposition_perGSdir_lambert_default`
- 模型：`CV3s_stage1_mlp`
- 训练脚本 / 日志：`run.sh` / `train_stage1.log`
- 后处理脚本：`post_train.sh`
- 全时序渲染：`CV3s_stage1_mlp/renders_stage1_insights/ours_35000`
- independent normal GT 评估：`normal_gt_eval_independent/ours_35000`
- 完整命令、渲染参数和验收记录：实验根目录 `README.md`

## 4. 启动与验收

训练在 GPU0 上持久运行，成功完成后自动执行 120 帧 RGB / alpha / normal / separation 渲染和 independent normal GT 评估。

验收必须回填：

1. RGB 五项指标；
2. independent normal mean / median / p95；
3. alpha 覆盖率与时序差；
4. RGB、alpha、normal、albedo / separation 四类目检；
5. 最终 `PASS` / `FAILED`。失败结果不得删除或覆盖。

## 5. 执行结果

- 35,000 iter 正常完成，训练约 32 分 54 秒；point cloud、deformation、photometric 三类 iteration 35000 checkpoint 完整。
- 最佳 RGB 位于 iteration 30000：PSNR 46.99108、SSIM 0.99852、LPIPS 0.00478、MS-SSIM 0.99950、Alex-LPIPS 0.00116。与 0824-04 的 46.96436 dB 基本持平。
- 梯度审计 75,528 条记录均为有限值；live / mv 按 preset 设计没有进入损失。
- 120 帧完整渲染、原自动 independent normal 评估和本轮显式 independent / GS 双法线评估均完成。

双法线相对 Blender world-space GT normal（alpha ≥ 0.5；120 帧逐帧统计后等权平均）：

| 法线源 | cosine loss | mean | median | p95 | train / test mean | 有效像素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| independent | 0.311187 | **41.8409°** | 42.2986° | 84.7740° | 41.8314° / 41.9079° | 5,533,315 |
| GS / 2DGS raster | 0.109623 | 20.1673° | 12.3391° | 69.5150° | 20.1660° / 20.1763° | 5,533,315 |

相对 0824-04（启用 live=0.01 / mv=0.02），independent mean 恶化 **+23.4831°**，而 GS mean 改善 **-0.9547°**。这说明 live / mv 对 independent shading normal 是必要约束，但没有证据表明它们改善 GS 几何 normal。

Alpha 覆盖率 mean 0.171057（0.168269–0.179735），帧间绝对差 4.45227e-4，略差于 0824-04 的 3.61412e-4。

## 6. 四类目检与结论

1. RGB：主体清晰、曝光稳定；末帧圆台下方有与 0824-04 / 0823 同型的拖影。
2. Alpha：主体轮廓连续，末帧底部拖尾可见；时序稳定性略差于 0824-04。
3. Normal：普通彩色 normal 图连续，但 GT error 图显示 independent normal 全主体高误差；GS error 图明显更暗，和定量结果一致。
4. Albedo / separation：主体完整、背景大体干净；末帧底部拖尾同步出现，没有 `gs_wrapping` 的大面积孔洞或噪声。

代表图片、视频、完整渲染命令与日志见实验 `README.md`；双法线数据位于 `normal_gt_eval_dual_audit/{independent,gs}/ours_35000/`。

最终结论：**FAILED**。训练和后处理完整、RGB 通过，但在没有 live / mv 与 GT normal 监督时，independent normal 退化到 mean 41.84°，不能作为可用的 independent-normal 配置。RGB 几乎无损而法线严重错误，也进一步证明当前单灯 RGB 存在强 albedo-normal 歧义。失败 checkpoint、渲染、统计与日志全部保留。
