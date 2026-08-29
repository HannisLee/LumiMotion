# 0824-04 SH 基线评估与梯度审计兼容修正

- 日期：2026-08-24
- 关联实验：`0824-03-CV3s-SH_default2DGSnormal`

## 修改内容

`scripts/eval_stage1_normals_gt.py` 原先虽然已实现 `gaussians.use_photometric_normal == False` 时的 GS raster normal 评估分支，但入口只允许 `photometric_lambertian`，因此无法加载没有 Lambertian renderer checkpoint 的 `original_sh` 模型。

现改为允许 `original_sh` 与 `photometric_lambertian`：

- `original_sh` 不创建或加载 Lambertian renderer，直接渲染并评估 `rend_normal`（默认 2DGS raster normal）。
- `photometric_lambertian` 保持原有 renderer 加载和 independent normal 评估行为。
- 不改变训练路径、loss、模型参数或已有 Lambertian 实验的 checkpoint 兼容性。

## 影响说明

这是离线评估兼容性修正，仅影响显式执行该评估脚本的 SH checkpoint。训练数值和已有训练结果不受影响。

## 冒烟中发现并修复的梯度审计问题

首次 SH 冒烟在 iteration 1、`--gradient_audit_interval 50` 时退出。原因是审计器无条件将 `_photometric_albedo` 放进 `torch.autograd.grad()` 的输入；`original_sh` 路径中该参数不需要梯度，PyTorch 会在 `allow_unused=True` 生效前直接拒绝输入。

现仅让 `requires_grad=True` 的参数参与该次 `autograd.grad()`，再将未参与的参数组按既有语义记录为零梯度。该修改只影响诊断 CSV 的取样，不改变损失、反向传播、优化器或 checkpoint。

失败冒烟的模型、TensorBoard event、梯度审计 CSV 与日志保留在 `output/smoke_test/0824-03-CV3s-SH_default2DGSnormal/`，重试会使用新的、不覆盖的目录。

## 后处理完成判据修正

实验目录的 `post_train.sh` 原先依赖训练日志中的 `Training complete.`。`--quiet` 会使该 stdout 标记不进入日志，即使 35,000 iter 已正常结束也会错误拒绝渲染与评估。

现改为同时检查 `point_cloud/iteration_35000` 与 `deform/iteration_35000` 目录存在。这只影响该实验的后处理启动门槛，不修改训练、模型或数值结果。
