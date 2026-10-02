# 0824-03 CV3s：SH 默认 2DGS normal 基线训练与对照方案

- 日期：2026-08-24
- 服务器/环境：`garuda` / `lumimotion-garuda`
- 状态：实验 A 已完成，**PASS**（2026-08-24）；实验 B 按用户要求未启动。
- 数据集：`data/LH-data/transfer-static/only_clothV3`（CV3s，105 训练帧 / 15 测试帧）

## 1. 两组实验的解释

当前实现中，`original_sh` 使用 SH 颜色与 2DGS rotation/deformation 导出的默认几何 normal，**没有 independent normal 参数**；`photometric_lambertian` 会创建 independent `photometric_normal`，而 `lambertian_normal3` 会在它上面启用 live 和 multi-view normal consistency，并保留 2DGS 几何 normal 一致性。

因此本草案将“两组”定义为：

| 实验 | 颜色管线 | normal 表达 / 损失 | 目的 |
| --- | --- | --- | --- |
| A | SH (`original_sh`) | 无 independent normal；默认 2DGS normal；`sh_baseline` | SH 默认颜色与默认损失基线 |
| B | Lambertian | independent normal；`lambertian_normal3` | independent normal 的三法线一致性方案 |

此对照同时改变颜色管线与损失，不能单独归因于 independent normal。若要严格消融 independent normal，需要另加“Lambertian、无 independent normal”的第三组；当前训练入口没有这个开关，故本次不擅自改代码或扩大范围。

## 2. 共同控制变量

| 项目 | 取值 |
| --- | --- |
| 训练 / 评测 | `--eval`、`--resolution 2`、`--gt_alpha_mask_as_scene_mask` |
| 预算 | 35,000 iter，warm-up 500 |
| densify | 1500–5000，每 200 iter，阈值 0.0004，最多 20,000 Gaussians |
| 公共项 | separation=0.005，d_xyz=0.001，d_color=0.01，depth_ratio=1.0 |
| 记录节点 | test: 500/1000/5000/10000/20000/30000/35000；save: 499 及上述节点 |
| 监督 | 不使用 GT normal 训练监督；仅在训练完成后用 EXR normal 离线评估 |

## 3. 实验 A：SH 默认颜色 / 默认 2DGS normal

- 实验目录：`output/0824-03-CV3s-SH_default2DGSnormal/`
- 冒烟目录：`output/smoke_test/0824-03-CV3s-SH_default2DGSnormal/`
- 颜色：`--render_mode original_sh`，SH degree=3（默认）。
- 损失：`--loss_preset sh_baseline`。保持默认 `lambda_gs_normal=0.02`、`lambda_dist=1000`、`start_normal_reg=8000`；不提前到 500。
- normal：仅 `rend_normal`，不出现 `photometric_normal` 参数组或 `photometric_normal_*` loss。

正式训练命令草案：

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n lumimotion-garuda python -u -m scripts.train_stage1 \
  --source_path data/LH-data/transfer-static/only_clothV3 \
  --model_path output/0824-03-CV3s-SH_default2DGSnormal/CV3s_stage1 \
  --train_light_folder images --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 --iterations 35000 --warm_up 500 \
  --densify_from_iter 1500 --densify_until_iter 5000 --densification_interval 200 \
  --densify_grad_threshold 0.0004 --opacity_reset_interval 3000 --min_opacity 0.01 \
  --prune_from_iter -1 --max_gaussians 20000 --binarization_warm_up 1000 \
  --lambda_separation 0.005 --d_xyz_loss_weight 0.001 --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 --render_mode original_sh --loss_preset sh_baseline \
  --gradient_audit_interval 25 \
  --test_iterations 500 1000 5000 10000 20000 30000 35000 \
  --save_iterations 499 500 1000 5000 10000 20000 30000 35000 --quiet
```

冒烟为 1200 iter。由于默认 GS normal/distortion 门控在 8000，冒烟只验数据、SH 训练、数值稳定、checkpoint 和没有 independent normal；不要求 `normal` / `distortion` 项非零。完训后使用 `--render_mode original_sh` 渲染，再用 `eval_stage1_normals_gt` 评估默认 GS normal。

## 4. 实验 B：Lambertian independent normal / 三法线

- 实验目录：`output/0824-04-CV3s-Lambertian_indnormal3/`
- 冒烟目录：`output/smoke_test/0824-04-CV3s-Lambertian_indnormal3/`
- 颜色：`photometric_lambertian`，从 iter 1 起启用；CV3s 的 `lights.json` 固定 GT directional 光，白光强度 5.5043499。
- normal：独立 `photometric_normal`，iter 1 由 2DGS normal 初始化，LR=1e-4。
- 损失：`--loss_preset lambertian_normal3`，其中 GS normal=0.02（iter 500 后），live=0.01（iter 500 后），multi-view=0.02（iter 1000 后、2000 iter ramp）；GT normal 仅离线评估。

正式训练命令草案：

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n lumimotion-garuda python -u -m scripts.train_stage1 \
  --source_path data/LH-data/transfer-static/only_clothV3 \
  --model_path output/0824-04-CV3s-Lambertian_indnormal3/CV3s_stage1 \
  --train_light_folder images --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 --iterations 35000 --warm_up 500 \
  --densify_from_iter 1500 --densify_until_iter 5000 --densification_interval 200 \
  --densify_grad_threshold 0.0004 --opacity_reset_interval 3000 --min_opacity 0.01 \
  --prune_from_iter -1 --max_gaussians 20000 --binarization_warm_up 1000 \
  --lambda_separation 0.005 --d_xyz_loss_weight 0.001 --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 --start_normal_reg 500 \
  --render_mode photometric_lambertian --photometric_start_iter 1 \
  --photometric_light_mode gt_directional \
  --photometric_gt_lights_path data/LH-data/static/only_clothV3/lights.json \
  --photometric_gt_light_intensity 5.5043499 --photometric_gt_light_color 1.0,1.0,1.0 \
  --photometric_albedo_lr 0.001 --photometric_normal_lr 0.0001 \
  --photometric_normal_start_iter 1 --lambda_photometric_normal_init 0.0 \
  --loss_preset lambertian_normal3 --gradient_audit_interval 25 \
  --test_iterations 500 1000 5000 10000 20000 30000 35000 \
  --save_iterations 499 500 1000 5000 10000 20000 30000 35000 --quiet
```

冒烟为 1200 iter，临时把 `photometric_normal_mv_ramp_iters` 设为 200 以验证完整路径；正式训练保持 2000。验收要求：independent normal 参数组已创建，GS `normal` 与 live 在 500 后、multi-view 在 1000 后均被审计到，且无 NaN/Inf、三类 checkpoint 完整。完训后以 `--render_mode photometric_lambertian` 渲染；normal 评估器会自动评估 independent normal。

## 5. 产物与启动门槛

每个实验目录会在确认后创建 `README.md`、`run.sh`、`post_train.sh`、训练/渲染/评估日志、checkpoint、全时序 RGB/alpha/albedo-normal/separation 可视化以及 normal 指标。README 会记录完整命令、source/model/iteration、输出、定量结果、代表图像/视频、四类目检和最终 PASS/FAILED；失败产物保留且不覆盖。

启动顺序：先复核当前未提交的 loss 相关基线并运行相关单元测试；再分别完成冒烟；仅冒烟 PASS 的组启动 35,000 iter 正式训练。

## 6. 实验 A 执行记录与结论

- 训练前：`tests.test_stage1_loss` 为 14/14 PASS；首次 SH 冒烟发现梯度审计器会对无梯度的 photometric 参数组调用 `autograd.grad`。该诊断器问题已作最小兼容修正，训练损失、反传、优化器与 checkpoint 不受影响；首次失败产物保留在 `output/smoke_test/0824-03-CV3s-SH_default2DGSnormal/`，重试冒烟 PASS。
- 正式训练：GPU 0（A40），35,000 iter 完成，用时约 23 分 54 秒；best PSNR 52.56204（iter 30000），SSIM 0.99940，LPIPS 0.00261，MS-SSIM 0.99985，Alex-LPIPS 0.00031。
- 渲染 / 评估：全时序 120 帧渲染与 GT EXR normal 离线评估均 exit=0；默认 GS raster normal 的 mean / median / p95 为 12.54203° / 8.86624° / 37.41768°，alpha 覆盖率 0.168259、时序差 6.68999e-6。
- 目检：RGB 轮廓、颜色和褶皱稳定；alpha 连续且无闪烁；normal 主体连续，高误差集中在高频褶皱/轮廓；separation 整体稳定但接触区有少量残余。
- 结论：**实验 A PASS**，完整产物见 `output/0824-03-CV3s-SH_default2DGSnormal/README.md`。实验 B 未启动，仍需用户重新确认其与 0823-01 的差异和目的。

## 6. 实验 A 执行记录（2026-08-24）

- 已确认服务器 / 环境：`garuda` / `lumimotion-garuda`；GPU 0（A40 46 GB）空闲后使用。
- Git HEAD：`7e3c8b5 Archive baseline before Stage1 loss refactor`；当前实验固定使用尚未提交的 Stage1 loss 集中化工作树。相关单元测试 `tests.test_stage1_loss` 为 **14/14 PASS**。
- 原始冒烟目录 `output/smoke_test/0824-03-CV3s-SH_default2DGSnormal/` 在 iteration 1 的 `gradient_audit` 因 SH 非活跃参数被传入 `autograd.grad` 而失败；日志、模型初始化产物和 CSV 已完整保留。
- 已作最小审计修复（不改变 loss / 反传 / 优化器），见 `DOC/修改/0824-04-eval_stage1_normals_gt支持SH基线.md`；重试使用不覆盖的新目录 `output/smoke_test/0824-04-CV3s-SH_default2DGSnormal_retry01/`。
- 重试冒烟：**PASS**。1200 iter 正常结束，无数值异常或 OOM；499 / 500 / 1000 / 1200 checkpoint 完整。iter 500：PSNR 29.83697、SSIM 0.97835、LPIPS 0.02876；iter 1000：PSNR 34.51687、SSIM 0.98912、LPIPS 0.02075。梯度审计正常记录且无 independent normal 项。
- 实验 B 尚未启动，等待用户进一步讨论。
