# LumiMotion Blender 训练指导

更新日期：2026-08-17
仓库：`/home/han.li/reproduce/LumiMotion-perlight`

本指导默认数据转换、相机参数、alpha 和 canonical PLY 已经检查完成。

当前 clothV3 实验必须成对运行两个数据集，并保持除路径外的超参数一致：

| 别称 | source | lights | GT normal（仅离线评估） |
| --- | --- | --- | --- |
| CV3 | `data/LH-data/transfer-dynamic/only_clothV3` | `data/LH-data/danamic/only_clothV3/lights.json` | `data/LH-data/danamic/only_clothV3/normal_exr` |
| CV3L | `data/LH-data/transfer-dynamic/only_clothV3_lambertian` | `data/LH-data/danamic/only_clothV3_lambertian/lights.json` | `data/LH-data/danamic/only_clothV3_lambertian/normal_exr` |

CV3L 共 120 帧，使用 105 train / 15 test 划分，`lights.json` 同样包含 120 条记录。每次新训练必须同时记录 CV3 与 CV3L 输出目录；若其中一组未启动或失败，README 必须明确说明，不能只报告成功的一组。

------

## 1. Stage 1 训练策略

当前统一使用有限 densification：

1. `iteration 1500–5000`：进行有限 densification 和正常 prune。
2. `iteration 5000–35000`：停止 Gaussian 数量变化，继续训练 Gaussian 参数和 deformation MLP。

停止 densification **不等于冻结 deformation**。5000 iter 后 deformation、位置、颜色、opacity 等参数仍继续训练。

推荐参数：

```text
iterations              = 35000

warm_up                 = 500
densify_from_iter       = 1500
densify_until_iter      = 5000
densification_interval  = 200
densify_grad_threshold  = 0.0004

opacity_reset_interval  = 3000
min_opacity             = 0.01
prune_from_iter         = -1
max_gaussians           = 20000

lambda_separation       = 0.005
d_xyz_loss_weight       = 0.001
d_color_reg_loss_weight = 0.01
```

不要通过延长 `densify_until_iter` 或提高 `max_gaussians` 来解决动态几何问题。

------

## 2. 基础 Stage 1 训练命令

```bash
cd /home/han.li/reproduce/LumiMotion

SOURCE=data/LH-data/transfer-dynamic/only_clothV3
OUT=output/Baseline/<experiment>
MODEL="$OUT/only_clothV3_stage1"

GPU=<GPU>
ENV=<Conda环境>

mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n "$ENV" \
python -u -m scripts.train_stage1 \
  --source_path "$SOURCE" \
  --model_path "$MODEL" \
  --train_light_folder images \
  --is_blender \
  --eval \
  --gt_alpha_mask_as_scene_mask \
  --resolution 2 \
  --iterations 35000 \
  --warm_up 500 \
  --densify_from_iter 1500 \
  --densify_until_iter 5000 \
  --densification_interval 200 \
  --densify_grad_threshold 0.0004 \
  --opacity_reset_interval 3000 \
  --min_opacity 0.01 \
  --prune_from_iter -1 \
  --max_gaussians 20000 \
  --binarization_warm_up 1000 \
  --lambda_separation 0.005 \
  --d_xyz_loss_weight 0.001 \
  --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 \
  --test_iterations 500 1000 5000 10000 20000 30000 35000 \
  --save_iterations 499 500 1000 5000 10000 20000 30000 35000 \
  --quiet \
  2>&1 | tee "$OUT/train_stage1.log"
```

训练必须使用真实 RGBA soft alpha，并显式加入：

```text
--gt_alpha_mask_as_scene_mask
```

训练实际模型目录通常会自动增加 `_mlp` 后缀，后续渲染和评估时需要使用真实目录。

------

## 3. Per-light Lambertian 训练

如果训练目标包含光照方向和 Lambertian normal 恢复，使用 directional light。

### 3.1 当前独立 Normal 恢复实验

不要再通过放开 GS rotation 同时优化 shading normal 和 Gaussian covariance。当前 Lambertian 管线使用独立 `photometric_normal`：在 photometric 模式切换时由当时的 GS normal 初始化，随后只参与 `N·L`，不参与 covariance 或屏幕覆盖。

推荐配置：

```text
--photometric_start_iter 10001
--photometric_light_mode gt_directional
--photometric_gt_lights_path <lights.json>
--photometric_gt_light_intensity <按数据集标定>
--photometric_gt_light_color 1.0,1.0,1.0
--photometric_albedo_lr 0.001
--photometric_normal_lr 0.0001
--photometric_normal_start_iter 10501
--photometric_albedo_freeze_iter 10501
--lambda_photometric_normal_init 0.01
--photometric_staged_training
--photometric_deform_unfreeze_iter 40000
--photometric_rotation_unfreeze_iter 40000
```

示例：

```bash
--photometric_start_iter 10001 \
--photometric_light_mode gt_directional \
--photometric_gt_lights_path data/LH-data/danamic/only_clothV3/lights.json \
--photometric_gt_light_intensity 5.5043499 \
--photometric_gt_light_color 1.0,1.0,1.0 \
--photometric_albedo_lr 0.001 \
--photometric_normal_lr 0.0001 \
--photometric_normal_start_iter 10501 \
--photometric_albedo_freeze_iter 10501 \
--lambda_photometric_normal_init 0.01 \
--photometric_staged_training \
--photometric_deform_unfreeze_iter 40000 \
--photometric_rotation_unfreeze_iter 40000
```

Directional 模式只使用每帧 light direction：

- 不使用 Gaussian 到灯的距离；
- 不使用 `1/r²`；
- CV3 (`only_clothV3`) 的重新标定白光 irradiance 使用 `5.5043499`；
- CV3L (`only_clothV3_lambertian`) 不得复用 5.5，重新标定值使用 `7.8434867`；
- iteration 1～10000 先得到可用的 SH geometry/deformation；
- iteration 10001 由当前 GS normal 初始化独立 normal；
- iteration 10001～10500 固定 independent normal，只训练 photometric albedo；
- iteration 10501 起固定 albedo，只训练 independent normal；
- 原 GS position、rotation、scale、opacity、deformation 与 GT light 在本轮一直固定；
- `lambda_photometric_normal_init=0.01` 只约束 independent normal 不要过快偏离其 GS 初始化值，不是 GT-normal loss；
- 训练损失不使用 GT-normal loss，GT EXR normal 只用于离线评价。

自 2026-08-23 起，Stage 1 全部损失项集中在 `scripts/loss.py`（`compute_stage1_loss`，由 `scripts/train_stage1.py` 的 `train_step` 调用），原 `train_step` 内联实现行为不变。命名损失组合可用 `--loss_preset` 选择（`sh_baseline` / `lambertian_default` / `lambertian_normal2` / `lambertian_normal3` / `gs_wrapping` / `pbr_default` / `n0_gt_normal_oracle`）；不传（`auto`）时行为完全由各 `lambda_*` 参数决定，既有实验不受影响，显式 CLI 参数永远优先于预设默认值。其中 `lambertian_normal2` 为 0824-01 二法线一致性消融组合（`lambertian_normal3` 去掉 GS 几何自一致法线，`lambda_gs_normal=0`）；GS 几何自一致法线权重自 2026-08-24 起由 `--lambda_gs_normal` 控制（默认 0.02 = 历史硬编码值，置 0 可在不影响 distortion 的前提下单独关闭该项）。另自 2026-08-24 起新增 `gs_wrapping` 预设：按 Gaussian Wrapping（From Blobs to Spokes，arXiv:2604.07337）论文 §4.1 权重方案映射——`lambda_gs_normal=0.05`（L_DN 深度-法线一致性）、`lambda_photometric_normal_live=0.05`（L_N 法线对齐）、`lambda_photometric_normal_mv=0.02`（L_gc 多视角几何一致性）；论文 L_pc 多视角光度一致性本仓库无对应项，调度沿用本仓库（live@500、mv@1000+ramp、`start_normal_reg` 门控），详见 `DOC/修改/0824-03-gs_wrapping预设.md`。自 2026-08-26 起，各原子 loss 与 preset 已改为独立对象组合，权重在构造时固定；`pbr_default` 仅为历史注册名，`photometric_perlight_pbr` Stage1 loss 已停用并在启动时明确报错，当前只支持 `original_sh` 与 `photometric_lambertian`。

注意：历史版本的 `photometric_staged_training` 虽然将阶段打印为 `albedo_only`，但没有关闭 independent normal 的学习率，实际是 albedo 与 normal 联合优化。使用新参数后必须在 gradient audit 中核验：10001～10500 的 normal LR/gradient 为 0，10501 起 albedo LR/gradient 为 0。

旧的 `photometric_start_iter=1`、自由 GS rotation 配置仅作为历史 baseline 保留，不再作为独立 normal 恢复的默认方案。

Lambertian 必须在线性 RGB 空间计算：

```text
sRGB albedo
    ↓
linear RGB
    ↓
albedo / π × irradiance × max(N · L, 0)
    ↓
alpha compositing
    ↓
sRGB
```

不能直接对 sRGB albedo 乘 `N·L`。

LH Blender EXR normal 为**世界坐标系 normal**，light direction、normal 和 renderer normal 应统一在世界坐标系中计算。

------

## 4. 训练过程中检查

建议重点检查：

```text
500
1000
5000
10000
20000
30000
35000
```

关注：

- Gaussian 数量；
- canonical XYZ bbox；
- Gaussian scale；
- opacity；
- dynamic/static separation；
- RGB、alpha 和 normal。

其中最重要的是 Gaussian 数量。

正常情况下：

```text
初始化        ≈ 4k
5000 iter     ≈ 15k
5000以后      点数基本固定
```

如果 5000 iter 前已经明显接近或超过 `20000`，优先检查数据、FOV、alpha、scene extent 和 PLY 对齐，而不是继续提高 `max_gaussians`。

------

## 5. 完整时序渲染

训练结束后运行：

```bash
ROOT=/home/han.li/reproduce/LumiMotion
SOURCE="$ROOT/data/LH-data/transfer-dynamic/only_clothV3"
MODEL="$ROOT/output/Baseline/<experiment>/<model>_mlp"
ITER=35000

CUDA_VISIBLE_DEVICES="$GPU" \
conda run --no-capture-output -n "$ENV" \
python -m scripts.render_stage1_insights \
  --source_path "$SOURCE" \
  --model_path "$MODEL" \
  --train_light_folder images \
  --is_blender \
  --eval \
  --resolution 2 \
  --render_mode photometric_lambertian \
  --load_iter "$ITER" \
  --depth_ratio 0.0 \
  --quiet
```

120 timestep 数据应完整输出 120 帧：

- RGB
- alpha
- albedo
- normal
- separation

渲染 photometric 模型时必须显式指定：

```text
--render_mode photometric_lambertian
```

避免使用默认 `original_sh` 渲染结果误判 Lambertian 训练效果。

------

## 6. 定量验收指标

训练完成后主要输出两类角度误差。

### 6.1 Light Direction Angular Error

比较每个 timestep：

```text
Predicted Light Direction
            vs
GT Light Direction
```

计算：

```text
θ = arccos(
    clamp(
        normalize(L_pred) · normalize(L_gt),
        -1,
        1
    )
)
```

输出：

```text
Light Direction Angular Error

Mean   = xx.xx°
Median = xx.xx°
P95    = xx.xx°
Max    = xx.xx°
```

同时建议保存逐 timestep 结果：

```text
timestep, angle_deg
0,   ...
1,   ...
...
119, ...
```

用于观察是否存在特定时间段 light direction 明显漂移。

------

### 6.2 Normal Angular Error

比较训练得到的 renderer normal 与 Blender GT normal：

```text
Predicted World Normal
          vs
GT World Normal
```

计算：

```text
θ = arccos(
    clamp(
        normalize(N_pred) · normalize(N_gt),
        -1,
        1
    )
)
```

只统计有效前景区域。

输出：

```text
Normal Angular Error

Mean   = xx.xx°
Median = xx.xx°
P95    = xx.xx°
Max    = xx.xx°
```

LH Blender EXR normal 本身就是**世界坐标系**，因此应直接与 renderer 的世界 normal 比较，**不要再进行相机坐标 Y/Z 翻转**。

建议至少评估：

```text
iteration 10000
iteration 20000
iteration 30000
iteration 35000
```

最终整理成：

```text
Iteration | Light Mean | Light Median | Light P95 | Normal Mean | Normal Median | Normal P95
----------|------------|--------------|-----------|-------------|---------------|-----------
10000     |            |              |           |             |               |
20000     |            |              |           |             |               |
30000     |            |              |           |             |               |
35000     |            |              |           |             |               |
```

------

## 7. 恢复训练

恢复前确认同时存在：

```text
<MODEL>_mlp/point_cloud/iteration_<ITER>/point_cloud.ply
<MODEL>_mlp/deform/iteration_<ITER>/deform.pth
```

恢复时加入：

```text
--load_iteration <ITER>
```

并保持以下配置与原训练一致：

- `source_path`
- `model_path`
- resolution
- alpha mask
- deformation 配置
- photometric 配置
- loss 配置
- densification 配置

恢复训练后 deformation 仍会正常继续更新。

------

## 8. 实验记录

每个正式实验建议保存：

```text
output/Baseline/<experiment>/
├── README.md
├── train_stage1.log
├── render_stage1_insights.log
├── light_angle_error.*
├── normal_angle_error.*
└── <model>_mlp/
```

`README.md` 至少记录：

```text
Dataset
GPU
Conda environment
Training command
Non-default parameters
Checkpoint
Light angular error
Normal angular error
```

实验对比时主要比较：

```text
Light Direction Angular Error
Normal Angular Error
```

并结合完整 120 timestep 渲染观察训练结果。
