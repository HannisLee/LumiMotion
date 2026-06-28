# LH static Stage1 V2 photometric initialization

本文记录当前分支上 Stage1 V2 photometric initialization 的总体设计、运行流程和本次 smoke test 结果。

## 1. 版本目标

V2 在 LumiMotion / 2DGS baseline 上继续保留原始动态几何建模能力：

```text
canonical 2DGS
deformation model
dynamic position / rotation / scale / opacity
2DGS rasterizer
RGB reconstruction loss
mask / opacity loss
densification / pruning
```

只在 `render_mode="photometric_lambertian"` 时，把 Stage1 主 appearance 从原 SH / free color 改为：

```text
C_i(t) = rho_i * max(0, normal_i_t dot light_dir_t)
```

其中：

- `rho_i` 是 per-Gaussian albedo；
- `normal_i_t` 由动态 Gaussian rotation 的局部 normal 轴得到；
- `light_dir_t` 是第 `t` 帧 learnable directional light；
- light intensity 固定为 1；
- 不使用 point light、distance attenuation、ambient、residual color、roughness/metallic/BRDF、light MLP。

## 2. 与 V1 的差异

V1 photometric 路径是 per-frame light table 的最小版本。V2 覆盖 V1 photometric 实现，主要升级为：

- 默认原始模式改名为 `render_mode="original_sh"`，旧的 `"original"` 仍作为兼容别名。
- 新增 `DirectionalLightModel`，支持：
  - `photometric_light_param="per_frame"`
  - `photometric_light_param="bspline"`
- 默认推荐 B-spline control points，而不是直接学习每帧 `[T, 3]`。
- light 初始化改为圆形上半球轨迹。
- normal 支持 `photometric_normal_axis="+z"` / `"-z"`。
- 新增一阶/二阶 light smoothness、upper-hemisphere prior、albedo prior。
- 支持 Stage1A / Stage1C / Stage1D 三种训练入口阶段。
- photometric checkpoint 保存 V2 config、light trajectory `.npy/.json`。
- `LH_Utils.export_light_directions` 支持 V2 B-spline checkpoint。

## 3. 训练阶段

### Stage 1A: original SH geometry warm-up

目标是先用原 LumiMotion appearance 学一个可用的几何初始化：

```text
render_mode = original_sh
```

优化内容：

```text
position / rotation / scale / opacity / deformation / SH appearance
```

不启用：

```text
photometric albedo
directional light model
Lambertian color
```

### Stage 1C: light-only calibration

从 S1A checkpoint 加载：

```text
render_mode = photometric_lambertian
photometric_stage = s1c_light_calib
```

冻结：

```text
position / deformation / rotation / scale / opacity / SH
```

优化：

```text
directional light model
```

默认 `photometric_s1c_albedo_lr=0.0`，即 albedo 使用 SH DC 初始化后不动。

### Stage 1D: photometric joint fine-tune

从 S1C checkpoint 加载：

```text
render_mode = photometric_lambertian
photometric_stage = s1d_joint
```

默认优化：

```text
light: small lr
albedo: small lr
rotation: small lr
scale: small lr
```

默认冻结：

```text
position
deformation
SH appearance
```

这一步仍然只使用 uniform-intensity directional Lambertian，不引入更复杂的光照或 BRDF。

## 4. 关键参数

```bash
--render_mode original_sh
--render_mode photometric_lambertian
--photometric_stage s1a_original_warmup
--photometric_stage s1c_light_calib
--photometric_stage s1d_joint
--photometric_light_param bspline
--photometric_light_param per_frame
--photometric_num_ctrl_points 16
--photometric_init_r_xy 0.8
--photometric_init_z 0.6
--photometric_init_phase 0.0
--photometric_init_direction_sign 1
--photometric_normal_axis +z
--photometric_use_hemi_prior
--photometric_hemi_axis 0,0,1
--photometric_hemi_margin 0.0
--lambda_photometric_light_smooth1 0.01
--lambda_photometric_light_smooth2 0.01
--lambda_photometric_hemi 0.001
--lambda_photometric_albedo_prior 0.005
```

旧参数仍兼容：

```bash
--render_mode original
--lambda_photometric_light_smooth
--lambda_photometric_albedo_reg
```

## 5. 本次 smoke test

测试数据：

```text
data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1
```

输出目录：

```text
output/LH-test/stage1v2_spheres_static
```

注意：这是极短迭代 smoke test，只验证代码链路，不代表最终质量。

### 5.1 Stage 1A 命令

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/LH-test/stage1v2_spheres \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 4 \
  --iterations 2 \
  --save_iterations 2 \
  --test_iterations 2 \
  --deform-type static \
  --render_mode original_sh \
  --photometric_stage s1a_original_warmup \
  --train_light_folder chapel_day_4k_32x16_rot0 \
  --depth_ratio 1.0
```

结果：

```text
iteration_2 saved
test PSNR 8.40
train PSNR 8.59
```

### 5.2 Stage 1C 命令

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/LH-test/stage1v2_spheres \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 4 \
  --iterations 4 \
  --load_iter 2 \
  --save_iterations 4 \
  --test_iterations 4 \
  --deform-type static \
  --render_mode photometric_lambertian \
  --photometric_stage s1c_light_calib \
  --photometric_light_param bspline \
  --photometric_num_ctrl_points 8 \
  --photometric_s1c_light_lr 0.001 \
  --photometric_s1c_albedo_lr 0.0 \
  --lambda_photometric_light_smooth1 0.01 \
  --lambda_photometric_light_smooth2 0.01 \
  --lambda_photometric_hemi 0.001 \
  --photometric_use_hemi_prior \
  --lambda_photometric_albedo_prior 0.005 \
  --train_light_folder chapel_day_4k_32x16_rot0 \
  --depth_ratio 1.0
```

结果：

```text
iteration_4 saved
photometric/iteration_4/photometric.pth saved
test PSNR 9.02
train PSNR 8.76
```

### 5.3 Stage 1D 命令

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/LH-test/stage1v2_spheres \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 4 \
  --iterations 6 \
  --load_iter 4 \
  --save_iterations 6 \
  --test_iterations 6 \
  --deform-type static \
  --render_mode photometric_lambertian \
  --photometric_stage s1d_joint \
  --photometric_light_param bspline \
  --photometric_num_ctrl_points 8 \
  --photometric_s1d_light_lr 0.0001 \
  --photometric_s1d_albedo_lr 0.001 \
  --photometric_s1d_rotation_lr 0.0001 \
  --photometric_s1d_scaling_lr 0.0001 \
  --photometric_s1d_position_lr 0.0 \
  --photometric_s1d_deformation_lr 0.0 \
  --lambda_photometric_light_smooth1 0.01 \
  --lambda_photometric_light_smooth2 0.01 \
  --lambda_photometric_hemi 0.001 \
  --photometric_use_hemi_prior \
  --lambda_photometric_albedo_prior 0.005 \
  --train_light_folder chapel_day_4k_32x16_rot0 \
  --depth_ratio 1.0
```

结果：

```text
iteration_6 saved
photometric/iteration_6/photometric.pth saved
test PSNR 9.02
train PSNR 8.76
```

## 6. 评测与可视化

### 6.1 Stage1 eval

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.eval_stage1_dynamic \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/LH-test/stage1v2_spheres \
  --is_blender --eval \
  --resolution 4 \
  --load_iter 6 \
  --deform-type static \
  --render_mode photometric_lambertian \
  --train_light_folder chapel_day_4k_32x16_rot0 \
  --depth_ratio 1.0
```

输出：

```text
output/LH-test/stage1v2_spheres_static/results_stage1_dynamic.json
output/LH-test/stage1v2_spheres_static/eval_stage1_dynamic/ours_6/*_comparison.png
output/LH-test/stage1v2_spheres_static/eval_stage1_dynamic/ours_6/*_render.png
output/LH-test/stage1v2_spheres_static/eval_stage1_dynamic/ours_6/*_albedo.png
output/LH-test/stage1v2_spheres_static/eval_stage1_dynamic/ours_6/*_normal.png
```

平均指标：

```text
PSNR       11.4666
SSIM        0.7705
LPIPS VGG   0.2896
MS-SSIM     0.6318
LPIPS Alex  0.2983
L1          0.1753
```

### 6.2 导出 light trajectory

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m LH_Utils.export_light_directions \
  --model_path output/LH-test/stage1v2_spheres_static \
  --iteration 6 \
  --output output/LH-test/stage1v2_spheres_static/light_directions_it6.csv
```

输出 CSV 为六列：

```text
raw_x, raw_y, raw_z, dir_x, dir_y, dir_z
```

### 6.3 绘制 light 图

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m LH_Utils.plot_light_polar \
  --csv output/LH-test/stage1v2_spheres_static/light_directions_it6.csv \
  --output output/LH-test/stage1v2_spheres_static/light_polar_it6.png
```

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m LH_Utils.plot_light_timeseries \
  --csv output/LH-test/stage1v2_spheres_static/light_directions_it6.csv \
  --output output/LH-test/stage1v2_spheres_static/light_timeseries_it6.png
```

## 7. Checkpoint 内容

V2 photometric checkpoint：

```text
output/LH-test/stage1v2_spheres_static/photometric/iteration_6/photometric.pth
```

已确认包含：

```text
photometric_version = stage1_v2_directional_uniform_light
config.light_param = bspline
config.num_ctrl_points = 8
config.normal_axis = +z
state_dict.light_model._light_ctrl
state_dict.light_model.timesteps
```

同时保存：

```text
photometric/iteration_6/light_dirs.npy
photometric/iteration_6/light_dirs.json
```

本次导出的 light direction 数量为 `99`，来自 train/test camera 的 unique timestep；norm 范围：

```text
min 0.9999999
max 1.0000001
```

## 8. Multistart smoke

V2 还提供独立 light initialization selection 工具。本次用 S1A `iteration_2` checkpoint 跑了最小 multistart smoke：

```bash
conda run --no-capture-output -n lumimotion-cu129 \
python -m LH_Utils.select_light_init \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/LH-test/stage1v2_spheres \
  --is_blender --eval \
  --resolution 4 \
  --load_iter 2 \
  --deform-type static \
  --num_phases 2 \
  --try_reverse_direction \
  --short_iters 1 \
  --candidate_lr 0.001 \
  --photometric_light_param bspline \
  --photometric_num_ctrl_points 8 \
  --train_light_folder chapel_day_4k_32x16_rot0 \
  --depth_ratio 1.0
```

输出：

```text
output/LH-test/stage1v2_spheres_static/photometric_multistart/iteration_2/best_light_init.pth
output/LH-test/stage1v2_spheres_static/photometric_multistart/iteration_2/best_light_init.json
```

候选结果：

```text
phase=0 sign=+1 loss=0.334802
phase=1 sign=+1 loss=0.362303
phase=0 sign=-1 loss=0.294634
phase=1 sign=-1 loss=0.314689
```

最优：

```text
best_phase = 0.0
best_direction_sign = -1
best_candidate_loss = 0.294634
```

## 9. 当前限制

- 本次只做短迭代 smoke test，没有跑标准 35000 iteration。
- B-spline 使用可微 cubic B-spline control-point 插值，目标是先跑通稳定初始化接口。
- `normal_exr` / GT normal 不参与训练监督。
- `lights.json` 只用于导出后对比可视化，不参与训练。
- 当前 V2 仍然不处理 point light、distance attenuation、shadow、residual、BRDF。
