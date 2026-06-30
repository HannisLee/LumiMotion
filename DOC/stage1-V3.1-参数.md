# Stage1 V3.1 推荐参数

本文记录 `PS-stage1-V3` 分支上 Stage1 V3.1 的推荐训练参数。V3.1 的核心变化是：光照方向固定使用 per-frame 参数表，每帧直接学习一个 raw light direction，渲染前 normalize；不再使用 B-spline 或 control points。light smooth 恢复为一阶 / 二阶 L2 正则，用权重控制逐帧抖动。

## 1. 推荐结论

默认推荐：

```text
photometric_light_param          = per_frame
photometric_multistart_enabled   = True
photometric_multistart_num_phases = 16
lambda_photometric_light_smooth1 = 0.001
lambda_photometric_light_smooth2 = 0.001
lambda_photometric_hemi          = 0.001
lambda_photometric_albedo_prior  = 0.005
```

要点：

- `photometric_light_param` 固定使用 `per_frame`。
- `photometric_num_ctrl_points` 不参与 V3.1 light model，不要再用它调轨迹容量。
- `photometric_multistart_num_phases=16` 会在正式训练前试 16 个圆形初始化相位，只选择 per-frame light table 的起点，不恢复曲线或 control points。
- `lambda_photometric_light_smooth1` 是相邻帧 light direction 的一阶 L2 smooth。
- `lambda_photometric_light_smooth2` 是二阶差分 L2 smooth，用于抑制逐帧曲率抖动。
- 如果真实光照轨迹有明显局部速度变化，优先降低 smooth 权重做对照。

## 2. 标准训练命令

以 LH static 单场景为例。服务器 `minakshi` 使用 `lumimotion-cu126`；旧服务器使用 `lumimotion-cu129` 时只替换 conda 环境名。

```bash
SCENE=brass_vase
GPU=0
OUT_ROOT=output/LH-staticv3.1

CUDA_VISIBLE_DEVICES=${GPU} conda run --no-capture-output -n lumimotion-cu126 \
python -m scripts.train_stage1 \
  --source_path data/LH-data/transfer-static/${SCENE} \
  --model_path ${OUT_ROOT}/${SCENE}_stage1v31 \
  --images images \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --render_mode photometric_lambertian \
  --photometric_stage s1d_joint \
  --photometric_light_param per_frame \
  --photometric_init_r_xy 0.8 \
  --photometric_init_z 0.6 \
  --photometric_multistart_enabled \
  --photometric_multistart_num_phases 16 \
  --photometric_multistart_short_iters 1000 \
  --photometric_s1d_light_lr 0.0001 \
  --photometric_s1d_albedo_lr 0.001 \
  --photometric_s1d_rotation_lr 0.0001 \
  --photometric_s1d_scaling_lr 0.0001 \
  --photometric_s1d_position_lr 0.0 \
  --photometric_s1d_deformation_lr 0.0 \
  --photometric_s1d_opacity_lr 0.0 \
  --photometric_use_hemi_prior \
  --lambda_photometric_light_smooth1 0.001 \
  --lambda_photometric_light_smooth2 0.001 \
  --lambda_photometric_hemi 0.001 \
  --lambda_photometric_albedo_prior 0.005 \
  --deform-type static \
  --resolution 2 \
  --iterations 35000 \
  --densify_until_iter 8000 \
  --opacity_reset_interval 100000 \
  --min_opacity 0.005 \
  --binarization_warm_up 1000 \
  --lambda_separation 0.005 \
  --d_xyz_loss_weight 0.001 \
  --d_color_reg_loss_weight 0.01 \
  --depth_ratio 1.0 \
  --test_iterations 1000 5000 10000 20000 30000 35000 \
  --save_iterations 1 1000 5000 10000 20000 30000 35000 \
  --quiet
```

训练脚本会自动追加 `_<deform_type>` 后缀，所以上面命令的实际输出目录是：

```text
output/LH-staticv3.1/<scene>_stage1v31_static
```

## 3. 参数档位

推荐先只调 smooth 权重：

| 档位 | `lambda_photometric_light_smooth1` | `lambda_photometric_light_smooth2` | 用途 |
| --- | ---: | ---: | --- |
| 主推 | `0.001` | `0.001` | 弱 L2 平滑，保留防抖能力。 |
| 放宽 | `0.0003` | `0.0003` | 真实光照轨迹有局部速度变化时优先测试。 |
| 只保留一阶 | `0.001` | `0.0` | 判断二阶曲率约束是否限制拟合。 |
| 无正则对照 | `0.0` | `0.0` | 判断正则影响；可能出现逐帧抖动。 |

不要通过 `photometric_num_ctrl_points` 调 V3.1，因为 V3.1 不再使用曲线控制点。

## 4. 不推荐参数

V3.1 不推荐继续使用曲线参数化相关配置：

```bash
--photometric_light_param bspline
--photometric_num_ctrl_points 16
```

原因：

- V3.1 active model 已固定为 per-frame light table。
- `photometric_num_ctrl_points` 对 V3.1 训练无效，继续保留只会造成实验记录混乱。

## 5. 训练后检查

训练完成后建议导出 light direction，并和 GT `lights.json` 对比：

```bash
MODEL=output/LH-staticv3.1/${SCENE}_stage1v31_static

python -m LH_Utils.export_light_directions \
  --model_path ${MODEL} \
  --iteration 35000 \
  --output ${MODEL}/light_directions_it35000.csv

python -m LH_Utils.plot_light_polar \
  --csv ${MODEL}/light_directions_it35000.csv \
  --lights_json data/LH-data/static/${SCENE}/lights.json \
  --output ${MODEL}/light_polar_compare_it35000.png

python -m LH_Utils.plot_light_timeseries \
  --csv ${MODEL}/light_directions_it35000.csv \
  --lights_json data/LH-data/static/${SCENE}/lights.json \
  --output ${MODEL}/light_timeseries_compare_it35000.png

python -m LH_Utils.light_direction_loss \
  --csv ${MODEL}/light_directions_it35000.csv \
  --lights_json data/LH-data/static/${SCENE}/lights.json \
  --output ${MODEL}/light_direction_loss_it35000.txt
```

重点看：

- polar 图是否能覆盖真实轨迹；
- timeseries 中 x/y/z 是否出现逐帧抖动；
- `light_direction_loss_it35000.txt` 的 mean angular error 是否相对 V2 降低；
- 渲染图是否出现明显逐帧闪烁。
