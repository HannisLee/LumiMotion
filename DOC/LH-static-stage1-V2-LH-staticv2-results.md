# LH-static Stage1 V2 完整训练结果

本文记录在新服务器 `minakshi` 上完成的 LH-static Stage1 V2 photometric initialization 训练、评测和光源可视化导出。

## 1. 运行环境

- 代码版本：`52d085c`，分支 `feature/photometric-lambertian-v1`。
- Conda 环境：`lumimotion-cu126`。
- GPU：NVIDIA RTX 6000 Ada Generation。
- CUDA/PyTorch：PyTorch `2.1.0`，`torch.version.cuda=12.1`。
- native 扩展：`diff_surfel_rasterization`、`simple_knn`、`surfel_tracer`、`nvdiffrast` 均已在该环境编译安装。
- 输出根目录：`output/LH-staticv2`。

## 2. 数据集

使用转换后的 LH static Blender 格式数据：

```text
data/LH-data/transfer-static/brass_vase
data/LH-data/transfer-static/concrete_cat
data/LH-data/transfer-static/garden_gnome
data/LH-data/transfer-static/rubber_duck_toy
```

GT 光源对比使用原始静态数据中的：

```text
data/LH-data/static/<scene>/lights.json
```

## 3. 三阶段训练

本次总 iteration 保持标准 Stage1 的 `35000`，内部拆成：

| 阶段 | iteration | render mode | 目的 |
| --- | ---: | --- | --- |
| S1A | 1-30000 | `original_sh` | 原始 LumiMotion / 2DGS 几何 warm-up |
| S1C | 30001-32000 | `photometric_lambertian` | 冻结几何，校准 B-spline directional light |
| S1D | 32001-35000 | `photometric_lambertian` | light / albedo / rotation / scale 小学习率联合微调 |

关键公共参数：

```bash
--images images
--train_light_folder images
--is_blender --eval --gt_alpha_mask_as_scene_mask
--deform-type static
--resolution 2
--densify_until_iter 8000
--opacity_reset_interval 100000
--min_opacity 0.005
--binarization_warm_up 1000
--lambda_separation 0.005
--d_xyz_loss_weight 0.001
--d_color_reg_loss_weight 0.01
--depth_ratio 1.0
```

S1A 使用：

```bash
--render_mode original_sh
--photometric_stage s1a_original_warmup
--iterations 30000
--save_iterations 1000 5000 10000 20000 30000
--test_iterations 30000
```

S1C 使用：

```bash
--load_iter 30000
--render_mode photometric_lambertian
--photometric_stage s1c_light_calib
--photometric_light_param bspline
--photometric_num_ctrl_points 16
--photometric_s1c_light_lr 0.001
--photometric_s1c_albedo_lr 0.0
--photometric_use_hemi_prior
--lambda_photometric_light_smooth1 0.01
--lambda_photometric_light_smooth2 0.01
--lambda_photometric_hemi 0.001
--lambda_photometric_albedo_prior 0.005
--iterations 32000
--save_iterations 32000
--test_iterations 32000
```

S1D 使用：

```bash
--load_iter 32000
--render_mode photometric_lambertian
--photometric_stage s1d_joint
--photometric_light_param bspline
--photometric_num_ctrl_points 16
--photometric_s1d_light_lr 0.0001
--photometric_s1d_albedo_lr 0.001
--photometric_s1d_rotation_lr 0.0001
--photometric_s1d_scaling_lr 0.0001
--photometric_s1d_position_lr 0.0
--photometric_s1d_deformation_lr 0.0
--photometric_s1d_opacity_lr 0.0
--photometric_use_hemi_prior
--lambda_photometric_light_smooth1 0.01
--lambda_photometric_light_smooth2 0.01
--lambda_photometric_hemi 0.001
--lambda_photometric_albedo_prior 0.005
--iterations 35000
--save_iterations 35000
--test_iterations 35000
```

## 4. 训练输出

每个场景最终都写出了：

```text
output/LH-staticv2/<scene>_stage1v2_static/point_cloud/iteration_35000/point_cloud.ply
output/LH-staticv2/<scene>_stage1v2_static/deform/iteration_35000/deform.pth
output/LH-staticv2/<scene>_stage1v2_static/photometric/iteration_35000/photometric.pth
```

训练日志：

```text
output/LH-staticv2/logs/<scene>_s1a.log
output/LH-staticv2/logs/<scene>_s1c.log
output/LH-staticv2/logs/<scene>_s1d.log
```

四个场景的 S1A、S1C、S1D 退出状态均为 `0`。

## 5. 评测

评测命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> conda run --no-capture-output -n lumimotion-cu126 \
python -m scripts.eval_stage1_dynamic \
  --source_path data/LH-data/transfer-static/<scene> \
  --model_path output/LH-staticv2/<scene>_stage1v2_static \
  --images images \
  --train_light_folder images \
  --is_blender --eval \
  --resolution 2 \
  --render_mode photometric_lambertian \
  --deform_type static \
  --load_iter 35000
```

评测日志：

```text
output/LH-staticv2/logs/<scene>_eval.log
```

评测输出：

```text
output/LH-staticv2/<scene>_stage1v2_static/results_stage1_dynamic.json
output/LH-staticv2/<scene>_stage1v2_static/eval_stage1_dynamic/ours_35000/
```

该目录中包含：

```text
*_comparison.png
*_render.png
*_gt.png
*_error.png
*_mask.png
*_albedo.png
*_normal.png
```

## 6. 指标汇总

| 场景 | Gaussian 数 | L1 | PSNR | SSIM | LPIPS-VGG | MS-SSIM | LPIPS-Alex |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brass_vase` | 31074 | 0.000902 | 35.693 | 0.99269 | 0.00893 | 0.98643 | 0.02143 |
| `concrete_cat` | 41424 | 0.003471 | 30.020 | 0.98717 | 0.01631 | 0.98266 | 0.02555 |
| `garden_gnome` | 34089 | 0.001686 | 33.497 | 0.99354 | 0.00902 | 0.99122 | 0.01205 |
| `rubber_duck_toy` | 46033 | 0.002920 | 32.078 | 0.98629 | 0.01765 | 0.98087 | 0.02086 |

## 7. 光源导出和可视化

对每个场景执行：

```bash
python -m LH_Utils.export_light_directions \
  --model_path output/LH-staticv2/<scene>_stage1v2_static \
  --iteration 35000 \
  --output output/LH-staticv2/<scene>_stage1v2_static/light_directions_it35000.csv

python -m LH_Utils.plot_light_polar \
  --csv output/LH-staticv2/<scene>_stage1v2_static/light_directions_it35000.csv \
  --lights_json data/LH-data/static/<scene>/lights.json \
  --output output/LH-staticv2/<scene>_stage1v2_static/light_polar_compare_it35000.png

python -m LH_Utils.plot_light_timeseries \
  --csv output/LH-staticv2/<scene>_stage1v2_static/light_directions_it35000.csv \
  --lights_json data/LH-data/static/<scene>/lights.json \
  --output output/LH-staticv2/<scene>_stage1v2_static/light_timeseries_compare_it35000.png
```

每个 CSV 有 120 条 light 数据，另有 1 行表头，每行 6 列：

```text
raw_x, raw_y, raw_z, dir_x, dir_y, dir_z
```

已生成：

```text
output/LH-staticv2/<scene>_stage1v2_static/light_directions_it35000.csv
output/LH-staticv2/<scene>_stage1v2_static/light_polar_compare_it35000.png
output/LH-staticv2/<scene>_stage1v2_static/light_timeseries_compare_it35000.png
```

## 8. 注意事项

- 本次 V2 使用 directional light，不使用 point light、distance attenuation、per-frame intensity、ambient、residual color 或 BRDF。
- GT `lights.json` 只用于导出后对比图，不参与训练。
- `light_polar_compare_it35000.png` 和 `light_timeseries_compare_it35000.png` 中的 GT 方向由 `light_pos_world - target` 归一化得到，默认 target 为 `[0, 0, 0]`。
- 本次新服务器记录相关文档改动包括 `AGENTS.md`、`CLAUDE.md`、`Version.md` 和本文档；训练产物均在 `output/LH-staticv2`。
