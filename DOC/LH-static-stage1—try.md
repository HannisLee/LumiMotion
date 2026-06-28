# LH static 数据集 PS mode 标准 Stage1 训练记录

本文档记录 `data/LH-data/static` 静态数据的完整链路：

1. 原始静态数据转换为 LumiMotion Blender 格式；
2. 使用 PS mode，也就是 `render_mode=photometric_lambertian`，进行标准 Stage1 训练；
3. 使用 `iteration_35000` checkpoint 评测并导出 comparison 图片；
4. 汇总训练命令、评测命令、输出路径、指标和本次发现的问题。

本版已覆盖之前的 100-iteration smoke test 输出。当前 `output/LH-static` 下是重新训练后的标准 Stage1 结果。

## 1. 数据位置

原始静态数据：

```bash
data/LH-data/static
```

转换后数据：

```bash
data/LH-data/transfer-static
```

本次包含 4 个场景：

| 场景 | 原始帧数 | 图像分辨率 | object_pose.json |
| --- | ---: | --- | --- |
| `brass_vase` | 120 | 768x768 | 无 |
| `concrete_cat` | 120 | 768x768 | 无 |
| `garden_gnome` | 120 | 768x768 | 无 |
| `rubber_duck_toy` | 120 | 768x768 | 无 |

每个原始场景目录包含：

```text
image/
albedo/
normal_exr/
camera.json
lights.json
static_processing_report.json
export_summary.json
```

当前 Stage1 PS v0 实际训练只使用转换后的 RGB / alpha mask / camera / time。`albedo/`、`normal_exr/`、`lights.json` 会被转换脚本记录到 manifest，但本版训练没有把它们作为监督或真实光照输入。

## 2. 转换脚本

转换脚本：

```bash
data/LH-data/prepare_lumimotion.py
```

为支持这批 static 数据，转换逻辑做了兼容：

- `object_pose.json` 从必需变为可选；
- 有 `object_pose.json` 时，仍按 LH 动态数据使用每帧物体位姿；
- 没有 `object_pose.json` 时，按 fixed camera / static object 处理，使用单位物体位姿；
- `conversion_summary.json` 中记录 `has_object_pose=false`；
- `manifest.json` 中 `source_metadata.object_pose=null`。

这个改动不影响原来带 `object_pose.json` 的动态 LH 数据。

## 3. 数据转换命令

检查数据：

```bash
conda run -n lumimotion-cu129 python data/LH-data/prepare_lumimotion.py \
  --origin-root data/LH-data/static \
  --transfer-root data/LH-data/transfer-static \
  --validate-only
```

检查结果：

```text
[OK] brass_vase: frames=120 train=105 test=15 resolution=768x768 mask=image-background
[OK] concrete_cat: frames=120 train=105 test=15 resolution=768x768 mask=image-background
[OK] garden_gnome: frames=120 train=105 test=15 resolution=768x768 mask=image-background
[OK] rubber_duck_toy: frames=120 train=105 test=15 resolution=768x768 mask=image-background
```

正式转换：

```bash
conda run -n lumimotion-cu129 python data/LH-data/prepare_lumimotion.py \
  --origin-root data/LH-data/static \
  --transfer-root data/LH-data/transfer-static \
  --overwrite
```

转换后每个场景包含：

```text
images/
transforms_train.json
transforms_test.json
manifest.json
conversion_summary.json
point3d.ply
```

含义：

| 文件或目录 | 作用 |
| --- | --- |
| `images/` | 转换后的 RGBA PNG。RGB 来自原始 `image/`，alpha 由背景色推断。 |
| `transforms_train.json` | LumiMotion Blender loader 的训练相机和 time 信息。 |
| `transforms_test.json` | LumiMotion Blender loader 的测试相机和 time 信息。 |
| `manifest.json` | 原始 image / albedo / normal / light / camera 来源索引。 |
| `conversion_summary.json` | 转换统计信息。 |
| `point3d.ply` | 初始化点云，用于 LumiMotion 初始化 Gaussian。 |

转换后划分：

| 场景 | images | train | test | fixed camera | has object pose |
| --- | ---: | ---: | ---: | --- | --- |
| `brass_vase` | 120 | 105 | 15 | 是 | 否 |
| `concrete_cat` | 120 | 105 | 15 | 是 | 否 |
| `garden_gnome` | 120 | 105 | 15 | 是 | 否 |
| `rubber_duck_toy` | 120 | 105 | 15 | 是 | 否 |

## 4. PS mode 模型说明

本次训练使用当前分支中的 PS mode：

```bash
--render_mode photometric_lambertian
```

颜色模型为：

```text
C_i(t) = albedo_i * max(0, normal_i_t · light_dir_t)
```

其中：

- `albedo_i` 是每个 Gaussian 的可学习 diffuse albedo；
- `normal_i_t` 来自当前 Gaussian rotation 的 z 轴；
- `light_dir_t` 是每帧可学习 directional light，forward 时 normalize；
- 光照强度固定为 1；
- 不读取 `lights.json` 中的真实点光源或面积光源；
- 不使用 `normal_exr` 做法线监督；
- 不使用 `albedo/` 图像做 albedo 监督。

因为这批数据是 fixed camera / static object，本次训练使用：

```bash
--deform-type static
```

## 5. 本次必要代码修复

标准 Stage1 使用：

```bash
--d_xyz_loss_weight 0.001
```

但在 `--deform-type static` 下，deformation 不返回 tensor 形式的 `d_xyz`，原代码会在 warmup 后执行：

```python
(d_xyz ** 2).mean()
```

从而报错：

```text
AttributeError: 'float' object has no attribute 'mean'
```

已在 `scripts/train_stage1.py` 做最小兼容修复：

```python
if self.opt.d_xyz_loss_weight > 0 and torch.is_tensor(d_xyz):
    loss += ((d_xyz**2).mean()) * self.opt.d_xyz_loss_weight
```

这个修复只跳过 static 模式下没有 tensor offset 的 `d_xyz` loss，不改变 MLP deformation 返回 tensor 时的原始行为。

## 6. 标准 Stage1 训练配置

本次不是 100-iteration smoke test，而是标准 Stage1：

```text
iterations = 35000
```

同时沿用 LH 单目数据之前验证过的稳定参数：

```text
resolution             = 2
densify_until_iter     = 8000
opacity_reset_interval = 100000
min_opacity            = 0.005
```

原因是固定单目数据在默认 opacity reset / densification 下可能把 Gaussian 裁空或点数增长过快。

训练输出目录：

```bash
output/LH-static
```

每个场景实际输出目录：

```text
output/LH-static/brass_vase_ps_stage1_static
output/LH-static/concrete_cat_ps_stage1_static
output/LH-static/garden_gnome_ps_stage1_static
output/LH-static/rubber_duck_toy_ps_stage1_static
```

训练日志：

```text
output/LH-static/logs/brass_vase_train.log
output/LH-static/logs/concrete_cat_train.log
output/LH-static/logs/garden_gnome_train.log
output/LH-static/logs/rubber_duck_toy_train.log
```

## 7. 标准 Stage1 训练命令

本次实际并行跑 4 个场景，分别绑定 GPU0、GPU1、GPU2、GPU5。

关键点：

- 使用 `conda run --no-capture-output`，避免 conda 默认捕获输出导致长任务状态不可见；
- 使用 `--train_light_folder images`，否则 Blender reader 会寻找默认 relighting 目录；
- 使用 `--images images`，和转换后的目录一致；
- 使用 `--load_iter` 只在 eval 阶段使用，训练阶段从头开始。

实际训练命令模板如下：

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path "data/LH-data/transfer-static/<SCENE>" \
  --model_path "output/LH-static/<SCENE>_ps_stage1" \
  --images images \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --render_mode photometric_lambertian \
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
  --lambda_photometric_light_smooth 0.001 \
  --lambda_photometric_albedo_reg 0.0 \
  --test_iterations 1000 5000 10000 20000 30000 35000 \
  --save_iterations 1000 5000 10000 20000 30000 35000 \
  --quiet
```

并行分配：

| 场景 | GPU | 完成状态 | 完成时间 |
| --- | ---: | --- | --- |
| `brass_vase` | 5 | status=0 | 2026-06-26 21:33:29 |
| `concrete_cat` | 0 | status=0 | 2026-06-26 21:57:25 |
| `garden_gnome` | 1 | status=0 | 2026-06-26 21:30:29 |
| `rubber_duck_toy` | 2 | status=0 | 2026-06-26 21:49:06 |

全部场景都写出了：

```text
point_cloud/iteration_35000/point_cloud.ply
deform/iteration_35000/deform.pth
photometric/iteration_35000/photometric.pth
```

## 8. 评测命令

评测使用 `iteration_35000`：

```bash
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.eval_stage1_dynamic \
  --source_path "data/LH-data/transfer-static/<SCENE>" \
  --model_path "output/LH-static/<SCENE>_ps_stage1_static" \
  --images images \
  --train_light_folder images \
  --is_blender --eval \
  --resolution 2 \
  --render_mode photometric_lambertian \
  --deform_type static \
  --load_iter 35000
```

注意：

- `eval_stage1_dynamic.py` 使用参数名 `--deform_type`，不是训练脚本里的 `--deform-type`；
- eval 脚本不接受 `--gt_alpha_mask_as_scene_mask`，该参数只用于训练；
- eval 输出路径是 `eval_stage1_dynamic/ours_35000/`。

每个场景评测输出：

```text
output/LH-static/<SCENE>_ps_stage1_static/results_stage1_dynamic.json
output/LH-static/<SCENE>_ps_stage1_static/eval_stage1_dynamic/ours_35000/
```

每个场景都有 15 张 comparison PNG。

## 9. 评测指标

测试集每个场景 15 帧。以下指标来自 `results_stage1_dynamic.json` 的 `metrics_average`。

| 场景 | test frames | Gaussians | L1 | PSNR | SSIM | LPIPS-VGG | MS-SSIM | LPIPS-Alex | comparison PNG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brass_vase` | 15 | 32898 | 0.000286 | 43.5159 | 0.9989 | 0.0019 | 0.9981 | 0.0077 | 15 |
| `concrete_cat` | 15 | 42280 | 0.000767 | 42.1007 | 0.9987 | 0.0024 | 0.9988 | 0.0041 | 15 |
| `garden_gnome` | 15 | 34440 | 0.000503 | 43.0164 | 0.9991 | 0.0018 | 0.9991 | 0.0026 | 15 |
| `rubber_duck_toy` | 15 | 48575 | 0.001767 | 36.3684 | 0.9946 | 0.0060 | 0.9922 | 0.0072 | 15 |

## 10. 图片输出

总览图：

```text
output/LH-static/eval_comparison_overview.png
```

每个场景首张 comparison：

```text
output/LH-static/brass_vase_ps_stage1_static/eval_stage1_dynamic/ours_35000/000_frame_0008_comparison.png
output/LH-static/concrete_cat_ps_stage1_static/eval_stage1_dynamic/ours_35000/000_frame_0008_comparison.png
output/LH-static/garden_gnome_ps_stage1_static/eval_stage1_dynamic/ours_35000/000_frame_0008_comparison.png
output/LH-static/rubber_duck_toy_ps_stage1_static/eval_stage1_dynamic/ours_35000/000_frame_0008_comparison.png
```

`*_comparison.png` 是横向拼接图，包含：

```text
GT | render | error
```

本次总览图已经人工检查过，不是空图；四个场景都能看到 GT、render 和 error。

## 11. 结果观察

本次标准 Stage1 全链路已经跑通：

- static 数据转换可用；
- PS mode `photometric_lambertian` 可在 `static` deformation 下完成 35k 标准训练；
- 4 个场景都成功保存 `iteration_35000` checkpoint；
- 4 个场景都成功完成 eval；
- 每个场景都输出 15 张 comparison PNG；
- 总览图已生成。

指标相比 100-iteration smoke test 明显提升，尤其 L1 和 LPIPS 明显降低。当前测试图中 render 与 GT 基本对齐，error 图主要集中在高光、暗部边缘和局部材质差异。

## 12. 当前限制

虽然本次叫 PS mode，但当前实现仍是 Stage1 v0：

- 使用 learnable per-frame directional light；
- 不读取 `lights.json` 中的真实点光源位置、面积光参数或光强；
- 不使用 `normal_exr` 监督 normal；
- 不使用 `albedo/` 监督 albedo；
- 光强固定为 1，尺度主要由 albedo 吸收；
- 还没有 point light attenuation、shadow、BRDF 或 Neural BRDF。

因此，这版结果证明的是：

```text
LH static 数据 -> LumiMotion 格式 -> PS Lambertian v0 -> 标准 Stage1 35k -> eval 图片/指标
```

这条链路可运行且结果可视化正常。

下一步如果要更接近 photometric stereo，应优先把 `lights.json` 中的真实光源方向或点光源位置接入 renderer，而不是继续只学习 unconstrained directional light。
