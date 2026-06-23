# LH-data 接入 LumiMotion 原版两阶段训练指南

日期：2026-06-22

本文档说明 `/home/han.li/reproduce/LumiMotion/data/LH-data` 下两个自采集/自渲染场景如何转换为 LumiMotion 可读取的数据格式，以及如何运行 `render_mode=original` 的 Stage 1 和原版 Stage 2。最终实测结果、逐帧指标和渲染对比另见 `DOC/LH-data-two-stage-results.md`。

## 1. 结论概览

当前两个场景都可以转换并进入 LumiMotion 原版 Stage 1 训练：

- `cat`
- `rubber_duck_toy_dataset`

两个场景的共同属性：

- 每个场景 120 帧。
- RGB 分辨率为 `1280x720`。
- 相机固定，120 帧使用同一套内外参。
- 物体随时间运动。
- 点光源/面积光的位置随帧变化。
- RGB 图像背景颜色恒定，主背景色为 `[63, 63, 63]`。
- RGB PNG 自带 alpha 全为 255，不能直接作为物体 mask。
- albedo PNG 有有效 alpha，但本次按用户说明默认直接从 RGB 恒定背景生成 mask。
- normal EXR 有有效的 `Normal.X`、`Normal.Y`、`Normal.Z` 浮点通道。

原版 Stage 1 真正使用的是：

- RGB 图像。
- 物体 mask，用于 alpha supervision。
- 相机内参和外参。
- 每帧归一化时间 `time`。

原版 Stage 1 不会直接读取：

- albedo RGB。
- EXR normal。
- `lights.json`。
- `object_pose.json`。

因此，本次“原版训练”验证的是动态 Gaussian 重建 baseline，不是显式光照、法线或材质监督训练。

## 2. 原始数据检查结果

### 2.1 目录结构

两个场景原始结构均满足：

```text
data/LH-data/<scene>/
  image/
  albedo/
  normal_exr/
  camera.json
  lights.json
  object_pose.json
```

实际场景：

```text
data/LH-data/cat/
data/LH-data/rubber_duck_toy_dataset/
```

每个场景的 `image`、`albedo`、`normal_exr` 都有 120 个文件，frame id 范围为 `0001` 到 `0120`。

### 2.2 相机

两个场景使用相同固定相机参数：

```text
width  = 1280
height = 720
fx     = 1600
fy     = 1600
cx     = 640
cy     = 360
fov_x  = 0.7610127542 rad
fov_y  = 0.4426288847 rad
```

`camera.json` 提供：

- `K`。
- `fx/fy/cx/cy`。
- 水平和垂直 FOV。
- Blender `camera_to_world`。
- `world_to_camera`。
- Blender 相机约定：局部 `-Z` 为 forward，局部 `+Y` 为 up。

转换使用 `camera_to_world` 原值写入 `transform_matrix`。

### 2.3 RGB 和 mask

RGB 图像是 RGBA PNG，但原始 alpha 全部为 255，所以不能把原始 image alpha 当作物体 mask。

背景分析结果：

- 两个场景的背景主色均为 `[63, 63, 63]`。
- duck 场景边界背景完全一致。
- cat 场景边界存在少量 `±1` 的颜色变化。

默认 mask 算法：

1. 从图像四条边界统计最常见 RGB，作为本帧背景色。
2. 计算每个像素与背景色的最大通道差。
3. 当最大通道差大于 `1` 时，判定为前景。
4. 输出二值 alpha：背景 0，前景 255。
5. 输出 RGB 在背景区域清零，得到黑背景 RGBA 图像。

该 image-background mask 与 albedo alpha 的抽样 IoU 约为 `0.99`，符合当前训练使用要求。

转换后前景像素范围：

| 场景 | 每帧非零 mask 像素范围 |
| --- | ---: |
| cat | 28,743 - 45,593 |
| rubber duck | 20,161 - 31,601 |

### 2.4 Albedo

两个场景都有 120 帧 RGBA albedo PNG，alpha 有效。

本次原版 Stage 1：

- 不读取 albedo RGB。
- 默认不使用 albedo alpha，而是从 RGB 背景生成 mask。
- 转换脚本仍支持 `--mask-source albedo-alpha`，用于对照实验。

### 2.5 Normal EXR

EXR 文件有效，分辨率为 `1280x720`，通道名为：

```text
Normal.X
Normal.Y
Normal.Z
```

抽样数值约在 `[-1, 1]`，不是损坏的全黑 normal。OpenCV 默认按 RGB 通道读取时会失败或返回空，是因为这些 EXR 使用 Blender 自定义 channel name；可使用 Python `OpenEXR` 包读取。

原版 Stage 1 不使用这些 GT normal。后续如果增加 normal supervision，需要先明确：

- normal 是 world space、camera space 还是 object/local space。
- Blender normal 通道的轴方向。
- 图像坐标是否需要翻转。
- 动态物体 normal 是否已经包含 object pose。

### 2.6 Lights

`lights.json` 有 120 条记录，每帧包含：

- `light_pos_world`
- `light_rgb`
- `intensity`
- `source_type`
- area light size

当前记录是随时间移动的点光源/面积光，而原版 Stage 1 不读取这些字段。

### 2.7 Object pose

`object_pose.json` 有 120 帧 object-to-world pose，包括：

- world location。
- world quaternion。
- world Euler rotation。
- scale。
- 4x4 `matrix_world`。
- controller parent pose。

原版 LumiMotion deformation MLP 只根据 Gaussian canonical position 和时间学习动态变形，不使用已知 object pose。

注意：`rubber_duck_toy_dataset/object_pose.json` 和 `export_summary.json` 中的 object/controller 名称仍是 cat 相关名称，例如 `Dynamic_concrete_cat_statue`。这不影响本次 Stage 1，因为 pose 不被读取；后续使用 pose supervision 前应先确认导出对象命名是否只是命名遗留，还是 pose 确实来自错误对象。

## 3. LumiMotion Stage 1 数据需求映射

| 自有数据 | 原版 Stage 1 是否需要 | 本次处理方式 |
| --- | --- | --- |
| `image/*.png` | 必需 | 转成黑背景 RGBA，写入 `images/` |
| RGB 恒定背景 | 用于生成 mask | 边界众数 + 阈值 1 |
| `albedo/*.png` | 非必需 | 保留在原目录，可选用 alpha 生成 mask |
| `normal_exr/*.exr` | 不使用 | 保留，未来 normal supervision 使用 |
| `lights.json` | 不使用 | 保留，未来 photometric 模式使用 |
| `object_pose.json` | 不使用 | 保留，未来 motion prior/评估使用 |
| `camera.json` | 必需 | 转为 `camera_angle_x/y` 和 `transform_matrix` |
| frame id | 必需 | 转为 `[0,1]` 的 `time` |
| `points3d.ply` | 可选 | 缺失时 LumiMotion 自动生成 100,000 个随机点 |

## 4. 单文件转换脚本

脚本位置：

```text
data/LH-data/prepare_lumimotion.py
```

该脚本是单文件实现，依赖：

- Python 3。
- NumPy。
- Pillow。

### 4.1 仅验证数据

从仓库根目录执行：

```bash
conda activate lumimotion-cu129
python data/LH-data/prepare_lumimotion.py --validate-only
```

当前验证结果：

```text
cat: 120 frames, 105 train, 15 test, 1280x720
rubber_duck_toy_dataset: 120 frames, 105 train, 15 test, 1280x720
```

### 4.2 转换全部场景

```bash
python data/LH-data/prepare_lumimotion.py
```

不传场景路径时，脚本自动扫描 `data/LH-data` 下包含 `camera.json` 的直接子目录。

### 4.3 转换指定场景

```bash
python data/LH-data/prepare_lumimotion.py data/LH-data/cat
```

### 4.4 重新生成

```bash
python data/LH-data/prepare_lumimotion.py --overwrite
```

`--overwrite` 只覆盖脚本已知的生成图像和 JSON，不删除原始 `image/albedo/normal_exr`。

### 4.5 使用 albedo alpha 作为 mask

```bash
python data/LH-data/prepare_lumimotion.py \
  --mask-source albedo-alpha \
  --output-name lumimotion_original_albedo_mask
```

### 4.6 调整测试帧划分

默认每 8 帧留出 1 帧测试：

```text
train: 105 frames
test:  15 frames
```

修改为每 10 帧留出 1 帧：

```bash
python data/LH-data/prepare_lumimotion.py --test-stride 10 --overwrite
```

不留测试帧：

```bash
python data/LH-data/prepare_lumimotion.py --test-stride 0 --overwrite
```

## 5. 转换后目录

每个场景生成：

```text
data/LH-data/<scene>/lumimotion_original/
  images/
    frame_0001.png
    ...
    frame_0120.png
  transforms_train.json
  transforms_test.json
  dataset_manifest.json
  points3d.ply                 # 首次训练时由 LumiMotion 自动生成
```

当前实际生成大小：

| 场景 | 转换目录大小 |
| --- | ---: |
| cat | 约 11 MB |
| rubber duck | 约 4.6 MB |

生成数据位于 `.gitignore` 中的 `data/` 下，不会自动提交到 Git。

## 6. transforms 设计

### 6.1 时间

120 帧映射到：

```text
time = frame_ordinal / 119
```

因此：

- frame 1 的 `time=0.0`。
- frame 120 的 `time=1.0`。

测试帧仍使用完整 120 帧序列中的归一化时间，不会因为 train/test split 改变时间。

### 6.2 相机

所有帧复用 `camera.json` 中固定的 `camera_to_world`。

transforms 同时写入：

- `camera_angle_x`
- `camera_angle_y`
- `fl_x/fl_y`
- `cx/cy`
- `w/h`

### 6.3 固定相机 extent

原始 LumiMotion 用训练相机中心分布计算 `cameras_extent`。固定相机的所有 camera center 完全相同，会得到 radius 0，导致位置学习率和 densification 尺度退化。

转换文件写入：

```json
"camera_extent": 1.0
```

reader 只在自动计算 radius 接近 0 时使用该 fallback。可通过转换脚本修改：

```bash
python data/LH-data/prepare_lumimotion.py --camera-extent 1.0 --overwrite
```

## 7. Reader 兼容修复

为读取该固定相机数据，本分支对 `scene/dataset_readers.py` 做了三处最小兼容修复。

### 7.1 Pillow RGBA dtype

旧代码使用：

```python
dtype=np.byte
```

`np.byte` 是 signed int8，新版 Pillow 无法从该 RGBA array 创建图像。现在改为：

```python
dtype=np.uint8
```

### 7.2 非方形图像 FOV

旧 reader 只读取 `camera_angle_x`，且历史路径中 `FovX/FovY` 存在交换行为。

为避免影响旧数据：

- 旧 transforms 不含 `camera_angle_y` 时，保持原有行为。
- 本转换器生成的 transforms 显式包含 `camera_angle_y`，reader 使用正确的 `FovX=fov_x` 和 `FovY=fov_y`。

### 7.3 固定相机 radius

当相机中心分布 radius 接近 0 时：

- 读取 transforms 中的 `camera_extent`。
- 默认使用 `1.0`。
- 非固定相机数据继续使用原来的自动计算结果。

这些改动不修改 Gaussian、deformation model 或 rasterizer 结构。

## 8. Smoke test

### 8.1 不要使用 resolution=8 做带评估的 smoke test

原图 `1280x720` 使用 `--resolution 8` 后变为 `160x90`。仓库使用的 MS-SSIM 要求短边大于 160，因此在 test iteration 会报：

```text
AssertionError: Image size should be larger than 160 due to the 4 downsamplings in ms-ssim
```

训练和 rasterizer 本身已能运行，但评估失败。因此 smoke test 使用 `--resolution 4`，输出 `320x180`。

### 8.2 Cat smoke test

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original-smoke/cat_r4 \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --render_mode original \
  --resolution 4 \
  --iterations 2 \
  --test_iterations 2 \
  --save_iterations 2 \
  --quiet
```

结果：成功，生成 iteration 2 Gaussian 和 deformation checkpoint。

### 8.3 Rubber duck smoke test

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/LH-data/rubber_duck_toy_dataset/lumimotion_original \
  --model_path output/LH-original-smoke/rubber_duck_r4 \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --render_mode original \
  --resolution 4 \
  --iterations 2 \
  --test_iterations 2 \
  --save_iterations 2 \
  --quiet
```

结果：成功，生成 iteration 2 Gaussian 和 deformation checkpoint。

## 9. 正式原版 Stage 1 训练

本次采用仓库 synthetic baseline 的主要设置：

- `render_mode=original`
- `resolution=2`，实际训练分辨率 `640x360`
- `iterations=35000`
- `densify_until_iter=8000`
- `depth_ratio=1.0`
- `binarization_warm_up=1000`
- `lambda_separation=0.005`
- `d_xyz_loss_weight=0.001`
- `d_color_reg_loss_weight=0.01`
- `opacity_reset_interval=100000`
- `min_opacity=0.005`

这三项是当前固定单相机数据采用的稳定性设置。使用仓库默认的 `opacity_reset_interval=3000` 时，实测在 iteration 3000 重置 opacity 后，iteration 3100 的 densify/prune 会把全部 Gaussian 裁掉，随后 rasterizer backward 因 SH 输入变成 `[0, 16, 3]` 而失败。禁用本次训练区间内的 opacity reset 后，3300 iteration 稳定性测试可以跨过该位置。

首次正式运行仍沿用论文脚本的 `densify_until_iter=20000`，但该固定单目数据在 iteration 10000 后出现点数膨胀：cat 从 iteration 5000 的约 19.8k 增长至 325.8k，随后两个场景的显存快速升至 18-31 GB 且仍持续增长。最终运行因此在 normal regularization 启动前将 densification 截止设为 8000。失败运行和日志保留在 `*_uncapped_failed*`，不用于最终指标。

### 9.1 Cat

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --render_mode original \
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
  --test_iterations 1000 5000 10000 20000 35000 \
  --save_iterations 1000 10000 20000 35000 \
  --quiet
```

实际输出目录会自动追加 deformation type：

```text
output/LH-original/cat_baseline_mlp
```

### 9.2 Rubber duck

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage1 \
  --source_path data/LH-data/rubber_duck_toy_dataset/lumimotion_original \
  --model_path output/LH-original/rubber_duck_baseline \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --render_mode original \
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
  --test_iterations 1000 5000 10000 20000 35000 \
  --save_iterations 1000 10000 20000 35000 \
  --quiet
```

实际输出：

```text
output/LH-original/rubber_duck_baseline_mlp
```

## 10. 原版 Stage 2 训练

Stage 2 从 iteration 35000 的 Stage 1 checkpoint 继续训练到 iteration 55000，实际执行 20,000 次材质与光照训练。它会冻结几何主干，优化 Gaussian albedo、roughness、opacity、deformation color head 和全局 `EnvLight`。

本次沿用论文 synthetic 脚本的主要设置：

- `load_iter=35000`
- `iterations=55000`
- `diffuse_sample_num=512`
- `trace_num_rays=262144`
- `depth_ratio=0.0`

Cat：

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage2 \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 \
  --iterations 55000 \
  --load_iter 35000 \
  --diffuse_sample_num 512 \
  --trace_num_rays 262144 \
  --depth_ratio 0.0 \
  --test_iterations 40000 50000 55000 \
  --save_iterations 40000 50000 55000 \
  --quiet
```

Rubber duck：

```bash
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n lumimotion-cu129 \
python -m scripts.train_stage2 \
  --source_path data/LH-data/rubber_duck_toy_dataset/lumimotion_original \
  --model_path output/LH-original/rubber_duck_baseline \
  --train_light_folder images \
  --is_blender --eval --gt_alpha_mask_as_scene_mask \
  --resolution 2 \
  --iterations 55000 \
  --load_iter 35000 \
  --diffuse_sample_num 512 \
  --trace_num_rays 262144 \
  --depth_ratio 0.0 \
  --test_iterations 40000 50000 55000 \
  --save_iterations 40000 50000 55000 \
  --quiet
```

Stage 2 会在同一个模型目录中新增：

```text
point_cloud/iteration_55000/point_cloud.ply
deform/iteration_55000/deform.pth
envmap/iteration_55000/envmap.pth
envmap/iteration_55000/envmap.hdr
```

## 11. 统一渲染与评估

仓库原训练报告在最终 iteration 仍只计算前 4 个测试帧。为比较两阶段，本分支增加两个评估入口，对同一 `transforms_test.json` 中全部 15 帧计算：L1、PSNR、SSIM、LPIPS(VGG)、MS-SSIM 和 LPIPS(Alex)。两者都按 GT alpha mask 清理背景，并保存逐帧 JSON 和 `GT | Render | Absolute Error` 对比图。

Stage 2 评估还会按 frame id 精确读取原场景 `albedo/albedo_XXXX.png`，参照仓库原 material evaluator 在全部测试前景像素上计算 per-channel median scale，再对线性 RGB albedo 计算同样六项指标。该步骤解决了原脚本假定 GT 位于转换目录且使用文件名模糊匹配的问题。

Stage 1：

```bash
python -m scripts.eval_stage1_dynamic \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images --is_blender --eval \
  --resolution 2 --render_mode original --load_iter 35000 --quiet
```

Stage 2：

```bash
python -m scripts.eval_stage2_dynamic \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images --is_blender --eval \
  --resolution 2 --load_iter 55000 \
  --diffuse_sample_num 512 --trace_num_rays 262144 \
  --depth_ratio 0.0 --quiet
```

将 `cat` 路径替换为 `rubber_duck_toy_dataset`，并将模型路径替换为 `rubber_duck_baseline`，即可评估 duck。

完整 Stage 1 动画和 Stage 2 材质动画：

```bash
python -m scripts.render_stage1_insights ... --load_iter 35000
python -m scripts.render_materials ... --load_iter 55000 --depth_ratio 0.0
```

## 12. 输出说明

训练输出目录主要包含：

```text
output/LH-original/<scene>_mlp/
  cfg_args
  cameras.json
  input.ply
  events.out.tfevents.*
  point_cloud/
    iteration_*/point_cloud.ply
  deform/
    iteration_*/deform.pth
```

TensorBoard：

```bash
conda activate lumimotion-cu129
tensorboard --logdir output/LH-original --port 6006
```

## 13. 正式训练执行记录

本节在训练完成后记录最终 checkpoint、指标和异常。

评估语义需要特别注意：

- `utils/train_report_utils.py` 在非最终 test iteration 不读取 test cameras，只评估 3 个训练 camera/time 样本。
- 只有最后一个 `test_iterations` 才读取 `transforms_test.json`。
- 即使最终有 15 个测试帧，当前代码仍通过 `if idx > 3: break` 最多评估前 4 个测试帧。
- 因为所有 frame 使用同一个固定相机，最终 test 指标代表同视角时间留帧/插值，不代表 novel-view 指标。

iteration 1000 的中间 train subset 指标：

| 场景 | PSNR | SSIM | LPIPS | MS-SSIM | Alex-LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| cat | 17.94241 | 0.95484 | 0.05425 | 0.86443 | 0.14709 |
| rubber duck | 20.08021 | 0.96828 | 0.05074 | 0.90614 | 0.12524 |

| 场景 | Stage 1 | Stage 2 | Gaussian 数 | 输出目录 |
| --- | --- | --- | ---: | --- |
| cat | 完成，35000，约 62.0 分钟 | 完成，55000，约 61.6 分钟 | 93,751 | `output/LH-original/cat_baseline_mlp` |
| rubber duck | 完成，35000，约 52.8 分钟 | 完成，55000，约 30.4 分钟 | 19,994 | `output/LH-original/rubber_duck_baseline_mlp` |

两个场景均已生成 Stage 1/Stage 2 checkpoint、EnvLight、全 15 帧 RGB/albedo JSON 指标、Stage 1 诊断视频和 Stage 2 材质视频。最终数值、逐帧表和对比图见 `DOC/LH-data-two-stage-results.md`。

## 14. 结果解释边界

### 14.1 固定单目带来的限制

当前只有一个固定相机。虽然有 120 个时间帧，但每个时间点只有一个观测方向。因此：

- 模型可以拟合该相机下的时间序列。
- test split 只评估同一视角下的时间插值。
- 无法可靠评估 novel-view synthesis。
- 遮挡区域和物体背面的三维几何没有多视角约束。
- 学到的 Gaussian 几何可能是满足当前视图的退化解。

如果目标是稳定三维重建或新视角渲染，建议后续采集：

- 同一时间点的多相机同步图像，或
- 相机运动且相机轨迹已标定的数据。

### 14.2 移动光源带来的限制

原版 Stage 1 不使用 `lights.json`。逐帧移动光源产生的亮度和阴影变化可能被模型解释为：

- SH / color 变化。
- deformation color head 的 `d_color`。
- 几何或 opacity 变化。

所以原版训练结果不具备严格的 albedo-light 分离能力。它只能作为“不显式使用光源信息”的 baseline。

### 14.3 原版 Stage 2 不会直接使用点光源 JSON

LumiMotion 原版 Stage 2 面向 environment map relighting，renderer 使用一个可学习的全局 `EnvLight`。当前采集的是 per-frame point/area light position，不是原版 Stage 2 直接支持的 envmap 输入。

本次仍完整运行 Stage 2，目的是获得“未经点光源适配的原版算法 baseline”，验证材质分解和重建表现。但结果不能解释为正确使用了真实光源参数：逐帧光照变化仍可能被 deformation color head、albedo、roughness 或全局环境光共同吸收。若要建立物理一致的点光源模型，后续需要：

- 把点光源信息接入现有 photometric 分支。
- 将 point light 转成对应的方向/强度监督。
- 新增真正依赖 Gaussian 位置的 point-light direction 和距离衰减。

## 15. 后续建议

建议按以下顺序继续：

1. 检查本次 original 两阶段 baseline 的渲染与逐帧指标。
2. 检查是否存在明显漂浮 Gaussian、背景噪声和 mask 边缘问题。
3. 对比 image-background mask 与 albedo-alpha mask 的训练结果。
4. 检查 `object_pose.json` 中 duck 场景的对象命名。
5. 为 photometric 模式设计 point light 版本，而不是直接把当前 directional Lambertian 模型当作点光源模型。
6. 明确 normal EXR 坐标系后，再考虑 normal supervision 或 normal validation。
