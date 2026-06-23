# LH 数据集转换、两阶段训练、评测与渲染

本文档总结如何将 `/home/han.li/reproduce/LumiMotion/data/LH-data` 中的两个场景转换为 LumiMotion 可读取的数据，完成 Stage 1、Stage 2 训练，并进行评测、材质渲染和指定相机视角渲染。

适用场景：

- `cat`
- `rubber_duck_toy_dataset`

完整实验指标与对比图见：

- `DOC/LH-data-two-stage-results.md`
- `DOC/LH-data-original-training.md`

## 1. 环境与输出约定

从仓库根目录执行：

```bash
cd /home/han.li/reproduce/LumiMotion
conda activate lumimotion-cu129
```

所有训练、评测、渲染、checkpoint 和日志统一放在 `output/` 下，不要在仓库根目录创建 `outputs_*` 等平行目录。

本次正式模型目录：

```text
output/LH-original/cat_baseline_mlp
output/LH-original/rubber_duck_baseline_mlp
```

LumiMotion 会根据 deformation type 自动给 `--model_path` 追加 `_mlp`。例如传入：

```text
--model_path output/LH-original/cat_baseline
```

实际目录是：

```text
output/LH-original/cat_baseline_mlp
```

## 2. 原始 LH 数据内容

每个场景原始目录为：

```text
data/LH-data/<scene>/
├── image/             # 120 帧带真实光照的 RGB/RGBA 图像
├── albedo/            # 120 帧 GT albedo PNG
├── normal_exr/        # 120 帧 GT normal EXR
├── camera.json        # 相机内参、外参和分辨率
├── lights.json        # 每帧点光源/面积光位置、颜色和强度
└── object_pose.json   # 每帧物体位姿
```

两个场景均为：

- 120 帧。
- 原始分辨率 `1280x720`。
- 固定单相机。
- 物体随时间运动。
- 光源随时间运动。
- RGB 图像背景接近恒定颜色 `[63, 63, 63]`。
- 原 RGB alpha 全为 255，不能直接用作 mask。

原版 LumiMotion Stage 1 实际使用：

- RGB 图像。
- 物体 alpha mask。
- 相机内外参。
- 归一化时间 `time`。

原版训练不会直接使用：

- `lights.json`。
- `object_pose.json`。
- GT normal EXR。
- GT albedo RGB。

本仓库的 Stage 2 评测脚本会额外读取父目录中的 GT albedo，用于材质定量评测，但 GT albedo 不参与训练。

## 3. 数据集转换脚本

转换脚本：

```text
data/LH-data/prepare_lumimotion.py
```

该脚本会：

1. 检查 RGB、albedo、normal、light 和 pose 是否都包含相同的 120 帧。
2. 从图像边界统计背景颜色。
3. 将与背景色最大通道差大于 1 的像素作为前景。
4. 生成黑背景 RGBA 图像，alpha 为二值物体 mask。
5. 将 `camera.json` 转换为 Blender transforms 格式。
6. 将 120 帧时间线映射到 `[0, 1]`。
7. 默认每第 8 帧作为 test，得到 105 个 train 帧和 15 个 test 帧。
8. 为固定相机写入 `camera_extent=1.0`，避免相机中心完全相同时归一化 radius 变成 0。

### 3.1 只检查，不写文件

```bash
python data/LH-data/prepare_lumimotion.py --validate-only
```

预期结果：

```text
cat: 120 frames, 105 train, 15 test, 1280x720
rubber_duck_toy_dataset: 120 frames, 105 train, 15 test, 1280x720
```

### 3.2 转换全部场景

```bash
python data/LH-data/prepare_lumimotion.py
```

不指定场景时，脚本自动扫描 `data/LH-data` 下包含 `camera.json` 的直接子目录。

### 3.3 转换单个场景

```bash
python data/LH-data/prepare_lumimotion.py data/LH-data/cat
```

### 3.4 已有转换结果时重新生成

```bash
python data/LH-data/prepare_lumimotion.py --overwrite
```

### 3.5 使用 albedo alpha 生成 mask

默认使用 RGB 恒定背景生成 mask。若要对照使用 albedo alpha：

```bash
python data/LH-data/prepare_lumimotion.py \
  --mask-source albedo-alpha \
  --output-name lumimotion_original_albedo_mask
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output-name` | `lumimotion_original` | 每个场景内的转换输出目录名 |
| `--test-stride` | `8` | 每第 N 帧作为测试帧，0 表示不划分测试集 |
| `--mask-source` | `image-background` | mask 来源，可选 RGB 背景、albedo alpha、image alpha |
| `--background-threshold` | `1` | 与背景色的最大通道差阈值 |
| `--background` | `black` | 转换后 RGB 背景颜色 |
| `--camera-extent` | `1.0` | 固定相机的非零场景归一化半径 |
| `--overwrite` | false | 覆盖已生成文件 |
| `--validate-only` | false | 只验证，不转换 |

## 4. 转换后目录含义

以 cat 为例：

```text
data/LH-data/cat/lumimotion_original/
├── images/
│   ├── frame_0001.png
│   ├── ...
│   └── frame_0120.png
├── transforms_train.json
├── transforms_test.json
├── dataset_manifest.json
└── points3d.ply
```

### 4.1 `images/`

- 共 120 张 `frame_XXXX.png`。
- RGB 背景已经清零为黑色。
- alpha 通道是从恒定背景提取的二值物体 mask。
- 训练时通过 `--train_light_folder images` 读取。

### 4.2 `transforms_train.json`

- 包含 105 个训练帧。
- 每一帧记录图像路径、归一化时间和相机 `camera-to-world` 4x4 矩阵。
- 顶层记录水平/垂直 FOV、焦距、主点、图像宽高和 `camera_extent`。

单帧结构示例：

```json
{
  "file_path": "frame_0001.png",
  "time": 0.0,
  "source_frame": 1,
  "transform_matrix": [[...], [...], [...], [...]]
}
```

### 4.3 `transforms_test.json`

- 包含 15 个保留测试帧：8、16、24、...、120。
- 时间仍按完整 120 帧序列归一化，不会因 train/test split 改变。
- 当前所有 test 帧仍是同一个固定相机，所以这是同视角时间插值测试，不是真正的 novel-view 测试。

### 4.4 `dataset_manifest.json`

记录：

- 原始场景路径。
- train/test 帧编号。
- mask 生成方式和检测到的背景颜色。
- 每帧前景像素数。
- 相机和原始 albedo、normal、light、pose 数据的位置。

该文件主要用于追踪转换过程，不直接参与训练。

### 4.5 `points3d.ply`

- 原始数据没有 COLMAP 点云。
- LumiMotion 第一次加载时会随机生成初始点云并保存为 `points3d.ply`。
- 后续训练复用该初始点云，保证转换目录结构完整。

## 5. Stage 1：几何与动态外观训练

Stage 1 训练：

- Gaussian 位置、尺度、旋转和 opacity。
- SH/颜色参数。
- 时间 deformation MLP。
- static/dynamic separation feature。

当前固定单目数据不能使用仓库默认的 opacity reset 和长时间 densification：默认配置曾在 iteration 3100 将 Gaussian 全部裁空，继续 densify 又会造成点数爆炸。因此正式命令使用：

```text
opacity_reset_interval = 100000
min_opacity             = 0.005
densify_until_iter      = 8000
```

### 5.1 Cat

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_stage1 \
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
  --test_iterations 1000 5000 10000 20000 30000 35000 \
  --save_iterations 1000 5000 10000 20000 30000 35000 \
  --quiet
```

### 5.2 Rubber duck

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.train_stage1 \
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
  --test_iterations 1000 5000 10000 20000 30000 35000 \
  --save_iterations 1000 5000 10000 20000 30000 35000 \
  --quiet
```

Stage 1 关键输出：

```text
<model>/point_cloud/iteration_35000/point_cloud.ply
<model>/deform/iteration_35000/deform.pth
<model>/events.out.tfevents.*
```

## 6. Stage 2：材质与环境光训练

Stage 2 从 iteration 35000 加载 Stage 1，继续训练到 55000，实际执行 20,000 次迭代。

Stage 2 主要优化：

- albedo。
- roughness。
- opacity。
- deformation MLP 的 color head。
- 全局可学习 `EnvLight`。

注意：原版 Stage 2 不读取 `lights.json`，其全局环境光无法表达当前逐帧移动点光源。这是 baseline，不是物理正确的点光源分解。

### 6.1 Cat

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_stage2 \
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

### 6.2 Rubber duck

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.train_stage2 \
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

Stage 2 关键输出：

```text
<model>/point_cloud/iteration_55000/point_cloud.ply
<model>/deform/iteration_55000/deform.pth
<model>/envmap/iteration_55000/envmap.pth
<model>/envmap/iteration_55000/envmap.hdr
```

## 7. 评测方法

训练内置 report 最终只检查前 4 个 test 帧。本分支增加的评测脚本会对全部 15 个 test 帧评测。

### 7.1 Stage 1 RGB 评测

Cat：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_stage1_dynamic \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --render_mode original --load_iter 35000 --quiet
```

Duck 只需替换：

```text
--source_path data/LH-data/rubber_duck_toy_dataset/lumimotion_original
--model_path output/LH-original/rubber_duck_baseline
```

输出：

```text
<model>/results_stage1_dynamic.json
<model>/eval_stage1_dynamic/ours_35000/
```

JSON 包含全部 15 帧和平均：

- L1。
- PSNR。
- SSIM。
- LPIPS-VGG。
- MS-SSIM。
- LPIPS-Alex。

逐帧目录包含 render、GT、mask、error、albedo、normal 和 `GT | Render | Error` 对比图。

### 7.2 Stage 2 RGB 与 albedo 评测

Cat：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_stage2_dynamic \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter 55000 \
  --diffuse_sample_num 512 \
  --trace_num_rays 262144 \
  --depth_ratio 0.0 --quiet
```

输出：

```text
<model>/results_stage2_dynamic.json
<model>/eval_stage2_dynamic/ours_55000/
```

该 JSON 同时包含：

- Stage 2 RGB 的六项指标。
- Stage 2 albedo 的六项指标。
- 每帧指标。
- albedo linear RGB per-channel median scale。

Albedo GT 从转换目录的父目录读取：

```text
data/LH-data/<scene>/albedo/albedo_XXXX.png
```

### 7.3 当前评测的含义

当前 15 个 test 帧全部来自同一个固定相机，因此指标衡量：

```text
同视角 + 未参与训练的时间帧插值
```

它不衡量：

```text
真正的新视角合成质量
```

## 8. 常规渲染

### 8.1 Stage 1 RGB、alpha、normal 和 separation 动画

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.render_stage1_insights \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter 35000 --depth_ratio 0.0 --quiet
```

输出：

```text
<model>/renders_stage1_insights/ours_35000/
```

包含 120 帧 PNG，以及 full render、alpha、normal、albedo 和 separation MP4。

### 8.2 Stage 2 albedo 和 roughness 动画

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.render_materials \
  --source_path data/LH-data/cat/lumimotion_original \
  --model_path output/LH-original/cat_baseline \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter 55000 --depth_ratio 0.0 --quiet
```

输出：

```text
<model>/trained_materials/ours_55000/
```

包含 120 帧 albedo、120 帧 roughness，以及对应 MP4。

## 9. 指定新视角渲染

### 9.1 重要限制

当前训练数据只有一个固定相机，没有任何训练期多视角约束。因此可以把 checkpoint 放到新的 camera-to-world 下渲染，但结果不一定具有正确的三维几何，不能把它当作可靠 novel-view reconstruction。

仓库目前没有独立的任意相机轨迹 CLI。现有渲染脚本都使用：

```python
scene.getTestCameras()[:1]
```

因此最小可运行方法是为渲染创建一个独立的 transforms 目录，并让 `transforms_test.json` 的第一条记录包含目标相机。

### 9.2 创建渲染专用数据目录

不要修改训练使用的 `lumimotion_original`。复制一份：

```bash
cp -a \
  data/LH-data/cat/lumimotion_original \
  data/LH-data/cat/lumimotion_novel_view
```

编辑：

```text
data/LH-data/cat/lumimotion_novel_view/transforms_test.json
```

要求：

1. 至少保留一条 frame。
2. 第一条 frame 的 `transform_matrix` 改为目标相机的 4x4 Blender camera-to-world 矩阵。
3. `file_path` 可以继续指向已有 `frame_XXXX.png`，因为 loader 需要用它确定分辨率；该图不用于 `render_stage1_insights` 的渲染颜色。
4. 保留顶层 `camera_angle_x`、`camera_angle_y`、`w`、`h` 和 `camera_extent`。
5. 目标相机必须沿用 `camera.json` 的坐标系约定：Blender camera-to-world，局部 `-Z` 为 forward，局部 `+Y` 为 up。

示例结构：

```json
{
  "camera_angle_x": 0.7610127542,
  "camera_angle_y": 0.4426288847,
  "w": 1280,
  "h": 720,
  "camera_extent": 1.0,
  "frames": [
    {
      "file_path": "frame_0008.png",
      "time": 0.0588235294,
      "source_frame": 8,
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

上面的单位矩阵仅展示 JSON 结构，不是推荐相机位置。实际矩阵应由目标 camera position、look-at point 和 up direction 正确计算。

下面的示例会根据 `eye/target/up` 生成符合 Blender `-Z forward、+Y up` 约定的 camera-to-world，并写入第一条 test frame。坐标必须与原始 `camera.json` 位于同一世界坐标系：

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

path = Path("data/LH-data/cat/lumimotion_novel_view/transforms_test.json")
data = json.loads(path.read_text())

# 示例：从物体另一侧观察原点。请按实际场景修改。
eye = np.array([-5.2, -6.2, 3.2], dtype=np.float64)
target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

forward = target - eye
forward /= np.linalg.norm(forward)
right = np.cross(forward, world_up)
right /= np.linalg.norm(right)
up = np.cross(right, forward)
up /= np.linalg.norm(up)

c2w = np.eye(4, dtype=np.float64)
c2w[:3, 0] = right
c2w[:3, 1] = up
c2w[:3, 2] = -forward
c2w[:3, 3] = eye

data["frames"] = [data["frames"][0]]
data["frames"][0]["transform_matrix"] = c2w.tolist()
path.write_text(json.dumps(data, indent=2) + "\n")
print(c2w)
PY
```

为了不覆盖正式 baseline 的渲染目录，再复制一份模型 checkpoint。新实验仍放在 `output/` 下：

```bash
mkdir -p output/LH-novel-view
cp -a \
  output/LH-original/cat_baseline_mlp \
  output/LH-novel-view/cat_view01_mlp
```

### 9.3 用 Stage 1 checkpoint 渲染新视角时间序列

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.render_stage1_insights \
  --source_path data/LH-data/cat/lumimotion_novel_view \
  --model_path output/LH-novel-view/cat_view01 \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter 35000 --depth_ratio 0.0 --quiet
```

脚本使用新目录中的第一台 test camera，并遍历训练数据中的全部时间值。输出仍位于模型目录：

```text
output/LH-novel-view/cat_view01_mlp/renders_stage1_insights/ours_35000/
```

该目录与正式 baseline 分离，不会覆盖原视角结果。

### 9.4 用 Stage 2 checkpoint 在新视角和指定 HDR 下渲染

Stage 2 可用 `render_relight_with_hdr` 在上述第一台 test camera 下遍历全部时间：

渲染脚本会读取可选的 `albedo_scale_linear_dynamic.json`。本分支的 scale 位于 `results_stage2_dynamic.json`，可先在新模型目录生成兼容文件：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("output/LH-novel-view/cat_view01_mlp")
results = json.loads((root / "results_stage2_dynamic.json").read_text())
scale = results["albedo_scale_rgb_linear"]
(root / "albedo_scale_linear_dynamic.json").write_text(
    json.dumps({"0": [1.0, 1.0, 1.0], "2": scale}, indent=2) + "\n"
)
PY
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.render_relight_with_hdr \
  --source_path data/LH-data/cat/lumimotion_novel_view \
  --model_path output/LH-novel-view/cat_view01 \
  --train_light_folder images \
  --is_blender --eval --resolution 2 \
  --load_iter 55000 \
  --depth_ratio 0.0 \
  --diffuse_sample_num 512 \
  --hdr /absolute/path/to/environment.hdr \
  --quiet
```

输出：

```text
<model>/render_relight_with_hdr_<hdr_name>/ours_55000/
```

包含每个时间点的 PNG 和 15 FPS MP4。

注意：

- 该命令使用外部 HDR，不使用 `lights.json` 中的逐帧点光源。
- 当前仓库没有直接把 `lights.json` 转成 per-Gaussian 点光源 shading 的新视角渲染器。
- 如果目标是按真实采集光源进行新视角渲染，需要实现基于动态 Gaussian 世界坐标的点光源方向和距离衰减。

## 10. 已完成结果位置

Cat：

```text
output/LH-original/cat_baseline_mlp/
├── point_cloud/iteration_35000/
├── point_cloud/iteration_55000/
├── deform/iteration_35000/
├── deform/iteration_55000/
├── envmap/iteration_55000/
├── eval_stage1_dynamic/ours_35000/
├── eval_stage2_dynamic/ours_55000/
├── renders_stage1_insights/ours_35000/
├── trained_materials/ours_55000/
├── results_stage1_dynamic.json
└── results_stage2_dynamic.json
```

Duck：

```text
output/LH-original/rubber_duck_baseline_mlp/
```

本次最终平均结果：

| 场景 | Stage 1 RGB PSNR | Stage 2 RGB PSNR | Stage 2 albedo PSNR |
| --- | ---: | ---: | ---: |
| Cat | 22.458 | 21.804 | 19.656 |
| Rubber duck | 42.325 | 32.680 | 22.840 |

Stage 2 RGB 指标下降的主要原因是原版全局 EnvLight 与当前逐帧移动点光源数据不匹配。该结果应作为“未使用真实光源信息的原版 LumiMotion baseline”。
