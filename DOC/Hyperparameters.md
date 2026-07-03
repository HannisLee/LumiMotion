# LumiMotion 原始 Baseline 训练超参参考

> 本文档记录**复现 LumiMotion 原始 baseline(非 photometric / 非 LH-data 修改版)**时,Stage 1 与 Stage 2 的实际推荐训练参数,并对每个参数给出讲解。
>
> **来源与可信度**:全部取自原始 baseline 的两个权威文件——
> - `arguments/__init__.py`(所有参数默认值的事实来源)
> - `bash_scripts/*.sh`(规范复现流水线,各数据集推荐标志组合)
>
> 已用 git 核对:`bash_scripts/` 与 `arguments/__init__.py` 在 baseline 提交 `c81b937 origin model save`(原作者 joaxkal 的最后一次提交)之后**没有任何改动**,与原始 LumiMotion 完全一致。本文不含 `DOC/` 中任何修改版(photometric / LH-data / V1·V2·V3)的内容。
>
> 原始 baseline 覆盖三类数据集:**合成(`d-nerf-relight-spec32`)/ ENeRF / DNA**,均使用标准 `original` 渲染模式 + 形变 MLP。表中"默认"表示该数据集脚本不显式传该参数,沿用 `arguments/__init__.py` 默认值。

---

## 目录
1. [公共 / 必传参数](#1-公共--必传参数)
2. [Stage 1 推荐参数](#2-stage-1-推荐参数)
3. [Stage 2 推荐参数](#3-stage-2-推荐参数)
4. [合成场景逐场景子配置](#4-合成场景逐场景子配置)
5. [可直接复制的命令](#5-可直接复制的命令)
6. [必读陷阱](#6-必读陷阱)

---

## 1. 公共 / 必传参数

两 stage、所有数据集共享。

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--source_path` | 必传 | 数据根目录 |
| `--model_path` | 必传 | 输出目录;会自动加 `_mlp` 后缀,下游所有命令传**相同原始路径**,勿手动加后缀 |
| `--resolution` | `2` | 渲染下采样倍率,三数据集统一用 2 |
| `--eval` | 传 | 划分训练/测试相机集 |
| `--gt_alpha_mask_as_scene_mask` | 传 | 用 GT alpha mask 当 scene mask(限定有效渲染区域) |
| `--is_blender` | 合成 ✓ / DNA ✓ / **ENeRF ✗** | Blender 数据集标志;ENeRF 走 COLMAP 读取器,不能开 |
| `--iterations` | Stage 1 一律 `35000` | Stage 1 总步数(几何训练) |
| `--train_light_folder` | 合成必传(选哪个光照训练);ENeRF/DNA 不用 | 训练光照子目录,对应场景里的 `<envmap>` 文件夹 |
| `--depth_ratio` | 见下 | 混合"期望深度(0.0)"与"中值深度(1.0)"的深度正则项 |

**`--depth_ratio` 专项**(直接影响重光照质量,原始代码注释标注"可能严重破坏 relighting"):
- **Stage 1**:合成 `1.0` / DNA `1.0` / **ENeRF `0.0`**
- **Stage 2 及所有 render/eval**:三类一律 `0.0`

合成/DNA 在 Stage 1 用 1.0 是让深度正则更贴中值深度、得到更干净的几何喂给 Stage 2;ENeRF 是真实捕获数据,用 0.0。

---

## 2. Stage 1 推荐参数

脚本:`python -m scripts.train_stage1`。
训练目标:高斯几何 + 形变 MLP + **二值分离特征**(论文核心的动/静分割机制,无监督)。

| 参数 | 合成 / ENeRF / DNA | 说明 |
|---|---|---|
| `--binarization_warm_up` | `1000` / `100` / `100` | 第 N 步才启用二值分离(动/静分割);此前让几何先稳定 |
| `--warm_up` | 默认 `1000` / `100` / `100` | 形变 MLP 预热步,在此之前形变接近冻结 |
| `--lambda_separation` | `0.001`或`0.005`(见 §4) / `0.0` / `0.005` | **论文核心:二值分离损失权重**,把每个高斯的 `feature` 推向 {0,1} 实现无监督动/静分割。越大分割越"硬";ENeRF 设 0 表示不强制二值化 |
| `--d_xyz_loss_weight` | `0.0`或`0.001`(见 §4) / 默认 `0.001` / 默认 `0.001` | 形变 MLP 输出位移 `d_xyz` 的 L2 正则权重,约束动态点不漂移过远 |
| `--d_color_reg_loss_weight` | `0.01` / `0.001` / `0.01` | 形变 MLP 输出颜色形变 `d_color` 的正则权重 |
| `--densify_until_iter` | `20000` / `18000` / `20000` | 自适应致密化(分裂/克隆)终止步 |
| `--densification_interval` | 默认 `100` / `200` / `100` | 致密化间隔步 |
| `--lambda_dist` | 默认 `1000` / `0` / `0` | 2DGS 畸变正则权重;真实数据(ENeRF/DNA)设 0,合成用 1000 |
| `--min_opacity` | 默认 `0.01` / `0.05` / `0.05` | 透明度剪枝/重置阈值 |
| `--lambda_alpha_loss` | 默认 `0.1` / `0.005` / 默认 `0.1` | alpha mask BCE 权重 |
| `--start_normal_reg` | 默认 `8000` / `6000` / `6000` | 从第 N 步开始启用法线正则 |
| `--start_frame` / `--end_frame` | — / — / `40` / `140` | DNA 专用,只取第 40–140 帧 |

> 合成 Stage 1 命令最"干净",只显式传少数几个参数,其余走默认——这是有意为之,合成即基线配置。

---

## 3. Stage 2 推荐参数

脚本:`python -m scripts.train_stage2`。
训练目标:加载 Stage 1 的 `--load_iter 35000` 检查点,**冻结几何**,只训练 albedo / roughness / opacity + 形变 MLP 颜色头 + 可学习环境贴图 `EnvLight`。

| 参数 | 合成 / ENeRF / DNA | 说明 |
|---|---|---|
| `--iterations` | `55000` / `45000` / `50000` | Stage 2 总步数(从 35000 续训) |
| `--load_iter` | 一律 `35000` | 加载 Stage 1 的检查点 |
| `--diffuse_sample_num` | `512` / `2048` / `512` | PBR 追踪里**漫反射入射光采样数**;训练用较小值提速,relight/nvs 评估统一用 `2048` |
| `--envmap_resolution` | 默认 `32` / `128` / `128` | 可学习 `EnvLight` 立方体贴图分辨率;真实数据用 128 捕捉更细光照 |
| `--envmap_cubemap_lr` | 默认 `0.1` / `0.1` / `0.1` | EnvLight 立方体贴图学习率 |
| `--albedo_lr` | 默认 `0.01` / `0.01` / `0.01` | albedo(漫反射色)学习率 |
| `--roughness_lr` | 默认 `0.002` / `0.0005` / `0.001` | 粗糙度学习率 |
| `--opacity_lr` | 默认 `0.05` / `0` / `0.1` | 透明度学习率;ENeRF 设 0 表示 Stage 2 不动 opacity |
| `--trace_num_rays` | 默认 `2^18·1` / `2^18·16` / `2^18·1` | 屏幕空间可见性/阴影光线追踪采样数;越大阴影越准但越慢(ENeRF 用 ×16,脚本注释说 ×4 也可) |
| `--d_lower_hemisphere_weight` | 默认 / `0.0` / `0.0` | 环境贴图下半球(几乎无监督)的权重,设 0 避免下半球亮斑 |
| `--white_background` | — / — / DNA 传 | 白色背景 |
| `--depth_ratio` | 三类一律 `0.0` | 见 §1 |

---

## 4. 合成场景逐场景子配置

合成数据集的 `lambda_separation` 与 `d_xyz_loss_weight` **因场景×光照组合而异**,是 `synthetic_results_from_paper.sh` 硬编码的查表值(原始 baseline 权威来源):

| 光照组合(train→test) | 场景 | `--lambda_separation` | `--d_xyz_loss_weight` |
|---|---|---|---|
| chapelday→goldenbay | spheres | 0.005 | 0.001 |
| | jumpingjacks | 0.001 | 0.001 |
| | hook | 0.001 | 0.001 |
| | standup | 0.005 | 0.0 |
| | mouse | 0.005 | 0.001 |
| damwall→harbour | spheres | 0.005 | 0.0 |
| | jumpingjacks | 0.001 | 0.0 |
| | hook | 0.001 | 0.001 |
| | standup | 0.001 | 0.001 |
| | mouse | 0.001 | 0.0 |
| goldenbay→damwall | spheres | 0.005 | 0.001 |
| | jumpingjacks | 0.001 | 0.0 |
| | hook | 0.001 | 0.001 |
| | standup | 0.001 | 0.001 |
| | mouse | 0.005 | 0.001 |

其余合成通用:`--d_color_reg_loss_weight 0.01`、`--binarization_warm_up 1000`、`--densify_until_iter 20000`、`--depth_ratio 1.0`。

---

## 5. 可直接复制的命令

### 5.1 合成(Stage 1 → Stage 2)

```bash
conda activate lumimotion-cu129

# Stage 1 — 几何(以 hook + chapelday→goldenbay 为例)
python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/hook150_v5_spec32 \
  --model_path outputs/hook \
  --is_blender --eval --gt_alpha_mask_as_scene_mask --resolution 2 \
  --iterations 35000 --train_light_folder chapel_day_4k_32x16_rot0 \
  --densify_until_iter 20000 \
  --lambda_separation 0.001 --d_xyz_loss_weight 0.001 \
  --binarization_warm_up 1000 --depth_ratio 1.0 --d_color_reg_loss_weight 0.01

# Stage 2 — 材质/重光照
python -m scripts.train_stage2 \
  --source_path data/d-nerf-relight-spec32/hook150_v5_spec32 \
  --model_path outputs/hook \
  --is_blender --eval --gt_alpha_mask_as_scene_mask --resolution 2 \
  --iterations 55000 --load_iter 35000 \
  --diffuse_sample_num 512 --train_light_folder chapel_day_4k_32x16_rot0 \
  --depth_ratio 0.0
```

### 5.2 ENeRF / DNA

完整命令分别见 `bash_scripts/enerf-actor1.sh`、`bash_scripts/dna_hairdryer.sh`(均为原始 baseline,可直接用)。

---

## 6. 必读陷阱

1. **`--model_path` 自动加后缀**:`deform_type=mlp` → 加 `_mlp`。传 `outputs/hook` 实际写到 `outputs/hook_mlp/`。**下游命令传相同的原始路径**,后缀在内部重新添加,绝不手动加。
2. **`--depth_ratio`**:Stage 1 合成/DNA=`1.0`、ENeRF=`0.0`;Stage 2/eval 全部=`0.0`。用错会"严重破坏重光照"。
3. 两阶段写**同一个** `--model_path`,Stage 2 通过 `--load_iter 35000` 读取 Stage 1 产物。
4. **NVS 评估注意**:训练时若改过 `--envmap_activation`/`--envmap_resolution` 等 opt 参数,`eval_nvs_*` 也必须传相同值(否则学到 EnvLight 的形状对不上)。原始 baseline 默认值无需额外处理。
5. 完整端到端流水线:`bash bash_scripts/synthetic_results_from_paper.sh`(合成);ENeRF/DNA 各有独立脚本。

---

## 附:参数默认值速查(`arguments/__init__.py`)

| 参数 | 默认值 | 所属 |
|---|---|---|
| `sh_degree` | `3` | ModelParams |
| `deform_type` | `mlp` | ModelParams(决定 model_path 后缀) |
| `hyper_dim` | `1` | ModelParams(动静分离变量维度) |
| `pred_color` | `True` | ModelParams |
| `depth_ratio` | `1.0` | PipelineParams |
| `diffuse_sample_num` | `256` | PipelineParams |
| `light_t_min` | `0.1` | PipelineParams |
| `iterations` | `80000` | OptimizationParams |
| `warm_up` | `1000` | OptimizationParams |
| `binarization_warm_up` | `1000` | OptimizationParams |
| `position_lr_init/final` | `0.00016 / 0.0000016` | OptimizationParams |
| `feature_lr` | `0.004` | OptimizationParams(feature = 二值分离变量) |
| `opacity_lr` | `0.05` | OptimizationParams |
| `roughness_lr` | `0.002` | OptimizationParams |
| `scaling_lr` | `0.002` | OptimizationParams |
| `rotation_lr` | `0.001` | OptimizationParams |
| `percent_dense` | `0.01` | OptimizationParams |
| `lambda_dssim` | `0.2` | OptimizationParams |
| `densification_interval` | `100` | OptimizationParams |
| `opacity_reset_interval` | `3000` | OptimizationParams |
| `densify_from_iter` | `500` | OptimizationParams |
| `densify_until_iter` | `15000` | OptimizationParams |
| `densify_grad_threshold` | `0.0002` | OptimizationParams |
| `min_opacity` | `0.01` | OptimizationParams |
| `start_normal_reg` | `8000` | OptimizationParams |
| `lambda_dist` | `1000` | OptimizationParams |
| `albedo_lr` | `0.01` | OptimizationParams |
| `albedo_rest_lr` | `0.0005` | OptimizationParams(= albedo_lr/20) |
| `envmap_cubemap_lr` | `0.1` | OptimizationParams |
| `lambda_separation` | `0.005` | OptimizationParams |
| `d_color_reg_loss_weight` | `0.01` | OptimizationParams |
| `d_xyz_loss_weight` | `0.001` | OptimizationParams |
| `d_lower_hemisphere_weight` | `0.00001` | OptimizationParams |
| `lambda_alpha_loss` | `0.1` | OptimizationParams |
| `envmap_resolution` | `32` | OptimizationParams |
| `envmap_init_value` | `1.5` | OptimizationParams |
| `envmap_activation` | `exp` | OptimizationParams |
| `trace_num_rays` | `262144` (= 2^18·1) | OptimizationParams |
