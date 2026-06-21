# LumiMotion 中文工作指南

本文件记录 LumiMotion 项目的结构、两阶段训练流程、渲染评估顺序和常见注意事项。面向在本仓库中进行 baseline 复现、调试和小范围改动的开发者与 AI agents。

## 项目概述

LumiMotion 是一个可重光照的动态 2D Gaussian Splatting 研究代码库，论文为 “Improving Gaussian Relighting with Scene Dynamics”。整体目标是在训练环境光照下重建动态场景，并在未见过的环境光照下对场景重新打光。

代码主要由三部分组成：

- Stage 1：动态几何训练，继承 Dynamic-2DGS 思路。
- Stage 2：材质、粗糙度和环境光训练，参考 IRGS 与 Relightable3DGaussian。
- 渲染核心：2DGS surfel rasterizer、surfel tracer、nvdiffrast、PyTorch3D 等 CUDA/native 扩展。

## 环境与依赖

README 中记录的官方环境是 Python 3.8、CUDA 12.1 和 PyTorch 2.1.0。本地 baseline 环境为：

```bash
conda activate lumimotion-cu129
```

运行训练、渲染、评估或导入 native 扩展前，应确认当前环境中的 CUDA、PyTorch 和扩展版本。

官方依赖安装入口见 `readme.md`。Python 依赖集中在 `requirements.txt`，但 PyTorch、CUDA、rasterizer、nvdiffrast 和 PyTorch3D 应优先按 README 中的方式单独安装，以避免 CUDA ABI 不匹配。

## Native/CUDA 扩展

训练前通常需要安装或构建：

- `diff-surfel-rasterization`
- `submodules/simple-knn`
- `submodules/surfel_tracer`
- `nvdiffrast`
- `pytorch3d`

注意事项：

- rasterizer 版本会影响训练稳定性，不要随意升级。
- 合成数据和 ENeRF 可使用仓库内 vendored 版本：`submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization`。
- DNA 数据集使用非针孔相机模型，可能需要 README 中指定的新版本 rasterizer commit。
- `surfel_tracer` 需要先用 CMake 编译，再 `pip install`。
- rasterizer 构建失败并出现 glm 相关错误时，可安装 `glm`。

## 入口调用方式

从仓库根目录运行脚本，并优先使用模块方式：

```bash
python -m scripts.train_stage1 --source_path ... --model_path ...
python -m scripts.train_stage2 --source_path ... --model_path ... --load_iter ...
```

避免直接使用：

```bash
python scripts/train_stage1.py
```

除非已经确认当前 `PYTHONPATH` 和相对导入不会出问题。

## 两阶段训练流程

每个场景通常分两阶段训练，并写入同一个 `--model_path`。Stage 2 通过 `--load_iter` 加载 Stage 1 checkpoint。

### Stage 1：几何训练

入口：

```bash
python -m scripts.train_stage1 ...
```

主要训练内容：

- Gaussian geometry
- deformation MLP
- binary separation feature

相关重点文件：

- `scripts/train_stage1.py`
- `gaussian_renderer/__init__.py`
- `scene/gaussian_model.py`
- `scene/deform_model.py`
- `arguments/__init__.py`

Stage 1 使用 `gaussian_renderer/__init__.py:render` 进行光栅化，并返回几何正则化使用的法线、深度和畸变图。合成场景常见设置中，Stage 1 使用 `--depth_ratio 1.0`。

### Stage 2：材质与光照训练

入口：

```bash
python -m scripts.train_stage2 ... --load_iter ...
```

主要训练内容：

- albedo
- roughness
- opacity
- deformation MLP 的 color head
- `EnvLight`

相关重点文件：

- `scripts/train_stage2.py`
- `gaussian_renderer/render_ir.py`
- `scene/light.py`
- `scene/gaussian_model.py`

Stage 2 使用 `gaussian_renderer/render_ir.py:render_ir`，结合 surfel tracer 和环境光进行基于物理的重光照渲染。它依赖 Stage 1 checkpoint，通常继续使用同一个原始 `--model_path`。

## 渲染与评估顺序

完成 Stage 2 后，合成场景常见顺序为：

```text
render_materials
scale_albedo_static / scale_albedo_dynamic
eval_material_static / eval_material_dynamic
eval_relight_static / eval_relight_dynamic
eval_nvs_static / eval_nvs_dynamic
```

`scale_albedo_*` 必须在对应评估前运行。

优先参考现有 bash 脚本，而不是重新拼复杂命令：

- `bash_scripts/synthetic_results_from_paper.sh`
- `bash_scripts/enerf-actor1.sh`
- `bash_scripts/enerf-actor3.sh`
- `bash_scripts/dna_shoes.sh`
- `bash_scripts/dna_table.sh`
- `bash_scripts/dna_hairdryer.sh`

## 关键 CLI 陷阱

- `--model_path` 会被代码自动追加 deform 类型后缀，例如默认 `_mlp`。下游命令应继续传入相同原始路径，不要手动补后缀。
- `--depth_ratio` 会影响重光照质量。合成场景 Stage 1 常用 `--depth_ratio 1.0`。
- `--train_light_folder` 和 `--test_light_folder` 对应数据目录中的环境贴图和图像文件夹。
- OOM 时可考虑 `--load2gpu_on_the_fly`。
- 合成场景常见标志包括 `--is_blender --eval --gt_alpha_mask_as_scene_mask --resolution 2`。

## 数据目录

数据默认放在 `data/` 下，通常不提交到 git。

```text
data/
  d-nerf-relight-spec32/
  enerf_actors_1_3/
  dna/
```

合成 Blender 场景结构大致为：

```text
data/d-nerf-relight-spec32/<scene>/
  transforms_train.json
  transforms_test.json
  config.json
  points3d.ply
  <envmap_name>/
  <envmap_name>.hdr
  albedo/
  normal/
  depth/
  dynamic_mask/
```

ENeRF 和 DNA 数据通常不可再分发。相关说明在 `notebooks/` 中。

## 代码地图

- `arguments/__init__.py`：所有 CLI 参数和默认值。
- `scene/__init__.py`：数据集类型识别、相机列表和 Gaussian 初始化。
- `scene/dataset_readers.py`：Blender、ENeRF-COLMAP 和 DNA-Rendering 读取逻辑。
- `scene/gaussian_model.py`：GaussianModel、binary feature、材质参数和 BVH/trace。
- `scene/deform_model.py`：deformation network。
- `scene/light.py`：可学习环境光。
- `gaussian_renderer/__init__.py`：Stage 1 rasterization。
- `gaussian_renderer/render_ir.py`：Stage 2 relighting renderer。
- `utils/`：损失、图形学、渲染、日志和指标工具。
- `scripts/`：训练、渲染和评估入口。
- `bash_scripts/`：论文复现流水线。
- `notebooks/`：数据准备和结果整理辅助文档。

## 修改风险提示

- 改训练逻辑时，明确 checkpoint 兼容性风险。
- 改数据读取逻辑时，确认 Blender、ENeRF 和 DNA 三类路径不会互相破坏。
- 改 CUDA/native 相关代码时，说明需要重新编译哪些扩展。
- 对 baseline 有影响的实验性改动应放在新分支或清晰提交中。
