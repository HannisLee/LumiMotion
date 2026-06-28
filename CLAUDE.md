# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

LumiMotion 是一个**可重光照的动态 2D 高斯泼溅(Gaussian Splatting)**研究代码库(论文标题:"Improving Gaussian Relighting with Scene Dynamics")。它在训练环境光照下重建动态场景,然后在未见过的环境光照下对场景重新打光。它基于三个上游项目构建:Dynamic-2DGS(Stage 1 几何)、IRGS + Relightable3DGaussian(Stage 2 材质)以及 2DGS 的 diff-surfel 光栅化器。

## 环境

Conda 环境为 **`lumimotion-cu129`**(CUDA 12.9)。运行任何命令前都需先激活:
```bash
conda activate lumimotion-cu129
```
README 中记录的是另一个 `lumimotion` 环境(Python 3.8 + CUDA 12.1 + torch 2.1.0)。这里实际使用的是 cu129 变体。

## 原生依赖与构建顺序

训练可用之前,必须按顺序构建/安装三个原生扩展(见 `readme.md` §2)。顺序和**确切版本**都很重要:

- `diff-surfel-rasterization` —— 2DGS surfel 光栅化器。**版本被锁定的原因**:较新版本会导致训练不稳定,而 DNA 数据集使用非针孔相机,需要比合成场景更新的 commit。仓库中自带的 vendored 副本位于 `submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization`。
- `submodules/simple-knn` —— `pip install ./submodules/simple-knn`(CUDA KNN)。
- `submodules/surfel_tracer` —— 必须**先用 cmake 编译**,再 pip 安装:`cd submodules/surfel_tracer && rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../ && cd ../../`,然后 `python -m pip install ./submodules/surfel_tracer`。仅 Stage 2 使用,用于光线追踪可见性/阴影。

如果光栅化器构建失败并报 glm 错误:`conda install -y glm -c conda-forge`(或 `apt install libglm-dev`)。

## 运行 —— 调用约定

所有入口都**从仓库根目录作为模块调用**,绝不要用 `python scripts/foo.py`:
```bash
python -m scripts.train_stage1 --source_path ... --model_path ...
```

本仓库没有测试套件 / lint 配置。`bash_scripts/*.sh` 是用于复现论文结果的规范流水线 —— 在自创命令之前,请先阅读它们以了解确切的标志组合。

### Stage1 光源方向导出

`photometric_lambertian` 的光源 checkpoint 在 `output/<exp>/photometric/iteration_<iter>/photometric.pth`。V1 使用 `raw_light_dir`，V2 可能使用 `light_model._raw_light_dir_table` 或 B-spline `light_model._light_ctrl`；渲染实际使用单位化方向。导出六列 CSV：

```bash
python -m LH_Utils.export_light_directions --model_path output/<exp> --iteration 35000
```

再用 CSV 解耦画图；如有原始 `lights.json`，传入后会把 `light_pos_world` 相对默认目标点 `[0,0,0]` 单位化并作为 GT 对比：

```bash
python -m LH_Utils.plot_light_polar --csv output/<exp>/light_directions.csv --lights_json data/LH-data/static/<scene>/lights.json
python -m LH_Utils.plot_light_timeseries --csv output/<exp>/light_directions.csv --lights_json data/LH-data/static/<scene>/lights.json
```

## 两阶段流水线

每个场景分两阶段训练,写入**同一个** `--model_path`。Stage 2 通过 `--load_iter` 加载 Stage 1 的检查点。

1. **Stage 1 —— `scripts/train_stage1.py`**(几何)。训练高斯几何、形变 MLP 以及**二值分离特征**(论文的核心动/静分割机制;见 `GaussianModel.get_binary_feature`)。使用 `gaussian_renderer/render.py`(diff-surfel-rasterization)。关键参数:`--binarization_warm_up`、`--lambda_separation`、`--d_xyz_loss_weight`、`--d_color_reg_loss_weight`。合成场景在此阶段使用 `--depth_ratio 1.0`。
2. **Stage 2 —— `scripts/train_stage2.py`**(材质)。加载 Stage 1(`--load_iter`),然后**只训练参数的一个子集**(albedo_dc、albedo_rest、roughness、opacity + 形变 MLP 的 `mlp_color` 头)以及一个可学习的 `EnvLight`。使用 `gaussian_renderer/render_ir.py:render_ir`(surfel tracer + 环境光)在屏幕空间光线模式下运行。从 35000 步的 Stage 1 检查点继续,运行 `--iterations 55000`。

随后是**渲染 + 评估**链,按固定顺序执行(必须先完成 Stage 2):`render_materials` → `scale_albedo_{static,dynamic}` → `eval_material_{static,dynamic}` → `eval_relight_{static,dynamic}` → `eval_nvs_{static,dynamic}`。`scale_albedo_*` 步骤**必须**在评估步骤之前运行。同时存在 `static`(单个固定时间步)和 `dynamic`(完整运动)两种变体。

所有合成场景的端到端流程在 `bash_scripts/synthetic_results_from_paper.sh` 中;ENeRF 和 DNA 场景各有自己的 `bash_scripts/*.sh`。

## 架构地图

- **`arguments/__init__.py`** —— 三个 argparse `ParamGroup`(`ModelParams`、`OptimizationParams`、`PipelineParams`)定义了所有 CLI 标志及其默认值。这是所有超参数的唯一事实来源。Stage 2 会**原地修改** `opt`(把学习率缩放为 0.1,并把优化器参数组过滤到与材质相关的子集)。
- **`scene/__init__.py`** —— `Scene` 通过检查源路径字符串来自动检测数据集格式:路径包含 `enerf` → `ColmapENerf`,包含 `dna` → `DNA-Rendering`,否则若存在 `transforms_train.json` → `Blender`。构建训练/测试相机列表,加载或创建高斯点云,并从相机 `fid` 计算 `all_timesteps`。
- **`scene/dataset_readers.py`** —— 三种格式的读取器(Blender / ENeRF-COLMAP / DNA-Rendering)。Blender 读取器使用按光照文件夹划分的子目录(环境贴图为 `<envmap_name>.hdr` + 对应的图像文件夹)。
- **`scene/gaussian_model.py`** —— `GaussianModel`。除了标准 2DGS 属性外,它还持有:每个高斯一个可学习的 `feature` 标量(**二值分离**的输入,通过 `get_binary_feature` 中固定 `T=0.5` 的直通 sigmoid 进行二值化)、albedo DC/rest SH、roughness,以及 Stage 2 光线追踪使用的 BVH(`build_bvh`/`update_bvh`/`trace`)。
- **`scene/deform_model.py` + `utils/time_utils.py`** —— `DeformModel` 封装 `DeformNetwork`(`mlp`)或 `StaticNetwork`(`static`)。形变 MLP 接收 (xyz, time, binary_feature, camera_center) 并输出 `d_xyz/d_rotation/d_scaling/d_opacity/d_color`。Stage 1 训练整个 MLP;Stage 2 只保留 `mlp_color` 头。
- **`scene/light.py`** —— `EnvLight`:用于重光照的可学习环境贴图(默认 32-px 立方体贴图,`exp` 激活)。与 Stage 2 检查点一起保存/加载。
- **`gaussian_renderer/__init__.py:render`** —— Stage 1 光栅化;同时返回几何正则化所使用的法线/深度/畸变图。
- **`gaussian_renderer/render_ir.py:render_ir`** —— Stage 2 基于物理的重光照渲染器(通过追踪光线的漫反射 + 镜面反射 + 间接光)。
- **`utils/`** —— 损失、SH、图形学、渲染和日志相关的辅助函数。`utils/train_report_utils.py` 在训练和评估期间计算 PSNR/SSIM/LPIPS/MS-SSIM。

## 关键 CLI 陷阱

- **`--model_path` 会被自动加上后缀** `_<deform_type>`(默认 `_mlp`)。传入 `--model_path outputs/x` 实际会写入 `outputs/x_mlp/`。每个下游的 stage/render/eval 都必须传入相同的原始路径 —— 后缀会在内部被重新添加。不要自己手动加后缀。
- **`--depth_ratio`**(一个 `PipelineParam`)用于混合期望深度(0.0)与中值深度(1.0),并**直接影响重光照质量** —— `arguments/` 中明确标注它"可能严重破坏重光照"。合成场景 Stage 1 使用 `1.0`;Stage 2/render/eval 会透传该值。
- **`--train_light_folder` / `--test_light_folder`** 指定用于训练 / 用于重光照评估的环境贴图子目录。评估步骤需要 `test_light_folder`。它们对应场景目录中的 `<envmap>.hdr` 文件。
- **`--load2gpu_on_the_fly`** —— 如果遇到 OOM 就启用;它会把图像保留在 CPU 上并按相机逐个传输。
- `--is_blender --eval --gt_alpha_mask_as_scene_mask --resolution 2` 是合成场景的标准标志(见 bash 脚本)。

## 数据布局

所有数据都放在 `data/` 下(已被 gitignore)。合成 Blender 场景遵循如下结构:
```
data/d-nerf-relight-spec32/<scene>/
  transforms_train.json  transforms_test.json  config.json  points3d.ply
  <envmap_name>/        # 在该环境光照下渲染的 RGB 图像(训练 + 测试光照集)
  <envmap_name>.hdr     # 对应的环境贴图
  albedo/  normal/  depth/  dynamic_mask/   # GT 监督
```
`data/d-nerf-relight-spec32/` 是来自 Zenodo 的合成数据集;`data/enerf_actors_1_3/`(ENeRF)和 `data/dna/`(DNA)不可再分发 —— `notebooks/` 文档说明了如何获取并预处理它们(`enerf_use_our_colmap.ipynb`、`dna_prepare_visualise_data.ipynb`)。`example_envmaps/` 存放示例 `.hdr` 文件。

## 子模块

`.gitmodules` 为空;三个子模块(`2dgs_rasterizer_lumimotion`、`simple-knn`、`surfel_tracer`)直接提交在仓库中。`submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization` 是为 ENeRF/合成场景锁定的旧版光栅化器;DNA 需要 README 中提到的较新 commit。
