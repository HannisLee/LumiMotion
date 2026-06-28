# Version 记录

本文档只记录当前仓库各实验版本的简要说明。详细实现、训练命令和评测结果放在 `DOC/` 下的专题文档中。

## Version 1：LH static Stage1 photometric Lambertian

当前分支：

```text
feature/photometric-lambertian-v1
```

当前已推送基线提交：

```text
7c8dff4 feat: document LH static stage1 photometric workflow and fix static deform loss
```

### 目标

Version 1 在不破坏 LumiMotion 原生 `render_mode="original"` 流程的前提下，加入并验证 Stage1 的 photometric Lambertian 路径：

```text
C_i(t) = albedo_i * max(0, normal_i_t · light_dir_t)
```

核心设定：

- 保留 LumiMotion 原始 Gaussian、deformation、densification、rasterizer 和 RGB reconstruction loss。
- 新增可选 `render_mode="photometric_lambertian"`。
- 每个 Gaussian 学习 diffuse albedo。
- 每帧学习 directional light direction，并在 forward 中单位化。
- Stage1 v1 使用 uniform light intensity，不建模 per-frame intensity、point light attenuation、shadow、BRDF 或 residual color。

### 主要改动

- `arguments/__init__.py`：增加 photometric 相关默认参数，默认不影响 original mode。
- `gaussian_renderer/__init__.py`：在 rasterizer 前增加 photometric color 分支。
- `scene/photometric_lambertian.py`：封装 Lambertian color、albedo、light direction 和相关正则/日志。
- `scripts/train_stage1.py`：接入 photometric renderer，并修复 `--deform-type static` 下 `d_xyz` 非 tensor 导致的 loss 崩溃。
- `data/LH-data/prepare_lumimotion.py`：支持 LH static 数据没有 `object_pose.json` 的情况。
- `LH_Utils/`：增加光源方向导出和可视化工具。

### LH static 实验

实验数据：

```text
data/LH-data/static
```

转换后数据：

```text
data/LH-data/transfer-static
```

训练输出：

```text
output/LH-static
```

已跑通 4 个 static 场景：

- `brass_vase`
- `concrete_cat`
- `garden_gnome`
- `rubber_duck_toy`

每个场景完成了标准 Stage1 `35000` iteration 的 photometric Lambertian 训练，并导出：

- checkpoint；
- Stage1 eval comparison；
- `light_tensor.txt`；
- `light_directions.csv`；
- light polar / time-series 可视化；
- GT normal visualization。

### 光源导出工具

导出六列 CSV：

```bash
python -m LH_Utils.export_light_directions \
  --model_path output/LH-static/concrete_cat_ps_stage1_static \
  --iteration 35000
```

画 learned light 与原始 `lights.json` 的极坐标对比：

```bash
python -m LH_Utils.plot_light_polar \
  --csv output/LH-static/concrete_cat_ps_stage1_static/light_directions.csv \
  --lights_json data/LH-data/static/concrete_cat/lights.json
```

画 learned light 与原始 `lights.json` 的 x/y/z 时间曲线：

```bash
python -m LH_Utils.plot_light_timeseries \
  --csv output/LH-static/concrete_cat_ps_stage1_static/light_directions.csv \
  --lights_json data/LH-data/static/concrete_cat/lights.json
```

### 详细文档

Version 1 的具体细节见：

- [DOC/LH-static-stage1V1.md](DOC/LH-static-stage1V1.md)：LH static 数据转换、Stage1 训练、评测、输出和指标记录。
- [DOC/LH-static-stage1V1-Code.md](DOC/LH-static-stage1V1-Code.md)：Stage1 original / photometric 两条代码路径和核心实现说明。

### 注意事项

- Version 2 后 `render_mode` 默认改为 `original_sh`，旧的 `original` 仍作为兼容别名；原版 LumiMotion 训练不需要额外 photometric 参数。
- 当前 photometric 路径没有使用 `normal_exr` 作为训练监督；GT normal 只用于可视化输出。
- 当前 `lights.json` 只用于导出后的可视化对比，不参与 Stage1 训练。
- `output/` 下的模型、评测图片和 CSV/PNG 分析结果不纳入 git。

## Version 2：Stage1 V2 photometric initialization

### 目标

Version 2 覆盖 Version 1 的 photometric 实现，把 Stage1 photometric 初始化升级为：

```text
C_i(t) = rho_i * max(0, normal_i_t · light_dir_t)
```

核心设定：

- 保留 LumiMotion 原始 deformation model 和 2DGS 动态几何。
- 原始模式使用 `render_mode="original_sh"`，旧 `"original"` 仍兼容。
- photometric 模式使用 per-Gaussian albedo、dynamic normal、learnable directional light。
- light intensity 固定为 1。
- 不加入 point light、attenuation、ambient、residual、BRDF、light MLP。

### 主要改动

- `scene/photometric_lambertian.py`：新增 V2 `DirectionalLightModel`，支持 `per_frame` 和 `bspline`；加入圆形上半球初始化、normal 轴开关、light smooth/hemi loss、light trajectory 保存。
- `gaussian_renderer/__init__.py`：`photometric_lambertian` 分支在 rasterizer 前用 Lambertian color 替换 SH color。
- `scripts/train_stage1.py`：新增 `--load_iter`，支持 `s1a_original_warmup`、`s1c_light_calib`、`s1d_joint` 三个 Stage1 子阶段和分组 lr 控制。
- `scripts/eval_stage1_dynamic.py`：支持 `original_sh` 和 V2 photometric checkpoint。
- `LH_Utils/export_light_directions.py`：支持 V2 B-spline checkpoint 导出 CSV。
- `LH_Utils/select_light_init.py`：新增 multistart light initialization utility。

### 跑通验证

已用以下数据做短迭代 smoke test：

```text
data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1
```

输出：

```text
output/LH-test/stage1v2_spheres_static
```

完成：

- S1A original_sh `iteration_2`
- S1C photometric light-only `iteration_4`
- S1D photometric joint `iteration_6`
- `eval_stage1_dynamic`
- light CSV export
- polar / time-series plot
- `select_light_init` multistart smoke

V2 checkpoint 已确认：

```text
photometric_version = stage1_v2_directional_uniform_light
light_param = bspline
num_ctrl_points = 8
light direction norm ≈ 1
```

### 详细文档

- [DOC/LH-static-stage1-V2.md](DOC/LH-static-stage1-V2.md)：V2 总体架构、训练命令、评测命令和 smoke test 结果。
- [DOC/LH-static-stage1-V2-Code.md](DOC/LH-static-stage1-V2-Code.md)：V2 代码路径、核心类函数、loss、checkpoint 和工具细节。
- [DOC/LH-static-stage1-V2-LH-staticv2-results.md](DOC/LH-static-stage1-V2-LH-staticv2-results.md)：新服务器 `minakshi` 上 `output/LH-staticv2` 的完整 LH-static V2 训练、评测和光源可视化结果。
