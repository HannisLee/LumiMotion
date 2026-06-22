# LumiMotion photometric_lambertian v1 执行文档

日期：2026-06-21

本文档记录本次在 LumiMotion 代码库中新增可选 `photometric_lambertian` 渲染模式的代码阅读结论、实现范围、数据流、配置参数、运行方式、验证结果和已知问题。

## 0. 分支信息与维护边界

本版本代码保存在独立分支：

```text
feature/photometric-lambertian-v1
```

分支创建信息：

- 创建日期：2026-06-22。
- 基线分支：`main`。
- 基线提交：`c81b937697a10f3b9b57a7783b23352e477a18cc`，提交说明为 `origin model save`。
- 创建分支时，本地 `main` 与 `origin/main` 指向同一基线提交。
- 本分支承载 photometric Lambertian v1 的代码、配置和执行文档。
- `main` 分支不包含本版本的未验证实验代码，也不会因为本分支的提交和推送而移动。

本分支的设计定位是“可选实验能力”，不是替换原始 LumiMotion baseline：

- 默认仍使用 `render_mode="original"`。
- 新模式必须通过 `render_mode="photometric_lambertian"` 显式开启。
- 本版只接入 Stage 1，Stage 2 保持原样。
- deformation network 和 Gaussian rasterizer 内部实现保持原样。
- 原始 RGB / SH / feature 参数与渲染路径均保留。

建议分支使用方式：

1. 复现原始 baseline 时继续使用 `main`。
2. 开展 Lambertian photometric 实验时切换到 `feature/photometric-lambertian-v1`。
3. 在完成 original / photometric 两条路径的端到端验证前，不建议把该分支直接合并回 `main`。
4. 后续 Stage 2、checkpoint resume、normal transport 或更复杂光照模型应继续使用单独提交或后续版本分支，避免扩大 v1 的职责范围。

## 1. 本次目标

本次目标是在不破坏原始 LumiMotion 训练流程的前提下，新增一个可选的 Lambertian 光照颜色路径：

```text
color_i_t = albedo_i * light_rgb_t * clamp(dot(normal_i_t, light_dir_t), min=0)
```

其中：

- `albedo_i`：每个 Gaussian 的 diffuse albedo，shape 为 `[N, 3]`，可学习。
- `normal_i_t`：第 `t` 帧下每个 Gaussian 的 dynamic normal，shape 为 `[N, 3]`。
- `light_dir_t`：第 `t` 帧的光照方向，shape 为 `[3]`，由 per-frame 可学习参数归一化得到。
- `light_rgb_t`：第 `t` 帧的 RGB 光照强度，shape 为 `[3]`，由 per-frame 可学习参数经过 `softplus` 得到正值。

核心约束：

- 默认模式必须仍然是原始 LumiMotion 行为。
- 新模式只在 `render_mode="photometric_lambertian"` 时启用。
- 不删除、不替换原始 RGB / SH / feature 参数。
- 不修改 Gaussian rasterizer 内部实现。
- 不修改 deformation network 结构。
- 不引入 Neural BRDF、shadow modeling 或外部 normal supervision。

## 2. 当前实现范围

本版实现为 Stage 1 训练路径的最小可运行集成：

- `scripts/train_stage1.py` 已接入 `photometric_lambertian`。
- `scripts/train_stage2.py` 未接入，保持原样。
- 原始 `render_mode="original"` 是默认值，旧 config 不需要新增字段也能继续运行。
- `override_color` 仍优先于 photometric 颜色路径，用于原有可视化和新增 albedo / shading / normal 可视化。

也就是说，本版用于在 Stage 1 中验证 Lambertian photometric color 的训练行为，不是完整替换 LumiMotion 两阶段训练体系。

## 3. 代码阅读定位结果

### 3.1 Gaussian 参数定义、初始化和优化

主要位置：`scene/gaussian_model.py`

Gaussian 参数集中定义在 `GaussianModel` 中，核心字段包括：

- `_xyz`：Gaussian 中心位置。
- `_albedo_dc`：SH DC / 颜色主分量。
- `_albedo_dc_stage1`：Stage 1 使用的 albedo DC 相关参数。
- `_albedo_rest`：SH 高阶颜色分量。
- `_opacity`：opacity 参数。
- `_roughness`：roughness 参数。
- `_scaling`：Gaussian scale 参数。
- `_rotation`：Gaussian rotation quaternion 参数。
- `_features`：feature 参数。

初始化主要在：

- `create_from_pcd(...)`：从 point cloud 初始化 Gaussian 参数。
- `load_ply(...)`：从已有 PLY checkpoint 读取参数。

优化器设置主要在：

- `training_setup(...)`：给不同 Gaussian 参数建立 optimizer param group，例如位置、颜色、opacity、scale、rotation 等。

本次新增的 photometric albedo 也放在 `GaussianModel` 中，确保它跟随原有 Gaussian 的 prune、densify、clone、split 生命周期变化。

### 3.2 Deformation model 如何把 canonical Gaussian 变换到时间 t

主要位置：

- `scripts/train_stage1.py`
- `scene/deform_model.py`
- `utils/time_utils.py`
- `gaussian_renderer/__init__.py`

Stage 1 训练中，每个 camera/viewpoint 带有 `fid`：

```python
fid = viewpoint_cam.fid
time_input = fid.unsqueeze(0).expand(N, -1)
d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, iteration=self.iteration)
```

deformation model 返回：

- `d_xyz`
- `d_rotation`
- `d_scaling`
- `d_opacity`
- `d_color`

这些 deformation 输出传入 `gaussian_renderer.render(...)` 后，在 renderer 内应用到 canonical Gaussian：

- `means3D = pc.get_xyz + d_xyz`
- `scales = pc.get_scaling + d_scaling`
- `rotations = pc.get_rotation_bias(d_rotation)`
- `opacity = pc.get_opacity + d_opacity`，如果 `d_opacity` 存在。

因此，当前 canonical Gaussian 到时间 `t` 的变换主要由 `fid -> deformation model -> d_* -> renderer` 完成。

### 3.3 每个 Gaussian 的颜色 / SH / feature 在哪里计算

主要位置：

- `scene/gaussian_model.py`
- `gaussian_renderer/__init__.py`
- `utils/time_utils.py`

原始颜色路径中，Gaussian 颜色来自 SH / feature：

- `pc.get_features` 返回 SH features。
- 如果 deformation model 预测 `d_color`，renderer 中会使用 `d_color` 对 SH DC 颜色做 modulation。
- 如果 `pipe.convert_SHs_python=True`，renderer 中使用 `eval_sh(...)` 在 Python 侧把 SH 转成 RGB。
- 如果 `pipe.convert_SHs_python=False`，则把 `shs` 传给 rasterizer，由 rasterizer 侧完成 SH 到 RGB 的处理。

renderer 中原始逻辑大致为：

```python
if d_color is not None:
    final_color = RGB2SH(SH2RGB(pc.get_features[:, :1]) * (1 - shadowed_modulation))
    sh_features = torch.cat([final_color, pc.get_features[:, 1:]], dim=1)
else:
    sh_features = pc.get_features
```

本次 photometric 模式不会删除或覆盖这些原始颜色参数，只在渲染时用 Lambertian 计算出的 `colors_precomp` 替换传入 rasterizer 的颜色输入。

### 3.4 调用 Gaussian rasterizer 前颜色 tensor 如何传入

主要位置：`gaussian_renderer/__init__.py`

调用 rasterizer 前有两条颜色输入路径：

1. `shs`：把 SH features 传给 rasterizer。
2. `colors_precomp`：把预计算 RGB 颜色直接传给 rasterizer。

原始模式下：

- 如果 `pipe.convert_SHs_python=False`，使用 `shs=sh_features`。
- 如果 `pipe.convert_SHs_python=True`，先在 Python 中 `eval_sh(...)`，再使用 `colors_precomp`。
- 如果外部传入 `override_color`，直接使用 `colors_precomp=override_color`。

新增模式下：

- 当 `render_mode="photometric_lambertian"` 且没有 `override_color` 时，renderer 计算 `photometric_outputs["color"]`。
- 然后设置：

```python
colors_precomp = photometric_outputs["color"]
shs = None
```

之后仍然调用原始 rasterizer：

```python
output = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=cov3D_precomp,
)
```

因此 rasterizer 内部没有修改。

### 3.5 当前是否已有 Gaussian normal

当前代码中存在多类 normal 相关逻辑：

- `gaussian_renderer/__init__.py` 从 rasterizer 返回的 `allmap` 中读取 `rend_normal`。
- `gaussian_renderer/__init__.py` 使用 depth 生成 `surf_normal`，用于 normal regularization。
- `utils/normal_utils.py` 中有 `compute_normal_world_space(...)`，report / 可视化路径会基于 quaternion 和 scale 计算 world-space normal。
- `scene/gaussian_model.py` 的 ray tracing 相关逻辑中也会从 Gaussian rotation / scale 推导 normal。

本次 photometric v1 选择最小改动实现，没有新增复杂 normal transport，也没有改 deformation model。当前 `normal_i_t` 的计算方式为：

```python
normal_i_t = build_rotation(rotations)[:, :, 2]
```

其中 `rotations` 已经包含当前时间下 deformation 后的 Gaussian rotation：

```python
rotations = pc.get_rotation_bias(d_rotation)
```

这等价于使用每个 Gaussian 当前 rotation matrix 的第 3 个局部轴作为 dynamic normal，并在 Lambertian forward 中再次 normalize。

## 4. 修改文件清单

本次代码改动涉及：

- `arguments/__init__.py`
- `scene/gaussian_model.py`
- `scene/photometric_lambertian.py`
- `gaussian_renderer/__init__.py`
- `scripts/train_stage1.py`
- `utils/train_report_utils.py`

本文档新增：

- `DOC/version01.md`

未修改：

- `scripts/train_stage2.py`
- deformation network 结构
- Gaussian rasterizer 内部实现

## 5. 新增配置参数

新增参数都带默认值，保证旧 config / 旧命令不传这些参数时仍可运行。

| 参数 | 默认值 | 位置 | 作用 |
| --- | --- | --- | --- |
| `render_mode` | `"original"` | `PipelineParams` | 渲染模式，支持 `"original"` 和 `"photometric_lambertian"` |
| `photometric_albedo_lr` | `0.01` | `OptimizationParams` | per-Gaussian photometric albedo 的学习率 |
| `photometric_light_lr` | `0.01` | `OptimizationParams` | per-frame light direction / RGB 的学习率 |
| `lambda_photometric_light_smooth` | `0.0` | `OptimizationParams` | light smoothness regularization 权重 |
| `lambda_photometric_albedo_reg` | `0.0` | `OptimizationParams` | albedo regularization 权重 |

默认权重均不改变原始训练目标：

- `render_mode="original"` 时，photometric 模块不会创建，也不会参与 forward / loss / optimizer。
- 两个 photometric loss 默认权重为 `0.0`，即使进入 photometric 模式，也需要显式打开正则项才会加入 loss。

## 6. 新增模块：PhotometricLambertianRenderer

新增文件：`scene/photometric_lambertian.py`

模块名：

```python
PhotometricLambertianRenderer
```

职责：

1. 保存每个 timestep 的可学习光照参数。
2. 根据当前 `fid` 找到最近 timestep。
3. 归一化 light direction。
4. 使用 `softplus` 约束 light RGB 为正。
5. 根据 Lambertian 公式输出 per-Gaussian RGB color。
6. 提供 light smoothness loss。
7. 提供 photometric checkpoint 保存 / 加载方法。

关键参数：

- `raw_light_dir`：shape `[T, 3]`，learnable。
- `raw_light_rgb`：shape `[T, 3]`，learnable。

forward 输入：

- `albedo`：shape `[N, 3]`。
- `normal`：shape `[N, 3]`。
- `fid`：当前 view 的时间 id。

forward 输出：

- `color`：shape `[N, 3]`。
- `normal`：normalize 后的 normal。
- `light_dir`：当前 timestep 的 normalize 光照方向。
- `light_rgb`：当前 timestep 的正值 RGB 光强。
- `ndotl`：shape `[N, 1]`。
- `timestep_idx`：当前使用的 timestep index。

计算逻辑：

```python
light_dir_t = normalize(raw_light_dir[t])
light_rgb_t = softplus(raw_light_rgb[t])
normal_t = normalize(normal_i_t)
ndotl = clamp(sum(normal_t * light_dir_t, dim=-1, keepdim=True), min=0)
color = albedo * light_rgb_t[None, :] * ndotl
```

## 7. per-Gaussian albedo 实现

主要位置：`scene/gaussian_model.py`

新增字段：

- `use_photometric_albedo`
- `_photometric_albedo`
- `_photometric_albedo_init`

新增方法：

- `enable_photometric_albedo(...)`
- `get_photometric_albedo`
- `get_photometric_albedo_init`
- `photometric_albedo_reg_loss(...)`

### 7.1 初始化策略

当进入 `photometric_lambertian` 模式时，Stage 1 初始化流程会调用：

```python
self.gaussians.enable_photometric_albedo()
```

默认从当前 Gaussian 的颜色初始化：

```python
init_albedo = self.get_albedo.detach()
```

`get_albedo` 对 `_albedo_dc` 做 `SH2RGB(...)`，因此能复用已有 SH DC / RGB 颜色作为 diffuse albedo 初值。

初始化时会 clamp 到合理范围，避免进入 `inverse_sigmoid` 时出现数值问题：

```python
init_albedo = init_albedo.clamp(1e-4, 1.0 - 1e-4)
```

真实可优化参数保存为 raw logits：

```python
_photometric_albedo = inverse_sigmoid(init_albedo)
```

forward 使用：

```python
get_photometric_albedo = sigmoid(_photometric_albedo)
```

因此 albedo 始终约束在 `[0, 1]`。

### 7.2 优化器接入

当 `use_photometric_albedo=True` 时，`GaussianModel.training_setup(...)` 会额外加入 optimizer param group：

```python
{
    "params": [_photometric_albedo],
    "lr": training_args.photometric_albedo_lr,
    "name": "photometric_albedo",
}
```

原始颜色参数仍然存在并保持可训练，`original` 模式仍走原始颜色路径。

### 7.3 densify / prune 生命周期

因为 LumiMotion 训练过程中会对 Gaussian 做 densify、clone、split、prune，本次实现把 photometric albedo 同步接入这些路径：

- prune 时按 `valid_points_mask` 同步裁剪 `_photometric_albedo` 和 `_photometric_albedo_init`。
- clone / split 时复制 selected Gaussian 对应的 photometric albedo 和 init albedo。
- densification post-fix 时把新 albedo 拼接进 optimizer 管理的 tensor。

这样 photometric albedo 的数量始终与 Gaussian 数量一致。

### 7.4 PLY 保存 / 加载

当 `use_photometric_albedo=True` 时，`save_ply(...)` 会额外写入：

- `photometric_albedo_raw_0`
- `photometric_albedo_raw_1`
- `photometric_albedo_raw_2`
- `photometric_albedo_init_0`
- `photometric_albedo_init_1`
- `photometric_albedo_init_2`

当 PLY 中存在这些字段时，`load_ply(...)` 会恢复 photometric albedo，并设置：

```python
use_photometric_albedo = True
```

如果没有这些字段，则旧 PLY 仍按原逻辑加载。

## 8. Stage 1 接入方式

主要位置：`scripts/train_stage1.py`

### 8.1 初始化阶段

训练器初始化时读取：

```python
self.render_mode = getattr(pipe, "render_mode", "original")
```

只允许：

- `"original"`
- `"photometric_lambertian"`

如果为 photometric 模式：

```python
self.gaussians.enable_photometric_albedo()
self.photometric_renderer = PhotometricLambertianRenderer(self.scene.all_timesteps, device="cuda")
self.photometric_renderer.training_setup(opt)
```

同时会打印：

```text
Render mode: photometric_lambertian
```

或：

```text
Render mode: original
```

### 8.2 forward 阶段

Stage 1 每次迭代：

1. 从 camera 读取 `fid`。
2. deformation model 根据 `fid` 输出 `d_xyz / d_rotation / d_scaling / d_opacity / d_color`。
3. 调用 `render(...)`。
4. 当 `render_mode="photometric_lambertian"` 时，传入 `photometric_renderer`。
5. renderer 用当前 rotation 计算 dynamic normal。
6. photometric renderer 计算 per-Gaussian RGB color。
7. 该 color 通过 `colors_precomp` 传给原始 rasterizer。

### 8.3 loss 阶段

原始 loss 仍保留：

- photometric / L1 / SSIM 图像重建相关 loss。
- normal regularization。
- depth distortion loss。
- deformation regularization。
- `d_color` regularization。

新增可选 loss：

```python
loss_light_smooth = photometric_renderer.light_smoothness_loss()
loss_albedo_reg = gaussians.photometric_albedo_reg_loss()
```

加入总 loss 的条件：

```python
if lambda_photometric_light_smooth > 0:
    loss += lambda_photometric_light_smooth * loss_light_smooth

if lambda_photometric_albedo_reg > 0:
    loss += lambda_photometric_albedo_reg * loss_albedo_reg
```

### 8.4 optimizer step

Stage 1 原有 optimizer step 保持不变。

当 photometric renderer 存在时，额外执行：

```python
self.photometric_renderer.optimizer.step()
self.photometric_renderer.optimizer.zero_grad(set_to_none=True)
```

photometric albedo 自身在 `GaussianModel.optimizer` 里，因此跟随 Gaussian optimizer step。

### 8.5 checkpoint 保存

原始保存路径保持：

- `point_cloud/iteration_*/point_cloud.ply`
- `deform/iteration_*/deform.pth`

photometric 模式额外保存：

- `photometric/iteration_*/photometric.pth`

其中 `photometric.pth` 包含：

- `PhotometricLambertianRenderer.state_dict()`
- `timesteps`

## 9. renderer 数据流

### 9.1 original 模式

```text
Gaussian SH / feature
    + optional d_color modulation
    -> sh_features
    -> either eval_sh in Python or pass SH to rasterizer
    -> GaussianRasterizer
    -> rendered image
```

此路径与原始 LumiMotion 行为一致。

### 9.2 photometric_lambertian 模式

```text
viewpoint_cam.fid
    -> deformation model
    -> d_xyz / d_rotation / d_scaling / d_opacity / d_color

canonical Gaussian rotation + d_rotation
    -> rotations
    -> build_rotation(rotations)[:, :, 2]
    -> normal_i_t

Gaussian SH DC / RGB init
    -> photometric_albedo raw logits
    -> sigmoid
    -> albedo_i

viewpoint_cam.fid
    -> nearest timestep index
    -> normalize(raw_light_dir[t])
    -> softplus(raw_light_rgb[t])

albedo_i, normal_i_t, light_dir_t, light_rgb_t
    -> ndotl
    -> photometric color_i_t
    -> colors_precomp
    -> GaussianRasterizer
    -> rendered image
```

公式：

```text
normal_i_t = normalize(normal_i_t)
light_dir_t = normalize(light_dir_t)
light_rgb_t = softplus(raw_light_rgb_t)
ndotl = clamp(sum(normal_i_t * light_dir_t), min=0)
color_i_t = albedo_i * light_rgb_t * ndotl
```

## 10. 日志和可视化

### 10.1 scalar logging

主要位置：`scripts/train_stage1.py`

在 `photometric_lambertian` 模式且 TensorBoard writer 存在时，记录：

- `photometric/render_mode`
- `photometric/light_dir_mean`
- `photometric/light_dir_norm`
- `photometric/light_rgb_mean`
- `photometric/light_rgb_min`
- `photometric/light_rgb_max`
- `photometric/albedo_mean`
- `photometric/albedo_min`
- `photometric/albedo_max`
- `photometric/normal_norm_mean`
- `photometric/ndotl_mean`
- `photometric/ndotl_min`
- `photometric/ndotl_max`
- `photometric/color_mean`
- `photometric/color_min`
- `photometric/color_max`
- `photometric/loss_light_smooth`
- `photometric/loss_albedo_reg`

### 10.2 image logging

主要位置：`utils/train_report_utils.py`

在已有 validation / report image logging 中，photometric 模式额外输出：

- `photometric_render`：当前 photometric 渲染 RGB。
- `photometric_albedo`：把 per-Gaussian albedo 作为 `override_color` 重新 rasterize 的可视化。
- `photometric_shading`：把 `ndotl` repeat 到 RGB 后作为 `override_color` 重新 rasterize。
- `photometric_normal`：把 normalized normal 从 `[-1, 1]` 映射到 `[0, 1]` 后作为 `override_color` 重新 rasterize。

注意：这些图像日志依赖已有 report / test iteration 机制，不是每一步都一定输出。

## 11. 如何运行 original mode

因为默认值就是 `render_mode="original"`，旧命令不需要变化。

示例：

```bash
conda activate lumimotion-cu129

python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/spheres_v5_stage1_original \
  --is_blender \
  --eval
```

也可以显式指定：

```bash
python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/spheres_v5_stage1_original \
  --is_blender \
  --eval \
  --render_mode original
```

original mode 行为：

- 不创建 `PhotometricLambertianRenderer`。
- 不启用 photometric albedo。
- 不写入 photometric checkpoint。
- rasterizer 颜色输入仍来自原始 RGB / SH / feature 路径。

## 12. 如何运行 photometric_lambertian mode

示例：

```bash
conda activate lumimotion-cu129

python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/spheres_v5_spec32_statictimestep1 \
  --model_path output/spheres_v5_stage1_photometric \
  --is_blender \
  --eval \
  --render_mode photometric_lambertian \
  --photometric_albedo_lr 0.01 \
  --photometric_light_lr 0.01 \
  --lambda_photometric_light_smooth 1e-4 \
  --lambda_photometric_albedo_reg 1e-3
```

建议初始实验：

- 先用较小分辨率或较少 iteration 做 smoke test。
- 正则权重先从小值开始，例如：
  - `lambda_photometric_light_smooth=1e-4`
  - `lambda_photometric_albedo_reg=1e-3`
- 如果只想检查 photometric 数据流能否跑通，可以先把两个正则权重都设为 `0.0`。

photometric mode 输出：

- 原始 Stage 1 输出仍在 `model_path` 下。
- Gaussian PLY 会包含 photometric albedo 字段。
- 额外保存：

```text
model_path/photometric/iteration_*/photometric.pth
```

## 13. 验证记录

### 13.1 语法检查

已执行：

```bash
conda run -n lumimotion-cu129 python -m py_compile \
  arguments/__init__.py \
  scene/photometric_lambertian.py \
  scene/gaussian_model.py \
  gaussian_renderer/__init__.py \
  scripts/train_stage1.py \
  utils/train_report_utils.py
```

结果：通过。

### 13.2 diff 空白检查

已执行：

```bash
git diff --check
```

结果：通过。

### 13.3 CLI 参数检查

已执行：

```bash
conda run -n lumimotion-cu129 python -m scripts.train_stage1 --help | \
  rg "render_mode|photometric|lambda_photometric"
```

结果：能看到新增参数。

执行过程中出现的 `kornia` / `torchvision` warning 与本次改动无关。

### 13.4 Photometric 模块 tensor smoke test

已执行：

```bash
conda run -n lumimotion-cu129 python -c "\
import torch; \
from scene.photometric_lambertian import PhotometricLambertianRenderer; \
device='cuda' if torch.cuda.is_available() else 'cpu'; \
r=PhotometricLambertianRenderer(torch.tensor([0.0,0.5,1.0]), device=device); \
albedo=torch.full((5,3),0.5,device=device,requires_grad=True); \
normal=torch.nn.functional.normalize(torch.randn(5,3,device=device),dim=-1); \
out=r(albedo,normal,torch.tensor([0.5],device=device)); \
loss=out['color'].mean()+r.light_smoothness_loss(); \
loss.backward(); \
print(out['color'].shape, out['ndotl'].shape, round(float(r.light_dir.norm(dim=-1).mean()),6), bool((r.light_rgb>0).all()))"
```

输出：

```text
torch.Size([5, 3]) torch.Size([5, 1]) 1.0 True
```

含义：

- 输出 color shape 正确。
- 输出 ndotl shape 正确。
- light direction norm 为 1。
- light RGB 为正。
- backward 可执行。

### 13.5 训练 smoke test 状态

尝试运行 1 iteration 训练 smoke test 时，在进入训练前被数据加载逻辑阻断：

```text
TypeError: Cannot handle this data type: (1, 1, 4), |i1
```

定位到：

```text
scene/dataset_readers.py:201
```

触发代码中使用：

```python
np.array(arr_train_light * 255.0, dtype=np.byte)
```

`np.byte` 是 signed int8，Pillow 对 RGBA int8 array 处理失败。该问题发生在原始数据加载路径，且在 original mode smoke test 中即出现，因此不是 photometric 改动引入的问题。

建议后续单独修复为 `np.uint8` 后，再重新跑 original 和 photometric 两个 smoke test。

## 14. 向后兼容性说明

本次实现采取以下方式保持兼容：

1. `render_mode` 默认是 `"original"`。
2. old config 不包含新增参数时不会报错。
3. original mode 不创建 photometric renderer。
4. original mode 不启用 `_photometric_albedo`。
5. original mode 不改变 rasterizer 颜色输入。
6. 原始 SH / RGB / feature 参数没有删除。
7. `override_color` 优先级保持最高。
8. 未修改 Stage 2 训练脚本。
9. 未修改 deformation model 结构。
10. 未修改 rasterizer 内部实现。

## 15. 当前已知限制和风险

### 15.1 normal 计算是 v1 简化版

当前 photometric normal 使用：

```python
build_rotation(rotations)[:, :, 2]
```

优点是实现简单，并且使用了当前 timestep 的 deformation rotation。

限制是：

- 没有做 Jacobian normal transport。
- 没有额外处理 normal 朝向翻转。
- 没有结合 view direction 做 two-sided shading。
- 当 Gaussian 局部 z 轴与真实表面法线不一致时，Lambertian shading 会受影响。

### 15.2 光照模型是单方向 Lambertian

当前只实现：

- single directional light。
- per-frame light direction。
- per-frame RGB intensity。
- diffuse albedo。

没有实现：

- specular。
- BRDF。
- shadow。
- visibility。
- environment map。
- multi-light。

### 15.3 Stage 2 尚未接入

当前只接入 `scripts/train_stage1.py`。

如果后续需要完整两阶段 photometric 训练，需要另行设计 Stage 2 中的：

- photometric renderer 创建 / 加载。
- Stage 2 render 调用传参。
- Stage 2 optimizer step。
- Stage 2 checkpoint resume。
- Stage 2 loss 和 logging。

### 15.4 photometric checkpoint resume 需要后续补齐

`PhotometricLambertianRenderer` 已提供：

- `save_weights(...)`
- `load_weights(...)`

但当前 Stage 1 训练启动流程主要覆盖从头训练路径。若需要从已有 photometric checkpoint 恢复继续训练，需要在 train script 的 resume / load iteration 流程中显式调用 photometric renderer 的 `load_weights(...)`，并确认 Gaussian PLY 中 photometric albedo 字段同步加载。

### 15.5 数据加载 smoke test 有原始路径问题

当前 1 iteration smoke test 被 `scene/dataset_readers.py:201` 的 `np.byte` / Pillow RGBA 类型问题阻断。建议优先单独修复该问题，否则无法用该数据集完成 end-to-end 训练验证。

## 16. 建议后续执行顺序

1. 单独修复数据加载中的 `np.byte` 问题，建议改为 `np.uint8` 并确认 original mode 能跑 1 iteration。
2. 用 original mode 跑短训练，确认 baseline 行为不变。
3. 用 `photometric_lambertian` 且两个正则权重为 `0.0` 跑短训练，确认 forward / backward / checkpoint / logging 全部正常。
4. 打开较小正则权重，观察：
   - `photometric/light_rgb_*`
   - `photometric/light_dir_norm`
   - `photometric/albedo_*`
   - `photometric/ndotl_*`
   - `photometric/color_*`
5. 检查 image logging 中的：
   - `photometric_render`
   - `photometric_albedo`
   - `photometric_shading`
   - `photometric_normal`
6. 如果 normal visualization 明显方向错误，再考虑：
   - normal 翻转策略。
   - 基于已有 `compute_normal_world_space(...)` 的一致化。
   - 更完整的 normal transport。

## 17. 快速排查清单

如果 original mode 行为异常：

- 确认没有传 `--render_mode photometric_lambertian`。
- 确认输出日志打印 `Render mode: original`。
- 确认没有生成 `model_path/photometric/...`。

如果 photometric mode 报缺少 renderer：

- 确认 `scripts/train_stage1.py` 中创建了 `PhotometricLambertianRenderer`。
- 确认调用 `render(...)` 时传入了 `photometric_renderer=self.photometric_renderer`。

如果 albedo shape 和 Gaussian 数量不一致：

- 检查 densify / prune 路径是否被其他改动绕过。
- 检查 `_photometric_albedo` 是否在 optimizer param group 中。

如果 light RGB 出现负值：

- 检查是否使用了 `photometric_renderer.light_rgb` property，而不是直接使用 `raw_light_rgb`。
- 当前实现通过 `softplus(raw_light_rgb)` 保证正值。

如果 normal norm 不接近 1：

- 检查 `photometric/normal_norm_mean`。
- 当前 Lambertian forward 中会对 normal 再次 `normalize`。

## 18. 总结

本次 v1 改动完成了一个保守的 Stage 1 photometric Lambertian 实验路径：

- 原始模式默认不变。
- 新模式显式开启。
- albedo、light direction、light RGB 可学习。
- dynamic normal 来自当前 deformation 后的 Gaussian rotation。
- photometric color 通过 `colors_precomp` 接入原始 rasterizer。
- 新增 loss、scalar logging 和 image visualization。

当前最主要的未解决问题不是 photometric 模块本身，而是数据加载路径中的 Pillow / `np.byte` 类型问题导致 end-to-end 训练 smoke test 尚未完成。修复该 baseline 问题后，应优先对 original 和 photometric 两种模式分别跑短训练验证。
