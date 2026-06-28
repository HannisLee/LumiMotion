# Stage 1 两种渲染路径代码梳理

本文梳理当前分支中 Stage 1 的两条代码路径：

- 原生 LumiMotion 路径：`render_mode="original"`
- Stage 1 v0 Lambertian 路径：`render_mode="photometric_lambertian"`

两条路径共用 LumiMotion 的数据读取、canonical Gaussian、deformation model、动态几何、opacity、scale、rotation、densification/pruning、RGB reconstruction loss 和 rasterizer。区别只发生在 Gaussian 颜色进入 rasterizer 之前。

## 入口参数

参数定义在 `arguments/__init__.py`：

```python
class PipelineParams(ParamGroup):
    self.render_mode = "original"
```

默认值是 `original`，所以不传 `--render_mode` 时，原始 LumiMotion 行为保持不变。

photometric 相关优化参数也在 `OptimizationParams` 中定义：

```python
self.photometric_albedo_lr = 0.01
self.photometric_light_lr = 0.01
self.lambda_photometric_light_smooth = 0.0
self.lambda_photometric_albedo_reg = 0.0
```

默认正则权重为 0，不影响原生训练。

## 共同训练主流程

训练入口是 `scripts/train_stage1.py`。

`Trainer.__init__()` 中会读取：

```python
self.render_mode = getattr(pipe, "render_mode", "original")
```

并限制只能是：

```python
["original", "photometric_lambertian"]
```

随后初始化：

1. `DeformModel`
2. `GaussianModel`
3. `Scene`
4. 背景色
5. optimizer

训练 step 中，两种路径都先执行相同的 deformation：

```python
d_values = self.deform.step(
    self.gaussians.get_xyz,
    time_input + ast_noise,
    iteration=self.iteration,
    feature=self.gaussians.get_binary_feature(eval=False, T=self.T_current),
    camera_center=viewpoint_cam.camera_center,
)
```

得到：

```python
d_xyz
d_rotation
d_scaling
d_opacity
d_color
```

然后统一调用：

```python
render_pkg_re = render(
    viewpoint_cam,
    self.gaussians,
    self.pipe,
    self.background,
    d_xyz,
    d_rotation,
    d_scaling,
    d_opacity=d_opacity,
    d_color=d_color,
    photometric_renderer=self.photometric_renderer,
)
```

真正的 original / photometric 分叉发生在 `gaussian_renderer/__init__.py::render()`。

## 路径一：原生 LumiMotion original

### 参数与颜色来源

原生 Gaussian 颜色参数在 `scene/gaussian_model.py` 中：

```python
self._albedo_dc
self._albedo_rest
```

创建点云时，`create_from_pcd()` 初始化：

```python
self._albedo_dc = nn.Parameter(...)
self._albedo_rest = nn.Parameter(...)
```

训练 optimizer 中加入：

```python
{"params": [self._albedo_dc], "lr": training_args.albedo_lr, "name": "albedo_dc"}
{"params": [self._albedo_rest], "lr": training_args.albedo_rest_lr, "name": "albedo_rest"}
```

原生颜色通过：

```python
pc.get_features
```

返回：

```python
torch.cat((albedo_dc, albedo_rest), dim=1)
```

### render() 内部颜色分支

在 `gaussian_renderer/__init__.py::render()` 中，如果没有 `override_color`，并且：

```python
render_mode != "photometric_lambertian"
```

则进入原生路径。

如果 deformation 输出了 `d_color`，代码会用它调制 SH DC：

```python
shadowed_modulation = d_color[:, None, :3].clamp_max(1.0)
final_color = RGB2SH(SH2RGB(pc.get_features[:, :1]) * (1 - shadowed_modulation))
sh_features = torch.cat([final_color, pc.get_features[:, 1:]], dim=1)
```

否则直接使用：

```python
sh_features = pc.get_features
```

之后分两种情况：

1. `pipe.convert_SHs_python=True`

   Python 侧把 SH 转成 RGB：

   ```python
   sh2rgb = eval_sh(...)
   colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
   ```

   最终传给 rasterizer 的是：

   ```python
   colors_precomp
   ```

2. `pipe.convert_SHs_python=False`

   不在 Python 侧转换，直接把 SH 交给 rasterizer：

   ```python
   shs = sh_features
   ```

   最终传给 rasterizer 的是：

   ```python
   shs
   ```

### 原生路径数据流

```text
camera + frame time
    ↓
DeformModel
    ↓
d_xyz, d_rotation, d_scaling, d_opacity, d_color
    ↓
GaussianModel.get_features
    ↓
SH / learned appearance color
    ↓
GaussianRasterizer(shs or colors_precomp)
    ↓
rendered RGB
    ↓
RGB loss + normal/dist/alpha/regularization
```

## 路径二：photometric_lambertian v0

### 额外参数

当 `render_mode="photometric_lambertian"` 时，`scripts/train_stage1.py::Trainer.__init__()` 会执行：

```python
self.gaussians.enable_photometric_albedo()
self.photometric_renderer = PhotometricLambertianRenderer(self.scene.all_timesteps, device="cuda")
self.photometric_renderer.training_setup(opt)
```

这会新增两类参数。

第一类是 per-Gaussian albedo，定义在 `scene/gaussian_model.py`：

```python
self._photometric_albedo
self._photometric_albedo_init
```

forward 取值为：

```python
torch.sigmoid(self._photometric_albedo)
```

初始化来自原始 base color：

```python
init_albedo = self.get_albedo.detach()
self._photometric_albedo = inverse_sigmoid(init_albedo)
```

optimizer 中加入：

```python
{"params": [self._photometric_albedo], "lr": training_args.photometric_albedo_lr, "name": "photometric_albedo"}
```

第二类是 per-frame directional light direction，定义在 `scene/photometric_lambertian.py`：

```python
self.raw_light_dir: [T, 3]
```

forward 中归一化：

```python
light_dir = F.normalize(self.raw_light_dir, dim=-1)
```

Stage 1 v0 固定光强为 1，不再学习 `raw_light_rgb`。为了日志兼容，`light_rgb` 属性返回固定的全 1 tensor，但它不是优化参数。

### dynamic normal

photometric 路径没有新增 normal 参数，而是使用 2DGS rotation 得到每个 Gaussian 的动态 normal。

在 `gaussian_renderer/__init__.py::render()` 中：

```python
normal_rotations = rotations
normal_i_t = build_rotation(normal_rotations)[:, :, 2]
```

含义是：

```text
n_i^t = R_i^t e_z
```

其中 `R_i^t` 来自 canonical rotation 加 deformation rotation bias 之后的动态 rotation。

### Lambertian color

`scene/photometric_lambertian.py::PhotometricLambertianRenderer.forward()` 输入：

```python
albedo
normal
fid
```

内部根据当前 `fid` 找到最近的 timestep index：

```python
timestep_idx = self.timestep_index(fid)
light_dir_t = self.light_dir[timestep_idx]
```

然后计算：

```python
normal_t = F.normalize(normal, dim=-1)
ndotl = torch.sum(normal_t * light_dir_t[None, :], dim=-1, keepdim=True).clamp_min(0.0)
color = albedo * ndotl
```

这就是 Stage 1 v0 的核心公式：

```text
C_i(t) = rho_i * max(0, n_i^t · l_t)
```

不包含：

- per-frame light intensity
- point light
- distance attenuation
- ambient
- residual color
- BRDF
- shadow

### render() 内部颜色分支

在 `gaussian_renderer/__init__.py::render()` 中，如果：

```python
override_color is None
and render_mode == "photometric_lambertian"
```

则进入 photometric 路径：

```python
photometric_outputs = photometric_renderer(
    pc.get_photometric_albedo,
    normal_i_t,
    viewpoint_camera.fid,
)
colors_precomp = photometric_outputs["color"]
```

这意味着 rasterizer 收到的是已经计算好的 RGB：

```python
GaussianRasterizer(colors_precomp=photometric_color, shs=None)
```

原始 SH 不再作为主颜色参与该次渲染，但 `_albedo_dc` / `_albedo_rest` 没有删除，original mode 仍可使用。

### photometric 路径数据流

```text
camera + frame time
    ↓
DeformModel
    ↓
d_xyz, d_rotation, d_scaling, d_opacity
    ↓
dynamic rotation R_i^t
    ↓
normal_i^t = R_i^t e_z
    ↓
GaussianModel.get_photometric_albedo = sigmoid(raw_albedo)
    ↓
PhotometricLambertianRenderer
    ↓
light_dir_t = normalize(raw_light_dir[t])
    ↓
ndotl = max(0, normal_i^t · light_dir_t)
    ↓
color_i(t) = albedo_i * ndotl
    ↓
GaussianRasterizer(colors_precomp=color_i(t))
    ↓
rendered RGB
    ↓
RGB loss + normal/dist/alpha loss + optional light/albedo regularization
```

## Loss 差异

两种路径都使用原始 Stage 1 image loss：

```python
Ll1 = l1_loss(image, gt_image)
loss_img = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
```

也都保留：

- normal regularization
- depth distortion loss
- alpha / mask loss
- deformation regularization
- densification / pruning

photometric 模式额外可选：

```python
loss_light_smooth = self.photometric_renderer.light_smoothness_loss()
loss_albedo_reg = self.gaussians.photometric_albedo_reg_loss()
```

对应权重：

```python
--lambda_photometric_light_smooth
--lambda_photometric_albedo_reg
```

当前 v0 的 light smoothness 只约束方向：

```python
mean(||l[t+1] - l[t]||^2)
```

不约束 light RGB，因为 v0 中 light RGB 固定为 1。

## 保存与加载

### Gaussian PLY

`scene/gaussian_model.py::save_ply()` 会保存：

```text
photometric_albedo_raw_0/1/2
photometric_albedo_init_0/1/2
```

加载时 `load_ply()` 会读回这些字段，并设置：

```python
self.use_photometric_albedo = True
```

这样 photometric eval/resume 能正确使用保存的 albedo。

### Light checkpoint

`scene/photometric_lambertian.py::save_weights()` 保存到：

```text
<model_path>/photometric/iteration_<iter>/photometric.pth
```

当前 v0 checkpoint 包含：

```text
state_dict:
  raw_light_dir
  timesteps
photometric_version: directional_uniform_light_v0
```

旧实验 checkpoint 里如果存在 `raw_light_rgb`，加载时会被忽略：

```python
state_dict.pop("raw_light_rgb", None)
```

## 评测路径

评测入口是 `scripts/eval_stage1_dynamic.py`。

两种模式共用：

```python
DeformModel.load_weights(...)
GaussianModel(...)
Scene(..., load_iteration=load_iter)
```

original mode 直接调用 `render()`。

photometric mode 会额外执行：

```python
gaussians.enable_photometric_albedo()
photometric_renderer = PhotometricLambertianRenderer(scene.all_timesteps, device="cuda")
photometric_renderer.load_weights(dataset.model_path, scene.loaded_iter)
photometric_renderer.eval()
```

然后调用：

```python
render(..., photometric_renderer=photometric_renderer)
```

评测输出：

```text
<model_path>/results_stage1_dynamic.json
<model_path>/eval_stage1_dynamic/ours_<iter>/*_render.png
<model_path>/eval_stage1_dynamic/ours_<iter>/*_gt.png
<model_path>/eval_stage1_dynamic/ours_<iter>/*_error.png
<model_path>/eval_stage1_dynamic/ours_<iter>/*_comparison.png
<model_path>/eval_stage1_dynamic/ours_<iter>/*_albedo.png
<model_path>/eval_stage1_dynamic/ours_<iter>/*_normal.png
```

其中 `comparison` 的布局是：

```text
[ground truth | render | absolute error]
```

## 当前 smoke test 对比

同一数据集：

```text
data/d-nerf-relight-spec32/hook150_v5_spec32
```

同样设置：

```text
resolution = 4
iterations = 2
test frames = 15
gaussians = 100000
```

结果：

| 模式 | 输出目录 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS-VGG ↓ | MS-SSIM ↑ | LPIPS-Alex ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| original | `output/stage1_v0_original_smoke_r4_mlp` | 0.070824 | 16.9400 | 0.823245 | 0.246524 | 0.742367 | 0.207393 |
| photometric v0 | `output/stage1_v0_lambertian_smoke_r4_mlp` | 0.070797 | 16.9433 | 0.823278 | 0.246602 | 0.742419 | 0.207334 |

这只是 2 iteration 的跑通测试，不能代表最终收敛质量。它说明两条路径在同一数据集、同一短跑设置下都可以完成训练、保存、加载、评测和渲染输出。

## 最小运行命令

原生路径：

```bash
conda run -n lumimotion-cu129 python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/hook150_v5_spec32 \
  --model_path output/stage1_original \
  --is_blender --eval --resolution 4 \
  --iterations 2 \
  --test_iterations 2 \
  --save_iterations 2 \
  --densify_until_iter 0
```

photometric v0 路径：

```bash
conda run -n lumimotion-cu129 python -m scripts.train_stage1 \
  --source_path data/d-nerf-relight-spec32/hook150_v5_spec32 \
  --model_path output/stage1_lambertian_v0 \
  --is_blender --eval --resolution 4 \
  --render_mode photometric_lambertian \
  --iterations 2 \
  --test_iterations 2 \
  --save_iterations 2 \
  --densify_until_iter 0 \
  --lambda_photometric_light_smooth 0.001 \
  --lambda_photometric_albedo_reg 0.0
```

评测原生路径：

```bash
conda run -n lumimotion-cu129 python -m scripts.eval_stage1_dynamic \
  --source_path data/d-nerf-relight-spec32/hook150_v5_spec32 \
  --model_path output/stage1_original_mlp \
  --is_blender --eval --resolution 4 \
  --render_mode original \
  --load_iter 2
```

评测 photometric v0 路径：

```bash
conda run -n lumimotion-cu129 python -m scripts.eval_stage1_dynamic \
  --source_path data/d-nerf-relight-spec32/hook150_v5_spec32 \
  --model_path output/stage1_lambertian_v0_mlp \
  --is_blender --eval --resolution 4 \
  --render_mode photometric_lambertian \
  --load_iter 2
```

注意：`ModelParams.extract()` 会根据 `--deform_type mlp` 自动把输出目录补成 `_mlp` 后缀。
