# Stage1 V2 photometric code 改动细节

本文记录 Stage1 V2 photometric initialization 的代码级数据流。总体实验和命令见 `DOC/LH-static-stage1-V2.md`。

## 1. 参数入口

文件：

```text
arguments/__init__.py
```

### render mode

默认模式改为：

```python
render_mode = "original_sh"
```

兼容旧值：

```python
"original" -> "original_sh"
```

因此不启用 photometric 时仍然走原始 LumiMotion SH / appearance 路径。

### photometric 参数

新增核心参数：

```python
photometric_stage = "s1d_joint"
photometric_light_param = "bspline"
photometric_num_ctrl_points = 16
photometric_init_r_xy = 0.8
photometric_init_z = 0.6
photometric_init_phase = 0.0
photometric_init_direction_sign = 1
photometric_normal_axis = "+z"
photometric_use_hemi_prior = False
photometric_hemi_axis = "0,0,1"
photometric_hemi_margin = 0.0
```

新增 loss 权重：

```python
lambda_photometric_light_smooth1 = 0.0
lambda_photometric_light_smooth2 = 0.0
lambda_photometric_hemi = 0.0
lambda_photometric_albedo_prior = 0.0
```

保留旧兼容参数：

```python
lambda_photometric_light_smooth
lambda_photometric_albedo_reg
```

训练时会把旧 smooth/albedo 权重作为 V2 smooth1/albedo prior 的兼容别名。

## 2. Photometric 模块

文件：

```text
scene/photometric_lambertian.py
```

### DirectionalLightModel

V2 新增：

```python
class DirectionalLightModel(nn.Module)
```

支持两种 light 参数化：

```python
photometric_light_param = "per_frame"
photometric_light_param = "bspline"
```

#### per_frame

参数：

```python
_raw_light_dir_table: [T, 3]
```

forward：

```python
light_dir_t = normalize(_raw_light_dir_table[t])
```

#### bspline

参数：

```python
_light_ctrl: [K, 3]
```

其中 `K = photometric_num_ctrl_points`。

V2 通过 uniform cubic B-spline basis 从 control points 插值得到每个 timestep 的 raw direction：

```python
raw_light_dirs = B_spline(_light_ctrl)  # [T, 3]
light_dirs = normalize(raw_light_dirs)
```

接口：

```python
get_all_raw_light_dirs()  # [T, 3]
get_all_light_dirs()      # [T, 3], normalized
forward(frame_id)         # [3] or [B, 3]
```

### circle upper-hemisphere init

函数：

```python
circle_upper_hemisphere_init(...)
```

初始化公式：

```text
theta_t = sign * 2*pi*t/T + phase
light_init = normalize([r_xy*cos(theta_t), r_xy*sin(theta_t), z])
```

`per_frame` 直接初始化 `[T, 3]`；`bspline` 在 control point 时间位置采样同一条圆形轨迹。

### normal 计算

函数：

```python
get_gaussian_normal(rotation_t, normal_axis="+z")
```

支持：

```text
quaternion: [..., 4]
rotation matrix: [..., 3, 3]
```

默认：

```text
normal_i_t = R_i_t @ [0, 0, 1]
```

如果设置：

```bash
--photometric_normal_axis -z
```

则使用：

```text
normal_i_t = R_i_t @ [0, 0, -1]
```

当前代码注释明确：LumiMotion 的 Gaussian rotation 按 world-space rotation 传给 rasterizer，因此 V2 directional light 也按 world-space direction 建模。

### PhotometricLambertianRenderer

V2 renderer 输入：

```python
albedo: [N, 3]
normal: [N, 3]
fid: frame/time id
```

计算：

```python
light_dir = normalize(light_model(fid))
normal = normalize(normal)
ndotl = sum(normal * light_dir[None, :], dim=-1, keepdim=True)
shading = clamp(ndotl, min=0)
color = clamp(albedo * shading, 0, 1)
```

返回 aux：

```python
{
    "color",
    "albedo",
    "normal",
    "light_dir",
    "ndotl",
    "shading",
    "timestep_idx",
}
```

没有 `light_rgb`、没有 intensity、没有 ambient、没有 residual。

## 3. Gaussian albedo

文件：

```text
scene/gaussian_model.py
```

V1 已加入的 photometric albedo 继续保留并作为 V2 albedo：

```python
_photometric_albedo: [N, 3]       # raw logits
_photometric_albedo_init: [N, 3]  # init target
```

forward：

```python
get_photometric_albedo = sigmoid(_photometric_albedo)
```

初始化：

```python
init_albedo = get_albedo
get_albedo = clamp(SH2RGB(_albedo_dc), 0.03, 0.97)
_photometric_albedo = inverse_sigmoid(init_albedo)
_photometric_albedo_init = init_albedo.detach().clone()
```

因此 albedo 不是随机初始化，而是来自原始 SH DC / base color。

### densification / pruning

V1 已完成同步，V2 继续使用：

- prune 时同步 `_photometric_albedo` 和 `_photometric_albedo_init`；
- clone 时继承 parent albedo；
- split 时继承 parent albedo；
- optimizer state 通过现有 `cat_tensors_to_optimizer` / `_prune_optimizer` 更新。

## 4. Rasterizer 接入

文件：

```text
gaussian_renderer/__init__.py
```

### original_sh 路径

当：

```python
render_mode == "original_sh"
```

执行原始路径：

```text
SH / feature -> colors_precomp or shs -> GaussianRasterizer
```

旧值：

```python
render_mode == "original"
```

会被规范化为：

```python
original_sh
```

### photometric_lambertian 路径

当：

```python
render_mode == "photometric_lambertian"
```

且没有 `override_color` 时：

```python
normal_i_t = get_gaussian_normal(dynamic_rotation, normal_axis)
photometric_outputs = photometric_renderer(
    gaussians.get_photometric_albedo,
    normal_i_t,
    viewpoint_camera.fid,
)
colors_precomp = photometric_outputs["color"]
```

然后仍然调用原始 2DGS rasterizer：

```python
GaussianRasterizer(
    means3D,
    means2D,
    colors_precomp=colors_precomp,
    opacities,
    scales,
    rotations,
)
```

没有修改 rasterizer 内部实现。

## 5. Stage1 训练入口

文件：

```text
scripts/train_stage1.py
```

### checkpoint 加载

新增：

```bash
--load_iter <iter>
```

如果提供：

```python
Scene(..., load_iteration=load_iter)
DeformModel.load_weights(..., iteration=load_iter)
```

如果 photometric checkpoint 存在，也会加载：

```text
photometric/iteration_<iter>/photometric.pth
```

### stage 控制

`photometric_stage=s1a_original_warmup`：

```python
render_mode = "original_sh"
```

不创建 photometric renderer。

`photometric_stage=s1c_light_calib`：

```text
freeze Gaussian geometry
freeze deformation
freeze SH
optimize light model
optional albedo lr, default 0
disable densification
```

`photometric_stage=s1d_joint`：

```text
light small lr
photometric albedo small lr
rotation small lr
scale small lr
position default 0
deformation default 0
SH default 0
```

学习率通过 optimizer param group name 覆盖：

```python
xyz
albedo_dc
albedo_rest
opacity
roughness
scaling
rotation
feature
photometric_albedo
photometric_light
```

### V2 losses

在原始 RGB / mask / normal / dist loss 基础上，photometric 模式增加：

```python
loss_light_smooth1 = mean((l[t+1] - l[t]) ** 2)
loss_light_smooth2 = mean((l[t+1] - 2*l[t] + l[t-1]) ** 2)
loss_hemi = mean(relu(hemi_margin - dot(l[t], hemi_axis)) ** 2)
loss_albedo_prior = mean(abs(albedo - albedo_init))
```

权重：

```python
lambda_photometric_light_smooth1
lambda_photometric_light_smooth2
lambda_photometric_hemi
lambda_photometric_albedo_prior
```

photometric mode 不再使用 deformation color residual，也不对 `d_color` 加正则。

### logging

TensorBoard 中新增：

```text
photometric/render_mode
photometric/stage
photometric/light_dir_norm_min/max/mean
photometric/light_dir_x/y/z_mean
photometric/albedo_min/max/mean
photometric/normal_norm_mean
photometric/ndotl_min/max/mean
photometric/shading_min/max/mean
photometric/color_min/max/mean
photometric/loss_light_smooth1
photometric/loss_light_smooth2
photometric/loss_hemi
photometric/loss_albedo_prior
```

保存 photometric checkpoint 时，同时写：

```text
photometric/iteration_<iter>/photometric.pth
photometric/iteration_<iter>/light_dirs.npy
photometric/iteration_<iter>/light_dirs.json
```

## 6. Eval 和工具

### eval_stage1_dynamic

文件：

```text
scripts/eval_stage1_dynamic.py
```

变更：

- 支持 `original_sh`；
- 旧 `original` 自动映射到 `original_sh`；
- 支持 `--deform-type` 别名；
- photometric 模式下从 checkpoint config 恢复 B-spline/per-frame light model。

### export_light_directions

文件：

```text
LH_Utils/export_light_directions.py
```

V2 支持三种 checkpoint：

```text
V1 raw_light_dir
V2 light_model._raw_light_dir_table
V2 light_model._light_ctrl
```

对 B-spline checkpoint，工具会重建 `DirectionalLightModel`，导出每个 timestep 的插值 raw direction 和 normalize 后 direction。

CSV 固定六列：

```text
raw_x, raw_y, raw_z, dir_x, dir_y, dir_z
```

### select_light_init

文件：

```text
LH_Utils/select_light_init.py
```

新增独立 multistart utility：

```bash
python -m LH_Utils.select_light_init \
  --source_path <dataset> \
  --model_path <stage1a_model> \
  --load_iter <stage1a_iter> \
  --num_phases 8 \
  --try_reverse_direction \
  --short_iters 1000
```

流程：

1. 加载 S1A Gaussian / deformation checkpoint；
2. 冻结 geometry、deformation、SH、albedo；
3. 枚举 phase 和 direction sign；
4. 每个 candidate 只优化 light model；
5. 记录 photometric RGB L1；
6. 保存最优 light 初始化。

输出：

```text
photometric_multistart/iteration_<iter>/best_light_init.pth
photometric_multistart/iteration_<iter>/best_light_init.json
```

## 7. 已验证项

静态编译：

```bash
python -m py_compile \
  scene/photometric_lambertian.py \
  scripts/train_stage1.py \
  scripts/eval_stage1_dynamic.py \
  LH_Utils/export_light_directions.py \
  LH_Utils/plot_light_polar.py \
  LH_Utils/plot_light_timeseries.py \
  LH_Utils/select_light_init.py
```

CLI：

```bash
python -m scripts.train_stage1 --help
python -m LH_Utils.export_light_directions --help
python -m LH_Utils.select_light_init --help
```

Tensor smoke：

```text
per_frame light norm OK
bspline light norm OK
smooth1 / smooth2 / hemi loss backward OK
+z / -z normal sign OK
Lambertian color shape OK
```

End-to-end smoke：

```text
S1A original_sh iteration_2 OK
S1C photometric_lambertian iteration_4 OK
S1D photometric_lambertian iteration_6 OK
eval_stage1_dynamic OK
light CSV export OK
polar plot OK
time-series plot OK
select_light_init multistart OK
```

## 8. 尚未解决 / 后续建议

- 尚未跑标准 35000 iteration，因此当前 `output/LH-test` 只证明代码链路可运行。
- `select_light_init.py` 已通过最小 multistart smoke；后续正式使用时应把 `num_phases` 提到 8 或 16，把 `short_iters` 提到 500 到 1000。
- V2 未把 GT lights.json 纳入训练，只作为导出后可视化对比。
- V2 未引入 point light / attenuation / shadow / residual / BRDF，这些应放到后续版本。
