# Stage1 V3.2 photometric code 简要说明

本文简要记录 `PS-stage1-V3` 分支上 Stage1 V3.2 的 photometric 代码结构。训练参数见 `DOC/stage1-V3.2-参数.md`。

## 1. 版本目标

V3.2 的核心目标是删除 V2 B-spline / control points 曲线假设，改为逐帧直接学习 light direction，同时用正则和 multistart 控制抖动与起点不确定性。

```text
per-frame raw light table [T, 3]
        -> normalize
        -> Lambertian shading
        -> RGB reconstruction loss + light smooth losses
```

不改变：

- Lambertian 公式；
- Gaussian photometric albedo；
- normal 由 Gaussian rotation 得到；
- hemisphere prior / albedo prior；
- Stage1 的 densification、opacity、normal/dist loss 等基础训练逻辑。

## 2. 参数入口

文件：

```text
arguments/__init__.py
```

V3.2 关键默认值：

```python
photometric_light_param = "per_frame"
photometric_multistart_enabled = False
photometric_multistart_num_phases = 16
photometric_multistart_short_iters = 1000
lambda_photometric_light_smooth1 = 0.0
lambda_photometric_light_smooth2 = 0.0
```

说明：

- 代码默认不开 multistart，正式实验命令中显式传 `--photometric_multistart_enabled`。
- `photometric_num_ctrl_points` 仍可能作为 CLI 兼容参数存在，但 V3.2 active light model 不使用它。
- 推荐实验参数见 `DOC/stage1-V3.2-参数.md`。

## 3. Light model

文件：

```text
scene/photometric_lambertian.py
```

### DirectionalLightModel

V3.2 固定使用 per-frame light table：

```python
_raw_light_dir_table: [T, 3]
```

渲染时使用单位化方向：

```python
light_dirs = normalize(_raw_light_dir_table, dim=-1)
light_dir_t = light_dirs[timestep_index(fid)]
```

初始化仍使用圆形上半球轨迹：

```text
theta_t = sign * 2*pi*t/T + phase
raw_init_t = normalize([r_xy*cos(theta_t), r_xy*sin(theta_t), z])
```

### 已删除的 V2 曲线路径

V3.2 active model 不再包含：

```text
_light_ctrl
_bspline_raw_dirs()
periodic/uniform B-spline control point interpolation
```

也就是说，不再通过少量 control points 生成整段 light trajectory。

## 4. Smooth losses

文件：

```text
scene/photometric_lambertian.py
scripts/train_stage1.py
```

V3.2 使用双 L2 smooth：

```python
smooth1 = mean((l[t+1] - l[t]) ** 2)
smooth2 = mean((l[t+2] - 2*l[t+1] + l[t]) ** 2)
```

训练中对应：

```python
loss += lambda_photometric_light_smooth1 * smoothness_loss(order=1)
loss += lambda_photometric_light_smooth2 * smoothness_loss(order=2)
```

其中 `l[t]` 是 normalize 后的 light direction。

## 5. Multistart

文件：

```text
scripts/train_stage1.py
```

V3.2 在正式训练前可启用 light multistart：

```bash
--photometric_multistart_enabled
--photometric_multistart_num_phases 16
--photometric_multistart_short_iters 1000
```

流程：

1. 构造 16 个候选 phase：

```text
phase_k = 2*pi*k/16
```

2. 每个候选调用：

```python
light_model.reset_circle_init(phase=phase_k, direction_sign=sign)
```

3. 每个候选短跑若干 iteration，只优化 light model，不更新 Gaussian / deform。
4. 用候选短跑末段 photometric loss 均值打分。
5. 选择 loss 最低的候选 light table 作为正式训练初始值。
6. 将候选结果写入 photometric checkpoint 的 `multistart` metadata。

如果从已有 photometric checkpoint resume，multistart 会跳过，避免覆盖已训练 light。

## 6. Checkpoint 和导出

文件：

```text
scene/photometric_lambertian.py
```

保存内容：

```text
photometric/iteration_<iter>/photometric.pth
photometric/iteration_<iter>/light_dirs.npy
photometric/iteration_<iter>/light_dirs.json
```

checkpoint 版本：

```text
photometric_version = stage1_v3_directional_per_frame_light
state_dict.light_model._raw_light_dir_table
```

`light_dirs.json` 中同时保存：

- raw direction；
- normalize 后的 direction；
- timestep / fid；
- config；
- multistart metadata。

## 7. 与 V2 的主要差异

| 项目 | V2 B-spline | V3.2 |
| --- | --- | --- |
| light 参数 | `_light_ctrl [K,3]` | `_raw_light_dir_table [T,3]` |
| 轨迹生成 | B-spline 插值 | 每帧直接学习 |
| `photometric_num_ctrl_points` | 控制曲线容量 | 不参与 active model |
| smooth | 一阶/二阶 L2 | 一阶/二阶 L2 |
| 起点选择 | 独立工具或手动 phase | 训练前内置 multistart |
| 适用目标 | 平滑轨迹更稳定 | 不规则轨迹自由度更高 |

## 8. 已知边界

- V3.2 仍是 directional light，不是 point light。
- `lights.json` 不参与训练初始化，只用于训练后导出/画图/误差对比。
- 如果 smooth 权重太低，per-frame light 可能逐帧抖动。
- 如果 smooth 权重太高，可能重新限制不规则真实轨迹的拟合。
