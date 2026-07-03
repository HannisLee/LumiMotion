# Stage1 photometric 版本说明

本文只记录 Stage1 `photometric_lambertian` 路线中的核心改动。具体命令、参数和实验结果仍以各版本专门文档为准。

## V1：最小 Lambertian photometric baseline

参考分支-目前没有

V1 的目标是先把 LumiMotion Stage1 的 appearance 路径替换成一个可选的 Lambertian photometric 路径，同时尽量不动原始 baseline。

核心改动：

- 新增 `render_mode="photometric_lambertian"`，默认原始路径仍保留。
- Stage1 中使用：

```text
color_i_t = albedo_i * light_rgb_t * max(0, normal_i_t dot light_dir_t)
```

- 每个 Gaussian 新增可学习 diffuse albedo。
- 光照采用 per-frame learnable light direction；早期 V1 还包含 per-frame `light_rgb_t`。
- normal 来自 Gaussian rotation 推出的局部 normal 轴。
- 不使用 GT `albedo/`、`normal_exr/`、`lights.json` 作为训练监督；这些主要用于后处理检查或后续版本对比。
- Stage2、deformation network、rasterizer 内部实现保持原样。

V1 的意义是打通“RGB 图像 -> Lambertian color -> Stage1 训练”的最小闭环，但 light/albedo/normal 的可辨识性和初始化稳定性还很弱。

## V2：photometric initialization 版本

参考分支-PS-stage1-V2

V2 在 V1 的基础上，把 photometric light 建模和训练流程系统化。V2 可以理解成两个子形态。

### V2.1：per-frame directional light table

这是 V2 的 per-frame 形态，也可以看作从 V1 过渡来的版本。

核心改动：

- 引入 `DirectionalLightModel`，把 light 相关逻辑从 renderer 中拆出来。
- 使用：

```text
photometric_light_param = "per_frame"
_raw_light_dir_table: [T, 3]
light_dir_t = normalize(_raw_light_dir_table[t])
```

- light intensity 固定为 1，不再依赖 V1 的 per-frame `light_rgb_t`。
- 初始化改为圆形上半球轨迹，由 `photometric_init_r_xy`、`photometric_init_z`、`photometric_init_phase`、`photometric_init_direction_sign` 控制。
- 增加 normal axis 配置：`photometric_normal_axis="+z"` / `"-z"`。
- 增加一阶/二阶 light smooth、upper-hemisphere prior、albedo prior。
- checkpoint 开始保存 light config 和 light trajectory。

这个形态的优点是每帧自由度足够；缺点是如果 smooth 权重不合适，容易出现逐帧抖动。

### V2.2：B-spline / control points directional light

这是 V2 后来推荐使用、并完成 LH-static 训练记录的形态。

核心改动：

- 新增曲线参数化：

```text
photometric_light_param = "bspline"
_light_ctrl: [K, 3]
K = photometric_num_ctrl_points
raw_light_dirs = B_spline(_light_ctrl)  # [T, 3]
light_dirs = normalize(raw_light_dirs)
```

- 通过少量 control points 生成完整光照轨迹，让 light trajectory 天然更平滑。
- 推荐配置中通常使用 `photometric_num_ctrl_points=16`。
- 引入 Stage1A / Stage1C / Stage1D 三阶段：
  - S1A：原始 SH appearance warm-up，先学几何。
  - S1C：冻结几何，只校准 directional light。
  - S1D：light / albedo / rotation / scale 小学习率联合微调。
- 提供独立 light initialization selection 工具 `LH_Utils.select_light_init`，用于尝试不同 phase / 方向后选择初始光照。
- `LH_Utils.export_light_directions` 支持导出 V1 raw light、V2 per-frame table、V2 B-spline control checkpoint。

这个形态的优点是稳定、平滑；缺点是如果真实光照轨迹不是简单平滑曲线，control points 的容量会限制拟合。

## V3 / V3.1：删除曲线，回到 per-frame + 正则

参考分支-PS-stage1-V3

V3 的目标是取消 V2-b 中的 B-spline 曲线假设，让光照轨迹回到逐帧可学习，但保留 V2 中有效的初始化、正则和诊断能力。

核心改动：

- `DirectionalLightModel` 固定为 per-frame light table：

```text
photometric_light_param = "per_frame"
_raw_light_dir_table: [T, 3]
light_dir_t = normalize(_raw_light_dir_table[t])
```

- 删除 active training model 中的：
  - `_light_ctrl`
  - B-spline 插值逻辑
  - `photometric_num_ctrl_points` 对训练行为的影响
- 恢复双 L2 smooth：
  - `lambda_photometric_light_smooth1`：相邻帧一阶 L2。
  - `lambda_photometric_light_smooth2`：二阶差分 L2。
- 保留圆形上半球初始化，但正式训练前加入 V3-compatible multistart：
  - 默认尝试 `photometric_multistart_num_phases=16` 个初始化相位。
  - 每个候选只短跑优化 per-frame light table。
  - 选择末段 photometric loss 最低的候选进入正式训练。
  - 不恢复 B-spline 或 control points。
- `iteration_1/photometric.pth` 保存 multistart 选中后的初始化 light table，便于后续导出检查。

V3 的预期收益是提高不规则真实光照轨迹的拟合能力；代价是需要依赖 smooth1/smooth2 和 multistart 控制逐帧抖动与起点不确定性。

