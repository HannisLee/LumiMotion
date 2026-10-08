# Stage2 Albedo 真值评测与材质均衡指标

日期：2026-10-03；服务器：`venus`；环境：`lumimotion`；代码目录：`/home/lihan/reproduce/LumiMotion-stage2-perlight`。

本次增加独立的 albedo 评测入口，以未缩放的线性材质误差评估分离质量。已完成 only_clothV4 的 120 帧真值审计、1002-02 的四个 checkpoint，以及 1002-03/04 两组最终权重的对照。未重新训练，所有来源权重哈希在评测前后保持一致。

完整命令、JSON/CSV、浮点缓存、图片、日志和源码快照均保存在 [评测实验目录](../../../output/1003-06-onlyclothV4-albedo_GT评测/README.md)。`PASS` 表示评测流程和数值验证通过；材质质量尚未设定验收阈值。

## 1. 实际管线与权重存储

1. Blender 转换集 `transforms_train/test.json` 提供每帧相机、FOV 与 time，`images/frame_XXXX.png` 提供 RGB/RGBA。数据 loader 使用这些相机和 time；本次核验 105 个训练帧、15 个测试帧，无重叠。GT albedo RGB 和 shader Base Color 未作为这些模型的训练监督。
2. Stage1 的 `gaussian_renderer.render` 使用 SH 和 MLP 的颜色调制拟合 RGB，并学习 Gaussian 几何、opacity、二值动静分离与形变。因此 SH 的 DC 初始颜色包含外观和光照，不是已分离的物理材质。
3. 原始 Stage2 在 `scripts/train_stage2.py` 中优化 `albedo_dc`、`albedo_dc_stage1`、`albedo_rest`、roughness、opacity 和 EnvLight；GGX/漫反射、光线追踪及 SH 外观辅助损失共同参与 RGB 拟合。源码还保留 MLP color 优化器，但形变调用在 `torch.no_grad()` 内。本次逐 tensor 比较确认 35k→40k/50k/55k 的整个 deform 网络，包括 color head，变化为零；PLY 的 xyz/scale/rotation/feature 也不变，opacity 与 albedo 有变化。证据为 `original_parameter_audit.json`。
4. 新 `photometric_lambertian` Stage2 在 `scripts/stage2_runtime.py` 中冻结来源 SH 模型和 deform，训练 sigmoid 参数化的 albedo 与方向光表，以前景 RGB 和材质先验优化。GT 方向来自元数据；learned 方向在训练时刻优化、测试时刻插值。该直接 Lambertian 模型没有阴影可见性、镜面项或间接光；当前 AREA 灯被近似为方向，因此 GT-light 仍不是完整的物理光照 oracle。

| 文件 | 本次实际内容 | albedo 提取 |
| --- | --- | --- |
| 原始 IR `point_cloud/iteration_N/point_cloud.ply` | 几何、DC/SH、opacity、roughness、feature | `clamp(C0 * albedo_dc + 0.5, 0.03, 0.97)` 得 sRGB，再转线性；`C0=0.28209479177387814` |
| 原始 IR `deform/iteration_N/deform.pth` | 时间/形变/颜色网络 state_dict | 提供对应时刻的几何形变，文件本身不是 per-Gaussian albedo 表 |
| 原始 IR `envmap/iteration_N/envmap.pth` | 环境光 state_dict、activation | albedo 评测无需加载或查询环境光 |
| 新模式 `stage2_photometric/iteration_N/checkpoint.pth` | `material.raw_albedo`、light、来源哈希、配置、optimizer、RNG | `sigmoid(material.raw_albedo)` 得 sRGB，再转线性；几何从记录的 Stage1 恢复 |

1002-02 的 35k 是 SH 伪材质初始化，40k/50k/55k 为原始 Stage2；模型均有 44,136 个 Gaussian。本次保留原训练/渲染算法，增加专用材质提取路径。

## 2. 真值核验

- 训练 source：`/home/lihan/data/LH-data/transfer-static/only_clothV4`。
- GT albedo：`/home/lihan/data/LH-data/static/only_clothV4/albedo/albedo_XXXX.png`。
- Shader 来源：同目录 `chair_cloth_mvps_120views.blend`。只读解压和解析 SDNA，未使用 Blender 写入原文件。
- 120 个 PNG 均为 1280×720 RGBA，具有 sRGB chunk；各帧 alpha 与转换集训练图像逐像素相同。

| 材质 | Blender shader 线性 Base Color | PNG 模态 sRGB 8-bit | PNG 对 shader 的纯区域 MAE（线性） |
| --- | --- | --- | --- |
| 红布 | `[0.42, 0.018, 0.028]` | `[174, 36, 46]` | 0.001429 |
| 椅面 | `[0.055, 0.070, 0.085]` | `[66, 75, 82]` | 0.000506 |
| 木框 | `[0.220, 0.065, 0.025]` | `[129, 72, 43]` | 0.000510 |
| 底盘 | `[0.800, 0.800, 0.800]` | `[231, 231, 231]` | 0.000913 |

保存场景的 RGB view settings 为 Filmic / Medium High Contrast。albedo PNG 的四种颜色实际与 shader 的标准 sRGB 编码吻合；本次以这项数值核验确定其材质语义，不对 albedo 应用保存场景的 Filmic 设置。前景逐像素指标以解码的 PNG 为参考；纯材质指标直接使用 shader 数值，避免把 PNG 的量化/抖动当作材质误差。

GT 的 RGB 与 alpha 先在线性空间进行 alpha 加权双线性重采样，再除以重采样 alpha，匹配当前模型的 640×360 渲染分辨率。准确匹配尾部帧号，不用子串搜索，不复用单一静态时刻。本次 `dataset_audit.json` 保存全部帧的核验记录和哈希。

## 3. 评测定义

### 3.1 从模型提取无光照材质

专用 `gaussian_renderer/render_albedo.py` 使用当前帧的模型几何、形变和 opacity，将 per-Gaussian 线性 albedo 直接光栅化到零背景，得到累加材质 `A(u)` 和累加 alpha `a(u)`：

```text
rho_pred(u) = A(u) / max(a(u), 1e-6)
a(u) <= 1e-6 时 rho_pred(u) = 0
```

不查询环境光、方向光、阴影颜色调制或镜面项。原始 IR 的 `base_color_linear` 已在光栅化累加结果上额外乘 alpha，而新 Lambertian 返回的是除 alpha 后的材质；本次直接从底层累加结果提取，统一两条路径的语义。各 checkpoint 均在真实相机上与原渲染器进行了数值一致性检查，最大差值为零。

### 3.2 主指标：绝对前景 albedo MAE 与材质均衡 bMAE

GT 内部前景 `U` 定义为重采样的 GT alpha≥0.99，随后以 3×3 邻域收缩 2 像素。所有模型使用同一个 `U`，本次 15 帧共 286,773 个像素。

```text
foreground_MAE = sum_u_in_U sum_c |rho_pred(u,c) - rho_png_GT(u,c)| / (3 * |U|)
```

在 `U` 内，仅以 GT 颜色和已核验的四种材质生成区域：sRGB 各通道与最近的 PNG 模态颜色相差不超过 3/255，再收缩 1 像素以避开混合边界。每种材质的参考色为独立核验的 shader Base Color。

```text
MAE_material_m = sum_u_in_Um sum_c |rho_pred(u,c) - rho_shader_m(c)| / (3 * |Um|)
bMAE = (MAE_cloth + MAE_seat + MAE_wood + MAE_platform) / 4
```

本项目以 **bMAE 为材质分离的主比较指标**，同时报告红布 MAE 和 foreground MAE。bMAE 使四类材质各占四分之一权重，避免面积最大的底盘主导全局结果。所有主指标都使用未缩放、未额外裁剪的线性 RGB，越低越好。缺少某一材质时 bMAE 输出 null，不能拿其余材质冒充完整评分。

GT 评分像素不会被预测 alpha 筛掉；漏检像素仍参与主指标。另报 alpha≥0.9 的前景覆盖率、alpha IoU，以及高 alpha 区域的条件 MAE，后者仅作诊断。本次全部模型的覆盖率超过 99.996%，误差主要不是漏检导致的。

### 3.3 尺度与局部残差诊断

可选 `--calibrate_scale` 在 **105 个训练帧** 的 GT albedo 上拟合全模型的一个非负线性 scalar，或三通道非负系数：

```text
s = sum_train(rho_pred * rho_GT) / sum_train(rho_pred^2)
```

随后固定系数评测 15 个测试帧。保存 `raw`、`train_scalar_diagnostic`、`train_rgb_diagnostic` 三组；训练拟合使用的是额外 albedo 真值，因此缩放结果属于校准诊断，主排名使用 raw。缩放后的值不裁剪到 [0,1]，超过参考范围也计入误差。

同时报告 RMSE、逐像素平均绝对误差的 P95、误差低于 0.05 的比例、RGB bias、预测颜色均值/标准差。恒定 GT 材质中的预测标准差可辅助定位斑驳和阴影残留，也受几何误差、Gaussian 混合和视角采样影响。

## 4. 真实结果

相同 15 个测试帧，各帧使用自己的相机和 time；下表为跨帧像素汇总，未缩放的线性 RGB 指标。

| 模型 | iteration | foreground MAE | bMAE | 红布 MAE | 椅面 MAE | 木框 MAE | 底盘 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1002-02 SH 初始化 | 35000 | 0.191563 | 0.128391 | 0.140504 | 0.048699 | 0.085793 | 0.238569 |
| 1002-02 原始 IR | 40000 | 0.202425 | 0.128365 | 0.106746 | 0.054053 | 0.085196 | 0.267465 |
| 1002-02 原始 IR | 50000 | 0.217230 | 0.128482 | 0.087205 | 0.050369 | 0.078064 | 0.298290 |
| 1002-02 原始 IR | 55000 | 0.219698 | 0.128248 | 0.083853 | 0.049912 | 0.075922 | 0.303305 |
| 1002-03 GT-light | 55000 | 0.139378 | 0.089097 | 0.079757 | 0.034281 | 0.063963 | 0.178387 |
| 1002-04 learned | 55000 | 0.224833 | 0.127783 | 0.093257 | 0.035811 | 0.074950 | 0.307115 |

1002-02 从 35k 到 55k，红布 MAE 降低约 40.3%，底盘 MAE 增加约 27.1%，四材质均衡误差基本持平。因此 RGB 拟合改善与整体 albedo 恢复改善需要分开验证。

GT-light 的 raw bMAE 比原始 IR 55k 低约 30.5%。original 与 learned 的宏平均差距小；将掩码从 alpha≥0.99/收缩2像素改为 alpha≥0.5/不收缩时，这两者的微小排名会翻转。GT-light 在五组掩码设置中均保持明显领先，完整数值见 `mask_sensitivity.csv`。这属于本场景、已评 checkpoint 的结论。

| 55k 模型 | 训练帧拟合 scalar | raw foreground MAE | 校准后 foreground MAE | 校准后 bMAE |
| --- | --- | --- | --- | --- |
| original | 1.431550 | 0.219698 | 0.141884 | 0.108862 |
| GT-light | 0.990509 | 0.139378 | 0.138618 | 0.088390 |
| learned | 1.423856 | 0.224833 | 0.157402 | 0.112091 |

original 与 learned 有整体偏暗；全模型亮度校准能解释部分误差，局部残差仍很明显。原始 IR 55k 的底盘平均线性颜色为 `[0.4795, 0.5135, 0.5072]`，GT 为 `[0.8, 0.8, 0.8]`；红布平均为 `[0.2752, 0.0315, 0.0415]`，GT 为 `[0.42, 0.018, 0.028]`。三组的 albedo 可视化均存在非恒定颜色和阴影形状残留，GT-light 也未达到正确材质分离。

## 5. 使用与验证

新增文件：

- `scripts/eval_albedo_gt.py`：PLY/PTH 恢复、训练/测试分离、校准、评分、缓存和可视化。
- `gaussian_renderer/render_albedo.py`：不依赖光照的线性材质累加。
- `utils/albedo_metrics.py`：CPU 公式、掩码、材质划分和 scale fit。
- `arguments/albedo_onlyclothv4.json`：已独立核验的四材质 GT；显式传入，不会自动套到其他数据集。
- `tests/test_albedo_metrics.py`：10 项独立样例，覆盖背景分母、透明度归一化、漏检、sRGB、校准、区域划分和输入只读保护。

单个权重的基础评测不需校准，可以直接执行：

```bash
conda run --no-capture-output -n lumimotion python -m scripts.eval_albedo_gt \
  --model_path output/1002-02-onlyclothV4-stage2_original/model_mlp \
  --iterations 55000 \
  --gt_albedo_dir /home/lihan/data/LH-data/static/only_clothV4/albedo \
  --material_palette arguments/albedo_onlyclothv4.json \
  --output_path output/1003-07-onlyclothV4-albedo_GT复跑 \
  --save_float
```

增加 `--calibrate_scale` 可复现本次尺度诊断；`--iterations 35000 40000 50000 55000` 可评多个 checkpoint。输出目录必须是新目录。其他具有逐帧 sRGB albedo PNG 的 Blender 数据可先使用 foreground 指标；材质均衡指标需要提供独立核验的 palette。纹理材质不适用当前四个恒定颜色的分组方法。

验证结果：

- 10 项新增单测和原有 Stage2/相机回归测试，共 **27 项通过**。
- 6 个真实 checkpoint 的材质提取与原渲染器一致，GPU 数值核验通过。
- 使用保存的浮点 NPZ 独立复算 foreground MAE/RMSE、四材质 MAE 和 bMAE，误差小于 `1e-8`。
- 6 组输出的测试相机、投影、time、GT 和 GT core 完全相同，校准帧仅来自训练集合。
- 当前最终 CLI 的 GT-light 单帧冒烟通过；正式评测后新增的输入保护不改变评分公式，正式和最终源码及 SHA256 均已保留。

当前评测在可见表面的相机投影上进行，衡量多视角材质误差；它不是唯一三维表面点的面积积分。Gaussian 与 GT mesh 没有一对一索引，直接拿两份点颜色逐行相减没有对应意义。绝对材质和未知光照存在尺度自由度，因此同时保留 raw 和校准诊断，便于分辨尺度偏差与局部残差。

代表图：[checkpoint 曲线](../../../output/1003-06-onlyclothV4-albedo_GT评测/plots/checkpoint_metrics.png)、[分材质对照](../../../output/1003-06-onlyclothV4-albedo_GT评测/plots/material_comparison.png)、[0064 三模型与 GT](../../../output/1003-06-onlyclothV4-albedo_GT评测/plots/three_models_0064.png)。
