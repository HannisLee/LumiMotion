# 1003-02 only_clothV4 法线与深度真值评测

2026-10-07命名补充：本文历史主表Normal MAE均对应 **GS Render Normal MAE**。原 `depth_derived_angular_deg` 对应 **Depth Normal MAE**；GT网格与EXR法线差只用于真值核验。三类明确命名与真正模型Mesh评测已完成，见 [1007-02更新](/home/lihan/reproduce/LumiMotion-stage2-perlight/LIHAN/DOC/修改/1007-02-三类NormalMAE命名与模型Mesh评测.md)。本文历史数值与证据保持原含义。

日期：2026-10-03；venus / lumimotion；实现分支：1001-stage2-perlight。

## 入口与评测范围

直接比较预测几何和 Blender 真值，联报法线 MAE、深度 MAE、覆盖率，不把角度和距离合成单一分数。

- 入口：`python -m scripts.eval_geometry_gt`。
- 完整命令、结果、CSV、日志和代表图：`output/1003-06-onlyclothV4-geometry_gt_eval/README.md`。
- 冒烟：`output/smoke_test/1003-06-onlyclothV4-geometry_gt_eval/`，仅 0008/0064/0120 三帧。
- 正式：六个模型/iteration，每组完整 15 个测试视角；未启动新训练。

## 管线核对

**不能仅从 PTH 读取逐像素几何。** `point_cloud/iteration_N/point_cloud.ply` 保存 canonical 位置、opacity、切向尺度、四元数、材质和二值形变特征；`nx/ny/nz` 是零占位，不能用于评测。`deform/iteration_N/deform.pth` 保存时间编码、主干和位移/旋转/尺度/颜色 head 的权重，不是法线/深度图。`envmap.pth` 是光照。独立 Lambertian 的 `checkpoint.pth` 则引用冻结的 Stage1 PLY/PTH，并保存来源哈希。

实际预测为：canonical Gaussian → 当前 time 的形变 → 当前相机投影 → 深度排序和 alpha 合成。`utils/time_utils.py` 中位移/旋转增量乘二值特征，尺度增量目前返回 `scaling*0`；`get_rotation_bias()` 单位化 canonical quaternion 与旋转增量之和。栅格器从变形后四元数得到盘面短轴法线，向相机定向，再按 `alpha*transmittance` 合成。`rend_normal` 已转换到世界坐标，评测前再单位化。

深度导出法线是另一个来源：对预测 Z 深度重建点做邻域差分。Stage1 的 normal regularization 比较 Gaussian normal 和这种预测深度法线，不是 Blender GT normal 监督。GT EXR 在本次离线评测中才读入。

原始 IR Stage2 的 Gaussian optimizer 保留材质与 opacity，移除位置、旋转、尺度和二值特征；形变前向位于 `torch.no_grad()` 中。虽然 `mlp_color` 在 optimizer 中，本次 35000→55000 全部 32 个 deformation tensor 实测都没有改变。位置、尺度、旋转、`fea_0` 也未变，但 raw-opacity 最大差为 13.69880295；这是 logit 参数差，不能当作概率差。opacity 改变会改变像素混合，因此法线/深度仍会变化。

GT/learned Lambertian Stage2 连 opacity 也冻结，几何结果与 Stage1 35000 完全相同。这两组不能证明 Stage2 改善了法线或深度。

## 真值与对齐

训练路径：`/home/lihan/data/LH-data/transfer-static/only_clothV4`。原始真值：`/home/lihan/data/LH-data/static/only_clothV4`，包含 camera.json、normal_exr、albedo-alpha、baseline.ply 和原始 blend，没有深度文件。

法线 EXR 使用 `Normal.X/Y/Z`，按通道名读取；area resize 到 640×360，再单位化。GT 与预测均为世界坐标，不交换 RGB/BGR，也不额外施加相机坐标转换。Blender Normal pass 的世界坐标定义见 [官方手册](https://docs.blender.org/manual/en/4.4/render/layers/passes.html)。

深度生成分两步：

1. `scripts/export_blender_geometry.py` 用 Blender 5.2.2 LTS 后台读取原场景，按各帧 render modifier 设置导出 evaluated triangle mesh、逐角点世界法线、物体 ID。只改变进程内状态，不保存 blend。
2. `scripts/prepare_geometry_gt.py` 用 CPU Open3D RaycastingScene，从相同相机和像素中心发射不单位化射线，取首个三角形交点。Blender 局部方向为 `[(u+0.5-cx)/fx, -(v+0.5-cy)/fy, -1]`，故 `t_hit` 就是正的 camera Z。另存 ray distance 供核对，两种深度不混用。

没有用预测模型产生 GT，也没有做尺度/平移拟合、ICP 或图像 min/max 归一化。场景为 METRIC、scale_length=1，即一世界单位为一米；JSON 保留世界单位，表格转为 cm。这是理想中心射线三角面真值，不是 EEVEE Z pass。

真值核验：15 个测试帧相机最大矩阵差为 0，网格 hash 完全相同；baseline 50000 个采样点到网格的均值距离 2.0342e-7、P95 2.4777e-7、最大 9.0307e-5 世界单位；网格命中与 albedo-alpha 的 IoU 为 0.99949–1；网格插值法线与 EXR 的内缩前景角度 MAE 为 0.657–1.624°。后两种法线的采样/渲染方式不同，这项检查用于排除场景和坐标错误，不宣称逐像素完全等价。

## 指标协议

| 指标 | 定义 | 单位 |
| --- | --- | --- |
| Normal MAE | 单位世界法线的 `mean(acos(clamp(dot(pred,GT),-1,1)))` | ° |
| expected Depth MAE | `mean(abs(expected_Z-GT_Z))` | 世界单位；本场景为 m |
| 覆盖率 | 有效预测像素数 / GT 可评测像素数 | 比例 |

另报无方向轴误差 `acos(abs(dot))`、XYZ 分量平均绝对误差、余弦损失、深度导出法线 MAE、中值深度 MAE、相对深度误差、逐帧 median/P95/bias、Alpha IoU。缺失法线的附加指标按 180°惩罚。

`expected_Z=sum(alpha*T*Z)/sum(alpha*T)` 与当前 `depth_ratio=0` 的训练语义一致，建议长期作为主深度指标。`median_Z` 是栅格器通道 5 的中值输出，作为固定辅助项，不能每组实验挑更小者排名。

区域：`foreground` 为 GT alpha≥0.5 且命中网格；`interior` 为 GT alpha≥0.99 后内缩 2 个评测像素；`cloth_foreground` 用首交点物体 ID 限定 Falling_Cloth；`cloth_interior` 对布料区域自身内缩。预测 alpha≥0.5 且预测法线非零/有限、深度正且有限的交集用于条件 MAE；GT 定义覆盖率分母。深度导出法线还要求有效差分邻域，JSON 保留独立样本数。

`frame_mean` 是 15 帧等权平均，作为表格口径；`pixel_mean` 是所有有效像素加权平均。逐帧 median/P95 不冒充全局分位数。测试帧为 8、16、24、32、40、48、56、64、72、80、88、96、104、112、120，resolution=2，640×360。

## 完整测试集结果

| 模型 / 累计 iteration | 全前景法线 MAE ° | 仅布料法线 MAE ° | 全前景 expected 深度 cm | 仅布料 expected 深度 cm | 全前景 median 深度 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| SH 来源 / original 初始化 35000 | 7.5227 | 13.6213 | 2.7591 | 3.1978 | 2.5517 |
| original IR 40000 | 7.7025 | 13.8306 | 2.9426 | 3.1951 | 2.7614 |
| original IR 50000 | 7.9481 | 14.1407 | 3.0624 | 3.2404 | 2.9028 |
| original IR 55000 | 8.0163 | 14.2198 | 3.0959 | 3.2572 | 2.9394 |
| Lambertian GT 55000 | 7.5227 | 13.6213 | 2.7591 | 3.1978 | 2.5517 |
| Lambertian learned 55000 | 7.5227 | 13.6213 | 2.7591 | 3.1978 | 2.5517 |

55000 original：全前景覆盖率 99.7865%，布料 99.7620%；内缩区域覆盖率均为 100%。全前景/布料 XYZ 分量 MAE 为 0.0650649/0.1182651，深度导出法线 MAE 为 12.3835°/15.0624°。内缩后的法线 MAE 为 7.2896°/12.8898°，expected 深度 MAE 为 3.0597/2.8089 cm；内缩结果不能替代完整前景。

1002-02 Stage2 没有改善这组几何指标：全前景法线 MAE 上升 0.4936°，expected 深度 MAE 增加 0.3369 cm。冻结字段和 PTH 核验排除了位置、旋转、尺度和形变变化，差异来自 opacity 的混合变化。整个前景含大量较易的椅子与平台，后续实验应把布料独立结果作为重点。

同一 55000 权重的阈值敏感性：

| 预测 alpha 阈值 | 条件法线 MAE ° | expected 深度 MAE cm | 覆盖率 | 含缺失惩罚的法线 MAE ° |
| --- | ---: | ---: | ---: | ---: |
| 0.1 | 8.0940 | 3.0921 | 100.0000% | 8.0940 |
| 0.5 | 8.0163 | 3.0959 | 99.7865% | 8.4008 |
| 0.9 | 7.6663 | 3.1007 | 98.4189% | 10.4821 |

提高阈值降低了条件法线误差，同时丢弃更多像素。建议固定 0.5 并联报覆盖率；保存的 NPZ 在 0.5 下重算与正式 JSON 完全相同。

## 验证与边界

7 个新的已知值数值测试，加已有相机/Stage2 测试，共 24/24 通过。原 IR renderer 在三代表帧的 Alpha、单位世界法线、expected Z 深度与评测器逐像素差为 0；Lambertian 的 Alpha/深度差为 0，前景法线最大差 6.01e-8。35000 副本与真实 Stage1 来源的几何/opacity/形变也完全相同。

评测结束重新检查全部来源哈希；既有模型、数据和渲染保留。保留早期 Blender 5.0.1 的版本警告，以及汇总时二次单位化带来约 1e-15 差异的失败日志/文件；正式 GT 使用 5.2.2，修复后的重算检查通过，六组正式指标不受影响。

PASS 指评测功能、GT 对齐和产物通过，未设重建质量阈值。结果衡量既有场景在未训练视角上的重建误差，不是未知场景泛化或逐 Gaussian 最近邻法线误差。透明材质等复杂场景需要另行确认 GT 首交点语义，当前场景的几何/Alpha 核验已通过。

2026-10-07 更正初始化说明并补查世界尺度：1002-02 的 Stage1 来源模型保存的 `input.ply` 是 100000 个默认随机方块点，SHA256 为 `f9a2601f8b96bb7381e8b9086c70d97790c4d93026932e332838e0e9e256c971`；此前“baseline 参与该组初始化”的描述不准确，baseline 在本评测中只用于独立 GT 网格核验。保存的 120 个训练/测试相机与原 GT 相机的位置最大差为 3.5379151e-7 世界单位，内参一致；Blender loader 不执行世界坐标归一化。尺度诊断 s=0.9999999999999998，评测未应用该拟合。详见[尺度核查记录](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-13-onlyclothV4-几何评测流程与训练视角/scale_audit_1007.json)。已有几何指标不变。

后续建议固定联报 **cloth Normal MAE / cloth expected Depth MAE / coverage**，同步记录全前景和两种法线来源；先统一 split、分辨率、符号规则、深度定义、mask，再比较算法。
