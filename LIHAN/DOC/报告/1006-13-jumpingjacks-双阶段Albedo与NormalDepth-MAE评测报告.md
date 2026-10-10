# 1006-13 jumpingjacks 双阶段 Albedo 与 Normal/Depth MAE 评测报告

日期：2026-10-06（Asia/Tokyo）。实验：`1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段`。

**Stage2 55k 的全前景线性 RGB albedo MAE 为 0.061694；normal 与 depth 的真值 MAE 无法评测。** 原数据没有 normal 真值；150 张 depth PNG 解码后全部为纯白，不能还原有尺度的几何深度。本次完成双阶段 albedo 评测、真值可用性审计及报告，不将缺失的几何指标记为零或通过。

## 1. 主要结果

同一15帧测试集、400×400、GT Alpha ≥ 0.5 全前景，主值按所有评分像素汇总，线性 RGB 三通道绝对差平均。不拟合材质尺度，不裁剪预测颜色，不根据预测Alpha剔除GT前景。

| 阶段 | 迭代 | Albedo MAE↓ | Normal角度MAE↓ | Depth MAE↓ | 预测覆盖率（Alpha≥0.9） |
| --- | ---: | ---: | --- | --- | ---: |
| Stage1 SH基色代理 | 35000 | 0.08579548 | N/A：无GT normal | N/A：GT depth全白 | 98.6047% |
| Stage2 IR物理albedo参数 | 55000 | **0.06169353** | N/A：无GT normal | N/A：GT depth全白 | 98.7294% |

两个阶段均使用590,990个GT前景评分像素。Stage2相对Stage1代理的MAE降低0.02410195，约28.09%；15个测试帧的误差均降低。Stage1是含照明的SH基色代理，数值改善不能解释为两个阶段具有完全相同的物理材质含义，也不能作为法线或深度恢复改善的证据。

| 补充口径 | Stage1 35k | Stage2 55k |
| --- | ---: | ---: |
| 全前景逐帧等权MAE | 0.08580166 | 0.06170238 |
| 前景内区像素汇总MAE（GT Alpha≥0.99，腐蚀2像素） | 0.08566499 | 0.06209056 |
| 前景内区逐帧等权MAE | 0.08567640 | 0.06209169 |
| 前景内区预测覆盖率（Alpha≥0.9） | 99.9967% | 99.9980% |
| 全前景单像素三通道平均绝对误差P95 | 0.12812580 | 0.07919051 |

内区共541,083像素，排除软Alpha与轮廓边界；这是补充指标，不替换全前景主值。覆盖率只作诊断，所有GT评分像素保留，包括预测Alpha低的像素。

![15帧测试集的原始全前景Albedo MAE](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/albedo_mae_by_frame.png)

## 2. 数据与模型

- 数据：`/home/lihan/data/d-nerf-relight-spec32/jumpingjacks150_v5_spec32`；训练135帧、测试15帧。
- 测试帧：10、20、30、40、50、60、70、80、90、100、110、120、130、140、150。每帧采用其原始相机和`time`，不是固定视角全时序视频。
- 实际source：`/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/data_view`。
- 模型：同一实验下`model_mlp`，35000与55000的PLY及对应deform PTH；两阶段均146,065点。来源哈希记录于每阶段metrics.json。
- 环境：venus / lumimotion，GPU0。仅只读加载checkpoint，没有重新训练或修改模型、原始数据。
- 本评测目录：同一实验下`evaluation/mae_task`。

## 3. Albedo定义、掩码与帧号适配

GT文件名如`albedo/r_00100010.png`，前4位为相机记录帧号、后4位为动画帧；对应`transforms`中的`r_0010`、time=10/150。150帧均核对两个帧号相等，并匹配JSON时间，规范化为评测目录的`gt_albedo_rgba/r_0010.png`。

原始GT albedo PNG的Alpha全为255，不能用于定义物体前景。本次保持其RGB字节不变，采用对应Chapel Day RGBA图像的真实Alpha构造评测专用GT副本；不改原文件、不改变训练数据。完整来源、映射和SHA256见`audit/gt_availability.json`。

按原版`/home/lihan/reproduce/LumiMotion/scripts/eval_material_dynamic.py`的颜色约定，将GT PNG的sRGB颜色解码为线性RGB。PNG没有明确sRGB/gamma标记，使用显式`--assume_srgb`；这是一项与原版评测一致的颜色空间约定，不是根据本测试集拟合的转换。

预测从PLY中的原始基色经sRGB→线性RGB转换后直接光栅化；不加入SH高阶项、形变颜色head或光照。光栅累加颜色除以预测Alpha恢复表面颜色，Alpha≤1e-6记零。GT先在线性空间按Alpha加权双线性缩放，再除以缩放后的Alpha，避免透明背景污染。主指标：

`MAE = sum(|prediction_linear - GT_linear|) / (3 × GT前景像素数)`。

没有GT尺度拟合、RGB三系数校准、曝光修正或ICP对齐。主值基于保存的float32数组，不使用8位展示PNG计分。

## 4. Normal与Depth不可评测的证据

| 检查 | 结果 | 对指标的影响 |
| --- | --- | --- |
| 当前jumpingjacks数据目录normal候选文件 | 0 | 没有同相机同时间GT normal |
| depth目录图像 | 150张，800×800，RGBA uint8 | 不是可直接解释的浮点Z深度 |
| depth完整像素检查 | 150/150张全部四通道255，解码像素哈希只有1种 | 所有前景/背景/帧均无深度变化 |
| 原始tar中该动态场景的normal/.blend/.exr/.npy/.npz/.obj候选 | 0 | 归档不能补齐缺失几何真值 |
| 本机jumpingjacks/D-NeRF/Dynamic3GS匹配的.blend | 0 | 目前无法从原始动态几何重新导出真值 |

各depth文件的SHA256不同，是PNG元数据（帧号、时间等）不同；解码后像素均为同一纯白图。不能把255当成距离255，也不能把归一化1当成世界Z深度1。数据没有可用于恢复距离的编码、近远平面或尺度说明；全白图也不能生成有效表面法线。

官方[发布记录API](https://zenodo.org/api/records/18894615)列出`d-nerf-relight-spec32.tar`和`enerf_actors_1_3.tar`；本地原始合成tar已核查，没有发现该场景的原始Blender或normal真值。depth PNG元数据提及开发者本机`jumpingjacks_v5_specular32.blend`，但该文件未包含在已核查资源中。

需要补齐与当前相机、time及原始单位一致的世界法线和浮点深度，或完全对应的原始动态Blender/网格，才能继续完成几何MAE。预计normal主口径为单位世界法线的有向角度MAE（度）；depth主口径为正相机Z的绝对误差（原始世界单位）。现有预测normal视频只能说明视觉质量，不能替代GT。此次没有用预测深度推导的法线冒充normal真值。

## 5. 真实目检与误差表现

已真实查看测试帧0010、0080、0150的GT albedo、预测、线性误差热图与预测Alpha。预测保留人物、平台、服装和主要颜色，但衣服、头部/四肢仍有GT albedo中不存在的褶皱和明暗结构；鞋脚接触与轮廓局部误差较明显，平台有整体颜色偏差。此观察与此前“材质和照明尚未完全分离”的判断一致，不是法线/深度定量结论。

![Stage2测试帧0010的Albedo和误差](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/albedo/iteration_55000/0010_comparison.png)

[中间测试帧0080](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/albedo/iteration_55000/0080_comparison.png)；[最后测试帧0150](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/albedo/iteration_55000/0150_comparison.png)。这些热图采用GT前景内区，全前景主值包含更多轮廓像素，口径分别标明。

## 6. 验证、复现和产物

- 两个阶段的纯albedo渲染与原版IR material-only输出在真实权重上核验通过，最大线性颜色差为0。
- 30份保存浮点结果重新用NumPy汇总，逐帧和汇总内区MAE均与原评测JSON一致（容差1e-7）；全前景主值由同一浮点图独立汇总。
- 15帧完整、不重复，GT帧号/时间匹配；所有评分像素保留，覆盖率单独报告。
- 模型PLY/PTH及读取配置哈希在评测前后不变；GT和派生GT哈希可追溯。

完整albedo执行命令保存在`commands/eval_albedo.sh`；先准备GT，再运行评测及汇总：

```bash
cd /home/lihan/reproduce/LumiMotion-stage2-perlight
MAE_DIR=/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task
PYTHON=/home/lihan/miniconda3/envs/lumimotion/bin/python
"$PYTHON" "$MAE_DIR/commands/prepare_gt.py"
bash "$MAE_DIR/commands/eval_albedo.sh"
"$PYTHON" "$MAE_DIR/commands/finalize_mae.py"
```

评测器拒绝覆盖已有`albedo/`输出，重新渲染时应另设新输出目录并保留本次证据；已有浮点结果可直接重跑finalize。`prepare_gt.py`对已有派生GT核验RGB与Alpha，不盲目覆盖。

一手证据：

- [主指标与可用性状态JSON](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/metrics.json)
- [30行逐帧CSV](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/metrics_per_frame.csv)
- [Stage1内区指标、来源哈希与逐帧结果](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/albedo/iteration_35000/metrics.json)
- [Stage2内区指标、来源哈希与逐帧结果](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/albedo/iteration_55000/metrics.json)
- [GT完整审计与帧号映射](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/audit/gt_availability.json)
- [数值复核](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/audit/numerical_validation.json)
- [完整评测日志](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1006-11-jumpingjacks150_v5_spec32-原版随机方块默认两阶段/evaluation/mae_task/logs/albedo.log)

最终评测状态：**ALBEDO_COMPLETED_GEOMETRY_GT_UNAVAILABLE**。Albedo流程与数值复核通过；normal/depth尚未具备GT，不能标为三项MAE全部完成。该限制不改变此前训练和产物验收完成的事实。
