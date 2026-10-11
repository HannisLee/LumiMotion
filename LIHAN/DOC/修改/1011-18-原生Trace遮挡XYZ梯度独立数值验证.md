# 原生 Trace 遮挡 XYZ 梯度独立数值验证

## 目的和边界

为后续阴影梯度分流提供最低限度的运算证据：原生 Gaussian/BVH trace 是否对遮挡 Gaussian 的位置提供正确的局部梯度，以及独立阴影路径能否阻断接收点、方向、旋转、尺度、透明度、材质和光强的梯度。

本次只新增 `LIHAN/scripts/goal_trace_xyz_probe.py`。不修改正式核心、冻结快照、评测口径或 T01/T02/T03 参数，不启动图像训练，不读取训练/测试图像及 GT 几何或光强，不建立优化器。探针结果不代表全 15 帧法线 MAE 改善，也不决定下一轮正式实验配置。

## 数值设计

- 构造接收点 `(0, 0, 0)`，方向 `normalize(0.2, -0.1, 1)`，单个平面 Gaussian 中心 `(0.42, -0.04, 1)`、尺度 `(0.25, 0.20)`、单位四元数和透明度 `0.7`。
- 经正式 `GaussianModel.trace` 计算可见度，以固定 Lambert 系数乘可见度得到标量响应。遮挡路径仅保留遮挡物 XYZ 计算图，其余指定输入显式 `detach`。
- 在避开离散命中/截断的位置，使用独立闭式表达和中心差分核验 XYZ 导数。有限差分步长为 `0.002`、`0.001` 和 `0.0005`，每次扰动重新构建 BVH，梯度相对 L2 误差阈值为 `1%`，Alpha 绝对误差阈值为 `2e-5`。
- 去除全部 `detach` 作为负对照：相同输入实际参与计算、梯度非零；避免把未连接的输入误当作成功阻断的路径。
- 遮挡物沿 X 移动 `-0.08/0/+0.08`，核验可见度按解析关系单调变化。
- 对透明度 `0.99` 的近中心射线，同时调用正式包装器和其下层原生 tracer，核验包装器在 `alpha >= 0.97` 后强制输出 `1` 导致的 XYZ 零梯度。
- 构造距接收点前方约 `0.02 m` 的真实遮挡物，对比 `0.001/0.03 m` 起点偏移，检查较大偏移可能直接越过近邻遮挡的限制。

单高斯交点在世界坐标为 `(0.2 z, -0.1 z, z)`，令 `u=(x-0.2z, y+0.1z)`，则 `alpha=0.7 exp[-0.5 sum((u/sigma)^2)]`。光照因子为 `0.7*1.3/sqrt(1.05)`。此闭式表达独立于 CUDA backward，可同时检查 forward 和 backward，而不只是比较同一实现的两个自动梯度。

## 环境、产物和命令

- 服务器：`ushas`。
- Python：`/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python`。
- GPU：仅在 evaluator 释放的短窗口使用 GPU3，UUID 为 `GPU-9a65d91a-3fcb-5bca-6baa-b7e4ec262d2b`；脚本核验环境绑定和实际进程归属。
- 输出预留：`output/smoke_test/1011-07-staticDL-原生Trace遮挡XYZ梯度`。
- 产物：`command.sh`、`probe.log`、`result.json`、`README.md`，失败不覆盖。
- source 为解析构造，model/iteration 均不适用；无渲染图片和视频，四类可视化目检不适用。

```bash
CUDA_VISIBLE_DEVICES=GPU-9a65d91a-3fcb-5bca-6baa-b7e4ec262d2b \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/han.li/miniconda3/envs/lumimotion-ushas/bin/python \
LIHAN/scripts/goal_trace_xyz_probe.py \
  --output_dir /home/han.li/reproduce/LumiMotion-stage2-perlight/output/smoke_test/1011-07-staticDL-原生Trace遮挡XYZ梯度 \
  --expected_gpu_uuid GPU-9a65d91a-3fcb-5bca-6baa-b7e4ec262d2b
```

脚本记录实际 Python wrapper、原生 `.so`、`gaussian_model.py` 和探针自身的 SHA256。实际加载的 wrapper 与仓库源码 SHA256 必须一致。本次使用正式训练同一环境已安装的原生 tracer；没有重新编译，因此不把 SHA 记录表述为独立重建源码与二进制一致性的证明。

## 执行结果

2026-10-11 08:07:36—08:07:43 UTC，在 evaluator 明确确认 T01 全 15 帧评测退出并释放 GPU3 后执行。脚本 `py_compile` 通过，GPU 探针退出码 `0`，五项数值测试全部 **PASS**，耗时 `6.774 s`。实际进程 `1995049` 的 GPU UUID 与预定一致，初始化时显存 `326 MiB`；PyTorch 峰值分配 `17,591,808 B`、峰值预留 `23,068,672 B`（不包含全部 OptiX/CUDA 开销，不能当作总显存峰值）。结束后已立即通知 evaluator/trainer，GPU3 继续 T03 评测。

| 核验项 | 实测结果 | 判定 |
| --- | --- | --- |
| 平滑区 forward Alpha | `0.4543556869`，闭式值 `0.4543556835` | PASS |
| 遮挡 XYZ 自动梯度 | `(1.4203171730, 0.6052489281, -0.2235385478)` | 有限且非零 |
| 自动梯度对独立闭式导数 | 相对 L2 误差 `9.02145e-8` | PASS |
| 中心差分 `epsilon=0.002` | 相对 L2 误差 `4.03405e-5` | PASS |
| 中心差分 `epsilon=0.001` | 相对 L2 误差 `3.06168e-5` | PASS |
| 中心差分 `epsilon=0.0005` | 相对 L2 误差 `5.94791e-5` | PASS |
| 禁止路径 | 接收起点、光方向、接收 q、遮挡 q、尺度、透明度、材质、光强梯度全部为 `None` | PASS |
| 去掉 detach 的负对照 | 上述全部参数组梯度均有限且范数非零 | PASS |
| 遮挡物 X 位移 `-0.08/0/+0.08` | 可见度 `0.42791915/0.54564428/0.67426622`，单调变化 | PASS |
| 包装器饱和 | 原生 Alpha `0.98987323`，XYZ 梯度非零；包装后 Alpha `1.0`、XYZ 梯度精确全零 | PASS，确认限制 |
| 近邻遮挡与 bias | `bias=0.001` 时 V=`0.30502188`；`bias=0.03` 时 V=`1.0` | PASS，确认漏遮挡限制 |

三个差分步长的最大相对误差为 `5.95e-5`，远低于预设 `1e-2` 阈值。负对照只有“各参数组范数非零”的要求；单位四元数某些分量的导数自然为零，不应把每个分量均非零当作正确性的条件。

实际 wrapper 与仓库 wrapper 的 SHA256 均为 `8f066f9df45b48f3ac4b4a468e9c5d6615d684a5f71708790a1311bbe7a5184b`，实际原生 `.so` SHA256 为 `5b666e2b3b9fcae5b3137f3689b1ea43e44e2f892874c08ffb0257244ac8809e`。完整梯度数组、SHA、环境及时间见输出目录 `result.json`，执行脚本原样另存为 `probe.py`。

**本次结论：原生 trace 在所测平滑单高斯情形下支持正确的遮挡 XYZ 局部导数，显式 detach 能阻断所列输入梯度。已具备尝试下一阶段完整训练路径审计的必要运算证据；尚不能宣称完整训练分流正确、几何恢复有效或 Goal 达标。**

## 解释边界

1. 当前 T03 的可见度仍在 `no_grad` 中，本探针不改变它，也不把 T03 总损失存在 XYZ 梯度称为遮挡路径已有 XYZ 梯度。
2. 平滑衰减区的局部导数正确，不代表 BVH 命中集合变化、Alpha 下阈值、单点 `0.99` 上限、透射率 `0.03` 早停等离散或饱和位置可微。
3. 单条射线对沿射线方向的遮挡物平移存在零空间；XYZ 三分量各自非零并不代表位置的三个自由度均可观测。
4. 本次尚未接入主相机 Alpha/depth 合成、共享形变网络、增密统计等路径，因此不能替代完整训练路径的无旁路审计。
5. 偏移实验只说明过大的 bias 可以漏掉近邻真实遮挡，不能证明实际 DL 场景中 `0.03 m` 最优，也没有完整验证自遮挡消除。
6. 冻结旋转、尺度和透明度是本次受控分流策略；不代表阴影在物理上普遍不约束这些参数。
