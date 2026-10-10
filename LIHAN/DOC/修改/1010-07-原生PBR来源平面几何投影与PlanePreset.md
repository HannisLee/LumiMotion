# 1010-07 原生PBR来源平面几何投影与Plane Preset

> 后续更正：初版GeometryRun未消费Anchor/Plane，本文初次“全部离线入口共享”声明不成立；已由1010-08修改接通并通过真实GPU buffer逐像素验证。B04旧几何18.91235°作废，正式更正18.77743°；完整报告1010-11，旧失败产物保留。B05训练参数未改，最终评测使用修复入口。

## 修改动机

修改前已读取GOAL全部最新B04结果。B04 Anchor消除移动XYZ查询源MLP的放大，完整20k后GS Normal仍18.91234950°、图像质量FAILED。全135训练时刻只读审计发现此前静态feature支持包括运动点，canonical平面loss不能保证实际渲染world平面。B05新增来源预测驱动的实际几何投影，不能预先断言降低MAE。

## 实现与边界

新增scene/pbr_plane_projection.py以及独立pbr_gt_directional_plane Preset，继承Anchor参数；旧默认/旧Preset全部保持projection关闭。源35k未改PLY/MLP、所有135训练相机中心/time产生固定支持资产，不使用GT normal/plane/mesh/albedo/roughness/depth/光强/光色。

Source RANSAC/SVD拟合band .08、seed1010不变；支持要求feature<.5、Alpha>.05、canonical及全部time有效XYZ距plane≤.2m、形变坐标全time box-span范数≤.01m、canonical及全time有效normal轴与plane绝对cos≥.5。真实99827点支持15187，未删除任一点；其余点输出完全不改。支持/plane/polarity及全time统计存NPZ，source PLY/PTH/camera SHA存JSON；load时从真实source完整重算逐数组/元数据比较，不接受只修改mask并重写SHA；每个实际checkpoint marker绑定asset SHA。

在原MLP sourceXYZ输入输出后，对支持的actual world XYZ投影到source plane，最短弧将actual quaternion Z轴旋到source polarity×plane，保留切向旋转和raw norm。投影位于MLP no_grad之外，可对实际pc参数求导。RGB、原生PBR/BVH/GGX、周期报告、FrozenScene以及全部离线GS Normal/Depth/Mesh/HDR/转灯共享相同实际几何，未在评分之后替换法线。实际MLP32tensor完全不改。

Canonical保存PLY仍用原.08m/60°边界，但actual world几何额外经过投影，不能称总world位移受.08m限制；plane假设可能错误平坦化来源误分类点或真实细节，需要全量MAE和目检。材质、光强、loss权重、lr/warmup及其它参数继承B04；全15帧与全FG评分口径、coverage/common/missing180不改。

## 验证证据

173项CPU tests实际exit0。新增10项覆盖source roundtrip、支持的真实位置和旋转/梯度、body及材质不改、反向轴四元数、零q拒绝、source/nooverwrite、伪mask重新SHA拒绝、marker绑定、变MLP/相机拒绝以及默认/Preset。

真实venus GPU零optimizer步：Native Trainer创建full135来源支持，FrozenScene完整重算一致；分别查询1/中/尾相机，实际world plane距离<5e-7m、轴cos误差<1e-5，非支持形变逐元素不改；修改当前pc XYZ .002后train/FrozenScene一致。真正周期报告3camera调用输入仍sourceXYZ并应用相同projection；原生PBR/BVH/GGX真实前后向，XYZ/rotation/albedo/roughness/全局正光强均有finite非零梯度，sourceNN无梯度、源SHA未变。实际exit0；不把0步源35k当新训练checkpoint，不使用smoke作为最终MAE。

命令：

```bash
/home/lihan/miniconda3/envs/lumimotion/bin/python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/lihan/miniconda3/envs/lumimotion/bin/python -u output/smoke_test/1010-10-dir-PBR来源平面投影验证/commands/check_gpu.py
```

证据output/smoke_test/1010-10-dir-PBR来源平面投影验证/{audit.json,logs/unit_tests.log,logs/gpu_check.log,commands/check_gpu.py}。git CRLF-aware diff --check PASS；train_stage2.py和render_ir.py原CRLF保留。

## 完整训练

新B05 output/1010-10-jumpingjacks_dir-PBR来源平面几何投影，35k→55k完整20000步；训练计划1010-10训练报告、源码/命令快照已保存。仍仅原RGB/自身RGBA/相机时间/来源预测/GT direction-only。最终七项MAE、质量和7°PENDING，不提前commit/push。失败完整保留。
