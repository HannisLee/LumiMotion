# 1010-03 原生PBR预测几何微调与来源平面约束Preset

日期：2026-10-10（JST）。状态：代码接入、143项CPU单元测试与真实权重单帧GPU梯度检查通过；尚未进行新完整训练，未取得新MAE，不标记质量或7°目标通过。

## 动机与输入边界

原生Stage2仅优化材质/SH/opacity，排除xyz/rotation。现有dir40k预览仍有人物斑驳与平台伪几何；新增可选pbr_gt_directional_geometry。基线pbr_gt_directional和原envmap范围保持。

训练只能用RGB/输入RGBA、相机/time、Stage1预测和GT方向；不加载GT albedo、normal、roughness、depth、mesh、blend或GT光颜色/强度。平面法线来自来源预测静态高opacity位置的种子1010 RANSAC和SVD，禁止硬编码世界法线或加载外部法线图。

## 原生参数与损失

实际原生PLY的xyz和rotation进入优化器；源MLP几何保持，所有离线渲染/三法线/depth/Mesh读取更新后的原生PLY。新增source几何记录pbr_geometry_source.json，无独立假法线展示分支。默认冻结opacity，尺度与静动态feature冻结，避免删减前景覆盖率。

2000步材质warmup后开放几何，1000步ramp，LR从xyz2e-5/rotation1e-3缓降到.1倍。每4步原生更新；原生xyz scheduler在此独立preset由新schedule替代。几何梯度norm裁剪10，位置距离最大.08世界单位，四元数最短弧最大60°，保持来源raw quaternion norm稳定MLP加法偏置。

附加损失：预测GS Normal-预测Depth Normal一致性.05（有效预测邻域，不用GT Normal）；来源平面法线一致性.5、位置一致性.1（band.08）；源XYZ/rotation弱锚定各.001；原始输入Alpha轮廓L1 .1。原生PBR RGB/SH/material smooth保留。主平面不足可靠支持时显式失败，禁止偷偷用GT/固定法线替代。

## 验证与限制

倾斜平面精度修复后142项CPU测试通过，再加入光强梯度/marker测试共143项通过。SVD以inlier均值中心拟合方向，offset取法线投影中位数。单帧真实35k模型训练frame0001的PBR/GGX/BVH前向/反向与投影通过，xyz/rotation/albedo/roughness梯度均有限且非零，源模型SHA未变；审计output/smoke_test/1010-03-dir-PBR几何GPU梯度。此检查不运行训练、无新权重或MAE。

本改动在1001-stage2-perlight准备，B00已有进程未启用；完整B00评测和人工审阅后决定目标1修复重训。每次训练必须独立输出、完整MAE、训练/报告/GOAL落档，质量验收后才commit/push，再从该版本建用户目标分支。

## 修改位置

arguments/stage2.py：独立native PBR几何开关/LR/信赖域/损失；scene/pbr_geometry.py：来源拟合、预测几何损失、参数投影、验证；scripts/loss_stage2.py：新preset与模式约束；scripts/train_stage2.py：优化组、warmup/ramp、附加项、几何投影、来源记录；tests/test_pbr_geometry.py：CPU回归。未改变原生renderer公式或评测定义。

## 训练RGB拟合全局光强（新增）

新geometry preset默认启用pbr_learn_light_intensity，仅以训练RGB优化log正标量，初值是CLI固定π（本实验），Adam LR.001、log初值锚定.001，范围[.01,100]。白色方向和相对单位权重固定，GT颜色/强度不读，不使用测试RGB标定。旧preset默认仍固定光强。

lighting marker的intensity_scale保持配置初值，另保存effective_intensity_scale与intensity_fitted_from_train_rgb=true；离线Stage2RenderRun先验证初值/快照，再恢复同iteration拟合值。缺少记录时拒绝渲染，不能回退初值。Roughness材质评测同样审核该marker。不声称拟合值是真实GT能量；与albedo有可辨识性限制，必须结合raw Albedo MAE如实报告。仍不支持原生完整optimizer resume。

修改另涉及scene/pbr_directional.py的拟合标记验证与scripts/render_stage2.py的真实光强恢复。新增CPU测试验证标量梯度与marker必需性。实际35k模型单帧GPU光强梯度验证使用训练frame0001，不运行训练、不保存候选权重。

## B01执行更新

B00全量训练/评测完成，人工质量FAILED；本修改已用于独立完整B01训练1010-04，PID206152，2026-10-10 01:54:52 JST启动。从同35k来源训练至55k，未重跑Stage1；真实配置已核对新preset/冻结范围/方向-only灯表。37700步实际TensorBoard记录光强4.64452362、预测GS-Depth一致性.120153、平面法线项.0454543、轮廓.0138896，均有限；这些训练损失不是GT MAE，不能证明达到7°。40k只读RGB预览已排队，最终全量评测等待55k。
