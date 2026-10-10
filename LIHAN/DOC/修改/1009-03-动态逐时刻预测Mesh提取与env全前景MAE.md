# 动态逐时刻预测Mesh提取与env全前景MAE

日期：2026-10-09；venus / lumimotion。

新增scripts/extract_geometry_mesh_dynamic.py，沿用静态入口的Expected Z、Open3D TSDF、像素中心和相机坐标逻辑，在每个目标时刻重新融合135个训练相机下的预测深度，生成逐帧predicted_mesh清单，包含模型、相机、时间、哈希与提取参数。禁止跨时间融合，GT几何和测试相机位姿不参与重建，既有三类Normal评测入口直接读取此动态manifest。

本次env从Blender5.2.2快照补出15个测试帧三角网格、GT正Z深度、粗糙度float EXR。透明Principled Alpha沿用源albedo导出的透明分支，roughness定义为可见Principled r的Alpha加权值；shading Normal与几何法线差记录为诊断。适配器放实验commands，不修改通用严格导出器默认行为。未改训练、模型或原始数据。

真实评测及模型来源不变核查、独立数值复算与代表图目检在1009-03实验执行。未添加或运行测试套件；只执行用户要求的真实权重数据评测。

[报告](../报告/1009-03-jumpingjacks_environment_light-各项MAE与三类Normal评测报告.md)。

完成：2026-10-09T19:22:16.860208+09:00，15逐时刻模型Mesh提取与全部MAE完成；三类法线独立CPU复算及来源哈希通过。GS/Depth/Mesh Normal MAE 13.88986765° / 18.92264049° / 14.95726286°；Albedo 0.19101779、Roughness 0.19992533，深度Expected/Median 0.05899602/0.05912712米。真实目检确认人体法线斑驳、材质明暗/结构误差、局部Alpha漏覆盖；流程PASS不表示模型质量达标。
