# 1007-02 三类Normal MAE命名与模型Mesh评测

日期：2026-10-07；当前项目；venus / lumimotion。

用户要求明确区分法线对象并修改评测。本次沿用1002-02原版Stage2/55k权重，完成三类法线评测及包含训练相机视角的流程图，未启动训练。

## 历史含义与修改

旧主表Normal MAE对应 **GS Render Normal MAE**。旧 `depth_derived_angular_deg` 对应 **Depth Normal MAE**。旧GT Mesh与GT EXR法线一致性核验不是模型重建质量。

当前明确输出：

1. **GS Render Normal MAE**：透明度/透射率混合并归一化的高斯平面法线，与GT Normal EXR比较。
2. **Depth Normal MAE**：模型Expected Z反投影后中心差分得到的世界法线，与同一GT Normal EXR比较。
3. **Mesh Normal MAE**：模型重建Mesh的第一交点处几何顶点法线插值，与同一GT Normal EXR比较；按相机射线定向。

三项均采用单位世界法线、有方向角度MAE（度），分别报告覆盖率，并增加共同有效像素比较。保留旧JSON字段，新增 `normal_metrics`；缺失模型Mesh时明确为 `not_available/null`。

代码：[评测入口](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/eval_geometry_gt.py)、[指标计算](/home/lihan/reproduce/LumiMotion-stage2-perlight/utils/geometry_eval_utils.py)、[模型Mesh提取](/home/lihan/reproduce/LumiMotion-stage2-perlight/scripts/extract_geometry_mesh.py)、[预测Mesh法线渲染](/home/lihan/reproduce/LumiMotion-stage2-perlight/utils/predicted_mesh.py)。

## 实际模型Mesh与结果

静态only_clothV4：仅融合105个训练视角的预测Expected Z，固定TSDF体素0.02、截断0.08、alpha≥0.5、resolution=2；不使用GT几何或测试视角重建，不重新拟合尺度/姿态。得到50159顶点、97754三角面。Mesh误差受到模型深度和提取参数共同影响；薄布边缘及孔洞用覆盖率反映。动态数据需要匹配frame/time的逐帧Mesh，不能跨时刻静态融合。

全部15个测试视角的全前景逐帧等权MAE依次为 **8.0163° / 12.3835° / 11.4455°**，覆盖率 **99.7865% / 95.6373% / 98.0005%**。仅布料前景为14.2198° / 15.0624° / 15.6191°。详细区域、共同像素结果、来源哈希与复现命令见 [产物README](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/README.md)。

15项指标/求交测试通过；旧GS、Depth Normal和深度逐帧及汇总数据与历史评测一致。评测代码、权重、相机、真值和预测Mesh来源均核验哈希。流程PASS表示计算与产物完整，未设模型质量合格阈值。

## 图与证据

[三类法线流程与105训练相机视角](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/normal_metrics_with_train_views.png)；[训练视角三类法线GT对照](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/training_three_normal_comparison.png)。均提供PNG/SVG/PDF。训练0001/0041/0081仅作可视化示例，主成绩来自15测试视角。

[最终测试指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/test_evaluation_final/metrics_summary.csv)；[逐帧法线指标](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/test_evaluation_final/normal_metrics_per_frame.csv)；[最终核验](/home/lihan/reproduce/LumiMotion-stage2-perlight/output/1007-02-onlyclothV4-三类法线MAE/final_audit.json)。
